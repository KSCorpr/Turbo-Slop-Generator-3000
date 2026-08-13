"""Sonde MiniMax-H3 : la ligne de commande doit être celle de la doc.

Ce script fait télécharger 26 Go puis tourner plusieurs minutes. S'il se
trompe d'un drapeau, on l'apprend après tout ça — ces tests le vérifient en
quelques millisecondes.

Le point le plus facile à rater : le modèle retenu est `ref2va`, un modèle
REFERENCE-to-video. Sans `-r`, il n'a rien à quoi se raccrocher. La première
version de la sonde n'en passait pas.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "try_minimax", ROOT / "scripts" / "try_minimax.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def _paths():
    return {role: Path(f"/m/{Path(name).name}")
            for role, _repo, name, _gb in T.WEIGHTS + [T.TURBO]}


class WeightSetTests(unittest.TestCase):
    def test_it_takes_the_lightest_published_set(self):
        """Le but est de savoir si ÇA PASSE, pas de faire beau : on prend le
        plus petit quant publié."""
        names = [n for _r, _repo, n, _gb in T.WEIGHTS]
        self.assertTrue(any("ref2va_pruned-Q2_K_M" in n for n in names), names)
        self.assertTrue(any("qwen3vl_32b" in n and "Q2_K_M" in n
                            for n in names), names)

    def test_the_turbo_lora_matches_the_diffusion_model(self):
        """Une LoRA `fl2v` sur un modèle `ref2va` ne s'appliquerait pas — et le
        test entier ne voudrait plus rien dire."""
        diffusion = next(n for _r, _repo, n, _g in T.WEIGHTS
                         if "minimax_h3_ref" in n or "minimax_h3_fl" in n)
        self.assertIn("ref2v", T.TURBO[2])
        self.assertIn("ref2va", diffusion)

    def test_the_announced_size_matches_the_files(self):
        total = sum(w[3] for w in T.WEIGHTS) + T.TURBO[3]
        self.assertAlmostEqual(T.NEEDED_GB, total, places=3)
        self.assertGreater(T.NEEDED_GB, 20, "annonce trop optimiste")


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.paths = _paths()

    def _cmd(self, turbo, placement=None):
        return T.build_cmd(Path("sd-cli"), self.paths, turbo,
                           Path("out.webm"), Path("/ref.png"),
                           placement=placement)

    def test_the_four_components_are_all_passed(self):
        """Quatre poids, quatre drapeaux distincts : oublier `--audio-vae` est
        l'erreur qui coûte le plus cher, elle ne sort qu'au chargement."""
        cmd = self._cmd(False)
        for flag in ("--diffusion-model", "--vae", "--audio-vae", "--llm"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--audio-vae") + 1],
                         "/m/minimax_h3_audio_vae_fp32.safetensors")
        self.assertEqual(cmd[cmd.index("--vae") + 1],
                         "/m/minimax_h3_video_vae_fp16.safetensors")

    def test_the_reference_image_is_always_passed(self):
        """ref2va EXIGE une référence. Les deux passes doivent la porter."""
        for turbo in (False, True):
            cmd = self._cmd(turbo)
            self.assertIn("-r", cmd)
            self.assertEqual(cmd[cmd.index("-r") + 1], "/ref.png")

    def test_the_mode_and_documented_flags_are_kept(self):
        cmd = self._cmd(False)
        self.assertEqual(cmd[cmd.index("-M") + 1], "vid_gen")
        for flag in ("--diffusion-fa", "--offload-to-cpu", "--rng", "--fps"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--cfg-scale") + 1], "1.0")

    def test_the_frame_count_sits_on_the_models_grid(self):
        """Le modèle impose une grille `17k + 5` : une valeur hors grille est
        remontée en silence, et la durée n'est plus celle qu'on croit."""
        cmd = self._cmd(False)
        frames = int(cmd[cmd.index("--video-frames") + 1])
        self.assertGreaterEqual(frames, 5)
        self.assertEqual((frames - 5) % 17, 0, f"{frames} hors grille 17k+5")

    def test_without_turbo_no_lora_and_no_step_override(self):
        cmd = self._cmd(False)
        self.assertNotIn("--lora-model-dir", cmd)
        self.assertNotIn("--steps", cmd)
        self.assertNotIn("<lora:", " ".join(cmd))

    def test_with_turbo_the_lora_is_named_in_the_prompt(self):
        """Syntaxe de sd.cpp : le nom SANS extension, dans le prompt, et le
        dossier via --lora-model-dir (forme documentée pour Wan 2.2)."""
        cmd = self._cmd(True)
        prompt = cmd[cmd.index("-p") + 1]
        self.assertIn("<lora:minimax_h3_ref2v_turbo_4step_v0.1_bf16:1>", prompt)
        self.assertNotIn(".safetensors", prompt)
        self.assertEqual(cmd[cmd.index("--lora-model-dir") + 1], "/m")
        self.assertEqual(cmd[cmd.index("--steps") + 1], "4")

    def test_the_prompt_refers_to_the_reference_picture(self):
        self.assertIn("<Picture 1>", T.PROMPT)

    def test_the_two_passes_write_to_different_files(self):
        self.assertNotEqual(T.out_path(False), T.out_path(True),
                            "la seconde passe écraserait la première")
        self.assertTrue(str(T.out_path(True)).endswith(".webm"))


class ReferencePickingTests(unittest.TestCase):
    def test_an_explicit_file_is_used(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            self.assertEqual(T.pick_reference(f.name), Path(f.name))

    def test_a_missing_explicit_file_is_refused_not_replaced(self):
        """Se rabattre en silence sur une autre image ferait tourner l'essai
        sur autre chose que ce qui était demandé."""
        self.assertIsNone(T.pick_reference("/nulle/part/x.png"))


if __name__ == "__main__":
    unittest.main()


class MemoryPlacementTests(unittest.TestCase):
    """Le premier essai réel a échoué là-dessus, et c'est arithmétique :
    Qwen3-VL 32B réclame 12 845 Mio de tampon de calcul, une RTX 3060 en a
    12 288 en tout. Carte vide, ça ne rentre pas."""

    def _gpu(self, index, vram, arch="ampere"):
        from atelier.hardware import Gpu
        return Gpu(index=index, name=f"carte{index}", vram_gb=vram,
                   arch=arch, tensor_cores=True)

    def test_two_cards_get_the_split_placement_first(self):
        """LE point manqué au premier essai : avec deux cartes, l'encodeur peut
        être RÉPARTI (12 + 11 = 23 Go) au lieu de tomber sur le processeur.
        C'est le placement le plus rapide, il doit passer en tête."""
        ladder = T.placements((self._gpu(0, 12.0), self._gpu(1, 11.0, "pascal")))
        self.assertEqual(ladder[0][1], ["--backend", "te=cuda0&cuda1"])
        self.assertIn("réparti", ladder[0][0])

    def test_the_split_uses_the_real_card_indices(self):
        """Coder « cuda0&cuda1 » en dur serait faux dès qu'une carte manque à
        l'appel ou que l'ordre change."""
        ladder = T.placements((self._gpu(0, 12.0), self._gpu(2, 24.0)))
        self.assertEqual(ladder[0][1], ["--backend", "te=cuda0&cuda2"])

    def test_a_single_card_falls_back_to_the_processor(self):
        ladder = T.placements((self._gpu(0, 11.0, "turing"),))
        self.assertEqual(ladder[0][1], ["--backend", "te=cpu"])
        self.assertTrue(all("cuda" not in " ".join(e) for _l, e in ladder))

    def test_the_ladder_always_ends_with_a_vram_budget(self):
        for gpus in ((self._gpu(0, 12.0),),
                     (self._gpu(0, 12.0), self._gpu(1, 11.0))):
            last = T.placements(gpus)[-1][1]
            self.assertIn("--max-vram", last)
            self.assertEqual(last[last.index("--max-vram") + 1], "-1")

    def test_the_budget_is_not_used_before_it_is_needed(self):
        """Un budget VRAM coûte en performance : il ne sert qu'en dernier
        recours, pas d'entrée de jeu."""
        for gpus in ((self._gpu(0, 12.0),),
                     (self._gpu(0, 12.0), self._gpu(1, 11.0))):
            self.assertNotIn("--max-vram", T.placements(gpus)[0][1])

    def test_offload_stays_on_top_of_the_placement(self):
        """`--offload-to-cpu` range les POIDS en RAM mais ramène le calcul sur
        le GPU : c'est ce qui a échoué. Il reste utile, il ne remplace pas le
        placement de l'encodeur."""
        cmd = T.build_cmd(Path("sd-cli"), _paths(), False, Path("o.webm"),
                          Path("/ref.png"),
                          placement=["--backend", "te=cpu"])
        self.assertIn("--offload-to-cpu", cmd)
        self.assertEqual(cmd[cmd.index("--backend") + 1], "te=cpu")

    def test_no_placement_is_baked_into_the_common_arguments(self):
        """Le placement se décide d'après le matériel : le figer dans
        BASE_ARGS ramènerait le défaut d'origine."""
        self.assertNotIn("--backend", T.BASE_ARGS)
        self.assertNotIn("--max-vram", T.BASE_ARGS)

    def test_out_of_memory_is_recognised_in_the_engine_output(self):
        """Les trois formulations relevées dans la sortie réelle du moteur."""
        for line in ("cudaMalloc failed: out of memory",
                     "alloc_tensor_range: failed to allocate CUDA0 buffer",
                     "ggml_backend_cuda_buffer_type_alloc_buffer: "
                     "allocating 12844.50 MiB on device 0: cudaMalloc failed"):
            self.assertTrue(any(sig in line.lower() for sig in T._OOM), line)

    def test_an_ordinary_error_is_not_taken_for_a_memory_problem(self):
        """Relancer avec un budget VRAM sur une erreur qui n'en est pas une
        ferait perdre une seconde passe entière."""
        for line in ("GGML_ASSERT(!hidden_states.empty()) failed",
                     "unknown argument: --nope"):
            self.assertFalse(any(sig in line.lower() for sig in T._OOM), line)
