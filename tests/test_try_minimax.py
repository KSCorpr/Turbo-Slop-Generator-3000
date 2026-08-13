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

    def _cmd(self, turbo, budget=False):
        return T.build_cmd(Path("sd-cli"), self.paths, turbo,
                           Path("out.webm"), Path("/ref.png"), budget=budget)

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

    def _cmd(self, budget=False):
        return T.build_cmd(Path("sd-cli"), _paths(), False,
                           Path("out.webm"), Path("/ref.png"), budget=budget)

    def test_the_text_encoder_runs_on_the_cpu(self):
        cmd = self._cmd()
        self.assertIn("--backend", cmd)
        self.assertEqual(cmd[cmd.index("--backend") + 1], "te=cpu")

    def test_offload_alone_is_not_relied_upon(self):
        """`--offload-to-cpu` range les POIDS en RAM mais ramène le calcul sur
        le GPU : c'est exactement ce qui a échoué. Il reste utile, il ne
        remplace pas le placement de l'encodeur."""
        cmd = self._cmd()
        self.assertIn("--offload-to-cpu", cmd)
        self.assertIn("te=cpu", cmd)

    def test_the_vram_budget_is_only_for_the_retry(self):
        self.assertNotIn("--max-vram", self._cmd(budget=False))
        cmd = self._cmd(budget=True)
        self.assertEqual(cmd[cmd.index("--max-vram") + 1], "-1")

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
