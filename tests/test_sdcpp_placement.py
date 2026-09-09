import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import benchmark, hardware, registry
from atelier.engine import generate, sdcpp


class ParamsBackendTests(unittest.TestCase):
    def _cmd(self, req, options):
        with patch.object(sdcpp, "_require", lambda *a, **k: None), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset(options)):
            return sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))

    def test_explicit_residency_wins_over_legacy_offload(self):
        """`--offload-to-cpu` est de la RÉSIDENCE : il cède à params-backend."""
        mapping = "diffusion=cuda0,vae=cuda0,te=cuda1"
        req = sdcpp.GenRequest(
            diffusion_model=Path("model.gguf"), params_backend=mapping,
            flags={"offload_to_cpu": True})
        cmd = self._cmd(req, {"--params-backend"})
        self.assertEqual(cmd[cmd.index("--params-backend") + 1], mapping)
        self.assertNotIn("--offload-to-cpu", cmd)

    def test_cpu_computation_survives_explicit_residency(self):
        """`--clip-on-cpu` est du CALCUL : il n'a pas à céder, et c'est le bug.

        La reprise après manque de VRAM demande justement le calcul CPU de
        l'encodeur. Quand une résidence explicite l'effaçait — donc sur toute
        machine multi-GPU — la seconde tentative relançait une commande
        IDENTIQUE à celle qui venait d'échouer, après avoir rechargé le modèle
        en entier. Elle échouait pareil, forcément.
        """
        req = sdcpp.GenRequest(
            diffusion_model=Path("model.gguf"),
            params_backend="diffusion=cuda0,vae=cuda0,te=cpu",
            flags={"clip_on_cpu": True})
        cmd = self._cmd(req, {"--params-backend", "--clip-on-cpu"})
        self.assertIn("--clip-on-cpu", cmd)

    def test_old_engine_keeps_the_compatible_offload(self):
        req = sdcpp.GenRequest(
            diffusion_model=Path("model.gguf"),
            params_backend="diffusion=cuda0,vae=cuda0,te=cuda1",
            flags={"offload_to_cpu": True})
        cmd = self._cmd(req, set())
        self.assertNotIn("--params-backend", cmd)
        self.assertIn("--offload-to-cpu", cmd)


class BenchmarkPlanTests(unittest.TestCase):
    def test_exact_combo_compares_resident_and_staged_encoder(self):
        gpus = (
            hardware.Gpu(0, "NVIDIA GeForce RTX 3060", 12.0, "ampere", True),
            hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal", False),
        )
        with patch.object(hardware, "auto_profile") as auto:
            auto.return_value.flags.return_value = {
                "diffusion_fa": True, "offload_to_cpu": True,
                "vae_tiling": True, "clip_on_cpu": False,
                "vae_on_cpu": False,
            }
            modes = benchmark.placement_candidates({"gpu_index": 0}, gpus)
        self.assertEqual([m.key for m in modes],
                         ["single-staged", "single-autofit",
                          "dual-resident", "dual-staged"])
        resident = modes[2].prefs_patch
        self.assertEqual(resident["params_backend"],
                         "diffusion=cuda0,vae=cuda0,te=cuda1")
        self.assertFalse(resident["flags"]["offload_to_cpu"])
        self.assertEqual(modes[3].prefs_patch["params_backend"], "*=cpu")

    def test_the_safest_profile_stays_first_so_a_tie_changes_nothing(self):
        """`_winner` départage les ex æquo en faveur du PREMIER de la liste.

        Un nouveau profil placé en tête gagnerait donc chaque égalité, et le
        banc d'essai déplacerait le comportement de la machine sans qu'aucune
        mesure ne le justifie.
        """
        gpus = (hardware.Gpu(0, "NVIDIA GeForce RTX 2080 Ti", 11.0,
                             "turing", True),)
        with patch.object(hardware, "auto_profile") as auto:
            auto.return_value.flags.return_value = {}
            modes = benchmark.placement_candidates({"gpu_index": 0}, gpus)
        self.assertEqual(modes[0].key, "single-staged")
        self.assertFalse(modes[0].prefs_patch["auto_fit"])

    def test_the_auto_fit_profile_carries_nothing_that_would_disable_it(self):
        """Auto-fit refuse de planifier dès qu'une affectation est présente.

        `--clip-on-cpu`, `--vae-on-cpu` et `--offload-to-cpu` en sont : l'amont
        les traduit en entrées de `--backend` / `--params-backend`. Un profil
        « auto-fit » qui en laisserait passer un mesurerait donc le placement
        d'à côté, et le rapport dirait le contraire de ce qui a tourné.
        """
        gpus = (hardware.Gpu(0, "NVIDIA GeForce RTX 3060", 12.0,
                             "ampere", True),)
        with patch.object(hardware, "auto_profile") as auto:
            auto.return_value.flags.return_value = {
                "offload_to_cpu": True, "clip_on_cpu": True,
                "vae_on_cpu": True, "diffusion_fa": True}
            modes = benchmark.placement_candidates({"gpu_index": 0}, gpus)
        patch_ = next(m.prefs_patch for m in modes if m.key == "single-autofit")
        self.assertTrue(patch_["auto_fit"])
        self.assertEqual(patch_["params_backend"], "")
        self.assertIsNone(patch_["encoder_gpu_index"])
        for legacy in ("offload_to_cpu", "clip_on_cpu", "vae_on_cpu"):
            self.assertFalse(patch_["flags"][legacy], legacy)
        # Ce qui ne relève PAS du placement doit survivre : le banc mesure
        # auto-fit, pas une machine déshabillée.
        self.assertTrue(patch_["flags"]["diffusion_fa"])


class PcieDetectionTests(unittest.TestCase):
    def tearDown(self):
        hardware.detect_gpus.cache_clear()
        hardware._apple_gpu.cache_clear()

    def test_link_width_is_reported_without_breaking_old_driver_fallback(self):
        responses = {
            "index,name,memory.total,compute_cap,driver_version":
                "0, NVIDIA GeForce RTX 3060, 12288, 8.6, 610.0\n",
            "index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current":
                "0, 00000000:01:00.0, 3, 16\n",
        }
        hardware.detect_gpus.cache_clear()
        with patch.object(hardware, "_apple_gpu", return_value=None), \
             patch.object(hardware, "_nvidia_smi",
                          side_effect=lambda fields: responses.get(fields)):
            gpu = hardware.detect_gpus()[0]
        self.assertEqual(gpu.pcie_label, "Gen3 x16")
        self.assertEqual(gpu.bus_id, "00000000:01:00.0")


if __name__ == "__main__":
    unittest.main()
