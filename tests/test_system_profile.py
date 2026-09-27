"""Model-specific system settings, including the actual CLI request."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import hardware, registry, settings, system_profile
from atelier.engine import generate, sdcpp


OPTIONS = frozenset({"--auto-fit", "--params-backend", "--max-vram",
                     "--stream-layers"})
RTX = hardware.Gpu(1, "RTX 2080 Ti", 11.0, "turing", True)
PASCAL = hardware.Gpu(0, "GTX 1080 Ti", 11.0, "pascal", False)


def weight(path, gib):
    with open(path, "wb") as fh:
        fh.truncate(int(gib * 1024 ** 3))  # sparse: no multi-GB download
    return path


class AutomaticSystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.image = weight(root / "image.gguf", 6.6)
        self.vae = weight(root / "vae.safetensors", 0.2)
        self.encoder = weight(root / "encoder.gguf", 3.5)
        self.files = {"diffusion": self.image, "model_path": None,
                      "vae": self.vae, "enc": self.encoder,
                      "tokenizer": None, "uncond": None, "t5xxl": None,
                      "clip_l": None, "llm_vision": None}
        self.model = registry.BaseModel(
            "z-image-turbo", "Z-Image", "z_image", [], "", [],
            {"steps": 8, "sampler": "euler"}, 6, [])
        self.prefs = {**settings.DEFAULT_PREFS, "system_mode": "auto"}
        self.gpus = patch.object(hardware, "detect_gpus",
                                 return_value=(PASCAL, RTX))
        self.ram = patch.object(hardware, "detect_ram_gb", return_value=64)
        self.options = patch.object(sdcpp, "supported_options",
                                    return_value=OPTIONS)
        self.gpus.start(); self.addCleanup(self.gpus.stop)
        self.ram.start(); self.addCleanup(self.ram.stop)
        self.options.start(); self.addCleanup(self.options.stop)

    def plan(self, **kw):
        return system_profile.plan_for_model(
            self.model, self.files, self.prefs, Path("sd-cli"),
            available_gib=9.5, **kw)

    def test_small_model_keeps_diffusion_on_rtx_and_encoder_in_ram(self):
        plan = self.plan()
        self.assertEqual(plan.gpu_index, RTX.index)
        self.assertEqual(plan.params_backend,
                         "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertFalse(plan.flags["offload_to_cpu"])
        self.assertFalse(plan.auto_fit)
        self.assertEqual(plan.max_vram, "")

    def test_large_model_lets_sdcpp_fit_weights_and_sets_budget(self):
        weight(self.image, 11.5)
        plan = self.plan()
        self.assertTrue(plan.auto_fit)
        self.assertEqual(plan.params_backend, "")
        self.assertEqual(plan.max_vram, "-1")

    def test_high_resolution_and_reference_use_more_headroom(self):
        plan = self.plan(width=2048, height=2048, has_reference=True)
        self.assertTrue(plan.auto_fit)
        self.assertTrue(plan.flags["vae_tiling"])

    def test_vision_encoder_counts_toward_reference_image_budget(self):
        self.files["llm_vision"] = weight(Path(self.tmp.name) / "vision.gguf", 2.0)
        self.assertFalse(self.plan().auto_fit)
        self.assertTrue(self.plan(has_reference=True).auto_fit)

    def test_legacy_manual_card_does_not_pin_automatic_card(self):
        self.prefs["gpu_index"] = PASCAL.index
        self.assertEqual(self.plan().gpu_index, RTX.index)
        self.prefs["auto_gpu_index"] = PASCAL.index
        self.assertEqual(self.plan().gpu_index, PASCAL.index)
        self.assertEqual(settings.generation_gpu_index(self.prefs), PASCAL.index)
        self.prefs["gpu_index"] = RTX.index
        self.prefs["system_mode"] = "manual"
        self.assertEqual(settings.generation_gpu_index(self.prefs), RTX.index)

    def test_older_engine_falls_back_to_ram_instead_of_unknown_flags(self):
        with patch.object(sdcpp, "supported_options", return_value=frozenset()):
            plan = self.plan()
        self.assertEqual(plan.params_backend, "")
        self.assertTrue(plan.flags["offload_to_cpu"])

    def test_explicit_catalog_weight_survives_automatic_quantization(self):
        selected = {**self.prefs, "quant": "Q3_K_S",
                    "model_files": {"z-image-turbo": {
                        "diffusion": "z_image_turbo-Q8_0.gguf"}}}
        model = registry.get_base_model("z-image-turbo", selected)
        diffusion = next(c for c in model.components if c.role == "diffusion")
        self.assertEqual(diffusion.requested(), "z_image_turbo-Q8_0.gguf")
        self.assertEqual(registry.effective_quants(selected)[0], "Q4_K_M")

    def test_generation_rechecks_actual_weight_and_ignores_stale_manual_flags(self):
        calls = []

        def capture(_cli, req, _out):
            calls.append(req)
            return ["sd-cli"]

        with patch.object(generate.settings, "load_prefs", return_value=self.prefs), \
             patch.object(generate.settings, "find_sd_cli",
                          return_value=Path("sd-cli")), \
             patch.object(generate.registry, "get_base_model",
                          return_value=self.model), \
             patch.object(generate, "resolve_model_files",
                          return_value=self.files), \
             patch.object(hardware, "free_vram_gb", return_value=9.5), \
             patch.object(sdcpp, "build_gen_cmd", side_effect=capture), \
             patch.object(sdcpp, "run"), \
             patch.object(sdcpp, "collect_outputs", return_value=[]):
            self.prefs["auto_optimize"] = False
            self.prefs["auto_fit"] = True
            self.prefs["params_backend"] = "*=cpu"
            self.prefs["flags"] = {"offload_to_cpu": True}
            generate.generate("z-image-turbo", "a cat", "", 8, 1.0,
                              1024, 1024, 42, 1, save_prompt=False)
            weight(self.image, 11.5)
            generate.generate("z-image-turbo", "a cat", "", 8, 1.0,
                              1024, 1024, 42, 1, save_prompt=False)
        self.assertFalse(calls[0].auto_fit)
        self.assertEqual(calls[0].params_backend,
                         "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertTrue(calls[1].auto_fit)
        self.assertEqual(calls[1].params_backend, "")


if __name__ == "__main__":
    unittest.main()
