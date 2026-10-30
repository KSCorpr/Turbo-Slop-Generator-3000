"""Retirement keeps user files recoverable and preserves the active SAM cutout."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from atelier import addons, inventory, registry, settings
from atelier.engine import generate, sdcpp, tools
from atelier.ui import manage_tab


class RetiredFeatureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, path in {"ROOT": self.root, "MODELS_DIR": self.root / "models",
                           "OUTPUT_DIR": self.root / "outputs", "TMP_DIR": self.root / "tmp",
                           "USERDATA_DIR": self.root / "userdata"}.items():
            self.stack.enter_context(patch.object(settings, name, path))
        self.stack.enter_context(patch.object(settings, "ensure_dirs"))

    def file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"user data")
        return path

    def test_cleanup_is_visible_optional_and_keeps_shared_models_and_outputs(self):
        retired = self.file(settings.model_repo_dir("akhaliq/flux-3-image-klein-sd-gguf") / "old.gguf")
        zimage = self.file(settings.model_repo_dir("Comfy-Org/z_image_turbo") / "vae.safetensors")
        custom = self.file(self.root / "tools_repo/upscale/checkpoints/personal.safetensors")
        shared_encoder = self.file(settings.model_repo_dir("Qwen/Qwen3-VL-4B-Instruct-GGUF") / "encoder.gguf")
        shared_vae = self.file(settings.model_repo_dir("Comfy-Org/flux2-klein-9B") / "vae.safetensors")
        cutout = self.file(addons.root("sam2") / "model/model.safetensors")
        output = self.file(settings.OUTPUT_DIR / "old-result.png")
        items = inventory.items({})
        legacy = {item.key: item for item in items if item.category == "Removed features"}
        self.assertEqual(set(legacy), {"retired_flux3", "retired_zimage_native", "retired_upscale"})
        self.assertTrue(legacy["retired_upscale"].protected)
        self.assertTrue(all(p.exists() for p in (retired, zimage, custom)))
        choices, summary = manage_tab._choices_and_summary()
        self.assertIn("Removed features", summary)
        self.assertIn("retired_flux3", {key for _, key in choices})
        self.assertTrue(any("⚠️" in label and key == "retired_upscale" for label, key in choices))
        inventory.delete(["retired_flux3", "retired_zimage_native"], {})
        self.assertFalse(retired.exists())
        self.assertFalse(zimage.exists())
        self.assertTrue(all(p.exists() for p in (shared_encoder, shared_vae, cutout, output, custom)))

    def test_retired_models_cannot_start_engine_even_with_old_saved_preferences(self):
        for model in ("flux3-klein", "z-image-turbo"):
            prefs = {"model_files": {model: {"diffusion": "old.gguf"}}}
            self.assertIsNone(registry.get_base_model(model, prefs))
            with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                 patch.object(sdcpp, "run") as run, self.assertRaisesRegex(sdcpp.EngineError, "Unknown model"):
                generate.generate(model, "test", "", 8, 1, 1024, 1024, 42, 1, prefs_override=prefs)
            run.assert_not_called()

    def test_sam2_cutout_still_uses_its_existing_runtime_cache_and_generation_gpu(self):
        def child(command, *args, **kwargs):
            self.assertEqual(command[0], str(addons.python("sam2")))
            self.assertEqual(command[command.index("--backend") + 1], "sam2")
            self.assertEqual(command[command.index("--model-dir") + 1], str(addons.root("sam2") / "model"))
            self.assertEqual(kwargs["gpu_index"], 1)
            dest = Path(command[command.index("--output-dir") + 1])
            dest.mkdir(exist_ok=True)
            Image.new("RGBA", (32, 32), "red").save(dest / "source_sam.png")
        settings.TMP_DIR.mkdir()
        settings.OUTPUT_DIR.mkdir()
        with patch.object(addons, "ready", return_value=True), \
             patch.object(tools, "_run_tool", side_effect=child), \
             patch.object(tools, "_gen_gpu_index", return_value=1):
            cut, _ = tools.sam_segment(Image.new("RGB", (32, 32)), 10, 10, backend="sam2")
        self.assertTrue(cut.is_file())
        with Image.open(cut) as image:
            self.assertEqual(image.mode, "RGBA")

    def test_sam2_installer_route_does_not_offer_layer_models(self):
        from atelier import model_catalog
        from scripts import setup_sam2
        with patch.object(model_catalog.JOB, "run") as run, \
             patch.object(model_catalog, "ready", return_value=True):
            model_catalog.install("addon:sam2", log=lambda _: None)
        self.assertTrue(str(run.call_args.args[0][-2]).endswith("setup_sam2.py"))
        self.assertEqual(run.call_args.args[0][-1], "sam2")
        with self.assertRaises(ValueError):
            setup_sam2.install("qwen-layered")
