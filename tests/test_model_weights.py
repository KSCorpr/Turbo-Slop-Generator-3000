"""GGUF sizes, explicit weights and reuse of compatible encoders."""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import downloader, registry, settings


class ModelWeightTests(unittest.TestCase):
    def test_catalog_download_uses_the_selected_weight_after_ui_save(self):
        import app
        from atelier.ui import library_tab

        demo = app.build_app()
        handler = next(f.fn for f in demo.fns.values()
                       if getattr(f.fn, "__name__", "") == "handler" and
                       getattr(f.fn, "__closure__", None) and any(
                           k == "model_id" and c.cell_contents == "qwen-image-2.1"
                           for k, c in zip(f.fn.__code__.co_freevars,
                                           f.fn.__closure__)))
        prefs = {"model_files": {}}
        captured = []

        def download(model, log):
            captured.append(model)
            yield "downloaded"

        with patch.object(settings, "load_prefs", side_effect=lambda: deepcopy(prefs)), \
             patch.object(settings, "save_prefs",
                          side_effect=lambda data: prefs.update(deepcopy(data))), \
             patch.object(library_tab.downloader, "download_model",
                          side_effect=download), \
             patch.object(library_tab.registry, "recommend", return_value={}):
            list(handler("qwen_image_2.1-Q6_K.gguf",
                         "Qwen3VL-8B-Instruct-Q4_K_M.gguf"))
        self.assertEqual(prefs["model_files"]["qwen-image-2.1"], {
            "diffusion": "qwen_image_2.1-Q6_K.gguf",
            "text_encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"})
        self.assertEqual(next(c for c in captured[0].components
                              if c.role == "diffusion").requested(),
                         "qwen_image_2.1-Q6_K.gguf")

    def test_explicit_weight_is_the_one_loaded_after_download(self):
        prefs = {"model_files": {"qwen-image-2.1": {
            "diffusion": "qwen_image_2.1-Q6_K.gguf",
            "text_encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"}}}
        with patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            model = registry.get_base_model("qwen-image-2.1", prefs)
        comp = {c.role: c for c in model.components}
        self.assertEqual(comp["diffusion"].requested(),
                         "qwen_image_2.1-Q6_K.gguf")
        self.assertEqual(comp["text_encoder"].requested(),
                         "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
        self.assertIsNone(comp["diffusion"].token)
        self.assertEqual(comp["diffusion"].base_glob(),
                         "qwen_image_2.1-*.gguf")
        # An invalid filename in persisted preferences never escapes the repo.
        prefs["model_files"]["qwen-image-2.1"]["diffusion"] = "../other.gguf"
        with patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            model = registry.get_base_model("qwen-image-2.1", prefs)
        self.assertEqual(next(c for c in model.components if c.role == "diffusion")
                         .requested(), "qwen_image_2.1-Q4_K.gguf")

    def test_auto_encoder_reuses_a_matching_weight_even_if_quant_changed(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            comp = next(c for c in registry.get_base_model(
                "qwen-image-2.1", {}).components if c.role == "text_encoder")
            old = settings.model_repo_dir(comp.repo) / \
                "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"installed weight")
            with patch("huggingface_hub.list_repo_files",
                       side_effect=AssertionError("unneeded hub query")), \
                 patch("huggingface_hub.hf_hub_download",
                       side_effect=AssertionError("duplicate download")):
                self.assertEqual(downloader.download_component(comp), old)

    def test_actual_hub_sizes_and_installed_status_in_gguf_menu(self):
        def sibling(name, size):
            return SimpleNamespace(rfilename=name, size=size, lfs=None)

        def model_info(repo_id, files_metadata):
            self.assertTrue(files_metadata)
            if "Qwen-Image-2.1" in repo_id:
                return SimpleNamespace(siblings=[
                    sibling("qwen_image_2.1-Q4_K.gguf", 6_000_000_000),
                    sibling("qwen_image_2.1-Q6_K.gguf", 8_000_000_000)])
            return SimpleNamespace(siblings=[
                sibling("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5_200_000_000),
                sibling("mmproj-Qwen3VL-8B-Instruct-F16.gguf", 1_000_000_000)])

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = model_info
            model = registry.get_base_model("qwen-image-2.1", {
                "model_files": {"qwen-image-2.1": {
                    "diffusion": "qwen_image_2.1-Q6_K.gguf"}}})
            ggufs = downloader.gguf_choices(model)
        self.assertEqual([value for _, value in ggufs["diffusion"]],
                         ["qwen_image_2.1-Q4_K.gguf",
                          "qwen_image_2.1-Q6_K.gguf"])
        self.assertIn("6.00 GB", ggufs["diffusion"][0][0])
        self.assertEqual([value for _, value in ggufs["text_encoder"]],
                         ["Qwen3VL-8B-Instruct-Q4_K_M.gguf"])


if __name__ == "__main__":
    unittest.main()
