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
                       if getattr(f.fn, "__name__", "") == "handler")
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
            list(handler("image:flux2-klein-9b", "flux-2-klein-9b-Q6_K.gguf",
                         "Qwen3-8B-Q4_K_M.gguf"))
        self.assertEqual(prefs["model_files"]["flux2-klein-9b"], {
            "diffusion": "flux-2-klein-9b-Q6_K.gguf",
            "text_encoder": "Qwen3-8B-Q4_K_M.gguf"})
        self.assertEqual(next(c for c in captured[0].components
                              if c.role == "diffusion").requested(),
                         "flux-2-klein-9b-Q6_K.gguf")

    def test_explicit_weight_is_the_one_loaded_after_download(self):
        prefs = {"model_files": {"flux2-klein-9b": {
            "diffusion": "flux-2-klein-9b-Q6_K.gguf",
            "text_encoder": "Qwen3-8B-Q4_K_M.gguf"}}}
        with patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            model = registry.get_base_model("flux2-klein-9b", prefs)
        comp = {c.role: c for c in model.components}
        self.assertEqual(comp["diffusion"].requested(),
                         "flux-2-klein-9b-Q6_K.gguf")
        self.assertEqual(comp["text_encoder"].requested(),
                         "Qwen3-8B-Q4_K_M.gguf")
        self.assertIsNone(comp["diffusion"].token)
        self.assertEqual(comp["diffusion"].base_glob(),
                         "*klein-9b**.gguf")
        # An invalid filename in persisted preferences never escapes the repo.
        prefs["model_files"]["flux2-klein-9b"]["diffusion"] = "../other.gguf"
        with patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            model = registry.get_base_model("flux2-klein-9b", prefs)
        self.assertEqual(next(c for c in model.components if c.role == "diffusion")
                         .requested(), "*klein-9b*Q4_K_M.gguf")

    def test_auto_encoder_reuses_a_matching_weight_even_if_quant_changed(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch.object(registry, "effective_quants",
                          return_value=("Q4_K_M", "Q8_0")):
            comp = next(c for c in registry.get_base_model(
                "flux2-klein-9b", {}).components if c.role == "text_encoder")
            old = settings.model_repo_dir(comp.repo) / \
                "Qwen3-8B-Q4_K_M.gguf"
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
            if "FLUX.2-klein-9B" in repo_id:
                return SimpleNamespace(siblings=[
                    sibling("flux-2-klein-9b-Q4_K_M.gguf", 6_000_000_000),
                    sibling("flux-2-klein-9b-Q6_K.gguf", 8_000_000_000)])
            return SimpleNamespace(siblings=[
                sibling("Qwen3-8B-Q4_K_M.gguf", 5_200_000_000),
                sibling("mmproj-Qwen3-8B-F16.gguf", 1_000_000_000)])

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = model_info
            model = registry.get_base_model("flux2-klein-9b", {
                "model_files": {"flux2-klein-9b": {
                    "diffusion": "flux-2-klein-9b-Q6_K.gguf"}}})
            ggufs = downloader.gguf_choices(model)
        self.assertEqual([value for _, value in ggufs["diffusion"]],
                         ["flux-2-klein-9b-Q4_K_M.gguf",
                          "flux-2-klein-9b-Q6_K.gguf"])
        self.assertIn("6.00 GB", ggufs["diffusion"][0][0])
        self.assertEqual([value for _, value in ggufs["text_encoder"]],
                         ["Qwen3-8B-Q4_K_M.gguf"])


if __name__ == "__main__":
    unittest.main()
