"""Texture Fix selection, downloads and generation; no GPU or network needed."""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import downloader, inventory, registry, settings
from atelier.engine import generate, sdcpp
from scripts import maintenance


MODEL = "qwen-image-2.1-turbo"
FIX_REPO = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
FIX_FILE = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
ORIGINAL_REPO = "Comfy-Org/Qwen-Image-2.1"
ORIGINAL_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def preferences(vae=None):
    return {"model_files": {MODEL: {"vae": vae} if vae else {}}}


def installed_file(repo, filename):
    path = settings.model_repo_dir(repo) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"installed weight")
    return path


class TextureVaeTests(unittest.TestCase):
    def test_original_selection_switches_repo_and_fix_can_be_selected_again(self):
        for filename, repo in ((ORIGINAL_FILE, ORIGINAL_REPO),
                               (FIX_FILE, FIX_REPO), (None, FIX_REPO),
                               ("../other.safetensors", FIX_REPO),
                               ("unapproved.safetensors", FIX_REPO)):
            with self.subTest(filename=filename):
                model = registry.get_base_model(MODEL, preferences(filename))
                vae = next(c for c in model.components if c.role == "vae")
                self.assertEqual(vae.repo, repo)
                self.assertEqual(vae.requested(),
                                 ORIGINAL_FILE if repo == ORIGINAL_REPO else FIX_FILE)
                self.assertEqual({v.repo for v in vae.alternatives},
                                 {FIX_REPO, ORIGINAL_REPO})

    def test_new_vae_download_reuses_every_other_installed_component(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch.object(settings, "ensure_dirs"):
            model = registry.get_base_model(MODEL, {})
            for comp in model.components:
                if comp.role != "vae":
                    installed_file(comp.repo, comp.requested())
            old = installed_file(ORIGINAL_REPO, ORIGINAL_FILE)

            def transfer(repo_id, filename, local_dir):
                self.assertEqual((repo_id, filename), (FIX_REPO, FIX_FILE))
                return str(installed_file(repo_id, filename))

            self.assertFalse(registry.model_is_ready(model))
            with patch("huggingface_hub.list_repo_files", return_value=[FIX_FILE]) as listing, \
                 patch("atelier.transfers.hf_hub_download", side_effect=transfer) as download:
                list(downloader.download_model(model))
            listing.assert_called_once_with(FIX_REPO)
            download.assert_called_once()
            self.assertTrue(old.is_file())
            self.assertTrue(registry.model_is_ready(model))
            with patch("huggingface_hub.list_repo_files", side_effect=AssertionError("offline")), \
                 patch("atelier.transfers.hf_hub_download", side_effect=AssertionError("duplicate")):
                list(downloader.download_model(model))

    def test_metadata_menu_lists_only_approved_vae_files_with_real_sizes(self):
        def info(repo_id, files_metadata):
            self.assertTrue(files_metadata)
            names = {FIX_REPO: [(FIX_FILE, 676_123_456),
                                ("diffusion_pytorch_model.safetensors", 1_352_000_000)],
                     ORIGINAL_REPO: [(ORIGINAL_FILE, 676_123_456),
                                     ("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", 7_260_000_000)]}
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=n, size=s, lfs=None)
                for n, s in names.get(repo_id, [])])

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch("huggingface_hub.HfApi") as api:
            installed_file(ORIGINAL_REPO, ORIGINAL_FILE)
            api.return_value.model_info.side_effect = info
            model = registry.get_base_model(MODEL, preferences(ORIGINAL_FILE))
            choices = downloader.weight_choices(model)["vae"]
        self.assertEqual([v for _, v in choices], [FIX_FILE, ORIGINAL_FILE])
        self.assertIn("0.68 GB", choices[0][0])
        self.assertIn("installed", choices[1][0])
        self.assertEqual(sum(c.kwargs["repo_id"] == ORIGINAL_REPO
                             for c in api.return_value.model_info.call_args_list), 1)

    def test_selected_vae_reaches_cli_for_generation_and_reference_editing(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)):
            for filename in (FIX_FILE, ORIGINAL_FILE):
                prefs = preferences(filename)
                model = registry.get_base_model(MODEL, prefs)
                for comp in model.components:
                    installed_file(comp.repo, comp.requested())
                for editing in (False, True):
                    with self.subTest(filename=filename, editing=editing):
                        files = generate.resolve_model_files(model, want_vision=editing)
                        expected = settings.model_repo_dir(
                            FIX_REPO if filename == FIX_FILE else ORIGINAL_REPO) / filename
                        req = sdcpp.GenRequest(
                            diffusion_model=files["diffusion"], vae=files["vae"],
                            text_encoder=files["enc"], llm_vision=files["llm_vision"],
                            ref_image=[Path("reference.png")] if editing else None)
                        with patch.object(sdcpp, "_require"), \
                             patch.object(sdcpp, "supported_options", return_value=frozenset()):
                            command = sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))
                        self.assertEqual(command[command.index("--vae") + 1], str(expected))
                        self.assertEqual("-r" in command, editing)

    def test_inventory_and_cleanup_keep_both_vae_repositories(self):
        prefs = preferences(ORIGINAL_FILE)
        model = registry.get_base_model(MODEL, prefs)
        self.assertIn(settings.model_repo_dir(FIX_REPO), inventory._model_dirs_for(model))
        with patch.object(settings, "load_prefs", return_value=prefs):
            self.assertIn(settings.model_repo_dir(FIX_REPO).name,
                          maintenance._expected_model_dirs())

    def test_ui_persists_original_then_resets_to_texture_fix_default(self):
        import gradio as gr
        from atelier.ui import library_tab

        prefs = preferences()
        downloaded = []
        load_models = registry.load_base_models

        def download(model, log):
            downloaded.append(next(c for c in model.components if c.role == "vae"))
            yield "ready"

        with patch.object(settings, "load_prefs", side_effect=lambda: deepcopy(prefs)), \
             patch.object(settings, "save_prefs", side_effect=lambda p: prefs.update(deepcopy(p))), \
             patch.object(registry, "load_base_models", side_effect=lambda p: [
                 m for m in load_models(p) if m.id == MODEL]), \
             patch.object(registry, "recommend", return_value={}), \
             patch.object(downloader, "download_model", side_effect=download):
            with gr.Blocks() as demo:
                library_tab.build_library_tab()
            handler = next(f.fn for f in demo.fns.values()
                           if getattr(f.fn, "__name__", "") == "handler")
            list(handler("image:" + MODEL, library_tab._AUTO, library_tab._AUTO, ORIGINAL_FILE))
            self.assertEqual(prefs["model_files"][MODEL]["vae"], ORIGINAL_FILE)
            list(handler("image:" + MODEL, library_tab._AUTO, library_tab._AUTO, library_tab._AUTO))
            self.assertNotIn(MODEL, prefs["model_files"])
            scanner = next(f.fn for f in demo.fns.values()
                           if getattr(f.fn, "__name__", "") == "scan_weights")
            with patch.object(downloader, "weight_choices", side_effect=RuntimeError("offline")):
                self.assertEqual(len(scanner("image:" + MODEL)), 4)
            with patch.object(downloader, "weight_choices", return_value={
                    "diffusion": [("Turbo", "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors")],
                    "vae": [("Texture Fix", FIX_FILE), ("Original", ORIGINAL_FILE)]}):
                updates = scanner("image:" + MODEL)
                self.assertEqual(len(updates), 4)
                self.assertEqual(updates[2]["value"], FIX_FILE)
        self.assertEqual([c.repo for c in downloaded], [ORIGINAL_REPO, FIX_REPO])
        menu, value = library_tab._weight_menu(
            registry.get_base_model(MODEL, preferences(ORIGINAL_FILE)), "vae")
        self.assertEqual(value, ORIGINAL_FILE)
        self.assertIn(FIX_FILE, [v for _, v in menu])


if __name__ == "__main__":
    unittest.main()
