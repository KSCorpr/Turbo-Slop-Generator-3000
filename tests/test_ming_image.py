"""Ming Image integration: checked weights, tokenizer routing and engine age."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import downloader, registry, sampling, settings
from atelier.engine import generate, sdcpp


class MingImageTests(unittest.TestCase):
    def test_exact_variant_selection_and_required_files(self):
        model = registry.get_base_model("ming-image-design", {})
        self.assertEqual(model.family, "ming_image")
        self.assertEqual((model.defaults["steps"], model.defaults["cfg_scale"]),
                         (12, 1.0))
        components = {c.role: c for c in model.components}
        self.assertEqual(set(components),
                         {"diffusion", "text_encoder", "vae", "tokenizer"})
        self.assertEqual(components["diffusion"].requested(),
                         "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors")
        self.assertEqual(components["tokenizer"].requested(),
                         "mllm/tokenizer.json")
        self.assertIn("12 steps", sampling.rationale("ming_image"))

        chosen = {"model_files": {"ming-image-design": {
            "diffusion": "diffusion_models/ming_image_0.1_design_bf16.safetensors"}}}
        selected = registry.get_base_model("ming-image-design", chosen)
        self.assertEqual(selected.components[0].requested(),
                         "diffusion_models/ming_image_0.1_design_bf16.safetensors")
        # A forged file cannot redirect a download to another checkpoint.
        chosen["model_files"]["ming-image-design"]["diffusion"] = \
            "diffusion_models/ming_image_0.1_design_layer_bf16.safetensors"
        rejected = registry.get_base_model("ming-image-design", chosen)
        self.assertIn("int8_convrot", rejected.components[0].requested())

    def test_hub_sizes_exclude_layer_variants_and_share_one_repo_lookup(self):
        def sibling(filename, size):
            return SimpleNamespace(rfilename=filename, size=size, lfs=None)

        def info(repo_id, files_metadata):
            self.assertTrue(files_metadata)
            if repo_id == "Comfy-Org/Ming-Image":
                return SimpleNamespace(siblings=[
                    sibling("diffusion_models/ming_image_0.1_design_bf16.safetensors", 12_300_000_000),
                    sibling("diffusion_models/ming_image_0.1_design_int8_convrot.safetensors", 6_180_000_000),
                    sibling("diffusion_models/ming_image_0.1_design_layer_bf16.safetensors", 12_300_000_000),
                    sibling("text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors", 36_700_000_000),
                ])
            raise AssertionError(f"unexpected repo {repo_id}")

        with patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = info
            choices = downloader.weight_choices(
                registry.get_base_model("ming-image-design", {}))
        self.assertEqual(hub.return_value.model_info.call_count, 1)
        self.assertEqual(len(choices["diffusion"]), 2)
        self.assertEqual({c[1] for c in choices["diffusion"]}, {
            "diffusion_models/ming_image_0.1_design_bf16.safetensors",
            "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors",
        })
        self.assertTrue(any("6.18 GB" in label
                            for label, _ in choices["diffusion"]))
        self.assertIn("36.70 GB", choices["text_encoder"][0][0])

    def test_downloader_only_requests_selected_dit_and_required_files(self):
        model = registry.get_base_model("ming-image-design", {
            "model_files": {"ming-image-design": {
                "diffusion": "diffusion_models/ming_image_0.1_design_bf16.safetensors"}}})
        listed = {
            "Comfy-Org/Ming-Image": [c.requested() for c in model.components
                                      if c.repo == "Comfy-Org/Ming-Image"],
            "inclusionAI/Ming-Image-0.1-Design": ["mllm/tokenizer.json"],
        }
        listed["Comfy-Org/Ming-Image"].append(
            "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors")
        requested = []

        def fake_download(repo_id, filename, local_dir):
            requested.append((repo_id, filename))
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            return str(path)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch("huggingface_hub.list_repo_files",
                   side_effect=lambda repo: listed[repo]) as list_files, \
             patch("huggingface_hub.hf_hub_download",
                   side_effect=fake_download):
            lines = list(downloader.download_model(model))
            self.assertTrue(registry.model_is_ready(model))
            self.assertEqual(list_files.call_count, 2)
        self.assertIn("ready", lines[-1])
        self.assertEqual(len(requested), 4)
        self.assertEqual(requested[0][1],
                         "diffusion_models/ming_image_0.1_design_bf16.safetensors")
        self.assertEqual(requested[-1][1], "mllm/tokenizer.json")

    def test_command_passes_ling_tokenizer_and_png_preview_every_step(self):
        with patch.object(sdcpp, "_require"), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()):
            req = sdcpp.GenRequest(
                diffusion_model=Path("ming_image_0.1_design_int8_convrot.safetensors"),
                text_encoder=Path("ming_image_0.1_ling_mini_2.0_bf16.safetensors"),
                vae=Path("ming_image_vae_bf16.safetensors"),
                tokenizer=Path("tokenizer.json"),
                preview_path=Path("preview_%03d.png"), preview_method="vae",
                preview_interval=1, steps=12, cfg_scale=1.0)
            cmd = sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))
        for key, expected in (("--llm", req.text_encoder),
                              ("--tokenizer", req.tokenizer),
                              ("--vae", req.vae),
                              ("--diffusion-model", req.diffusion_model),
                              ("--preview", "vae"),
                              ("--preview-interval", "1"),
                              ("-o", "out.png")):
            self.assertEqual(cmd[cmd.index(key) + 1], str(expected))

    def test_generation_resolves_catalog_tokenizer_and_preserves_alpha_png(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            model = registry.get_base_model("ming-image-design", {})
            paths = {}
            for component in model.components:
                path = temp / component.requested()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                paths[component.role] = path
            cli = temp / "sd-cli"
            cli.write_bytes(b"fixture")
            out = temp / "ming_image.png"
            captured = []

            def run(cmd, **_kwargs):
                captured.extend(cmd)
                Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(out)

            with patch.object(settings, "BIN_DIR", temp), \
                 patch.object(settings, "find_sd_cli", return_value=cli), \
                 patch.object(registry, "resolve_component_path",
                              side_effect=lambda comp: paths.get(comp.role)), \
                 patch.object(generate, "_resolved_flags",
                              return_value=({}, None)), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset({"--tokenizer"})), \
                 patch.object(sdcpp, "run", side_effect=run), \
                 patch.object(sdcpp, "unique_output", return_value=out):
                result = generate.generate(
                    "ming-image-design", "cat sticker", "", 12, 1.0,
                    1024, 1024, 42, 1,
                    preview_path=temp / "preview_%03d.png",
                    save_prompt=False)
            self.assertEqual(result, [out])
            self.assertEqual(captured[captured.index("--tokenizer") + 1],
                             str(paths["tokenizer"]))
            self.assertEqual(captured[captured.index("--preview") + 1], "vae")
            with Image.open(out) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_old_official_build_rejected_even_with_tokenizer_flag(self):
        model = registry.get_base_model("ming-image-design", {})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "BIN_DIR", Path(tmp)), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--tokenizer"})):
            manifest = Path(tmp) / "engine-manifest.json"
            manifest.write_text(json.dumps({"tag": "master-922-9947eeb"}))
            with self.assertRaisesRegex(sdcpp.EngineError, "build 924"):
                generate._check_ming_engine(model, Path("sd-cli"))
            manifest.write_text(json.dumps({"tag": "master-924-f8890b9"}))
            generate._check_ming_engine(model, Path("sd-cli"))

    def test_generation_tab_offers_text_and_transparency_without_image_upload(self):
        import gradio as gr
        import app

        demo = app.build_app()
        tab = next(block for block in demo.blocks.values()
                   if isinstance(block, gr.Tab) and block.id == "ming-image-design")

        def in_tab(block):
            node = block.parent
            while node is not None:
                if node is tab:
                    return True
                node = node.parent
            return False

        blocks = [b for b in demo.blocks.values() if in_tab(b)]
        self.assertFalse(any(isinstance(b, gr.Image)
                             and b.label in ("Starting image", "Image to edit")
                             for b in blocks))
        self.assertTrue(any(isinstance(b, gr.Accordion)
                            and b.label == "Transparent PNG (RGBA)"
                            for b in blocks))
        self.assertEqual(next(b for b in blocks if isinstance(b, gr.Slider)
                              and b.label == "Steps").value, 12)


if __name__ == "__main__":
    unittest.main()
