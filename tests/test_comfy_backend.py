"""Verify the optional ComfyUI generation backend, Settings toggle, and API workflows."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import yaml
from PIL import Image

from atelier import registry, settings
from atelier.engine import comfy_backend, generate, sdcpp
from atelier.ui import settings_tab


def _png_bytes(width: int, height: int, color=(80, 120, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, *, json_data=None, content: bytes = b"", status_code: int = 200):
        self._json = json_data
        self.content = content
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(json_data) if json_data is not None else ""

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class ComfyBackendTests(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temp)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR", "CUSTOM_DIR", "LORA_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.enterContext(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        settings.save_prefs({"generation_backend": "sdcpp", "comfyui_url": comfy_backend.DEFAULT_URL})

    def _install_model(self, model_id: str) -> tuple[registry.BaseModel, dict[str, Path]]:
        model = registry.get_base_model(model_id, {})
        files: dict[str, Path] = {}
        for comp in model.components:
            path = settings.model_repo_dir(comp.repo) / comp.requested()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weights")
            files[comp.role] = path
        return model, files

    def test_settings_tab_exposes_backend_toggle_and_writes_extra_model_paths(self):
        self._install_model("minimax-h3-image")
        with gr.Blocks() as demo:
            settings_tab.build_settings_tab()
        self.addCleanup(demo.close)

        backend = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "generation-backend")
        comfy_url = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "comfyui-url")
        self.assertEqual(backend.value, "sdcpp")
        self.assertEqual(comfy_url.value, "http://127.0.0.1:8188")

        on_change = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "on_backend_change")
        grp_upd, note, status_upd = on_change("comfyui", "127.0.0.1:8188/")
        self.assertTrue(grp_upd["visible"])
        self.assertIn("extra-model-paths-config", note)
        self.assertIn("ComfyUI", status_upd["value"])

        prefs = settings.load_prefs()
        self.assertEqual(prefs["generation_backend"], "comfyui")
        self.assertEqual(prefs["comfyui_url"], "http://127.0.0.1:8188")

        yaml_path = comfy_backend.extra_model_paths_file()
        self.assertTrue(yaml_path.is_file())
        parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        section = parsed["turbo_slop_generator"]
        self.assertIn("Comfy-Org__MiniMax-H3-Image", section["diffusion_models"])
        self.assertIn(str(settings.LORA_DIR.resolve()), section["loras"])

    def test_resolve_comfy_weight_matches_suffix_windows_slashes_and_basename(self):
        extra = comfy_backend.extra_model_paths_file()
        local = settings.MODELS_DIR / "Comfy-Org__MiniMax-H3-Image" / "diffusion_models" / "minimax_h3_image_pruned_w4a8.safetensors"
        avail = [
            "other\\minimax_h3_image_pruned_w4a8.safetensors",
            "Comfy-Org__MiniMax-H3-Image\\diffusion_models\\minimax_h3_image_pruned_w4a8.safetensors",
        ]
        self.assertEqual(
            comfy_backend.resolve_comfy_weight(local, avail, kind="diffusion model", extra_paths=extra),
            avail[1],
        )
        self.assertEqual(
            comfy_backend.resolve_comfy_weight(local, ["minimax_h3_image_pruned_w4a8.safetensors"],
                                               kind="diffusion model", extra_paths=extra),
            "minimax_h3_image_pruned_w4a8.safetensors",
        )
        with self.assertRaisesRegex(sdcpp.EngineError, "extra-model-paths-config"):
            comfy_backend.resolve_comfy_weight(local, ["unrelated.safetensors"],
                                               kind="diffusion model", extra_paths=extra)

    def test_build_workflow_covers_all_model_families_and_gguf_loaders(self):
        # 1) MiniMax H3 Image with reference + batch + LCM
        h3 = registry.get_base_model("minimax-h3-image", {})
        wf = comfy_backend.build_workflow(
            model=h3, diffusion_name="minimax_h3_image_pruned_w4a8.safetensors", diffusion_is_gguf=False,
            encoder_name="p-h3-image_text_encoder_int8_convrot.safetensors", encoder_is_gguf=False,
            vae_name="minimax_h3_image_vae_fp16.safetensors", loras=[],
            prompt="Portrait of <Picture 1>", negative="", width=2048, height=2048,
            steps=6, cfg_scale=1.0, sampler="lcm", schedule="auto", seed=42, batch_count=2,
            uploaded_refs=["ref1.png"], uploaded_init=None, strength=0.6, vae_tiling=True,
        )
        classes = {n["class_type"]: n["inputs"] for n in wf.values()}
        self.assertIn("UNETLoader", classes)
        self.assertEqual(classes["CLIPLoader"]["type"], "minimax")
        self.assertIn("MiniMaxH3ImageConditioning", classes)
        self.assertIn("images.image_1", classes["MiniMaxH3ImageConditioning"])
        self.assertIn("RepeatLatentBatch", classes)
        self.assertEqual(classes["KSampler"]["sampler_name"], "lcm")
        self.assertIn("VAEDecodeTiled", classes)

        # 2) Juggernaut Moss Turbo with LoRA + euler_a -> euler_ancestral
        jmoss = registry.get_base_model("moss-image-juggernaut-turbo", {})
        wf_jm = comfy_backend.build_workflow(
            model=jmoss, diffusion_name="juggernaut_moss_turbo_v1_int8_convrot.safetensors", diffusion_is_gguf=False,
            encoder_name="Qwen3.5-9B-Q4_K_M.gguf", encoder_is_gguf=True,
            vae_name="flux2-vae_small.safetensors", loras=[("style.safetensors", 0.8)],
            prompt="Cinematic shot", negative="", width=1344, height=1792,
            steps=6, cfg_scale=1.0, sampler="euler_a", schedule="simple", seed=7, batch_count=1,
            uploaded_refs=[], uploaded_init=None, strength=0.6, vae_tiling=False,
        )
        classes_jm = {n["class_type"]: n["inputs"] for n in wf_jm.values()}
        self.assertEqual(classes_jm["CLIPLoaderGGUF"]["type"], "moss_image")
        self.assertIn("LoraLoaderModelOnly", classes_jm)
        self.assertIn("EmptyFlux2LatentImage", classes_jm)
        self.assertEqual(classes_jm["KSampler"]["sampler_name"], "euler_ancestral")
        self.assertIn("VAEDecode", classes_jm)

        # 3) Flux.2 Klein GGUF + ReferenceLatent
        flux = registry.get_base_model("flux2-klein-9b", {})
        wf_fl = comfy_backend.build_workflow(
            model=flux, diffusion_name="flux-2-klein-9b-Q4_K_M.gguf", diffusion_is_gguf=True,
            encoder_name="Qwen3-8B-Q4_K_M.gguf", encoder_is_gguf=True,
            vae_name="flux2-vae.safetensors", loras=[],
            prompt="Edit photo", negative="", width=1024, height=1024,
            steps=4, cfg_scale=1.0, sampler="euler", schedule="auto", seed=1, batch_count=1,
            uploaded_refs=["ref.png"], uploaded_init=None, strength=0.6,
        )
        classes_fl = {n["class_type"]: n["inputs"] for n in wf_fl.values()}
        self.assertIn("UnetLoaderGGUF", classes_fl)
        self.assertIn("ReferenceLatent", classes_fl)

        # 4) Fibo 2.0 Edit Turbo + Qwen 2.1 Turbo
        fibo = registry.get_base_model("fibo2-edit-turbo", {})
        wf_fb = comfy_backend.build_workflow(
            model=fibo, diffusion_name="fibo2_edit_turbo_int8_convrot.safetensors", diffusion_is_gguf=False,
            encoder_name="Qwen3VL-4B-Instruct-Q4_K_M.gguf", encoder_is_gguf=True,
            vae_name="flux2-vae.safetensors", loras=[],
            prompt="Replace background", negative="", width=1024, height=1024,
            steps=4, cfg_scale=1.0, sampler="euler", schedule="auto", seed=2, batch_count=1,
            uploaded_refs=["ref1.png"], uploaded_init=None, strength=0.6,
        )
        self.assertIn("TextEncodeFibo2", {n["class_type"] for n in wf_fb.values()})

        qwen = registry.get_base_model("qwen-image-2.1-turbo", {})
        wf_qw = comfy_backend.build_workflow(
            model=qwen, diffusion_name="qwen_image_2.1_turbo_int8_convrot.safetensors", diffusion_is_gguf=False,
            encoder_name="Qwen3VL-8B-Instruct-Q4_K_M.gguf", encoder_is_gguf=True,
            vae_name="texture_fix_vae_for_qwen_image_2.1_bf16.safetensors", loras=[],
            prompt="Edit text", negative="", width=1024, height=1024,
            steps=8, cfg_scale=1.0, sampler="euler", schedule="auto", seed=3, batch_count=1,
            uploaded_refs=["ref1.png"], uploaded_init=None, strength=0.6,
        )
        self.assertIn("TextEncodeQwenImage21", {n["class_type"] for n in wf_qw.values()})

    def test_generate_routes_to_comfyui_without_sd_cli_and_preserves_h3_resize_and_sidecars(self):
        model, files = self._install_model("minimax-h3-image")
        ref = self.root / "input_ref.png"
        ref.write_bytes(_png_bytes(512, 512))
        settings.save_prefs({"generation_backend": "comfyui", "comfyui_url": "http://127.0.0.1:8188"})

        queued_workflows = []
        interrupted = []

        def fake_get(url, **kwargs):
            if url.endswith("/system_stats"):
                return _FakeResponse(json_data={
                    "system": {"comfyui_version": "0.3.99"},
                    "devices": [{"name": "RTX 3060", "vram_total": 12 * 1024**3}],
                })
            if url.endswith("/object_info/UNETLoader"):
                return _FakeResponse(json_data={
                    "UNETLoader": {"input": {"required": {"unet_name": [[files["diffusion"].name]]}}}
                })
            if url.endswith("/object_info/CLIPLoader"):
                return _FakeResponse(json_data={
                    "CLIPLoader": {"input": {"required": {"clip_name": [[files["text_encoder"].name]]}}}
                })
            if url.endswith("/object_info/VAELoader"):
                return _FakeResponse(json_data={
                    "VAELoader": {"input": {"required": {"vae_name": [[files["vae"].name]]}}}
                })
            if url.endswith("/object_info/MiniMaxH3ImageConditioning"):
                return _FakeResponse(json_data={"MiniMaxH3ImageConditioning": {"input": {"required": {}}}})
            if url.endswith("/history/prompt-123"):
                return _FakeResponse(json_data={
                    "prompt-123": {
                        "status": {"status_str": "success"},
                        "outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}},
                    }
                })
            if url.endswith("/view"):
                # Return a 4 MP 2048x2048 image; generate() must downscale to requested 1024x1024.
                return _FakeResponse(content=_png_bytes(2048, 2048))
            raise AssertionError(f"Unexpected GET {url}")

        def fake_post(url, **kwargs):
            if url.endswith("/upload/image"):
                return _FakeResponse(json_data={"name": "uploaded_ref.png", "subfolder": ""})
            if url.endswith("/prompt"):
                queued_workflows.append(kwargs["json"]["prompt"])
                return _FakeResponse(json_data={"prompt_id": "prompt-123"})
            if url.endswith("/interrupt"):
                interrupted.append(url)
                return _FakeResponse(json_data={})
            raise AssertionError(f"Unexpected POST {url}")

        logs = []
        with patch.object(settings, "find_sd_cli", return_value=None), \
             patch.object(comfy_backend.requests, "get", side_effect=fake_get), \
             patch.object(comfy_backend.requests, "post", side_effect=fake_post):
            paths = generate.generate(
                model_id="minimax-h3-image",
                prompt="Portrait of <Picture 1>",
                negative="",
                steps=6,
                cfg_scale=1.0,
                width=1024,
                height=1024,
                seed=99,
                batch_count=1,
                ref_image=[ref],
                save_prompt=True,
                log=logs.append,
            )
            cancel_msg = generate.cancel()

        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].is_file())
        with Image.open(paths[0]) as im:
            self.assertEqual(im.size, (1024, 1024))
        self.assertTrue(paths[0].with_suffix(".txt").is_file())
        meta = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual((meta["width"], meta["height"]), (1024, 1024))
        self.assertEqual((meta["canvas_width"], meta["canvas_height"]), (2048, 2048))
        self.assertEqual(len(queued_workflows), 1)
        self.assertTrue(any("ComfyUI" in line for line in logs), logs)
        self.assertEqual(len(interrupted), 1)
        self.assertIn("ComfyUI", cancel_msg)


if __name__ == "__main__":
    unittest.main()
