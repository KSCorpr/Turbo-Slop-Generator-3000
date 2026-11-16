"""Verify the MiniMax H3 Image package and its native sd-cli requests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atelier import downloader, model_catalog, registry, settings
from atelier.engine import generate, sdcpp

MODEL = "minimax-h3-image"


class MiniMaxH3ImageTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.manifest = settings.BIN_DIR / "engine-manifest.json"
        self.manifest.write_text('{"tag":"master-1088-32d516c"}', encoding="utf-8")
        self.enterContext(patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"))
        self.enterContext(patch.object(sdcpp, "binary_help", return_value="--sigmas minimax-h3-image --rng"))
        self.model = registry.get_base_model(MODEL, {})
        self.files = {}
        for component in self.model.components:
            path = settings.model_repo_dir(component.repo) / component.requested()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.files[component.role] = path
        self.request = dict(model_id=MODEL, prompt="A studio portrait", negative="", steps=6,
                            cfg_scale=1, width=1024, height=1024, seed=42, batch_count=1,
                            prefs_override={"auto_optimize": False}, save_prompt=False)

    def test_package_reuses_video_encoder_and_uses_its_own_image_vae(self):
        self.assertIsNotNone(model_catalog.get("image:" + MODEL))
        self.assertTrue(registry.model_is_ready(self.model))
        self.assertEqual(set(self.files), {"diffusion", "text_encoder", "vae"})
        self.assertEqual(self.files["diffusion"].relative_to(settings.MODELS_DIR).as_posix(),
                         "Comfy-Org__MiniMax-H3-Image/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors")
        self.assertEqual(self.files["vae"].name, "minimax_h3_image_vae_fp16.safetensors")
        self.assertEqual(self.files["text_encoder"].relative_to(settings.MODELS_DIR).as_posix(),
                         "leejet__MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")

        # Official Comfy-Org Safetensors diffusion checkpoints resolve cleanly.
        for selected in (
            "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
        ):
            prefs = {"model_files": {MODEL: {"diffusion": selected}}}
            comp = next(c for c in registry.get_base_model(MODEL, prefs).components if c.role == "diffusion")
            self.assertEqual((comp.repo, comp.requested()), ("Comfy-Org/MiniMax-H3-Image", selected))

        # Official sd.cpp text encoder alternatives (PrunaAI P-H3 4B, Comfy-Org 32B INT8, leejet GGUF) resolve cleanly.
        for selected, expected_repo in (
            ("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "leejet/MiniMax-H3-GGUF"),
            ("qwen3vl_32b_minimax_h3-Q2_K_M.gguf", "leejet/MiniMax-H3-GGUF"),
            ("text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors", "PrunaAI/P-H3-Image-Comfy"),
            ("text_encoders/p-h3-image_text_encoder_bf16.safetensors", "PrunaAI/P-H3-Image-Comfy"),
            ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3-Image"),
        ):
            prefs = {"model_files": {MODEL: {"text_encoder": selected}}}
            comp = next(c for c in registry.get_base_model(MODEL, prefs).components if c.role == "text_encoder")
            self.assertEqual((comp.repo, comp.requested()), (expected_repo, selected))

        # If only Q2_K_M is installed, downloader upgrades to Q4_K_M in automatic mode, while resolve_model_files can fall back if offline.
        installed_encoder = self.files["text_encoder"]
        q2_encoder = installed_encoder.with_name(installed_encoder.name.replace("Q4_K_M", "Q2_K_M"))
        installed_encoder.rename(q2_encoder)
        q2_encoder.write_bytes(b"q2")
        self.assertEqual(generate.resolve_model_files(self.model, want_vision=True)["enc"], q2_encoder)
        enc_comp = next(c for c in self.model.components if c.role == "text_encoder")
        repo_files = {"leejet/MiniMax-H3-GGUF": [installed_encoder.name, q2_encoder.name]}
        with patch("atelier.transfers.hf_hub_download", return_value=str(installed_encoder)) as fetch:
            result = downloader.download_component(enc_comp, _repo_files=repo_files)
        fetch.assert_called_once_with(repo_id="leejet/MiniMax-H3-GGUF",
                                      filename="qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                                      local_dir=str(installed_encoder.parent))
        self.assertEqual(result, installed_encoder)

    def test_native_generation_and_ordered_edit_use_sd_cpp_recipe_and_unlocked_steps(self):
        refs = [self.root / f"reference-{i}.png" for i in range(16)]
        for ref in refs:
            ref.touch()
        for ordered in ([], refs[::-1]):
            prompt = "Edit <Picture 1> using <Picture 2>." if ordered else self.request["prompt"]
            with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prompt": prompt, "width": 1025}, ref_image=ordered)
            command = run.call_args.args[0]
            for flag, value in (
                ("--diffusion-model", self.files["diffusion"]), ("--vae", self.files["vae"]),
                ("--llm", self.files["text_encoder"]), ("--sigmas", MODEL), ("--rng", "cpu"),
                ("--steps", 6), ("--sampling-method", "euler"), ("-W", 1056), ("-p", prompt),
            ):
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertIn("--vae-tiling", command)
            self.assertNotIn("--llm_vision", command)
            self.assertEqual([command[i+1] for i, arg in enumerate(command) if arg == "-r"],
                             [str(path) for path in ordered])

        # Custom step counts omit --sigmas minimax-h3-image so sd-cli interpolates the PDD schedule across N steps.
        with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
            generate.generate(**{**self.request, "steps": 12})
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--steps") + 1], "12")
        self.assertNotIn("--sigmas", command)

    def test_rejects_an_old_engine_and_unsupported_requests_before_inference(self):
        with patch.object(sdcpp, "run") as run:
            for changes, message in (
                ({"ref_image": [self.root / "ref.png"] * 17}, "at most 16"),
                ({"steps": 0}, "must be positive"),
                ({"init_image": self.root / "ref.png"}, "reference-image editor"),
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(sdcpp.EngineError, message):
                    generate.generate(**{**self.request, **changes})
            self.manifest.write_text('{"tag":"master-1087-f8417d0"}', encoding="utf-8")
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1088"):
                generate.generate(**self.request)
        run.assert_not_called()
