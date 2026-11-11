"""Verify the MiniMax H3 Image package and its native sd-cli requests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atelier import model_catalog, registry, settings
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
        self.assertEqual(self.files["diffusion"].name, "minimax_h3_image_pruned_int8_convrot.safetensors")
        self.assertEqual(self.files["vae"].name, "minimax_h3_image_vae_fp16.safetensors")
        self.assertEqual(self.files["text_encoder"].relative_to(settings.MODELS_DIR).as_posix(),
                         "leejet__MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q2_K_M.gguf")

        # A retained Q4 video encoder is compatible with the automatic Q2 target.
        installed_encoder = self.files["text_encoder"]
        legacy_encoder = installed_encoder.with_name(installed_encoder.name.replace("Q2_K_M", "Q4_K_M"))
        installed_encoder.rename(legacy_encoder)
        self.assertEqual(generate.resolve_model_files(self.model, want_vision=True)["enc"], legacy_encoder)

    def test_native_generation_and_ordered_edit_use_the_fixed_pdd_recipe(self):
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

    def test_rejects_an_old_engine_and_unsupported_requests_before_inference(self):
        with patch.object(sdcpp, "run") as run:
            for changes, message in (
                ({"ref_image": [self.root / "ref.png"] * 17}, "at most 16"),
                ({"steps": 8}, "official PDD sigmas"),
                ({"init_image": self.root / "ref.png"}, "reference-image editor"),
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(sdcpp.EngineError, message):
                    generate.generate(**{**self.request, **changes})
            self.manifest.write_text('{"tag":"master-1087-f8417d0"}', encoding="utf-8")
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1088"):
                generate.generate(**self.request)
        run.assert_not_called()
