"""Unit tests for MiniMax H3 Image new weights, MiniMax H3 Video, and Steppe-Wan2.1."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier import addons, h3_video, model_catalog, registry, settings, steppe
from atelier.engine import generate, h3_video as h3v_engine, sdcpp, steppe as steppe_engine


class H3ImageNewWeightsTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR", "CUSTOM_DIR", "LORA_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.enterContext(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        self.manifest = settings.BIN_DIR / "engine-manifest.json"
        self.manifest.write_text('{"tag":"master-1117-0d71f96"}', encoding="utf-8")
        self.enterContext(patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"))
        self.enterContext(patch.object(sdcpp, "binary_help", return_value="--sigmas minimax-h3-image --rng"))

    def test_w4a8_and_unsloth_gguf_weights_resolve_and_guard_old_engine(self):
        for selected, expected_repo in (
            ("diffusion_models/minimax_h3_image_pruned_w4a8.safetensors", "Comfy-Org/MiniMax-H3-Image"),
            ("minimax_h3_image_pruned-UD-Q4_K_XL.gguf", "unsloth/MiniMax-H3-Image-GGUF"),
            ("minimax_h3_image_pruned-UD-Q3_K_XL.gguf", "unsloth/MiniMax-H3-Image-GGUF"),
        ):
            prefs = {"model_files": {"minimax-h3-image": {"diffusion": selected}}}
            comp = next(c for c in registry.get_base_model("minimax-h3-image", prefs).components if c.role == "diffusion")
            self.assertEqual((comp.repo, comp.requested()), (expected_repo, selected))

        prefs_w4a8_enc = {
            "model_files": {
                "minimax-h3-image": {
                    "diffusion": "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
                    "text_encoder": "text_encoders/p-h3-image_text_encoder_w4a8.safetensors",
                }
            }
        }
        model = registry.get_base_model("minimax-h3-image", prefs_w4a8_enc)
        for comp in model.components:
            p = settings.model_repo_dir(comp.repo) / comp.requested()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"ok")

        # Older engine (< 1110) must be rejected before launching sd-cli for W4A8
        self.manifest.write_text('{"tag":"master-1098-b521db4"}', encoding="utf-8")
        with patch.object(sdcpp, "run") as run:
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1110"):
                generate.generate(
                    model_id="minimax-h3-image",
                    prompt="test",
                    negative="",
                    steps=6,
                    cfg_scale=1.0,
                    width=2048,
                    height=2048,
                    seed=42,
                    batch_count=1,
                    prefs_override={**prefs_w4a8_enc, "auto_optimize": False},
                    save_prompt=False,
                )
        run.assert_not_called()


class MiniMaxH3VideoTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.enterContext(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        settings.save_prefs({"system_mode": "auto"})
        self.manifest = settings.BIN_DIR / "engine-manifest.json"
        self.manifest.write_text('{"tag":"master-1117-0d71f96"}', encoding="utf-8")
        self.enterContext(patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"))
        self.help_flags = (
            "--mode --audio-vae --video-frames --fps --sigmas --end-img --init-img "
            "--temporal-tiling --model-args --auto-fit on|off --max-vram --diffusion-fa "
            "--vae-tiling --hires --hires-upscaler --hires-width --hires-height --hires-steps "
            "--hires-denoising-strength --hires-sigmas --rng"
        )
        self.enterContext(patch.object(sdcpp, "binary_help", return_value=self.help_flags))

    def test_grid_alignment_and_catalog_registration(self):
        self.assertEqual(h3v_engine.align_dimension(850), 864)
        self.assertEqual(h3v_engine.align_dimension(480), 480)
        self.assertEqual(h3v_engine.align_frames(1), 5)
        self.assertEqual(h3v_engine.align_frames(5), 5)
        self.assertEqual(h3v_engine.align_frames(50), 56)
        self.assertEqual(h3v_engine.align_frames(56), 56)
        self.assertIsNotNone(model_catalog.get("video:minimax-h3"))

    def test_generate_video_builds_hyperflow_cmd_with_11gb_memory_guards(self):
        for entry in (
            h3_video.DIFFUSION_VARIANTS["hyperflow-fl2va-q4"],
            h3_video.ENCODER_VARIANTS["p-h3-int8"],
            h3_video.VIDEO_VAE_VARIANTS["sol-int8"],
            h3_video.AUDIO_VAE,
        ):
            p = h3_video.entry_path(entry)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"weights")

        self.assertTrue(h3_video.ready("hyperflow-fl2va-q4", "p-h3-int8", "sol-int8"))

        def fake_run(cmd, **kwargs):
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_bytes(b"fake-mp4")

        with patch.object(sdcpp, "run", side_effect=fake_run) as run:
            out_path, used_seed = h3v_engine.generate_video(
                prompt="A rally car drifting in snow",
                diffusion_key="hyperflow-fl2va-q4",
                encoder_key="p-h3-int8",
                vae_key="sol-int8",
                width=864,
                height=480,
                video_frames=56,
                steps=6,
                seed=777,
                attn_chunk_budget=0.5,
                temporal_tiling=True,
                hires_enable=True,
                hires_width=1216,
                hires_height=672,
                hires_steps=4,
            )
        self.assertTrue(out_path.is_file())
        self.assertEqual(used_seed, 777)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--mode") + 1], "vid_gen")
        self.assertEqual(cmd[cmd.index("--sigmas") + 1], "hyperflow-h3-flash-6")
        self.assertEqual(cmd[cmd.index("--video-frames") + 1], "56")
        self.assertEqual(cmd[cmd.index("--model-args") + 1], "minimax_h3_attn_chunk_budget=0.5")
        self.assertIn("--temporal-tiling", cmd)
        self.assertIn("--vae-tiling", cmd)
        self.assertIn("--diffusion-fa", cmd)
        self.assertIn("--hires", cmd)
        self.assertEqual(cmd[cmd.index("--hires-width") + 1], "1216")
        self.assertEqual(cmd[cmd.index("--hires-height") + 1], "672")
        self.assertTrue(out_path.with_suffix(".txt").is_file())


class SteppeWan21Tests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.enterContext(patch.object(settings, "ROOT", self.root))
        for name in ("OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR"):
            path = self.root / name
            path.mkdir(parents=True, exist_ok=True)
            self.enterContext(patch.object(settings, name, path))
        self.enterContext(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))

    def test_catalog_and_validation(self):
        self.assertIsNotNone(model_catalog.get("addon:steppe"))
        self.assertIn("steppe", addons.ADDON_NAMES)
        self.assertEqual(steppe_engine.target_size((512, 512), 2), (1024, 1024))
        self.assertEqual(steppe_engine.target_size((512, 512), 4), (2048, 2048))
        with self.assertRaises(ValueError):
            steppe_engine.target_size((4096, 4096), 4)
        with self.assertRaises(ValueError):
            steppe_engine.validate_controls("unknown", 2, 0.9, 1.0, 20, "auto", 42, 8)

    def test_ready_checks_pinned_commit_vae_and_checkpoint(self):
        root = addons.root("steppe")
        py = addons.python("steppe")
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("", encoding="utf-8")
        (root / "source" / "steppe").mkdir(parents=True, exist_ok=True)
        (root / "source" / "steppe" / "__init__.py").write_text("", encoding="utf-8")
        (root / "installed.json").write_text(
            json.dumps({"commit": steppe.SOURCE_COMMIT, "variant": "200m-1step"}),
            encoding="utf-8",
        )
        self.assertFalse(steppe.ready("200m-1step"))

        for rel in steppe.VAE_FILES:
            p = steppe.model_dir() / "wan21_vae" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"vae")
        ckpt = steppe.checkpoint_path("200m-1step")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"ckpt")
        self.assertTrue(steppe.ready("200m-1step"))
        self.assertTrue(model_catalog.ready("addon:steppe", "200m-1step"))
        self.assertFalse(steppe.ready("20m-1step"))

    def test_upscale_invokes_isolated_runner_for_image_and_verifies_dimensions(self):
        root = addons.root("steppe")
        py = addons.python("steppe")
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("", encoding="utf-8")
        (root / "source" / "steppe").mkdir(parents=True, exist_ok=True)
        (root / "source" / "steppe" / "__init__.py").write_text("", encoding="utf-8")
        (root / "installed.json").write_text(
            json.dumps({"commit": steppe.SOURCE_COMMIT, "variant": "200m-1step"}),
            encoding="utf-8",
        )
        for rel in steppe.VAE_FILES:
            p = steppe.model_dir() / "wan21_vae" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"vae")
        ckpt = steppe.checkpoint_path("200m-1step")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"ckpt")

        src = self.root / "in.png"
        Image.new("RGBA", (128, 96), (100, 150, 200, 255)).save(src)

        def fake_job_run(cmd, log, env=None):
            out = Path(cmd[cmd.index("--output") + 1])
            Image.new("RGBA", (256, 192), (100, 150, 200, 255)).save(out)

        with patch.object(steppe_engine.JOB, "run", side_effect=fake_job_run) as run:
            out_path = steppe_engine.upscale(src, variant="200m-1step", scale=2, hff=0.9, guidance=1.0)
        self.assertTrue(out_path.is_file())
        with Image.open(out_path) as im:
            self.assertEqual(im.size, (256, 192))
        cmd = [str(x) for x in run.call_args.args[0]]
        self.assertIn("--mode", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "image")
        self.assertEqual(cmd[cmd.index("--scale") + 1], "2")


if __name__ == "__main__":
    unittest.main()
