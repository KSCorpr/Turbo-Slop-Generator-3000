"""Media integration gates: validation, cancellation and output handling."""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from atelier import addons, settings
from atelier.engine import ltx25, seedvr2, splat
from atelier.engine.local_jobs import LocalJob


class CommandTests(unittest.TestCase):
    def test_seed_presets_bound_video_memory_and_disable_noise_injection(self):
        for key in addons.SEED_MODELS:
            cmd = seedvr2.build_command(Path("in file.mp4"), Path("out.mp4"), key,
                                        1080, is_video=True)
            self.assertEqual(cmd[cmd.index("--batch_size") + 1], "5")
            self.assertEqual(cmd[cmd.index("--chunk_size") + 1], "33")
            self.assertIn("--vae_decode_tiled", cmd)
            self.assertEqual(cmd[cmd.index("--input_noise_scale") + 1], "0")
            self.assertNotIn("--cuda_device", cmd)  # mapping comes from child env
        with self.assertRaises(ValueError):
            seedvr2.build_command(Path("in.png"), Path("out.png"), "../model", 1080, is_video=False)

    def test_ltx_rejects_wrong_vae_and_unpaired_last_frame(self):
        paths = [Path("model"), Path("encoder"), Path("ltx-2.5-video-vae-bf16.safetensors")]
        with self.assertRaisesRegex(ValueError, "convolutional"):
            ltx25.build_command(Path("sd"), paths, "a scene", Path("out"))
        with self.assertRaisesRegex(ValueError, "first frame"):
            ltx25.build_command(Path("sd"), paths, "a scene", Path("out"), last=Path("last.png"))

    def test_ltx_paths_and_memory_controls_are_real_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / n for n in ["model.gguf", "encoder.gguf",
                     "ltx-2.5-video-vae-conv-bf16.safetensors"]]
            for p in paths:
                p.write_bytes(b"test")
            opts = {"--video-frames", "--auto-fit", "--max-vram", "--llm",
                    "--vae-tiling", "--diffusion-fa"}
            with patch.object(ltx25.sdcpp, "supported_options", return_value=opts), \
                 patch.object(ltx25.sdcpp, "max_vram_arg", return_value="10GiB"):
                cmd = ltx25.build_command(Path("sd"), paths, "a scene", Path("out.avi"))
            self.assertIn("--auto-fit", cmd)
            self.assertIn("10GiB", cmd)
            self.assertNotIn("--audio-vae", cmd)
            self.assertEqual(cmd[cmd.index("--cfg-scale") + 1], "3.0")

    def test_brush_uses_release_cli_not_unreleased_renamed_flags(self):
        cmd = splat.brush_command(Path("brush"), Path("data"), Path("out"), "scene", "standard")
        self.assertIn("--total-steps", cmd)
        self.assertNotIn("--total-train-iters", cmd)
        self.assertEqual(cmd[cmd.index("--max-splats") + 1], "1000000")


class CaptureTests(unittest.TestCase):
    def test_duplicate_names_do_not_overwrite_and_originals_remain_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photos = []
            for i in range(8):
                folder = root / str(i)
                folder.mkdir()
                photo = folder / "same.png"
                Image.new("RGB", (1800, 900), (i, 0, 0)).save(photo)
                photos.append(str(photo))
            splat.prepare_photos(photos, root / "prepared")
            self.assertEqual(len(list((root / "prepared").glob("*.png"))), 8)
            with Image.open(root / "prepared" / "frame_00000.png") as image:
                self.assertEqual(image.size, (1600, 800))
            with Image.open(photos[0]) as image:
                self.assertEqual(image.size, (1800, 900))

    def test_capture_rejects_mixed_sources_and_too_few_photos(self):
        with self.assertRaisesRegex(ValueError, "either"):
            splat.reconstruct(["photo"], "video")
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            splat.prepare_photos(["photo"], Path(temp))

    def test_previous_capture_cannot_escape_outputs(self):
        with self.assertRaises(ValueError):
            splat.resolve_saved("../../secret.ply")


class JobTests(unittest.TestCase):
    def test_cancellation_stops_process_and_prevents_next_stage(self):
        job = LocalJob()
        ready = threading.Event()
        errors = []
        def worker():
            try:
                with job.session():
                    job.run([sys.executable, "-u", "-c",
                             "import time; print('ready'); time.sleep(60)"],
                            lambda line: ready.set() if line == "ready" else None)
                    errors.append("incorrectly reached second stage")
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(ready.wait(10))
        job.cancel()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("cancelled", errors[0])
        self.assertIsNone(job._proc)
        # A new job must not inherit the cancelled state.
        with job.session():
            job.run([sys.executable, "-c", "print('new job')"], lambda line: None)

    def test_logging_failure_reaps_worker(self):
        job = LocalJob()
        def log(line):
            if line == "ready":
                raise ValueError("consumer failed")
        with self.assertRaisesRegex(ValueError, "consumer failed"), job.session():
            job.run([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"], log)
        self.assertIsNone(job._proc)


if __name__ == "__main__":
    unittest.main()
