"""Scale and detail controls must change real output, not only UI labels."""
import io
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from atelier import addons, settings
from atelier.engine import seedvr2
from atelier.engine.local_jobs import LocalJob


class SeedControlsFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source…é_日本.png"
        self.output = self.root / "result.png"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.job = LocalJob()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(seedvr2, "JOB", self.job))
        self.stack.enter_context(patch.object(settings, "TMP_DIR", self.scratch))
        self.stack.enter_context(patch.object(addons, "ready", return_value=True))
        self.stack.enter_context(patch.object(addons, "root", return_value=self.root))
        self.stack.enter_context(patch.object(seedvr2.sdcpp, "unique_output",
                                             side_effect=lambda *args: self.output))


class SeedControlsTests(SeedControlsFixture):
    def test_x2_and_x4_reach_the_engine_with_source_based_resolutions(self):
        Image.new("RGB", (320, 240)).save(self.source)
        original = self.source.read_bytes()
        for factor, short, size in ((2, 480, (640, 480)), (4, 960, (1280, 960))):
            with self.subTest(factor=factor):
                emitted = []

                def run(command, log, **kwargs):
                    self.assertEqual(command[command.index("--resolution") + 1], str(short))
                    self.assertEqual(Path(command[3]).read_bytes(), original)
                    target = Path(command[command.index("--output") + 1])
                    Image.new("RGB", size, "red").save(target)
                    emitted.append(target.read_bytes())

                with patch.object(self.job, "run", side_effect=run):
                    result = seedvr2.restore(self.source, scale=factor, log=lambda line: None)
                self.assertEqual(result.read_bytes(), emitted[0])  # 100% never re-encodes the image.
                with Image.open(result) as image:
                    self.assertEqual(image.size, size)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual(list(self.scratch.iterdir()), [])

    def test_detail_strength_blends_pixels_and_preserves_transparency(self):
        for mode in ("RGB", "RGBA"):
            with self.subTest(mode=mode):
                source_color = (20, 40, 60, 80) if mode == "RGBA" else (20, 40, 60)
                ai_color = (220, 160, 100, 240) if mode == "RGBA" else (220, 160, 100)
                expected = (70, 70, 70, 120) if mode == "RGBA" else (70, 70, 70)
                Image.new(mode, (320, 240), source_color).save(self.source)
                original = self.source.read_bytes()

                def run(command, log, **kwargs):
                    Image.new(mode, (640, 480), ai_color).save(
                        command[command.index("--output") + 1])

                with patch.object(self.job, "run", side_effect=run):
                    seedvr2.restore(self.source, scale=2, detail_strength=25, log=lambda line: None)
                with Image.open(self.output) as image:
                    self.assertEqual(image.size, (640, 480))
                    self.assertEqual(image.mode, mode)
                    pixel = image.getpixel((100, 100))
                    for actual, wanted in zip(pixel, expected):
                        self.assertLessEqual(abs(actual - wanted), 2)
                self.assertEqual(self.source.read_bytes(), original)

    def test_zero_details_is_a_simple_upscale_without_an_installed_model(self):
        Image.new("RGBA", (320, 240), (30, 60, 90, 120)).save(self.source)
        with patch.object(addons, "ready", return_value=False), \
             patch.object(self.job, "run") as engine:
            seedvr2.restore(self.source, scale=4, detail_strength=0, log=lambda line: None)
        engine.assert_not_called()
        with Image.open(self.output) as image:
            self.assertEqual(image.size, (1280, 960))
            pixel = image.getpixel((100, 100))
            self.assertEqual(pixel[3], 120)
            for actual, wanted in zip(pixel[:3], (30, 60, 90)):
                self.assertLessEqual(abs(actual - wanted), 2)

    def test_custom_short_edge_still_works(self):
        Image.new("RGB", (320, 240)).save(self.source)
        seedvr2.restore(self.source, resolution=300, detail_strength=0, log=lambda line: None)
        with Image.open(self.output) as image:
            self.assertEqual(image.size, (400, 300))

    def test_invalid_details_and_scale_do_not_start_a_job_or_publish_a_file(self):
        Image.new("RGB", (320, 240)).save(self.source)
        for controls in ({"detail_strength": -1}, {"detail_strength": 101},
                         {"detail_strength": float("nan")}, {"detail_strength": float("inf")},
                         {"scale": 3}, {"scale": "4"}, {"scale": 2.0},
                         {"resolution": None}, {"resolution": float("inf")}):
            with self.subTest(controls=controls), patch.object(self.job, "run") as engine:
                with self.assertRaises(ValueError):
                    seedvr2.restore(self.source, log=lambda line: None, **controls)
                engine.assert_not_called()
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.scratch.iterdir()), [])

    def test_out_of_range_x4_is_rejected_without_silently_reducing_the_factor(self):
        Image.new("RGB", (320, 240)).save(self.source)
        with patch.object(seedvr2, "source_size", return_value=(3000, 2300)), \
             patch.object(self.job, "run") as engine:
            with self.assertRaisesRegex(ValueError, "9200 px"):
                seedvr2.restore(self.source, scale=4, log=lambda line: None)
        engine.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])


class VideoControlsTests(SeedControlsFixture):
    def setUp(self):
        super().setUp()
        self.source = self.root / "video…é_日本.mp4"
        self.output = self.root / "result.mp4"
        self.ffmpeg = seedvr2.video._ffmpeg_exe(log=lambda line: None)
        self.make_clip(self.source, "black", "160x128", audio=True)

    def make_clip(self, path, color, size, *, audio=False):
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "lavfi", "-i", f"color={color}:s={size}:r=8:d=0.5"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.5",
                    "-c:a", "aac", "-shortest"]
        subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                       capture_output=True, check=True, timeout=30)

    def assert_clip(self, red):
        self.assertEqual(seedvr2.source_size(self.output), (320, 256))
        frame = subprocess.run(
            [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(self.output),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, check=True, timeout=30)
        with Image.open(io.BytesIO(frame.stdout)) as image:
            pixel = image.convert("RGB").getpixel((100, 100))
            self.assertLessEqual(abs(pixel[0] - red), 8, pixel)
            self.assertLessEqual(max(pixel[1:]), 8, pixel)
        probe = subprocess.run([self.ffmpeg, "-hide_banner", "-nostdin", "-i", str(self.output),
                                "-map", "0:v:0", "-frames:v", "0", "-an", "-f", "null", "-"],
                               capture_output=True, text=True, check=True, timeout=30)
        self.assertIn("Audio:", probe.stderr)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_real_ffmpeg_x2_and_zero_details_keep_video_and_audio(self):
        original = self.source.read_bytes()
        with patch.object(addons, "ready", return_value=False):
            seedvr2.restore(self.source, scale=2, detail_strength=0, log=lambda line: None)
        self.assertEqual(self.source.read_bytes(), original)
        self.assert_clip(0)

    def test_real_ffmpeg_blends_details_and_uses_snapshot_after_upload_eviction(self):
        real_run = self.job.run
        calls = []

        def run(command, log, **kwargs):
            calls.append(command)
            if "--dit_model" in command:
                self.assertEqual(command[command.index("--resolution") + 1], "256")
                target = Path(command[command.index("--output") + 1])
                self.make_clip(target, "red", "320x256")
                self.source.unlink()
            else:
                real_run(command, log, **kwargs)

        with patch.object(self.job, "run", side_effect=run):
            seedvr2.restore(self.source, scale=2, detail_strength=50, log=lambda line: None)
        self.assertEqual(len(calls), 3)  # SeedVR2, detail mix, audio attachment.
        self.assert_clip(127)

    def test_failed_detail_mix_does_not_publish_a_partial_video(self):
        def run(command, log, **kwargs):
            if "--dit_model" in command:
                self.make_clip(Path(command[command.index("--output") + 1]), "red", "320x256")
            else:
                raise RuntimeError("detail mix failed")

        with patch.object(self.job, "run", side_effect=run), \
             self.assertRaisesRegex(RuntimeError, "detail mix failed"):
            seedvr2.restore(self.source, scale=2, detail_strength=50, log=lambda line: None)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.source.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
