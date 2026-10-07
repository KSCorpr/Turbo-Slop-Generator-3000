"""Source previews must load Unicode uploads without modifying engine inputs."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from atelier.ui.media_tabs import seed_source_preview, seed_size_controls


class SourcePreviewTests(unittest.TestCase):
    def test_unicode_jpeg_respects_exif_and_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brutalisme…é_日本.jpg"
            exif = Image.Exif()
            exif[274] = 6  # portrait orientation stored as landscape pixels
            Image.new("RGB", (1600, 800), "red").save(path, exif=exif)
            original = path.read_bytes()
            update, info = seed_source_preview(str(path))
            self.assertTrue(update["visible"])
            self.assertEqual(update["value"].size, (512, 1024))
            self.assertIn("800 × 1600", info)
            self.assertEqual(path.read_bytes(), original)

    def test_png_preview_preserves_transparency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image…png.png"
            Image.new("RGBA", (30, 20), (10, 40, 80, 70)).save(path)
            update, _ = seed_source_preview(path)
            self.assertEqual(update["value"].mode, "RGBA")
            self.assertEqual(update["value"].getpixel((0, 0))[3], 70)

    def test_clear_missing_file_and_video_do_not_leave_a_stale_image(self):
        for path in (None, "missing.jpg", "clip.mp4"):
            with self.subTest(path=path):
                update, _ = seed_source_preview(path)
                self.assertFalse(update["visible"])
                self.assertIsNone(update["value"])

    def test_unreadable_upload_returns_actionable_preview_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"not an image")
            update, info = seed_source_preview(path)
            self.assertFalse(update["visible"])
            self.assertIn("preview unavailable", info)


class ScalePreviewTests(unittest.TestCase):
    def test_x2_x4_and_custom_show_sizes_and_only_custom_shows_the_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source…é_日本.jpg"
            Image.new("RGB", (1024, 768)).save(path)
            for scale, size, visible in (("2", "2048 × 1536", False),
                                         ("4", "4096 × 3072", False),
                                         ("custom", "1440 × 1080", True)):
                with self.subTest(scale=scale):
                    field, info = seed_size_controls(path, scale, 1080)
                    self.assertEqual(field["visible"], visible)
                    self.assertIn(size, info)

    def test_preview_does_not_transcode_videos_and_reports_excessive_factors(self):
        field, info = seed_size_controls("missing-video.mp4", "4", 1080)
        self.assertFalse(field["visible"])
        self.assertIn("when restoration starts", info)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            Image.new("RGB", (2400, 2100)).save(path)
            _, info = seed_size_controls(path, "4", 1080)
            self.assertIn("8400", info)


if __name__ == "__main__":
    unittest.main()
