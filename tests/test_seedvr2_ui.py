"""Source previews must load Unicode uploads without modifying engine inputs."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from atelier.ui.media_tabs import seed_source_preview


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


if __name__ == "__main__":
    unittest.main()
