"""Check spherical geometry, all catalogue conditioning paths and cancellation."""
import json
import math
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from atelier import registry, settings
from atelier.engine import multiview as mv, panorama as pano


def direction_image(view, size=128):
    """Known analytic environment: RGB encodes its world-space ray direction."""
    axis = ((np.arange(size) + .5) / size * 2 - 1) * math.tan(math.radians(pano.FOV / 2))
    u, v = np.meshgrid(axis, -axis)
    forward, right, up = pano.basis(view)
    rays = forward + u[..., None] * right + v[..., None] * up
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return Image.fromarray(np.round((rays + 1) * 127.5).astype(np.uint8))


class PanoramaTests(unittest.TestCase):
    def test_projection_matches_known_sphere_including_poles_and_wrap(self):
        width = 512
        images = [direction_image(view) for view in pano.SCENE_VIEWS]
        actual = np.asarray(pano.equirectangular(images, width=width)).astype(float)
        lon = ((np.arange(width) + .5) / width * 2 - 1) * np.pi
        lat = (.5 - (np.arange(width // 2) + .5) / (width // 2)) * np.pi
        expected = np.stack((np.cos(lat)[:, None] * np.sin(lon)[None, :],
                             np.broadcast_to(np.sin(lat)[:, None], actual.shape[:2]),
                             np.cos(lat)[:, None] * np.cos(lon)[None, :]), axis=-1)
        self.assertLess(np.abs(actual - (expected + 1) * 127.5).max(), 2)
        # Catch inverted north/south and a discontinuity at the 180-degree wrap.
        self.assertGreater(actual[0, width // 2, 1], 250)
        self.assertLess(actual[-1, width // 2, 1], 5)
        self.assertLess(np.abs(actual[:, 0] - actual[:, -1]).max(), 5)

    def test_constant_scene_has_no_uncovered_or_dark_regions(self):
        images = [Image.new("RGB", (16, 16), (35, 67, 201)) for _ in pano.SCENE_VIEWS]
        result = np.asarray(pano.equirectangular(images, width=256))
        self.assertTrue(np.all(result == (35, 67, 201)))

    def test_rejects_missing_poles_nonsquare_and_bad_geometry(self):
        square = Image.new("RGB", (16, 16))
        for kwargs in ({"images": [square] * 8},
                       {"images": [Image.new("RGB", (32, 16))] * 10},
                       {"images": [square] * 10, "fov": 70},
                       {"images": [square] * 10, "width": 8194}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pano.equirectangular(**kwargs)


class MultiviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(settings, "OUTPUT_DIR", self.root / "outputs"))
        self.stack.enter_context(patch.object(settings, "ensure_dirs", lambda: settings.OUTPUT_DIR.mkdir(exist_ok=True)))
        self.job = mv.ViewJob()
        self.source = Image.new("RGB", (100, 70), "red")
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        path = self.root / f"render-{len(self.calls)}.png"
        Image.new("RGB", (kwargs["width"], kwargs["height"]),
                  (len(self.calls), 50, 100)).save(path)
        return [path]

    def test_all_five_engines_use_their_supported_conditioning_and_defaults(self):
        models = registry.load_base_models(settings.load_prefs())
        self.assertEqual(len(models), 5)
        with patch.object(mv.gen_engine, "generate", side_effect=self.generate), \
             patch.object(mv.tools, "describe_is_installed", return_value=True), \
             patch.object(mv.tools, "image_to_prompt", return_value=["A red sculptural object."]) as read:
            for model in models:
                self.calls.clear()
                with self.subTest(model=model.id):
                    result = mv.generate_views(self.source, model.id, size=512, seed=123, job=self.job)
                    self.assertEqual(len(self.calls), 10)
                    self.assertEqual(len(result.views), 10)
                    self.assertEqual([c["seed"] for c in self.calls], list(range(123, 133)))
                    self.assertEqual({c["steps"] for c in self.calls}, {model.defaults["steps"]})
                    method = mv.conditioning(model)
                    for call in self.calls:
                        self.assertEqual("ref_image" in call, method == "reference")
                        self.assertEqual("init_image" in call, method == "img2img")
                        if method == "description":
                            self.assertIn("A red sculptural object.", call["prompt"])
                    manifest = json.loads((result.directory / "manifest.json").read_text())
                    self.assertEqual(manifest["conditioning"], method)
                    self.assertEqual(manifest["sampling"]["steps"], model.defaults["steps"])
                    self.assertEqual(manifest["status"], "completed")
                    with zipfile.ZipFile(result.archive) as zf:
                        self.assertIn("manifest.json", zf.namelist())
                        self.assertIn("contact-sheet.jpg", zf.namelist())
                        self.assertEqual(len([n for n in zf.namelist() if n[:2].isdigit()]), 10)
            self.assertEqual(read.call_count, 1)  # Ming reads once, not ten times

    def test_cancel_between_views_keeps_partial_export_and_never_starts_third(self):
        updates = []

        def stop(result):
            updates.append(result)
            if len(result.views) == 2:
                self.job.cancelled.set()

        with patch.object(mv.gen_engine, "generate", side_effect=self.generate):
            with self.assertRaisesRegex(RuntimeError, "Cancelled"):
                mv.generate_views(self.source, "flux2-klein-9b", size=512, seed=42,
                                  job=self.job, update=stop)
        self.assertEqual(len(self.calls), 2)
        result = updates[-1]
        self.assertTrue(result.archive.is_file())
        self.assertFalse(self.job.busy)
        manifest = json.loads((result.directory / "manifest.json").read_text())
        self.assertEqual((manifest["status"], manifest["completed"]), ("cancelled", 2))

    def test_cancel_kills_active_engine_and_caption_but_idle_stop_does_nothing(self):
        with patch.object(mv.gen_engine, "cancel") as engine, patch.object(mv.tools, "cancel") as caption:
            self.job.cancel()
            engine.assert_not_called()
            caption.assert_not_called()
            with self.job.session():
                self.job.cancel()
                with self.assertRaises(RuntimeError):
                    self.job.check()
            engine.assert_called_once()
            caption.assert_called_once()
            with self.job.session():
                self.job.check()  # next task does not inherit cancellation

    def test_invalid_requests_do_not_start_generation(self):
        with patch.object(mv.gen_engine, "generate") as engine:
            for kwargs in ({"make_panorama": True}, {"size": 1000},
                           {"mode": "anything"}, {"strength": 0},
                           {"seed": 2**31}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    mv.generate_views(self.source, "flux2-klein-9b", job=self.job, **kwargs)
            engine.assert_not_called()

    def test_ming_requires_vision_module_and_never_silently_ignores_the_image(self):
        with patch.object(mv.tools, "describe_is_installed", return_value=False), \
             patch.object(mv.gen_engine, "generate") as engine:
            with self.assertRaisesRegex(ValueError, "Install Image"):
                mv.generate_views(self.source, "ming-image-design", job=self.job)
            engine.assert_not_called()

    def test_scene_generation_and_reconversion_preserve_order_and_projection(self):
        def generate_scene(**kw):
            index = len(self.calls)
            self.calls.append(kw)
            path = self.root / f"sphere-{index}.png"
            direction_image(pano.SCENE_VIEWS[index], size=512).save(path)
            return [path]

        with patch.object(mv.gen_engine, "generate", side_effect=generate_scene):
            result = mv.generate_views(self.source, "flux2-klein-9b", mode="scene",
                                       size=512, make_panorama=True, panorama_width=2048, job=self.job)
        self.assertEqual(Image.open(result.panorama).size, (2048, 1024))
        self.assertIn("position must stay fixed", self.calls[0]["prompt"])
        self.assertIn("straight UP", self.calls[8]["prompt"])
        self.assertIn("straight DOWN", self.calls[9]["prompt"])
        converted = mv.convert_views([p for p, _ in reversed(result.views)], width=2048, job=self.job)
        self.assertEqual(result.panorama.read_bytes(), converted.panorama.read_bytes())
        with self.assertRaisesRegex(ValueError, "Object views"):
            mv.convert_views([self.root / f"{i:02d}-object.png" for i in range(10)], job=self.job)


if __name__ == "__main__":
    unittest.main()
