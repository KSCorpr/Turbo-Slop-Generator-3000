"""Pixal3D uses TRELLIS's existing decoders and chooses a family per request."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import get_trellis  # noqa: E402
from atelier.engine import trellis  # noqa: E402
from atelier.trellis_models import (PIXAL_1024, PIXAL_512, TRELLIS_FILES,
                                   all_present)  # noqa: E402


class PixalInstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "trellis"
        self.q4 = self.root / "q4"
        self.q4.mkdir(parents=True)
        for name in TRELLIS_FILES:
            (self.q4 / name).write_bytes(b"base decoder")
        patcher = mock.patch.object(get_trellis, "MODELS_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_512_downloads_only_three_weights_and_links_into_q4(self):
        def download(**kw):
            self.assertEqual(kw["repo_id"], "vegax87/Pixal3D")
            self.assertEqual(set(kw["allow_patterns"]), set(PIXAL_512))
            for name in kw["allow_patterns"]:
                (self.root / name).write_bytes(name.encode())

        hub = SimpleNamespace(snapshot_download=mock.Mock(side_effect=download))
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertTrue(get_trellis.install_pixal_models(
                variant="q4", resolution=512, log=lambda *_: None))
            self.assertTrue(get_trellis.install_pixal_models(
                variant="q4", resolution=512, log=lambda *_: None))
        hub.snapshot_download.assert_called_once()
        self.assertFalse((self.q4 / "pixal3d_tex_flow_1024.gguf").exists())
        for name in PIXAL_512:
            self.assertTrue(os.path.samefile(self.root / name, self.q4 / name))

    def test_upgrading_to_1024_only_downloads_the_other_two(self):
        for name in PIXAL_512:
            (self.root / name).write_bytes(name.encode())

        def download(**kw):
            self.assertEqual(set(kw["allow_patterns"]),
                             set(PIXAL_1024) - set(PIXAL_512))
            for name in kw["allow_patterns"]:
                (self.root / name).write_bytes(name.encode())

        hub = SimpleNamespace(snapshot_download=mock.Mock(side_effect=download))
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertTrue(get_trellis.install_pixal_models(
                variant="q4", resolution=1024, log=lambda *_: None))
        hub.snapshot_download.assert_called_once()
        for name in PIXAL_1024:
            self.assertTrue(os.path.samefile(self.root / name, self.q4 / name))

    def test_missing_shared_decoder_cannot_report_pixal_as_ready(self):
        for name in PIXAL_512:
            (self.q4 / name).write_bytes(b"pixal")
        with mock.patch.object(trellis, "MODELS_DIR", self.root):
            self.assertTrue(trellis.pixal_ready("q4", 512))
            self.assertFalse(trellis.pixal_ready("q4", 1024))
            (self.q4 / "shape_dec.gguf").unlink()
            self.assertFalse(trellis.pixal_ready("q4", 512))

    def test_incomplete_base_install_cannot_be_skipped(self):
        (self.root / PIXAL_512[0]).write_bytes(b"incomplete")
        self.assertFalse(get_trellis.has_models("f16"))
        self.assertTrue(get_trellis.has_models("q4"))

    def test_maintenance_keeps_the_shared_3d_model_directory(self):
        from scripts import maintenance
        self.assertIn("trellis", maintenance._expected_model_dirs())


class PixalRequestTests(unittest.TestCase):
    def test_pixal_camera_is_sent_per_request_in_multipart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "in.png"
            path.write_bytes(b"image")
            out = Path(temp) / "out.glb"
            result = SimpleNamespace(ok=True, content=b"GLB", status_code=200)
            with mock.patch.object(trellis.requests, "post",
                                   return_value=result) as post:
                trellis._post_generate(path, out, 1024, 17,
                                       trellis.BG_AUTO, 8000, lambda *_: None,
                                       family="pixal3d", fov=38,
                                       mesh_scale=1.2, extend_pixel=0)
            self.assertEqual(out.read_bytes(), b"GLB")
            data = post.call_args.kwargs["data"]
            self.assertEqual(data, {"resolution": "1024", "model": "pixal3d",
                                    "fov": "38.0", "mesh_scale": "1.2",
                                    "seed": "17"})

    def test_default_camera_and_trellis_family_do_not_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "in.png"
            path.write_bytes(b"image")
            result = SimpleNamespace(ok=True, content=b"GLB", status_code=200)
            with mock.patch.object(trellis.requests, "post",
                                   return_value=result) as post:
                trellis._post_generate(path, Path(temp) / "out.glb", 512,
                                       None, trellis.BG_AUTO, 8000,
                                       lambda *_: None)
            self.assertEqual(post.call_args.kwargs["data"],
                             {"resolution": "512", "model": "trellis"})

    def test_binary_support_is_checked_from_help(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "trellis-server.exe"
            binary.write_bytes(b"binary")
            with mock.patch.object(trellis, "find_server", return_value=binary), \
                    mock.patch.object(trellis.subprocess, "run",
                                      return_value=SimpleNamespace(
                                          stdout="--model trellis|pixal3d",
                                          stderr="")) as run:
                trellis._pixal_option.cache_clear()
                self.assertTrue(trellis.supports_pixal3d())
            run.assert_called_once()

    def test_size_menu_uses_actual_remote_file_sizes(self):
        from atelier.ui import threed_tab
        sizes = {name: (i + 1) * 1_000_000_000
                 for i, name in enumerate(PIXAL_1024)}
        siblings = [SimpleNamespace(rfilename=name, size=size)
                    for name, size in sizes.items()]
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(
            model_info=lambda **kw: SimpleNamespace(siblings=siblings)))
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(trellis, "MODELS_DIR", Path(temp)), \
                mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            (Path(temp) / PIXAL_512[0]).write_bytes(b"installed")
            msg = threed_tab._pixal_sizes(512)
        self.assertIn("6.00 GB", msg)
        self.assertIn("Still to download: 5.00 GB", msg)
        self.assertIn("installed", msg)
        self.assertNotIn("pixal3d_tex_flow_1024.gguf", msg)

    def test_gradio_exposes_the_family_and_camera(self):
        import app
        demo = app.build_app()
        handler = next(f.fn for f in demo.fns.values()
                       if getattr(f.fn, "__name__", "") == "do_generate3d")
        event = next(f for f in demo.fns.values() if f.fn is handler)
        inputs = {component.label: component for component in event.inputs
                  if hasattr(component, "label")}
        self.assertEqual(inputs["3D model family"].value, "trellis")
        self.assertIn("Pixal3D camera FOV (degrees)", inputs)
        self.assertIn("Pixal3D mesh scale", inputs)
        self.assertIn("Pixal3D extend pixel", inputs)
        self.assertEqual(len(event.outputs), 7)

    def test_ui_passes_selected_family_and_camera_to_engine(self):
        import app
        from atelier import settings

        demo = app.build_app()
        handler = next(f.fn for f in demo.fns.values()
                       if getattr(f.fn, "__name__", "") == "do_generate3d")
        event = next(f for f in demo.fns.values() if f.fn is handler)
        inputs = {component.label: i for i, component in
                  enumerate(event.inputs) if hasattr(component, "label")}
        values = [getattr(component, "value", None)
                  for component in event.inputs]
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            image = temp_path / "source.png"
            Image.new("RGB", (8, 8), "white").save(image)
            values[inputs["Input image (a single object)"]] = str(image)
            values[inputs["3D model family"]] = "pixal3d"
            values[inputs["Pixal3D camera FOV (degrees)"]] = 38
            values[inputs["Pixal3D mesh scale"]] = 1.2
            values[inputs["Pixal3D extend pixel"]] = 12
            values[inputs["Weights used"]] = "q4"

            def fake_generate(source, target, **kwargs):
                target.write_bytes(b"GLB")
                kwargs["meta"]["seed"] = 41

            with mock.patch.object(settings, "TMP_DIR", temp_path), \
                    mock.patch.object(settings, "OUTPUT_DIR", temp_path), \
                    mock.patch.object(settings, "ensure_dirs"), \
                    mock.patch.object(trellis, "is_ready", return_value=True), \
                    mock.patch.object(trellis, "pixal_ready",
                                      return_value=True), \
                    mock.patch.object(trellis, "generate",
                                      side_effect=fake_generate) as generate:
                outputs = list(handler(*values))
            opts = generate.call_args.kwargs
            self.assertEqual((opts["family"], opts["variant"], opts["fov"],
                              opts["mesh_scale"], opts["extend_pixel"]),
                             ("pixal3d", "q4", 38.0, 1.2, 12))
            self.assertTrue(outputs[-1][0].startswith("✅ 3D generated"))
            self.assertTrue(generate.call_args.args[1].name.startswith("pixal3d-"))


if __name__ == "__main__":
    unittest.main()
