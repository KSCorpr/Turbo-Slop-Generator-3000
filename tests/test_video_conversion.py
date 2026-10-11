"""Playable video output and recovery after a converter failure."""
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import settings
from atelier.engine import sdcpp
from atelier.engine import video as engine


class VideoConversionTests(unittest.TestCase):
    def test_missing_converter_repairs_the_running_portable_python(self):
        calls = []

        def pip_install(command, **kwargs):
            calls.append(command)
            sys.modules["imageio_ffmpeg"] = SimpleNamespace(
                get_ffmpeg_exe=lambda: "ffmpeg-installed.exe")
            return SimpleNamespace(returncode=0, stdout="installed", stderr="")

        with patch.dict(sys.modules, {"imageio_ffmpeg": None}), \
             patch.object(engine.shutil, "which", return_value=None), \
             patch.object(engine.subprocess, "run", side_effect=pip_install):
            self.assertEqual(engine._ffmpeg_exe(), "ffmpeg-installed.exe")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("imageio-ffmpeg>=0.6,<1", calls[0])

    def test_can_recover_previous_avi_and_keeps_it_when_converter_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "OUTPUT_DIR", Path(tmp)):
            avi = Path(tmp) / "ltx-2.5-previous.avi"
            avi.write_bytes(b"saved video")
            self.assertIn(avi.name, engine.saved_avis())
            with patch.object(engine, "_ffmpeg_exe",
                              side_effect=sdcpp.EngineError("offline")):
                with self.assertRaisesRegex(sdcpp.EngineError, "offline"):
                    engine.convert_saved_avi(avi.name)
            self.assertTrue(avi.exists())
            with self.assertRaises(sdcpp.EngineError):
                engine.convert_saved_avi("../other.avi")

            def encoded(command, **kwargs):
                Path(command[-1]).write_bytes(b"playable mp4")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(engine, "_ffmpeg_exe", return_value="ffmpeg.exe"), \
                 patch.object(engine.subprocess, "run", side_effect=encoded):
                mp4 = engine.convert_saved_avi(avi.name)
            self.assertEqual(mp4.read_bytes(), b"playable mp4")
            self.assertFalse(avi.exists())

    def test_recovery_rejects_path_traversal_after_valid_prefix(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "OUTPUT_DIR", Path(tmp)):
            avi = Path(tmp) / "ltx-2.5-previous.avi"
            avi.write_bytes(b"saved video")
            self.assertIn(avi.name, engine.saved_avis())
            with patch.object(engine, "encode_mp4", return_value=avi.with_suffix(".mp4")):
                self.assertEqual(engine.convert_saved_avi(avi.name).suffix, ".mp4")
            with self.assertRaises(sdcpp.EngineError):
                engine.convert_saved_avi("ltx-2.5-../other.avi")

