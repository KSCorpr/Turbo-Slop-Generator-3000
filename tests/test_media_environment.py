"""Recovery for isolated media tool environments."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import setup_media


class MediaEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name) / "Turbo Slop"
        self.enterContext(patch.object(setup_media.settings, "ROOT", project))
        self.enterContext(patch.dict(sys.modules, {"uv": object()}))
        self.py = setup_media.addons.python("vosr")
        self.venv = self.py.parents[1]
        self.config = self.venv / "pyvenv.cfg"
        self.py.parent.mkdir(parents=True)
        self.py.touch()
        self.config.touch()
        self.neighbor = self.venv.parent / "constraints.txt"
        self.neighbor.write_text("keep me", encoding="utf-8")

    def create_venv(self, *args):
        self.assertEqual(args, (sys.executable, "-m", "uv", "venv", "--python",
                                "3.12", "--managed-python", self.venv))
        self.assertFalse(self.venv.exists())
        self.py.parent.mkdir(parents=True)
        self.py.touch()
        self.config.touch()

    def test_working_environment_is_reused(self):
        with patch.object(setup_media, "_probe_environment",
                          return_value=subprocess.CompletedProcess([], 0)) as probe, \
             patch.object(setup_media, "run") as run:
            self.assertEqual(setup_media.environment("vosr"), self.py)
        probe.assert_called_once_with(self.py)
        run.assert_not_called()

    def test_stale_windows_launcher_is_replaced_before_install(self):
        with patch.object(setup_media.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 2, stderr="No virtual environment found"),
            subprocess.CompletedProcess([], 0),
        ]) as probe, patch.object(setup_media, "run", side_effect=self.create_venv) as run:
            self.assertEqual(setup_media.environment("vosr"), self.py)
        run.assert_called_once()
        self.assertEqual(probe.call_count, 2)
        for call in probe.call_args_list:
            self.assertEqual(call.args[0][-4:], ["pip", "list", "--python", str(self.py)])
        self.assertEqual(self.neighbor.read_text(encoding="utf-8"), "keep me")

    def test_partial_environment_is_replaced_and_new_failure_is_reported(self):
        self.config.unlink()
        with patch.object(setup_media, "_probe_environment",
                          return_value=subprocess.CompletedProcess([], 2, stderr="Missing base Python")) as probe, \
             patch.object(setup_media, "run", side_effect=self.create_venv):
            with self.assertRaisesRegex(RuntimeError, "Missing base Python"):
                setup_media.environment("vosr")
        probe.assert_called_once_with(self.py)


if __name__ == "__main__":
    unittest.main()
