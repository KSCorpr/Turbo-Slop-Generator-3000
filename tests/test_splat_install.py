"""Exercise release extraction and executable discovery without GPU or downloads."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import addons
from scripts import setup_media


class BrushReleaseTests(unittest.TestCase):
    def test_release_and_legacy_names_are_found_in_nested_directories(self):
        for system, names in (("nt", ("brush_app.exe", "brush-app.exe", "brush.exe")),
                              ("posix", ("brush_app", "brush-app", "brush"))):
            for name in names:
                with self.subTest(system=system, name=name), tempfile.TemporaryDirectory() as temp:
                    target = Path(temp)
                    executable = target / "brush" / "release" / name
                    executable.parent.mkdir(parents=True)
                    executable.write_bytes(b"binary")
                    with patch.object(addons, "root", return_value=target), \
                         patch.object(addons, "os", SimpleNamespace(name=system)):
                        self.assertEqual(addons.brush(), executable)

    def test_directory_named_like_executable_is_not_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            (target / "brush" / "brush_app.exe").mkdir(parents=True)
            executable = target / "brush" / "release" / "brush_app.exe"
            executable.parent.mkdir()
            executable.write_bytes(b"binary")
            with patch.object(addons, "root", return_value=target), \
                 patch.object(addons, "os", SimpleNamespace(name="nt")):
                self.assertEqual(addons.brush(), executable)
                executable.unlink()
                self.assertIsNone(addons.brush())

    def exercise_install(self, *, invalid_checksum=False, missing_binary=False):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            py = target / "python.exe"
            py.write_bytes(b"python")
            urls = []

            def download(url, destination):
                urls.append(url)
                if url.endswith(".sha256"):
                    archive = destination.parent / "brush-app-x86_64-pc-windows-msvc.zip"
                    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
                    destination.write_text(("0" * 64 if invalid_checksum else checksum)
                                           + "  " + archive.name + "\n")
                else:
                    with zipfile.ZipFile(destination, "w") as bundle:
                        bundle.writestr("README.md", "fixture")
                        bundle.writestr("support.dll", b"companion")
                        if not missing_binary:
                            bundle.writestr("brush_app.exe", b"executable")

            with patch.object(addons, "root", return_value=target), \
                 patch.object(addons, "os", SimpleNamespace(name="nt")), \
                 patch.object(setup_media.platform, "system", return_value="Windows"), \
                 patch.object(setup_media.platform, "machine", return_value="AMD64"), \
                 patch.object(setup_media, "environment", return_value=py), \
                 patch.object(setup_media, "pip"), \
                 patch.object(setup_media, "download", side_effect=download), \
                 patch.object(setup_media, "run") as run:
                if invalid_checksum or missing_binary:
                    error = "checksum mismatch" if invalid_checksum else "Expected brush_app.exe"
                    with self.assertRaisesRegex(RuntimeError, error):
                        setup_media.install_splat()
                    self.assertFalse((target / "installed.json").exists())
                    run.assert_not_called()
                else:
                    setup_media.install_splat()
                    executable = target / "brush" / "brush_app.exe"
                    self.assertEqual(addons.brush(), executable)
                    run.assert_any_call(executable, "--help")
                    run.assert_any_call(py, "-c", "import pycolmap; print('COLMAP', pycolmap.__version__)")
                    self.assertEqual((target / "brush" / "support.dll").read_bytes(), b"companion")
                    self.assertEqual(json.loads((target / "installed.json").read_text())["brush"],
                                     addons.BRUSH_VERSION)
                self.assertTrue(all(f"/{addons.BRUSH_VERSION}/" in url for url in urls))

    def test_windows_archive_finishes_install_and_runs_discovered_executable(self):
        self.exercise_install()

    def test_bad_checksum_never_runs_executable_or_marks_install_ready(self):
        self.exercise_install(invalid_checksum=True)

    def test_missing_binary_never_marks_install_ready(self):
        self.exercise_install(missing_binary=True)


if __name__ == "__main__":
    unittest.main()
