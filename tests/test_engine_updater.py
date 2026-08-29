import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


_PATH = Path(__file__).resolve().parent.parent / "scripts" / "get_sdcpp.py"
_SPEC = importlib.util.spec_from_file_location("get_sdcpp_test", _PATH)
U = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(U)


def archive(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TransactionalUpdateTests(unittest.TestCase):
    def _globals(self, root):
        return patch.multiple(
            U, ROOT=root, BIN_DIR=root / "bin",
            PREVIOUS_DIR=root / ".engine-previous")

    def test_new_engine_is_validated_before_old_one_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "bin"
            old.mkdir()
            (old / "old.txt").write_text("working", encoding="utf-8")
            blob = archive({"sd-cli": "new"})
            with self._globals(root), \
                 patch.object(U, "_validate_staged",
                              side_effect=RuntimeError("broken")):
                with self.assertRaisesRegex(RuntimeError, "broken"):
                    U._transactional_install(blob, "engine.zip", {"tag": "x"})
            self.assertEqual((old / "old.txt").read_text(), "working")
            self.assertFalse((root / ".engine-previous").exists())

    def test_success_keeps_previous_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "bin"
            old.mkdir()
            (old / "old.txt").write_text("working", encoding="utf-8")
            blob = archive({"sd-cli": "new"})

            def valid(folder):
                return folder / "sd-cli", ["--mode", "--params-backend"]

            with self._globals(root), patch.object(U, "_validate_staged", valid):
                U._transactional_install(
                    blob, "engine.zip", {"source": "official", "tag": "master-830"})
            self.assertTrue((root / "bin" / "sd-cli").exists())
            self.assertTrue((root / ".engine-previous" / "old.txt").exists())
            manifest = json.loads(
                (root / "bin" / U.ENGINE_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["tag"], "master-830")
            self.assertEqual(manifest["source"], "official")
            self.assertIn("archive_sha256", manifest)

    def test_zip_slip_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "dangereuse"):
                U._extract_to(archive({"../escape": "bad"}), "bad.zip", root / "out")
            self.assertFalse((root / "escape").exists())

    def test_final_smoke_failure_restores_the_old_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "bin"
            old.mkdir()
            (old / "old.txt").write_text("working", encoding="utf-8")
            blob = archive({"sd-cli": "new"})

            def validate(folder):
                if folder == root / "bin":
                    raise RuntimeError("fails after swap")
                return folder / "sd-cli", ["--mode"]

            with self._globals(root), patch.object(U, "_validate_staged", validate):
                with self.assertRaisesRegex(RuntimeError, "after swap"):
                    U._transactional_install(blob, "engine.zip", {"tag": "x"})
            self.assertEqual((root / "bin" / "old.txt").read_text(), "working")
            self.assertFalse((root / ".engine-broken").exists())

    def test_rollback_swaps_current_and_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "bin"
            previous = root / ".engine-previous"
            current.mkdir()
            previous.mkdir()
            (current / "version.txt").write_text("new", encoding="utf-8")
            (previous / "version.txt").write_text("old", encoding="utf-8")
            with self._globals(root), \
                 patch.object(U, "_validate_staged",
                              side_effect=lambda folder: (folder / "sd-cli", [])):
                U._rollback()
            self.assertEqual((current / "version.txt").read_text(), "old")
            self.assertEqual((previous / "version.txt").read_text(), "new")


if __name__ == "__main__":
    unittest.main()
