"""Verify Kandinsky ImageSR 2.0 catalog registration, installer, engine runner, and UI."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import addons, inventory, kandinsky_sr as artifacts, model_catalog, settings
from atelier.engine import kandinsky_sr
from atelier.engine.local_jobs import LocalJob
from atelier.ui import kandinsky_sr_tab
from scripts import setup_media
from scripts.tools import run_kandinsky_sr


class KandinskySRTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(settings, "ROOT", self.root))
        self.stack.enter_context(patch.object(settings, "TMP_DIR", self.root / "tmp"))
        self.stack.enter_context(patch.object(settings, "OUTPUT_DIR", self.root / "outputs"))
        (self.root / "outputs").mkdir()
        self.source = self.root / "photo.png"
        Image.new("RGBA", (128, 96), (40, 120, 200, 180)).save(self.source)

    def _populate_installed(self):
        root = addons.root("kandinsky-sr")
        py = addons.python("kandinsky-sr")
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"py")
        pipeline = root / "source" / "src" / "kandinsky_image_sr" / "pipeline.py"
        pipeline.parent.mkdir(parents=True, exist_ok=True)
        pipeline.write_text("# stub\n", encoding="utf-8")
        for filename, size in artifacts.FILES.items():
            dest = artifacts.model_dir() / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                fh.truncate(size)
        (root / "installed.json").write_text(
            json.dumps({"commit": artifacts.SOURCE_COMMIT, "revision": artifacts.MODEL_REVISION}),
            encoding="utf-8",
        )

    def test_catalog_registration_install_command_and_inventory(self):
        self.assertEqual(
            [key for _, key in model_catalog.choices("Toolkit", "Kandinsky")],
            ["addon:kandinsky-sr"],
        )
        self.assertFalse(model_catalog.ready("addon:kandinsky-sr"))
        with patch.object(model_catalog.JOB, "run", side_effect=lambda *_, **__: self._populate_installed()) as run:
            model_catalog.install("addon:kandinsky-sr", log=lambda _: None)
        self.assertEqual(run.call_args.args[0][-1], "kandinsky-sr")

        entries = {item.key: item for item in inventory.items()}
        self.assertEqual(entries["kandinsky_sr_models"].paths, [artifacts.model_dir()])
        self.assertNotIn(artifacts.model_dir(), entries["engine_kandinsky_sr"].paths)

    def test_setup_media_install_kandinsky_sr_probes_downloads_and_writes_marker(self):
        root = addons.root("kandinsky-sr")
        py = addons.python("kandinsky-sr")
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"py")

        def fake_source(repo, commit, target):
            self.assertEqual(repo, artifacts.SOURCE_REPO)
            self.assertEqual(commit, artifacts.SOURCE_COMMIT)
            pipe = target / "src" / "kandinsky_image_sr" / "pipeline.py"
            pipe.parent.mkdir(parents=True, exist_ok=True)
            pipe.write_text("# stub\n", encoding="utf-8")

        def fake_download(repo_id, filename, revision, local_dir):
            self.assertEqual(repo_id, artifacts.MODEL_REPO)
            self.assertEqual(revision, artifacts.MODEL_REVISION)
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                fh.truncate(artifacts.FILES[filename])
            return str(dest)

        with patch.object(setup_media, "environment", return_value=py), \
             patch.object(setup_media, "pip") as pip_mock, \
             patch.object(setup_media, "_install_source", side_effect=fake_source), \
             patch.object(setup_media, "run") as run_mock, \
             patch.object(artifacts.transfers, "hf_hub_download", side_effect=fake_download):
            setup_media.main(["kandinsky-sr"])

        self.assertTrue(artifacts.ready())
        self.assertTrue(model_catalog.ready("addon:kandinsky-sr"))
        self.assertGreaterEqual(pip_mock.call_count, 2)
        self.assertIn("--probe", [str(a) for a in run_mock.call_args.args])
        marker = json.loads((root / "installed.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["commit"], artifacts.SOURCE_COMMIT)
        self.assertEqual(marker["revision"], artifacts.MODEL_REVISION)

    def test_engine_validation_and_offline_subprocess_execution(self):
        self._populate_installed()
        output = self.root / "outputs" / "kandinsky-sr2-test.png"

        def fake_run(cmd, log, env=None):
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            cmd_str = [str(c) for c in cmd]
            self.assertEqual(cmd_str[cmd_str.index("--upscale") + 1], "4")
            self.assertEqual(cmd_str[cmd_str.index("--creativity") + 1], "6")
            self.assertEqual(cmd_str[cmd_str.index("--overlap") + 1], "0.3")
            self.assertEqual(cmd_str[cmd_str.index("--dtype") + 1], "fp16")
            self.assertEqual(cmd_str[cmd_str.index("--memory-mode") + 1], "sequential")
            out_path = Path(cmd_str[cmd_str.index("--output") + 1])
            Image.new("RGBA", (512, 384), (10, 20, 30, 255)).save(out_path)

        job = LocalJob()
        with patch.object(kandinsky_sr, "JOB", job), \
             patch.object(job, "run", side_effect=fake_run), \
             patch.object(kandinsky_sr.sdcpp, "unique_output", return_value=output):
            result = kandinsky_sr.upscale(
                self.source, 4, 6, overlap=0.3, dtype="fp16", memory_mode="sequential", log=lambda _: None
            )
        self.assertEqual(result, output)
        self.assertTrue(output.is_file())

        # Validation guards
        for bad in (
            dict(scale=3),
            dict(creativity=9),
            dict(creativity=-1),
            dict(overlap=0.9),
            dict(dtype="int4"),
            dict(memory_mode="disk"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kandinsky_sr.upscale(self.source, **bad)
        # 4K -> 8K (67.1 MP) is allowed; 16K is rejected.
        self.assertEqual(kandinsky_sr.target_size((4096, 4096), 2), (8192, 8192))
        with self.assertRaises(ValueError):
            kandinsky_sr.target_size((4096, 4096), 4)

    def test_runner_memory_mode_auto_selection_for_11gb_16gb_24gb(self):
        class _FakeCuda:
            def __init__(self, gib):
                self._gib = gib

            def is_available(self):
                return True

            def get_device_properties(self, _idx):
                return type("P", (), {"total_memory": int(self._gib * 1024**3)})()

        for gib, expected in ((11.0, "sequential"), (12.0, "sequential"), (16.0, "model"), (24.0, "gpu")):
            fake_torch = type("T", (), {"cuda": _FakeCuda(gib)})()
            self.assertEqual(run_kandinsky_sr._resolve_memory_mode(fake_torch, "auto"), expected)

    def test_ui_tab_size_and_creativity_helpers_and_controls(self):
        self.assertIn("128 × 96 → **512 × 384 px**", kandinsky_sr_tab.size_info(self.source, 4))
        self.assertIn("Import", kandinsky_sr_tab.size_info(None, 4))
        self.assertIn("0.15", kandinsky_sr_tab.creativity_info(2, 4))
        self.assertIn("0.244", kandinsky_sr_tab.creativity_info(4, 4))
        self.assertIn("0.38", kandinsky_sr_tab.creativity_info(4, 8))

        with gr.Blocks() as demo:
            kandinsky_sr_tab.build_kandinsky_sr_tab()
        self.addCleanup(demo.close)
        labels = {getattr(b, "label", None) for b in demo.blocks.values()}
        self.assertIn("Upscale factor", labels)
        self.assertIn("Creativity (calibrated 0–8)", labels)
        self.assertIn("Memory mode", labels)


if __name__ == "__main__":
    unittest.main()
