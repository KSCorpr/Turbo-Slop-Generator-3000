"""VOSR install, shared model storage, controls, cancellation and image publication."""
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from PIL import Image

from atelier import addons, inventory, model_catalog, settings, vosr as artifacts
from atelier.engine import vosr
from atelier.engine.local_jobs import LocalJob
from atelier.ui import vosr_tab
from scripts import setup_media


class ArtifactTests(unittest.TestCase):
    def test_catalog_uses_one_install_and_only_the_five_pinned_files(self):
        self.assertEqual([key for _, key in model_catalog.choices("Toolkit", "VOSR")], ["addon:vosr"])
        with patch.object(model_catalog.JOB, "run") as run, patch.object(model_catalog, "ready", return_value=True):
            model_catalog.install("addon:vosr", log=lambda _: None)
        self.assertTrue(str(run.call_args.args[0][-2]).endswith("setup_media.py"))
        self.assertEqual(run.call_args.args[0][-1], "vosr")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], artifacts.MODEL_REPO)
                self.assertEqual(kwargs["revision"], artifacts.MODEL_REVISION)
                self.assertEqual(kwargs["local_dir"], str(root))
                path = root / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    stream.truncate(artifacts.FILES[kwargs["filename"]])
                return str(path)
            with patch.object(artifacts, "model_dir", return_value=root), \
                 patch.object(artifacts.transfers, "hf_hub_download", side_effect=download) as transfer:
                artifacts.download_models(log=lambda _: None)
                self.assertTrue(artifacts.weights_ready())
                (root / "Qwen-Image-vae-2d/config.json").write_bytes(b"incomplete")
                self.assertFalse(artifacts.weights_ready())
            self.assertEqual(transfer.call_count, 5)
            self.assertEqual({call.kwargs["filename"] for call in transfer.call_args_list}, set(artifacts.FILES))

    def test_incomplete_install_marker_and_missing_source_cannot_report_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(addons, "root", return_value=root), \
                 patch.object(addons, "ready", return_value=True), \
                 patch.object(artifacts, "weights_ready", return_value=True):
                self.assertFalse(artifacts.ready())
                (root / "installed.json").write_text(json.dumps({"commit": artifacts.SOURCE_COMMIT,
                    "revision": artifacts.MODEL_REVISION, "adapter": 1}))
                self.assertFalse(artifacts.ready())
                (root / "source").mkdir(); (root / "source/progress.py").touch()
                self.assertTrue(artifacts.ready())
                (root / "installed.json").write_text("broken")
                self.assertFalse(artifacts.ready())

    def test_gpu_probe_precedes_weight_download_and_failed_install_has_no_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []
            with patch.object(addons, "root", return_value=root), \
                 patch.object(setup_media, "environment", return_value=root / "python"), \
                 patch.object(setup_media, "pip") as pip, patch.object(setup_media, "_install_source"), \
                 patch.object(artifacts, "adapt_source"), patch.object(artifacts, "load_args"), \
                 patch.object(artifacts, "download_models", side_effect=lambda: calls.append("download")), \
                 patch.object(setup_media, "run", side_effect=lambda *args: calls.append("probe")):
                setup_media.main(["vosr"])
                self.assertEqual(calls, ["probe", "download"])
                self.assertTrue((root / "installed.json").is_file())
                self.assertIn("torch==2.7.1+cu126", pip.call_args_list[0].args)
            with patch.object(addons, "root", return_value=root), \
                 patch.object(setup_media, "environment", return_value=root / "python"), \
                 patch.object(setup_media, "pip"), patch.object(setup_media, "_install_source"), \
                 patch.object(artifacts, "adapt_source"), patch.object(artifacts, "download_models") as download, \
                 patch.object(setup_media, "run", side_effect=RuntimeError("no GPU")):
                with self.assertRaisesRegex(RuntimeError, "no GPU"):
                    setup_media.install_vosr()
                self.assertFalse((root / "installed.json").exists())
                download.assert_not_called()

    def test_inventory_separates_runtime_and_one_weight_bundle(self):
        entries = {item.key: item for item in inventory.items()}
        self.assertEqual(entries["vosr_models"].paths, [artifacts.model_dir()])
        self.assertNotIn(artifacts.model_dir(), entries["engine_vosr"].paths)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.source = self.root / "source…é_日本.png"
        Image.new("RGBA", (37, 25), (50, 100, 150, 75)).save(self.source)
        self.output = self.root / "output.png"
        self.job = LocalJob()
        for obj, name, value in ((vosr, "JOB", self.job), (settings, "TMP_DIR", self.root / "tmp")):
            self.stack.enter_context(patch.object(obj, name, value))
        self.stack.enter_context(patch.object(artifacts, "ready", return_value=True))
        self.stack.enter_context(patch.object(vosr.sdcpp, "unique_output", return_value=self.output))

    def test_unicode_input_is_snapshotted_and_selected_gpu_and_controls_reach_child(self):
        original = self.source.read_bytes()
        snapshots = []
        def run(command, log, **kwargs):
            source = Path(command[command.index("--input") + 1]); snapshots.append(source)
            self.assertTrue(source.name.isascii()); self.assertEqual(source.read_bytes(), original)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(command[command.index("--dtype") + 1], "bf16")
            self.assertIn("--keep-on-gpu", command)
            Image.open(source).resize((74, 50)).save(command[command.index("--output") + 1])
        with patch.object(self.job, "run", side_effect=run), \
             patch.object(settings, "generation_gpu_index", return_value=1):
            result = vosr.restore(self.source, 2, 123, dtype="bf16", offload=False, log=lambda _: None)
        self.assertEqual(result, self.output)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(snapshots[0].exists())
        with Image.open(result) as restored:
            self.assertEqual(restored.size, (74, 50)); self.assertEqual(restored.mode, "RGBA")

    def test_invalid_controls_and_oversized_images_fail_before_subprocess(self):
        for options in ({"scale": 3}, {"seed": -1}, {"tile": 0}, {"overlap": 512},
                        {"vae_overlap": 3}, {"dtype": "fp32"}, {"color": "invalid"}):
            with patch.object(self.job, "run") as run, self.assertRaises(ValueError):
                vosr.restore(self.source, **options)
            run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "32 megapixels"):
            vosr.target_size((4000, 4000), 4)
        with patch.object(artifacts, "ready", return_value=False), self.assertRaisesRegex(RuntimeError, "Install / repair"):
            vosr.restore(self.source)

    def test_cancel_failure_and_wrong_dimensions_do_not_publish_partial_outputs(self):
        for mode in ("cancel", "failure", "wrong-size", "corrupt"):
            def run(command, log, **kwargs):
                result = Path(command[command.index("--output") + 1])
                if mode == "corrupt":
                    result.write_bytes(b"broken")
                else:
                    Image.new("RGB", (2, 2)).save(result)
                if mode == "cancel":
                    self.job.cancel(); self.job.check()
                if mode == "failure":
                    raise RuntimeError("CUDA failure")
            with self.subTest(mode=mode), patch.object(self.job, "run", side_effect=run):
                with self.assertRaises((RuntimeError, OSError)):
                    vosr.restore(self.source, log=lambda _: None)
                self.assertFalse(self.output.exists())
                self.assertEqual(list((self.root / "tmp").iterdir()), [])
                self.assertTrue(self.source.is_file())

    def test_size_preview_accounts_for_exif_orientation(self):
        path = self.root / "rotated.jpg"
        exif = Image.Exif(); exif[274] = 6
        Image.new("RGB", (100, 50)).save(path, exif=exif)
        self.assertIn("100 × 200", vosr_tab.size_info(path, 2))
        self.assertIn("Import", vosr_tab.size_info(None, 4))


if __name__ == "__main__":
    unittest.main()
