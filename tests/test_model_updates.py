"""Content-based update detection, old installs, pinned tools and launcher wiring."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atelier import model_updates as updates, model_store, settings, transfers
from scripts import update_app


class ModelUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name, path in (("ROOT", self.root), ("MODELS_DIR", self.root / "models"),
                           ("USERDATA_DIR", self.root / "userdata"), ("LORA_DIR", self.root / "loras")):
            path.mkdir(exist_ok=True)
            self.stack.enter_context(patch.object(settings, name, path))
        self.stack.enter_context(patch("huggingface_hub.constants.HF_HUB_OFFLINE", False))

    def file(self, name="weight.gguf", data=b"old weights"):
        path = settings.MODELS_DIR / "test__model" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return path

    def remote(self, path, data=None, git=False):
        data = path.read_bytes() if data is None else data
        sha = (hashlib.sha1(f"blob {len(data)}\0".encode() + data) if git else hashlib.sha256(data)).hexdigest()
        return SimpleNamespace(rfilename=path.relative_to(settings.MODELS_DIR / "test__model").as_posix(),
            size=len(data), blob_id=sha if git else "blob", lfs=None if git else SimpleNamespace(sha256=sha))

    def run_check(self, files):
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value = SimpleNamespace(sha="new commit", siblings=files)
            report = updates.check(log=lambda _: None)
            return report, api.return_value.model_info.call_args_list

    def test_repo_commit_change_does_not_mark_unchanged_files_and_one_request_per_repo(self):
        a = self.file(); b = self.file("encoder.gguf", b"encoder")
        for path in (a, b):
            updates.record(path, repo="test/model", filename=path.name, revision="main",
                           commit="old commit", etag=hashlib.sha256(path.read_bytes()).hexdigest())
        files = [self.remote(a, b"old weights"), self.remote(b, b"encoder")]
        with patch.object(updates.hashlib, "sha256", side_effect=AssertionError("must not rehash")):
            report, calls = self.run_check(files)
        self.assertEqual(report["counts"]["up_to_date"], 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["files_metadata"], True)

    def test_same_size_changed_weights_are_detected_and_not_replaced(self):
        path = self.file(data=b"AAAA")
        model_store.remember(path)
        report, _ = self.run_check([self.remote(path, b"BBBB")])
        self.assertEqual(report["counts"]["update_available"], 1)
        self.assertEqual(path.read_bytes(), b"AAAA")

    def test_old_untracked_file_is_identified_once_and_cached_for_future_checks(self):
        path = self.file()
        remote = self.remote(path)
        report, _ = self.run_check([remote])
        self.assertEqual(report["counts"]["up_to_date"], 1)
        with patch.object(updates.hashlib, "sha256", side_effect=AssertionError("must not rehash")):
            report, _ = self.run_check([remote])
        self.assertEqual(report["counts"]["up_to_date"], 1)
        self.assertTrue((settings.USERDATA_DIR / "model-downloads.json").is_file())

    def test_small_config_uses_git_blob_hash_not_plain_sha1_or_sha256(self):
        path = self.file("config.json", b'{"dtype":"fp16"}')
        report, _ = self.run_check([self.remote(path, git=True)])
        self.assertEqual(report["counts"]["up_to_date"], 1)

    def test_pinned_tool_has_separate_update_status_and_keeps_original_revision(self):
        path = self.file()
        revision = "a" * 40
        updates.record(path, repo="test/model", filename=path.name, revision=revision,
                       commit=revision, etag=hashlib.sha256(path.read_bytes()).hexdigest())
        report, calls = self.run_check([self.remote(path, b"new version")])
        self.assertEqual(report["counts"]["pinned_update"], 1)
        self.assertEqual(calls[0].kwargs["revision"], "main")
        self.assertEqual(updates.discover()[0]["revision"], revision)
        self.assertEqual(path.read_bytes(), b"old weights")

    def test_local_edits_deleted_upstream_files_and_network_failures_are_not_up_to_date(self):
        path = self.file()
        updates.record(path, repo="test/model", filename=path.name, revision="main", commit="old",
                       etag=hashlib.sha256(path.read_bytes()).hexdigest())
        path.write_bytes(b"custom edited model")
        report, _ = self.run_check([self.remote(path, b"upstream")])
        self.assertEqual(report["counts"]["local_modified"], 1)
        path.write_bytes(b"upstream")
        report, _ = self.run_check([self.remote(path, b"upstream")])
        self.assertEqual(report["counts"]["up_to_date"], 1)
        report, _ = self.run_check([])
        self.assertEqual(report["counts"]["missing_upstream"], 1)
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.side_effect = OSError("offline")
            report = updates.check(log=lambda _: None)
        self.assertEqual(report["counts"]["unavailable"], 1)

    def test_offline_and_unmapped_weights_do_not_trigger_network_or_model_download(self):
        self.file()
        custom = settings.MODELS_DIR / "custom.gguf"; custom.write_bytes(b"custom")
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), \
             patch("huggingface_hub.HfApi") as api, patch.object(transfers, "hf_hub_download") as download:
            report = updates.check(log=lambda _: None)
        api.return_value.model_info.assert_not_called(); download.assert_not_called()
        self.assertEqual(report["counts"]["unavailable"], 1)
        self.assertEqual(report["counts"]["unmapped"], 1)
        self.assertEqual(custom.read_bytes(), b"custom")

    def test_snapshot_tracking_preserves_requested_branch_and_pinned_commits(self):
        for revision in ("main", "a" * 40):
            with patch("huggingface_hub.HfApi") as api, patch.object(transfers, "hf_hub_download") as download:
                api.return_value.model_info.return_value = SimpleNamespace(sha="b" * 40,
                    siblings=[SimpleNamespace(rfilename="model.safetensors")])
                transfers.snapshot_download(repo_id="test/model", local_dir=str(self.root), revision=revision)
            self.assertEqual(download.call_args.kwargs["revision"], "b" * 40)
            self.assertEqual(download.call_args.kwargs["_tracking_revision"], revision)

    def test_default_and_standalone_models_flags_run_fresh_code_and_check_mode_does_not(self):
        for argv, flag in (([], "--check-models"), (["--models"], "check_model_updates.py")):
            with patch.object(update_app.sys, "argv", ["update_app.py", *argv]), \
                 patch.object(update_app, "update", return_value=0), patch("subprocess.call", return_value=0) as call:
                self.assertEqual(update_app.main(), 0)
            command = " ".join(call.call_args.args[0])
            self.assertIn(flag, command)
        with patch.object(update_app.sys, "argv", ["update_app.py", "--models", "--check"]), \
             patch("subprocess.call") as call:
            self.assertEqual(update_app.main(), 0)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
