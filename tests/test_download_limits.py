"""Real HTTP transfer, resume safety, rate and persisted UI setting."""
import hashlib
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import settings, transfers


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = b"0123456789abcdef" * 16384
        self.limit = 1
        self.requests = []
        self.ignore_range = False
        self.invalid_range = False
        self.truncate = False
        self.meta = SimpleNamespace(size=len(self.data), commit_hash="a" * 40,
                                    etag=hashlib.sha256(self.data).hexdigest())
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
                owner.requests.append(offset)
                if owner.ignore_range:
                    offset = 0
                self.send_response(206 if offset else 200)
                if offset:
                    start = offset + 1 if owner.invalid_range else offset
                    self.send_header("Content-Range",
                                     f"bytes {start}-{len(owner.data)-1}/{len(owner.data)}")
                self.send_header("Content-Length", str(len(owner.data)-offset))
                self.end_headers()
                end = offset + transfers.CHUNK_SIZE if owner.truncate else len(owner.data)
                self.wfile.write(owner.data[offset:end])

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(settings, "USERDATA_DIR", self.root / "user"))
        self.stack.enter_context(patch.object(settings, "load_prefs", side_effect=lambda: {
            "download_limit_mbps": self.limit}))
        self.stack.enter_context(patch("huggingface_hub.constants.HF_HUB_OFFLINE", False))
        self.stack.enter_context(patch("huggingface_hub.get_hf_file_metadata", return_value=self.meta))
        self.stack.enter_context(patch("huggingface_hub.hf_hub_url", return_value=
                                     f"http://127.0.0.1:{server.server_port}/weight"))

    def download(self, name="weight.gguf"):
        return Path(transfers.hf_hub_download(repo_id="test/model", filename=name,
                                              local_dir=str(self.root / "models")))

    def partial(self, data):
        path = self.root / "models/weight.gguf.download"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.with_name(path.name + ".etag").write_text(
            f"test/model\n{self.meta.etag}\n{self.meta.size}")
        return path

    def test_real_http_cap_and_verified_atomic_output(self):
        start = time.monotonic()
        dest = self.download()
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, len(self.data) / 1_000_000 * .95)
        self.assertEqual(dest.read_bytes(), self.data)
        self.assertFalse(dest.with_name(dest.name + ".download").exists())
        self.assertEqual(self.requests, [0])

    def test_resume_only_downloads_remaining_bytes(self):
        self.partial(self.data[:transfers.CHUNK_SIZE])
        self.assertEqual(self.download().read_bytes(), self.data)
        self.assertEqual(self.requests, [transfers.CHUNK_SIZE])

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        self.partial(self.data[:transfers.CHUNK_SIZE])
        self.ignore_range = True
        self.assertEqual(self.download().read_bytes(), self.data)

    def test_invalid_range_is_rejected_without_publishing(self):
        partial = self.partial(self.data[:transfers.CHUNK_SIZE])
        self.invalid_range = True
        with self.assertRaisesRegex(RuntimeError, "Invalid resume"):
            self.download()
        self.assertEqual(partial.stat().st_size, transfers.CHUNK_SIZE)
        self.assertFalse(self.root.joinpath("models/weight.gguf").exists())

    def test_changed_etag_restarts_partial(self):
        partial = self.partial(b"old bytes")
        partial.with_name(partial.name + ".etag").write_text("old identity")
        self.assertEqual(self.download().read_bytes(), self.data)
        self.assertEqual(self.requests, [0])

    def test_truncated_http_file_is_preserved_for_resume(self):
        self.truncate = True
        with self.assertRaises(Exception):
            self.download()
        self.assertFalse(self.root.joinpath("models/weight.gguf").exists())
        self.truncate = False
        self.assertEqual(self.download().read_bytes(), self.data)
        self.assertGreater(self.requests[-1], 0)

    def test_bad_hash_never_becomes_a_model(self):
        self.meta.etag = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.download()
        self.assertFalse(self.root.joinpath("models/weight.gguf").exists())

    def test_completed_partial_is_finalized_without_another_get(self):
        self.partial(self.data)
        self.assertEqual(self.download().read_bytes(), self.data)
        self.assertFalse(self.requests)

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.download("../../escape.gguf")

    def test_unlimited_uses_hub_and_xet_disabled_after_import(self):
        self.limit = 0
        with patch("huggingface_hub.hf_hub_download", return_value="installed") as hub:
            self.assertEqual(str(self.download()), "installed")
        hub.assert_called_once()
        from huggingface_hub import constants
        self.assertTrue(constants.HF_HUB_DISABLE_XET)

    def test_snapshot_filters_and_pins_every_download_to_same_commit(self):
        info = SimpleNamespace(sha="fixed", siblings=[SimpleNamespace(rfilename=s)
                                                     for s in ["a.gguf", "q4/b.gguf", "README.md"]])
        with patch("huggingface_hub.HfApi") as api, \
             patch.object(transfers, "hf_hub_download") as download:
            api.return_value.model_info.return_value = info
            transfers.snapshot_download(repo_id="test/model", local_dir="models",
                                        allow_patterns="*.gguf", ignore_patterns=["q4/*"])
        download.assert_called_once_with(repo_id="test/model", local_dir="models",
                                         filename="a.gguf", revision="fixed")

    def test_live_limit_and_invalid_values(self):
        for value, expected in [(None, 5), ("bad", 5), (float("nan"), 5), (-1, 0), (3, 3)]:
            self.limit = value
            self.assertEqual(transfers.download_limit(), expected)
        self.limit = 0
        transfers.RateLimiter().wait(1_000_000)


if __name__ == "__main__":
    unittest.main()
