"""Real HTTP transfer, resume safety, rate and persisted UI setting."""
import hashlib
import math
import socket
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
        self.stack.enter_context(patch.object(settings, "MODELS_DIR", self.root / "models"))
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

    def test_identical_weights_from_another_repo_reuse_bytes_without_another_get(self):
        first = self.download()
        second = Path(transfers.hf_hub_download(repo_id="another/repo",filename="shared/encoder.safetensors",
                                               local_dir=str(self.root/"models")))
        self.assertEqual(self.requests,[0])
        self.assertTrue(first.samefile(second))
        first.unlink()
        self.assertEqual(second.read_bytes(),self.data)

    def test_same_size_changed_revision_replaces_only_the_selected_alias(self):
        first = self.download()
        alias = self.download("copy.gguf")
        self.data = b"x"*len(self.data)
        self.meta.etag = hashlib.sha256(self.data).hexdigest()
        changed = self.download()
        self.assertEqual(changed.read_bytes(),self.data)
        self.assertNotEqual(alias.read_bytes(),self.data)
        self.assertFalse(changed.samefile(alias))
        self.assertEqual(self.requests,[0,0])

    def test_cancelled_transfer_remains_partial_and_can_resume(self):
        calls=[0]
        def check():
            calls[0]+=1
            if calls[0] >= 4:
                raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError,"cancelled"), transfers.cancel_scope(check):
            self.download()
        self.assertFalse((self.root/"models/weight.gguf").exists())
        self.assertTrue((self.root/"models/weight.gguf.download").exists())
        self.assertEqual(self.download().read_bytes(),self.data)

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


class RateAndBufferTests(unittest.TestCase):
    def test_network_read_time_is_credited_instead_of_added_to_the_limit(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(transfers, "download_limit", return_value=1), \
             patch.object(transfers.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(transfers.time, "sleep", side_effect=sleep):
            limiter = transfers.RateLimiter()
            for _ in range(10):
                limiter.wait(100_000)
                clock[0] += .1  # source réseau déjà limitée à 1 Mo/s
        # L'ancien code prenait 2 s : 1 s réseau PLUS 1 s de pauses.
        self.assertAlmostEqual(clock[0], 1.1)

    def test_coarse_windows_timer_does_not_accumulate_oversleep(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += math.ceil(seconds / .015625) * .015625
        with patch.object(transfers, "download_limit", return_value=5), \
             patch.object(transfers.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(transfers.time, "sleep", side_effect=sleep):
            limiter = transfers.RateLimiter()
            for _ in range(200):
                limiter.wait(transfers.CHUNK_SIZE)
        ideal = 200 * transfers.CHUNK_SIZE / 5_000_000
        self.assertGreaterEqual(clock[0], ideal)
        self.assertLess(clock[0], ideal + .032)

    def test_no_large_burst_credit_accumulates_after_a_long_pause(self):
        clock = [0.0]
        sleeps = []
        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
        with patch.object(transfers, "download_limit", return_value=1), \
             patch.object(transfers.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(transfers.time, "sleep", side_effect=sleep):
            limiter = transfers.RateLimiter()
            limiter.wait(100_000)
            clock[0] += 60
            sleeps.clear()
            limiter.wait(100_000)
            limiter.wait(100_000)
            limiter.wait(100_000)
        self.assertAlmostEqual(sum(sleeps), .1)

    def test_receive_buffer_is_set_for_http_https_and_proxy_before_connect(self):
        with patch.object(transfers, "download_limit", return_value=1), \
             transfers.download_session() as session:
            option = (socket.SOL_SOCKET, socket.SO_RCVBUF, 100_000)
            for scheme in ["http://", "https://"]:
                adapter = session.get_adapter(scheme)
                self.assertIn(option, adapter.poolmanager.connection_pool_kw["socket_options"])
                proxy = adapter.proxy_manager_for("http://127.0.0.1:9999")
                self.assertIn(option, proxy.connection_pool_kw["socket_options"])

    def test_unlimited_leaves_socket_buffers_at_system_defaults(self):
        with patch.object(transfers, "download_limit", return_value=0), \
             transfers.download_session() as session:
            self.assertNotIn("socket_options", session.get_adapter("https://").poolmanager.connection_pool_kw)


if __name__ == "__main__":
    unittest.main()
