"""Installer retries real HTTP failures without replacing working files."""
import io
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scripts import setup_media


class MediaDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.destination = Path(self.tmp.name) / "source.zip"
        self.destination.write_bytes(b"existing working archive")
        self.calls = 0
        self.codes = [200]
        self.truncate_first = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                attempt = owner.calls
                owner.calls += 1
                code = owner.codes[min(attempt, len(owner.codes) - 1)]
                self.send_response(code)
                self.send_header("Content-Length", "7" if code == 200 else "0")
                self.end_headers()
                if code == 200:
                    self.wfile.write(b"part" if owner.truncate_first and attempt == 0 else b"archive")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}/source.zip"
        patcher = patch.object(setup_media.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_504_then_success_retries_real_http_and_publishes_only_complete_file(self):
        self.codes = [504, 200]
        setup_media.download(self.url, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"archive")
        self.assertEqual(self.calls, 2)
        self.sleep.assert_called_once_with(1)
        self.assertFalse(self.destination.with_suffix(".zip.part").exists())

    def test_repeated_504_has_bounded_attempts_and_preserves_previous_file(self):
        self.codes = [504]
        with self.assertRaisesRegex(RuntimeError, "after 4 attempts.*HTTP 504"):
            setup_media.download(self.url, self.destination)
        self.assertEqual(self.calls, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 4])
        self.assertEqual(self.destination.read_bytes(), b"existing working archive")
        self.assertFalse(self.destination.with_suffix(".zip.part").exists())

    def test_permanent_404_does_not_retry(self):
        self.codes = [404]
        with self.assertRaises(urllib.error.HTTPError):
            setup_media.download(self.url, self.destination)
        self.assertEqual(self.calls, 1)
        self.sleep.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"existing working archive")

    def test_truncated_response_restarts_without_appending_corrupt_bytes(self):
        self.truncate_first = True
        setup_media.download(self.url, self.destination)
        self.assertEqual(self.calls, 2)
        self.assertEqual(self.destination.read_bytes(), b"archive")

    def test_archive_host_fallback_preserves_the_pinned_commit(self):
        response = io.BytesIO(b"archive")
        response.headers = {"Content-Length": "7"}
        sha = "a" * 40
        original = f"https://github.com/owner/repo/archive/{sha}.zip"
        timeout = urllib.error.HTTPError(original, 504, "Gateway timeout", {}, None)
        with patch.object(setup_media.urllib.request, "urlopen",
                          side_effect=[timeout, response]) as open_url:
            setup_media.download(original, self.destination)
        self.assertEqual([c.args[0].full_url for c in open_url.call_args_list], [
            f"https://codeload.github.com/owner/repo/zip/{sha}", original])
        self.assertEqual(self.destination.read_bytes(), b"archive")

    def test_network_timeout_then_success_retries(self):
        response = io.BytesIO(b"archive")
        response.headers = {"Content-Length": "7"}
        with patch.object(setup_media.urllib.request, "urlopen",
                          side_effect=[TimeoutError("timed out"), response]):
            setup_media.download(self.url, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"archive")

    def test_failed_source_download_leaves_installed_source_intact(self):
        target = Path(self.tmp.name) / "installed-source"
        target.mkdir()
        sentinel = target / "inference_cli.py"
        sentinel.write_text("working source")
        with patch.object(setup_media, "download", side_effect=RuntimeError("HTTP 504")):
            with self.assertRaisesRegex(RuntimeError, "HTTP 504"):
                setup_media._install_source("owner/repo", "a" * 40, target)
        self.assertEqual(sentinel.read_text(), "working source")


if __name__ == "__main__":
    unittest.main()
