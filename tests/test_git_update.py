"""Real local Git repos reproduce ZIP-over-checkout updates without a network."""
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import update_app as U


@unittest.skipUnless(shutil.which("git"), "Git is needed for local updater integration tests")
class GitUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.remote = self.folder / "remote.git"
        self.source = self.folder / "source"
        self.local = self.folder / "local"
        self.source.mkdir()
        self.env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        self.git(self.folder, "init", "--bare", str(self.remote))
        self.git(self.source, "init", "--initial-branch=main")
        self.identity(self.source)
        self.write(self.source, "app.py", "print('old')\n")
        self.write(self.source, "atelier/__init__.py", "VERSION = 1\n")
        self.write(self.source, ".gitignore", "__pycache__/\n*.pyc\n.update-backup/\n")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-m", "initial")
        self.git(self.source, "remote", "add", "origin", str(self.remote))
        self.git(self.source, "push", "-u", "origin", "main")
        self.git(self.folder, "clone", "--branch", "main", str(self.remote), str(self.local))
        self.identity(self.local)
        self.original = self.git(self.local, "rev-parse", "HEAD").stdout.strip()
        self.write(self.source, "app.py", "print('new')\n")
        self.write(self.source, "atelier/[new]…日本.py", "NEW = 2\n")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-m", "incoming update")
        self.git(self.source, "push")
        self.target = self.git(self.source, "rev-parse", "HEAD").stdout.strip()

    def identity(self, root):
        for key, value in (("user.name", "Updater test"), ("user.email", "test@example.invalid"),
                           ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
            self.git(root, "config", key, value)

    def git(self, root, *args, check=True):
        return subprocess.run(["git", *args], cwd=root, env=self.env,
                              capture_output=True, text=True, encoding="utf-8", check=check)

    @staticmethod
    def write(root, rel, content):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))

    def update(self, root=None, **kwargs):
        root = root or self.local
        self.messages = io.StringIO()
        with patch.object(U, "ROOT", root), patch.dict(os.environ, self.env), \
             patch.object(U.sys.stdin, "isatty", return_value=False), redirect_stdout(self.messages):
            return U.update(**kwargs)

    def dirty(self):
        self.write(self.local, "app.py", "print('local staged')\n")
        self.git(self.local, "add", "app.py")
        self.write(self.local, "atelier/__init__.py", "VERSION = 99\n")
        self.write(self.local, "atelier/[new]…日本.py", "NEW = 99\n")

    def assert_old_head(self):
        self.assertEqual(self.git(self.local, "rev-parse", "HEAD").stdout.strip(), self.original)

    def assert_no_stash(self):
        self.assertNotEqual(self.git(self.local, "rev-parse", "--verify", "refs/stash", check=False).returncode, 0)

    def test_clean_checkout_fetches_and_fast_forwards_current_branch(self):
        self.assertEqual(self.update(), 0, self.messages.getvalue())
        self.assertEqual(self.git(self.local, "rev-parse", "HEAD").stdout.strip(), self.target)
        self.assertEqual(self.git(self.local, "branch", "--show-current").stdout.strip(), "main")
        self.assertEqual((self.local / "app.py").read_text(), "print('new')\n")
        self.assert_no_stash()

    def test_zip_over_checkout_backs_up_staged_unstaged_and_colliding_code_only(self):
        self.dirty()
        data = {"models/model.gguf": "weights", "outputs/image.png": "image",
                "userdata/preferences.json": "prefs", "loras/style.bin": "lora",
                "tools_repo/engine/binary": "engine", "python/python.exe": "python",
                "runtime/file": "runtime", "tmp/cache": "cache", "my-notes.txt": "unrelated"}
        for path, content in data.items():
            self.write(self.local, path, content)
        self.assertEqual(self.update(backup_local_code=True), 0, self.messages.getvalue())
        self.assertEqual((self.local / "app.py").read_text(), "print('new')\n")
        self.assertEqual((self.local / "atelier/[new]…日本.py").read_text(), "NEW = 2\n")
        for path, content in data.items():
            self.assertEqual((self.local / path).read_text(), content, path)
        saved = self.git(self.local, "rev-parse", "refs/stash").stdout.strip()
        self.assertEqual(self.git(self.local, "show", f"{saved}:app.py").stdout,
                         "print('local staged')\n")
        self.assertEqual(self.git(self.local, "show", f"{saved}^2:app.py").stdout,
                         "print('local staged')\n")
        self.assertEqual(self.git(self.local, "show", f"{saved}:atelier/__init__.py").stdout,
                         "VERSION = 99\n")
        self.assertEqual(self.git(self.local, "show", f"{saved}^3:atelier/[new]…日本.py").stdout,
                         "NEW = 99\n")
        untracked_saved = self.git(self.local, "ls-tree", "-r", "--name-only", "-z", f"{saved}^3").stdout
        self.assertEqual(untracked_saved, "atelier/[new]…日本.py\0")
        self.assertIn(saved, self.messages.getvalue())

    def test_noninteractive_dirty_checkout_stops_with_recovery_command(self):
        self.dirty()
        self.assertEqual(self.update(), 1)
        self.assert_old_head()
        self.assert_no_stash()
        self.assertEqual((self.local / "app.py").read_text(), "print('local staged')\n")
        self.assertIn("update.bat --backup-local-code", self.messages.getvalue())

    def test_interactive_yes_saves_code_and_continues(self):
        self.dirty()
        with patch.object(U, "ROOT", self.local), patch.dict(os.environ, self.env), \
             patch.object(U.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="oui") as prompt, redirect_stdout(io.StringIO()):
            self.assertEqual(U.update(), 0)
        prompt.assert_called_once()
        self.assertEqual(self.git(self.local, "rev-parse", "HEAD").stdout.strip(), self.target)

    def test_interactive_no_keeps_local_code_and_head(self):
        self.dirty()
        with patch.object(U, "ROOT", self.local), patch.dict(os.environ, self.env), \
             patch.object(U.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="non"), redirect_stdout(io.StringIO()):
            self.assertEqual(U.update(), 1)
        self.assert_old_head()
        self.assert_no_stash()

    def test_check_never_fetches_stashes_or_merges_even_with_backup_flag(self):
        self.dirty()
        cached = self.git(self.local, "rev-parse", "origin/main").stdout
        with patch.object(U, "_git", wraps=U._git) as commands, patch.object(U, "_fetch") as fetch:
            self.assertEqual(self.update(check_only=True, backup_local_code=True), 0)
        self.assertFalse(any(c.args[0] in ("fetch", "stash", "merge") for c in commands.call_args_list))
        fetch.assert_not_called()
        self.assert_old_head()
        self.assert_no_stash()
        self.assertEqual(self.git(self.local, "rev-parse", "origin/main").stdout, cached)
        self.assertEqual((self.local / "app.py").read_text(), "print('local staged')\n")

    def test_divergence_stops_before_any_backup(self):
        self.write(self.local, "app.py", "print('local commit')\n")
        self.git(self.local, "add", "app.py")
        self.git(self.local, "commit", "-m", "local commit")
        current = self.git(self.local, "rev-parse", "HEAD").stdout
        self.write(self.local, "atelier/__init__.py", "VERSION = 99\n")
        self.assertEqual(self.update(backup_local_code=True), 1)
        self.assertEqual(self.git(self.local, "rev-parse", "HEAD").stdout, current)
        self.assert_no_stash()
        self.assertIn("diverged", self.messages.getvalue())

    def test_detached_commit_and_missing_upstream_are_explained(self):
        self.git(self.local, "checkout", "--detach")
        self.assertEqual(self.update(), 1)
        self.assertIn("detached", self.messages.getvalue())
        self.git(self.local, "checkout", "main")
        self.git(self.local, "branch", "--unset-upstream")
        self.assertEqual(self.update(), 1)
        self.assertIn("no upstream", self.messages.getvalue())
        self.assert_old_head()

    def test_failed_fetch_keeps_local_changes_and_creates_no_stash(self):
        self.dirty()
        self.git(self.local, "remote", "set-url", "origin", str(self.folder / "missing.git"))
        self.assertEqual(self.update(backup_local_code=True), 1)
        self.assert_old_head()
        self.assert_no_stash()
        self.assertEqual((self.local / "app.py").read_text(), "print('local staged')\n")

    def test_failed_stash_stops_without_updating_code(self):
        self.dirty()
        original = U._git

        def git(*args, **kwargs):
            if args[:2] == ("stash", "push"):
                raise RuntimeError("Simulated backup failure")
            return original(*args, **kwargs)

        with patch.object(U, "_git", side_effect=git):
            self.assertEqual(self.update(backup_local_code=True), 1)
        self.assert_old_head()
        self.assert_no_stash()
        self.assertEqual((self.local / "app.py").read_text(), "print('local staged')\n")

    def test_failed_fast_forward_keeps_the_code_backup(self):
        self.dirty()
        original = U._git

        def git(*args, **kwargs):
            if args[0] == "merge":
                raise RuntimeError("Simulated merge failure")
            return original(*args, **kwargs)

        with patch.object(U, "_git", side_effect=git):
            self.assertEqual(self.update(backup_local_code=True), 1)
        self.assert_old_head()
        self.assertEqual(self.git(self.local, "show", "refs/stash:app.py").stdout,
                         "print('local staged')\n")
        self.assertIn("Simulated merge failure", self.messages.getvalue())

    def test_incoming_data_changes_are_rejected_before_backup(self):
        self.write(self.source, "outputs/image.png", "remote image")
        self.git(self.source, "add", "outputs/image.png")
        self.git(self.source, "commit", "-m", "unexpected data")
        self.git(self.source, "push")
        self.dirty()
        self.write(self.local, "outputs/image.png", "my image")
        self.assertEqual(self.update(backup_local_code=True), 1)
        self.assert_old_head()
        self.assert_no_stash()
        self.assertEqual((self.local / "outputs/image.png").read_text(), "my image")

    def test_worktree_git_file_uses_its_current_tracking_branch(self):
        worktree = self.folder / "worktree"
        self.git(self.local, "worktree", "add", "-b", "preview", str(worktree))
        self.git(worktree, "branch", "--set-upstream-to=origin/main")
        self.assertTrue((worktree / ".git").is_file())
        self.assertEqual(self.update(root=worktree), 0, self.messages.getvalue())
        self.assertEqual(self.git(worktree, "branch", "--show-current").stdout.strip(), "preview")
        self.assert_old_head()

    def test_unfinished_operation_missing_git_and_zip_rollback_do_not_mutate(self):
        merge = self.local / ".git" / "MERGE_HEAD"
        merge.write_text(self.original)
        self.assertEqual(self.update(backup_local_code=True), 1)
        self.assertIn("unfinished", self.messages.getvalue())
        merge.unlink()
        with patch.object(U.shutil, "which", return_value=None):
            self.assertEqual(self.update(), 1)
        self.assertIn("Install Git for Windows", self.messages.getvalue())
        with patch.object(U, "ROOT", self.local), redirect_stdout(io.StringIO()):
            self.assertEqual(U._rollback(), 1)
        self.assert_old_head()


if __name__ == "__main__":
    unittest.main()
