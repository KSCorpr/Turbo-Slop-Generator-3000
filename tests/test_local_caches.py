"""Verify that all caches and temporary directories stay inside the app folder."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import inventory, settings
from scripts import maintenance


class LocalCacheIsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_local_cache_env_keeps_every_cache_and_temp_dir_inside_app_root(self):
        mapping = settings.local_cache_env(self.root)
        expected_keys = {
            "UV_CACHE_DIR",
            "UV_PYTHON_INSTALL_DIR",
            "PIP_CACHE_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "XDG_CACHE_HOME",
            "MPLCONFIGDIR",
            "NUMBA_CACHE_DIR",
            "GRADIO_TEMP_DIR",
            "TMP",
            "TEMP",
            "TMPDIR",
        }
        self.assertTrue(expected_keys.issubset(mapping.keys()))
        resolved_root = self.root.resolve()
        for key, raw_path in mapping.items():
            resolved = Path(raw_path).resolve()
            self.assertTrue(
                resolved == resolved_root or resolved_root in resolved.parents,
                f"{key}={raw_path} escapes {self.root}",
            )
        # Managed uv Python lives under runtime/, NOT disposable .cache/.
        self.assertNotIn(".cache", Path(mapping["UV_PYTHON_INSTALL_DIR"]).relative_to(self.root).parts)
        self.assertIn("runtime", Path(mapping["UV_PYTHON_INSTALL_DIR"]).relative_to(self.root).parts)

    def test_configure_local_caches_overrides_external_vars_and_creates_dirs(self):
        env = {
            "UV_CACHE_DIR": "/external/uv",
            "HF_HOME": "/external/hf",
            "TEMP": "/external/tmp",
        }
        mapping = settings.configure_local_caches(env, root=self.root, create=True)
        for key, val in mapping.items():
            self.assertEqual(env[key], val)
            self.assertTrue(Path(val).is_dir(), f"{key} directory missing: {val}")

    def test_child_env_inherits_in_app_cache_paths(self):
        with patch.object(settings, "CACHE_DIR", self.root / ".cache"), \
             patch.object(settings, "TMP_DIR", self.root / "tmp"), \
             patch.object(settings, "RUNTIME_DIR", self.root / "runtime"):
            env = settings.child_env(gpu_index=0)
        self.assertEqual(env["UV_CACHE_DIR"], str(self.root / ".cache" / "uv"))
        self.assertEqual(env["PIP_CACHE_DIR"], str(self.root / ".cache" / "pip"))
        self.assertEqual(env["HF_HOME"], str(self.root / ".cache" / "huggingface"))
        self.assertEqual(env["TORCH_HOME"], str(self.root / ".cache" / "torch"))
        self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(self.root / "runtime" / "uv-python"))
        self.assertEqual(env["TEMP"], str(self.root / ".cache" / "tmp"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue((self.root / ".cache" / "tmp").is_dir())

    def test_sync_local_pip_configs_and_maintenance_migration(self):
        # Simulate an existing installation with portable python, two addon venvs,
        # and legacy caches (tmp/uv-cache and hf_cache/).
        (self.root / "python").mkdir()
        (self.root / "tools_repo" / "kandinsky-sr" / ".venv").mkdir(parents=True)
        (self.root / "tools_repo" / "comfyui-backend" / ".venv").mkdir(parents=True)
        legacy_uv = self.root / "tmp" / "uv-cache" / "wheels"
        legacy_uv.mkdir(parents=True)
        (legacy_uv / "sample.whl").write_bytes(b"wheel")
        legacy_hf = self.root / "hf_cache" / "hub"
        legacy_hf.mkdir(parents=True)
        (legacy_hf / "meta.json").write_text("{}", encoding="utf-8")

        moved = maintenance.migrate_legacy_caches(self.root)
        self.assertEqual(moved, 2)
        self.assertTrue((self.root / ".cache" / "uv" / "wheels" / "sample.whl").is_file())
        self.assertTrue((self.root / ".cache" / "huggingface" / "hub" / "meta.json").is_file())
        self.assertFalse((self.root / "tmp" / "uv-cache").exists())
        self.assertFalse((self.root / "hf_cache").exists())

        old_env = dict(os.environ)
        try:
            fake_local = self.root.parent / f"{self.root.name}_localappdata"
            (fake_local / "uv" / "cache").mkdir(parents=True, exist_ok=True)
            self.addCleanup(lambda: __import__("shutil").rmtree(fake_local, ignore_errors=True))
            os.environ["LOCALAPPDATA"] = str(fake_local)
            found = maintenance._external_cache_candidates(self.root)
            self.assertIn((fake_local / "uv" / "cache").resolve(), found)
            maintenance.fix_local_caches(self.root)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        cfg_name = "pip.ini" if os.name == "nt" else "pip.conf"
        expected_cache = str(self.root / ".cache" / "pip")
        for env_dir in (
            self.root / "python",
            self.root / "tools_repo" / "kandinsky-sr" / ".venv",
            self.root / "tools_repo" / "comfyui-backend" / ".venv",
        ):
            cfg = env_dir / cfg_name
            self.assertTrue(cfg.is_file(), f"missing {cfg}")
            self.assertIn(f"cache-dir = {expected_cache}", cfg.read_text(encoding="utf-8"))

    def test_inventory_exposes_disposable_cache_folder_only(self):
        entries = {it.key: it for it in inventory.items()}
        self.assertIn("cache", entries)
        self.assertEqual(entries["cache"].paths, [settings.CACHE_DIR])

    def test_all_launchers_and_updaters_pin_local_caches(self):
        required = (
            "UV_CACHE_DIR",
            "UV_PYTHON_INSTALL_DIR",
            "PIP_CACHE_DIR",
            "HF_HOME",
            "TORCH_HOME",
            "GRADIO_TEMP_DIR",
            "TEMP",
        )
        for name in ("run.bat", "run.sh", "update.bat", "update.sh", "install.bat", "install.sh"):
            text = (settings.ROOT / name).read_text(encoding="utf-8")
            for var in required:
                self.assertIn(var, text, f"{name} is missing {var}")


if __name__ == "__main__":
    unittest.main()
