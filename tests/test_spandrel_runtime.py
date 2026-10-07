"""Modern upscalers bootstrap their own backend, independently of Faces."""
import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier import registry, settings
from atelier.engine import tools
from scripts import setup_tools


class RuntimeInstallTests(unittest.TestCase):
    def test_ready_runtime_does_not_install_or_contact_package_servers(self):
        with patch.object(setup_tools.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 0, stdout="")) as probe, \
             patch.object(setup_tools, "install_spandrel") as install:
            setup_tools.ensure_spandrel()
        install.assert_not_called()
        self.assertEqual(probe.call_args.args[0][0], sys.executable)
        self.assertIn("from spandrel import", probe.call_args.args[0][2])

    def test_missing_or_broken_import_prepares_backend_before_inference(self):
        with patch.object(setup_tools.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 1, stdout="No module named 'spandrel'")), \
             patch.object(setup_tools, "install_spandrel") as install, \
             patch("builtins.print") as log:
            setup_tools.ensure_spandrel()
        install.assert_called_once_with()
        self.assertTrue(any("No module named 'spandrel'" in str(c) for c in log.call_args_list))

    def test_installer_preserves_existing_torch_build_and_verifies_real_imports(self):
        commands, constraints = [], []

        def run(command):
            commands.append(command)
            if "install" in command:
                constraints.append(Path(command[command.index("-c") + 1]).read_text())

        versions = {"torch": "2.4.1+cu121", "torchvision": "0.19.1+cu121"}
        with patch.object(setup_tools.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 0)) as probe, \
             patch.object(setup_tools.importlib.metadata, "version", side_effect=versions.__getitem__), \
             patch.object(setup_tools, "ensure_torch_cuda") as torch_setup, \
             patch.object(setup_tools, "sh", side_effect=run):
            setup_tools.install_spandrel()
        torch_setup.assert_not_called()
        self.assertEqual(probe.call_args.args[0][0], sys.executable)
        self.assertEqual(constraints, ["torch==2.4.1+cu121\ntorchvision==0.19.1+cu121\n"])
        self.assertTrue(all(c[0] == sys.executable for c in commands))
        self.assertIn(setup_tools.SPANDREL_PIN, commands[0])
        self.assertIn("torch.from_numpy", commands[-1][-1])

    def test_missing_torch_uses_the_existing_hardware_aware_installer(self):
        with patch.object(setup_tools.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 1)), \
             patch.object(setup_tools, "ensure_torch_cuda") as torch_setup, \
             patch.object(setup_tools.importlib.metadata, "version",
                          side_effect=importlib.metadata.PackageNotFoundError), \
             patch.object(setup_tools, "sh"):
            setup_tools.install_spandrel()
        torch_setup.assert_called_once_with()


class ModernUpscaleTests(unittest.TestCase):
    def test_modern_upscale_runs_without_face_models_on_the_selected_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "4xNomos8kHAT-L_otf.pth"
            weights.write_bytes(b"fixture")
            commands = []

            def run(command, log, message, **kwargs):
                commands.append(command)
                self.assertEqual(kwargs["gpu_index"], 1)
                self.assertEqual(command[0], sys.executable)
                Image.new("RGB", (32, 16)).save(command[command.index("--output") + 1])

            with patch.object(settings, "TMP_DIR", root / "tmp"), \
                 patch.object(settings, "OUTPUT_DIR", root / "outputs"), \
                 patch.object(settings, "ensure_dirs", side_effect=lambda: (
                     (root / "tmp").mkdir(exist_ok=True), (root / "outputs").mkdir(exist_ok=True))), \
                 patch.object(registry, "upscaler_path", return_value=weights), \
                 patch.object(tools, "face_is_installed", return_value=False), \
                 patch.object(tools, "_gen_gpu_index", return_value=1), \
                 patch.object(tools, "_run_tool", side_effect=run):
                result = tools.modern_upscale(Image.new("RGB", (8, 4)), weights.name)
            self.assertEqual(len(commands), 1)
            self.assertIn("run_spandrel.py", commands[0][1])
            with Image.open(result) as image:
                self.assertEqual(image.size, (32, 16))

    def test_cancelled_pre_enlargement_does_not_continue_with_lanczos(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.png"
            Image.new("RGB", (8, 8)).save(source)

            def cancelled(*args, **kwargs):
                tools._CANCELLED = True
                raise tools.ToolError("Cancelled by the user.")

            logs = []
            with patch.object(tools, "upscale_is_installed", return_value=True), \
                 patch.object(tools, "_CANCELLED", False), \
                 patch.object(tools, "modern_upscale", side_effect=cancelled), \
                 patch.object(tools, "_run_tool") as run:
                with self.assertRaisesRegex(tools.ToolError, "Cancelled"):
                    tools.ultimate_upscale(source, base_model=source,
                                           esrgan_model="4xNomos8kHAT-L_otf.pth", log=logs.append)
            run.assert_not_called()
            self.assertFalse(any("Lanczos fallback" in line for line in logs))
