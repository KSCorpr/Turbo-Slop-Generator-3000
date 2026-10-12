"""ConvRot integration: native loader, isolation and cancellation without GPU weights."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from atelier import addons, settings
from atelier.engine import seedvr2
from atelier.engine.local_jobs import LocalJob
from scripts import setup_media
from scripts.tools import run_seedvr2_convrot as wrapper


class ConvRotCommandTests(unittest.TestCase):
    def test_all_quantized_variants_use_standard_loader_and_shared_vae_cache(self):
        for model, filename in addons.SEED_CONVROT_MODELS.items():
            with self.subTest(model=model):
                command = seedvr2.build_command(Path("entrée 日本.png"), Path("out.png"),
                                                model, 1080, is_video=False)
                self.assertEqual(command[0], str(addons.python("seedvr2-convrot")))
                self.assertIn("--addon-root", command)
                self.assertEqual(command[command.index("--dit_model") + 1], filename)
                self.assertEqual(command[command.index("--model_dir") + 1],
                                 str(addons.root("seedvr2") / "models"))
                self.assertNotIn("--swap_io_components", command)
                self.assertFalse(any("distorch" in arg.lower() for arg in command))
                if model.endswith("w4a8"):
                    self.assertEqual(command[command.index("--blocks_to_swap") + 1], "0")
                self.assertIn("--vae_decode_tiled", command)

    def test_classic_keeps_its_existing_runtime_and_swap_profile(self):
        command = seedvr2.build_command(Path("in.png"), Path("out.png"), "3b", 1080, is_video=False)
        self.assertEqual(command[0], str(addons.python("seedvr2")))
        self.assertIn("--swap_io_components", command)
        self.assertEqual(command[command.index("--blocks_to_swap") + 1], "32")

    def test_cancelling_native_gpu_probe_never_starts_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (16, 16)).save(source)
            job = LocalJob()

            def stop(command, log, **kwargs):
                self.assertIn("--probe", [str(arg) for arg in command])
                job.cancel()
                job.check()

            with patch.object(settings, "TMP_DIR", root), \
                 patch.object(addons, "ready", return_value=True), \
                 patch.object(seedvr2, "JOB", job), \
                 patch.object(job, "run", side_effect=stop) as run, \
                 patch.object(seedvr2, "_run_inference") as inference, \
                 patch.object(seedvr2.sdcpp, "unique_output", return_value=root / "out.png"):
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    seedvr2.restore(source, model="3b-w4a8", log=lambda line: None)
                run.assert_called_once()
                inference.assert_not_called()
            self.assertTrue(source.exists())
            self.assertFalse((root / "out.png").exists())


class ConvRotInstallerTests(unittest.TestCase):
    def test_failed_gpu_probe_never_marks_ready_or_alters_classic_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            classic = project / "seedvr2"
            classic.mkdir()
            sentinel = classic / "installed.json"
            sentinel.write_text("classic stays ready")
            target = project / "seedvr2-convrot"

            def install_source(repo, commit, destination):
                destination.mkdir()
                if destination.name == "source":
                    (destination / "inference_cli.py").write_text("script_dir + ':'")

            def run(*command):
                if "--probe" in command:
                    raise RuntimeError("native kernel failed")

            with patch.object(addons, "root", side_effect=lambda name: project / name), \
                 patch.object(setup_media, "environment", return_value=project / "python"), \
                 patch.object(setup_media, "pip") as pip, \
                 patch.object(setup_media, "_install_source", side_effect=install_source), \
                 patch.object(setup_media, "run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "native kernel failed"):
                    setup_media.install_seed_convrot()
            self.assertFalse((target / "installed.json").exists())
            self.assertEqual(sentinel.read_text(), "classic stays ready")
            self.assertEqual(pip.call_args_list[0].args[1:3],
                             ("torch==2.13.0+cu130", "torchvision==0.28.0+cu130"))
            self.assertIn("https://download.pytorch.org/whl/cu130", pip.call_args_list[0].args)
            self.assertIn("os.pathsep", (target / "source" / "inference_cli.py").read_text())

    def test_old_driver_stops_before_downloading_torch_or_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "seedvr2-convrot"
            target.mkdir()
            (target / "installed.json").write_text('{"torch_cuda":"12.6"}')
            with patch.object(addons, "root", return_value=target), \
                 patch.object(setup_media, "environment", return_value=target / "python"), \
                 patch.object(setup_media, "run", side_effect=RuntimeError("driver CUDA 12.6")) as run, \
                 patch.object(setup_media, "pip") as pip, \
                 patch.object(setup_media, "_install_source") as sources:
                with self.assertRaisesRegex(RuntimeError, "driver CUDA 12.6"):
                    setup_media.install_seed_convrot()
            self.assertIn("--check-driver", run.call_args.args)
            pip.assert_not_called()
            sources.assert_not_called()
            self.assertFalse((target / "installed.json").exists())

    def test_repair_pins_cuda_build_in_both_install_and_dependency_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "seedvr2-convrot"
            target.mkdir()
            (target / "installed.json").write_text('{"torch_cuda":"12.6"}')
            isolated_python = target / ".venv" / "python"

            def install_source(repo, commit, destination):
                destination.mkdir()
                if destination.name == "source":
                    (destination / "inference_cli.py").write_text("script_dir + ':'")

            with patch.object(addons, "root", return_value=target), \
                 patch.object(setup_media, "environment", return_value=isolated_python), \
                 patch.object(setup_media, "pip") as pip, \
                 patch.object(setup_media, "_install_source", side_effect=install_source), \
                 patch.object(setup_media, "run") as run:
                setup_media.install_seed_convrot()
            self.assertEqual(pip.call_args_list[0].args[:3],
                             (isolated_python, "torch==2.13.0+cu130", "torchvision==0.28.0+cu130"))
            constraints = target / "torch-constraints.txt"
            self.assertIn("torch==2.13.0+cu130", constraints.read_text())
            self.assertIn("torchvision==0.28.0+cu130", constraints.read_text())
            self.assertIn(constraints, pip.call_args_list[1].args)
            self.assertTrue(all(c.args[0] == isolated_python for c in pip.call_args_list))
            self.assertIn("--check-driver", run.call_args_list[0].args)
            self.assertIn("--probe", run.call_args_list[-2].args)
            self.assertEqual(json.loads((target / "installed.json").read_text())["torch_cuda"], "13.0")


class ConvRotWrapperTests(unittest.TestCase):
    def test_cuda126_is_rejected_before_loading_comfy_or_quantized_ops(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda i: (8, 6)),
            version=SimpleNamespace(cuda="12.6"), __version__="2.13.0+cu126")
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "Install / repair SeedVR2"):
                wrapper.probe("w4a8")

    def test_runtime_accepts_the_matching_cuda13_build(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda i: (8, 6)),
            version=SimpleNamespace(cuda="13.0"), __version__="2.13.0+cu130")
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertIs(wrapper.require_runtime(), torch)

    def test_driver_version_is_checked_without_installed_sources_or_torch(self):
        with patch.object(wrapper, "driver_cuda_version", return_value=12060):
            with self.assertRaisesRegex(RuntimeError, "driver 580"):
                wrapper.main(["--addon-root", "not-installed", "--check-driver"])
        with patch.object(wrapper, "driver_cuda_version", return_value=13000), \
             patch.object(wrapper, "configure") as configure, \
             patch.dict(sys.modules, {"torch": None}), patch("builtins.print"):
            wrapper.main(["--addon-root", "not-installed", "--check-driver"])
            configure.assert_not_called()

    def test_driver_api_is_read_directly(self):
        def version(pointer):
            pointer._obj.value = 13040
            return 0

        driver = SimpleNamespace(cuDriverGetVersion=version)
        with patch.object(wrapper.sys, "platform", "linux"), \
             patch.object(wrapper.ctypes, "CDLL", return_value=driver):
            self.assertEqual(wrapper.driver_cuda_version(), 13040)
        with patch.object(wrapper.sys, "platform", "win32"), \
             patch.object(wrapper.ctypes, "WinDLL", return_value=driver, create=True) as load:
            self.assertEqual(wrapper.driver_cuda_version(), 13040)
            load.assert_called_once_with("nvcuda.dll")
        with patch.object(wrapper.sys, "platform", "linux"), \
             patch.object(wrapper.ctypes, "CDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA CUDA driver"):
                wrapper.driver_cuda_version()

    def test_disabled_native_backend_is_not_reported_as_a_success(self):
        kitchen = SimpleNamespace(
            registry=SimpleNamespace(is_available=lambda name: False),
            list_backends=lambda: {"cuda": {"unavailable_reason": "missing CUDA DLL"}})
        with patch.dict(sys.modules, {"comfy": SimpleNamespace(),
                                      "comfy.quant_ops": SimpleNamespace(),
                                      "comfy_kitchen": kitchen}):
            with self.assertRaisesRegex(RuntimeError, "missing CUDA DLL"):
                wrapper.require_native_backend()

    def test_cuda_and_old_gpu_fail_before_loading_quantized_dependencies(self):
        for available, capability, message in ((False, (8, 6), "CUDA"), (True, (6, 1), "RTX")):
            cuda = SimpleNamespace(is_available=lambda: available, get_device_capability=lambda i: capability)
            with self.subTest(available=available), patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
                with self.assertRaisesRegex(RuntimeError, message):
                    wrapper.probe("w4a8")

    def test_runtime_paths_and_upstream_arguments_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            (root / "source" / "inference_cli.py").write_text("")
            (root / "comfy" / "comfy").mkdir(parents=True)
            (root / "comfy" / "comfy" / "ops.py").write_text("")
            with patch.object(sys, "path", list(sys.path)), patch.object(sys, "argv", ["wrapper.py"]), \
                 patch.dict(os.environ, {"PYTHONPATH": "existing"}), \
                 patch.object(wrapper, "require_runtime") as runtime, \
                 patch.object(wrapper.runpy, "run_path") as run:
                wrapper.main(["--addon-root", str(root), "in file.png", "--output", "out file.png"])
                self.assertEqual(sys.argv[1:], ["in file.png", "--output", "out file.png"])
                self.assertEqual(sys.path[:2], [str(root / "source"), str(root / "comfy")])
                self.assertEqual(os.environ["PYTHONPATH"],
                                 os.pathsep.join([str(root / "source"), str(root / "comfy"), "existing"]))
                run.assert_called_once_with(str(root / "source" / "inference_cli.py"), run_name="__main__")
                runtime.assert_called_once()


if __name__ == "__main__":
    unittest.main()
