"""Optional wheels must preserve Torch and verify kernels on the selected GPU."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from atelier import addons, settings, seedvr2_acceleration as acceleration
from atelier.engine import seedvr2
from atelier.engine.local_jobs import LocalJob
from atelier.ui import media_tabs
from scripts import setup_media


def hardware(capability=(8, 6), **overrides):
    return {"system": "Windows", "machine": "AMD64", "python": [3, 12],
            "torch": "2.7.1+cu126", "cuda": "12.6",
            "compute_dtype": "bfloat16",
            "gpu": {"name": "Selected GPU", "capability": list(capability)}, **overrides}


class CompatibilityTests(unittest.TestCase):
    def test_bundled_compiler_and_cuda_override_broken_windows_tools_in_child_only(self):
        with tempfile.TemporaryDirectory(prefix="SeedVR2 env ") as temporary:
            packages = Path(temporary) / "Lib" / "site-packages"
            compiler = packages / "triton" / "runtime" / "tcc" / "tcc.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            cuda = packages / "triton" / "backends" / "nvidia"
            for part in ("bin/ptxas.exe", "include/cuda.h", "lib/x64/cuda.lib"):
                path = cuda / part
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"bundled")
            original = {"CC": r"C:\missing SDK\cl.exe", "CUDA_PATH": "CUDA 13.1",
                        "CUDA_VISIBLE_DEVICES": "1", "PATH": "original path"}
            with patch.object(acceleration.platform, "system", return_value="Windows"):
                child = acceleration.triton_env(original, packages)
            self.assertEqual(child["CC"], str(compiler))
            self.assertEqual(child["CUDA_PATH"], str(cuda))
            self.assertEqual(child["CUDA_HOME"], str(cuda))
            self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(child["PATH"], "original path")
            self.assertEqual(original["CC"], r"C:\missing SDK\cl.exe")
            self.assertEqual(original["CUDA_PATH"], "CUDA 13.1")

    def test_missing_bundled_tools_and_linux_preserve_the_existing_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = {"CC": "custom compiler", "CUDA_PATH": "custom CUDA"}
            with patch.object(acceleration.platform, "system", return_value="Windows"):
                self.assertEqual(acceleration.triton_env(env, Path(temporary)), env)
            with patch.object(acceleration.platform, "system", return_value="Linux"):
                self.assertEqual(acceleration.triton_env(env, Path(temporary)), env)

    def test_turing_bf16_keeps_sdpa_and_fp16_requires_an_experimental_probe(self):
        self.assertEqual(acceleration.wheel_plan(hardware())[0],
                         [acceleration.TRITON, acceleration.SAGE, acceleration.FLASH])
        wheels, message = acceleration.wheel_plan(hardware((7, 5)))
        self.assertEqual(wheels, [])
        self.assertIn("BF16", message)
        self.assertIn("SDPA", message)
        wheels, message = acceleration.wheel_plan(hardware((7, 5), compute_dtype="float16"))
        self.assertEqual(wheels, [acceleration.TRITON_TURING, acceleration.SAGE])
        self.assertIn("experimental", message)

    def test_compute_precision_matches_seedvr2_cublas_probe_instead_of_gpu_generation(self):
        torch = SimpleNamespace(bfloat16="BF16", randn=Mock(return_value="matrix"), matmul=Mock())
        self.assertEqual(acceleration._compute_dtype(torch), "bfloat16")
        torch.randn.assert_called_once_with(8, 8, dtype="BF16", device="cuda:0")
        torch.matmul.assert_called_once_with("matrix", "matrix")
        torch.matmul.side_effect = RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        self.assertEqual(acceleration._compute_dtype(torch), "float16")
        torch.matmul.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            acceleration._compute_dtype(torch)

    def test_turing_bf16_rejects_an_old_positive_report_for_auto_and_explicit_attention(self):
        report = {**hardware((7, 5)), "backends": {
            mode: {"ok": True} for mode in acceleration.BACKENDS}}
        for dtype in ("bfloat16", None):
            if dtype is None:
                report.pop("compute_dtype")  # Old reports did not record precision.
            for mode in ("auto", *acceleration.BACKENDS):
                with self.subTest(dtype=dtype, mode=mode):
                    self.assertEqual(acceleration.choose_mode(mode, report, lambda line: None), "sdpa")
        with self.assertRaisesRegex(RuntimeError, "BF16.*sm_75"):
            acceleration._test_kernel("sageattn_2", (7, 5), "bfloat16")

    def test_probe_passes_the_actual_precision_to_the_varlen_test(self):
        for dtype in ("bfloat16", "float16"):
            with self.subTest(dtype=dtype), \
                 patch.object(acceleration, "hardware_info", return_value=hardware(compute_dtype=dtype)), \
                 patch.object(acceleration, "_test_kernel") as kernel:
                acceleration.probe(("sageattn_2",))
                kernel.assert_called_once_with("sageattn_2", (8, 6), dtype)

    def test_unsupported_gpu_python_os_or_torch_is_rejected_before_installation(self):
        cases = [hardware((6, 1)), hardware((7, 0)), hardware((12, 0)),
                 hardware(python=[3, 11]), hardware(system="Linux"), hardware(machine="ARM64"),
                 hardware(torch="2.8.0+cu128"), hardware(cuda="12.8"),
                 hardware(error="No CUDA GPU available")]
        for info in cases:
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                acceleration.wheel_plan(info)

    def test_import_or_installation_alone_never_marks_a_backend_usable(self):
        with patch.object(acceleration, "hardware_info", side_effect=hardware), \
             patch.object(acceleration.importlib.metadata, "version", return_value="installed"), \
             patch.object(acceleration, "_test_kernel", side_effect=[RuntimeError("bad kernel"), None]):
            report = acceleration.probe()
        self.assertFalse(report["backends"]["sageattn_2"]["ok"])
        self.assertIn("bad kernel", report["backends"]["sageattn_2"]["reason"])
        self.assertEqual(acceleration.choose_mode("auto", report, lambda line: None), "flash_attn_2")
        self.assertEqual(acceleration.choose_mode("sageattn_2", report, lambda line: None), "sdpa")

    def test_probe_cli_without_torch_reports_failure_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contrôle…é.json"
            subprocess.run([sys.executable, "-S", acceleration.__file__, "--output", str(output)],
                           check=True, timeout=30, env=settings.child_env())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("torch", report["error"])
            self.assertTrue(all(not result["ok"] for result in report["backends"].values()))


class InstallerTests(unittest.TestCase):
    def test_turing_bf16_does_not_download_incompatible_wheels_or_change_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / "installed.json"
            installed.write_text('{"commit":"unchanged"}')

            def run(*command):
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps(hardware((7, 5))))

            with patch.object(addons, "root", return_value=root), \
                 patch.object(addons, "ready", return_value=True), \
                 patch.object(setup_media, "run", side_effect=run), \
                 patch.object(setup_media, "download") as download, \
                 patch.object(setup_media, "pip") as pip, patch("builtins.print") as log:
                setup_media.install_seed_optimizations()
            download.assert_not_called()
            pip.assert_not_called()
            self.assertEqual(installed.read_text(), '{"commit":"unchanged"}')
            self.assertIn("SeedVR2 remains ready with SDPA", "\n".join(c.args[0] for c in log.call_args_list))

    def exercise_install(self, *, bad_checksum=False, broken=False, info=None):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            installed = root / "installed.json"
            installed.write_text('{"commit":"unchanged"}')
            payload = b"fixture wheel"
            digest = hashlib.sha256(payload).hexdigest()
            wheels = [replace(wheel, sha256=digest) for wheel in
                      (acceleration.TRITON, acceleration.SAGE, acceleration.FLASH)]
            stack.enter_context(patch.object(addons, "root", return_value=root))
            stack.enter_context(patch.object(addons, "python", return_value=root / "seed-python.exe"))
            stack.enter_context(patch.object(addons, "ready", return_value=True))
            # Preserve real environment/GPU validation, substituting only wheel hashes.
            stack.enter_context(patch.object(acceleration, "TRITON", wheels[0]))
            stack.enter_context(patch.object(acceleration, "SAGE", wheels[1]))
            stack.enter_context(patch.object(acceleration, "FLASH", wheels[2]))

            def run(*args):
                report_path = Path(args[args.index("--output") + 1])
                report = info if info is not None else hardware()
                if args[-1] == "all":
                    report = {**report, "backends": {mode: {"ok": not broken, "reason": "kernel failed"}
                                                    for mode in acceleration.BACKENDS}}
                report_path.write_text(json.dumps(report), encoding="utf-8")

            def download(url, destination):
                destination.write_bytes(b"corrupt" if bad_checksum and "triton_windows" in url else payload)

            stack.enter_context(patch.object(setup_media, "run", side_effect=run))
            downloads = stack.enter_context(patch.object(setup_media, "download", side_effect=download))
            installer = stack.enter_context(patch.object(setup_media, "pip"))
            if info is not None:
                with self.assertRaises(RuntimeError):
                    setup_media.install_seed_optimizations()
            else:
                with patch("builtins.print") as messages:
                    setup_media.install_seed_optimizations()
                if broken:
                    printed = "\n".join(str(call.args[0]) for call in messages.call_args_list)
                    self.assertIn("SeedVR2 remains ready with SDPA", printed)
                    self.assertNotIn("optimizations ready", printed)
            self.assertEqual(installed.read_text(), '{"commit":"unchanged"}')
            self.assertTrue(all(call.args[0] == root / "seed-python.exe" for call in installer.call_args_list))
            for call in installer.call_args_list:
                self.assertEqual(call.args[1:4], ("--no-deps", "--only-binary", ":all:"))
                self.assertNotIn("%2B", call.args[-1].name)
            if info is not None:
                downloads.assert_not_called()
                installer.assert_not_called()
                self.assertFalse((root / acceleration.MARKER).exists())
            else:
                report = json.loads((root / acceleration.MARKER).read_text())
                self.assertEqual(len(installer.call_args_list), 2 if bad_checksum else 3)
                self.assertEqual(bool(report["installation_failures"]), bad_checksum)
                if bad_checksum:
                    self.assertNotIn("triton", " ".join(c.args[-1].name for c in installer.call_args_list))
                if broken:
                    self.assertTrue(all(not result["ok"] for result in report["backends"].values()))

    def test_verified_wheels_use_only_seedvr2_python_without_resolving_torch(self):
        self.exercise_install()

    def test_corrupt_wheel_is_not_installed_and_failure_is_recorded(self):
        self.exercise_install(bad_checksum=True)

    def test_failed_gpu_check_keeps_base_install_and_reports_no_usable_optimization(self):
        self.exercise_install(broken=True)

    def test_mismatched_torch_never_downloads_or_installs_packages(self):
        self.exercise_install(info=hardware(torch="2.8.0+cu128"))

    def test_cli_dispatches_optional_install_without_reinstalling_models(self):
        with patch.object(setup_media, "install_seed") as base, \
             patch.object(setup_media, "install_seed_optimizations") as optional:
            setup_media.main(["seedvr2", "--optimizations"])
            optional.assert_called_once_with()
            base.assert_not_called()
            setup_media.main(["seedvr2"])
            base.assert_called_once_with()

    def test_install_button_uses_the_active_gpu_mapping(self):
        job = LocalJob()
        prefs = {"system_mode": "auto", "auto_gpu_index": 1, "gpu_index": 0}
        with patch.object(settings, "load_prefs", return_value=prefs), \
             patch.object(job, "run") as run:
            media_tabs.install_action("seedvr2", job, print, optimizations=True)
        self.assertIn("--optimizations", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")

    def test_install_button_passes_the_bundled_compiler_to_setup_and_its_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = root / ".venv" / "Lib" / "site-packages"
            compiler = packages / "triton" / "runtime" / "tcc" / "tcc.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            job = LocalJob()
            with patch.object(addons, "root", return_value=root), \
                 patch.object(acceleration.platform, "system", return_value="Windows"), \
                 patch.object(job, "run") as run:
                media_tabs.install_action("seedvr2", job, print, optimizations=True)
            self.assertEqual(run.call_args.kwargs["env"]["CC"], str(compiler))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source…é.png"
        Image.new("RGB", (320, 240), "red").save(self.source)
        self.output = self.root / "output.png"
        (self.root / acceleration.MARKER).write_text("{}")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.job = LocalJob()
        self.stack.enter_context(patch.object(seedvr2, "JOB", self.job))
        self.stack.enter_context(patch.object(addons, "root", return_value=self.root))
        self.stack.enter_context(patch.object(addons, "ready", return_value=True))
        self.stack.enter_context(patch.object(settings, "TMP_DIR", self.root))
        self.stack.enter_context(patch.object(seedvr2.sdcpp, "unique_output", return_value=self.output))
        self.stack.enter_context(patch.object(settings, "load_prefs", return_value={"gpu_index": 1}))

    def test_probe_and_inference_use_the_same_gpu_and_verified_backend_argument(self):
        compiler = self.root / ".venv" / "Lib" / "site-packages" / "triton" / "runtime" / "tcc" / "tcc.exe"
        compiler.parent.mkdir(parents=True)
        compiler.write_bytes(b"compiler")
        for healthy in (True, False):
            commands = []

            def run(command, log, **kwargs):
                commands.append(command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
                self.assertEqual(kwargs["env"]["CC"], str(compiler))
                output = Path(command[command.index("--output") + 1])
                if output.suffix == ".json":
                    output.write_text(json.dumps({**hardware(), "backends": {
                        "sageattn_2": {"ok": healthy}, "flash_attn_2": {"ok": False}}}))
                else:
                    self.assertEqual(command[command.index("--attention_mode") + 1],
                                     "sageattn_2" if healthy else "sdpa")
                    self.assertEqual(kwargs["env"]["SEEDVR2_OPTIMIZATIONS_LOGGED"], "1")
                    Image.open(self.source).save(output)

            with self.subTest(healthy=healthy), patch.object(self.job, "run", side_effect=run), \
                 patch.object(acceleration.platform, "system", return_value="Windows"):
                seedvr2.restore(self.source, log=lambda line: None)
            self.assertEqual(len(commands), 2)

    def test_sdpa_and_zero_details_do_not_load_optional_gpu_kernels(self):
        with patch.object(self.job, "run") as run:
            seedvr2.restore(self.source, scale=2, detail_strength=0, log=lambda line: None)
        run.assert_not_called()
        with patch.object(self.job, "run") as run:
            self.assertEqual(seedvr2.select_attention("sdpa", self.root, {}, lambda line: None), "sdpa")
        run.assert_not_called()

    def test_probe_crash_falls_back_but_cancellation_does_not_start_inference(self):
        with patch.object(self.job, "run", side_effect=RuntimeError("probe crashed")):
            self.assertEqual(seedvr2.select_attention("auto", self.root, {}, lambda line: None), "sdpa")
        with self.job.session():
            def cancel(*args, **kwargs):
                self.job.cancel()
                self.job.check()
            with patch.object(self.job, "run", side_effect=cancel), self.assertRaisesRegex(RuntimeError, "cancelled"):
                seedvr2.select_attention("auto", self.root, {}, lambda line: None)

    def test_generation_error_is_not_retried_or_hidden_as_an_attention_failure(self):
        with patch.object(seedvr2, "select_attention", return_value="sageattn_2"), \
             patch.object(self.job, "run", side_effect=RuntimeError("CUDA out of memory")) as run:
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                seedvr2.restore(self.source, log=lambda line: None)
        run.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_ptx_failure_before_the_shutdown_footer_retries_once_with_the_same_settings(self):
        commands, logs = [], []

        def run(command, log, **kwargs):
            commands.append((list(command), kwargs))
            output = Path(command[command.index("--output") + 1])
            if len(commands) == 1:
                output.write_bytes(b"incomplete output")
                log("ptxas fatal : Ptx assembly aborted due to errors")
                for _ in range(20):
                    log("upstream shutdown footer")
                raise RuntimeError("Python exited with code 1. upstream shutdown footer")
            self.assertFalse(output.exists())
            Image.open(self.source).save(output)

        with patch.object(seedvr2, "select_attention", return_value="sageattn_2"), \
             patch.object(self.job, "run", side_effect=run):
            output = seedvr2.restore(self.source, scale=2, seed=1234, log=logs.append)
        self.assertEqual(output, self.output)
        self.assertEqual(len(commands), 2)
        expected = list(commands[0][0])
        expected[expected.index("--attention_mode") + 1] = "sdpa"
        self.assertEqual(commands[1][0], expected)
        self.assertEqual(commands[0][1], commands[1][1])
        self.assertTrue(any("Retrying once" in line for line in logs))

    def test_ptx_error_in_the_exception_also_retries_but_sdpa_failure_is_propagated(self):
        for backend in acceleration.BACKENDS:
            with self.subTest(backend=backend), \
                 patch.object(seedvr2, "select_attention", return_value=backend), \
                 patch.object(self.job, "run", side_effect=[
                     RuntimeError("ptxas fatal : Ptx assembly aborted due to errors"),
                     RuntimeError("SDPA failure")]) as run:
                with self.assertRaisesRegex(RuntimeError, "SDPA failure"):
                    seedvr2.restore(self.source, log=lambda line: None)
            self.assertEqual(run.call_count, 2)
            self.assertFalse(self.output.exists())

    def test_ptx_failure_in_sdpa_is_not_retried(self):
        with patch.object(seedvr2, "select_attention", return_value="sdpa"), \
             patch.object(self.job, "run", side_effect=RuntimeError("ptxas fatal : error")) as run:
            with self.assertRaisesRegex(RuntimeError, "ptxas fatal"):
                seedvr2.restore(self.source, log=lambda line: None)
        run.assert_called_once()

    def test_stop_after_an_optional_compiler_failure_never_launches_a_retry(self):
        def run(command, log, **kwargs):
            log("ptxas fatal : Ptx assembly aborted due to errors")
            self.job.cancel()
            raise RuntimeError("ptxas fatal : Ptx assembly aborted due to errors")

        with patch.object(seedvr2, "select_attention", return_value="sageattn_2"), \
             patch.object(self.job, "run", side_effect=run) as runner:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                seedvr2.restore(self.source, log=lambda line: None)
        runner.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_real_process_preserves_the_compiler_error_before_a_long_shutdown_footer(self):
        directory = self.root / "source"
        directory.mkdir()
        script = directory / "fixture_backend.py"
        script.write_text(
            "import pathlib, sys\n"
            "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "attention = sys.argv[sys.argv.index('--attention_mode') + 1]\n"
            "if attention != 'sdpa':\n"
            "    output.write_bytes(b'incomplete video')\n"
            "    print('ptxas fatal : Ptx assembly aborted due to errors', flush=True)\n"
            "    for i in range(20): print('shutdown footer', flush=True)\n"
            "    sys.exit(1)\n"
            "assert not output.exists(), 'partial output was not removed'\n"
            "output.write_bytes(b'complete video')\n", encoding="utf-8")
        logs = []
        with self.job.session():
            seedvr2._run_inference(
                [sys.executable, script, "--output", self.output, "--attention_mode", "sageattn_2"],
                self.output, settings.child_env(), logs.append)
        self.assertEqual(self.output.read_bytes(), b"complete video")
        self.assertEqual(sum(line.startswith("$") for line in logs), 2)
        self.assertIn("ptxas fatal : Ptx assembly aborted due to errors", logs)


if __name__ == "__main__":
    unittest.main()
