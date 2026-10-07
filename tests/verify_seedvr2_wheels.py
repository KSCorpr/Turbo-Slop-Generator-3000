"""Windows CI: check wheels, TinyCC and actual Sage varlen PTX without a GPU.

Run only in a disposable Python 3.12 / Torch 2.7.1+cu126 environment.
"""
import argparse
import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atelier import settings, seedvr2_acceleration as acceleration
from scripts import setup_media


def verify_compiler():
    # Simulate an inherited Visual Studio shell with a broken/missing SDK.
    inherited = {**os.environ, "CC": r"C:\missing Visual Studio\cl.exe",
                 "WindowsSdkDir": r"C:\missing Windows SDK",
                 "CUDA_PATH": r"C:\missing CUDA Toolkit"}
    os.environ.update(acceleration.triton_env(inherited))
    from triton.runtime import build
    compiler = Path(os.environ["CC"])
    assert compiler.name == "tcc.exe" and Path(build.get_cc()) == compiler
    # Windows keeps an imported extension DLL locked until this child exits.
    with tempfile.TemporaryDirectory(prefix="seedvr2 compiler ", ignore_cleanup_errors=True) as temporary:
        stage = Path(temporary)
        source = stage / "turbo_cc_check.c"
        source.write_text('''#define Py_LIMITED_API 0x03090000
#include <Python.h>
#include <stdio.h>
static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "turbo_cc_check", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_turbo_cc_check(void) {
    PyObject *m = PyModule_Create(&def);
    if (m) PyModule_AddIntConstant(m, "answer", 42);
    return m;
}
''')
        binary = build._build("turbo_cc_check", str(source), str(stage), [], [], [])
        spec = importlib.util.spec_from_file_location("turbo_cc_check", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.answer == 42
        # Compile the actual helper from the wheel that failed for the user.
        # Loading it needs an NVIDIA driver; compilation needs no GPU or SDK.
        backend = Path(sysconfig.get_paths()["platlib"]) / "triton" / "backends" / "nvidia"
        cuda_source = stage / "cuda_utils.c"
        cuda_source.write_bytes((backend / "driver.c").read_bytes())
        cuda_binary = build._build("cuda_utils", str(cuda_source), str(stage),
                                   [str(backend / "lib" / "x64")],
                                   [str(backend / "include")], ["cuda"])
        assert Path(cuda_binary).is_file()
    print("TinyCC compiled and loaded a Python extension, and compiled Triton's real CUDA helper.")


def verify_varlen_ptx(profile):
    """Compile the checksum-verified Sage wheel's kernel; never load or launch it."""
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from sageattention.triton.attn_qk_int8_block_varlen import _attn_fwd

    cases = [(80, "bf16", True)]
    if profile == "turing":
        cases = [(75, "fp16", True), (75, "bf16", False), *cases]
    for head_dim in (64, 128):
        constants = {"H": 4, "num_kv_groups": 1, "HEAD_DIM": head_dim,
                     "BLOCK_M": 128, "BLOCK_N": 64, "STAGE": 1}
        for arch, dtype, succeeds in cases:
            # Triton 3.3 indexes constexpr arguments in the complete signature;
            # omitting them works in 3.2 but breaks its newer ASTSource API.
            signature = {name: "constexpr" if name in constants else "i32"
                         for name in _attn_fwd.arg_names}
            for name in ("Q", "K"):
                signature[name] = "*i8"
            signature["V"] = "*fp16"
            signature["Out"] = "*" + dtype
            for name in ("cu_seqlens_q", "cu_seqlens_k", "cu_seqlens_q_scale", "cu_seqlens_k_scale"):
                signature[name] = "*i32"
            for name in ("Q_scale", "K_scale"):
                signature[name] = "*fp32"
            try:
                compiled = triton.compile(
                    ASTSource(_attn_fwd, signature, constants), target=GPUTarget("cuda", arch, 32),
                    options={"num_warps": 4 if head_dim == 64 else 8,
                             "num_stages": 3 if head_dim == 64 else 4})
            except RuntimeError as exc:
                assert not succeeds, str(exc)
                diagnostic = str(exc)
                assert acceleration.kernel_compile_failed(diagnostic), diagnostic
                assert "bf16" in diagnostic and "sm_80" in diagnostic, diagnostic
                print(f"Sage varlen head={head_dim}: sm_{arch} BF16 rejected by ptxas as expected.")
            else:
                assert succeeds, "Turing BF16 unexpectedly compiled; review the compatibility guard."
                assert compiled.asm["cubin"]
                print(f"Sage varlen head={head_dim}: sm_{arch} {dtype} compiled without a GPU.")


def managed_checks():
    if platform.system() != "Windows":
        raise RuntimeError("These compilation checks require Windows.")
    with tempfile.TemporaryDirectory(prefix="seedvr2 managed ") as temporary:
        managed_dir = Path(temporary) / "Managed Python"
        setup_media.run(sys.executable, "-m", "uv", "python", "install", "3.12",
                        "--install-dir", managed_dir, "--no-bin", "--no-registry")
        # uv also creates a cpython-3.12 alias; select the versioned install.
        interpreters = list(managed_dir.glob("cpython-3.12.*/python.exe"))
        assert len(interpreters) == 1, interpreters
        for profile in ("turing", "ampere"):
            env_dir = Path(temporary) / (profile + " Python env")
            # python -m uv can prefer its parent interpreter despite
            # --managed-python. An explicit path and base-prefix assertion
            # ensure this tests uv's downloaded Python, as in the user's setup.
            setup_media.run(sys.executable, "-m", "uv", "venv", "--python",
                            interpreters[0], env_dir)
            py = env_dir / "Scripts" / "python.exe"
            setup_media.run(py, "-c",
                            "import sys; from pathlib import Path; "
                            "assert Path(sys.base_prefix).is_relative_to(Path(sys.argv[1])), sys.base_prefix; "
                            "print('Managed Python base:', sys.base_prefix)", managed_dir)
            setup_media.pip(py, "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126")
            # The verifier installs wheels from its own disposable interpreter.
            setup_media.pip(py, "einops", "packaging", "numpy", "uv>=0.8,<1")
            subprocess.run([str(py), "-u", __file__, "--profile", profile],
                           check=True, env=settings.child_env())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-python", action="store_true")
    parser.add_argument("--profile", choices=["turing", "ampere"], default="ampere")
    args = parser.parse_args(argv)
    if args.managed_python:
        managed_checks()
        return
    import torch
    if (platform.system() != "Windows" or sys.version_info[:2] != (3, 12)
            or torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.6"):
        raise RuntimeError("Use a disposable Windows Python 3.12 / Torch 2.7.1+cu126 environment.")
    before = torch.__version__
    with tempfile.TemporaryDirectory() as temporary:
        triton_wheel = acceleration.TRITON_TURING if args.profile == "turing" else acceleration.TRITON
        for wheel in (triton_wheel, acceleration.SAGE, acceleration.FLASH):
            archive = Path(temporary) / unquote(Path(urlsplit(wheel.url).path).name)
            setup_media.download(wheel.url, archive)
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != wheel.sha256:
                    raise RuntimeError(wheel.name + " checksum mismatch.")
            setup_media.pip(sys.executable, "--no-deps", "--only-binary", ":all:", archive)
    verify_compiler()  # Configure before importing any attention modules.
    import triton
    import flash_attn_2_cuda  # noqa: F401
    from flash_attn import flash_attn_varlen_func
    from sageattention import sageattn_varlen
    assert callable(flash_attn_varlen_func) and callable(sageattn_varlen)
    assert torch.__version__ == before
    assert triton.__version__.split(".")[:2] == ["3", "2" if args.profile == "turing" else "3"]
    verify_varlen_ptx(args.profile)
    print(f"Windows {args.profile} checks passed: wheels, DLL imports, TinyCC, varlen PTX, unchanged Torch.")


if __name__ == "__main__":
    main()
