"""Optional SeedVR2 attention: pinned Windows wheels and isolated GPU checks.

The probe runs with SeedVR2's Python, never importing Torch into the studio.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

MODES = ("auto", "sdpa", "sageattn_2", "flash_attn_2")
BACKENDS = ("sageattn_2", "flash_attn_2")
MARKER = "optimizations.json"


@dataclass(frozen=True)
class Wheel:
    name: str
    url: str
    sha256: str


SAGE = Wheel(
    "SageAttention 2.2.0",
    "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/"
    "sageattention-2.2.0%2Bcu128torch2.7.1-cp312-cp312-win_amd64.whl",
    "7cee3f8cdcd48bdf0ec59603a24be8723e45fc2f6329ebe06beecabd5184b1ba")
# Upstream FlashAttention wheels match the Torch major/minor ABI (2.7).
# This Windows build uses 2.7.0; the GPU probe must pass on our 2.7.1 runtime.
FLASH = Wheel(
    "FlashAttention 2.8.3",
    "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
    "916ae4d818d2b5a02b3b25e8431251b88a01ce98d08315495dd78242d81f7182")
TRITON = Wheel(
    "Triton Windows 3.3.1.post21",
    "https://files.pythonhosted.org/packages/e2/5e/"
    "2730d1a18ac336eaa3bea1e9f2fbe718d3868b71e794f46d572ce3fcd845/"
    "triton_windows-3.3.1.post21-cp312-cp312-win_amd64.whl",
    "c75e473e2d7edb41037b50477bda5e9fd01939ce9ac25a143c9c149bc9d8adf5")
TRITON_TURING = Wheel(
    "Triton Windows 3.2.0.post21",
    "https://files.pythonhosted.org/packages/8d/96/"
    "7a9dd9d1e891b41e3469bff13ab7d4a2eb4ff4d891097a5a0e299fc7946b/"
    "triton_windows-3.2.0.post21-cp312-cp312-win_amd64.whl",
    "f0294d4f85e98e735942d4db0771c4f05d7b13a481c99b520f1d9701f3e83966")


def wheel_plan(info: dict) -> tuple[list[Wheel], str]:
    """Reject mismatched environments before modifying any packages."""
    if info.get("error"):
        raise RuntimeError("Cannot inspect SeedVR2: " + info["error"])
    if (info.get("system") != "Windows"
            or info.get("machine", "").lower() not in ("amd64", "x86_64")
            or info.get("python") != [3, 12]):
        raise RuntimeError("Automatic optimizations require Windows x64 and SeedVR2 Python 3.12. "
                           "SDPA remains available.")
    if info.get("torch", "").split("+")[0] != "2.7.1" or info.get("cuda") != "12.6":
        raise RuntimeError("These wheels require SeedVR2 Torch 2.7.1 / CUDA 12.6. "
                           "Use Install / repair SeedVR2 first; no Torch version was changed.")
    capability = tuple(info.get("gpu", {}).get("capability", (0, 0)))
    if capability == (7, 5):
        return [TRITON_TURING, SAGE], (
            "Turing: trying SageAttention with Triton 3.2. FlashAttention 2 requires Ampere or newer. "
            "Triton 3.2 with Torch 2.7 is experimental; activation requires a successful GPU test.")
    if capability in ((8, 0), (8, 6), (8, 9), (9, 0)):
        return [TRITON, SAGE, FLASH], "Installing prebuilt Windows attention wheels in SeedVR2 only."
    raise RuntimeError("No supported optimization wheels for this selected GPU "
                       "(Pascal / GTX 10xx is unsupported). SDPA remains available.")


def hardware_info() -> dict:
    info = {"system": platform.system(), "machine": platform.machine(),
            "python": list(sys.version_info[:2])}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            info["gpu"] = {"name": torch.cuda.get_device_name(0),
                           "capability": list(torch.cuda.get_device_capability(0))}
        else:
            info["error"] = "No CUDA GPU available in the selected environment."
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _test_kernel(mode: str, capability: tuple) -> None:
    """Exercise the varlen API used by SeedVR2, including both head sizes."""
    import torch
    from torch.nn import functional as F

    if mode == "sageattn_2":
        if capability != (7, 5) and capability not in ((8, 0), (8, 6), (8, 9), (9, 0)):
            raise RuntimeError("SageAttention needs a supported Turing or newer GPU.")
        import triton
        major_minor = tuple(int(v) for v in triton.__version__.split(".")[:2])
        if capability == (7, 5) and major_minor > (3, 2):
            raise RuntimeError("Turing requires Triton <= 3.2; reinstall optimizations on this GPU.")
        from sageattention import sageattn_varlen as kernel
    else:
        if capability not in ((8, 0), (8, 6), (8, 9), (9, 0)):
            raise RuntimeError("FlashAttention 2 requires an Ampere, Ada or Hopper GPU.")
        import flash_attn_2_cuda  # noqa: F401
        from flash_attn import flash_attn_varlen_func as kernel

    torch.cuda.set_device(0)
    # BF16 is the upstream DiT default on Ampere+, FP16 on Turing.
    dtype = torch.bfloat16 if capability >= (8, 0) else torch.float16
    torch.manual_seed(42)
    offsets = (0, 128, 224)
    lengths = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        for head_dim in (64, 128):
            q, k, v = [torch.randn((224, 4, head_dim), device="cuda", dtype=dtype)
                       for _ in range(3)]
            if mode == "flash_attn_2":
                result = kernel(q, k, v, lengths, lengths, 128, 128, causal=False)
            else:
                result = kernel(q, k, v, lengths, lengths, 128, 128, False, head_dim ** -0.5)
            torch.cuda.synchronize()
            if result.shape != q.shape or not torch.isfinite(result).all().item():
                raise RuntimeError("Attention kernel produced invalid output.")
            for start, end in zip(offsets, offsets[1:]):
                reference = F.scaled_dot_product_attention(
                    *[x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v)])
                actual = result[start:end].transpose(0, 1).unsqueeze(0)
                torch.testing.assert_close(actual, reference, rtol=0.1, atol=0.1)


def probe(backends=BACKENDS) -> dict:
    info = hardware_info()
    info["packages"] = {}
    for package in ("sageattention", "flash-attn", "triton-windows", "triton"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    results = info["backends"] = {}
    capability = tuple(info.get("gpu", {}).get("capability", (0, 0)))
    for mode in backends:
        try:
            if info.get("error"):
                raise RuntimeError(info["error"])
            _test_kernel(mode, capability)
            results[mode] = {"ok": True, "reason": "GPU varlen test passed."}
        except Exception as exc:
            results[mode] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:1800]}
    return info


def choose_mode(requested: str, report: dict, log=print) -> str:
    if requested not in MODES:
        raise ValueError("Choose Auto, SDPA, SageAttention 2 or FlashAttention 2.")
    modes = BACKENDS if requested == "auto" else (requested,)
    for mode in modes:
        if mode == "sdpa":
            return mode
        result = report.get("backends", {}).get(mode, {})
        if result.get("ok") is True:
            return mode
        log(f"SeedVR2 {mode} unavailable: {result.get('reason', 'GPU check unavailable.')}")
    return "sdpa"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["info", "all", *BACKENDS], default="all")
    args = parser.parse_args()
    report = hardware_info() if args.mode == "info" else probe(
        BACKENDS if args.mode == "all" else (args.mode,))
    args.output.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
