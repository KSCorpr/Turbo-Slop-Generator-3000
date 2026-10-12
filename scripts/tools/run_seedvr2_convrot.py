#!/usr/bin/env python3
"""Run the pinned ConvRot CLI with Comfy internals, without a node editor/server."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys


def driver_cuda_version() -> int:
    """Read the installed driver's CUDA API, without Torch or a CUDA Toolkit."""
    try:
        driver = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
                  else ctypes.CDLL("libcuda.so.1"))
        get_version = driver.cuDriverGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        version = ctypes.c_int()
        if get_version(ctypes.byref(version)) != 0:
            raise OSError("cuDriverGetVersion failed")
        return version.value
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Cannot read the NVIDIA CUDA driver. Install a compatible "
                           "NVIDIA driver before repairing SeedVR2 ConvRot.") from exc


def check_driver() -> int:
    version = driver_cuda_version()
    if version < 13000:
        raise RuntimeError(
            f"The NVIDIA driver supports CUDA {version // 1000}.{version % 1000 // 10}; "
            "SeedVR2 ConvRot requires CUDA 13.0 support (NVIDIA driver 580 or newer). "
            "Installing the CUDA Toolkit does not update the driver. "
            "Classic SeedVR2 remains available with CUDA 12.6.")
    print(f"NVIDIA driver supports CUDA {version // 1000}.{version % 1000 // 10}.", flush=True)
    return version


def require_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ConvRot requires a working NVIDIA CUDA GPU. Select an RTX GPU in System; "
                           "the NVIDIA driver must support CUDA 13.0.")
    if torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("ConvRot requires RTX Tensor Cores. Select an RTX GPU; "
                           "use classic GGUF on GTX 10xx.")
    cuda = getattr(torch.version, "cuda", None)
    if cuda is None or tuple(map(int, cuda.split(".")[:2])) < (13, 0):
        raise RuntimeError(
            f"SeedVR2 ConvRot is using PyTorch {torch.__version__} / CUDA {cuda or 'unavailable'}, "
            "but its optimized kernels require PyTorch CUDA 13.0 or newer. "
            "Use Install / repair SeedVR2 with a ConvRot model selected. "
            "The repair changes only ConvRot's isolated Python; installing a CUDA Toolkit "
            "does not change PyTorch's CUDA build.")
    return torch


def require_native_backend():
    # Import after Comfy has configured its registry: checking the wheel alone
    # misses a backend explicitly disabled by comfy.quant_ops.
    import comfy.quant_ops  # noqa: F401
    import comfy_kitchen as kitchen

    if not kitchen.registry.is_available("cuda"):
        info = kitchen.list_backends().get("cuda", {})
        reason = info.get("unavailable_reason") or "CUDA backend disabled"
        raise RuntimeError("SeedVR2 ConvRot optimized CUDA kernels are unavailable: "
                           f"{reason}. Use Install / repair SeedVR2 with ConvRot selected.")


def configure(root: Path) -> Path:
    source, comfy = root / "source", root / "comfy"
    if not (source / "inference_cli.py").is_file() or not (comfy / "comfy" / "ops.py").is_file():
        raise RuntimeError("Install / repair SeedVR2 ConvRot first.")
    # Windows child processes require os.pathsep, not the upstream literal ':'.
    paths = [str(source), str(comfy)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    return source


def probe(precision: str) -> dict:
    """Exercise the selected loader and native GPU matmul before downloading GBs."""
    torch = require_runtime()
    capability = torch.cuda.get_device_capability(0)
    dtype, device = torch.float16, torch.device("cuda:0")
    if precision == "w4a8":
        from src.optimization.w4a8_native_ops import get_w4a8_mixed_precision_ops
        ops = get_w4a8_mixed_precision_ops(dtype)
        config = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
        state = {
            "weight": torch.full((256, 128), 17, dtype=torch.int8),
            "weight_s_rel": torch.ones((256, 16)).to(torch.float8_e4m3fn),
            "weight_s_channel": torch.full((256,), 0.01),
        }
    else:
        from src.optimization.int8_native_ops import get_hswq_mixed_precision_ops
        ops = get_hswq_mixed_precision_ops(dtype)
        config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        state = {
            "weight": torch.ones((256, 256), dtype=torch.int8),
            "weight_scale": torch.tensor(0.01),
        }
    require_native_backend()
    state["comfy_quant"] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    layer = ops.Linear(256, 256, bias=False, device=device, dtype=dtype)
    layer.load_state_dict(state, strict=True)
    with torch.inference_mode():
        result = layer(torch.ones((32, 256), device=device, dtype=dtype))
        torch.cuda.synchronize()
    packed = getattr(layer.weight, "_qdata", None)
    if result.shape != (32, 256) or not torch.isfinite(result).all().item():
        raise RuntimeError("ConvRot GPU test produced invalid values.")
    if packed is None or packed.dtype != torch.int8:
        raise RuntimeError("ConvRot loader expanded the weights; native INT8 storage is required.")
    return {"gpu": torch.cuda.get_device_name(0), "capability": list(capability),
            "precision": precision, "torch": torch.__version__, "cuda": torch.version.cuda,
            "backend": "cuda", "native_gpu_test": True}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--check-driver", action="store_true")
    parser.add_argument("--precision", choices=("int8", "w4a8"), default="w4a8")
    args, remaining = parser.parse_known_args(argv)
    if args.check_driver:
        check_driver()
        return
    source = configure(args.addon_root)
    if args.probe:
        print(json.dumps(probe(args.precision)), flush=True)
        return
    if not any(arg in ("-h", "--help") for arg in remaining):
        require_runtime()
    sys.argv = [str(source / "inference_cli.py"), *remaining]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
