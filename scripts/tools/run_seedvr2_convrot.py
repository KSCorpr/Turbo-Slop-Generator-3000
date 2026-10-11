#!/usr/bin/env python3
"""Run the pinned ConvRot CLI with Comfy internals, without a node editor/server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


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
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ConvRot requires a working NVIDIA CUDA GPU. Select the RTX 3060 in System.")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError("ConvRot requires RTX Tensor Cores. Select the RTX 3060; use classic GGUF on GTX 10xx.")
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
            "precision": precision, "torch": torch.__version__, "native_gpu_test": True}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--precision", choices=("int8", "w4a8"), default="w4a8")
    args, remaining = parser.parse_known_args(argv)
    source = configure(args.addon_root)
    if args.probe:
        print(json.dumps(probe(args.precision)), flush=True)
        return
    sys.argv = [str(source / "inference_cli.py"), *remaining]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
