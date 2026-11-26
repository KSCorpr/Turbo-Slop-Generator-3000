"""Run Kandinsky ImageSR 2.0 offline inside its isolated PyTorch/Diffusers venv."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps


def _add_source(addon_root: Path) -> None:
    src = addon_root / "source" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _resolve_dtype(torch, requested: str):
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_memory_mode(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "gpu"
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    # 5B transformer is ~10.25 GB: stream layers on <=14 GB GPUs (RTX 2080 Ti / 3060),
    # stage whole modules on 15-20 GB, keep fully resident on >=21 GB.
    if total_gib <= 14.5:
        return "sequential"
    if total_gib < 21.0:
        return "model"
    return "gpu"


def _probe(addon_root: Path) -> None:
    _add_source(addon_root)
    import diffusers
    import torch
    from kandinsky_image_sr import KandinskyImageSRPipeline  # noqa: F401

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        print(f"Kandinsky ImageSR 2.0 probe OK: {name} ({vram:.1f} GB VRAM, bf16={bf16}, "
              f"torch={torch.__version__}, diffusers={diffusers.__version__})", flush=True)
    else:
        print(f"Kandinsky ImageSR 2.0 probe OK (CPU mode, torch={torch.__version__}, "
              f"diffusers={diffusers.__version__})", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Kandinsky ImageSR 2.0 standalone runner")
    parser.add_argument("--addon-root", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--upscale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--creativity", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--memory-mode", choices=("auto", "sequential", "model", "gpu"), default="auto")
    args = parser.parse_args(argv)

    addon_root = Path(args.addon_root)
    if args.probe:
        _probe(addon_root)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --probe is set.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _add_source(addon_root)

    import torch
    from kandinsky_image_sr import KandinskyImageSRPipeline

    model_dir = addon_root / "models"
    dtype = _resolve_dtype(torch, args.dtype)
    mem_mode = _resolve_memory_mode(torch, args.memory_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kandinsky ImageSR 2.0 ({dtype}, memory={mem_mode}, device={device})…", flush=True)

    t0 = time.monotonic()
    pipe = KandinskyImageSRPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    if device == "cuda":
        if mem_mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif mem_mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    print(f"Pipeline ready in {time.monotonic() - t0:.1f}s.", flush=True)

    with Image.open(args.input) as raw:
        raw = ImageOps.exif_transpose(raw)
        has_alpha = raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info)
        alpha = raw.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = raw.convert("RGB")

    t1 = time.monotonic()
    result = pipe(
        rgb,
        upscale_factor=args.upscale,
        creativity=args.creativity,
        overlap=args.overlap,
        tile_batch_size=1,
    ).images[0]

    if alpha is not None:
        alpha_up = alpha.resize(result.size, Image.Resampling.LANCZOS)
        result = result.convert("RGBA")
        result.putalpha(alpha_up)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output, format="PNG")
    print(f"Done: {result.size[0]} × {result.size[1]} saved in {time.monotonic() - t1:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
