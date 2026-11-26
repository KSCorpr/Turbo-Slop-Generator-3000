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
    if torch.cuda.is_available():
        # PyTorch's is_bf16_supported() defaults to including_emulation=True,
        # which returns True on Turing (RTX 2080 Ti, sm_75) and runs BF16 in
        # slow software emulation. Require native BF16 Tensor Cores (Ampere+, sm_80+).
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _resolve_memory_mode(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "gpu"
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    # 5B transformer is ~10.25 GB: stream transformer blocks on <=14 GB GPUs
    # (RTX 2080 Ti / 3060), stage whole modules on 15-20 GB, keep resident on >=21 GB.
    if total_gib <= 14.5:
        return "sequential"
    if total_gib < 21.0:
        return "model"
    return "gpu"


def _configure_memory(pipe, torch, mem_mode: str, device: str, dtype) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"
    # Keep the 167 MB FLUX VAE in float32 when running FP16 on Turing to avoid
    # FP16 VAE overflow/black tiles while costing only ~335 MB of VRAM.
    if dtype == torch.float16 and hasattr(pipe, "vae"):
        pipe.vae.to(dtype=torch.float32)

    if mem_mode == "sequential":
        # Block-level group offload is 5-15x faster than accelerate's leaf-level
        # sequential offload while keeping peak VRAM under ~3.5 GB.
        if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "enable_group_offload"):
            cuda_dev = torch.device("cuda")
            cpu_dev = torch.device("cpu")
            for use_stream in (True, False):
                try:
                    pipe.transformer.enable_group_offload(
                        onload_device=cuda_dev,
                        offload_device=cpu_dev,
                        offload_type="block_level",
                        num_blocks_per_group=1,
                        use_stream=use_stream,
                    )
                    pipe.vae.to(cuda_dev)
                    return "block_stream" if use_stream else "block"
                except Exception as exc:
                    print(f"Group offload (stream={use_stream}) unavailable ({exc}); trying fallback…",
                          flush=True)
        pipe.enable_sequential_cpu_offload()
        return "sequential"
    if mem_mode == "model":
        pipe.enable_model_cpu_offload()
        return "model"
    pipe.to("cuda")
    return "gpu"


def _probe(addon_root: Path) -> None:
    _add_source(addon_root)
    import diffusers
    import torch
    from kandinsky_image_sr import KandinskyImageSRPipeline  # noqa: F401

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        major, minor = torch.cuda.get_device_capability(0)
        bf16_native = major >= 8
        print(f"Kandinsky ImageSR 2.0 probe OK: {name} (sm_{major}{minor}, {vram:.1f} GB VRAM, "
              f"native_bf16={bf16_native}, torch={torch.__version__}, diffusers={diffusers.__version__})",
              flush=True)
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
    from kandinsky_image_sr.tiling import tile_positions

    model_dir = addon_root / "models"
    dtype = _resolve_dtype(torch, args.dtype)
    mem_mode = _resolve_memory_mode(torch, args.memory_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kandinsky ImageSR 2.0 ({dtype}, memory={mem_mode}, device={device})…", flush=True)

    t0 = time.monotonic()
    try:
        pipe = KandinskyImageSRPipeline.from_pretrained(
            str(model_dir),
            dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        pipe = KandinskyImageSRPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            local_files_only=True,
        )
    active_mem = _configure_memory(pipe, torch, mem_mode, device, dtype)
    print(f"Pipeline ready in {time.monotonic() - t0:.1f}s (offload={active_mem}).", flush=True)

    with Image.open(args.input) as raw:
        raw = ImageOps.exif_transpose(raw)
        has_alpha = raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info)
        alpha = raw.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = raw.convert("RGB")

    tile_size = int(getattr(pipe.config, "tile_size", 1024))
    lq_tile = tile_size // int(args.upscale)
    stride = max(1, int(lq_tile * (1.0 - float(args.overlap))))
    n_rows = len(tile_positions(max(rgb.height, lq_tile), lq_tile, stride))
    n_cols = len(tile_positions(max(rgb.width, lq_tile), lq_tile, stride))
    total_tiles = n_rows * n_cols
    print(
        f"Upscaling {total_tiles} tile(s) ({n_cols} × {n_rows}, tile={tile_size}px, overlap={args.overlap:.2f})…",
        flush=True,
    )

    t1 = time.monotonic()
    done_tiles = [0]
    orig_infer = getattr(pipe, "_infer_tiles", None)
    if callable(orig_infer):
        def _logged_infer_tiles(pixels, timestep, dev):
            t_tile = time.monotonic()
            out = orig_infer(pixels, timestep, dev)
            done_tiles[0] += int(pixels.shape[0])
            dt = time.monotonic() - t_tile
            elapsed = time.monotonic() - t1
            rem = max(0, total_tiles - done_tiles[0]) * (elapsed / max(1, done_tiles[0]))
            print(
                f"Tile {done_tiles[0]}/{total_tiles} done ({dt:.1f}s · elapsed {elapsed:.0f}s · ETA {rem:.0f}s)",
                flush=True,
            )
            return out

        pipe._infer_tiles = _logged_infer_tiles

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
