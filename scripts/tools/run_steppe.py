#!/usr/bin/env python3
"""Standalone TensorForger/Steppe-Wan2.1 latent upscaler for images and videos.

Runs inside the isolated `tools_repo/steppe/.venv` CUDA 12.6 environment with
offline model loading, tiled Wan 2.1 VAE encode/decode, and stage offloading
so 11-12 GB cards (RTX 2080 Ti / RTX 3060) handle 2x/4x images and 2x/2x-FPS
videos without OOM.
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_steppe_pkg(addon_root: Path):
    src = addon_root / "source"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import steppe  # noqa: E402

    return steppe


def _resolve_dtype(torch_mod, name: str):
    if name == "bf16":
        if not torch_mod.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable on the selected GPU. Use Auto or FP16.")
        return torch_mod.bfloat16
    if name == "fp16":
        return torch_mod.float16
    if name == "fp32":
        return torch_mod.float32
    # Auto: BF16 on Ampere+ (compute capability >= 8.0), FP16 on Turing/Pascal.
    major, _ = torch_mod.cuda.get_device_capability(0)
    return torch_mod.bfloat16 if major >= 8 and torch_mod.cuda.is_bf16_supported() else torch_mod.float16


def _require_runtime(dtype_name: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Steppe-Wan2.1 requires an NVIDIA CUDA GPU. Select it in System and repair Steppe if needed."
        )
    device = torch.device("cuda:0")
    dtype = _resolve_dtype(torch, dtype_name)
    print(f"Steppe GPU: {torch.cuda.get_device_name(0)} · precision: {dtype}", flush=True)
    return torch, device, dtype


def _load_vae(addon_root: Path, torch_mod):
    from diffusers import AutoencoderKLWan

    vae_path = addon_root / "models" / "wan21_vae" / "vae"
    if not (vae_path / "config.json").is_file():
        raise RuntimeError("Wan 2.1 VAE is missing. Run Install / repair Steppe-Wan2.1 first.")
    vae = AutoencoderKLWan.from_pretrained(
        str(vae_path),
        torch_dtype=torch_mod.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    return vae


def _pad_spatial_tensor(tensor, multiple: int = 16):
    """Replicate-pad [B, C, T, H, W] so H and W are multiples of `multiple`."""
    import torch.nn.functional as F

    _, _, _, h, w = tensor.shape
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    # F.pad on 5D uses (w_left, w_right, h_top, h_bottom, t_front, t_back)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    return padded, h, w


def _run_image(args, steppe_mod, torch, device, dtype):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(args.input) as src:
        oriented = ImageOps.exif_transpose(src)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        rgba = oriented.convert("RGBA") if has_alpha else None
        rgb = oriented.convert("RGB")

    orig_w, orig_h = rgb.size
    passes = 2 if args.scale == 4 else 1
    target_w, target_h = orig_w * args.scale, orig_h * args.scale
    print(f"Steppe image: {orig_w}×{orig_h} → {target_w}×{target_h} (×{args.scale}, {passes} pass(es))", flush=True)

    arr = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    pixels, h0, w0 = _pad_spatial_tensor(pixels, multiple=16)

    print("Loading Wan 2.1 VAE…", flush=True)
    vae = _load_vae(args.addon_root, torch)
    print("Encoding source image to Wan 2.1 latent…", flush=True)
    vae.to(device)
    with torch.no_grad():
        latents = vae.encode(pixels.to(device=device, dtype=torch.float32)).latent_dist.mode().cpu()
    if not args.keep_on_gpu:
        vae.cpu()
        torch.cuda.empty_cache()

    print(f"Loading Steppe checkpoint ({args.checkpoint.name})…", flush=True)
    upscaler = steppe_mod.load_upscaler_file(str(args.checkpoint), dtype=dtype)
    upscaler.model.to(device)

    hff_val = None if args.hff < 0 else float(args.hff)
    cfg = steppe_mod.UpscaleSettings(
        spatial=True,
        temporal=False,
        hff=hff_val,
        guidance=float(args.guidance),
        steps=int(args.steps),
        shift=float(args.shift),
        seed=int(args.seed),
    )

    for idx in range(passes):
        step_counter = [0]

        def _tick():
            step_counter[0] += 1
            print(f"Steppe pass {idx + 1}/{passes} · eval {step_counter[0]}", flush=True)

        latents = upscaler.upscale(latents, cfg, device=device, on_pass=_tick)
        torch.cuda.empty_cache()

    if not args.keep_on_gpu:
        upscaler.model.cpu()
        del upscaler
        gc.collect()
        torch.cuda.empty_cache()

    print("Decoding upscaled Wan 2.1 latent…", flush=True)
    vae.to(device)
    with torch.no_grad():
        decoded = vae.decode(latents.to(device=device, dtype=torch.float32)).sample.cpu()
    vae.cpu()
    torch.cuda.empty_cache()

    # Crop padded pixels back to exact target_w x target_h
    decoded = decoded[0, :, 0, :target_h, :target_w]
    out_np = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    out_img = Image.fromarray(out_np, mode="RGB")
    if rgba is not None:
        alpha = rgba.getchannel("A").resize((target_w, target_h), Image.Resampling.LANCZOS)
        out_img = out_img.convert("RGBA")
        out_img.putalpha(alpha)
    out_img.save(args.output, format="PNG")
    print(f"Saved Steppe image: {args.output.name} ({out_img.width}×{out_img.height})", flush=True)


def _mux_audio(src_video: Path, silent_video: Path, final_video: Path) -> None:
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        silent_video.replace(final_video)
        return
    cmd = [
        ffmpeg, "-y",
        "-i", str(silent_video),
        "-i", str(src_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_video),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and final_video.is_file() and final_video.stat().st_size > 0:
        silent_video.unlink(missing_ok=True)
    else:
        silent_video.replace(final_video)


def _run_video(args, steppe_mod, torch, device, dtype):
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(args.input))
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps") or 24.0)
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError("No frames could be read from the source video.")

    src_t = len(frames)
    src_h, src_w = frames[0].shape[:2]
    spatial = bool(args.spatial)
    temporal = bool(args.temporal)
    if not spatial and not temporal:
        raise ValueError("Enable 2x spatial resolution and/or 2x temporal frame rate.")

    out_fps = src_fps * 2.0 if temporal else src_fps
    target_w = src_w * 2 if spatial else src_w
    target_h = src_h * 2 if spatial else src_h
    print(
        f"Steppe video: {src_w}×{src_h} @ {src_fps:.2f} fps ({src_t} frames) → "
        f"{target_w}×{target_h} @ {out_fps:.2f} fps (spatial={spatial}, temporal={temporal})",
        flush=True,
    )

    arr = np.stack([f[:, :, :3] for f in frames], axis=0).astype(np.float32) / 127.5 - 1.0  # [T, H, W, 3]
    pixels = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    pixels, _, _ = _pad_spatial_tensor(pixels, multiple=16)

    print("Loading Wan 2.1 VAE…", flush=True)
    vae = _load_vae(args.addon_root, torch)
    print("Encoding video frames to Wan 2.1 latents…", flush=True)
    vae.to(device)
    with torch.no_grad():
        latents = vae.encode(pixels.to(device=device, dtype=torch.float32)).latent_dist.mode().cpu()
    if not args.keep_on_gpu:
        vae.cpu()
        torch.cuda.empty_cache()

    print(f"Loading Steppe checkpoint ({args.checkpoint.name})…", flush=True)
    upscaler = steppe_mod.load_upscaler_file(str(args.checkpoint), dtype=dtype)
    upscaler.model.to(device)

    hff_val = None if args.hff < 0 else float(args.hff)
    cfg = steppe_mod.UpscaleSettings(
        spatial=spatial,
        temporal=temporal,
        hff=hff_val,
        guidance=float(args.guidance),
        steps=int(args.steps),
        shift=float(args.shift),
        seed=int(args.seed),
        temporal_window=int(args.temporal_window),
        temporal_context=2,
    )
    step_counter = [0]

    def _tick():
        step_counter[0] += 1
        print(f"Steppe video eval {step_counter[0]}", flush=True)

    latents = upscaler.upscale(latents, cfg, device=device, on_pass=_tick)
    if not args.keep_on_gpu:
        upscaler.model.cpu()
        del upscaler
        gc.collect()
        torch.cuda.empty_cache()

    print("Decoding upscaled Wan 2.1 video latents…", flush=True)
    vae.to(device)
    with torch.no_grad():
        decoded = vae.decode(latents.to(device=device, dtype=torch.float32)).sample.cpu()
    vae.cpu()
    torch.cuda.empty_cache()

    # decoded: [1, 3, T_out, H_pad, W_pad]
    decoded = decoded[0, :, :, :target_h, :target_w]
    out_np = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()

    silent_tmp = args.output.with_name(args.output.stem + "_silent.mp4")
    writer = imageio.get_writer(str(silent_tmp), fps=out_fps, codec="libx264", quality=8, macro_block_size=1)
    for idx in range(out_np.shape[0]):
        writer.append_data(out_np[idx])
    writer.close()

    _mux_audio(args.input, silent_tmp, args.output)
    print(f"Saved Steppe video: {args.output.name} ({out_np.shape[0]} frames @ {out_fps:.2f} fps)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--hff", type=float, default=0.9)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-window", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--keep-on-gpu", action="store_true")
    args = parser.parse_args()

    steppe_mod = _load_steppe_pkg(args.addon_root)
    torch, device, dtype = _require_runtime(args.dtype)
    if args.probe:
        import diffusers  # noqa: F401
        import imageio  # noqa: F401
        print(f"Steppe {steppe_mod.__version__} probe OK on {torch.cuda.get_device_name(0)}.", flush=True)
        return

    if not args.input or not args.output or not args.checkpoint:
        raise SystemExit("--input, --output, and --checkpoint are required.")
    if args.mode == "video":
        _run_video(args, steppe_mod, torch, device, dtype)
    else:
        _run_image(args, steppe_mod, torch, device, dtype)


if __name__ == "__main__":
    main()
