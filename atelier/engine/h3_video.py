"""MiniMax H3 Video & HyperFlow H3 Flash runner using native sd-cli (`--mode vid_gen`).

Tuned for 11-12 GB NVIDIA GPUs paired with 64 GB system RAM (e.g. RTX 2080 Ti
11 GB, or RTX 3060 12 GB + GTX 1080):
- Automatically selects the main RTX GPU via `settings.generation_gpu_index`
  (never falling back to FP16-slow Pascal cards for primary diffusion compute).
- Uses `--auto-fit on` (or `--offload-to-cpu` + `--stream-layers` fallback)
  so large weights stream cleanly from 64 GB RAM without VRAM OOM.
- Enforces `--diffusion-fa`, `--vae-tiling`, `--temporal-tiling`, `--rng cpu`,
  and `--model-args minimax_h3_attn_chunk_budget=<GiB>` to cap peak attention
  activation memory inside 11-12 GB VRAM.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable

from .. import h3_video as catalog
from .. import hardware, settings
from . import generate as gen_engine
from . import sdcpp


FPS = 24

RESOLUTION_PRESETS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("480p Widescreen 16:9 · 864×480 (recommended · 11-12 GB)", (864, 480)),
    ("480p Portrait 9:16 · 480×864 (vertical)", (480, 864)),
    ("Square 1:1 · 640×640", (640, 640)),
    ("Fast Draft 16:9 · 640×384 (fastest)", (640, 384)),
    ("HD 720p 16:9 · 1216×672 (high VRAM / hires target)", (1216, 672)),
)

FRAME_PRESETS: tuple[tuple[str, int], ...] = (
    ("22 frames (~0.9 s · ultra-fast test)", 22),
    ("39 frames (~1.6 s · fast)", 39),
    ("56 frames (~2.3 s · recommended 11-12 GB)", 56),
    ("73 frames (~3.0 s · standard)", 73),
    ("90 frames (~3.8 s · long · use 0.25-0.5 GiB chunk budget)", 90),
    ("107 frames (~4.5 s · very long · use 0.25 GiB chunk budget)", 107),
    ("124 frames (~5.2 s · max · use 0.25 GiB chunk budget)", 124),
)

HYPERFLOW_SIGMAS: dict[int, str] = {
    4: "hyperflow-h3-flash-4",
    6: "hyperflow-h3-flash-6",
    8: "hyperflow-h3-flash-8",
}


def align_dimension(px: int, minimum: int = 320, maximum: int = 1920) -> int:
    """MiniMax H3 Video requires width and height to be multiples of 32."""
    val = max(minimum, min(maximum, int(px)))
    return ((val + 31) // 32) * 32


def align_frames(frames: int) -> int:
    """MiniMax H3 Video aligns frame count upward to the `17k + 5` grid (min 5)."""
    n = max(5, int(frames))
    if n <= 5:
        return 5
    k = (n - 5 + 16) // 17
    return 17 * k + 5


def memory_plan(
    sd_cli: Path | None,
    prefs: dict,
    *,
    attn_chunk_budget: float = 0.5,
) -> dict:
    """Build a safe VRAM/RAM placement plan for 11-12 GB GPUs + 64 GB RAM."""
    gpu_index = settings.generation_gpu_index(prefs)
    gpus = hardware.detect_gpus()
    selected = next((g for g in gpus if g.index == gpu_index), None)
    if selected is None and gpus:
        selected = hardware.auto_profile().gpu
        gpu_index = selected.index if selected else None

    known = sdcpp.supported_options(sd_cli) if sd_cli else frozenset()
    use_auto_fit = "--auto-fit" in known
    flags = {
        "diffusion_fa": True,
        "vae_tiling": True,
        "offload_to_cpu": not use_auto_fit,
        "clip_on_cpu": False,
        "vae_on_cpu": False,
        "conv_direct_diffusion": False,
        "conv_direct_vae": False,
    }
    max_vram = sdcpp.max_vram_arg("auto", spare_gib=1.2) if "--max-vram" in known else ""
    stream_layers = (not use_auto_fit) and ("--stream-layers" in known)
    budget_val = max(0.0, float(attn_chunk_budget))
    model_args = [f"minimax_h3_attn_chunk_budget={budget_val:g}"]
    placement = (
        f"sd.cpp auto-fit + RAM offload · attn chunk {budget_val:g} GiB · spatial+temporal VAE tiling"
        if use_auto_fit
        else f"CPU RAM offload + layer streaming · attn chunk {budget_val:g} GiB · spatial+temporal VAE tiling"
    )
    return {
        "gpu_index": gpu_index,
        "gpu_name": selected.name if selected else "CPU",
        "vram_gb": selected.vram_gb if selected else 0.0,
        "flags": flags,
        "auto_fit": use_auto_fit,
        "max_vram": max_vram,
        "stream_layers": stream_layers,
        "model_args": model_args,
        "placement": placement,
    }


def _check_engine(sd_cli: Path, entries: list[catalog.WeightEntry]) -> None:
    required_build = max(e.min_build for e in entries)
    build = gen_engine._engine_build_number()
    if build is not None and build < required_build:
        raise sdcpp.EngineError(
            f"Selected MiniMax H3 Video weights require sd.cpp build {required_build} or newer. "
            "Run update.bat (or ./update.sh on Linux/Mac)."
        )
    known = sdcpp.supported_options(sd_cli)
    for flag in ("--audio-vae", "--video-frames"):
        if flag not in known:
            raise sdcpp.EngineError(
                f"MiniMax H3 Video requires a newer sd.cpp engine with {flag}. "
                "Run update.bat (or ./update.sh on Linux/Mac)."
            )


def generate_video(
    *,
    prompt: str,
    diffusion_key: str = catalog.DEFAULT_DIFFUSION,
    encoder_key: str = catalog.DEFAULT_ENCODER,
    vae_key: str = catalog.DEFAULT_VAE,
    width: int = 864,
    height: int = 480,
    video_frames: int = 56,
    steps: int = 6,
    cfg_scale: float = 1.0,
    seed: int = -1,
    init_image: Path | None = None,
    end_image: Path | None = None,
    ref_images: list[Path] | None = None,
    attn_chunk_budget: float = 0.5,
    temporal_tiling: bool = True,
    hires_enable: bool = False,
    hires_width: int = 1216,
    hires_height: int = 672,
    hires_steps: int = 4,
    hires_denoise: float = 0.4,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, int]:
    """Generate a MiniMax H3 MP4 video (with stereo audio) via `sd-cli --mode vid_gen`."""
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Enter a video prompt first.")

    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "The sd-cli binary was not found. Run install.bat or update.bat first."
        )

    diff_entry = catalog.DIFFUSION_VARIANTS.get(diffusion_key)
    enc_entry = catalog.ENCODER_VARIANTS.get(encoder_key)
    vae_entry = catalog.VIDEO_VAE_VARIANTS.get(vae_key)
    if diff_entry is None or enc_entry is None or vae_entry is None:
        raise ValueError("Choose listed MiniMax H3 Video weights.")

    _check_engine(sd_cli, [diff_entry, enc_entry, vae_entry, catalog.AUDIO_VAE])

    files = catalog.resolve_files(diffusion_key, encoder_key, vae_key)
    missing = [role for role, path in files.items() if path is None]
    if missing:
        raise sdcpp.EngineError(
            f"Missing MiniMax H3 Video weights ({', '.join(missing)}). "
            "Click 'Install / download selected H3 Video pack' first."
        )

    refs = [Path(p) for p in (ref_images or []) if p]
    if diff_entry.kind == "ref2va" and (init_image or end_image):
        raise ValueError(
            "Ref2VA checkpoints use reference images (-r) and cannot be combined with First/Last frame."
        )
    if diff_entry.kind == "fl2va" and refs:
        raise ValueError(
            "FL2VA checkpoints use First/Last frame conditioning. Switch to a Ref2VA checkpoint for reference images."
        )

    w = align_dimension(width)
    h = align_dimension(height)
    frames = align_frames(video_frames)
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = -1
    if s < 0:
        s = random.randint(0, 2**31 - 1)

    sigmas = ""
    actual_steps = max(1, int(steps))
    if diff_entry.hyperflow:
        # HyperFlow H3 Flash uses distilled sigma presets for 4, 6, or 8 steps at CFG 1.
        if actual_steps in HYPERFLOW_SIGMAS:
            sigmas = HYPERFLOW_SIGMAS[actual_steps]
        cfg_scale = 1.0

    hires_params = None
    hires_sigmas = ""
    if hires_enable:
        if init_image or end_image:
            if log:
                log("Note: 2-stage latent hires is skipped when First/Last frame conditioning is active (sd.cpp limitation).")
        else:
            hw = align_dimension(max(w, int(hires_width)))
            hh = align_dimension(max(h, int(hires_height)))
            hires_params = sdcpp.HiresParams(
                upscaler="Latent (bicubic)",
                denoise=float(hires_denoise),
                steps=max(1, int(hires_steps)),
                target_width=hw,
                target_height=hh,
            )

    prefs = settings.load_prefs()
    plan = memory_plan(sd_cli, prefs, attn_chunk_budget=attn_chunk_budget)

    req = sdcpp.VideoRequest(
        diffusion_model=files["diffusion"],
        vae=files["vae"],
        audio_vae=files["audio_vae"],
        text_encoder=files["encoder"],
        prompt=clean_prompt,
        steps=actual_steps,
        cfg_scale=float(cfg_scale),
        sampler="euler",
        sigmas=sigmas,
        width=w,
        height=h,
        video_frames=frames,
        fps=FPS,
        seed=s,
        init_image=Path(init_image) if init_image else None,
        end_image=Path(end_image) if end_image else None,
        ref_image=refs or None,
        extra_flags=["--rng", "cpu"],
        model_args=plan["model_args"],
        temporal_tiling=bool(temporal_tiling),
        hires=hires_params,
        hires_sigmas=hires_sigmas,
        sage_attn=bool(prefs.get("sage_attention")),
        flags=plan["flags"],
        gpu_index=plan["gpu_index"],
        auto_fit=plan["auto_fit"],
        max_vram=plan["max_vram"],
        stream_layers=plan["stream_layers"],
    )

    settings.ensure_dirs()
    out_path = sdcpp.unique_output("h3-video", "mp4")
    cmd = sdcpp.build_vid_gen_cmd(sd_cli, req, out_path)

    if log:
        log(f"MiniMax H3 Video: {diff_entry.label}")
        log(f"GPU: {plan['gpu_name']} ({plan['vram_gb']:.0f} GB) · {plan['placement']}")
        log(f"Canvas: {w}×{h} · {frames} frames @ {FPS} fps (~{frames / FPS:.1f} s) · Seed: {s}")
        if hires_params:
            log(f"2-stage latent hires: {w}×{h} → {hires_params.target_width}×{hires_params.target_height} ({hires_params.steps} steps)")

    sdcpp.run(
        cmd,
        log=log,
        gpu_index=plan["gpu_index"],
        all_gpus=plan["auto_fit"],
    )

    if not out_path.is_file():
        # sd-cli sometimes writes `<stem>.avi` or `<stem>.mp4` depending on build
        candidates = sorted(out_path.parent.glob(f"{out_path.stem}.*"))
        media = [p for p in candidates if p.suffix.lower() in {".mp4", ".avi", ".webm", ".mov"}]
        if media:
            out_path = media[0]
        else:
            raise sdcpp.EngineError("sd-cli finished without producing a video file.")

    _write_sidecar(
        out_path,
        prompt=clean_prompt,
        diffusion=diff_entry.key,
        encoder=enc_entry.key,
        vae=vae_entry.key,
        width=w,
        height=h,
        frames=frames,
        steps=actual_steps,
        sigmas=sigmas,
        seed=s,
        hires=hires_params,
    )
    return out_path, s


def _write_sidecar(
    out_path: Path,
    *,
    prompt: str,
    diffusion: str,
    encoder: str,
    vae: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    sigmas: str,
    seed: int,
    hires: sdcpp.HiresParams | None,
) -> None:
    lines = [
        f"File: {out_path.name}",
        "Model: minimax-h3-video",
        f"Diffusion: {diffusion}",
        f"Text encoder: {encoder}",
        f"Video VAE: {vae}",
        f"Prompt: {prompt}",
        f"Size: {width}x{height} · {frames} frames @ {FPS} fps",
        f"Steps: {steps}" + (f" ({sigmas})" if sigmas else ""),
        f"Seed: {seed}",
    ]
    if hires:
        lines.append(f"Hires: {hires.target_width}x{hires.target_height} ({hires.steps} steps, denoise {hires.denoise})")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        out_path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass
