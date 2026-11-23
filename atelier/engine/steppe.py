"""Cancellable TensorForger/Steppe-Wan2.1 image & video latent upscaler runner."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .. import addons, settings, steppe as artifacts
from . import sdcpp
from .local_jobs import LocalJob

JOB = LocalJob()
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MEDIA = IMAGES | VIDEOS
MAX_OUTPUT_PIXELS = 33_554_432  # up to 4096×8192 / 32 MP


def validate_controls(
    variant: str,
    scale: int,
    hff: float,
    guidance: float,
    steps: int,
    dtype: str,
    seed: int,
    temporal_window: int,
) -> None:
    if variant not in artifacts.MODELS:
        raise ValueError("Choose a listed Steppe-Wan2.1 model variant.")
    if type(scale) is not int or scale not in (2, 4):
        raise ValueError("Choose ×2 or ×4 image scale.")
    if not (-1.0 <= float(hff) <= 1.0):
        raise ValueError("HFF sharpness must be -1 (auto) or between 0.0 and 1.0.")
    if not (1.0 <= float(guidance) <= 10.0):
        raise ValueError("Guidance must be between 1.0 and 10.0.")
    if not (1 <= int(steps) <= 100):
        raise ValueError("Steps must be between 1 and 100.")
    if dtype not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("Choose a listed precision mode.")
    if type(seed) is not int or not (0 <= seed <= 2**32 - 1):
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    if int(temporal_window) < 0 or int(temporal_window) > 64:
        raise ValueError("Temporal window must be between 0 and 64 latent frames.")


def target_size(size: tuple[int, int], scale: int) -> tuple[int, int]:
    width, height = int(size[0]) * int(scale), int(size[1]) * int(scale)
    if width < 32 or height < 32:
        raise ValueError("Steppe output must be at least 32 pixels on each side.")
    if width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("Steppe output exceeds 32 megapixels. Use ×2 or crop the source first.")
    return width, height


def upscale(
    source: str | Path,
    *,
    variant: str = artifacts.DEFAULT_VARIANT,
    scale: int = 2,
    video_spatial: bool = True,
    video_temporal: bool = False,
    hff: float = 0.9,
    guidance: float = 1.0,
    steps: int = 20,
    dtype: str = "auto",
    seed: int = 42,
    temporal_window: int = 8,
    offload: bool = True,
    log=print,
) -> Path:
    validate_controls(variant, int(scale), float(hff), float(guidance), int(steps), dtype, int(seed), int(temporal_window))
    path = Path(source or "")
    suffix = path.suffix.lower()
    if not path.is_file() or suffix not in MEDIA:
        raise ValueError("Import an image or video first.")
    is_video = suffix in VIDEOS
    if is_video and not (video_spatial or video_temporal):
        raise ValueError("Enable 2× spatial resolution and/or 2× frame rate interpolation for videos.")

    if not artifacts.ready(variant):
        raise RuntimeError(
            f"Install / repair Steppe-Wan2.1 ({variant}) from Models or this tab first."
        )

    expected = None
    if not is_video:
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        expected = target_size((width, height), int(scale))

    out_ext = "mp4" if is_video else "png"
    output = sdcpp.unique_output("steppe-wan21", out_ext)
    settings.TMP_DIR.mkdir(parents=True, exist_ok=True)
    env = settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    with JOB.session(), tempfile.TemporaryDirectory(prefix="steppe-", dir=settings.TMP_DIR) as temporary:
        source_copy = Path(temporary) / ("input" + suffix)
        result = Path(temporary) / ("upscaled." + out_ext)
        JOB.check()
        shutil.copyfile(path, source_copy)

        command = [
            addons.python("steppe"), "-u", settings.ROOT / "scripts/tools/run_steppe.py",
            "--addon-root", addons.root("steppe"),
            "--mode", "video" if is_video else "image",
            "--input", source_copy,
            "--output", result,
            "--checkpoint", artifacts.checkpoint_path(variant),
            "--scale", str(int(scale)),
            "--hff", f"{float(hff):g}",
            "--guidance", f"{float(guidance):g}",
            "--steps", str(int(steps)),
            "--seed", str(int(seed)),
            "--temporal-window", str(int(temporal_window)),
            "--dtype", dtype,
        ]
        if is_video and video_spatial:
            command.append("--spatial")
        if is_video and video_temporal:
            command.append("--temporal")
        if not offload:
            command.append("--keep-on-gpu")

        if is_video:
            log(f"Steppe-Wan2.1 ({variant}): video upscale (2× res={video_spatial}, 2× FPS={video_temporal}).")
        else:
            log(f"Steppe-Wan2.1 ({variant}): {width} × {height} → {expected[0]} × {expected[1]} (×{scale}).")
        JOB.run(command, log, env=env)
        JOB.check()
        if not result.is_file():
            raise RuntimeError("Steppe-Wan2.1 finished without producing an output file.")
        if not is_video and expected is not None:
            with Image.open(result) as restored:
                if restored.size != expected:
                    raise RuntimeError("Steppe-Wan2.1 returned unexpected dimensions; no image was exported.")
        JOB.check()
        shutil.move(result, output)
    return output
