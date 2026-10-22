"""Cancellable VOSR 2.0 image restoration in an isolated Python process."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .. import addons, settings, vosr as artifacts
from . import sdcpp
from .local_jobs import LocalJob

JOB = LocalJob()
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_OUTPUT_PIXELS = 32_000_000


def validate_controls(scale, dtype, color, tile, overlap, vae_tile, vae_overlap, seed):
    if type(scale) is not int or scale not in (1, 2, 4):
        raise ValueError("Choose ×1, ×2 or ×4.")
    if dtype not in {"fp16", "bf16"} or color not in {"wavelet", "adain", "none"}:
        raise ValueError("Choose a listed precision and color alignment.")
    for size, margin, label in ((tile, overlap, "DiT"), (vae_tile, vae_overlap, "VAE")):
        if (type(size) is not int or size not in (128, 256, 512, 1024)
                or type(margin) is not int or not 0 <= margin < size or margin % 16):
            raise ValueError(f"{label}: use a listed tile size and an overlap smaller than the tile, in multiples of 16.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")


def target_size(size, scale):
    width, height = (int(side) * scale for side in size)
    if width < 16 or height < 16:
        raise ValueError("VOSR output must be at least 16 pixels on each side.")
    if width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("VOSR output exceeds 32 megapixels. Reduce the scale or crop the source first.")
    return width, height


def restore(source, scale=4, seed=42, *, dtype="fp16", color="wavelet", tile=512,
            overlap=64, vae_tile=512, vae_overlap=64, offload=True, log=print):
    validate_controls(scale, dtype, color, tile, overlap, vae_tile, vae_overlap, seed)
    path = Path(source or "")
    if not path.is_file() or path.suffix.lower() not in IMAGES:
        raise ValueError("Import an image first. VOSR 2.0 processes still images.")
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    expected = target_size((width, height), scale)
    if not artifacts.ready():
        raise RuntimeError("Install / repair VOSR 2.0 from Models or this tab first.")
    output = sdcpp.unique_output("vosr2", "png")
    settings.TMP_DIR.mkdir(parents=True, exist_ok=True)
    env = settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
    # Inference must use only the selected, already downloaded files.
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with JOB.session(), tempfile.TemporaryDirectory(prefix="vosr-", dir=settings.TMP_DIR) as temporary:
        source_copy = Path(temporary) / ("input" + path.suffix.lower())
        result = Path(temporary) / "restored.png"
        JOB.check()
        shutil.copyfile(path, source_copy)
        command = [addons.python("vosr"), "-u", settings.ROOT / "scripts/tools/run_vosr.py",
                   "--addon-root", addons.root("vosr"), "--input", source_copy, "--output", result,
                   "--upscale", str(scale), "--seed", str(seed), "--dtype", dtype,
                   "--color", color, "--tile-size", str(tile), "--tile-overlap", str(overlap),
                   "--vae-tile-size", str(vae_tile), "--vae-tile-overlap", str(vae_overlap)]
        if not offload:
            command.append("--keep-on-gpu")
        log(f"VOSR 2.0: {width} × {height} → {expected[0]} × {expected[1]}; one-step restoration.")
        JOB.run(command, log, env=env)
        JOB.check()
        if not result.is_file():
            raise RuntimeError("VOSR finished without an image.")
        with Image.open(result) as restored:
            restored.verify()
        with Image.open(result) as restored:
            if restored.size != expected:
                raise RuntimeError("VOSR returned unexpected dimensions; no image was exported.")
        JOB.check()
        shutil.move(result, output)
    return output
