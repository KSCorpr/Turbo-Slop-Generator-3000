"""Cancellable Kandinsky ImageSR 2.0 upscaling in an isolated Python process."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .. import addons, kandinsky_sr as artifacts, settings
from . import sdcpp
from .local_jobs import LocalJob

JOB = LocalJob()
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# Allow up to 8K (8192 x 8192 = 67.1 MP), matching the model card's 4K -> 8K benchmark.
MAX_OUTPUT_PIXELS = 70_000_000
VALID_DTYPES = ("auto", "bf16", "fp16")
VALID_MEMORY_MODES = ("auto", "sequential", "model", "gpu")


def validate_controls(scale, creativity, overlap, dtype, memory_mode):
    if type(scale) is not int or scale not in (2, 4):
        raise ValueError("Choose ×2 or ×4 upscale factor.")
    if type(creativity) is not int or not 0 <= creativity <= 8:
        raise ValueError("Creativity must be an integer from 0 to 8 (4 is the calibrated default).")
    if not isinstance(overlap, (int, float)) or isinstance(overlap, bool) or not 0.0 <= float(overlap) <= 0.5:
        raise ValueError("Tile overlap must be between 0.0 and 0.5.")
    if dtype not in VALID_DTYPES:
        raise ValueError("Choose a listed precision (auto, bf16 or fp16).")
    if memory_mode not in VALID_MEMORY_MODES:
        raise ValueError("Choose a listed memory mode (auto, sequential, model or gpu).")


def target_size(size: tuple[int, int], scale: int) -> tuple[int, int]:
    width, height = (int(side) * int(scale) for side in size)
    if width < 16 or height < 16:
        raise ValueError("Kandinsky ImageSR 2.0 output must be at least 16 pixels on each side.")
    if width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("Output exceeds 70 megapixels (above 8K). Use ×2 or crop the source first.")
    return width, height


def upscale(
    source,
    scale: int = 4,
    creativity: int = artifacts.DEFAULT_CREATIVITY,
    *,
    overlap: float = 0.25,
    dtype: str = "auto",
    memory_mode: str = "auto",
    log=print,
) -> Path:
    validate_controls(scale, creativity, overlap, dtype, memory_mode)
    path = Path(source or "")
    if not path.is_file() or path.suffix.lower() not in IMAGES:
        raise ValueError("Import an image first. Kandinsky ImageSR 2.0 processes still images.")
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    expected = target_size((width, height), scale)
    if not artifacts.ready():
        raise RuntimeError("Install / repair Kandinsky ImageSR 2.0 from Models or this tab first.")
    output = sdcpp.unique_output("kandinsky-sr2", "png")
    settings.TMP_DIR.mkdir(parents=True, exist_ok=True)
    env = settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
    # Inference must run strictly offline using only downloaded files.
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    timestep = artifacts.CREATIVITY_TIMESTEPS[scale][creativity]
    with JOB.session(), tempfile.TemporaryDirectory(prefix="ksr2-", dir=settings.TMP_DIR) as temporary:
        source_copy = Path(temporary) / ("input" + path.suffix.lower())
        result = Path(temporary) / "upscaled.png"
        JOB.check()
        shutil.copyfile(path, source_copy)
        command = [
            addons.python("kandinsky-sr"),
            "-u",
            settings.ROOT / "scripts/tools/run_kandinsky_sr.py",
            "--addon-root",
            addons.root("kandinsky-sr"),
            "--input",
            source_copy,
            "--output",
            result,
            "--upscale",
            str(scale),
            "--creativity",
            str(creativity),
            "--overlap",
            f"{float(overlap):g}",
            "--dtype",
            dtype,
            "--memory-mode",
            memory_mode,
        ]
        log(
            f"Kandinsky ImageSR 2.0: {width} × {height} → {expected[0]} × {expected[1]} "
            f"(×{scale}, creativity {creativity}/8, t={timestep:g}, memory={memory_mode})."
        )
        JOB.run(command, log, env=env)
        JOB.check()
        if not result.is_file():
            raise RuntimeError("Kandinsky ImageSR 2.0 finished without an image.")
        with Image.open(result) as upscaled:
            upscaled.verify()
        with Image.open(result) as upscaled:
            if upscaled.size != expected:
                raise RuntimeError("Kandinsky ImageSR 2.0 returned unexpected dimensions; no image was exported.")
        JOB.check()
        shutil.move(result, output)
    return output
