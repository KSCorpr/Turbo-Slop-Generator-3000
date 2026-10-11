"""Shared local video conversion and recovery helpers."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .. import settings
from . import sdcpp

OUTPUT_PREFIXES = ("ltx-2.5-",)


def _ffmpeg_exe(log: Callable[[str], None] | None = None) -> str:
    """Find a converter, repairing the portable Python when only it is missing.

    Code updates don't install newly added requirements. The first video
    generated after an update must be able to install its converter without
    asking the user to reinstall the whole application and its engines.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        pass

    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    if log:
        log("Installing the MP4 converter in the application Python (one time)…")
    try:
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input",
             "--disable-pip-version-check", "imageio-ffmpeg>=0.6,<1"],
            capture_output=True, text=True, errors="replace", timeout=300)
        if installed.returncode:
            raise RuntimeError((installed.stderr or installed.stdout)[-400:])
        importlib.invalidate_caches()
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        raise sdcpp.EngineError(
            "The MP4 converter could not be installed in the application "
            f"Python: {exc}. Run install.bat (or ./install.sh) and then "
            "convert the saved AVI after reinstalling the converter.") from exc


def encode_mp4(avi: Path, log: Callable[[str], None] | None = None) -> Path:
    """sd-cli writes AVI; browser playback needs an H.264 MP4 with no audio."""
    if not avi.is_file() or not avi.stat().st_size:
        raise sdcpp.EngineError(f"Video file not found: {avi}")
    try:
        ffmpeg = _ffmpeg_exe(log=log)
    except sdcpp.EngineError as exc:
        raise sdcpp.EngineError(f"Video saved as {avi}. {exc}") from exc
    mp4 = avi.with_suffix(".mp4")
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(avi), "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(mp4)], capture_output=True, text=True, errors="replace")
        if result.returncode or not mp4.is_file() or not mp4.stat().st_size:
            raise sdcpp.EngineError(
                f"Video saved as {avi}, but MP4 encoding failed: "
                f"{result.stderr[-400:]}")
    except OSError as exc:
        raise sdcpp.EngineError(f"Video saved as {avi}; MP4 encoding failed: {exc}") from exc
    avi.unlink()  # Keep only the playable result after a successful conversion.
    return mp4


def saved_avis() -> list[str]:
    """Offer only supported videos generated in the app's output folder."""
    return [p.name for p in sorted(
        (p for prefix in OUTPUT_PREFIXES
         for p in settings.OUTPUT_DIR.glob(f"{prefix}*.avi")),
        key=lambda p: p.stat().st_mtime, reverse=True) if p.is_file()]


def convert_saved_avi(name: str,
                      log: Callable[[str], None] | None = None) -> Path:
    """Recover a completed video run without loading the model or using the GPU."""
    if (not name or "/" in name or "\\" in name or Path(name).name != name
            or not name.startswith(OUTPUT_PREFIXES) or not name.endswith(".avi")):
        raise sdcpp.EngineError("Choose a saved AVI from the video tab.")
    return encode_mp4(settings.OUTPUT_DIR / name, log=log)
