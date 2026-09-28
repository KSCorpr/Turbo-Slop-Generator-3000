"""MiniMax H3 Turbo video-only inference with sd.cpp and local MP4 encoding."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .. import settings
from . import sdcpp

FRAMES = (22, 39, 56)  # MiniMax H3 aligns video to 17k+5, at 24 fps.
SIZES = ((640, 352), (864, 480), (960, 544))
STEPS = 4


def build_command(sd_cli: Path, diffusion: Path, encoder: Path, vae: Path,
                  prompt: str, output: Path, *, frames: int = 22,
                  width: int = 864, height: int = 480, seed: int = -1,
                  first: Path | None = None, last: Path | None = None,
                  gpu_index: int | None = None) -> list[str]:
    if not prompt.strip():
        raise sdcpp.EngineError("Describe the video before generating it.")
    if (width, height) not in SIZES or frames not in FRAMES:
        raise sdcpp.EngineError("Choose a listed H3 resolution and duration.")
    if last and not first:
        raise sdcpp.EngineError("A last frame also requires a first frame.")
    sdcpp._require(diffusion, encoder, vae, first, last)
    known = sdcpp.supported_options(sd_cli)
    needed = {"--video-frames", "--diffusion-model", "--vae", "--llm",
              "--auto-fit", "--max-vram", "--flow-shift"}
    if not needed.issubset(known) or (last and "--end-img" not in known):
        raise sdcpp.EngineError(
            "MiniMax H3 Turbo needs a recent sd.cpp video engine with auto-fit "
            "and segmented compute. Run update.bat (or ./update.sh).")
    request = sdcpp.GenRequest(
        auto_fit=True, max_vram=sdcpp.max_vram_arg("auto"),
        flags={"diffusion_fa": "--diffusion-fa" in known,
               "vae_tiling": "--vae-tiling" in known},
        gpu_index=gpu_index)
    cmd = [str(sd_cli), "-M", "vid_gen",
           "--diffusion-model", str(diffusion), "--vae", str(vae),
           "--llm", str(encoder), "-p", prompt,
           "--steps", str(STEPS), "--cfg-scale", "1.0",
           "--sampling-method", "euler", "--flow-shift", "6",
           "-W", str(width), "-H", str(height),
           "--video-frames", str(frames), "--fps", "24", "-s", str(seed)]
    if "--rng" in known:
        cmd += ["--rng", "cpu"]
    if first:
        cmd += ["-i", str(first)]
    if last:
        cmd += ["--end-img", str(last)]
    # No --audio-vae: sd.cpp skips audio decoding and writes video only.
    cmd += sdcpp.memory_args(sd_cli, request)
    cmd += ["-o", str(output), "-v"]
    return cmd


def encode_mp4(avi: Path) -> Path:
    """sd-cli writes AVI; browser playback needs an H.264 MP4 with no audio."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError) as exc:
        raise sdcpp.EngineError(
            f"Video saved as {avi}, but the bundled FFmpeg is missing. "
            "Run install.bat (or ./install.sh) to install dependencies.") from exc
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


def generate(diffusion: Path, encoder: Path, vae: Path, prompt: str, *,
             frames: int = 22, width: int = 864, height: int = 480,
             seed: int = -1, first: Path | None = None,
             last: Path | None = None,
             log: Callable[[str], None] | None = None) -> Path:
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError("sd-cli not found. Run install.bat or ./install.sh.")
    gpu_index = settings.generation_gpu_index(settings.load_prefs())
    output = sdcpp.unique_output("minimax-h3-turbo", "avi")
    cmd = build_command(sd_cli, diffusion, encoder, vae, prompt, output,
                        frames=frames, width=width, height=height, seed=seed,
                        first=first, last=last, gpu_index=gpu_index)
    try:
        sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    except sdcpp.VramError:
        if (width, height) == SIZES[0] or sdcpp.was_cancelled():
            raise
        if log:
            log("GPU memory was insufficient; retrying at 640×352 with the "
                "same prompt, seed and frame count.")
        output.unlink(missing_ok=True)
        cmd = build_command(sd_cli, diffusion, encoder, vae, prompt, output,
                            frames=frames, width=640, height=352, seed=seed,
                            first=first, last=last, gpu_index=gpu_index)
        sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    if not output.is_file() or not output.stat().st_size:
        raise sdcpp.EngineError("sd-cli finished without a video file. Check the log.")
    return encode_mp4(output)
