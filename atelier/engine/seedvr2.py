"""SeedVR2 standalone image/video restoration with bounded memory presets."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image
from .. import addons, settings
from . import sdcpp, video
from .local_jobs import LocalJob

JOB = LocalJob()
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_command(source: Path, output: Path, model: str, resolution: int,
                  *, is_video: bool, seed: int = 42) -> list[str]:
    if model not in addons.SEED_MODELS or not 256 <= resolution <= 8192:
        raise ValueError("Choose a SeedVR2 model and a short edge between 256 and 8192 px.")
    return [str(addons.python("seedvr2")), "-u",
            str(addons.root("seedvr2") / "source" / "inference_cli.py"),
            str(source), "--output", str(output),
            "--model_dir", str(addons.root("seedvr2") / "models"),
            "--dit_model", addons.SEED_MODELS[model],
            "--resolution", str(resolution), "--seed", str(seed),
            "--batch_size", "5" if is_video else "1",
            "--blocks_to_swap", "32" if model == "3b" else "36",
            "--swap_io_components", "--dit_offload_device", "cpu",
            "--vae_offload_device", "cpu", "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", "512", "--vae_decode_tile_size", "512",
            "--attention_mode", "sdpa", "--color_correction", "lab",
            "--input_noise_scale", "0", "--latent_noise_scale", "0",
            *(["--chunk_size", "33", "--temporal_overlap", "1", "--uniform_batch_size",
               "--video_backend", "ffmpeg", "--output_format", "mp4"]
              if is_video else ["--output_format", "png"])]


def restore(source, model="3b", resolution=1080, seed=42, log=print):
    source = Path(source or "")
    if not source.is_file() or source.suffix.lower() not in IMAGES | VIDEOS:
        raise ValueError("Import an image or a video first.")
    if not addons.ready("seedvr2"):
        raise RuntimeError("Install SeedVR2 using the button in this tab first.")
    is_video = source.suffix.lower() in VIDEOS
    output = sdcpp.unique_output("seedvr2", "mp4" if is_video else "png")
    command = build_command(source, output, model, int(resolution),
                            is_video=is_video, seed=int(seed))
    env = settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
    # imageio's bundled executable has a versioned name; upstream invokes 'ffmpeg'.
    if is_video:
        binary_dir = addons.root("seedvr2") / "ffmpeg"
        binary_dir.mkdir(parents=True, exist_ok=True)
        binary = binary_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not binary.is_file():
            shutil.copy2(video._ffmpeg_exe(log), binary)
            binary.chmod(binary.stat().st_mode | 0o111)
        env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")
    with JOB.session():
        JOB.run(command, log, cwd=addons.root("seedvr2") / "source", env=env)
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("SeedVR2 finished without an output file.")
        if not is_video:
            with Image.open(output) as image:
                image.verify()
        else:
            # Upstream writes video only. Reattach optional source audio without
            # re-encoding the restored frames; AAC keeps browser playback portable.
            muxed = output.with_name(output.stem + "-audio.mp4")
            JOB.run([video._ffmpeg_exe(log), "-y", "-nostdin", "-i", output,
                     "-i", source, "-map", "0:v:0", "-map", "1:a:0?",
                     "-c:v", "copy", "-c:a", "aac", "-shortest",
                     "-movflags", "+faststart", muxed], log)
            muxed.replace(output)
    return output
