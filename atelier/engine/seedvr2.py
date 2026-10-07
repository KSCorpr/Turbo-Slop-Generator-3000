"""SeedVR2 standalone image/video restoration with bounded memory presets."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from .. import addons, settings, seedvr2_acceleration as acceleration
from . import sdcpp, video
from .local_jobs import LocalJob

JOB = LocalJob()
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def source_size(source: Path, log=print) -> tuple[int, int]:
    """Read dimensions without loading a full video or importing OpenCV."""
    if source.suffix.lower() in IMAGES:
        with Image.open(source) as image:
            return image.size
    try:
        result = subprocess.run(
            [video._ffmpeg_exe(log), "-hide_banner", "-nostdin", "-i", str(source),
             "-map", "0:v:0", "-frames:v", "0", "-an", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Cannot read video dimensions: {exc}") from exc
    sizes = re.findall(r"Video:[^\n]*?\b([1-9]\d*)x([1-9]\d*)\b", result.stderr)
    if result.returncode or not sizes:
        raise ValueError("Cannot read video dimensions. Import a playable video again. "
                         + result.stderr[-400:])
    return tuple(map(int, sizes[-1]))


def output_resolution(source: Path, resolution: int, scale: int | None = None,
                      log=print) -> int:
    if scale is not None and (type(scale) is not int or scale not in (2, 4)):
        raise ValueError("Choose ×2, ×4 or a custom short edge.")
    if scale is None:
        try:
            target = int(resolution)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Enter a custom short edge between 256 and 8192 px.") from exc
    else:
        target = min(source_size(source, log)) * scale
    if not 256 <= target <= 8192:
        raise ValueError(f"The requested short edge is {target} px. Choose a custom "
                         "short edge between 256 and 8192 px.")
    return target


def _target_size(size: tuple[int, int], resolution: int, *, even=False) -> tuple[int, int]:
    short = min(size)
    width, height = (max(1, int(edge * resolution / short)) for edge in size)
    if even:  # H.264 yuv420p requires even dimensions.
        width, height = max(2, width // 2 * 2), max(2, height // 2 * 2)
    return width, height


def _image_detail(source: Path, output: Path, strength: float,
                  resolution: int | None = None) -> None:
    """Blend the resized source and AI output; zero skips AI entirely."""
    if resolution is None:
        with Image.open(output) as restored:
            mode = "RGBA" if "A" in restored.getbands() else "RGB"
            detail = restored.convert(mode)
        size = detail.size
    with Image.open(source) as original:
        if resolution is not None:
            mode = "RGBA" if "A" in original.getbands() or "transparency" in original.info else "RGB"
            size = _target_size(original.size, resolution)
        # Match upstream's IMREAD_UNCHANGED pixel orientation, without changing
        # the uploaded image or baking its display-only EXIF preview into it.
        baseline = original.convert(mode).resize(size, Image.Resampling.LANCZOS)
    result = baseline if resolution is not None else Image.blend(baseline, detail, strength / 100)
    result.save(output)  # All input handles are closed before replacing a PNG on Windows.


def _video_detail(source: Path, output: Path, strength: float, log,
                  resolution: int | None = None) -> None:
    ffmpeg = video._ffmpeg_exe(log)
    encoding = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if resolution is not None:
        width, height = _target_size(source_size(source, log), resolution, even=True)
        JOB.run([ffmpeg, "-y", "-nostdin", "-i", source,
                 "-map", "0:v:0", "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
                 *encoding, output], log)
        return
    width, height = source_size(output, log)
    amount = strength / 100
    mixed = output.with_name("restored-detail.mp4")
    filters = (f"[1:v:0]scale={width}:{height}:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[base];"
               "[0:v:0]setsar=1,setpts=PTS-STARTPTS[ai];"
               f"[base][ai]blend=all_expr='A*{1-amount:.6f}+B*{amount:.6f}':shortest=1[v]")
    JOB.run([ffmpeg, "-y", "-nostdin", "-i", output, "-i", source,
             "-filter_complex", filters, "-map", "[v]", *encoding, mixed], log)
    JOB.check()
    mixed.replace(output)


def build_command(source: Path, output: Path, model: str, resolution: int,
                  *, is_video: bool, seed: int = 42, attention: str = "sdpa") -> list[str]:
    if model not in addons.SEED_MODELS or not 256 <= resolution <= 8192:
        raise ValueError("Choose a SeedVR2 model and a short edge between 256 and 8192 px.")
    if attention not in ("sdpa", *acceleration.BACKENDS):
        raise ValueError("Choose SDPA, SageAttention 2 or FlashAttention 2.")
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
            "--attention_mode", attention, "--color_correction", "lab",
            "--input_noise_scale", "0", "--latent_noise_scale", "0",
            *(["--chunk_size", "33", "--temporal_overlap", "1", "--uniform_batch_size",
               "--video_backend", "ffmpeg", "--output_format", "mp4"]
              if is_video else ["--output_format", "png"])]


def select_attention(requested, directory: Path, env, log=print):
    if requested == "sdpa" or (requested == "auto" and not
            (addons.root("seedvr2") / acceleration.MARKER).is_file()):
        log("SeedVR2 attention: SDPA. Optional optimizations can be installed from this tab.")
        return "sdpa"
    report_path = directory / "attention-check.json"
    try:
        log("SeedVR2: checking optional attention kernels on the selected GPU…")
        JOB.run([addons.python("seedvr2"), "-u", Path(acceleration.__file__),
                 "--output", report_path, "--mode", "all" if requested == "auto" else requested],
                log, env=env)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        log("SeedVR2 selected GPU: " + report.get("gpu", {}).get("name", "unavailable"))
        log("SeedVR2 compute precision: " + report.get("compute_dtype", "unavailable"))
        selected = acceleration.choose_mode(requested, report, log)
    except (OSError, ValueError, RuntimeError) as exc:
        JOB.check()  # Cancellation must never start an SDPA inference.
        log(f"SeedVR2 optimization check failed: {exc}. Using SDPA.")
        selected = "sdpa"
    log("SeedVR2 attention: " + selected)
    return selected


def _run_inference(command, output: Path, env: dict, log) -> None:
    """Retry an optional kernel compilation failure once, in a fresh SDPA process."""
    compile_failed = False

    def inference_log(line):
        nonlocal compile_failed
        # LocalJob's short exception tail may contain only upstream's shutdown
        # footer. Remember the compiler diagnostic while forwarding the full log.
        compile_failed |= acceleration.kernel_compile_failed(line)
        log(line)

    try:
        JOB.run(command, inference_log, cwd=addons.root("seedvr2") / "source", env=env)
    except RuntimeError as exc:
        JOB.check()  # Stop must never launch another GPU process.
        attention_index = command.index("--attention_mode") + 1
        if command[attention_index] == "sdpa" or not (
                compile_failed or acceleration.kernel_compile_failed(str(exc))):
            raise
        output.unlink(missing_ok=True)  # Discard incomplete images/videos from the failed process.
        log("SeedVR2: optional attention kernel compilation failed. "
            "Retrying once with SDPA, the same source, seed and restoration settings.")
        fallback = list(command)
        fallback[attention_index] = "sdpa"
        JOB.run(fallback, log, cwd=addons.root("seedvr2") / "source", env=env)


def restore(source, model="3b", resolution=1080, seed=42, log=print,
            *, scale: int | None = None, detail_strength: float = 100, attention: str = "auto"):
    source = Path(source or "")
    if not source.is_file() or source.suffix.lower() not in IMAGES | VIDEOS:
        raise ValueError("Import an image or a video first.")
    strength = float(detail_strength)
    if not math.isfinite(strength) or not 0 <= strength <= 100:
        raise ValueError("Choose an added detail strength between 0 and 100%.")
    if scale is not None and (type(scale) is not int or scale not in (2, 4)):
        raise ValueError("Choose ×2, ×4 or a custom short edge.")
    if attention not in acceleration.MODES:
        raise ValueError("Choose Auto, SDPA, SageAttention 2 or FlashAttention 2.")
    if strength > 0 and not addons.ready("seedvr2"):
        raise RuntimeError("Install SeedVR2 using the button in this tab first.")
    is_video = source.suffix.lower() in VIDEOS
    output = sdcpp.unique_output("seedvr2", "mp4" if is_video else "png")
    # Validate controls before copying a potentially large video.
    build_command(source, output, model, output_resolution(source, resolution) if scale is None else 256,
                  is_video=is_video, seed=int(seed))
    env = acceleration.triton_env(
        settings.child_env(settings.generation_gpu_index(settings.load_prefs())),
        addons.root("seedvr2") / ".venv" / "Lib" / "site-packages")
    # imageio's bundled executable has a versioned name; upstream invokes 'ffmpeg'.
    if is_video and strength > 0:
        binary_dir = addons.root("seedvr2") / "ffmpeg"
        binary_dir.mkdir(parents=True, exist_ok=True)
        binary = binary_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not binary.is_file():
            shutil.copy2(video._ffmpeg_exe(log), binary)
            binary.chmod(binary.stat().st_mode | 0o111)
        env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")
    with JOB.session(), tempfile.TemporaryDirectory(
            prefix="seedvr2-", dir=settings.TMP_DIR) as temporary:
        # OpenCV's Windows imread/imwrite cannot reliably open Unicode names.
        # Keep the user's file untouched and snapshot Gradio's cached upload
        # before the model loads. Copy bytes; never recompress the input image.
        staged_source = Path(temporary) / ("input" + source.suffix.lower())
        staged_output = Path(temporary) / ("restored.mp4" if is_video else "restored.png")
        JOB.check()
        shutil.copyfile(source, staged_source)
        JOB.check()
        log("SeedVR2: using a temporary input with a simple filename.")
        target = output_resolution(staged_source, resolution, scale, log)
        log(f"SeedVR2: {'×' + str(scale) if scale else 'custom size'}, "
            f"short edge {target} px; added detail strength {strength:g}%.")
        if strength == 0:
            log("Simple Lanczos upscale; the SeedVR2 model is not loaded.")
            if is_video:
                _video_detail(staged_source, staged_output, 0, log, resolution=target)
            else:
                _image_detail(staged_source, staged_output, 0, resolution=target)
        else:
            selected = select_attention(attention, Path(temporary), env, log)
            # Our GPU check replaces upstream's generic pip suggestion, which
            # neither targets SeedVR2's Python nor checks Windows/GPU compatibility.
            env["SEEDVR2_OPTIMIZATIONS_LOGGED"] = "1"
            command = build_command(staged_source, staged_output, model, target,
                                    is_video=is_video, seed=int(seed), attention=selected)
            _run_inference(command, staged_output, env, log)
        if not staged_output.is_file() or not staged_output.stat().st_size:
            raise RuntimeError("SeedVR2 finished without an output file.")
        if not is_video:
            with Image.open(staged_output) as image:
                image.verify()
            if 0 < strength < 100:
                JOB.check()
                _image_detail(staged_source, staged_output, strength)
        else:
            if 0 < strength < 100:
                JOB.check()
                _video_detail(staged_source, staged_output, strength, log)
            # Reuse the snapshot for audio too: Gradio may evict the original
            # cached upload while a long restoration is running.
            muxed = Path(temporary) / "restored-audio.mp4"
            JOB.run([video._ffmpeg_exe(log), "-y", "-nostdin", "-i", staged_output,
                     "-i", staged_source, "-map", "0:v:0", "-map", "1:a:0?",
                     "-c:v", "copy", "-c:a", "aac", "-shortest",
                     "-movflags", "+faststart", muxed], log)
            muxed.replace(staged_output)
        JOB.check()
        shutil.move(staged_output, output)
    return output
