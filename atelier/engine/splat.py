"""Photo/video capture -> COLMAP poses -> Brush Gaussian splats."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps
from .. import addons, settings
from . import sdcpp, seedvr2, video
from .local_jobs import LocalJob

JOB = LocalJob()
PROFILES = {
    "object": {"resolution": 1024, "splats": 500000},
    "scene": {"resolution": 1280, "splats": 1000000},
}
QUALITY = {"preview": 5000, "standard": 15000, "fine": 30000}


def prepare_photos(files, destination, check=lambda: None):
    if not files or not 8 <= len(files) <= 300:
        raise ValueError("Import 8–300 overlapping photographs (30–150 recommended).")
    destination.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(files):
        check()
        path = Path(filename)
        if path.suffix.lower() not in seedvr2.IMAGES:
            raise ValueError(f"Unsupported photograph: {path.name}")
        # Normalize names and orientation without changing the original files.
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            image.save(destination / f"frame_{index:05d}.png")


def brush_command(binary, dataset, export, mode, quality):
    if mode not in PROFILES or quality not in QUALITY:
        raise ValueError("Choose an object/scene mode and a quality preset.")
    profile = PROFILES[mode]
    steps = QUALITY[quality]
    return [str(binary), str(dataset), "--total-steps", str(steps),
            "--max-resolution", str(profile["resolution"]),
            "--max-splats", str(profile["splats"]), "--sh-degree", "2",
            "--growth-stop-iter", str(min(steps // 2, 15000)),
            "--export-every", str(steps), "--export-path", str(export),
            "--export-name", "scene.ply"]


def reconstruct(files=None, clip=None, mode="object", quality="standard", fps=1,
                brush_device="", log=print):
    if bool(files) == bool(clip):
        raise ValueError("Import either photographs or one video, then clear the other input.")
    if not addons.ready("splat") or addons.brush() is None:
        raise RuntimeError("Install COLMAP + Brush using the button in this tab first.")
    if mode not in PROFILES or quality not in QUALITY:
        raise ValueError("Choose an object/scene mode and quality.")
    if float(fps) not in (0.5, 1, 2, 4):
        raise ValueError("Choose a listed video sampling rate.")
    device = str(brush_device or "").strip()
    if device and (not device.isdigit() or len(device) > 2):
        raise ValueError("Brush adapter index must be a number, or empty for automatic selection.")
    work = sdcpp.unique_output("capture", "work")
    work.mkdir(parents=True)
    images = work / "images"
    images.mkdir()
    (work / "capture.json").write_text(json.dumps({
        "mode": mode, "quality": quality, "fps": fps,
        "source_count": len(files or []), "video": Path(clip).name if clip else None,
        "brush_version": addons.BRUSH_VERSION, "brush_device": device or "auto",
    }, indent=2), encoding="utf-8")
    with JOB.session():
        if clip:
            source = Path(clip)
            if not source.is_file() or source.suffix.lower() not in seedvr2.VIDEOS:
                raise ValueError("Import a supported video file.")
            log("Extracting up to 300 frames. Keep the subject static and move the camera.")
            JOB.run([video._ffmpeg_exe(log), "-nostdin", "-y", "-i", source,
                     "-vf", f"fps={float(fps)},scale=1600:1600:force_original_aspect_ratio=decrease",
                     "-frames:v", "300", images / "frame_%05d.png"], log)
            count = len(list(images.glob("*.png")))
            if count < 8:
                raise ValueError("Fewer than eight video frames. Use a longer clip or higher sampling rate.")
            if count == 300:
                log("The 300-frame limit was reached; remaining video frames were not used.")
        else:
            prepare_photos(files, images, JOB.check)
        command = [addons.python("splat"), "-u",
                   settings.ROOT / "scripts" / "tools" / "run_colmap.py", work]
        if clip:
            command.append("--sequential")
        JOB.run(command, log)
        export = work / "export"
        export.mkdir()
        env = settings.child_env()
        # WebGPU enumeration is independent of CUDA/nvidia-smi enumeration.
        if device:
            env["CUBECL_DEFAULT_DEVICE"] = device
        log("Training Gaussian splats. Brush reports its chosen graphics adapter below.")
        JOB.run(brush_command(addons.brush(), work / "dataset", export, mode, quality),
                log, env=env)
        result = export / "scene.ply"
        if not result.is_file() or result.stat().st_size < 100:
            raise RuntimeError(f"Brush produced no PLY. Capture and calibration are preserved at {work}.")
        with result.open("rb") as stream:
            header = stream.read(8192)
        if not header.startswith(b"ply") or b"f_dc_0" not in header or b"scale_0" not in header:
            raise RuntimeError("The output is not a Gaussian-splat PLY.")
        log("Finished. PLY retains the full splat data; the viewer is an interactive preview.")
    return result


def saved():
    return [str(p.relative_to(settings.OUTPUT_DIR)) for p in
            sorted(settings.OUTPUT_DIR.glob("capture-*.work/export/scene.ply"),
                   key=lambda p: p.stat().st_mtime, reverse=True)]


def resolve_saved(name):
    path = (settings.OUTPUT_DIR / (name or "")).resolve()
    if not path.is_relative_to(settings.OUTPUT_DIR.resolve()) or name not in saved():
        raise ValueError("Choose a saved reconstruction.")
    return str(path)
