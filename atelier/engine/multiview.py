"""Ten sequential image-conditioned views using the existing image engines."""
from __future__ import annotations

import json
import secrets
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .. import registry, settings
from . import generate as gen_engine
from . import panorama, tools


class ViewJob:
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.busy = False

    @contextmanager
    def session(self):
        with self.lock:
            if self.busy:
                raise RuntimeError("A views task is already running.")
            self.busy = True
            self.cancelled.clear()
        try:
            yield
        finally:
            with self.lock:
                self.busy = False

    def check(self):
        if self.cancelled.is_set():
            raise RuntimeError("Cancelled. Completed images have been kept.")

    def cancel(self):
        with self.lock:
            if not self.busy:
                return "No views task running."
            self.cancelled.set()
            gen_engine.cancel()
            tools.cancel()
        return "Stopping. Completed images will be kept."


JOB = ViewJob()


@dataclass
class Result:
    directory: Path
    views: list[tuple[Path, str]] = field(default_factory=list)
    panorama: Path | None = None
    archive: Path | None = None
    status: str = "running"


def conditioning(model) -> str:
    if model.defaults.get("edit") in (True, "full"):
        return "reference"
    if model.defaults.get("supports_img2img", True):
        return "img2img"
    return "description"


def model_note(model_id):
    model = registry.get_base_model(model_id, settings.load_prefs()) if model_id else None
    if not model:
        return "Choose an image model."
    mode = conditioning(model)
    return {
        "reference": "Native image reference. Best option for following view-change instructions.",
        "img2img": "Image-to-image fallback. Large angle changes and subject identity are less reliable.",
        "description": "No image input in this engine. The local Image → prompt module reads the source once; generation follows its text description and can change identity. Install that module below first.",
    }[mode]


def view_prompt(view, mode, description):
    subject = (description or "").strip()
    common = ("Keep the same subject, materials, colours, lighting and visual style "
              "as the reference image. Infer hidden areas consistently. Single image, "
              "no grid, no collage, no text labels. ")
    if mode == "object":
        instruction = ("Render the same object from a different camera angle. "
                       "Orbit the camera around the object's centre, maintaining its "
                       "scale and distance. The reference camera defines yaw 0 degrees. "
                       f"Target camera yaw {view.yaw:g} degrees clockwise, elevation "
                       f"{view.pitch:g} degrees above the horizon, looking at the object. ")
    else:
        instruction = ("Render another perspective direction inside the SAME environment. "
                       "The camera position must stay fixed at the reference camera centre; "
                       "rotate its orientation only, without translation. The reference "
                       f"direction defines yaw 0. Turn right by {view.yaw:g} degrees; "
                       f"look at elevation {view.pitch:g} degrees. Square rectilinear view, "
                       f"{panorama.FOV:g} degree horizontal and vertical field of view, "
                       "zero roll, for an overlapping spherical panorama. ")
        if view.name == "zenith":
            instruction += "Look straight UP at the sky or ceiling, not at the horizon. "
        elif view.name == "nadir":
            instruction += "Look straight DOWN at the ground or floor, not at the horizon. "
    return (subject + "\n" if subject else "") + common + instruction


def _read_image(source):
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(source) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _contact_sheet(result):
    sheet = Image.new("RGB", (5 * 224, 2 * 246), "#20242b")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(result.views):
        x, y = (i % 5) * 224, (i // 5) * 246
        with Image.open(path) as im:
            sheet.paste(ImageOps.fit(im.convert("RGB"), (220, 220)), (x + 2, y + 2))
        draw.text((x + 5, y + 226), label, fill="white")
    sheet.save(result.directory / "contact-sheet.jpg", quality=90)


def _package(result, manifest):
    manifest["status"] = result.status
    manifest["completed"] = len(result.views)
    manifest["panorama"] = result.panorama.name if result.panorama else None
    (result.directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.views:
        _contact_sheet(result)
    archive = result.directory.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(result.directory.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)
    temporary.replace(archive)
    result.archive = archive


def generate_views(source, model_id, *, mode="object", description="", size=768,
                   strength=.8, seed=-1, make_panorama=False, panorama_width=4096,
                   auto_describe=False, log=None, update=None, job=JOB) -> Result:
    if source is None:
        raise ValueError("Import a source image first.")
    if mode not in ("object", "scene"):
        raise ValueError("Choose object orbit or fixed-camera 360 scene.")
    if make_panorama and mode != "scene":
        raise ValueError("A 360 panorama requires fixed-camera scene views; object orbits cannot be converted.")
    size, panorama_width = int(size), int(panorama_width)
    if size not in (512, 768, 1024):
        raise ValueError("Choose 512, 768 or 1024 pixels per view.")
    if make_panorama and panorama_width not in (2048, 4096, 8192):
        raise ValueError("Choose a panorama width of 2048, 4096 or 8192.")
    if not .1 <= float(strength) <= 1:
        raise ValueError("Image-to-image strength must be between 0.1 and 1.")
    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise ValueError("Choose an image model.")
    method = conditioning(model)
    if (auto_describe or method == "description") and not tools.describe_is_installed():
        raise ValueError("Install Image → prompt before using automatic image description or Ming.")
    views = panorama.OBJECT_VIEWS if mode == "object" else panorama.SCENE_VIEWS
    base_seed = int(seed)
    if base_seed < 0:
        base_seed = secrets.randbelow(2**31 - 10)
    if base_seed > 2**31 - 11:
        raise ValueError("Seed must be at most 2147483637 (ten consecutive seeds are used).")
    with job.session():
        settings.ensure_dirs()
        directory = settings.OUTPUT_DIR / f"views-{mode}-{time.time_ns()}"
        directory.mkdir()
        result = Result(directory)
        image = _read_image(source)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        reference = directory / "source.png"
        image.save(reference)
        initial = directory / "img2img-input.png"
        if method == "img2img":
            # Pad rather than crop: preserve the entire source for square outputs.
            ImageOps.pad(image, (size, size), color=(127, 127, 127)).save(initial)
        manifest = {"version": 1, "mode": mode, "model_id": model_id,
                    "conditioning": method, "seed": base_seed, "size": size,
                    "strength": float(strength), "fov": panorama.FOV if mode == "scene" else None,
                    "projection": "yaw 0 = +Z; positive yaw toward +X; pitch up; roll 0",
                    "description": description, "views": [],
                    "limitation": "Generated hidden geometry and camera angles are approximate. Independent views may disagree; this is not measured photogrammetry."}
        d = model.defaults
        sampling = dict(steps=int(d.get("steps", 8)),
                        cfg_scale=float(d.get("cfg_scale", 1)),
                        sampler=d.get("sampler", "euler"),
                        schedule=d.get("scheduler", "auto"),
                        flow_shift=float(d.get("flow_shift", 0)))
        manifest["sampling"] = sampling

        def emit(message):
            job.check()  # also stops a process started in the cancellation race
            if log:
                log(message)

        try:
            job.check()
            if auto_describe or method == "description":
                emit("Reading the source image locally; vision weights are released before generation.")
                detected = tools.image_to_prompt(reference, mode="plain", log=emit)[0]
                description = detected + ("\n" + description.strip() if description.strip() else "")
                manifest["description"] = description
            emit(model_note(model_id))
            for i, view in enumerate(views):
                job.check()
                prompt = view_prompt(view, mode, description)
                emit(f"View {i + 1}/10: {view.name} · seed {base_seed + i}")
                kw = dict(model_id=model_id, prompt=prompt, negative="", width=size,
                          height=size, **sampling, seed=base_seed + i,
                          batch_count=1, log=emit, save_prompt=True)
                if method == "reference":
                    kw["ref_image"] = reference
                elif method == "img2img":
                    kw.update(init_image=initial, strength=float(strength))
                paths = gen_engine.generate(**kw)
                job.check()
                if not paths:
                    raise RuntimeError(f"No image returned for {view.name}.")
                path = directory / f"{i + 1:02d}-{view.name}.png"
                with Image.open(paths[0]) as generated:
                    if generated.size != (size, size):
                        raise RuntimeError("The engine returned a different view size; geometry would be incorrect.")
                    generated.convert("RGB").save(path)
                result.views.append((path, view.name))
                manifest["views"].append({**asdict(view), "file": path.name,
                                          "seed": base_seed + i, "prompt": prompt})
                if update:
                    update(result)
            if make_panorama:
                emit("Reprojecting 10 perspective views to 360×180; blending overlaps.")
                images = [_read_image(p) for p, _ in result.views]
                pano = panorama.equirectangular(images, width=panorama_width, check=job.check)
                job.check()
                result.panorama = directory / "equirectangular-360.png"
                pano.save(result.panorama)
                for im in images:
                    im.close()
            result.status = "completed"
        except Exception as exc:
            result.status = "cancelled" if job.cancelled.is_set() else "failed"
            manifest["error"] = str(exc)
            raise
        finally:
            _package(result, manifest)
            if update:
                update(result)
        return result


def convert_views(files, *, width=4096, log=None, job=JOB) -> Result:
    """Re-convert exported scene images without spending GPU time again."""
    if not files or len(files) != 10:
        raise ValueError("Import the 10 numbered scene PNGs from a previous export.")
    expected = [f"{i + 1:02d}-{v.name}.png" for i, v in enumerate(panorama.SCENE_VIEWS)]
    paths = {Path(p).name: Path(p) for p in files}
    if set(paths) != set(expected):
        raise ValueError("Use 01-yaw-000.png … 08-yaw-315.png, 09-zenith.png and 10-nadir.png. Object views cannot be converted.")
    if int(width) not in (2048, 4096, 8192):
        raise ValueError("Choose 2048, 4096 or 8192 for panorama width.")
    with job.session():
        job.check()
        images = [_read_image(paths[name]) for name in expected]
        if log:
            log("Reprojecting the fixed-camera views; no image generation.")
        pano = panorama.equirectangular(images, width=int(width), check=job.check)
        settings.ensure_dirs()
        directory = settings.OUTPUT_DIR / f"views-converted-{time.time_ns()}"
        directory.mkdir()
        result = Result(directory, panorama=directory / "equirectangular-360.png", status="completed")
        pano.save(result.panorama)
        for im in images:
            im.close()
        return result
