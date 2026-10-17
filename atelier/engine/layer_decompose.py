"""Isolated layer models and portable RGBA/PSD exports; no Torch in the UI."""
from pathlib import Path
import json
import math
import shutil
import sys
import tempfile
import uuid
import zipfile

from PIL import Image, ImageOps
from .. import addons, settings
from .local_jobs import LocalJob
from .tools import _gen_gpu_index

JOB = LocalJob()
DEFAULTS = {"layers": 4, "steps": 50, "seed": 777, "resolution": 640, "cfg": 4.0,
            "memory": "sequential", "vram": 0, "ram": 0}


def validate(options):
    if set(options) != set(DEFAULTS):
        raise ValueError("Unknown or missing Qwen layer control.")
    limits = {"layers": (2, 8), "steps": (1, 100), "seed": (0, 2**32-1),
              "resolution": (640, 1024), "cfg": (1, 10), "vram": (0, 128), "ram": (0, 1024)}
    for key, (lo, hi) in limits.items():
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"Qwen layers: {key} must be between {lo} and {hi}.")
        if key in {"layers", "steps", "seed", "resolution"} and int(value) != value:
            raise ValueError(f"Qwen layers: {key} must be a whole number.")
    if options["resolution"] not in (640, 1024) or options["memory"] not in {"sequential", "model", "cpu"}:
        raise ValueError("Choose a listed resolution and memory mode.")


def install(name, log=print):
    if name not in {"sam2", "qwen-layered"}:
        raise ValueError("Choose SAM 2 or Qwen-Image-Layered.")
    with JOB.session():
        JOB.run([sys.executable, "-u", settings.ROOT / "scripts/setup_layers.py", name], log,
                env=settings.child_env(_gen_gpu_index()))


def arguments(options):
    merged = {**DEFAULTS, **(options or {})}
    validate(merged)
    result = []
    for key, value in merged.items():
        if key in {"layers", "steps", "seed", "resolution"}:
            value = int(value)
        result += ["--"+key, str(value)]
    return result


def diagnose(options=None, log=print):
    if not addons.ready("qwen-layered"):
        raise RuntimeError("Install Qwen-Image-Layered first.")
    with JOB.session():
        JOB.run([addons.python("qwen-layered"), "-u", settings.ROOT / "scripts/tools/run_qwen_layered.py",
                 "--addon-root", addons.root("qwen-layered"), "--probe", *arguments(options)], log,
                env=settings.child_env(_gen_gpu_index()))


def run(image, options=None, prompt="", log=print):
    opts = arguments(options)
    if not addons.ready("qwen-layered"):
        raise RuntimeError("Install Qwen-Image-Layered first.")
    if image is None:
        raise ValueError("Provide an image to decompose.")
    if len(str(prompt)) > 4000:
        raise ValueError("Keep the image description below 4,000 characters.")
    output = settings.OUTPUT_DIR / ("layers-qwen-"+uuid.uuid4().hex[:12])
    with JOB.session(), tempfile.TemporaryDirectory(prefix="turboslop-qwen-layers-") as temporary:
        work = Path(temporary)
        if isinstance(image, Image.Image):
            source = ImageOps.exif_transpose(image).convert("RGBA")
        else:
            with Image.open(image) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGBA")
        if source.width*source.height > 40_000_000 or max(source.size)/min(source.size) > 4:
            raise ValueError("Use an image up to 40 megapixels and an aspect ratio no greater than 4:1.")
        source.save(work / "source.png")
        (work / "description.txt").write_text(str(prompt).strip(), encoding="utf-8")
        JOB.run([addons.python("qwen-layered"), "-u", settings.ROOT / "scripts/tools/run_qwen_layered.py",
            "--addon-root", addons.root("qwen-layered"), "--input", "source.png", "--output", "export",
            "--prompt-file", "description.txt", *opts], log, cwd=work, env=settings.child_env(_gen_gpu_index()))
        JOB.check()
        for name in ("layers.psd", "layers.zip", "composite.png", "manifest.json"):
            file = work / "export" / name
            if not file.is_file() or not file.stat().st_size:
                raise RuntimeError("Qwen did not create a complete layer export.")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "export"), output)
    return output


def export_layers(images, output, metadata):
    """Preserve generated alpha and layer order, with an honest flattened preview."""
    import numpy as np
    from .psd import write_psd
    if not 2 <= len(images) <= 8:
        raise ValueError("Qwen must return two to eight RGBA layers.")
    if any(image.mode != "RGBA" or image.size != images[0].size for image in images):
        raise ValueError("Qwen returned inconsistent RGBA layer sizes or modes.")
    if not any(image.getchannel("A").getextrema()[1] for image in images):
        raise ValueError("Qwen returned only empty layers.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    composite = Image.new("RGBA", images[0].size)
    layers, entries = [], []
    for number, image in enumerate(images, 1):
        name = f"Layer {number:02d}"
        filename = f"layer-{number:02d}.png"
        image.save(output / filename)
        composite = Image.alpha_composite(composite, image)
        layers.append((name, np.asarray(image)))
        entries.append({"name": name, "file": filename, "order": number})
    composite.save(output / "composite.png")
    white = Image.new("RGBA", composite.size, "white")
    flattened = Image.alpha_composite(white, composite).convert("RGB")
    write_psd(output / "layers.psd", np.asarray(flattened), layers)
    report = {**metadata, "layers": entries, "size": list(composite.size),
              "format": "RGBA raster layers", "order": "background to foreground",
              "warning": "Generative reconstruction: shapes and lettering may change. Compare with the source."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "layers.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file.name != "layers.zip":
                archive.write(file, file.name)
    return output
