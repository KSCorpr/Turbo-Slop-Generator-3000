"""SVG jobs stay outside the Gradio process and release all models on exit."""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .. import addons, settings
from .local_jobs import LocalJob
from .tools import _gen_gpu_index

JOB = LocalJob()
MODES = {"trace": "svg", "layer4": "svg", "layer9": "svg",
         "hivg-image": "hivg", "hivg-text": "hivg"}


def install(name, log=print):
    if name not in {"svg", "hivg"}:
        raise ValueError("Unknown SVG engine.")
    with JOB.session():
        JOB.run([sys.executable, "-u", settings.ROOT / "scripts/setup_svg.py", name], log,
                env=settings.child_env(_gen_gpu_index()))


def run(mode, source, prompt, tokens=2048, seed=0, threshold=5, log=print):
    if mode not in MODES:
        raise ValueError("Choose a listed SVG mode.")
    name = MODES[mode]
    if not addons.ready(name):
        raise RuntimeError("Install the selected SVG engine first.")
    if mode != "hivg-text" and (not source or not Path(source).is_file()):
        raise ValueError("Import an image first.")
    prompt = str(prompt or "").strip()
    if mode == "hivg-text" and not prompt:
        raise ValueError("Enter a description first.")
    if len(prompt) > 4000:
        raise ValueError("Use a description of at most 4000 characters.")
    values = (tokens, seed, threshold)
    if any(not math.isfinite(float(v)) or float(v) != int(v) for v in values):
        raise ValueError("Token budget, seed and threshold must be whole numbers.")
    tokens, seed, threshold = map(int, values)
    if not (256 <= tokens <= 4096 and 0 <= seed <= 2**32-1 and 1 <= threshold <= 50):
        raise ValueError("SVG options are outside their allowed ranges.")
    output = settings.OUTPUT_DIR / ("svg-" + uuid.uuid4().hex[:12])
    with JOB.session(), tempfile.TemporaryDirectory(prefix="turboslop-svg-") as temporary:
        work = Path(temporary)
        if mode != "hivg-text":
            with Image.open(source) as uploaded:
                image = ImageOps.exif_transpose(uploaded)
                if image.width * image.height > 16_000_000:
                    raise ValueError("Use an image up to 16 megapixels.")
                if mode == "trace":
                    image = image.convert("RGBA")
                else:
                    rgba = image.convert("RGBA")
                    image = Image.new("RGB", rgba.size, "white")
                    image.paste(rgba, mask=rgba.getchannel("A"))
                image.save(work / "source.png")
        script = "run_hivg.py" if name == "hivg" else "run_svg.py"
        command = [addons.python(name), "-u", settings.ROOT / "scripts/tools" / script,
            "--mode", mode, "--output", "export", "--addon-root", addons.root(name)]
        if mode != "hivg-text":
            command += ["--source", "source.png"]
        if name == "hivg":
            command += ["--prompt", prompt, "--tokens", str(tokens), "--seed", str(seed)]
        else:
            command += ["--threshold", str(threshold)]
        JOB.run(command, log, cwd=work, env=settings.child_env(_gen_gpu_index()))
        JOB.check()
        for filename in ("result.svg", "preview.png", "vectors.zip", "manifest.json"):
            file = work / "export" / filename
            if not file.is_file() or not file.stat().st_size:
                raise RuntimeError("The engine did not create a complete SVG export.")
        # Only complete results enter persistent history. Temporary offload tensors are removed.
        offload = work / "export/offload"
        if offload.exists():
            shutil.rmtree(offload)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "export"), output)
    return output
