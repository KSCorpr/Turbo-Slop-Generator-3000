"""SVG jobs stay outside the Gradio process and release all models on exit."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
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
         "hivg-image": "hivg", "hivg-text": "hivg",
         "omnisvg-image": "omnisvg", "omnisvg-text": "omnisvg"}
OMNI_DEFAULTS = {"memory": "auto", "vram": 0, "ram": 0, "temperature": .3, "top_p": .9,
                 "top_k": 50, "repetition": 1.05, "variants": 1, "timeout": 300,
                 "cache": True, "keep_aspect": True}


class SVGDiagnosticError(RuntimeError):
    def __init__(self, message, directory):
        super().__init__(message)
        self.directory = Path(directory)


def omni_arguments(options=None, *, tokens=2048, seed=0):
    from scripts.tools.run_omnisvg import validate_options
    options = {**OMNI_DEFAULTS, **(options or {})}
    if set(options) != set(OMNI_DEFAULTS):
        raise ValueError("Unknown OmniSVG control.")
    if any(not isinstance(options[key], bool) for key in ("cache", "keep_aspect")):
        raise ValueError("Cache and image proportions controls must be on or off.")
    validate_options(SimpleNamespace(**options, tokens=tokens, seed=seed))
    result = []
    for name, value in options.items():
        if isinstance(value, bool):
            result += ["--" + ("" if value else "no-") + name.replace("_", "-")]
        else:
            if name in {"variants", "top_k", "timeout"}:
                value = int(value)
            result += ["--" + name.replace("_", "-"), str(value)]
    return result


def diagnose(options=None, log=print):
    if not addons.ready("omnisvg"):
        raise RuntimeError("Install OmniSVG first, then run the memory check.")
    arguments = omni_arguments(options)
    with JOB.session():
        JOB.run([addons.python("omnisvg"), "-u", settings.ROOT / "scripts/tools/run_omnisvg.py",
            "--addon-root", addons.root("omnisvg"), "--probe", *arguments], log,
            env=settings.child_env(_gen_gpu_index()))


def install(name, log=print):
    if name not in {"svg", "hivg", "omnisvg"}:
        raise ValueError("Unknown SVG engine.")
    with JOB.session():
        JOB.run([sys.executable, "-u", settings.ROOT / "scripts/setup_svg.py", name], log,
                env=settings.child_env(_gen_gpu_index()))


def run(mode, source, prompt, tokens=None, seed=0, threshold=5, log=print, *, omni=None):
    if mode not in MODES:
        raise ValueError("Choose a listed SVG mode.")
    name = MODES[mode]
    if not addons.ready(name):
        raise RuntimeError("Install the selected SVG engine first.")
    text_mode = mode in {"hivg-text", "omnisvg-text"}
    if not text_mode and (not source or not Path(source).is_file()):
        raise ValueError("Import an image first.")
    prompt = str(prompt or "").strip()
    if text_mode and not prompt:
        raise ValueError("Enter a description first.")
    if len(prompt) > 4000:
        raise ValueError("Use a description of at most 4000 characters.")
    if tokens is None:
        tokens = 4096 if name == "hivg" else 2048
    values = (tokens, seed, threshold)
    if any(not math.isfinite(float(v)) or float(v) != int(v) for v in values):
        raise ValueError("Token budget, seed and threshold must be whole numbers.")
    tokens, seed, threshold = map(int, values)
    token_limit = 16384 if name == "hivg" else 4096
    if not (256 <= tokens <= token_limit and 0 <= seed <= 2**32-1 and 1 <= threshold <= 50):
        raise ValueError("SVG options are outside their allowed ranges.")
    extra = omni_arguments(omni, tokens=tokens, seed=seed) if name == "omnisvg" else []
    output = settings.OUTPUT_DIR / ("svg-" + uuid.uuid4().hex[:12])
    diagnostic = output.with_name("svg-diagnostic-" + output.name.removeprefix("svg-"))
    with JOB.session(), tempfile.TemporaryDirectory(prefix="turboslop-svg-") as temporary:
        work = Path(temporary)
        if not text_mode:
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
        script = {"hivg": "run_hivg.py", "omnisvg": "run_omnisvg.py"}.get(name, "run_svg.py")
        command = [addons.python(name), "-u", settings.ROOT / "scripts/tools" / script,
            "--mode", mode, "--output", "export", "--addon-root", addons.root(name)]
        if not text_mode:
            command += ["--source", "source.png"]
        if name in {"hivg", "omnisvg"}:
            command += ["--prompt", prompt, "--tokens", str(tokens), "--seed", str(seed)]
            command += extra
            if name == "hivg":
                command += ["--diagnostics-dir", str(diagnostic)]
        else:
            command += ["--threshold", str(threshold)]
        try:
            JOB.run(command, log, cwd=work, env=settings.child_env(_gen_gpu_index()))
        except RuntimeError as exc:
            if name == "hivg" and not JOB.cancelled.is_set() and (diagnostic / "diagnostics.zip").is_file():
                report = json.loads((diagnostic / "diagnostic.json").read_text(encoding="utf-8"))
                raise SVGDiagnosticError(report["error"], diagnostic) from exc
            raise
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
