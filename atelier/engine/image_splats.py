"""Cancellable image-to-object generation; isolated upstream neural inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
import secrets
import sys

from PIL import Image, ImageOps

from .. import addons, image_splats as models, settings
from .local_jobs import LocalJob
from . import sdcpp, trellis

JOB = LocalJob()
COUNTS = (32768, 65536, 131072, 262144, 524288, 1048576)


def validate_controls(backend, count, steps, cfg, shift, mode, seed):
    models.artifacts(backend)
    maximum = 262144 if backend == "triposplat" else 1048576
    if count not in COUNTS or count > maximum:
        raise ValueError(f"Choose a Gaussian count up to {maximum:,} for {models.BACKENDS[backend]}.")
    if int(steps) != steps or not 1 <= steps <= 100:
        raise ValueError("Use 1–100 sampling steps.")
    if not all(math.isfinite(float(n)) for n in (cfg, shift)) or not 1 <= cfg <= 10 or not 0.1 <= shift <= 10:
        raise ValueError("Use CFG 1–10 and time shift 0.1–10.")
    if mode not in {"real", "synthetic"}:
        raise ValueError("Choose photograph or rendered / generated input.")
    if int(seed) != seed or not -1 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be -1 or an integer from 0 to 4294967295.")


def install(backend, log=print):
    models.artifacts(backend)
    with JOB.session():
        env = settings.child_env(models.selected_gpu_index())
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        JOB.run([sys.executable, "-u", settings.ROOT / "scripts/setup_image_splats.py", backend], log, env=env)
        if not models.ready(backend):
            raise RuntimeError("The installation is incomplete. See the log and retry Install / repair.")


def validate_ply(path, count):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("The model returned no Gaussian PLY.")
    with path.open("rb") as stream:
        header = stream.read(8192)
    marker = b"end_header\n"
    if marker not in header:
        raise RuntimeError("Invalid Gaussian PLY header.")
    top = header.split(marker, 1)[0]
    if (not top.startswith(b"ply\nformat binary_little_endian 1.0\n")
            or f"element vertex {int(count)}\n".encode() not in top
            or not all(f"property float {name}\n".encode() in top for name in
                       ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"))):
        raise RuntimeError("The output is not the requested 3D Gaussian splat PLY.")
    properties = sum(line.startswith(b"property float ") for line in top.splitlines())
    expected = header.index(marker) + len(marker) + int(count) * properties * 4
    if path.stat().st_size != expected:
        raise RuntimeError("Incomplete Gaussian PLY data.")
    import numpy as np
    with path.open("rb") as stream:
        stream.seek(header.index(marker) + len(marker))
        while chunk := stream.read(1024 * 1024):
            if not np.isfinite(np.frombuffer(chunk, dtype="<f4")).all():
                raise RuntimeError("Non-finite Gaussian values were exported. Try a different seed or fewer Gaussians.")


def generate(source, backend="triposplat", count=131072, steps=20, cfg=3.0, shift=3.0,
             mode="synthetic", seed=42, log=print):
    validate_controls(backend, count, steps, cfg, shift, mode, seed)
    path = Path(source or "")
    if not path.is_file():
        raise ValueError("Import an image first.")
    if not models.ready(backend):
        raise RuntimeError(f"Install / repair {models.BACKENDS[backend]} from Models or this tab first.")
    with JOB.session():
        paths = models.validate(backend, JOB.check)
        if trellis.resident_is_running():
            log(trellis.resident_stop())
        work = sdcpp.unique_output("image-splats", "work")
        work.mkdir(parents=True)
        seed = secrets.randbelow(2**32) if seed == -1 else int(seed)
        with Image.open(path) as original:
            if original.width * original.height > 40_000_000:
                raise ValueError("Crop the source below 40 megapixels first.")
            image = ImageOps.exif_transpose(original)
            if "A" in image.getbands() and image.getchannel("A").getbbox() is None:
                raise ValueError("The image is fully transparent. Import an object with visible pixels.")
            image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB").save(work / "source.png")
        request = dict(backend=backend, count=int(count), steps=int(steps), cfg=float(cfg), shift=float(shift),
                       mode=mode, seed=seed, source=str(work / "source.png"), output=str(work),
                       paths=paths, code_commit=models.SOURCES[backend][1], revisions=models.identity(backend)["revisions"])
        manifest = work / "request.json"
        manifest.write_text(json.dumps(request, indent=2), encoding="utf-8")
        env = settings.child_env(models.selected_gpu_index())
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ATTN_BACKEND="sdpa", PYTHONUNBUFFERED="1")
        JOB.run([addons.python("image-splats"), "-u", settings.ROOT / "scripts/tools/run_image_splats.py",
                 "--backend", backend, "--source", models.source(backend), "--request", manifest], log, env=env)
        JOB.check()
        result = work / "object.ply"
        validate_ply(result, count)
        if not (work / "input.png").is_file():
            raise RuntimeError("The model returned no preprocessed input image.")
        log(f"Finished: {models.BACKENDS[backend]}, {int(count):,} Gaussians, seed {seed}. Export: {result}")
        return result
