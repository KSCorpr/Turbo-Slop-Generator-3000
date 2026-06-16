"""Upscale par diffusion : SeedVR2-3B et NVIDIA PiD (PyTorch, à la demande).

Chaque upscaler est installé séparément (scripts/setup_upscalers.py) : dépôt
cloné sous upscalers_repo/<id> et poids sous upscalers_repo/<id>/ckpts. On lance
ensuite un script runner dédié (scripts/upscalers/run_<id>.py) en sous-process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import hardware, settings


class UpscaleError(RuntimeError):
    pass


_RUNNERS = {
    "seedvr2": "run_seedvr2.py",
    "nvidia-pid": "run_pid.py",
}


def is_installed(upscaler_id: str) -> bool:
    return (settings.UPSCALERS_REPO_DIR / upscaler_id).is_dir()


def _gpu_index() -> int | None:
    prefs = settings.load_prefs()
    if prefs.get("gpu_index") is not None:
        return prefs["gpu_index"]
    prof = hardware.auto_profile()
    return prof.gpu.index if prof.gpu else None


def upscale(
    upscaler_id: str,
    image: Image.Image | str | Path,
    scale: int = 4,
    log: Callable[[str], None] | None = None,
) -> Path:
    if upscaler_id not in _RUNNERS:
        raise UpscaleError(f"Upscaler inconnu : {upscaler_id}")
    if not is_installed(upscaler_id):
        raise UpscaleError(
            f"« {upscaler_id} » n'est pas installé.\n"
            f"Lancez : python scripts/setup_upscalers.py {upscaler_id}")

    settings.ensure_dirs()
    # Normalise l'entrée vers un PNG sur disque.
    if isinstance(image, (str, Path)):
        src = Path(image)
    else:
        src = settings.TMP_DIR / f"upscale_src_{int(time.time()*1000)}.png"
        image.save(src)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"upscale_out_{upscaler_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = settings.ROOT / "scripts" / "upscalers" / _RUNNERS[upscaler_id]
    repo_dir = settings.UPSCALERS_REPO_DIR / upscaler_id
    cmd = [sys.executable, str(runner),
           "--repo-dir", str(repo_dir),
           "--input", str(src),
           "--output-dir", str(out_dir),
           "--scale", str(scale)]

    env = dict(os.environ)
    gi = _gpu_index()
    if gi is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gi)

    if log:
        log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip("\n"))
    if proc.wait() != 0:
        raise UpscaleError(f"{upscaler_id} a échoué (voir le journal).")

    produced = sorted(p for p in out_dir.rglob("*")
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not produced:
        raise UpscaleError(f"{upscaler_id} n'a produit aucune image.")

    final = settings.OUTPUT_DIR / f"upscale-{upscaler_id}-{stamp}.png"
    Image.open(produced[0]).save(final)
    return final
