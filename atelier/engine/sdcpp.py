"""Exécuteur stable-diffusion.cpp : construction et lancement des commandes sd-cli."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .. import settings


class EngineError(RuntimeError):
    pass


@dataclass
class GenRequest:
    diffusion_model: Path
    vae: Path
    text_encoder: Path | None = None
    uncond_model: Path | None = None
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    schedule: str = ""          # vide = laisser le scheduler par défaut du modèle
    width: int = 1024
    height: int = 1024
    seed: int = -1
    batch_count: int = 1
    init_image: Path | None = None
    strength: float = 0.6
    lora_dir: Path | None = None       # --lora-model-dir
    flags: dict[str, bool] = field(default_factory=dict)
    gpu_index: int | None = None


def _flag_args(flags: Mapping[str, bool]) -> list[str]:
    mapping = {
        "diffusion_fa": "--diffusion-fa",
        "offload_to_cpu": "--offload-to-cpu",
        "vae_tiling": "--vae-tiling",
        "clip_on_cpu": "--clip-on-cpu",
        "vae_on_cpu": "--vae-on-cpu",
    }
    return [opt for key, opt in mapping.items() if flags.get(key)]


def _require(*paths: Path | None) -> None:
    for p in paths:
        if p is not None and not Path(p).is_file():
            raise EngineError(
                f"Fichier requis introuvable : {p}\n"
                "Téléchargez le modèle depuis l'onglet Bibliothèque.")


def build_gen_cmd(sd_cli: Path, req: GenRequest, output: Path) -> list[str]:
    _require(req.diffusion_model, req.vae, req.text_encoder,
             req.uncond_model, req.init_image)

    cmd: list[str] = [str(sd_cli), "--mode", "img_gen",
                      "--diffusion-model", str(req.diffusion_model)]
    if req.uncond_model:
        cmd += ["--uncond-diffusion-model", str(req.uncond_model)]
    cmd += ["--vae", str(req.vae)]
    if req.text_encoder:
        cmd += ["--llm", str(req.text_encoder)]

    cmd += ["-p", req.prompt]
    if req.negative and req.cfg_scale > 1.0:
        cmd += ["-n", req.negative]

    cmd += [
        "--cfg-scale", f"{req.cfg_scale}",
        "--steps", f"{req.steps}",
        "--sampling-method", req.sampler,
        "-W", f"{req.width}", "-H", f"{req.height}",
        "-s", f"{req.seed}", "-b", f"{req.batch_count}",
    ]
    if req.schedule:
        cmd += ["--scheduler", req.schedule]
    if req.init_image:
        cmd += ["-i", str(req.init_image), "--strength", f"{req.strength}"]
    if req.lora_dir:
        cmd += ["--lora-model-dir", str(req.lora_dir)]

    cmd += _flag_args(req.flags)
    cmd += ["-o", str(output), "-v"]
    return cmd


def build_upscale_cmd(sd_cli: Path, init_image: Path, upscale_model: Path,
                      output: Path, offload: bool = True) -> list[str]:
    _require(upscale_model, init_image)
    cmd = [str(sd_cli), "--mode", "upscale", "-i", str(init_image),
           "--upscale-model", str(upscale_model)]
    if offload:
        cmd.append("--offload-to-cpu")
    cmd += ["-o", str(output), "-v"]
    return cmd


def run(cmd: list[str], log: Callable[[str], None] | None = None,
        gpu_index: int | None = None) -> None:
    env = None
    if gpu_index is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    if log:
        log("$ " + " ".join(_q(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip("\n"))
    if proc.wait() != 0:
        raise EngineError(f"sd-cli s'est terminé avec le code {proc.returncode}.")


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


def unique_output(prefix: str, ext: str = "png") -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ms = int(time.time() * 1000) % 1000
    return settings.OUTPUT_DIR / f"{prefix}-{stamp}-{ms:03d}.{ext}"


def collect_outputs(output: Path, batch_count: int) -> list[Path]:
    if batch_count <= 1 and output.is_file():
        return [output]
    found = sorted(output.parent.glob(f"{output.stem}_*{output.suffix}"))
    return found or ([output] if output.is_file() else [])
