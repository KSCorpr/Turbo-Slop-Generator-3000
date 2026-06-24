"""Exécuteur stable-diffusion.cpp : construction et lancement des commandes sd-cli."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .. import settings


class EngineError(RuntimeError):
    pass


# Registre des process sd-cli en cours, pour pouvoir les annuler.
_ACTIVE: set[subprocess.Popen] = set()
_LOCK = threading.Lock()
_CANCELLED = False


def cancel_active() -> str:
    """Termine tous les process sd-cli en cours (bouton « Annuler »)."""
    global _CANCELLED
    with _LOCK:
        procs = list(_ACTIVE)
    if not procs:
        return "Aucune génération en cours."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Génération annulée."


@dataclass
class GenRequest:
    diffusion_model: Path | None = None   # modèle de diffusion seul (GGUF flow)
    vae: Path | None = None
    model_path: Path | None = None        # checkpoint complet -> -m
    text_encoder: Path | None = None       # --llm (modèles à encodeur LLM)
    t5xxl: Path | None = None              # --t5xxl (FLUX.1, etc.)
    clip_l: Path | None = None             # --clip_l
    uncond_model: Path | None = None
    extra_flags: list[str] = field(default_factory=list)
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    schedule: str = ""          # vide = laisser le scheduler par défaut du modèle
    flow_shift: float = 0.0     # 0 = auto (ne pas passer --flow-shift)
    width: int = 1024
    height: int = 1024
    seed: int = -1
    batch_count: int = 1
    init_image: Path | None = None     # img2img classique (-i + --strength)
    strength: float = 0.6
    ref_image: Path | None = None      # édition (-r / --ref-image, Flux.2 / PiD)
    vae_format: str = ""               # --vae-format (ex. "flux" pour PiD)
    rng: str = ""                      # --rng (ex. "cpu" pour PiD)
    lora_dir: Path | None = None       # --lora-model-dir
    preview_path: Path | None = None   # aperçu temps réel (--preview proj)
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
    _require(req.model_path, req.diffusion_model, req.vae, req.text_encoder,
             req.t5xxl, req.clip_l, req.uncond_model, req.init_image, req.ref_image)

    cmd: list[str] = [str(sd_cli), "--mode", "img_gen"]
    if req.model_path:
        # Checkpoint complet : CLIP + VAE inclus.
        cmd += ["-m", str(req.model_path)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
    else:
        cmd += ["--diffusion-model", str(req.diffusion_model)]
        if req.uncond_model:
            cmd += ["--uncond-diffusion-model", str(req.uncond_model)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
        if req.text_encoder:
            cmd += ["--llm", str(req.text_encoder)]
        if req.t5xxl:
            cmd += ["--t5xxl", str(req.t5xxl)]
        if req.clip_l:
            cmd += ["--clip_l", str(req.clip_l)]

    if req.vae_format:
        cmd += ["--vae-format", req.vae_format]
    cmd += list(req.extra_flags)
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
    if req.rng:
        cmd += ["--rng", req.rng]
    if req.flow_shift and req.flow_shift > 0:
        cmd += ["--flow-shift", f"{req.flow_shift}"]
    if req.init_image:
        cmd += ["-i", str(req.init_image), "--strength", f"{req.strength}"]
    if req.ref_image:
        # Édition d'image (Flux.2 / Kontext) : pilotée par le prompt, sans strength.
        cmd += ["-r", str(req.ref_image)]
    if req.lora_dir:
        cmd += ["--lora-model-dir", str(req.lora_dir)]

    if req.preview_path:
        cmd += ["--preview", "proj", "--preview-path", str(req.preview_path),
                "--preview-interval", "1"]
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
    global _CANCELLED
    env = None
    if gpu_index is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    if log:
        log("$ " + " ".join(_q(c) for c in cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
                            encoding="utf-8", errors="replace")
    with _LOCK:
        _ACTIVE.add(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log:
                log(line.rstrip("\n"))
        code = proc.wait()
    finally:
        with _LOCK:
            _ACTIVE.discard(proc)
    if _CANCELLED:
        raise EngineError("Interrompu par l'utilisateur.")
    if code != 0:
        raise EngineError(f"sd-cli s'est terminé avec le code {code}.")


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
