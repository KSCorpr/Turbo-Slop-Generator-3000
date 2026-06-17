"""Pipeline de génération : assemble un GenRequest depuis la bibliothèque, les
préférences matérielles et les LoRA, puis lance stable-diffusion.cpp.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from .. import hardware, registry, settings
from . import sdcpp
from .sdcpp import GenRequest


def list_loras() -> list[str]:
    """Noms des LoRA disponibles dans loras/ (sans extension)."""
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.LORA_DIR.glob("*")):
        if p.suffix.lower() in (".safetensors", ".gguf", ".ckpt", ".pt"):
            out.append(p.stem)
    return out


def _component(model: registry.BaseModel, role: str) -> Path | None:
    comp = next((c for c in model.components if c.role == role), None)
    if comp is None:
        return None
    return registry.resolve_component_path(comp)


def _resolved_flags(prefs: dict) -> tuple[dict[str, bool], int | None]:
    """Flags d'optimisation effectifs + index GPU."""
    if prefs.get("auto_optimize", True):
        prof = hardware.auto_profile(prefs.get("gpu_index"))
        flags = prof.flags()
        gpu_index = prof.gpu.index if prof.gpu else None
    else:
        flags = dict(prefs.get("flags", {}))
        gpu_index = prefs.get("gpu_index")
    return flags, gpu_index


def _apply_loras(prompt: str, loras: list[tuple[str, float]]) -> str:
    """Ajoute la syntaxe <lora:nom:poids> au prompt (consommée par sd.cpp)."""
    tags = "".join(f" <lora:{name}:{weight:g}>" for name, weight in loras if name)
    return (prompt or "") + tags


def generate(
    model_id: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    batch_count: int,
    sampler: str | None = None,
    schedule: str = "auto",
    init_image: Path | None = None,
    strength: float = 0.6,
    loras: list[tuple[str, float]] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "Binaire sd-cli introuvable. Lancez l'installation "
            "(install.bat) ou « python scripts/get_sdcpp.py ».")

    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Modèle inconnu : {model_id}")
    missing = registry.missing_components(model)
    if missing:
        roles = ", ".join(c.role for c in missing)
        raise sdcpp.EngineError(
            f"« {model.name} » incomplet (manque : {roles}). "
            "Téléchargez-le depuis l'onglet Bibliothèque.")

    diffusion = _component(model, "diffusion")
    vae = _component(model, "vae")
    enc = _component(model, "text_encoder")
    uncond = _component(model, "uncond")

    flags, gpu_index = _resolved_flags(prefs)
    lora_dir = settings.LORA_DIR if loras else None
    final_prompt = _apply_loras(prompt, loras or [])

    req = GenRequest(
        diffusion_model=diffusion, vae=vae, text_encoder=enc, uncond_model=uncond,
        prompt=final_prompt, negative=negative,
        steps=steps, cfg_scale=cfg_scale,
        sampler=sampler or model.defaults.get("sampler", "euler"),
        schedule="" if schedule in (None, "", "auto") else schedule,
        width=width, height=height, seed=seed, batch_count=batch_count,
        init_image=init_image, strength=strength,
        lora_dir=lora_dir, flags=flags, gpu_index=gpu_index,
    )
    out = sdcpp.unique_output(model.family)
    cmd = sdcpp.build_gen_cmd(sd_cli, req, out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    return sdcpp.collect_outputs(out, batch_count)
