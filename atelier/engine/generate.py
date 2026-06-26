"""Pipeline de génération : assemble un GenRequest depuis la bibliothèque, les
préférences matérielles et les LoRA, puis lance stable-diffusion.cpp.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import hardware, registry, settings
from . import sdcpp
from .sdcpp import GenRequest


def cancel() -> str:
    """Annule la génération en cours (termine le process sd-cli)."""
    return sdcpp.cancel_active()


def list_custom_models() -> list[str]:
    """Noms des fichiers de modèle déposés dans models/custom/ (téléchargés
    manuellement ailleurs)."""
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.CUSTOM_DIR.glob("*")):
        if p.suffix.lower() in (".gguf", ".safetensors", ".sft", ".pth", ".ckpt"):
            out.append(p.name)
    return out


def custom_path(name: str | None) -> Path | None:
    if not name:
        return None
    p = settings.CUSTOM_DIR / name
    return p if p.is_file() else None


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


def _write_prompt_sidecars(paths: list[Path], req: "GenRequest",
                           model: "registry.BaseModel", base_seed: int) -> None:
    """Écrit un .txt (style A1111) à côté de chaque image, pour retrouver le
    prompt et les réglages directement dans le dossier outputs/."""
    import time
    date = time.strftime("%Y-%m-%d %H:%M:%S")
    for i, p in enumerate(paths):
        seed = base_seed + i if (base_seed is not None and base_seed >= 0) \
            else base_seed
        lines = [req.prompt or ""]
        if req.negative:
            lines.append(f"Negative prompt: {req.negative}")
        lines.append(f"Model: {model.name} ({model.id})")
        lines.append(
            f"Steps: {req.steps}, CFG: {req.cfg_scale}, Sampler: {req.sampler}, "
            f"Scheduler: {req.schedule or 'auto'}, Size: {req.width}x{req.height}, "
            f"Seed: {seed}, Flow shift: {req.flow_shift:g}")
        if req.init_image:
            lines.append(f"img2img strength: {req.strength}")
        lines.append(f"Date: {date}")
        try:
            Path(p).with_suffix(".txt").write_text("\n".join(lines),
                                                   encoding="utf-8")
        except OSError:
            pass


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
    flow_shift: float = 0.0,
    init_image: Path | None = None,
    strength: float = 0.6,
    ref_image: "Path | list[Path] | None" = None,
    loras: list[tuple[str, float]] | None = None,
    diffusion_override: Path | None = None,
    vae_override: Path | None = None,
    encoder_override: Path | None = None,
    preview_path: Path | None = None,
    save_prompt: bool = True,
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

    # Famille « checkpoint complet » : un seul fichier via -m.
    has_full = any(c.role == "model" for c in model.components)
    if has_full:
        model_path = Path(diffusion_override) if diffusion_override \
            else _component(model, "model")
        vae = Path(vae_override) if vae_override else _component(model, "vae")
        diffusion = enc = uncond = t5xxl = clip_l = None
        if model_path is None or not Path(model_path).is_file():
            raise sdcpp.EngineError(
                f"« {model.name} » : checkpoint manquant. Téléchargez-le "
                "(onglet Catalogue de modèles) ou fournissez un fichier local.")
    else:
        model_path = None
        diffusion = Path(diffusion_override) if diffusion_override else _component(model, "diffusion")
        vae = Path(vae_override) if vae_override else _component(model, "vae")
        enc = Path(encoder_override) if encoder_override else _component(model, "text_encoder")
        uncond = _component(model, "uncond")
        t5xxl = _component(model, "t5xxl")
        clip_l = _component(model, "clip_l")
        # Composants de texte requis : au moins un encodeur (llm OU t5xxl).
        need = {"diffusion": diffusion, "vae": vae}
        if not t5xxl:
            need["text_encoder"] = enc
        absent = [role for role, p in need.items() if p is None or not Path(p).is_file()]
        if t5xxl is not None and not Path(t5xxl).is_file():
            absent.append("t5xxl")
        if absent:
            raise sdcpp.EngineError(
                f"« {model.name} » : fichiers manquants ({', '.join(absent)}). "
                "Téléchargez le modèle (onglet Catalogue de modèles) ou fournissez des "
                "fichiers locaux valides.")

    flags, gpu_index = _resolved_flags(prefs)
    lora_dir = settings.LORA_DIR if loras else None
    final_prompt = _apply_loras(prompt, loras or [])

    req = GenRequest(
        diffusion_model=diffusion, vae=vae, model_path=model_path,
        text_encoder=enc, t5xxl=t5xxl, clip_l=clip_l, uncond_model=uncond,
        extra_flags=list(model.defaults.get("extra_flags", [])),
        prompt=final_prompt, negative=negative,
        steps=steps, cfg_scale=cfg_scale,
        sampler=sampler or model.defaults.get("sampler", "euler"),
        schedule="" if schedule in (None, "", "auto") else schedule,
        flow_shift=float(flow_shift or 0.0),
        width=width, height=height, seed=seed, batch_count=batch_count,
        init_image=init_image, strength=strength, ref_image=ref_image,
        lora_dir=lora_dir, preview_path=preview_path,
        flags=flags, gpu_index=gpu_index,
    )
    out = sdcpp.unique_output(model.family)
    cmd = sdcpp.build_gen_cmd(sd_cli, req, out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    paths = sdcpp.collect_outputs(out, batch_count)
    if save_prompt and paths:
        _write_prompt_sidecars(paths, req, model, int(seed))
    return paths


def generate_video(prompt: str, negative: str = "", mode: str = "t2v",
                   init_image: Path | None = None, end_image: Path | None = None,
                   width: int = 1280, height: int = 720, frames: int = 33,
                   fps: int = 24, cfg_scale: float = 6.0, steps: int = 30,
                   log: Callable[[str], None] | None = None) -> Path:
    """Génère une vidéo via LTX-2.3 (sd.cpp -M vid_gen). mode : t2v|i2v|flf2v."""
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "Binaire sd-cli introuvable. Lancez l'installation (install.bat).")
    cfg = registry.ltx_config()
    paths = registry.ltx_paths(prefs)
    need = ("diffusion", "vae", "audio_vae", "embeddings_connectors", "llm")
    missing = [r for r in need
               if not (paths.get(r) and Path(paths[r]).is_file())]
    if missing:
        raise sdcpp.EngineError(
            f"LTX-2.3 non installé (composants manquants : {', '.join(missing)}). "
            "Installez-le depuis l'onglet Vidéo.")

    settings.ensure_dirs()
    flags, gpu_index = _resolved_flags(prefs)
    out = sdcpp.unique_output("ltx", ext="webm")
    cmd = sdcpp.build_vid_cmd(
        sd_cli, diffusion=paths["diffusion"], vae=paths["vae"],
        audio_vae=paths["audio_vae"], llm=paths["llm"],
        connectors=paths["embeddings_connectors"],
        prompt=prompt, negative=negative or cfg.get("negative", ""),
        cfg_scale=float(cfg_scale), steps=int(steps),
        width=int(width), height=int(height), frames=int(frames), fps=int(fps),
        mode=mode, init_image=init_image, end_image=end_image,
        flags=flags, output=out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    if out.is_file():
        return out
    found = sorted(out.parent.glob(f"{out.stem}*.webm")) or \
        sorted(out.parent.glob(f"{out.stem}*.mp4"))
    if found:
        return found[0]
    raise sdcpp.EngineError("LTX-2.3 n'a produit aucune vidéo.")

