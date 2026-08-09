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
    """Annule la génération en cours (process sd-cli)."""
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
    # Convolution directe : réglage DÉLIBÉRÉ de l'utilisateur, pas une
    # déduction matérielle. On l'ajoute après coup pour qu'il survive aussi
    # bien au profil automatique qu'aux presets « 1 clic », qui remplacent le
    # dictionnaire de flags en entier.
    flags["conv_direct_diffusion"] = bool(prefs.get("conv_direct_diffusion"))
    flags["conv_direct_vae"] = bool(prefs.get("conv_direct_vae"))
    return flags, gpu_index


def _apply_loras(prompt: str, loras: list[tuple[str, float]]) -> str:
    """Ajoute la syntaxe <lora:nom:poids> au prompt (consommée par sd.cpp CLI)."""
    tags = "".join(f" <lora:{name}:{weight:g}>" for name, weight in loras if name)
    return (prompt or "") + tags


def _lora_file(name: str) -> Path | None:
    """Résout le fichier LoRA à partir de son nom (sans extension)."""
    for ext in (".safetensors", ".gguf", ".ckpt", ".pt"):
        p = settings.LORA_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    return None


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
    mask_image: Path | None = None,
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
        diffusion = enc = uncond = t5xxl = clip_l = llm_vision = None
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
        # Projecteur vision (mmproj) : chargé UNIQUEMENT quand une image de
        # référence est fournie (édition Krea 2 / Ostris Edit) — inutile en
        # text-to-image pur, et ça évite son coût mémoire.
        llm_vision = _component(model, "text_encoder_vision") if ref_image else None
        # On exige UNIQUEMENT les composants que le modèle déclare : certains
        # modèles peuvent ne pas avoir de VAE, ou utiliser t5xxl/clip_l au lieu
        # de l'encodeur llm. Robuste et sans hypothèse sur l'architecture.
        declared = {c.role for c in model.components}
        need: dict[str, "Path | None"] = {"diffusion": diffusion}
        if "vae" in declared:
            need["vae"] = vae
        if "text_encoder" in declared:
            need["text_encoder"] = enc
        if "t5xxl" in declared:
            need["t5xxl"] = t5xxl
        if "clip_l" in declared:
            need["clip_l"] = clip_l
        absent = [role for role, p in need.items()
                  if p is None or not Path(p).is_file()]
        if absent:
            raise sdcpp.EngineError(
                f"« {model.name} » : fichiers manquants ({', '.join(absent)}). "
                "Téléchargez le modèle (onglet Catalogue de modèles) ou fournissez des "
                "fichiers locaux valides.")

    flags, gpu_index = _resolved_flags(prefs)
    loras = loras or []

    # Multi-GPU. auto-fit répartit tout le modèle sur les GPU visibles (prioritaire) ;
    # sinon split d'encodeur sur un 2e GPU (ex. 1080 Ti). Les deux nécessitent que
    # tous les GPU soient visibles (all_gpus) avec l'ordre CUDA par bus PCI.
    auto_fit = bool(prefs.get("auto_fit"))
    enc_gpu = prefs.get("encoder_gpu_index")
    split_gpu = (not auto_fit) and enc_gpu is not None and enc_gpu != gpu_index
    all_gpus = auto_fit or split_gpu
    if auto_fit:
        # Harmonisation : auto-fit gère lui-même le placement et IGNORE
        # --offload-to-cpu (le forcer en même temps ne fait qu'ajouter de la
        # confusion : sd.cpp met tout en VRAM). On le retire, et on force le VAE
        # tiling pour réduire le pic mémoire du décodage VAE — principale cause
        # d'OOM quand DiT + VAE atterrissent sur la même carte.
        flags = {**flags, "offload_to_cpu": False, "vae_tiling": True}

    # LoRA : via tags <lora:…> dans le prompt + --lora-model-dir (mode sd-cli).
    final_prompt = _apply_loras(prompt, loras)
    lora_dir = settings.LORA_DIR if loras else None

    req = GenRequest(
        diffusion_model=diffusion, vae=vae, model_path=model_path,
        text_encoder=enc, llm_vision=llm_vision,
        t5xxl=t5xxl, clip_l=clip_l, uncond_model=uncond,
        extra_flags=list(model.defaults.get("extra_flags", [])),
        prompt=final_prompt, negative=negative,
        steps=steps, cfg_scale=cfg_scale,
        sampler=sampler or model.defaults.get("sampler", "euler"),
        schedule="" if schedule in (None, "", "auto") else schedule,
        flow_shift=float(flow_shift or 0.0),
        width=width, height=height, seed=seed, batch_count=batch_count,
        init_image=init_image, strength=strength, mask_image=mask_image,
        ref_image=ref_image,
        lora_dir=lora_dir, preview_path=preview_path,
        flags=flags, gpu_index=gpu_index,
        encoder_gpu_index=enc_gpu if split_gpu else None,
        auto_fit=auto_fit, split_mode=prefs.get("split_mode") or "",
        cache_mode=prefs.get("cache_mode") or "",
        cache_option=prefs.get("cache_option") or "",
    )
    out = sdcpp.unique_output(model.family)
    cmd = sdcpp.build_gen_cmd(sd_cli, req, out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index, all_gpus=all_gpus)
    paths = sdcpp.collect_outputs(out, batch_count)

    if save_prompt and paths:
        _write_prompt_sidecars(paths, req, model, int(seed))
    return paths


def upscale_image(image, model_name: str, repeats: int = 1,
                  log: Callable[[str], None] | None = None) -> Path:
    """Agrandissement SIMPLE via un upscaler ESRGAN GGUF (sd.cpp --mode upscale).

    Déterministe, 100% GPU, aucun prompt. `repeats` ré-applique le modèle (un
    modèle ×2 appliqué 2 fois = ×4).

    La taille de tuile du réseau est choisie ici, pas laissée au défaut de
    sd.cpp (128 px) : voir `sdcpp.upscale_tile_size`. Si la tuile élargie ne
    passe pas en VRAM, on relance une fois au défaut plutôt que de rendre une
    erreur — l'image sortira comme avant, pas mieux, mais elle sortira.
    """
    from PIL import Image
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "Binaire sd-cli introuvable. Lancez l'installation (install.bat).")
    model = registry.upscaler_path(model_name)
    if model is None:
        raise sdcpp.EngineError(
            f"Upscaler introuvable : « {model_name} ». Téléchargez les upscalers "
            "depuis l'onglet Toolkit → Agrandir.")

    settings.ensure_dirs()
    src = settings.TMP_DIR / "upscale_in.png"
    im = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) \
        else image.convert("RGB")
    im.save(src)
    w, h = im.size

    _, gpu_index = _resolved_flags(prefs)
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    vram = prof.gpu.vram_gb if prof.gpu else None
    tile = sdcpp.upscale_tile_size(w, h, vram)
    out = sdcpp.unique_output("upscale")
    if log:
        log(f"Upscale ESRGAN « {model_name} » (×{repeats or 1}) sur le GPU…")
        if "--upscale-tile-size" in sdcpp.supported_options(sd_cli):
            log(f"[esrgan] tuiles de {tile} px"
                + (" — image entière en une passe, aucune couture."
                   if tile >= max(w, h) else
                   f" (au lieu de {sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px :"
                   " moins de coutures et d'aliasing)."))
        else:
            log("[esrgan] sd-cli trop ancien pour --upscale-tile-size : tuiles "
                f"de {sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px (coutures possibles). "
                "Mettez le moteur à jour (update-engine.bat).")

    def _cmd(tile_px: int) -> list[str]:
        return sdcpp.build_upscale_cmd(sd_cli, src, model, out,
                                       repeats=int(repeats or 1),
                                       tile_size=tile_px)

    # Les commandes sont construites AVANT le try : une erreur de construction
    # (fichier manquant) n'a rien à voir avec la VRAM et ne doit pas déclencher
    # une seconde tentative sous un message trompeur.
    first, fallback = _cmd(tile), _cmd(sdcpp.SDCPP_DEFAULT_UPSCALE_TILE)
    try:
        sdcpp.run(first, log=log, gpu_index=gpu_index)
    except sdcpp.EngineError:
        # Annulation utilisateur : ne surtout pas relancer.
        if sdcpp.was_cancelled() or first == fallback:
            raise
        if log:
            log(f"[esrgan] échec avec des tuiles de {tile} px (VRAM ?) → "
                f"nouvelle tentative au défaut sd.cpp "
                f"({sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px).")
        sdcpp.run(fallback, log=log, gpu_index=gpu_index)
    if out.is_file():
        return out
    found = sorted(out.parent.glob(f"{out.stem}*{out.suffix}"))
    if found:
        return found[0]
    raise sdcpp.EngineError("L'upscale n'a produit aucune image.")
