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
    """Annule la génération en cours."""
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


_SLOW_ENCODER_GPU = (
    "ℹ️ Text encoder moved back onto the generation GPU: the secondary card "
    "has no tensor cores and runs fp16 at a fraction of its speed (Pascal: "
    "1/64). It remains perfect for STORING weights, not for computing on "
    "them.")


# Réserve creusée par la reprise après OOM sous auto-fit. 3 Gio et non 1 : le
# but n'est pas de frôler la limite une seconde fois mais de faire basculer un
# module entier hors de la carte, et le module le plus petit (le VAE) pèse déjà
# plus d'un gigaoctet une fois son espace de calcul compté.
_OOM_RELIEF_SPARE_GIB = 3.0


def _te_on_cpu(params_backend: str) -> str:
    """La même résidence, mais avec l'encodeur en RAM.

    Utilisé par la reprise après OOM. On réécrit le champ `te=` au lieu de le
    concaténer : `"…,te=cuda1,te=cpu"` laisserait sd.cpp arbitrer selon son
    ordre de parsing, ce qui est précisément le genre de dépendance qu'on
    évite partout ailleurs.
    """
    if not params_backend:
        return ""
    parts = [p for p in params_backend.split(",") if not p.startswith("te=")]
    return ",".join([*parts, "te=cpu"])


def encoder_gpu_too_slow(enc_index: int | None,
                         gen_index: int | None) -> bool:
    """La 2e carte est-elle un mauvais endroit pour CALCULER l'encodeur ?

    Le partage d'encodeur a été pensé comme du stockage : la carte secondaire
    tient des poids que la principale n'a plus la place de garder. Mais elle
    les calcule aussi — et sur une Pascal grand public le fp16 tourne à 1/64
    de la vitesse fp32, ce qui transformait un encodage d'une seconde en
    trente-huit. On suit les TENSOR CORES, pas les noms de cartes : c'est le
    même critère que pour la flash-attention, et il ne se trompe pas sur les
    GTX 16xx.
    """
    if enc_index is None or gen_index is None or enc_index == gen_index:
        return False
    gpus = {g.index: g for g in hardware.detect_gpus()}
    enc, gen = gpus.get(enc_index), gpus.get(gen_index)
    if enc is None or gen is None:
        return False
    return gen.tensor_cores and not enc.tensor_cores


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


def resolve_model_files(model: "registry.BaseModel",
                        diffusion_override=None, vae_override=None,
                        encoder_override=None, want_vision: bool = False
                        ) -> dict:
    """Chemins des fichiers d'UN modèle du catalogue, ou une erreur lisible.

    Extrait de `generate()` pour être partagé avec le mode ADetailer : les
    deux ont besoin exactement du même modèle, résolu exactement pareil, et
    une seconde copie de cette logique dériverait — c'est déjà ce qui était
    arrivé entre la ligne de commande et le serveur résident.
    """
    # Famille « checkpoint complet » : un seul fichier via -m.
    has_full = any(c.role == "model" for c in model.components)
    if has_full:
        model_path = Path(diffusion_override) if diffusion_override \
            else _component(model, "model")
        vae = Path(vae_override) if vae_override else _component(model, "vae")
        diffusion = enc = uncond = t5xxl = clip_l = llm_vision = None
        if model_path is None or not Path(model_path).is_file():
            raise sdcpp.EngineError(
                f"“{model.name}”: checkpoint missing. Download it "
                "(Model catalog tab) or supply a local file.")
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
        llm_vision = _component(model, "text_encoder_vision") if want_vision else None
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
                f"“{model.name}”: missing files ({', '.join(absent)}). "
                "Download the model (Model catalog tab) or supply valid "
                "local files.")

    return {"model_path": model_path, "diffusion": diffusion, "vae": vae,
            "enc": enc, "uncond": uncond, "t5xxl": t5xxl, "clip_l": clip_l,
            "llm_vision": llm_vision}


def adetailer_command(model_id: str, source: Path, output: Path,
                      detector: Path, prompt: str = "", negative: str = "",
                      denoise: float = 0.4, steps: int = 0, seed: int = -1,
                      extra: dict | None = None) -> list[str]:
    """Commande `sd-cli -M adetailer` : détecter, redessiner, recoller.

    Le modèle, sa quantification et le placement mémoire viennent du MÊME
    chemin que la génération normale (`resolve_model_files` + `memory_args`).
    C'est voulu : la zone redessinée doit sortir du modèle qui a fait l'image,
    sinon le raccord se voit — et une seconde copie de cette résolution
    dériverait, comme celle du serveur résident l'avait fait.
    """
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError("The sd-cli binary was not found. "
                                "Run install.bat.")
    if not sdcpp.adetailer_supported(sd_cli):
        raise sdcpp.EngineError(
            "Your sd.cpp engine does not know ADetailer yet (the `--ad-model` "
            "option). Run update.bat, then restart the application.")
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Unknown model: {model_id}")
    files = resolve_model_files(model)
    flags, gpu_index = _resolved_flags(prefs)
    d = dict(model.defaults)

    req = GenRequest(
        model_path=files["model_path"], diffusion_model=files["diffusion"],
        vae=files["vae"], text_encoder=files["enc"],
        uncond_model=files["uncond"], t5xxl=files["t5xxl"],
        clip_l=files["clip_l"],
        prompt=prompt, negative=negative,
        steps=int(steps or d.get("steps", 8) or 8),
        cfg_scale=float(d.get("cfg_scale", 1.0)),
        sampler=d.get("sampler", "euler"), schedule=d.get("schedule", ""),
        seed=int(seed), strength=float(denoise),
        flags=flags, gpu_index=gpu_index,
        params_backend=(prefs.get("params_backend") or ""),
    )

    cmd = [str(sd_cli), "-M", "adetailer",
           "-i", str(source), "-o", str(output),
           "--ad-model", str(detector)]
    if req.model_path:
        cmd += ["-m", str(req.model_path)]
    else:
        cmd += ["--diffusion-model", str(req.diffusion_model)]
        if req.uncond_model:
            cmd += ["--uncond-diffusion-model", str(req.uncond_model)]
        if req.text_encoder:
            cmd += ["--llm", str(req.text_encoder)]
        if req.t5xxl:
            cmd += ["--t5xxl", str(req.t5xxl)]
        if req.clip_l:
            cmd += ["--clip_l", str(req.clip_l)]
    if req.vae:
        cmd += ["--vae", str(req.vae)]
    cmd += ["-p", req.prompt, "--steps", str(req.steps),
            "--cfg-scale", str(req.cfg_scale),
            "--sampling-method", req.sampler,
            "--strength", f"{req.strength:.2f}", "-s", str(req.seed)]
    if req.negative and req.cfg_scale > 1.0:
        cmd += ["-n", req.negative]
    if req.schedule:
        cmd += ["--scheduler", req.schedule]
    cmd += sdcpp.memory_args(sd_cli, req)
    if extra:
        cmd += ["--extra-ad-args",
                ",".join(f"{k}={v}" for k, v in extra.items() if v != "")]
    cmd += ["-v"]
    return cmd


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
    hires: "sdcpp.HiresParams | None" = None,
    # None = suivre la préférence utilisateur ; une chaîne force la valeur
    # (la passe HD s'en sert pour activer la découpe même si la préférence
    # laisse la génération ordinaire tranquille).
    max_vram: str | None = None,
    # Même logique que `max_vram` ci-dessus : la passe HD force le streaming
    # des couches pour une tentative précise, sans toucher aux préférences.
    stream_layers: bool | None = None,
    log: Callable[[str], None] | None = None,
    # Banc d'essai : préférences en mémoire, sans toucher au fichier utilisateur.
    prefs_override: dict | None = None,
) -> list[Path]:
    prefs = prefs_override if prefs_override is not None else settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "The sd-cli binary was not found. Run the installation "
            "(install.bat) or “python scripts/get_sdcpp.py”.")

    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Unknown model: {model_id}")

    files = resolve_model_files(model, diffusion_override, vae_override,
                                encoder_override, want_vision=bool(ref_image))
    model_path, diffusion, vae = (files["model_path"], files["diffusion"],
                                  files["vae"])
    enc, uncond = files["enc"], files["uncond"]
    t5xxl, clip_l, llm_vision = (files["t5xxl"], files["clip_l"],
                                 files["llm_vision"])

    flags, gpu_index = _resolved_flags(prefs)
    loras = loras or []

    # Placement. auto-fit laisse sd.cpp planifier la résidence de chaque module
    # (prioritaire) ; sinon split d'encodeur sur un 2e GPU (ex. 1080 Ti). Les
    # deux rendent toutes les cartes visibles (all_gpus) avec l'ordre CUDA par
    # bus PCI — auto-fit parce que la RAM d'une autre carte est l'un de ses
    # paliers, pas parce qu'il calculerait sur plusieurs.
    auto_fit = bool(prefs.get("auto_fit"))
    enc_gpu = prefs.get("encoder_gpu_index")
    # La carte d'encodeur est écartée ICI, avant tout le reste : plusieurs
    # branches plus bas re-déduisent le split à partir de `enc_gpu`, et une
    # décision prise en aval leur échapperait. Le banc d'essai, lui, a le droit
    # de mesurer le placement qu'on refuse — c'est sa raison d'être.
    slow_encoder = (not prefs.get("encoder_placement_forced")
                    and encoder_gpu_too_slow(enc_gpu, gpu_index))
    if slow_encoder:
        if log:
            log(_SLOW_ENCODER_GPU)
        enc_gpu = None
    split_gpu = (not auto_fit) and enc_gpu is not None and enc_gpu != gpu_index
    all_gpus = auto_fit or split_gpu
    if auto_fit:
        # Auto-fit gère lui-même la résidence : VRAM si ça tient, puis partage
        # temporel, RAM/disque et découpe des modules si nécessaire. On retire
        # les anciens raccourcis CPU pour ne pas court-circuiter cet arbitrage,
        # et on garde le VAE tiling comme garde-fou au décodage.
        flags = {**flags, "offload_to_cpu": False, "vae_tiling": True}

    # LoRA : via tags <lora:…> dans le prompt + --lora-model-dir (mode sd-cli).
    final_prompt = _apply_loras(prompt, loras)
    lora_dir = settings.LORA_DIR if loras else None

    raw_params_backend = prefs.get("params_backend") or ""
    params_backend = "" if auto_fit else raw_params_backend
    if slow_encoder and not auto_fit:
        # La préférence enregistrée dit encore « te=cuda<Pascal> » : sans cette
        # réécriture, sortir l'encodeur du split ne servirait à rien, la
        # résidence l'y renverrait. Ses poids passent en RAM et son calcul
        # revient sur la carte de génération, remappée en cuda0 puisqu'on n'est
        # plus en multi-GPU.
        params_backend = "diffusion=cuda0,vae=cuda0,te=cpu"
    # Si l'utilisateur choisit le split d'encodeur et que le moteur moderne est
    # installé, le placement attendu est la valeur sûre : poids ET calcul sur
    # la même carte. L'ancien mode global RAM reste mesuré par le benchmark.
    if split_gpu and not params_backend:
        # En mono-GPU, CUDA_VISIBLE_DEVICES remappe la carte choisie en cuda0.
        # En split, toutes les cartes restent visibles et gardent leurs index.
        g = (gpu_index if gpu_index is not None else 0) if split_gpu else 0
        params_backend = f"diffusion=cuda{g},vae=cuda{g},te=cuda{enc_gpu}"

    stream_layers = (bool(prefs.get("stream_layers")) if stream_layers is None
                     else bool(stream_layers))
    cache_mode = prefs.get("cache_mode") or ""
    cache_option = prefs.get("cache_option") or ""
    targeted = (prefs.get("cache_by_model") or {}).get(model_id) or {}
    if not cache_mode and isinstance(targeted, dict):
        cache_mode = targeted.get("mode") or ""
        cache_option = targeted.get("option") or ""

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
        params_backend=params_backend,
        auto_fit=auto_fit, split_mode=prefs.get("split_mode") or "",
        cache_mode=cache_mode, cache_option=cache_option,
        hires=hires,
        max_vram=(max_vram if max_vram is not None
                  else sdcpp.max_vram_arg(prefs.get("max_vram") or "")),
        stream_layers=stream_layers,
    )
    out = sdcpp.unique_output(model.family)
    base_max_vram = req.max_vram

    def _attempt(relief: bool) -> list[Path]:
        if relief and auto_fit:
            # Sous auto-fit, `--clip-on-cpu` ne sert à RIEN : le raccourci
            # est effacé par `memory_args` (auto-fit décide seul du
            # placement), et il désactiverait de toute façon auto-fit en
            # amont puisqu'il se traduit par une affectation `--backend`.
            # La reprise relancerait donc la commande qui vient d'échouer.
            #
            # Le levier prévu pour ce cas est le BUDGET. Une valeur négative
            # se lit « mémoire libre moins tant de Gio » : en creusant la
            # réserve, le planificateur descend d'un cran dans sa cascade
            # (VRAM → RAM → autre carte → disque) au lieu de retenter le
            # même placement. On écrase le budget précédent sans état d'âme :
            # il vient de prouver qu'il ne passait pas.
            req.flags = flags
            req.params_backend = params_backend
            req.max_vram = sdcpp.max_vram_arg(sdcpp.MAX_VRAM_AUTO,
                                              spare_gib=_OOM_RELIEF_SPARE_GIB)
            return _run_attempt()
        req.flags = {**flags, "clip_on_cpu": True} if relief else flags
        # Sortir l'encodeur du GPU demande DEUX choses : que son calcul parte
        # sur le CPU (`--clip-on-cpu`) et que ses poids n'y restent pas
        # résidents. Sans la seconde, la résidence explicite le ramène sur la
        # carte et la reprise relance une commande identique à celle qui vient
        # d'échouer — un rechargement complet du modèle pour rater pareil.
        req.params_backend = (_te_on_cpu(params_backend)
                              if relief else params_backend)
        # Et un BUDGET, si l'utilisateur n'en avait pas posé. C'est l'autre
        # moitié du problème, et souvent la seule qui compte : sans
        # `--max-vram`, le moteur découpe son graphe sans cible et découvre au
        # troisième segment qu'il ne tient pas. Sortir l'encodeur ne l'aide
        # pas — l'encodeur a déjà fini de travailler quand la diffusion
        # échoue. On ne touche pas à un budget CHOISI : celui-là est une
        # décision, pas un défaut.
        req.max_vram = (base_max_vram or
                        sdcpp.max_vram_arg(sdcpp.MAX_VRAM_AUTO)) if relief \
            else base_max_vram
        return _run_attempt()

    def _run_attempt() -> list[Path]:
        cmd = sdcpp.build_gen_cmd(sd_cli, req, out)
        sdcpp.run(cmd, log=log, gpu_index=gpu_index, all_gpus=all_gpus)
        return sdcpp.collect_outputs(out, batch_count)

    try:
        # Sous auto-fit la reprise n'est PAS « l'encodeur en RAM » mais un
        # budget plus serré : partir directement en reprise dépenserait ce
        # levier avant même d'avoir essayé le placement demandé.
        paths = _attempt(not auto_fit and bool(flags.get("clip_on_cpu")))
    except sdcpp.VramError:
        # L'encodeur revenu sur le GPU de génération peut faire déborder une
        # carte juste. Plutôt que de renvoyer l'utilisateur à ses réglages, on
        # reprend une fois avec l'encodeur en RAM : plus lent que le GPU, mais
        # toujours bien plus rapide qu'une Pascal, et ça ne coûte pas un octet
        # de VRAM.
        if flags.get("clip_on_cpu") and not auto_fit:
            raise
        if log:
            if auto_fit:
                log("↻ Not enough GPU memory — retrying with a tighter budget "
                    f"(--max-vram -{_OOM_RELIEF_SPARE_GIB:g}), so automatic "
                    "placement moves more weights out of the card.")
            elif base_max_vram:
                log("↻ Not enough GPU memory — retrying with the text encoder "
                    "in RAM (--clip-on-cpu).")
            else:
                log("↻ Not enough GPU memory — retrying with a compute budget "
                    "(--max-vram auto) so the engine can cut its graph to "
                    "fit, and with the text encoder in RAM.")
        paths = _attempt(True)

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
            "The sd-cli binary was not found. Run the installation (install.bat).")
    model = registry.upscaler_path(model_name)
    if model is None:
        raise sdcpp.EngineError(
            f"Upscaler not found: “{model_name}”. Download the upscalers "
            "from the Toolkit → Enlarge tab.")

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
        log(f"ESRGAN upscale “{model_name}” (×{repeats or 1}) on the GPU…")
        if "--upscale-tile-size" in sdcpp.supported_options(sd_cli):
            log(f"[esrgan] tiles of {tile} px"
                + (" — the whole image in one pass, no seam."
                   if tile >= max(w, h) else
                   f" (instead of {sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px:"
                   " fewer seams and less aliasing)."))
        else:
            log("[esrgan] sd-cli too old for --upscale-tile-size: tiles of "
                f"{sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px (seams possible). "
                "Update the engine (update.bat).")

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
            log(f"[esrgan] failed with tiles of {tile} px (VRAM ?) → "
                f"retrying at the sd.cpp default "
                f"({sdcpp.SDCPP_DEFAULT_UPSCALE_TILE} px).")
        sdcpp.run(fallback, log=log, gpu_index=gpu_index)
    if out.is_file():
        return out
    found = sorted(out.parent.glob(f"{out.stem}*{out.suffix}"))
    if found:
        return found[0]
    raise sdcpp.EngineError("The upscale produced no image.")


# Alignement des tailles de la passe HD.
#
# On aligne sur 16 px, PAS sur la grille « native » de chaque famille (32 px
# Flux.2, 64 px Krea 2), et vers le HAUT. Deux raisons :
#
#  • 16 est un diviseur commun à toutes ces grilles, donc c'est le pas qui
#    déforme le moins — et surtout il préserve tel quel tout ce que
#    l'application produit elle-même (1184×880, 1152×896, 752…), alors qu'un
#    arrondi à 64 les déplacerait ;
#  • sd.cpp aligne de toute façon LUI-MÊME vers le haut sur son vrai multiple
#    (`vae_scale_factor × diffusion_model_down_factor`) et le journalise. Il est
#    l'autorité sur ce chiffre, pas nous : on lui donne une taille propre et on
#    le laisse compléter s'il lui faut davantage.
#
# Vers le haut, jamais vers le bas : arrondir à la baisse rognerait du cadrage.
HD_ALIGN = 16


def _align_up(v: int, step: int = HD_ALIGN) -> int:
    return max(step, -(-int(v) // step) * step)


# Plafond de côté pour la passe HD. Au-delà, le second débruitage se fait sur
# une image que le modèle n'a jamais vue à cette échelle : il se met à répéter
# des motifs (« doublons »).
HD_MAX_SIDE = 3072

# --------------------------------------------------------------------------- #
#  Budget mémoire de la passe HD.
#
#  Le second débruitage porte sur l'image ENTIÈRE : son tampon de calcul est
#  proportionnel au nombre de pixels, et il s'AJOUTE aux poids du modèle déjà
#  résidents sur la carte. C'est ce cumul qui déborde, pas la taille seule —
#  d'où un budget calculé, et non un plafond fixe en pixels.
#
#  La constante ci-dessous vient d'un OOM réel de sd.cpp : Krea 2 en 2304×1792
#  (4,13 Mpx) a réclamé un tampon de 4961670528 octets, soit ~1200 octets par
#  pixel. C'est un ordre de grandeur, pas une loi — il dépend du modèle et de la
#  version du moteur. Le vrai filet de sécurité est la reprise automatique en
#  plus petit, qui, elle, ne dépend d'aucune estimation.
# --------------------------------------------------------------------------- #
HD_COMPUTE_BYTES_PER_PX = 1200

# Plancher du budget : en dessous, l'agrandissement n'aurait plus d'intérêt et
# c'est à la reprise automatique de constater l'échec, pas à l'estimation de
# proposer une taille absurde.
HD_MIN_PIXELS = 512 * 512

# Fraction de la VRAM qu'on s'autorise (le reste : contexte CUDA, bureau,
# fragmentation).
HD_VRAM_USABLE = 0.90

# Réduction linéaire appliquée à chaque nouvelle tentative après un OOM.
# 0,8 en côté = 0,64 en pixels : assez pour changer le résultat, assez peu pour
# ne pas sacrifier l'agrandissement dès le premier échec.
HD_RETRY_FACTOR = 0.8
HD_MAX_RETRIES = 2

# Débruitage du PREMIER passage (img2img avant l'agrandissement). sd.cpp calcule
# t_enc = pas × force : à 0,15 sur 8 pas ça fait un seul pas, à très bas sigma —
# l'image d'origine est pratiquement recopiée. On ne peut pas le supprimer (la
# passe HD s'accroche derrière une génération), alors on le rend négligeable.
HD_FIRST_PASS_STRENGTH = 0.15


def _weights_bytes(model: registry.BaseModel) -> int:
    """Taille sur disque des poids qui atterrissent sur le GPU.

    Approximation volontairement grossière — mais mesurée, pas devinée : un
    GGUF occupe en VRAM à peu près sa taille de fichier. On ne compte QUE la
    diffusion : l'encodeur de texte est déchargé en RAM, et le VAE est
    négligeable devant le reste.
    """
    for role in ("diffusion", "model"):
        p = _component(model, role)
        if p is not None:
            try:
                return Path(p).stat().st_size
            except OSError:
                pass
    return 0


def hd_pixel_budget(model: registry.BaseModel, vram_gb: float | None,
                    free_gb: float | None = None) -> int:
    """Nombre de pixels que la passe HD peut viser sur cette carte.

    `free_gb` = VRAM réellement libre à l'instant T. Quand on l'a, elle prime :
    budgéter sur la VRAM TOTALE revient à supposer la carte vide, ce qui est
    faux dès qu'un navigateur, un jeu ou une autre application y a pris sa part
    — et c'est alors l'OOM garanti. On garde le total comme repli, et on ne
    dépasse jamais ce que le libre autorise.

    0 = AUCUNE mesure disponible (pas de GPU détecté) : on ne plafonne pas, on
    laisse la reprise automatique faire le travail plutôt que d'inventer une
    limite. À ne pas confondre avec « mesuré, et ça ne rentre pas » : là on
    renvoie le plancher, c'est-à-dire le plafond le plus SERRÉ possible. Rendre
    0 dans ce cas revenait à tenter la pleine taille précisément quand la
    mémoire manquait le plus — soit exactement l'inverse du but.
    """
    usable = None
    for candidate in (free_gb, vram_gb):
        if candidate and candidate > 0:
            v = candidate * (1024 ** 3) * HD_VRAM_USABLE
            usable = v if usable is None else min(usable, v)
    if usable is None:
        return 0
    return max(HD_MIN_PIXELS,
               int((usable - _weights_bytes(model)) / HD_COMPUTE_BYTES_PER_PX))


def hd_upscale(model_id: str, image, scale: float = 2.0,
               upscaler: str = "Lanczos", denoise: float = 0.4,
               prompt: str = "", negative: str = "", steps: int = 0,
               hd_steps: int = 0, seed: int = -1,
               preview_path: Path | None = None,
               log: Callable[[str], None] | None = None) -> list[Path]:
    """Passe **HD** native sd.cpp : agrandir puis re-débruiter l'image ENTIÈRE.

    Ni PyTorch, ni SDXL, ni découpage en tuiles — donc aucune couture possible,
    et c'est le modèle de génération installé (Krea 2, Flux.2) qui redessine le
    détail, dans le style qu'il connaît déjà.

    Déroulé côté sd.cpp, en une seule commande : img2img très léger à la taille
    d'origine → agrandissement (latent, Lanczos ou un ESRGAN) → second
    débruitage à la taille finale. `denoise` règle ce second passage : c'est le
    seul réglage qui compte vraiment ici.
    """
    from PIL import Image
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "The sd-cli binary was not found. Run the installation (install.bat).")
    if not sdcpp.hires_supported(sd_cli):
        raise sdcpp.EngineError(
            "Your sd.cpp engine does not know the HD pass yet (the “--hires” "
            "option).\n→ Run update.bat to fetch a recent version, "
            "then restart the application.")
    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Unknown model: {model_id}")

    settings.ensure_dirs()
    src = settings.TMP_DIR / "hd_in.png"
    im = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) \
        else image.convert("RGB")
    im.save(src)
    ow, oh = im.size

    # Taille de travail : c'est à cette taille que sd.cpp redimensionne l'image
    # d'entrée pour le premier passage.
    bw, bh = _align_up(ow), _align_up(oh)

    # Deux plafonds, et on réduit toujours le FACTEUR plutôt que de rogner :
    # le cadrage demandé doit rester celui qui sort.
    #   1. le côté, au-delà duquel le modèle quitte son échelle d'entraînement ;
    #   2. le nombre de PIXELS que la VRAM peut porter — c'est celui qui mord en
    #      pratique, parce que le tampon du second débruitage s'ajoute aux poids.
    scale = float(scale)
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    vram = prof.gpu.vram_gb if prof.gpu else None
    if max(bw, bh) * scale > HD_MAX_SIDE:
        scale = max(1.25, HD_MAX_SIDE / max(bw, bh))
        if log:
            log(f"[hd] factor lowered to ×{scale:.2f} to stay under "
                f"{HD_MAX_SIDE} px on a side.")
    free = hardware.free_vram_gb(prof.gpu.index if prof.gpu else None)

    # Exécution segmentée : quand le moteur sait DÉCOUPER son graphe pour tenir
    # dans un budget, notre estimation en pixels cesse d'être la contrainte —
    # ce n'est plus « ça rentre d'un bloc ou ça casse ». On l'active ici même si
    # la préférence laisse la génération ordinaire tranquille : c'est la passe HD
    # qui souffre du tout-ou-rien, pas une génération à la taille native.
    #
    # Le budget en pixels est alors ABANDONNÉ plutôt que relâché à moitié :
    # le garder brimerait précisément ce qu'on vient de rendre possible. Le vrai
    # plancher est trouvé par la reprise automatique, qui mesure au lieu d'estimer.
    cut = sdcpp.max_vram_arg(prefs.get("max_vram") or sdcpp.MAX_VRAM_AUTO)
    if cut and not sdcpp.max_vram_supported(sd_cli):
        cut = ""
    if cut:
        if log:
            log(f"[hd] segmented run (--max-vram {cut}): the engine splits "
                "its graph to fit in the free VRAM instead of allocating "
                "in one block.")
    else:
        budget = hd_pixel_budget(model, vram, free)
        if budget and bw * bh * scale * scale > budget:
            scale = max(1.25, (budget / (bw * bh)) ** 0.5)
            if log:
                have = (f"{free:.1f} GB free" if free else
                        f"{vram:.0f} GB of VRAM" if vram else
                        "the available VRAM")
                log(f"[hd] factor lowered to ×{scale:.2f}: past that, the "
                    f"second denoise's buffer does not fit in {have} with "
                    "this model loaded.")
    if log and free and vram and free < vram * 0.75:
        # Utile à savoir AVANT de lancer : de la VRAM est prise ailleurs.
        log(f"[hd] {free:.1f} GB free of {vram:.0f} — another application is "
            "holding the card. Close it to aim higher.")

    d = dict(model.defaults)
    base_steps = int(steps or d.get("steps", 8) or 8)

    def _attempt(sc: float, stream: bool = False) -> list[Path]:
        tw, th = _align_up(bw * sc), _align_up(bh * sc)
        hires = sdcpp.HiresParams(
            scale=sc, upscaler=upscaler or "Lanczos",
            upscalers_dir=registry.upscalers_dir(),
            denoise=float(denoise), steps=int(hd_steps or 0),
            target_width=tw, target_height=th,
            # Ne sert que si l'agrandisseur intermédiaire est un ESRGAN : même
            # correction de tuilage que pour l'agrandissement simple.
            tile_size=sdcpp.upscale_tile_size(tw, th, vram))
        if log:
            log(f"HD “{model.name}”: {ow}×{oh} → {tw}×{th} (×{sc:.2f}), "
                f"enlarger “{hires.upscaler}”, detail {denoise:g}.")
            log("[hd] second denoise over the WHOLE image: no tiles, so no "
                "seam is possible.")
        got = generate(
            model_id=model_id, prompt=prompt, negative=negative,
            steps=base_steps,
            cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
            width=bw, height=bh, seed=seed, batch_count=1,
            sampler=d.get("sampler") or "euler",
            schedule=("" if d.get("scheduler") in (None, "", "auto")
                      else d["scheduler"]),
            init_image=src, strength=HD_FIRST_PASS_STRENGTH,
            preview_path=preview_path, hires=hires, max_vram=cut,
            stream_layers=True if stream else None, log=log)
        # Taille RÉELLE : sd.cpp peut avoir arrondi au-dessus de notre demande
        # pour tomber sur son propre multiple. Autant lire le résultat que
        # d'affirmer une taille qu'on n'a pas vérifiée.
        if log and got:
            try:
                with Image.open(got[0]) as res:
                    if res.size != (tw, th):
                        log(f"[hd] actual final size: {res.width}×"
                            f"{res.height} (aligned by sd.cpp).")
            except OSError:
                pass
        return got

    # Reprise automatique sur manque de VRAM. Le budget calculé plus haut n'est
    # qu'une estimation ; ceci est la mesure. Un OOM est le seul échec qui vaille
    # une nouvelle tentative — tout le reste échouerait à l'identique.
    #
    # ORDRE DES REPRISES. Baisser le facteur coûte des pixels, définitivement.
    # Le streaming des couches coûte de la bande passante PCIe, donc du temps —
    # et rend l'image demandée. Entre perdre 20 % de côté et attendre plus
    # longtemps, c'est à l'utilisateur de trancher, mais le défaut raisonnable
    # est de tenter d'abord ce qui ne sacrifie rien du résultat.
    #
    # Le moteur documente cette échelle (docs/performance.md) : « --offload-to-cpu
    # → + --max-vram → + --stream-layers », et annonce des modèles 3 à 4 fois
    # plus gros que la VRAM brute une fois les trois cumulés. On avait déjà les
    # deux premiers barreaux ; celui-ci manquait.
    stream = sdcpp.stream_layers_possible(sd_cli, prefs.get("flags") or {},
                                          prefs.get("params_backend") or "")
    streaming_on = bool(prefs.get("stream_layers"))
    # On compte les BAISSES DE FACTEUR, pas les tentatives : la tentative en
    # streaming ne sacrifie aucun pixel, elle ne doit donc pas consommer un
    # barreau de l'échelle de repli. Sans ça, activer le streaming coûterait
    # une réduction de taille — exactement ce qu'il sert à éviter.
    drops = 0
    while True:
        try:
            return _attempt(scale, stream=streaming_on)
        except sdcpp.VramError:
            last = scale
            if sdcpp.was_cancelled():
                raise
            # 1er recours : garder le facteur, charger les couches depuis la
            # RAM. Une seule fois — si ça n'a pas suffi, insister ne changera
            # rien et il faut vraiment descendre.
            if stream and not streaming_on:
                streaming_on = True
                if log:
                    log(f"[hd] not enough VRAM at ×{last:.2f} → we keep the factor "
                        "and stream the model's layers from RAM as the "
                        "computation goes (--stream-layers). It is slower, "
                        "but the image stays at the size you asked for.")
                continue
            if drops >= HD_MAX_RETRIES:
                raise
            scale = max(1.25, scale * HD_RETRY_FACTOR)
            if scale >= last:
                raise      # plancher atteint : insister ne changerait rien
            drops += 1
            if log:
                log(f"[hd] not enough VRAM at ×{last:.2f} → retrying at "
                    f"×{scale:.2f}.")
