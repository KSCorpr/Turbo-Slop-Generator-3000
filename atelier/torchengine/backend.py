"""`generate()` — la même fonction, le même contrat, un autre moteur.

Toute l'interface passe par UNE fonction : `atelier.engine.generate.generate`.
Les onglets de génération, Xanax, la passe HD, l'outpaint, ADetailer et le banc
d'essai l'appellent tous avec la même signature. C'est ce qui rend cette
branche possible : il suffit de la remplacer.

Le contrat qu'on tient, parce que l'interface en dépend :

* les images sont ÉCRITES sur le disque et la fonction rend leurs chemins ;
* `batch_count` produit des graines CONSÉCUTIVES, une image chacune — c'est ce
  que fait sd.cpp, et c'est ce que promet le bandeau « Images » ;
* `preview_path` reçoit un aperçu au fil des pas ;
* `cancel()` interrompt entre deux pas, pas à la fin ;
* une graine donnée rejoue la même image (le bruit est tiré sur le CPU, voir
  `runtime.generator_for`).

Ce qu'on ne tient pas, on le DIT dans le journal plutôt que de le contourner :
un mode que diffusers ne publie pas pour ce modèle, un échantillonneur sans
équivalent, une image de départ sur un modèle qui n'a pas de classe img2img.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .. import hardware, registry, settings
from ..engine import sdcpp
from . import catalog, placement, runtime, schedulers

#  Annulation coopérative. Le drapeau est lu par le rappel de pas : sd.cpp tue
#  un processus, ici il n'y en a pas — on lève entre deux pas, ce qui laisse le
#  pipeline dans un état propre et réutilisable au lieu de le corrompre.
_CANCEL = threading.Event()


class Cancelled(RuntimeError):
    """Génération interrompue à la demande."""


def cancel() -> str:
    _CANCEL.set()
    return "⏹️ Stopping after the current step…"


def _log(log: Callable | None, msg: str) -> None:
    if log:
        log(msg)


def _gpu(prefs: dict | None = None) -> "hardware.Gpu | None":
    """La carte de génération — en respectant `prefs_override`.

    Lire `settings.load_prefs()` ici serait plus court et faux : le banc
    d'essai passe ses préférences en mémoire précisément pour ne pas toucher
    au fichier de l'utilisateur, et il choisit la carte qu'il veut mesurer.
    """
    prefs = settings.load_prefs() if prefs is None else prefs
    gpus = hardware.detect_gpus()
    if not gpus:
        return None
    wanted = prefs.get("gpu_index")
    return next((g for g in gpus if g.index == wanted), None) or \
        max(gpus, key=lambda g: g.vram_gb)


def component_files(model: catalog.TorchModel, prefs: dict) -> dict:
    """Les fichiers du catalogue PRINCIPAL, par rôle.

    C'est le pont entre les deux moteurs, et il est plus court qu'on ne le
    croyait : les mêmes fichiers servent aux deux. Le rôle (`diffusion`,
    `text_encoder`, `vae`) est le nom du composant dans `models.yaml`, et la
    quantification a déjà été choisie au téléchargement par l'échelle VRAM.
    """
    from ..engine import generate as gen
    base = registry.get_base_model(model.id, prefs)
    if base is None:
        return {}
    out = {}
    for part in model.parts.values():
        path = gen._component(base, part.role)
        if path is not None:
            out[part.role] = path
    return out


def sizes_gb(model: catalog.TorchModel, files: dict) -> dict[str, float]:
    """Le poids réel de chaque composant, lu sur le disque.

    Lu et non estimé : c'est le gain le plus net du chemin GGUF. Un calcul en
    octets par paramètre se trompe d'un facteur deux dès qu'un modèle mélange
    les précisions par couche — ce que fait précisément un GGUF « _K_M ».
    """
    out: dict[str, float] = {}
    for name, part in model.parts.items():
        path = files.get(part.role)
        try:
            out[name] = path.stat().st_size / (1024 ** 3) if path else 0.0
        except OSError:
            out[name] = 0.0
    out.update(_supplement_sizes(model))
    return out


def _supplement_sizes(model: catalog.TorchModel) -> dict[str, float]:
    """Le poids des composants qui NE viennent pas du GGUF.

    Sans eux le planificateur ne voyait que le transformer. Sur Krea 2 il
    annonçait « 8,3 Go, tout tient sur 12 » pour un pipeline qui en pèse près
    de 18 une fois l'encodeur compté — et promettait la résidence complète
    juste avant de manquer de mémoire. Un plan qui ignore la moitié du modèle
    n'est pas optimiste, il est faux.
    """
    if model.supplement is None:
        return {}
    root = model.supplement.local_dir
    out: dict[str, float] = {}
    for name in ("text_encoder", "vae"):
        folder = root / name
        if not folder.is_dir():
            #  Absent = inconnu, pas vide. Zéro ferait croire que ça tient.
            out[name] = 0.0
            continue
        total = 0
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix in (".safetensors", ".bin", ".gguf"):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        out[name] = total / (1024 ** 3)
    return out


def plan_for(model: catalog.TorchModel, gpu: "hardware.Gpu | None",
             prefs: dict | None = None,
             files: dict | None = None) -> placement.Plan:
    """Le placement retenu pour ce modèle sur cette carte.

    Deux chemins, parce qu'il y a deux façons d'arriver. Depuis le GGUF les
    poids sont déjà quantifiés et leur taille se LIT ; il ne reste que le
    placement. Depuis un dépôt complet (Krea 2, faute de chargement
    fichier-unique en amont) l'ancienne échelle int8/NF4 s'applique.

    `torch_allow_quant: False` désactive ces crans. Ce n'est pas un réglage de
    confort mais ce que le banc d'essai a besoin de pouvoir forcer pour
    comparer le plan retenu à celui d'à côté — sans quoi il mesurerait deux
    fois la même chose.
    """
    arch = gpu.arch if gpu else "unknown"
    vram = gpu.vram_gb if gpu else None
    if model.from_gguf:
        chosen = placement.plan_from_files(vram, sizes_gb(model, files or {}),
                                           arch)
        return placement.forced(chosen,
                                (prefs or {}).get("torch_force_placement"))
    allow = True
    if prefs is not None and "torch_allow_quant" in prefs:
        allow = bool(prefs.get("torch_allow_quant"))
    return placement.plan(
        vram_gb=vram,
        resident_gb=model.full_repo_gb,
        largest_module_gb=(model.full_repo_gb / 2 if model.full_repo_gb
                           else None),
        arch=arch, allow_quant=allow)


def _source(model: catalog.TorchModel) -> str | Path:
    """Pour le repli dépôt complet : le dossier local s'il est là, sinon
    l'identifiant du dépôt.

    Laisser passer l'identifiant permet à diffusers de télécharger lui-même au
    premier lancement — et c'est aussi ce qui produit, sur un dépôt fermé,
    l'erreur d'authentification qu'on veut voir plutôt qu'un « fichier
    introuvable » qui n'expliquerait rien.
    """
    local = model.local_dir
    return local if (local / "model_index.json").is_file() else model.repo


def token_advice(model: catalog.TorchModel) -> str:
    """Le message quand un jeton manque — et où aller le mettre.

    Surtout pas « huggingface-cli login ». Cette application tourne sur un
    Python portable, sans console et sans PATH : la commande n'existe pas pour
    celui qui lit le message, et l'envoyer dans un terminal qu'il n'a pas est
    une impasse polie.

    Fonction séparée pour être vérifiable : la consigne est le genre de détail
    qu'on recopie d'une version à l'autre sans le relire.
    """
    return (f"[torch] “{model.id}” needs a Hugging Face token: "
            f"{model.needs_token}."
            "\n→ Accept the licence on that model's page, then paste a read "
            "token in Settings → 🌍 Theme and accounts. "
            "“🔍 Check what is still missing” confirms it.")


def _resolve_mode(model: catalog.TorchModel, init_image, ref_image, mask_image,
                  log) -> str:
    wanted = catalog.mode_for(model, init_image=init_image,
                              ref_image=ref_image, mask_image=mask_image)
    if model.can(wanted):
        return wanted
    _log(log, f"⚠️ “{model.id}” has no {wanted.replace('_', '-')} pipeline in "
              "diffusers — that image is ignored and a plain text-to-image "
              "render is produced instead.")
    return catalog.TEXT_TO_IMAGE


def _call_kwargs(mode: str, model: catalog.TorchModel, *, prompt, negative,
                 steps, cfg_scale, width, height, init_image, ref_image,
                 mask_image, strength) -> dict[str, Any]:
    """Les arguments d'appel du pipeline, par mode.

    Les trois familles ne prennent pas les mêmes : Flux.2 Klein reçoit ses
    références sous `image=` sans `strength` (ce n'est pas un point de départ
    bruité mais un conditionnement), Z-Image en img2img veut `image=` ET
    `strength=`, et Krea 2 n'a que le mode texte. Écrit ici plutôt que déduit
    au moment de l'appel : un `TypeError` sur un argument de pipeline arrive
    après le chargement des poids, donc après vingt secondes d'attente.
    """
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": max(1, int(steps)),
        "guidance_scale": float(cfg_scale),
        "width": int(width), "height": int(height),
    }
    # Le prompt négatif n'a de sens qu'avec un guidage : à CFG 1.0 il n'y a
    # pas de passe non conditionnée où l'appliquer. Et tous les modèles ne
    # l'exposent pas — l'envoyer quand même lèverait un TypeError.
    if negative and model.negative_prompt and float(cfg_scale) > 1.0:
        kwargs["negative_prompt"] = negative

    if mode == catalog.EDIT:
        refs = ref_image if isinstance(ref_image, (list, tuple)) else [ref_image]
        kwargs["image"] = [_open(p) for p in refs if p]
    elif mode == catalog.IMAGE_TO_IMAGE:
        src = init_image or (ref_image[0] if isinstance(ref_image, (list, tuple))
                             else ref_image)
        kwargs["image"] = _open(src)
        kwargs["strength"] = float(strength)
    elif mode == catalog.INPAINT:
        kwargs["image"] = _open(init_image)
        kwargs["mask_image"] = _open(mask_image)
        kwargs["strength"] = float(strength)
    return kwargs


def _open(path):
    from PIL import Image
    if path is None:
        return None
    if hasattr(path, "convert"):
        return path
    return Image.open(str(path)).convert("RGB")


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
    max_vram: str | None = None,
    stream_layers: bool | None = None,
    log: Callable[[str], None] | None = None,
    prefs_override: dict | None = None,
) -> list[Path]:
    """Signature identique à `engine.generate.generate`. Volontairement.

    Les paramètres propres à sd.cpp (`max_vram`, `stream_layers`, les chemins
    de composants) n'ont pas d'équivalent ici : un pipeline diffusers se charge
    en bloc, et sa mémoire est gouvernée par le plan de placement. Ils sont
    acceptés et IGNORÉS, avec une ligne de journal quand ils étaient réellement
    demandés — les retirer de la signature aurait cassé tous les appelants pour
    ne rien gagner.
    """
    _CANCEL.clear()
    model = catalog.get(model_id)
    if model is None:
        raise runtime.TorchEngineError(
            f"“{model_id}” has no PyTorch entry. See config/models_torch.yaml.")
    prefs = (prefs_override if prefs_override is not None
             else settings.load_prefs())
    base = registry.get_base_model(model_id, prefs)

    for name, value in (("--max-vram", max_vram),
                        ("layer streaming", stream_layers),
                        ("a local diffusion file", diffusion_override),
                        ("a local VAE", vae_override),
                        ("a local text encoder", encoder_override)):
        if value:
            _log(log, f"[torch] {name} is a stable-diffusion.cpp setting and "
                      "has no equivalent here — ignored.")
    if hires:
        _log(log, "[torch] the HD pass is run by the toolkit on this engine, "
                  "not inside the sampler — ignored here.")

    if not model.usable:
        raise runtime.TorchEngineError(
            f"“{model_id}” does not run on the PyTorch engine yet. Its "
            "transformer loads fine from the GGUF you already have; what is "
            "missing is around it:\n" + model.blocked_summary
            + "\n→ Switch to stable-diffusion.cpp in Settings for this model.")
    if model.needs_supplement and not model.supplement.present:
        #  Dit AVANT le chargement, et en nommant ce qui manque. « Modèle
        #  introuvable » serait faux : le gros du modèle est là, en GGUF.
        raise runtime.TorchEngineError(
            f"“{model_id}” needs its text encoder and VAE in addition to the "
            "GGUF you already have — neither can be read from GGUF on this "
            "engine.\n→ Model catalog tab, Download: about "
            f"{model.supplement.download_gb:.0f} GB, not the "
            f"{model.supplement.download_gb + (model.supplement.skipped_gb or 0):.0f} "
            "GB of the full repository.")

    # Le jeton et le point d'accès sont posés AVANT toute requête au Hub :
    # les configurations d'architecture partent chercher un dépôt qui peut
    # être fermé, et un jeton collé dans les réglages doit agir sans redémarrer.
    settings.configure_hf_env()
    #  AVANT tout import de diffusers : les deux variables sont lues une seule
    #  fois, au chargement de son module de quantification. Les poser ensuite
    #  n'aurait aucun effet, et l'utilisateur croirait avoir activé quelque
    #  chose.
    kernels_on = bool(prefs.get("torch_gguf_kernels"))
    runtime.configure_kernels(kernels_on)

    gpu = _gpu(prefs)
    files = component_files(model, prefs) if model.from_gguf else {}
    plan = plan_for(model, gpu, prefs, files)
    _log(log, placement.describe(plan))
    if model.from_gguf:
        _log(log, "[torch] weights come from the files already installed for "
                  "stable-diffusion.cpp — nothing extra to download.")
        _log(log, runtime.dequant_report(kernels_on))
    if model.needs_token:
        _log(log, token_advice(model))

    mode = _resolve_mode(model, init_image, ref_image, mask_image, log)
    cls = model.pipeline_for(mode)
    key = runtime.LoadKey(model_id=model_id, mode=mode, cls=cls,
                          dtype=plan.dtype, quant=plan.quant,
                          placement=plan.mode,
                          gpu_index=gpu.index if gpu else None)

    lora_files = tuple((str(p), float(w)) for p, w in (loras or []))
    choice = schedulers.resolve(sampler, schedule, flow_shift)
    for note in choice.notes:
        _log(log, f"[torch] {note}")

    out: list[Path] = []
    with runtime.lock():
        pipe = runtime.load_pipeline(key, plan, _source(model), log,
                                     model=model, files=files)
        runtime.set_loras(pipe, lora_files, log)
        pipe.scheduler = runtime.build_scheduler(pipe, choice, log)

        for i in range(max(1, int(batch_count))):
            if _CANCEL.is_set():
                raise Cancelled("stopped before image "
                                f"{i + 1}/{batch_count}")
            this_seed = int(seed) + i
            kwargs = _call_kwargs(
                mode, model, prompt=prompt, negative=negative, steps=steps,
                cfg_scale=cfg_scale, width=width, height=height,
                init_image=init_image, ref_image=ref_image,
                mask_image=mask_image, strength=strength)
            kwargs["generator"] = runtime.generator_for(this_seed,
                                                        key.gpu_index)
            kwargs["callback_on_step_end"] = _step_callback(
                pipe, preview_path, steps, log)
            _log(log, f"[torch] image {i + 1}/{batch_count} · seed "
                      f"{this_seed} · {width}×{height} · {steps} steps")
            result = pipe(**kwargs)
            path = sdcpp.unique_output(base.family if base else "torch")
            result.images[0].save(path)
            out.append(path)

    if save_prompt and out:
        _write_sidecars(out, model, prompt, negative, steps, cfg_scale,
                        width, height, int(seed), sampler, schedule, plan)
    return out


def _step_callback(pipe, preview_path: Path | None, steps: int, log):
    """Aperçu au fil des pas, et point d'annulation.

    Le décodage d'un aperçu coûte une passe de VAE ; on n'en fait donc pas à
    chaque pas. Sur 4 à 8 pas — le régime de tous nos modèles — un aperçu au
    milieu est la seule occasion utile, et elle sert surtout à montrer que
    quelque chose avance.
    """
    halfway = max(1, int(steps) // 2)

    def _cb(pipeline, step: int, timestep, kwargs):
        if _CANCEL.is_set():
            raise Cancelled("stopped between two steps")
        if preview_path is not None and step == halfway:
            try:
                _write_preview(pipeline, kwargs.get("latents"), preview_path)
            except Exception as exc:  # noqa: BLE001
                _log(log, f"[torch] no preview ({exc}) — generation continues.")
        return kwargs

    return _cb


def _write_preview(pipe, latents, dest: Path) -> None:
    if latents is None:
        return
    import torch as _t
    with _t.no_grad():
        image = pipe.vae.decode(
            latents / getattr(pipe.vae.config, "scaling_factor", 1.0)).sample
    image = (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu()
    from PIL import Image
    Image.fromarray((image.numpy() * 255).astype("uint8")).save(dest)


def _write_sidecars(paths, model, prompt, negative, steps, cfg, w, h, seed,
                    sampler, schedule, plan) -> None:
    """Le même fichier `.txt` que côté sd.cpp — c'est ce que l'utilisateur relit.

    Une ligne de plus, et elle compte : quel moteur a produit l'image. Deux
    branches qui écrivent le même nom de modèle dans le même dossier sans dire
    laquelle a tourné, c'est une comparaison impossible à faire trois jours
    plus tard.
    """
    from ..fileio import atomic_write_text
    for i, path in enumerate(paths):
        lines = [
            prompt or "",
            f"Negative: {negative}" if negative else "",
            f"Model: {model.id} ({model.repo})",
            "Engine: PyTorch / diffusers",
            f"Placement: {plan.mode} · {plan.dtype}"
            + (f" · {plan.quant}" if plan.quant != "none" else ""),
            f"Steps: {steps} · CFG: {cfg} · Size: {w}×{h}",
            f"Seed: {seed + i}",
            f"Sampler: {sampler or 'euler'} · Scheduler: {schedule or 'auto'}",
        ]
        atomic_write_text(path.with_suffix(".txt"),
                          "\n".join(x for x in lines if x) + "\n")
