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


def _gpu() -> "hardware.Gpu | None":
    prefs = settings.load_prefs()
    gpus = hardware.detect_gpus()
    if not gpus:
        return None
    wanted = prefs.get("gpu_index")
    return next((g for g in gpus if g.index == wanted), None) or \
        max(gpus, key=lambda g: g.vram_gb)


def plan_for(model: catalog.TorchModel,
             gpu: "hardware.Gpu | None") -> placement.Plan:
    """Le placement retenu pour ce modèle sur cette carte."""
    return placement.plan(
        vram_gb=gpu.vram_gb if gpu else None,
        resident_gb=model.resident_bf16_gb,
        largest_module_gb=model.largest_module_bf16_gb,
        arch=gpu.arch if gpu else "unknown")


def _source(model: catalog.TorchModel) -> str | Path:
    """Le dossier local s'il est là, sinon l'identifiant du dépôt.

    Laisser passer l'identifiant permet à diffusers de télécharger lui-même au
    premier lancement — et c'est aussi ce qui produit, sur un dépôt fermé,
    l'erreur d'authentification qu'on veut voir plutôt qu'un « fichier
    introuvable » qui n'expliquerait rien.
    """
    local = model.local_dir
    return local if (local / "model_index.json").is_file() else model.repo


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
    base = registry.get_base_model(model_id, prefs_override
                                   or settings.load_prefs())

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

    gpu = _gpu()
    plan = plan_for(model, gpu)
    _log(log, placement.describe(plan))
    if model.gated:
        _log(log, f"[torch] “{model.repo}” is a gated repository: it needs a "
                  "Hugging Face account, the licence accepted on the model "
                  "page, and a token (huggingface-cli login).")

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
        pipe = runtime.load_pipeline(key, plan, _source(model), log)
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
