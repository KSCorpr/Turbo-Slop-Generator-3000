"""Génération vidéo LTX-2.3 via stable-diffusion.cpp (« -M vid_gen »).

Même principe que engine/generate.py pour l'image : on résout les composants du
catalogue, on applique les réglages matériels, puis on lance sd-cli. La
différence tient aux fichiers supplémentaires que LTX réclame — un VAE audio et
des « embeddings connectors » — et aux contraintes de grille (32 px, 8k+1
images) appliquées dans sdcpp.build_vid_cmd.

Trois modes :
  · t2v   texte → vidéo ;
  · i2v   image → vidéo (une image de départ, le modèle l'anime) ;
  · flf2v début → fin (deux images, le modèle fabrique l'entre-deux).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .. import registry, settings
from . import sdcpp
from .generate import _component, _resolved_flags
from .sdcpp import VidRequest

MODES = ("t2v", "i2v", "flf2v", "ref")

_ROLE_LABELS = {
    "diffusion": "modèle de diffusion",
    "vae": "VAE vidéo",
    "audio_vae": "VAE audio",
    "text_encoder": "encodeur de prompt",
    "embeddings_connectors": "connecteurs d'embeddings",
}


def is_reference_model(model: registry.BaseModel) -> bool:
    """Modèle piloté par IMAGES DE RÉFÉRENCE (MiniMax-H3 Ref2VA).

    Ces modèles-là ne gèrent ni image de départ ni image de fin : c'est une
    contrainte du modèle, pas un choix d'interface."""
    return bool(model.defaults.get("reference"))


def modes_for(model: registry.BaseModel | None) -> tuple[str, ...]:
    """Modes réellement possibles avec CE modèle."""
    if model is None:
        return ("t2v",)
    if is_reference_model(model):
        return ("t2v", "ref")
    return ("t2v", "i2v", "flf2v")


def grid(model: registry.BaseModel | None) -> dict:
    """Contraintes de forme du modèle (alignement, paliers d'images, fps)."""
    d = model.defaults if model else {}
    return {
        "size_align": int(d.get("size_align", sdcpp.VAE_SCALE)),
        "frame_step": int(d.get("frame_step", sdcpp.FRAME_STEP)),
        "frame_base": int(d.get("frame_base", sdcpp.FRAME_BASE)),
        "fps": int(d.get("fps", 24) or 24),
        "fps_fixed": bool(d.get("fps_fixed")),
    }


def cancel() -> str:
    """Annule la génération vidéo en cours (process sd-cli)."""
    return sdcpp.cancel_active()


def available_models(prefs: dict | None = None) -> list[registry.BaseModel]:
    return registry.video_models(prefs or settings.load_prefs())


def engine_ready(model: registry.BaseModel | None = None) -> tuple[bool, str]:
    """(le moteur peut-il générer cette vidéo, message d'explication).

    Deux niveaux : le binaire sait-il faire de la vidéo du tout, puis connaît-il
    les options propres à CE modèle. Un sd-cli de juin sait faire du LTX mais
    ignore MiniMax-H3 : sans ce second contrôle, l'utilisateur téléchargerait
    35 Go avant de découvrir que son moteur ne peut pas les charger."""
    from ..i18n import t
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        return False, t("Binaire `sd-cli` introuvable. Lancez `install.bat` "
                        "(ou `python scripts/get_sdcpp.py`).")
    if not sdcpp.supports_video(sd_cli):
        return False, t("Votre `sd-cli` ne connaît pas le mode vidéo "
                        "(`-M vid_gen`). Mettez le moteur à jour avec "
                        "**update-engine.bat**.")
    if model is not None:
        absent = sdcpp.missing_options(
            sd_cli, list(model.defaults.get("engine_options") or []))
        if absent:
            return False, t("Votre `sd-cli` est trop ancien pour « {name} » "
                            "(option {opt} inconnue). Mettez le moteur à jour "
                            "avec **update-engine.bat** — inutile de "
                            "télécharger les poids avant.").format(
                name=model.name, opt="`" + "`, `".join(absent) + "`")
    return True, ""


def missing_parts(model: registry.BaseModel) -> list[str]:
    """Libellés lisibles des composants requis encore absents du disque.

    On s'appuie sur ce que le modèle DÉCLARE (composants non facultatifs) au
    lieu d'une liste figée : LTX a des connecteurs d'embeddings, MiniMax-H3
    n'en a pas."""
    from ..i18n import t
    return [t(_ROLE_LABELS.get(c.role, c.role))
            for c in registry.missing_components(model)]


def spatial_upscaler(model: registry.BaseModel) -> Path | None:
    """Upscaler latent ×2 de LTX, s'il a été téléchargé."""
    return _component(model, "spatial_upscaler")


def plan(width: int, height: int, frames: int, fps: int,
         model: registry.BaseModel | None = None) -> dict:
    """Ce qui sera RÉELLEMENT généré, après alignement sur la grille du modèle.

    L'utilisateur demande « 720p, 2 secondes » ; le modèle, lui, ne sait produire
    que des multiples de 32 px et des paquets d'images (8k+1 pour LTX-2.3,
    17k+5 pour MiniMax-H3), et certains imposent leur cadence. Autant le calculer
    et l'afficher avant de lancer plusieurs minutes de génération."""
    g = grid(model)
    w = sdcpp.snap_size(width, g["size_align"])
    h = sdcpp.snap_size(height, g["size_align"])
    n = sdcpp.snap_frames(frames, g["frame_step"], g["frame_base"])
    f = g["fps"] if g["fps_fixed"] else max(1, int(fps))
    return {"width": w, "height": h, "frames": n, "fps": f,
            "seconds": n / f,
            "fps_forced": g["fps_fixed"] and int(fps or f) != f,
            "snapped": (w != int(width) or h != int(height)
                        or n != int(frames))}


def describe(p: dict) -> str:
    from ..i18n import t
    return t("{w}×{h} · {n} images à {fps} i/s · {s} s").format(
        w=p["width"], h=p["height"], n=p["frames"], fps=p["fps"],
        s=f"{p['seconds']:.1f}")


def generate_video(
    model_id: str,
    prompt: str,
    negative: str = "",
    mode: str = "t2v",
    init_image: Path | None = None,
    end_image: Path | None = None,
    ref_images: "list[Path] | None" = None,
    width: int = 1280,
    height: int = 704,
    frames: int = 33,
    fps: int = 24,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    hires: bool = False,
    hires_steps: int = 4,
    save_prompt: bool = True,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Génère un clip .webm (vidéo + audio) et renvoie son chemin."""
    if mode not in MODES:
        raise sdcpp.EngineError(f"Mode vidéo inconnu : {mode}")

    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    if model is None or not model.is_video:
        raise sdcpp.EngineError(f"Modèle vidéo inconnu : {model_id}")

    ok, why = engine_ready(model)
    if not ok:
        raise sdcpp.EngineError(why)
    sd_cli = settings.find_sd_cli()

    if mode not in modes_for(model):
        raise sdcpp.EngineError(
            f"« {model.name} » ne gère pas ce mode. "
            + ("Ce modèle fonctionne par images de RÉFÉRENCE : il n'accepte ni "
               "image de départ ni image de fin."
               if is_reference_model(model) else
               "Ce modèle n'accepte pas d'images de référence — utilisez la "
               "variante Ref2VA pour ça."))

    absent = missing_parts(model)
    if absent:
        raise sdcpp.EngineError(
            f"« {model.name} » n'est pas installé — il manque : "
            f"{', '.join(absent)}.\nTéléchargez-le depuis l'onglet "
            "« 📚 Catalogue de modèles ».")

    # Le conditionnement n'a de sens que dans le mode qui l'utilise — et les
    # deux familles s'excluent : références OU image de départ/fin, jamais les
    # deux (contrainte MiniMax-H3 Ref2VA).
    ref_images = [Path(r) for r in (ref_images or []) if r]
    if mode == "t2v":
        init_image = end_image = None
        ref_images = []
    elif mode == "i2v":
        end_image = None
        ref_images = []
        if init_image is None:
            raise sdcpp.EngineError("Mode image → vidéo : fournissez l'image "
                                    "de départ.")
    elif mode == "flf2v":
        ref_images = []
        if init_image is None or end_image is None:
            raise sdcpp.EngineError("Mode début → fin : fournissez les deux "
                                    "images (départ et fin).")
    else:  # ref
        init_image = end_image = None
        if not ref_images:
            raise sdcpp.EngineError("Mode référence : fournissez au moins une "
                                    "image de référence.")

    d = model.defaults
    flags, gpu_index = _resolved_flags(prefs)
    # LTX-2.3 en 22 B ne tient dans 11-12 Go QUE par décharge en RAM, et le
    # décodage VAE d'une vidéo est le pic mémoire le plus violent du pipeline.
    # On force donc les deux, quelles que soient les préférences : sans ça,
    # l'utilisateur récolte un OOM au lieu d'un clip.
    flags = {**flags, "offload_to_cpu": True, "vae_tiling": True}

    auto_fit = bool(prefs.get("auto_fit"))
    enc_gpu = prefs.get("encoder_gpu_index")
    split_gpu = (not auto_fit) and enc_gpu is not None and enc_gpu != gpu_index
    all_gpus = auto_fit or split_gpu
    if auto_fit:
        # auto-fit place tout en VRAM et ignore --offload-to-cpu : incompatible
        # avec un 22 B sur 11-12 Go, mais c'est un choix explicite de l'utilisateur.
        flags = {**flags, "offload_to_cpu": False}

    up = spatial_upscaler(model) if hires else None
    if hires and up is None and log:
        log("⚠️ Upscaler spatial LTX absent : « Détail ×2 » ignoré. "
            "Re-téléchargez le modèle pour l'obtenir.")

    g = grid(model)
    req = VidRequest(
        diffusion_model=_component(model, "diffusion"),
        vae=_component(model, "vae"),
        audio_vae=_component(model, "audio_vae"),
        text_encoder=_component(model, "text_encoder"),
        connectors=_component(model, "embeddings_connectors"),
        llm_vision=_component(model, "text_encoder_vision"),
        prompt=prompt or "",
        negative=negative if negative is not None else d.get("negative", ""),
        steps=int(steps if steps is not None else d.get("steps", 8)),
        cfg_scale=float(cfg_scale if cfg_scale is not None
                        else d.get("cfg_scale", 1.0)),
        sampler=d.get("sampler") or "euler",
        schedule="" if d.get("scheduler") in (None, "", "auto")
                 else d["scheduler"],
        width=int(width), height=int(height),
        frames=int(frames),
        # Certains modèles imposent leur cadence (MiniMax-H3 : 24 i/s) et
        # écrasent toute autre valeur ; autant l'envoyer juste du premier coup.
        fps=(g["fps"] if g["fps_fixed"] else int(fps)),
        seed=int(seed),
        init_image=init_image, end_image=end_image, ref_images=ref_images,
        size_align=g["size_align"], frame_step=g["frame_step"],
        frame_base=g["frame_base"], rng=(model.defaults.get("rng") or ""),
        hires_upscaler=up, hires_steps=int(hires_steps),
        flags=flags, gpu_index=gpu_index,
        encoder_gpu_index=enc_gpu if split_gpu else None,
        auto_fit=auto_fit, split_mode=prefs.get("split_mode") or "",
    )

    out = sdcpp.unique_output("ltx", ext="webm")
    cmd = sdcpp.build_vid_cmd(sd_cli, req, out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index, all_gpus=all_gpus)

    if not out.is_file():
        # sd.cpp remplace l'extension quand le conteneur demandé ne convient
        # pas ; on récupère alors ce qu'il a effectivement écrit.
        found = sorted(out.parent.glob(f"{out.stem}.*"))
        found = [f for f in found if f.suffix.lower() != ".txt"]
        if not found:
            raise sdcpp.EngineError(
                "LTX-2.3 n'a produit aucune vidéo. Voir le journal : c'est "
                "presque toujours un manque de mémoire (baissez la résolution "
                "ou la durée).")
        out = found[0]

    if save_prompt:
        _sidecar(out, req, model, mode, up is not None)
    return out


def _sidecar(path: Path, req: VidRequest, model: registry.BaseModel,
             mode: str, hires: bool) -> None:
    """Journal .txt à côté du clip, comme pour les images et les GLB."""
    p = plan(req.width, req.height, req.frames, req.fps, model)
    lines = [
        req.prompt or "",
        f"Negative prompt: {req.negative}" if req.negative else "",
        f"Modèle: {model.name} ({model.id})",
        f"Mode: {mode}",
        (f"Références: {', '.join(Path(r).name for r in req.ref_images)}"
         if req.ref_images else ""),
        f"Vidéo: {describe(p)}",
        f"Steps: {req.steps}, CFG: {req.cfg_scale}, Sampler: {req.sampler}, "
        f"Seed: {req.seed}",
        f"Détail ×2 (upscaler latent): {'oui' if hires else 'non'}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    try:
        path.with_suffix(".txt").write_text(
            "\n".join(x for x in lines if x), encoding="utf-8")
    except OSError:
        pass
