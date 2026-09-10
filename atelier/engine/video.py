"""Génération de VIDÉO — même moteur, autre mode.

Ce fichier a déjà existé, pour LTX-2.3 puis MiniMax-H3, et il a été retiré les
deux fois. La raison était toujours la même, et elle n'était pas l'interface :
l'encodeur de texte de ces modèles ne tenait pas sur une carte de 12 Go, donc
il n'y avait rien à afficher. La sonde MiniMax est d'ailleurs restée dans le
dépôt le temps de répondre à cette question-là, et sa réponse était non.

Wan 2.2 TI2V 5B change le calcul, et il faut voir les trois chiffres ensemble :

    diffusion Q4_K_M   3,4 Go   sur la carte
    VAE 2.2            1,4 Go   sur la carte
    umt5-xxl Q4_K_M    3,7 Go   en RAM (`te=cpu`), comme tous nos encodeurs

La carte porte donc **4,8 Go de poids**, pas 8,5. C'est ce qui fait tenir la
vidéo sur une 2080 Ti de 11 Go aussi bien que sur une 3060 de 12 Go — la même
répartition que pour les images, pour la même raison.

Restait le VAE, dont la doc amont dit qu'il « demande vraiment beaucoup de
VRAM ». Cet avertissement est plus vieux que le code : sd.cpp sait maintenant
reprendre un décodage raté en tuilant AUSSI dans le temps, pas seulement dans
l'espace (`prepare_vae_decode_retry_tiling`, PR #1926 et #1932). Un moteur à
jour se rattrape donc tout seul là où l'avertissement disait d'aller chercher
un TAE dégradé.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import registry, settings
from . import sdcpp
from .generate import _resolved_flags, resolve_model_files
from .sdcpp import VidRequest

#  Le négatif de la doc Wan, en chinois dans le texte — et ce n'est pas une
#  coquetterie. Wan est entraîné bilingue et ses propres exemples utilisent
#  cette chaîne : traduite, elle ne désigne plus les mêmes régions de son
#  espace de représentation. On la garde donc telle quelle, avec sa traduction
#  en commentaire pour qui veut la modifier.
#
#  « couleurs criardes, surexposé, statique, détails flous, sous-titres, style,
#    œuvre, peinture, image, immobile, gris d'ensemble, pire qualité, basse
#    qualité, résidus de compression JPEG, laid, mutilé, doigts en trop, mains
#    mal dessinées, visage mal dessiné, difforme, défiguré, membres déformés,
#    doigts fusionnés, image figée, arrière-plan encombré, trois jambes,
#    foule en arrière-plan, marche à reculons »
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def default_model_id() -> str | None:
    """Le premier modèle vidéo du catalogue, ou None s'il n'y en a pas."""
    models = registry.load_video_models(settings.load_prefs())
    return models[0].id if models else None


def is_ready(model_id: str | None = None) -> bool:
    prefs = settings.load_prefs()
    model_id = model_id or default_model_id()
    if not model_id:
        return False
    model = registry.get_base_model(model_id, prefs)
    return bool(model) and registry.model_is_ready(model)


def engine_ready() -> str:
    """"" si le moteur installé sait faire de la vidéo, sinon POURQUOI pas.

    Une option absente doit dire ce qui l'empêche et le geste qui la débloque,
    pas disparaître : on cherche sinon dans l'interface un bouton dont on vient
    de lire la description.
    """
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        return ("⚠️ The `sd-cli` engine is not installed. Run `install.bat`.")
    if not sdcpp.video_supported(sd_cli):
        return ("⚠️ **Your engine cannot generate video yet** — it does not "
                "know the `--video-frames` option. Run **`update.bat`**: it "
                "updates the engine right after the code.")
    return ""


def frames_for(seconds: float, fps: int) -> int:
    """Nombre d'images aligné 4n+1 pour une durée demandée."""
    return sdcpp.align_video_frames(max(1, round(seconds * fps)))


def generate_video(
    prompt: str,
    negative: str = "",
    model_id: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int = -1,
    video_frames: int | None = None,
    fps: int | None = None,
    flow_shift: float | None = None,
    start_image: Path | None = None,
    output_format: str = "webm",
    log: Callable[[str], None] | None = None,
) -> Path:
    """Fabrique une vidéo et renvoie le fichier produit.

    Le placement mémoire n'est PAS recalculé ici : `_resolved_flags` et
    `memory_args` sont ceux des images. C'est le même moteur sur la même carte,
    et une seconde façon de décider où vont les poids finirait par diverger de
    la première — c'est exactement ce qui était arrivé entre la ligne de
    commande et l'ancien serveur résident.
    """
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError("The sd-cli binary was not found. "
                                "Run install.bat.")
    model_id = model_id or default_model_id()
    if not model_id:
        raise sdcpp.EngineError("No video model in the catalog.")
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Unknown model: {model_id}")
    if model.kind != registry.VIDEO:
        raise sdcpp.EngineError(f"“{model.name}” is not a video model.")

    files = resolve_model_files(model)
    flags, gpu_index = _resolved_flags(prefs)
    d = dict(model.defaults)

    #  Le VAE en tuiles est FORCÉ, pas suggéré. Sur une vidéo, le décodage est
    #  le seul moment où l'on manipule toutes les images à la fois : c'est là
    #  que la mémoire lâche, et c'est le seul poste dont le coût ne se voit pas
    #  dans la taille des poids. Le laisser au profil automatique reviendrait à
    #  faire dépendre le succès d'un réglage pensé pour une image unique.
    flags = dict(flags)
    flags["vae_tiling"] = True

    frames = sdcpp.align_video_frames(
        int(video_frames or d.get("video_frames", 33) or 33))
    rate = int(fps or d.get("fps", 24) or 24)

    ext = output_format if output_format in ("webm", "avi", "png") else "webm"
    if ext == "png":
        #  Suite d'images : sd.cpp remplit lui-même le %04d. UN DOSSIER par
        #  vidéo — sinon 121 fichiers viennent se mêler aux images du jour dans
        #  outputs/, et on ne retrouve plus ni les unes ni les autres.
        folder = sdcpp.unique_output("video", "png").with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "frame_%04d.png"
    else:
        output = sdcpp.unique_output("video", ext)

    req = VidRequest(
        diffusion_model=files["diffusion"], vae=files["vae"],
        t5xxl=files["t5xxl"], text_encoder=files["enc"],
        prompt=prompt,
        negative=negative if negative is not None else "",
        steps=int(steps or d.get("steps", 20) or 20),
        cfg_scale=float(cfg_scale if cfg_scale is not None
                        else d.get("cfg_scale", 6.0)),
        sampler=d.get("sampler", "euler"),
        schedule=("" if d.get("scheduler", "auto") == "auto"
                  else d.get("scheduler", "")),
        flow_shift=float(flow_shift if flow_shift is not None
                         else d.get("flow_shift", 0.0) or 0.0),
        width=int(width or d.get("width", 704)),
        height=int(height or d.get("height", 1280)),
        seed=int(seed),
        video_frames=frames, fps=rate,
        start_image=Path(start_image) if start_image else None,
        flags=flags, gpu_index=gpu_index,
        params_backend=(prefs.get("params_backend") or ""),
        auto_fit=bool(prefs.get("auto_fit")),
        encoder_gpu_index=prefs.get("encoder_gpu_index"),
        #  Un BUDGET par défaut, contrairement aux images. Le décodage du
        #  VAE d'une vidéo est le seul moment où toutes les images existent en
        #  même temps : sans budget, le moteur découpe son graphe sans cible et
        #  découvre au milieu qu'il ne tient pas — le mode d'échec qu'on a
        #  passé deux jours à diagnostiquer sur les images. Un budget CHOISI
        #  reste prioritaire : celui-là est une décision.
        max_vram=sdcpp.max_vram_arg(
            prefs.get("max_vram") or sdcpp.MAX_VRAM_AUTO),
        lora_dir=settings.LORA_DIR if settings.LORA_DIR.is_dir() else None,
    )

    if log:
        seconds = frames / float(rate)
        log(f"🎬 {frames} frames at {rate} fps — {seconds:.1f} s of video, "
            f"{req.width}×{req.height}, {req.steps} steps.")
        if frames != int(video_frames or frames):
            log(f"ℹ️ Frame count aligned to {frames}: Wan's temporal VAE "
                "works in groups of four plus one.")

    cmd = sdcpp.build_vid_cmd(sd_cli, req, output)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)

    if ext == "png":
        produced = sorted(output.parent.glob("frame_*.png"))
        if not produced:
            raise sdcpp.EngineError("The engine wrote no frame.")
        if log:
            log(f"🖼️ {len(produced)} frames written to {output.parent}")
        return output.parent
    if not output.is_file():
        raise sdcpp.EngineError(f"The engine wrote no file to {output}.")
    return output
