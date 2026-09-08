"""Les fonctions qui ne passaient PAS par `generate()` — portées ici.

L'outpaint, lui, n'avait rien à porter : il appelle déjà `generate()` avec une
image de départ et un masque, donc il a changé de moteur en même temps que
l'aiguille. C'est ce que vaut un point de passage unique.

Restaient trois choses qui parlaient à sd.cpp directement :

* **la passe HD** — `sd-cli --hires`, une commande qui agrandit puis redébruite
  en interne. Ici c'est reconstruit en clair : agrandissement puis img2img sur
  le pipeline déjà chargé. Le déroulé est le même, mais il est VISIBLE, et le
  facteur n'est plus limité par la capacité du moteur à découper son graphe —
  c'est le plan de placement qui s'en charge.
* **ADetailer** — `sd-cli -M adetailer`. La détection change de format et il
  faut le dire : sd.cpp lit un `.safetensors` converti, Ultralytics veut le
  `.pt` d'origine. Les deux moteurs ne peuvent pas partager le même fichier.
* **l'agrandissement ESRGAN** — les poids sont en GGUF, que seul sd.cpp lit.
  La fonction existe toujours et refuse clairement, en pointant les upscalers
  modernes qui, eux, tournent déjà sous PyTorch (spandrel) et rendent mieux.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .. import registry, settings
from . import backend, catalog

#  Côté sd.cpp la passe HD plafonne le côté long parce que le modèle quitte son
#  échelle d'entraînement au-delà. Ce plafond-là n'a rien de spécifique au
#  moteur : c'est une propriété du modèle, donc il reste.
HD_MAX_SIDE = 3072
#  Grille d'alignement commune aux trois familles (16 = plus petit multiple
#  commun de leurs facteurs VAE × patch). Même raison que côté sd.cpp.
HD_ALIGN = 16


def _align_up(v: float, step: int = HD_ALIGN) -> int:
    return int((int(v) + step - 1) // step * step)


def supports_mask(model_id: str) -> bool:
    """Ce moteur sait-il inpeindre AVEC un masque pour ce modèle ?

    La question était posée au binaire (`sdcpp.mask_flag`) ; sur cette branche
    il n'y a pas de binaire, et la réponse dépend du modèle : diffusers publie
    une classe d'inpainting pour Z-Image et Flux.2 Klein, aucune pour Krea 2.
    Sans cette fonction, l'outpaint retombait en img2img sur TOUS les modèles —
    y compris ceux qui savent faire mieux.
    """
    model = catalog.get(model_id)
    return bool(model and model.can(catalog.INPAINT))


# --------------------------------------------------------------------------- #
#  Passe HD
# --------------------------------------------------------------------------- #
def hd_upscale(model_id: str, image, scale: float = 2.0,
               upscaler: str = "Latent", denoise: float = 0.4,
               prompt: str = "", negative: str = "", steps: int = 0,
               hd_steps: int = 0, seed: int = -1,
               preview_path: Path | None = None,
               log: Callable[[str], None] | None = None) -> list[Path]:
    """Agrandir, puis laisser le modèle redessiner le détail à la taille finale.

    Signature identique à `engine.generate.hd_upscale`. Le déroulé aussi, à un
    détail près qui est en fait une amélioration : l'agrandissement est fait
    ici, avec le rééchantillonneur choisi, avant d'être passé au modèle. Côté
    sd.cpp il était interne, donc invisible et non réutilisable.

    `denoise` reste le seul réglage qui compte. Trop bas, l'agrandissement
    reste flou ; trop haut, le modèle réinvente au lieu de préciser.
    """
    from PIL import Image
    model = catalog.get(model_id)
    if model is None:
        raise RuntimeError(f"“{model_id}” has no PyTorch entry.")
    if not model.can(catalog.IMAGE_TO_IMAGE):
        raise RuntimeError(
            f"“{model_id}” has no image-to-image pipeline in diffusers, so it "
            "cannot redraw an enlarged image. Use Z-Image Turbo for the HD "
            "pass, or the modern upscalers in the Toolkit.")

    im = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) \
        else image.convert("RGB")
    ow, oh = im.size

    scale = float(scale)
    if max(ow, oh) * scale > HD_MAX_SIDE:
        scale = max(1.25, HD_MAX_SIDE / max(ow, oh))
        if log:
            log(f"[hd] factor lowered to ×{scale:.2f} to stay under "
                f"{HD_MAX_SIDE} px on a side.")
    tw, th = _align_up(ow * scale), _align_up(oh * scale)

    big = _enlarge(im, tw, th, upscaler, log)
    settings.ensure_dirs()
    staged = settings.TMP_DIR / f"hd_in_{int(time.time() * 1000)}.png"
    big.save(staged)

    d = dict((registry.get_base_model(model_id, settings.load_prefs())
              or _Empty()).defaults or {})
    if log:
        log(f"[hd] {ow}×{oh} → {tw}×{th} · denoise {denoise:g} · "
            f"{int(hd_steps or steps or d.get('steps', 8) or 8)} steps")
    return backend.generate(
        model_id=model_id, prompt=prompt or "", negative=negative,
        steps=int(hd_steps or steps or d.get("steps", 8) or 8),
        cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
        width=tw, height=th, seed=int(seed), batch_count=1,
        sampler=d.get("sampler"), schedule=d.get("scheduler", "auto"),
        flow_shift=float(d.get("flow_shift", 0.0) or 0.0),
        init_image=staged, strength=float(denoise),
        preview_path=preview_path, save_prompt=False, log=log)


class _Empty:
    defaults: dict = {}


def _enlarge(im, tw: int, th: int, upscaler: str, log=None):
    """L'agrandissement AVANT le redébruitage.

    « Latent » n'a plus de sens hors de sd.cpp — il n'y a pas ici de latent à
    agrandir avant d'entrer dans le sampler. On le traite comme « Lanczos »,
    en le disant : c'est le rééchantillonneur le plus net des classiques, et
    de toute façon le modèle va redessiner par-dessus.

    Un vrai modèle d'agrandissement (DAT, SPAN…) est utilisé s'il est nommé et
    installé : il donne au modèle une base bien plus propre à préciser, donc
    un `denoise` plus bas suffit et l'image d'origine est moins réinventée.
    """
    from PIL import Image
    name = (upscaler or "Latent").strip()
    if name.lower() in ("", "latent", "none", "lanczos"):
        if name.lower() == "latent" and log:
            log("[hd] “Latent” is a stable-diffusion.cpp notion — enlarging "
                "with Lanczos instead, then letting the model redraw.")
        return im.resize((tw, th), Image.LANCZOS)

    weights = registry.upscaler_path(name)
    if weights is None or registry.upscaler_engine(name) != "spandrel":
        if log:
            log(f"[hd] “{name}” is not usable on this engine (GGUF upscalers "
                "are read by stable-diffusion.cpp only) — Lanczos instead.")
        return im.resize((tw, th), Image.LANCZOS)

    from ..engine import tools
    staged = settings.TMP_DIR / f"hd_pre_{int(time.time() * 1000)}.png"
    settings.ensure_dirs()
    im.save(staged)
    if log:
        log(f"[hd] enlarging with “{name}” before the redraw.")
    out = tools.modern_upscale(staged, name, log=log)
    big = Image.open(out).convert("RGB")
    # Le facteur vient du modèle et tombe rarement juste : on RÉDUIT vers la
    # cible (du suréchantillonnage, propre) et jamais l'inverse.
    return big if big.size == (tw, th) else big.resize((tw, th), Image.LANCZOS)


# --------------------------------------------------------------------------- #
#  ADetailer
# --------------------------------------------------------------------------- #
#  Ultralytics dépicklise un `.pt`, donc l'exécute. Même dépôt que la version
#  sd.cpp (`Bingsu/adetailer`), même fichiers d'origine — c'est la conversion
#  en safetensors qui n'a plus lieu d'être, pas la source.
DETECTOR_SUFFIX = ".pt"


def detector_for(name: str) -> Path:
    """Le `.pt` correspondant au détecteur nommé côté sd.cpp.

    Les deux moteurs ne peuvent pas partager le fichier : sd.cpp exige un
    `.safetensors` converti (les `.pt` sont des pickles, il refuse de les
    exécuter), Ultralytics ne lit que le `.pt`. On garde donc les mêmes NOMS
    et on change l'extension, pour que l'interface reste identique.
    """
    from ..engine.tools import ADETAILER_DIR
    return ADETAILER_DIR / (Path(name).stem + DETECTOR_SUFFIX)


def adetailer_reason() -> str:
    """Ce qui manque, nommé — ou "" si tout est prêt."""
    from importlib.util import find_spec
    from ..engine.tools import ADETAILER_DIR, ADETAILER_MODELS
    if find_spec("ultralytics") is None:
        return ("ADetailer on this engine needs the “ultralytics” package "
                "(AGPL-3.0). Install it with setup-adetailer.bat.")
    present = [n for n, _ in ADETAILER_MODELS
               if detector_for(n).is_file()]
    if not present:
        return (f"No detector in {ADETAILER_DIR}. Run setup-adetailer.bat — "
                "on this engine it keeps the original .pt files rather than "
                "converting them.")
    return ""


def detect(image_path: Path, detector: Path, confidence: float,
           only_largest: int = 0, log=None) -> list[tuple[int, int, int, int]]:
    """Les boîtes trouvées, de la plus grande à la plus petite.

    Séparé du redessin exprès : c'est la moitié qui peut se vérifier sans
    charger un modèle de diffusion, et la moitié dont on veut pouvoir dire
    « il n'a rien trouvé » plutôt que de rendre l'image inchangée en silence.
    """
    from ultralytics import YOLO
    model = YOLO(str(detector))
    result = model(str(image_path), conf=float(confidence), verbose=False)[0]
    boxes = [tuple(int(v) for v in b) for b in
             result.boxes.xyxy.cpu().numpy().tolist()]
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    if only_largest and int(only_largest) > 0:
        boxes = boxes[:int(only_largest)]
    if log:
        log(f"[adetailer] {len(boxes)} area(s) detected.")
    return boxes


def _expand(box, padding: int, size) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = size
    return (max(0, x1 - padding), max(0, y1 - padding),
            min(w, x2 + padding), min(h, y2 + padding))


def adetailer(model_id: str, image, detector: str, prompt: str = "",
              negative: str = "", denoise: float = 0.4, steps: int = 0,
              confidence: float = 0.3, padding: int = 32,
              mask_blur: int = 4, only_largest: int = 0, seed: int = -1,
              log: Callable[[str], None] | None = None) -> Path:
    """Détecte, redessine chaque zone, recolle — visage par visage.

    sd.cpp faisait les trois en une commande. Ici c'est explicite, et le
    recollage est fait avec un fondu comme celui de l'outpaint : sans lui, un
    visage redessiné laisse un rectangle net autour de lui, ce qui se voit plus
    que le défaut qu'on corrigeait.
    """
    from PIL import Image, ImageDraw, ImageFilter
    reason = adetailer_reason()
    if reason:
        raise RuntimeError(reason)
    model = catalog.get(model_id)
    if model is None or not model.can(catalog.IMAGE_TO_IMAGE):
        raise RuntimeError(
            f"“{model_id}” has no image-to-image pipeline in diffusers, so it "
            "cannot redraw a detected area.")

    settings.ensure_dirs()
    src = Path(image) if isinstance(image, (str, Path)) else None
    im = (Image.open(src) if src else image).convert("RGB")
    if src is None:
        src = settings.TMP_DIR / f"adetail_in_{int(time.time() * 1000)}.png"
        im.save(src)

    boxes = detect(Path(src), detector_for(detector), confidence,
                   only_largest, log)
    if not boxes:
        raise RuntimeError(
            "ADetailer found nothing to redraw. Lower the confidence, or pick "
            "a detector that matches what you are after (faces vs hands).")

    d = dict((registry.get_base_model(model_id, settings.load_prefs())
              or _Empty()).defaults or {})
    out = im.copy()
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = _expand(box, int(padding), im.size)
        # Le modèle travaille sur sa grille : on lui donne une découpe alignée
        # et carrée-ish, puis on remet à la taille de la zone au recollage.
        crop = out.crop((x1, y1, x2, y2))
        cw, ch = _align_up(crop.width), _align_up(crop.height)
        staged = settings.TMP_DIR / f"adetail_crop_{i}.png"
        crop.resize((cw, ch), Image.LANCZOS).save(staged)
        if log:
            log(f"[adetailer] area {i}/{len(boxes)} · "
                f"{crop.width}×{crop.height} at ({x1}, {y1})")
        redrawn = backend.generate(
            model_id=model_id, prompt=prompt or "", negative=negative,
            steps=int(steps or d.get("steps", 8) or 8),
            cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
            width=cw, height=ch, seed=int(seed) + i, batch_count=1,
            sampler=d.get("sampler"), schedule=d.get("scheduler", "auto"),
            flow_shift=float(d.get("flow_shift", 0.0) or 0.0),
            init_image=staged, strength=float(denoise),
            save_prompt=False, log=log)
        patch = Image.open(redrawn[0]).convert("RGB").resize(crop.size,
                                                            Image.LANCZOS)
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).rectangle(
            [int(padding / 2), int(padding / 2),
             crop.width - int(padding / 2), crop.height - int(padding / 2)],
            fill=255)
        if mask_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(int(mask_blur)))
        out.paste(patch, (x1, y1), mask)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = settings.OUTPUT_DIR / f"adetail-{stamp}.png"
    out.save(dest)
    return dest


# --------------------------------------------------------------------------- #
#  Agrandissement ESRGAN (GGUF)
# --------------------------------------------------------------------------- #
def upscale_image(image, model_name: str, repeats: int = 1,
                  log: Callable[[str], None] | None = None) -> Path:
    """Refuse, et dit vers quoi aller.

    Les upscalers ESRGAN du catalogue sont des GGUF, un format que seul sd.cpp
    lit. Convertir n'aurait aucun intérêt : ce sont des RRDBNet de 2018, et les
    upscalers modernes du catalogue (DAT, SPAN, PLKSR…) tournent déjà sous
    PyTorch via spandrel — ils sont la RAISON pour laquelle ils ont été ajoutés.
    """
    raise RuntimeError(
        f"“{model_name}” is a GGUF upscaler, read by stable-diffusion.cpp "
        "only. On the PyTorch engine use a modern upscaler (Toolkit → "
        "Enlarge): same click, and it does not leave the painted look this "
        "one does.")
