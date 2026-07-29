"""Outpaint directionnel (façon Midjourney) : étendre une image à gauche,
à droite, en haut, en bas — ou tout autour.

Méthode volontairement INDÉPENDANTE du modèle et du prompt :

1. on agrandit la toile dans les directions choisies ;
2. les nouvelles zones sont pré-remplies par **effet miroir** des bords (la
   continuité des couleurs/textures donne au modèle un point de départ
   plausible, bien meilleur qu'un flou ou du bruit) ;
3. on passe le tout en **img2img** avec le modèle choisi ;
4. on **recolle l'original** par-dessus, avec un **fondu** sur quelques pixels.

L'étape 4 garantit que la zone d'origine est préservée au pixel près, sans
dépendre d'un masque d'inpainting côté moteur — donc ça fonctionne avec
n'importe quel modèle du catalogue.
"""
from __future__ import annotations

from pathlib import Path

DIRECTIONS = ["left", "right", "top", "bottom"]


def plan(size: tuple[int, int], directions: list[str], amount: float,
         multiple: int = 16, max_side: int = 2048) -> dict:
    """Calcule la nouvelle toile et la position de l'original.

    `amount` = proportion de la dimension d'origine ajoutée par côté (0.25 = +25 %).
    La toile est alignée sur `multiple` (contrainte des modèles) et plafonnée.
    """
    w, h = size
    dirs = [d for d in (directions or []) if d in DIRECTIONS]
    pad_x = int(round(w * amount)) if amount > 0 else 0
    pad_y = int(round(h * amount)) if amount > 0 else 0
    left = pad_x if "left" in dirs else 0
    right = pad_x if "right" in dirs else 0
    top = pad_y if "top" in dirs else 0
    bottom = pad_y if "bottom" in dirs else 0

    new_w, new_h = w + left + right, h + top + bottom
    # Plafond : on réduit les marges proportionnellement si on dépasse.
    def _clamp(new, orig, a, b):
        if new <= max_side:
            return a, b, new
        excess = new - max_side
        total = a + b
        if total <= 0:
            return a, b, new
        a2 = max(0, a - int(round(excess * a / total)))
        b2 = max(0, b - (excess - (a - a2)))
        return a2, b2, orig + a2 + b2

    left, right, new_w = _clamp(new_w, w, left, right)
    top, bottom, new_h = _clamp(new_h, h, top, bottom)

    # Alignement sur `multiple` : on rallonge la marge existante (jamais 0, pour
    # ne pas déplacer l'original si un côté n'est pas étendu).
    def _snap(new, a, b):
        rem = new % multiple
        if rem:
            add = multiple - rem
            if b > 0:
                b += add
            elif a > 0:
                a += add
            else:
                return a, b, new      # aucune extension sur cet axe
            new += add
        return a, b, new

    left, right, new_w = _snap(new_w, left, right)
    top, bottom, new_h = _snap(new_h, top, bottom)
    return {"left": left, "right": right, "top": top, "bottom": bottom,
            "width": new_w, "height": new_h, "orig": (w, h)}


def build_canvas(img, p: dict):
    """Toile agrandie, nouvelles zones remplies par effet MIROIR des bords."""
    from PIL import Image, ImageOps
    src = img.convert("RGB")
    w, h = src.size
    left, top = p["left"], p["top"]
    right, bottom = p["right"], p["bottom"]
    if not any((left, right, top, bottom)):
        return src.copy()
    # ImageOps.expand ne sait pas faire du miroir : on assemble à la main en
    # réfléchissant des bandes prélevées sur les bords.
    canvas = Image.new("RGB", (p["width"], p["height"]))
    canvas.paste(src, (left, top))

    def _mirror_h(box_w: int, from_left: bool):
        if box_w <= 0:
            return None
        band = src.crop((0, 0, min(box_w, w), h)) if from_left else \
            src.crop((max(0, w - box_w), 0, w, h))
        band = ImageOps.mirror(band)
        if band.width < box_w:      # marge plus large que l'image : on répète
            band = band.resize((box_w, h))
        return band

    if left:
        b = _mirror_h(left, True)
        canvas.paste(b, (left - b.width, top))
    if right:
        b = _mirror_h(right, False)
        canvas.paste(b, (left + w, top))
    # Bandes verticales : on réfléchit la bande horizontale DÉJÀ complétée pour
    # que les coins soient cohérents.
    mid = canvas.crop((0, top, p["width"], top + h))
    if top:
        band = mid.crop((0, 0, p["width"], min(top, h)))
        band = ImageOps.flip(band)
        if band.height < top:
            band = band.resize((p["width"], top))
        canvas.paste(band, (0, top - band.height))
    if bottom:
        band = mid.crop((0, max(0, h - bottom), p["width"], h))
        band = ImageOps.flip(band)
        if band.height < bottom:
            band = band.resize((p["width"], bottom))
        canvas.paste(band, (0, top + h))
    return canvas


def composite_back(generated, original, p: dict, feather: int = 24):
    """Recolle l'original sur le résultat, avec un fondu sur `feather` pixels.

    Garantit que la zone d'origine est PRÉSERVÉE, quel que soit le modèle et
    sans masque côté moteur. Le fondu évite une couture visible.
    """
    from PIL import Image, ImageFilter
    out = generated.convert("RGB").copy()
    src = original.convert("RGB")
    w, h = p["orig"]
    left, top = p["left"], p["top"]
    if out.size != (p["width"], p["height"]):
        out = out.resize((p["width"], p["height"]))

    # Masque : blanc = on garde l'original. On rétrécit puis on floute pour
    # obtenir un dégradé UNIQUEMENT vers l'intérieur de la zone d'origine.
    mask = Image.new("L", (p["width"], p["height"]), 0)
    f = max(0, int(feather))
    inner = (left + f, top + f, left + w - f, top + h - f)
    if inner[2] > inner[0] and inner[3] > inner[1]:
        mask.paste(255, inner)
        if f:
            mask = mask.filter(ImageFilter.GaussianBlur(f / 2.0))
    else:                                  # image trop petite pour le fondu
        mask.paste(255, (left, top, left + w, top + h))
    out.paste(src, (left, top), mask.crop((left, top, left + w, top + h)))
    return out


def describe(p: dict) -> str:
    parts = [f"{k} +{p[k]}px" for k in DIRECTIONS if p.get(k)]
    return (f"{p['orig'][0]}×{p['orig'][1]} → {p['width']}×{p['height']}"
            + (" (" + ", ".join(parts) + ")" if parts else " (aucune extension)"))
