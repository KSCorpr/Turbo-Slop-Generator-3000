"""Constructeur de prompt structuré pour Ideogram 4.

Reproduit le format JSON de l'éditeur d-daley/ideogram4-editor :
    {
      "high_level_description": "...",
      "style_description": {aesthetics, lighting, medium, art_style|photo, color_palette[]},
      "compositional_deconstruction": {background, elements[{type,bbox,text?,desc,color_palette?}]}
    }
Les bbox sont en coordonnées normalisées 0–1000, au format [y1, x1, y2, x2].
"""
from __future__ import annotations

import json
import re


def parse_colors(raw: str | None) -> list[str]:
    """Extrait des couleurs hex (#RRGGBB) d'une chaîne libre, max 16."""
    if not raw:
        return []
    found = re.findall(r"#?[0-9a-fA-F]{6}", raw)
    out = []
    for c in found:
        c = c if c.startswith("#") else "#" + c
        out.append(c.upper())
    return out[:16]


def _clamp1000(v) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(1000, n))


def build_prompt(
    mode: str,
    high_level: str,
    aesthetics: str,
    lighting: str,
    medium: str,
    style_or_photo: str,
    background: str,
    global_colors: str,
    element_rows: list[list],
) -> str:
    """Assemble le prompt JSON Ideogram 4. `mode` = 'photo' ou 'art_style'."""
    style: dict = {
        "aesthetics": aesthetics or "",
        "lighting": lighting or "",
        "medium": medium or "",
    }
    style["photo" if mode == "photo" else "art_style"] = style_or_photo or ""
    palette = parse_colors(global_colors)
    if palette:
        style["color_palette"] = palette

    elements: list[dict] = []
    for row in element_rows or []:
        row = list(row) + [""] * (8 - len(row))  # complète si tronqué
        etype, text, desc, x1, y1, x2, y2, colors = row[:8]
        etype = (str(etype).strip().lower() or "obj")
        etype = "text" if etype.startswith("t") else "obj"
        desc = (desc or "")
        text = (text or "")
        # Ligne vide -> ignorée
        if not str(desc).strip() and not str(text).strip():
            continue
        el: dict = {
            "type": etype,
            "bbox": [_clamp1000(y1), _clamp1000(x1), _clamp1000(y2), _clamp1000(x2)],
            "desc": str(desc),
        }
        if etype == "text":
            el["text"] = str(text)
        ecols = parse_colors(colors if isinstance(colors, str) else "")
        if ecols:
            el["color_palette"] = ecols[:5]
        elements.append(el)

    obj = {
        "high_level_description": high_level or "",
        "style_description": style,
        "compositional_deconstruction": {
            "background": background or "",
            "elements": elements,
        },
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


# Exemple pré-rempli pour le tableau des éléments (colonnes ci-dessous).
EXAMPLE_ELEMENTS = [
    ["text", "FEDERALL", "titre en haut, lettres dorées", 50, 100, 200, 900, "#E7C84B"],
    ["obj", "", "chat roux pelucheux au centre", 300, 300, 850, 700, ""],
]
ELEMENT_HEADERS = ["type (obj/text)", "texte", "description",
                   "x1", "y1", "x2", "y2", "couleurs (#hex)"]
