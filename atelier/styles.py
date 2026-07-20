"""Préréglages de prompt système / style, persistés en JSON.

Un style est un simple texte (préfixe appliqué en tête du prompt) enregistré
sous un nom. Les styles sont GLOBAUX : enregistrés une fois, ils sont
disponibles dans tous les onglets de modèle.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from . import settings

STYLES_FILE = settings.USERDATA_DIR / "style_presets.json"

# Banques de styles Krea (cumulables), fournies avec l'app :
#  • PHOTO : styles photographiques (qualité, lumière, objectif, pellicule…),
#    AVEC négatifs. Source github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2 (MIT, ghleg).
#  • ART   : styles artistiques (anime, cartoon, BD, dessin, design, peinture…),
#    SANS négatifs. Collection Krea fournie par l'utilisateur (provenance à confirmer).
PHOTO_STYLES_FILE = settings.CONFIG_DIR / "krea2_styles.csv"
ART_STYLES_FILE = settings.CONFIG_DIR / "krea2_art_styles.csv"


def _load() -> Dict[str, str]:
    settings.ensure_dirs()
    if STYLES_FILE.is_file():
        try:
            data = json.loads(STYLES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write(data: Dict[str, str]) -> None:
    settings.ensure_dirs()
    STYLES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def list_styles() -> list[str]:
    """Noms des styles enregistrés (triés)."""
    return sorted(_load().keys(), key=str.lower)


def get_style(name: str | None) -> str:
    if not name:
        return ""
    return _load().get(name, "")


def save_style(name: str, text: str) -> str:
    """Enregistre/écrase un style. Retourne le nom nettoyé."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Donnez un nom au style.")
    if not (text or "").strip():
        raise ValueError("Le style est vide.")
    data = _load()
    data[name] = text.strip()
    _write(data)
    return name


def delete_style(name: str | None) -> None:
    if not name:
        return
    data = _load()
    if name in data:
        del data[name]
        _write(data)


# --------------------------------------------------------------------------
#  Banques de styles Krea (cumulables), fournies avec l'app.
#  CSV : name,prompt,negative_prompt. Les lignes « [ Catégorie ],, » sont des
#  en-têtes ; les lignes « # … » sont des commentaires (crédit / notes).
#  « {prompt} » (styles photo) est remplacé par le sujet ; les styles SANS
#  « {prompt} » (styles artistiques) sont ajoutés APRÈS le sujet comme addon.
#  On empile plusieurs styles en chaînant leurs phrases ; les négatifs éventuels
#  sont combinés. L'étiquette d'un style est « Catégorie › Nom » (unique).
# --------------------------------------------------------------------------
_SEP = " › "
_bank_cache: Dict[Path, List[Dict[str, str]]] = {}


def _load_bank(path: Path) -> List[Dict[str, str]]:
    """Parse une banque de styles CSV une fois (avec cache par fichier).
    Retourne une liste ordonnée d'entrées {category, name, label, prompt,
    negative}."""
    cached = _bank_cache.get(path)
    if cached is not None:
        return cached
    entries: List[Dict[str, str]] = []
    if not path.is_file():
        _bank_cache[path] = entries
        return entries
    category = ""
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                first = (row[0] or "").strip()
                if not first or first.startswith("#"):
                    continue
                rest = "".join(row[1:]).strip()
                # En-tête de catégorie : « [ Nom ],, » (colonnes suivantes vides).
                if first.startswith("[") and first.endswith("]") and not rest:
                    category = first.strip("[] ").strip()
                    continue
                # Ligne d'en-tête du CSV.
                if first == "name" and len(row) > 1 and row[1].strip() == "prompt":
                    continue
                prompt = row[1].strip() if len(row) > 1 else ""
                negative = row[2].strip() if len(row) > 2 else ""
                if not prompt:
                    continue
                label = f"{category}{_SEP}{first}" if category else first
                entries.append({"category": category, "name": first,
                                "label": label, "prompt": prompt,
                                "negative": negative})
    except OSError:
        pass
    _bank_cache[path] = entries
    return entries


def bank_labels(path: Path) -> List[str]:
    """Étiquettes « Catégorie › Nom » d'une banque, dans l'ordre du fichier
    (regroupées par catégorie) — choix d'un menu multi-sélection."""
    return [e["label"] for e in _load_bank(path)]


def _bank_index(path: Path) -> Dict[str, Dict[str, str]]:
    return {e["label"]: e for e in _load_bank(path)}


def random_bank_label(path: Path, exclude: List[str] | None = None) -> str | None:
    """Tire un style au hasard dans une banque (comportement « wildcard » Krea).
    Ignore les étiquettes déjà présentes dans `exclude`. Retourne None si vide."""
    exclude = set(exclude or [])
    pool = [lab for lab in bank_labels(path) if lab not in exclude]
    return random.choice(pool) if pool else None


def apply_style_bank(subject: str, labels: List[str] | None,
                     path: Path) -> Tuple[str, List[str]]:
    """Applique les styles cumulables sélectionnés d'une banque au sujet.

    Style AVEC « {prompt} » (photo) : le sujet y est inséré. Style SANS
    « {prompt} » (artistique) : sa description est ajoutée après le sujet.
    Plusieurs styles chaînent leurs phrases. Retourne (prompt_final, négatifs).
    Sans sélection, le sujet est renvoyé inchangé.
    """
    subject = (subject or "").strip()
    if not labels:
        return subject, []
    index = _bank_index(path)
    combined = subject
    negatives: List[str] = []
    for lab in labels:
        entry = index.get(lab)
        if not entry:
            continue
        tmpl = entry["prompt"]
        if "{prompt}" in tmpl:
            # Partie descriptive = ce qui suit « {prompt} » (souvent « . … »).
            addon = tmpl.split("{prompt}", 1)[1]
        else:
            addon = tmpl
        addon = addon.strip().lstrip(".").strip()
        if addon:
            combined = f"{combined}. {addon}" if combined else addon
        neg = (entry.get("negative") or "").strip()
        if neg:
            negatives.append(neg)
    return combined.strip().strip(".").strip(), negatives


# Rétro-compat : anciens noms « photo » = banque photographique.
def photo_style_labels() -> List[str]:
    return bank_labels(PHOTO_STYLES_FILE)


def apply_photo_styles(subject: str,
                       labels: List[str] | None) -> Tuple[str, List[str]]:
    return apply_style_bank(subject, labels, PHOTO_STYLES_FILE)
