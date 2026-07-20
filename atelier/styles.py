"""Préréglages de prompt système / style, persistés en JSON.

Un style est un simple texte (préfixe appliqué en tête du prompt) enregistré
sous un nom. Les styles sont GLOBAUX : enregistrés une fois, ils sont
disponibles dans tous les onglets de modèle.
"""
from __future__ import annotations

import csv
import json
from typing import Dict, List, Tuple

from . import settings

STYLES_FILE = settings.USERDATA_DIR / "style_presets.json"

# Banque de styles photographiques Krea 2 (cumulables), fournie avec l'app.
# Source : github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2 (MIT, ghleg).
PHOTO_STYLES_FILE = settings.CONFIG_DIR / "krea2_styles.csv"


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
#  Banque de styles photographiques Krea 2 (cumulables, © ghleg — MIT).
#  CSV : name,prompt,negative_prompt. Les lignes « [ Catégorie ],, » sont des
#  en-têtes ; les lignes « # … » sont des commentaires (crédit). « {prompt} »
#  est remplacé par le sujet de l'utilisateur. On empile plusieurs styles en
#  chaînant leurs phrases après le sujet ; les négatifs sont combinés.
#  L'étiquette d'un style est « Catégorie › Nom » (unique, groupée visuellement).
# --------------------------------------------------------------------------
_SEP = " › "
_photo_cache: List[Dict[str, str]] | None = None


def _load_photo_styles() -> List[Dict[str, str]]:
    """Parse le CSV une fois. Retourne une liste ordonnée d'entrées
    {category, name, label, prompt, negative}."""
    global _photo_cache
    if _photo_cache is not None:
        return _photo_cache
    entries: List[Dict[str, str]] = []
    if not PHOTO_STYLES_FILE.is_file():
        _photo_cache = entries
        return entries
    category = ""
    try:
        with PHOTO_STYLES_FILE.open(encoding="utf-8", newline="") as fh:
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
    _photo_cache = entries
    return entries


def photo_style_labels() -> List[str]:
    """Étiquettes « Catégorie › Nom » dans l'ordre du fichier (regroupées par
    catégorie) — à utiliser comme choix d'un menu multi-sélection."""
    return [e["label"] for e in _load_photo_styles()]


def _photo_index() -> Dict[str, Dict[str, str]]:
    return {e["label"]: e for e in _load_photo_styles()}


def apply_photo_styles(subject: str,
                       labels: List[str] | None) -> Tuple[str, List[str]]:
    """Insère le sujet dans les styles photo sélectionnés (cumulables).

    Chaque style est une phrase « {prompt}. <description> ». On remplace
    « {prompt} » par le sujet dans le premier, puis on enchaîne les
    descriptions suivantes après le sujet. Retourne (prompt_final, négatifs).
    Sans sélection, le sujet est renvoyé inchangé.
    """
    subject = (subject or "").strip()
    if not labels:
        return subject, []
    index = _photo_index()
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
