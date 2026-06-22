"""Préréglages de prompt système / style, persistés en JSON.

Un style est un simple texte (préfixe appliqué en tête du prompt) enregistré
sous un nom. Les styles sont GLOBAUX : enregistrés une fois, ils sont
disponibles dans tous les onglets de modèle.
"""
from __future__ import annotations

import json
from typing import Dict

from . import settings

STYLES_FILE = settings.USERDATA_DIR / "style_presets.json"


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
