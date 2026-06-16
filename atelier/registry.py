"""Bibliothèque de modèles : chargement du catalogue, résolution des fichiers,
statut de téléchargement, et recommandations selon le matériel.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import hardware, settings


@dataclass
class Component:
    role: str            # diffusion | uncond | vae | text_encoder
    repo: str
    match: str           # motif avec jokers {quant}/{enc_quant} déjà résolus
    quantizable: bool


@dataclass
class BaseModel:
    id: str
    name: str
    family: str          # ideogram | zimage
    tags: list[str]
    description: str
    components: list[Component]
    defaults: dict[str, Any]
    vram_min_gb: float


@dataclass
class Upscaler:
    id: str
    name: str
    engine: str
    tags: list[str]
    description: str
    weights_repo: str
    code_repo: str
    vram_min_gb: float


def _catalog() -> dict[str, Any]:
    path = settings.CONFIG_DIR / "models.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def effective_quants(prefs: dict[str, Any]) -> tuple[str, str]:
    """(quant diffusion, quant encodeur) après prise en compte des préférences."""
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    quant = prefs.get("quant") or prof.quant
    enc_quant = prefs.get("enc_quant") or prof.enc_quant
    return quant, enc_quant


def load_base_models(prefs: dict[str, Any]) -> list[BaseModel]:
    quant, enc_quant = effective_quants(prefs)
    out: list[BaseModel] = []
    for m in _catalog().get("base_models", []):
        comps: list[Component] = []
        for role, spec in (m.get("sources") or {}).items():
            match = (spec["match"]
                     .replace("{quant}", quant)
                     .replace("{enc_quant}", enc_quant))
            comps.append(Component(role, spec["repo"], match,
                                   bool(spec.get("quantizable"))))
        out.append(BaseModel(
            id=m["id"], name=m["name"], family=m["family"],
            tags=m.get("tags", []), description=(m.get("description") or "").strip(),
            components=comps, defaults=m.get("defaults", {}),
            vram_min_gb=float(m.get("vram_min_gb", 0)),
        ))
    return out


def load_upscalers() -> list[Upscaler]:
    out: list[Upscaler] = []
    for u in _catalog().get("upscalers", []):
        out.append(Upscaler(
            id=u["id"], name=u["name"], engine=u.get("engine", "pytorch"),
            tags=u.get("tags", []), description=(u.get("description") or "").strip(),
            weights_repo=u.get("weights_repo", ""), code_repo=u.get("code_repo", ""),
            vram_min_gb=float(u.get("vram_min_gb", 0)),
        ))
    return out


def get_base_model(model_id: str, prefs: dict[str, Any]) -> BaseModel | None:
    return next((m for m in load_base_models(prefs) if m.id == model_id), None)


# --------------------------------------------------------------------------- #
#  Résolution des chemins locaux + statut
# --------------------------------------------------------------------------- #
def resolve_component_path(comp: Component) -> Path | None:
    """Cherche sur le disque le fichier correspondant au motif. None si absent."""
    repo_dir = settings.model_repo_dir(comp.repo)
    if not repo_dir.exists():
        return None
    pattern = comp.match
    # Correspondance exacte d'abord
    exact = repo_dir / pattern
    if "*" not in pattern and exact.is_file():
        return exact
    matches = sorted((p for p in repo_dir.rglob("*")
                      if p.is_file() and fnmatch.fnmatch(p.name, pattern)),
                     key=lambda p: len(p.name))
    return matches[0] if matches else None


def model_is_ready(model: BaseModel) -> bool:
    return all(resolve_component_path(c) is not None for c in model.components)


def missing_components(model: BaseModel) -> list[Component]:
    return [c for c in model.components if resolve_component_path(c) is None]


# --------------------------------------------------------------------------- #
#  Recommandations selon le matériel
# --------------------------------------------------------------------------- #
def recommend(prefs: dict[str, Any]) -> dict[str, list[str]]:
    """Retourne {model_id: [étiquettes de reco]} pour guider l'artiste."""
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    vram = prof.gpu.vram_gb if prof.gpu else 0.0
    out: dict[str, list[str]] = {}
    for m in load_base_models(prefs):
        labels: list[str] = []
        if vram and vram >= m.vram_min_gb:
            labels.append("✅ adapté à votre carte")
        elif vram:
            labels.append(f"⚠️ {m.vram_min_gb:.0f} Go conseillés (vous : {vram:.0f})")
        if m.family == "zimage" and (not vram or vram < 12):
            labels.append("👍 recommandé pour démarrer (rapide)")
        out[m.id] = labels
    return out
