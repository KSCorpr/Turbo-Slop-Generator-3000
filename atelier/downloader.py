"""Téléchargement à la demande des modèles depuis Hugging Face.

Pour un modèle de la bibliothèque, télécharge chaque composant (diffusion, vae,
encodeur...) en choisissant le bon fichier par correspondance souple sur le motif.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Callable, Iterator

from . import settings
from .registry import BaseModel, Component


def _pick_file(repo: str, pattern: str, files: list[str]) -> str | None:
    if pattern in files:
        return pattern
    cands = [f for f in files if fnmatch.fnmatch(f, pattern) or
             fnmatch.fnmatch(Path(f).name, pattern)]
    if not cands:
        # repli : tout fichier contenant la base du motif (avant le 1er joker)
        base = pattern.split("*")[0].split("{")[0].rstrip("-_.")
        if base:
            cands = [f for f in files if base.lower() in f.lower()]
    if not cands:
        return None
    # On écarte explicitement les projecteurs vision (mmproj) pour les encodeurs.
    cands = [c for c in cands if "mmproj" not in c.lower()] or cands
    return sorted(cands, key=lambda f: len(f))[0]


def download_component(comp: Component,
                       log: Callable[[str], None] | None = None) -> Path:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_ENDPOINT", settings.load_prefs().get(
        "hf_endpoint", "https://huggingface.co"))
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub manquant : "
                           "pip install -r requirements.txt") from exc

    files = list_repo_files(comp.repo)
    chosen = _pick_file(comp.repo, comp.match, files)
    if not chosen:
        raise RuntimeError(
            f"Aucun fichier de {comp.repo} ne correspond à « {comp.match} ».")

    local_dir = settings.model_repo_dir(comp.repo)
    dest = local_dir / chosen
    if dest.is_file():
        if log:
            log(f"  ✓ déjà présent : {comp.role} ({chosen})")
        return dest
    if log:
        log(f"  ↓ {comp.role} : {comp.repo}/{chosen}")
    path = hf_hub_download(repo_id=comp.repo, filename=chosen,
                           local_dir=str(local_dir))
    return Path(path)


def download_model(model: BaseModel,
                   log: Callable[[str], None] | None = None) -> Iterator[str]:
    """Télécharge tous les composants manquants d'un modèle. Yields des messages."""
    settings.ensure_dirs()
    yield f"Téléchargement de « {model.name} »…"
    for comp in model.components:
        try:
            download_component(comp, log=log)
            yield f"  ✓ {comp.role}"
        except Exception as exc:  # noqa: BLE001
            yield f"  ✗ {comp.role} : {exc}"
            return
    yield f"« {model.name} » est prêt. ✅"
