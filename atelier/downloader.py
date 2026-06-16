"""Téléchargement à la demande des modèles depuis Hugging Face.

Pour un modèle de la bibliothèque, télécharge chaque composant (diffusion, vae,
encodeur...) en choisissant le bon fichier par correspondance souple sur le motif.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Callable, Iterator

from . import quant, settings
from .registry import BaseModel, Component


def _fn(f: str, pattern: str) -> bool:
    return fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)


def _pick_file(comp: Component, files: list[str]) -> str | None:
    requested = comp.requested()
    if requested in files:
        return requested

    # 1) correspondance directe sur le motif demandé (nom ou chemin)
    direct = [f for f in files if _fn(f, requested)]
    if direct:
        return sorted(direct, key=len)[0]

    # 2) repli quant-tolérant : même motif de base, quant le plus proche
    if comp.token:
        cands = [f for f in files if _fn(f, comp.base_glob())]
        cands = [c for c in cands if "mmproj" not in c.lower()] or cands
        if cands and comp.quant:
            chosen = quant.best([Path(c).name for c in cands], comp.quant)
            return next((c for c in cands if Path(c).name == chosen), cands[0])
        if cands:
            return sorted(cands, key=len)[0]
    return None


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
    chosen = _pick_file(comp, files)
    if not chosen:
        raise RuntimeError(
            f"Aucun fichier de {comp.repo} ne correspond à "
            f"« {comp.requested()} » ni au motif « {comp.base_glob()} ».")

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
