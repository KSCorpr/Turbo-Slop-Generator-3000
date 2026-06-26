"""Téléchargement à la demande des modèles depuis Hugging Face.

Pour un modèle de la bibliothèque, télécharge chaque composant (diffusion, vae,
encodeur...) en choisissant le bon fichier par correspondance souple sur le motif.
"""
from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Callable, Iterator

from . import quant, settings
from .registry import BaseModel, Component


def download_lora(repo: str, log: Callable[[str], None] | None = None) -> str:
    """Télécharge le premier .safetensors d'un dépôt LoRA dans loras/.

    Renvoie le nom (sans extension) à utiliser dans la syntaxe <lora:nom:poids>.
    """
    settings.configure_hf_env()
    settings.LORA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    cands = [f for f in files if f.lower().endswith(".safetensors")]
    if not cands:
        raise RuntimeError(f"Aucun .safetensors dans {repo}")
    chosen = sorted(cands, key=len)[0]
    dest = settings.LORA_DIR / Path(chosen).name
    if not dest.is_file():
        if log:
            log(f"Téléchargement du LoRA {repo}/{chosen}…")
        got = Path(hf_hub_download(repo_id=repo, filename=chosen,
                                   local_dir=str(settings.LORA_DIR)))
        if got.resolve() != dest.resolve() and got.is_file():
            shutil.copy(got, dest)
    return dest.stem


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
    settings.configure_hf_env()
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub manquant : "
                           "pip install -r requirements.txt") from exc

    files = list_repo_files(comp.repo)
    chosen = _pick_file(comp, files)
    if not chosen:
        # On liste les fichiers pertinents pour diagnostiquer le vrai nommage.
        relevant = [f for f in files
                    if f.lower().endswith((".gguf", ".safetensors", ".sft"))]
        listing = "\n      - " + "\n      - ".join(sorted(relevant)[:40]) \
            if relevant else " (aucun .gguf/.safetensors trouvé)"
        raise RuntimeError(
            f"Aucun fichier de {comp.repo} ne correspond à "
            f"« {comp.requested()} » ni à « {comp.base_glob()} ».\n"
            f"    Fichiers disponibles dans le dépôt :{listing}")

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


def download_upscalers(log: Callable[[str], None] | None = None) -> Iterator[str]:
    """Télécharge tous les upscalers ESRGAN GGUF (wbruna/upscalers-sdcpp-gguf)."""
    from .registry import upscaler_config, upscalers_dir
    settings.configure_hf_env()
    settings.ensure_dirs()
    cfg = upscaler_config()
    repo = cfg.get("repo")
    files = cfg.get("files") or []
    if not repo or not files:
        yield "Upscalers non configurés."
        return
    from huggingface_hub import hf_hub_download

    local_dir = upscalers_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    yield f"Téléchargement de {len(files)} upscalers ESRGAN ({repo})…"
    ok = 0
    for fn in files:
        dest = local_dir / fn
        if dest.is_file():
            ok += 1
            yield f"  ✓ déjà présent : {fn}"
            continue
        try:
            hf_hub_download(repo_id=repo, filename=fn, local_dir=str(local_dir))
            ok += 1
            yield f"  ↓ {fn}"
        except Exception as exc:  # noqa: BLE001
            yield f"  ✗ {fn} : {exc}"
    yield f"Upscalers prêts ({ok}/{len(files)}). ✅"


def download_ltx(log: Callable[[str], None] | None = None) -> Iterator[str]:
    """Télécharge les composants de LTX-2.3 (diffusion 22B + Gemma-3-12B + VAE…)."""
    from .registry import ltx_components
    settings.ensure_dirs()
    comps = ltx_components()
    if not comps:
        yield "LTX-2.3 non configuré."
        return
    yield "Téléchargement de LTX-2.3 (gros : 22B + Gemma-3-12B + VAE)…"
    for comp in comps:
        try:
            download_component(comp, log=log)
            yield f"  ✓ {comp.role}"
        except Exception as exc:  # noqa: BLE001
            yield f"  ✗ {comp.role} : {exc}"
            return
    yield "LTX-2.3 est prêt. ✅"
