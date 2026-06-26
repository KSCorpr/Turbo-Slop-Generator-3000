"""Catalogue de modèles : chargement du catalogue, résolution des fichiers,
statut de téléchargement, et recommandations selon le matériel.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import hardware, quant, settings


@dataclass
class Component:
    role: str            # diffusion | uncond | vae | text_encoder
    repo: str
    template: str        # ex "*-{quant}.gguf" ou "vae/*.safetensors"
    quant: str | None    # quant résolu si le motif contient un token, sinon None

    @property
    def token(self) -> str | None:
        if "{quant}" in self.template:
            return "{quant}"
        if "{enc_quant}" in self.template:
            return "{enc_quant}"
        return None

    def requested(self) -> str:
        """Nom/motif exact souhaité (token remplacé par le quant choisi)."""
        if self.token and self.quant:
            return self.template.replace(self.token, self.quant)
        return self.template

    def base_glob(self) -> str:
        """Motif quant-agnostique (token remplacé par *) pour le repli."""
        if self.token:
            return self.template.replace(self.token, "*")
        return self.template


@dataclass
class BaseModel:
    id: str
    name: str
    family: str
    tags: list[str]
    description: str
    components: list[Component]
    defaults: dict[str, Any]
    vram_min_gb: float
    presets: list[dict] = None  # type: ignore[assignment]


def _catalog() -> dict[str, Any]:
    path = settings.CONFIG_DIR / "models.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def effective_quants(prefs: dict[str, Any]) -> tuple[str, str]:
    """(quant diffusion, quant encodeur) après prise en compte des préférences."""
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    q_diff = prefs.get("quant") or prof.quant
    q_enc = prefs.get("enc_quant") or prof.enc_quant
    return q_diff, q_enc


def load_base_models(prefs: dict[str, Any]) -> list[BaseModel]:
    q_diff, q_enc = effective_quants(prefs)
    out: list[BaseModel] = []
    for m in _catalog().get("base_models", []):
        comps: list[Component] = []
        for role, spec in (m.get("sources") or {}).items():
            template = spec["match"]
            if "{enc_quant}" in template:
                q = q_enc
            elif "{quant}" in template:
                q = q_diff
            else:
                q = None
            comps.append(Component(role, spec["repo"], template, q))
        out.append(BaseModel(
            id=m["id"], name=m["name"], family=m["family"],
            tags=m.get("tags", []), description=(m.get("description") or "").strip(),
            components=comps, defaults=m.get("defaults", {}),
            vram_min_gb=float(m.get("vram_min_gb", 0)),
            presets=m.get("presets", []),
        ))
    return out


def get_base_model(model_id: str, prefs: dict[str, Any]) -> BaseModel | None:
    return next((m for m in load_base_models(prefs) if m.id == model_id), None)


# --------------------------------------------------------------------------- #
#  LTX-2.3 (vidéo, natif sd.cpp)
# --------------------------------------------------------------------------- #
def ltx_config() -> dict[str, Any]:
    return _catalog().get("ltx", {}) or {}


def ltx_components(prefs: dict[str, Any] | None = None) -> list[Component]:
    prefs = prefs or settings.load_prefs()
    q_diff, q_enc = effective_quants(prefs)
    out: list[Component] = []
    for role, spec in (ltx_config().get("sources") or {}).items():
        t = spec["match"]
        q = q_enc if "{enc_quant}" in t else (q_diff if "{quant}" in t else None)
        out.append(Component(role, spec["repo"], t, q))
    return out


def ltx_paths(prefs: dict[str, Any] | None = None) -> dict[str, Path | None]:
    return {c.role: resolve_component_path(c) for c in ltx_components(prefs)}


def ltx_ready(prefs: dict[str, Any] | None = None) -> bool:
    comps = ltx_components(prefs)
    return bool(comps) and all(resolve_component_path(c) is not None for c in comps)


# --------------------------------------------------------------------------- #
#  Upscalers ESRGAN (GGUF, natif sd.cpp --mode upscale)
# --------------------------------------------------------------------------- #
def upscaler_config() -> dict[str, Any]:
    return _catalog().get("upscalers", {}) or {}


def upscalers_dir() -> Path:
    return settings.model_repo_dir(upscaler_config().get("repo", "upscalers"))


def list_upscalers() -> list[str]:
    """Noms des fichiers d'upscaler ESRGAN déjà téléchargés (triés)."""
    d = upscalers_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.gguf"))


def upscaler_path(name: str) -> Path | None:
    if not name:
        return None
    p = upscalers_dir() / name
    return p if p.is_file() else None


def upscalers_ready() -> bool:
    return bool(list_upscalers())


# --------------------------------------------------------------------------- #
#  Résolution des chemins locaux + statut
# --------------------------------------------------------------------------- #
def _match(files: list[Path], repo_dir: Path, pattern: str) -> list[Path]:
    out = []
    for p in files:
        rel = p.relative_to(repo_dir).as_posix()
        if fnmatch.fnmatch(p.name, pattern) or fnmatch.fnmatch(rel, pattern):
            out.append(p)
    return out


def resolve_component_path(comp: Component) -> Path | None:
    """Cherche sur le disque le fichier correspondant au composant. None si absent."""
    repo_dir = settings.model_repo_dir(comp.repo)
    if not repo_dir.exists():
        return None

    requested = comp.requested()
    exact = repo_dir / requested
    if "*" not in requested and exact.is_file():
        return exact

    files = [p for p in repo_dir.rglob("*") if p.is_file()]

    # 1) correspondance directe sur le motif demandé
    direct = _match(files, repo_dir, requested)
    if direct:
        return min(direct, key=lambda p: len(p.relative_to(repo_dir).as_posix()))

    # 2) repli quant-tolérant : tout fichier du même motif de base, quant le + proche
    if comp.token:
        cands = _match(files, repo_dir, comp.base_glob())
        cands = [c for c in cands if "mmproj" not in c.name.lower()] or cands
        if cands and comp.quant:
            chosen = quant.best([c.name for c in cands], comp.quant)
            return next((c for c in cands if c.name == chosen), cands[0])
        if cands:
            return cands[0]
    return None


def model_is_ready(model: BaseModel) -> bool:
    return all(resolve_component_path(c) is not None for c in model.components)


def missing_components(model: BaseModel) -> list[Component]:
    return [c for c in model.components if resolve_component_path(c) is None]


def delete_model(model: BaseModel, prefs: dict[str, Any]) -> list[str]:
    """Supprime les fichiers téléchargés de ce modèle, SANS toucher aux fichiers
    partagés avec un autre modèle (encodeur/VAE communs).
    Retourne la liste des fichiers supprimés."""
    mine = {resolve_component_path(c) for c in model.components}
    mine.discard(None)
    # Fichiers utilisés par les AUTRES modèles : à préserver.
    shared: set = set()
    for other in load_base_models(prefs):
        if other.id == model.id:
            continue
        for c in other.components:
            p = resolve_component_path(c)
            if p:
                shared.add(p)

    deleted: list[str] = []
    repo_dirs: set[Path] = set()
    for p in mine - shared:
        try:
            repo_dirs.add(p.parent)
            p.unlink()
            deleted.append(p.name)
        except OSError:
            pass
    # Nettoie les dossiers devenus vides (y compris parents type split_files/).
    for d in sorted(repo_dirs, key=lambda x: len(str(x)), reverse=True):
        cur = d
        while cur != settings.MODELS_DIR and cur.is_dir():
            try:
                next(cur.iterdir())
                break  # pas vide
            except StopIteration:
                parent = cur.parent
                try:
                    cur.rmdir()
                except OSError:
                    break
                cur = parent
    return deleted


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
        out[m.id] = labels
    return out
