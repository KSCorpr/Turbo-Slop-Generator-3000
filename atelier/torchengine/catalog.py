"""Ce que chaque modèle du catalogue devient une fois passé chez diffusers.

Le catalogue GGUF décrit des FICHIERS : un par rôle, choisi selon la VRAM. Un
modèle diffusers est un DÉPÔT — un dossier par composant et un index qui les
relie — et la quantification n'y est plus un fichier qu'on choisit mais une
transformation faite au chargement. Les deux descriptions n'ont presque aucun
champ en commun, d'où un second fichier plutôt qu'un champ de plus dans le
premier.

Les identifiants sont communs aux deux catalogues : c'est ce qui permet à
l'interface de ne pas savoir sur quel moteur elle tourne.

Le point important est `pipelines` : un MODE (texte→image, image→image,
édition, inpaint) n'existe que si diffusers publie une classe pour ce modèle.
Ce n'est pas une option qu'on active, c'est un fait qu'on constate — et il
diffère d'un modèle à l'autre, là où sd.cpp offrait les quatre partout.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .. import settings

CATALOG_FILE = settings.CONFIG_DIR / "models_torch.yaml"

TEXT_TO_IMAGE = "text_to_image"
IMAGE_TO_IMAGE = "image_to_image"
EDIT = "edit"
INPAINT = "inpaint"


@dataclass(frozen=True)
class TorchModel:
    id: str
    repo: str
    min_diffusers: str
    gated: bool
    license: str
    #  mode -> classe diffusers. Absent = le mode n'existe pas pour ce modèle.
    pipelines: dict = field(default_factory=dict)
    negative_prompt: bool = False
    #  None = inconnu (dépôt fermé : l'API du Hub ne publie pas les tailles).
    #  Zéro serait un mensonge commode, et se propagerait dans les décisions de
    #  placement comme un modèle qui ne pèse rien.
    download_gb: float | None = None
    resident_bf16_gb: float | None = None
    largest_module_bf16_gb: float | None = None

    @property
    def local_dir(self) -> Path:
        return settings.model_repo_dir(self.repo)

    def can(self, mode: str) -> bool:
        return bool(self.pipelines.get(mode))

    def pipeline_for(self, mode: str) -> str | None:
        return self.pipelines.get(mode)

    @property
    def modes(self) -> list[str]:
        return [m for m in (TEXT_TO_IMAGE, IMAGE_TO_IMAGE, EDIT, INPAINT)
                if self.can(m)]


@lru_cache(maxsize=4)
def _parse(path: str, mtime: float) -> tuple[TorchModel, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return tuple(
        TorchModel(
            id=str(e["id"]),
            repo=str(e["repo"]),
            min_diffusers=str(e.get("min_diffusers") or "0"),
            gated=bool(e.get("gated")),
            license=str(e.get("license") or "unknown"),
            pipelines=dict(e.get("pipelines") or {}),
            negative_prompt=bool(e.get("negative_prompt")),
            download_gb=_opt_float(e.get("download_gb")),
            resident_bf16_gb=_opt_float(e.get("resident_bf16_gb")),
            largest_module_bf16_gb=_opt_float(e.get("largest_module_bf16_gb")),
        )
        for e in (raw.get("models") or []))


def _opt_float(value) -> float | None:
    return None if value is None else float(value)


def load(path: Path | None = None) -> list[TorchModel]:
    """Le catalogue, relu dès que le fichier change sur le disque."""
    p = Path(path or CATALOG_FILE)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return []
    # Copie profonde : les entrées sont gelées mais `pipelines` est un dict
    # bien vivant, et le cache le rendrait partagé entre tous les appelants.
    return [copy.deepcopy(m) for m in _parse(str(p), mtime)]


def get(model_id: str, path: Path | None = None) -> TorchModel | None:
    return next((m for m in load(path) if m.id == model_id), None)


def mode_for(model: TorchModel, *, init_image=None, ref_image=None,
             mask_image=None) -> str:
    """Le mode que ces entrées demandent — indépendamment de sa disponibilité.

    L'appelant compare ensuite avec `model.can(...)`. Séparer les deux est
    volontaire : « ce que l'utilisateur a demandé » et « ce que le modèle sait
    faire » doivent pouvoir diverger pour qu'on puisse l'EXPLIQUER, au lieu de
    retomber en silence sur du texte→image en laissant croire que l'image de
    départ a servi.
    """
    if mask_image is not None:
        return INPAINT
    if ref_image:
        return EDIT if model.can(EDIT) else IMAGE_TO_IMAGE
    if init_image is not None:
        return IMAGE_TO_IMAGE
    return TEXT_TO_IMAGE
