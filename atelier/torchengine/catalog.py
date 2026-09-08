"""Ce que chaque modèle du catalogue devient une fois passé chez diffusers.

Le point important, et il a été corrigé après coup : **diffusers lit le
GGUF**. La première version de ce fichier supposait le contraire et décrivait
des dépôts complets à retélécharger — 33 Go pour un modèle qui en pèse 6,6 sur
le disque. C'était faux, et c'est ce qui rendait cette branche absurde.

`single_file` décrit donc le montage réel : chaque composant du pipeline est
chargé depuis LE FICHIER QUE L'APPLICATION A DÉJÀ, celui du catalogue principal,
choisi par la même échelle de quantification. `role` est le nom du composant
là-bas ; c'est le seul lien à maintenir entre les deux catalogues.

Il reste deux choses qui ne viennent pas du GGUF, et elles sont petites :

* **les configurations d'architecture** — quelques kilo-octets de JSON qui
  disent combien de couches et de quelle largeur. `config_repo` dit d'où. Pour
  Z-Image ce dépôt est ouvert ; pour Flux.2 Klein il est fermé, et c'est le
  seul jeton que cette branche demande (une fois, puis mis en cache) ;
* **Krea 2**, qui n'a pas de chargement fichier-unique en amont du tout et
  reste donc sur son dépôt complet.

Un mode (texte→image, image→image, édition, inpaint) n'existe que si diffusers
publie une classe pour ce modèle. Ce n'est pas une option qu'on active, c'est un
fait qu'on constate — et il diffère d'un modèle à l'autre, là où sd.cpp offrait
les quatre partout.
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

#  Composants du pipeline montés depuis un fichier unique, dans l'ordre où on
#  les charge. Le VAE en dernier : c'est le plus petit, et le seul dont une
#  panne se rattrape (on peut décoder sur le CPU).
SINGLE_FILE_PARTS = ("transformer", "text_encoder", "vae")


@dataclass(frozen=True)
class Part:
    """Un composant chargé depuis un fichier unique."""
    name: str
    cls: str
    role: str


@dataclass(frozen=True)
class TorchModel:
    id: str
    pipeline: str
    min_diffusers: str
    license: str
    negative_prompt: bool
    #  mode -> classe diffusers. Absent = le mode n'existe pas pour ce modèle.
    pipelines: dict = field(default_factory=dict)
    #  composant -> Part. Vide = pas de chargement fichier-unique possible.
    parts: dict = field(default_factory=dict)
    config_repo: str = ""
    config_gated: bool = False
    #  Repli dépôt complet.
    repo: str = ""
    gated: bool = False
    full_repo_gb: float | None = None

    @property
    def from_gguf(self) -> bool:
        """Ce modèle se monte-t-il à partir des fichiers déjà installés ?"""
        return bool(self.parts)

    @property
    def local_dir(self) -> Path:
        return settings.model_repo_dir(self.repo) if self.repo else Path()

    def can(self, mode: str) -> bool:
        return bool(self.pipelines.get(mode))

    def pipeline_for(self, mode: str) -> str | None:
        return self.pipelines.get(mode)

    @property
    def modes(self) -> list[str]:
        return [m for m in (TEXT_TO_IMAGE, IMAGE_TO_IMAGE, EDIT, INPAINT)
                if self.can(m)]

    @property
    def needs_token(self) -> str:
        """Pourquoi ce modèle réclame un jeton Hugging Face — ou "" s'il n'en
        réclame pas. La distinction compte : quelques kilo-octets de
        configuration et trente gigaoctets de poids ne se demandent pas de la
        même façon."""
        if self.from_gguf:
            return ("its architecture config (a few KB, once) comes from a "
                    f"gated repository: {self.config_repo}"
                    if self.config_gated else "")
        return (f"its weights come from a gated repository: {self.repo}"
                if self.gated else "")


@lru_cache(maxsize=4)
def _parse(path: str, mtime: float) -> tuple[TorchModel, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = []
    for e in raw.get("models") or []:
        sf = dict(e.get("single_file") or {})
        parts = {name: Part(name, str(spec["class"]), str(spec["role"]))
                 for name in SINGLE_FILE_PARTS
                 for spec in [sf.get(name)] if spec}
        out.append(TorchModel(
            id=str(e["id"]),
            pipeline=str(e["pipeline"]),
            min_diffusers=str(e.get("min_diffusers") or "0"),
            license=str(e.get("license") or "unknown"),
            negative_prompt=bool(e.get("negative_prompt")),
            pipelines=dict(e.get("pipelines") or {}),
            parts=parts,
            config_repo=str(sf.get("config_repo") or ""),
            config_gated=bool(sf.get("config_gated")),
            repo=str(e.get("repo") or ""),
            gated=bool(e.get("gated")),
            full_repo_gb=(None if e.get("full_repo_gb") is None
                          else float(e["full_repo_gb"])),
        ))
    return tuple(out)


def load(path: Path | None = None) -> list[TorchModel]:
    """Le catalogue, relu dès que le fichier change sur le disque."""
    p = Path(path or CATALOG_FILE)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return []
    # Copie profonde : les entrées sont gelées mais `pipelines` et `parts` sont
    # des dicts bien vivants, et le cache les rendrait partagés.
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
