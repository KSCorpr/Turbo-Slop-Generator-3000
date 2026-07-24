"""Moteur trellis.cpp : image → maillage 3D (GLB) via le binaire natif trellis-cli.

trellis-cli est un exécutable C++/GGML/CUDA autonome (aucun PyTorch), invoqué en
one-shot : `trellis-cli <image> <sortie.glb> --res 512|1024|1536 --models <DIR>`.
Le mode **512** (« light », sans cascade) est celui qui tient sur les cartes
modestes (≤ 12 Go) ; 1024/1536 exigent ~16 Go+.
"""
from __future__ import annotations

import platform
import shlex
import shutil
from pathlib import Path
from typing import Callable

from .. import settings
from . import sdcpp

TRELLIS_BIN_DIR = settings.BIN_DIR / "trellis"
MODELS_DIR = settings.MODELS_DIR / "trellis"

# Résolutions proposées. Seul le 512 tient sur 11-12 Go (les autres ~16 Go+).
RESOLUTIONS = [
    ("512 — léger (recommandé, ≤ 12 Go)", 512),
    ("1024 — cascade (≥ 16 Go)", 1024),
    ("1536 — haute (≥ 16 Go+)", 1536),
]


def _is_cli(p: Path) -> bool:
    """Détection souple du binaire CLI trellis (nom variable selon la release)."""
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    if stem.startswith("trellis") and "cli" in stem:
        return True
    return ("trellis" in stem and not any(
        x in stem for x in ("server", "test", "studio", "bench", "convert")))


def find_cli() -> Path | None:
    if settings.BIN_DIR.exists():
        # Priorité stricte (…cli…), puis repli sur tout exécutable « trellis ».
        strict = [p for p in settings.BIN_DIR.rglob("*")
                  if _is_cli(p) and "cli" in p.name.lower()]
        if strict:
            return strict[0]
        loose = [p for p in settings.BIN_DIR.rglob("*") if _is_cli(p)]
        if loose:
            return loose[0]
    for name in ("trellis-cli", "trellis"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def models_ready() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.rglob("*.gguf"))


def is_ready() -> bool:
    return find_cli() is not None and models_ready()


def build_cmd(cli: Path, image: Path, out: Path, res: int,
              extra: str = "") -> list[str]:
    cmd = [str(cli), str(image), str(out),
           "--res", str(int(res)), "--models", str(MODELS_DIR)]
    if extra and extra.strip():
        cmd += shlex.split(extra)
    return cmd


def generate(image_path: Path, out_path: Path, res: int = 512,
             extra: str = "", log: Callable[[str], None] | None = None,
             gpu_index: int | None = None) -> Path:
    """Génère un GLB 3D à partir d'une image. Bloquant (one-shot)."""
    cli = find_cli()
    if cli is None:
        raise sdcpp.EngineError(
            "trellis-cli introuvable — installez trellis.cpp (onglet 3D).")
    if not models_ready():
        raise sdcpp.EngineError(
            "Modèles trellis absents — installez-les (onglet 3D).")
    settings.ensure_dirs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(cli, image_path, out_path, res, extra)
    # Réutilise le lanceur streamé + annulable de sd.cpp (subprocess générique).
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    if not out_path.is_file():
        raise sdcpp.EngineError(
            "trellis-cli s'est terminé sans produire de GLB — voir le journal "
            "(VRAM insuffisante en 512 ? modèles incomplets ?).")
    return out_path
