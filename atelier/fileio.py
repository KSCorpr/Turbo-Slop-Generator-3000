"""Écriture de fichiers qui survit à une coupure. Bibliothèque standard seule.

Le cas qu'on évite : `preferences.json` est réécrit à CHAQUE changement de
réglage — c'est-à-dire souvent, et parfois pendant une génération. `write_text`
tronque le fichier PUIS écrit ; une coupure de courant, un plantage ou un
antivirus qui verrouille le fichier au mauvais moment laisse un JSON à moitié
écrit, et l'application ne redémarre plus. Perdre le dernier réglage est
acceptable ; perdre le fichier ne l'est pas.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Écrit `text` dans `path` en une seule opération visible.

    Le temporaire est créé DANS le dossier de destination : `os.replace` n'est
    atomique qu'à l'intérieur d'un même volume, et `%TEMP%` est régulièrement
    sur un autre disque que le projet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as fh:
            temp = Path(fh.name)
            fh.write(text)
            fh.flush()
            # Sans fsync, le contenu peut n'être qu'en cache d'écriture : le
            # remplacement serait atomique mais la cible vide après coupure.
            os.fsync(fh.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        # Un temporaire laissé derrière par un échec est un déchet caché dans
        # userdata/ ; après un remplacement réussi il n'existe plus.
        if temp is not None:
            temp.unlink(missing_ok=True)
