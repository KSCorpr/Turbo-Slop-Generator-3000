"""Quel moteur exécute la génération — et rien d'autre dans ce fichier.

Sur la branche principale il n'y a qu'un moteur, stable-diffusion.cpp, et cette
question ne se pose pas. Sur Test7000 il y en a deux, et tout le pari de la
branche est que l'INTERFACE ne s'en aperçoive pas : mêmes onglets, mêmes
réglages, mêmes identifiants de modèles, mêmes fichiers de sortie. La seule
chose qui change est ce qu'il y a derrière `generate.generate()`.

Le choix se lit dans les préférences, avec deux échappatoires utiles :

* la variable d'environnement `TURBOSLOP_BACKEND`, qui l'emporte sur tout —
  c'est ce qui permet de lancer deux fois la même application côte à côte, une
  par moteur, pour comparer sans rien modifier ;
* `prefs_override`, dont se sert le banc d'essai pour mesurer un moteur sans
  toucher au fichier de l'utilisateur.

Le défaut de la branche est PYTORCH. C'est sa raison d'être ; un défaut sd.cpp
aurait fait une branche où il faut penser à activer ce qu'on est venu essayer.
"""
from __future__ import annotations

import os

SDCPP = "sdcpp"
TORCH = "torch"
ALL = (SDCPP, TORCH)

#  Le défaut de CETTE branche. Sur `main`, la valeur est SDCPP.
DEFAULT = TORCH

ENV_VAR = "TURBOSLOP_BACKEND"

LABELS = {
    SDCPP: "stable-diffusion.cpp (native, GGUF)",
    TORCH: "PyTorch / diffusers",
}


def active(prefs: dict | None = None) -> str:
    """Le moteur à utiliser maintenant.

    Une valeur inconnue — un fichier de préférences édité à la main, une
    préférence héritée d'une autre branche — retombe sur le défaut au lieu de
    lever : on ne bloque pas le démarrage de l'application sur une chaîne de
    caractères.
    """
    forced = (os.environ.get(ENV_VAR) or "").strip().lower()
    if forced in ALL:
        return forced
    if prefs is None:
        from .. import settings
        prefs = settings.load_prefs()
    chosen = str(prefs.get("engine_backend") or "").strip().lower()
    return chosen if chosen in ALL else DEFAULT


def label(name: str) -> str:
    return LABELS.get(name, name)
