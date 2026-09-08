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

Le défaut est STABLE-DIFFUSION.CPP, et il le reste après la fusion de Test7000.
Ce n'est pas de la timidité : le moteur natif est livré avec l'application et
n'exige rien, là où PyTorch réclame plusieurs gigaoctets de paquets Python.
Basculer le défaut aurait cassé chaque installation existante à la mise à jour
suivante, pour une fonctionnalité que personne n'a demandée. Le moteur PyTorch
s'active dans les Réglages, sous « 🔧 Expert », ou par la variable
d'environnement.
"""
from __future__ import annotations

import os

SDCPP = "sdcpp"
TORCH = "torch"
ALL = (SDCPP, TORCH)

#  Le moteur livré et utilisé sans rien faire. La branche Test7000 met TORCH
#  ici : c'était sa raison d'être, et une branche où il faut penser à activer
#  ce qu'on vient essayer n'aurait servi à rien.
DEFAULT = SDCPP

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
