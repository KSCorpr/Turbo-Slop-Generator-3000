"""Paquet moteur : rien d'autre que le paquet.

Ce fichier a contenu trois fonctions du temps où le moteur pouvait rester
chargé en mémoire entre deux images (`sd-server`) : `resident_engine()`,
`release_resident_engine()`, et `engine_build_source()`, qui ne servait qu'à
expliquer pourquoi l'option n'était pas disponible.

L'option est retirée. Elle demandait à chaque outil du projet — sd-cli, la 3D,
la boîte à outils — de penser à rendre la carte avant de la prendre, et un
oubli ne se voyait qu'au moment d'un manque de mémoire, loin de sa cause. Le
moteur démarre, fait l'image, et sort.
"""
