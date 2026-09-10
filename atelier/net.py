"""Un seul besoin réseau : trouver un port libre pour l'interface locale.

Ce module a aussi su lister les adresses IP de la machine (`primary_ip`,
`lan_ips`), pour annoncer au démarrage l'adresse à partager avec les autres
postes du réseau. Le partage réseau est retiré — l'application n'écoute que
sur 127.0.0.1 — et ces deux fonctions sont parties avec lui.
"""
from __future__ import annotations

import socket


def find_free_port(start: int, count: int = 25, host: str = "") -> int:
    """Premier port libre à partir de `start`. host='' = toutes interfaces."""
    bind_host = "" if host in ("0.0.0.0", "") else host
    for p in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((bind_host, p))
                return p
            except OSError:
                continue
    return start
