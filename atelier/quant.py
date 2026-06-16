"""Sélection tolérante de la quantification d'un fichier GGUF.

Les dépôts n'ont pas toujours la quantification exacte demandée (ex. on veut
Q4_K_M mais seuls Q4_K et Q8_0 existent). On choisit alors la plus proche.
"""
from __future__ import annotations

# Échelle ordonnée approximative qualité/taille (du plus léger au plus lourd).
ORDER = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K", "Q3_K_L",
    "Q4_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K",
    "Q5_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K",
    "Q6_K", "Q8_0", "Q8_1", "F16", "BF16", "F32",
]
# Recherche du plus long token d'abord (Q4_K_M avant Q4_K).
_BY_LEN = sorted(ORDER, key=len, reverse=True)


def find_quant(name: str) -> str | None:
    up = name.upper()
    for q in _BY_LEN:
        if q in up:
            return q
    return None


def _idx(q: str | None) -> int | None:
    if q is None:
        return None
    try:
        return ORDER.index(q)
    except ValueError:
        return None


def best(names: list[str], requested: str) -> str | None:
    """Parmi `names`, renvoie celui dont le quant est le plus proche de `requested`.

    Tri : distance de quant croissante, puis qualité supérieure, puis nom court.
    Les fichiers sans quant reconnaissable sont fortement dépriorisés.
    """
    if not names:
        return None
    req = _idx(requested)

    def key(name: str):
        qi = _idx(find_quant(name))
        if qi is None:
            return (10_000, 0, len(name))
        if req is None:
            return (0, -qi, len(name))
        return (abs(qi - req), -qi, len(name))

    return sorted(names, key=key)[0]
