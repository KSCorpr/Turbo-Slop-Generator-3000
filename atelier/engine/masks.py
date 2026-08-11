"""Nettoyage et description de masques binaires — numpy pur.

Écrit à la main plutôt qu'avec scipy/OpenCV : l'add-on de segmentation pèse
déjà lourd (PyTorch + transformers), et ces opérations tiennent en quelques
dizaines de lignes. Ajouter une dépendance de 40 Mo pour un remplissage de trous
serait un mauvais échange.

Les composantes connexes passent par un union-find sur les PLAGES de chaque
ligne, pas sur les pixels : sur une image d'un million de pixels, une union-find
par pixel en Python prendrait des secondes, alors que le nombre de plages se
compte en milliers. La même primitive sert au remplissage des trous — un trou
n'étant qu'une composante du complément qui ne touche aucun bord.
"""
from __future__ import annotations

import numpy as np


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Plages contiguës de True dans une ligne : [(début, fin exclue), …]."""
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Étiquette les composantes connexes (voisinage 4). (étiquettes, nombre).

    0 = fond. Les étiquettes retournées sont compactées à partir de 1.
    """
    h, w = mask.shape
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rows: list[list[tuple[int, int, int]]] = []      # (début, fin, étiquette)
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        cur: list[tuple[int, int, int]] = []
        for s, e in _runs(mask[y]):
            lab = 0
            for ps, pe, plab in prev:
                if ps < e and s < pe:                # plages qui se chevauchent
                    lab = plab if lab == 0 else lab
                    union(lab, plab)
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            cur.append((s, e, lab))
        rows.append(cur)
        prev = cur

    # Compactage des étiquettes après résolution des équivalences.
    remap: dict[int, int] = {}
    out = np.zeros((h, w), np.int32)
    for y, cur in enumerate(rows):
        for s, e, lab in cur:
            root = find(lab)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[y, s:e] = remap[root]
    return out, len(remap)


def component_sizes(labels: np.ndarray, count: int) -> np.ndarray:
    """Nombre de pixels par étiquette (index 0 = fond)."""
    return np.bincount(labels.ravel(), minlength=count + 1)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Bouche les trous INTÉRIEURS (ce qui donne l'aspect « gruyère »).

    Un trou est une composante du complément qui ne touche aucun bord : le fond
    extérieur, lui, en touche forcément un.
    """
    inv = ~mask
    labels, n = label_components(inv)
    if n == 0:
        return mask.copy()
    outside = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist())
    outside.discard(0)
    holes = inv & ~np.isin(labels, list(outside))
    return mask | holes


def largest_components(mask: np.ndarray, min_area: int,
                       max_parts: int = 0) -> list[np.ndarray]:
    """Découpe un masque en ses morceaux connexes, du plus grand au plus petit.

    C'est le correctif le plus visible : un « calque » fait de trente taches
    éparpillées aux quatre coins de l'image n'est pas un calque, c'est du bruit
    que rien ne permet d'utiliser. Chaque morceau devient une zone à part, et
    ceux qui n'atteignent pas `min_area` disparaissent.
    """
    labels, n = label_components(mask)
    if n == 0:
        return []
    sizes = component_sizes(labels, n)
    order = np.argsort(sizes[1:])[::-1] + 1
    out = []
    for lab in order:
        if sizes[lab] < min_area:
            break
        out.append(labels == lab)
        if max_parts and len(out) >= max_parts:
            break
    return out


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Flou moyen séparable par sommes cumulées (rapide, sans dépendance)."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    pad = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    cs = np.cumsum(pad, axis=0, dtype=np.float32)
    cs = np.vstack([np.zeros((1, cs.shape[1]), np.float32), cs])
    a = (cs[k:, :] - cs[:-k, :]) / k
    cs = np.cumsum(a, axis=1, dtype=np.float32)
    cs = np.hstack([np.zeros((cs.shape[0], 1), np.float32), cs])
    return (cs[:, k:] - cs[:, :-k]) / k


def feather_alpha(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Alpha 0-255 avec un bord adouci.

    Les masques de SAM sont binaires : collés tels quels, les découpes ont ce
    bord en marches d'escalier qui trahit le détourage automatique. Un ou deux
    pixels de transition suffisent à le faire disparaître.
    """
    a = _box_blur(mask.astype(np.float32), radius)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Description : donner un NOM utilisable à une zone.
#
#  « Zone 9 — 0,48 % » n'apprend rien à personne. Trois informations qu'on a
#  déjà sous la main suffisent à rendre la liste de calques lisible sans
#  ouvrir chaque vignette : où c'est, quelle taille, et de quelle couleur.
# --------------------------------------------------------------------------- #
_COLOR_NAMES = [
    ((15, 15, 18), "noir"), ((70, 70, 75), "gris foncé"),
    ((140, 140, 145), "gris"), ((205, 205, 210), "gris clair"),
    ((248, 248, 248), "blanc"),
    ((190, 40, 40), "rouge"), ((235, 130, 40), "orange"),
    ((240, 220, 90), "jaune"), ((70, 160, 70), "vert"),
    ((35, 90, 55), "vert foncé"),
    ((60, 120, 210), "bleu"), ((150, 190, 230), "bleu clair"),
    ((25, 50, 110), "bleu foncé"),
    ((120, 70, 170), "violet"), ((215, 120, 170), "rose"),
    ((130, 90, 60), "brun"), ((235, 210, 175), "beige"),
]


def dominant_color_name(rgb: np.ndarray, mask: np.ndarray) -> str:
    """Nom de la couleur MÉDIANE de la zone (la médiane résiste aux reflets).

    Comparaison pondérée façon luminance : l'œil sépare d'abord le clair du
    sombre. Une distance euclidienne brute en RGB fait passer un gris anthracite
    pour du vert foncé — les trois canaux y pèsent pareil alors qu'ils ne
    comptent pas pareil.
    """
    if not mask.any():
        return ""
    med = np.median(rgb[mask].reshape(-1, 3), axis=0).astype(np.float32)
    best, score = "", None
    for ref, label in _COLOR_NAMES:
        c = np.array(ref, np.float32)
        # Écart de luminosité, puis écart de teinte une fois la luminosité ôtée.
        dl = abs(float(med.mean()) - float(c.mean()))
        dh = float(np.abs((med - med.mean()) - (c - c.mean())).sum())
        s = dl * 1.6 + dh
        if score is None or s < score:
            best, score = label, s
    return best


def position_name(mask: np.ndarray) -> str:
    """« haut gauche », « centre », « bas droite »… d'après le centre de masse."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ""
    h, w = mask.shape
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = "haut" if cy < 0.34 else ("bas" if cy > 0.66 else "milieu")
    horiz = "gauche" if cx < 0.34 else ("droite" if cx > 0.66 else "centre")
    if vert == "milieu" and horiz == "centre":
        return "centre"
    return f"{vert} {horiz}".replace("milieu ", "").replace(" centre", "")


def depth_band_name(rank: int, total: int) -> str:
    """Bande de profondeur : c'est ce qui compte le plus pour un calque."""
    if total <= 1:
        return "plan unique"
    r = rank / max(1, total - 1)
    if r < 0.34:
        return "arrière-plan"
    if r > 0.66:
        return "premier plan"
    return "plan médian"


def describe(mask: np.ndarray, rgb: np.ndarray, rank: int, total: int) -> str:
    """Nom de calque lisible : plan, position, couleur, taille."""
    pct = 100.0 * float(mask.sum()) / mask.size
    bits = [depth_band_name(rank, total)]
    pos = position_name(mask)
    if pos:
        bits.append(pos)
    col = dominant_color_name(rgb, mask)
    if col:
        bits.append(col)
    return " · ".join(bits) + f" — {pct:.1f}%"
