"""Pavage d'une inférence d'agrandissement — numpy pur, aucun torch ici.

Ce module existe parce que **la couture est le premier défaut visible** d'un
upscale, et qu'elle ne vient pas du modèle : elle vient de la façon dont on
découpe l'image avant de le lancer.

Trois décisions, dans l'ordre d'importance :

1. **Ne pas paver du tout quand ça tient.** Une image entière passée en une
   fois n'a aucune couture — pas « peu », aucune, par construction. Le pavage
   n'est pas une qualité qu'on règle, c'est un pis-aller qu'on évite.

2. **Jeter le halo.** Un réseau de super-résolution se comporte mal sur ses
   propres bords : il n'y voit pas de contexte et invente. On lui donne donc
   plus de pixels qu'on n'en garde, et on **jette** la marge au lieu de la
   fondre. Fondre deux bords ratés donne un bord raté flou.

3. **Fondre ce qui reste.** Après le halo il subsiste un écart de décision
   entre tuiles voisines — pas un artefact de bord, une divergence de contenu.
   Une rampe linéaire sur la zone de recouvrement la répartit au lieu de la
   concentrer sur une ligne.

Ce qu'on ne fait PAS, et c'est délibéré : aucun redimensionnement pour
atteindre une cible. Le facteur est celui du modèle, point. Interpoler après
coup pour tomber juste, c'est exactement le défaut qu'on cherche à fuir.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def plan_tiles(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Découpe [0, length) en segments `tile` qui se recouvrent de `overlap`.

    Le dernier segment est CALÉ SUR LA FIN plutôt qu'ajouté en débordement :
    sinon une image de 1030 px avec des tuiles de 512 produit un dernier
    segment de 6 px, dont le réseau ne peut rien faire de bon.
    """
    if tile <= 0 or overlap < 0 or tile <= overlap:
        raise ValueError("tile must be positive and larger than overlap")
    if length <= tile:
        return [(0, length)]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(s, s + tile) for s in starts]


def _ramp(n: int) -> np.ndarray:
    """Rampe 0→1 sur n points, sans jamais valoir 0 ni 1.

    Un poids nul rendrait une tuile inutile sur sa marge ; un poids de 1 des
    deux côtés d'un joint remettrait une ligne franche.
    """
    return (np.arange(1, n + 1, dtype=np.float32) / (n + 1))


def _window(start: int, end: int, limit: int, overlap: int) -> np.ndarray:
    """Poids d'une tuile le long d'un axe : rampe aux joints, plat ailleurs."""
    w = np.ones(end - start, dtype=np.float32)
    n = min(overlap, len(w))
    if n > 0:
        ramp = _ramp(n)
        if start > 0:                      # joint à gauche/en haut
            w[:n] *= ramp
        if end < limit:                    # joint à droite/en bas
            w[-n:] *= ramp[::-1]
    return w


def upscale_tiled(image: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
                  scale: int, tile: int = 0, overlap: int = 32, halo: int = 16,
                  log: Callable[[str], None] | None = None) -> np.ndarray:
    """Agrandit `image` (HWC float32 [0,1]) en appelant `predict` par morceaux.

    `predict` reçoit un bloc HWC [0,1] et doit rendre EXACTEMENT ×`scale`.
    `tile` à 0 (ou plus grand que l'image) = une seule passe, sans couture.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB HxWx3 array")
    h, w = image.shape[:2]

    # Le cas qui compte : ça tient, donc on ne coupe rien.
    if tile <= 0 or (tile >= h and tile >= w):
        if log:
            log(f"[upscale] {w}×{h} in a single pass — no tiles, no seam.")
        out = np.asarray(predict(image), dtype=np.float32)
        _check(out, (h * scale, w * scale, 3))
        return np.clip(out, 0.0, 1.0)

    rows, cols = plan_tiles(h, tile, overlap), plan_tiles(w, tile, overlap)
    if log:
        log(f"[upscale] {w}×{h} in {len(rows) * len(cols)} tile(s) of {tile} px "
            f"(overlap {overlap}, halo {halo} discarded)")

    accum = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    for y0, y1 in rows:
        for x0, x1 in cols:
            # Contexte supplémentaire donné au réseau, puis jeté.
            ty0, tx0 = max(0, y0 - halo), max(0, x0 - halo)
            ty1, tx1 = min(h, y1 + halo), min(w, x1 + halo)
            block = np.ascontiguousarray(image[ty0:ty1, tx0:tx1])
            pred = np.asarray(predict(block), dtype=np.float32)
            _check(pred, ((ty1 - ty0) * scale, (tx1 - tx0) * scale, 3))
            # Recadrage sur la tuile utile : le halo n'entre jamais dans la somme.
            crop = pred[(y0 - ty0) * scale:(y1 - ty0) * scale,
                        (x0 - tx0) * scale:(x1 - tx0) * scale]
            mask = (_window(y0, y1, h, overlap)[:, None, None].repeat(scale, 0)
                    * _window(x0, x1, w, overlap)[None, :, None].repeat(scale, 1))
            accum[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += crop * mask
            weight[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += mask

    if not (weight > 0).all():
        # Impossible avec plan_tiles, mais un trou passerait autrement pour un
        # aplat noir dans l'image finale — autant le dire.
        raise RuntimeError("tile coverage left a hole")
    return np.clip(accum / weight, 0.0, 1.0)


def _check(array: np.ndarray, expected: tuple[int, int, int]) -> None:
    if array.shape != expected:
        raise ValueError(
            f"the model returned {array.shape}, expected {expected}. "
            "Nothing was resized to hide the mismatch.")
    if not np.isfinite(array).all():
        # Typiquement du fp16 sur une architecture qui ne le supporte pas :
        # l'appelant réessaie en fp32 plutôt que d'écrire une image trouée.
        raise ArithmeticError("the model produced non-finite pixels")


def tile_for_vram(width: int, height: int, scale: int,
                  free_gb: float | None) -> int:
    """Taille de tuile tenant dans la VRAM libre. 0 = pas besoin de paver.

    L'estimation est volontairement prudente : se tromper vers le bas coûte
    des coutures qu'on sait atténuer, se tromper vers le haut coûte un échec
    et un rechargement. On vise ~1,5 octet par pixel de SORTIE et par canal,
    avec les activations intermédiaires — mesuré sur DAT et SPAN, les deux
    familles les plus gourmandes de la sélection.
    """
    if not free_gb or free_gb <= 0:
        return 1024
    budget = max(0.5, free_gb - 1.0) * (1024 ** 3)   # 1 Go gardé pour le reste
    per_src_pixel = 3 * 4 * scale * scale * 6        # sortie + activations
    fits = int((budget / per_src_pixel) ** 0.5)
    if fits >= max(width, height):
        return 0                                     # une passe, aucune couture
    return max(128, min(1024, (fits // 64) * 64))
