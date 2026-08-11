#!/usr/bin/env python3
"""Runner « décomposition en calques » : SAM automatique + tri par profondeur.

Produit un dossier de masques PNG (niveaux de gris) + un manifeste JSON décrivant
l'ordre d'empilement. L'assemblage du PSD se fait ensuite HORS de ce runner, dans
le Python principal — il n'a besoin ni de torch ni de transformers, et il n'y a
aucune raison de le faire tourner dans le sous-process lourd.

Deux difficultés, et elles ne sont pas dans le code de segmentation :

1. **SAM segmente l'apparence, pas le sens.** Sur une photo il rend volontiers
   quarante à quatre-vingts masques imbriqués — une chemise, un bouton, un pli,
   un reflet — et, pire, des « zones » faites de taches éparpillées aux quatre
   coins de l'image. Tout le travail utile est là : découper chaque masque en
   ses morceaux CONNEXES, boucher les trous intérieurs, écarter les miettes et
   les quasi-doublons, puis rendre les calques DISJOINTS pour que l'empilement
   reconstitue exactement l'image.

2. **SAM ne donne aucun ordre de profondeur.** On le récupère de Depth Anything
   V2 quand il est installé : profondeur médiane sous chaque masque, du plus
   loin au plus près. Sans lui, on trie par surface décroissante — un gros
   objet est plus souvent en arrière qu'un petit. C'est une approximation, et
   elle est annoncée comme telle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device  # noqa: E402


def _iou(a, b) -> float:
    inter = (a & b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float((a | b).sum())


def _filter_masks(masks, min_area, max_area, iou_max, log):
    """Réduit une soupe de masques à un jeu de calques exploitable.

    L'étape qui change tout est la PREMIÈRE : chaque masque brut est nettoyé et
    surtout DÉCOUPÉ en ses morceaux connexes. Sans elle, SAM rend volontiers une
    « zone » faite de trente taches éparpillées aux quatre coins de l'image — ce
    n'est pas un calque, c'est du bruit qu'aucun logiciel ne permet d'exploiter.
    Les trous intérieurs sont bouchés dans la foulée : ils donnaient aux
    découpes leur aspect de gruyère.

    Ensuite seulement : rejet des quasi-doublons. Volontairement PAS de rejet
    sur le recouvrement — une zone contenue dans une plus grande n'est pas
    redondante, elle est DEVANT (la voiture sur la route, le personnage devant
    un mur). La disjonction finale s'occupe du recouvrement en découpant
    l'arrière-plan.
    """
    from atelier.engine import masks as M

    pieces: list = []
    for m in masks:
        for part in M.largest_components(M.fill_holes(m), int(min_area)):
            pieces.append(part)
    log(f"[calques] {len(masks)} masque(s) bruts -> {len(pieces)} morceau(x) "
        "connexes après nettoyage.")

    kept: list = []
    dropped = {"grand": 0, "doublon": 0}
    for m in sorted(pieces, key=lambda x: int(x.sum()), reverse=True):
        area = int(m.sum())
        if area > max_area:
            # Masque « toute l'image » : c'est le fond, on l'a déjà.
            dropped["grand"] += 1
            continue
        # SEUL critère de rejet : le quasi-doublon. On ne rejette PLUS une zone
        # parce qu'une plus grande la recouvre — c'était le cas de la voiture
        # sur la route, du personnage devant un mur, de la fenêtre sur une
        # façade : contenu ne veut pas dire redondant, ça veut dire DEVANT.
        # La disjonction qui suit règle le recouvrement en découpant l'arrière,
        # ce qui rend ce filtre non seulement inutile mais nuisible.
        if any(_iou(m, k) > iou_max for k in kept):
            dropped["doublon"] += 1
            continue
        kept.append(m)
    log(f"[calques] {len(kept)} zone(s) retenue(s) — écartées : "
        + (", ".join(f"{v} {k}" for k, v in dropped.items() if v) or "aucune"))
    return kept


def _partition(ordered, log):
    """Rend les calques DISJOINTS : chaque pixel appartient à un seul.

    `ordered` va de l'arrière-plan vers le premier plan. On retire donc de
    chaque calque ce que les calques SITUÉS DEVANT lui recouvrent : c'est le
    comportement d'un vrai empilement, où l'avant-plan cache l'arrière, et ça
    garantit une propriété simple et vérifiable — tout afficher redonne l'image
    d'origine, sans qu'un pixel soit peint deux fois.
    """
    import numpy as np
    out = []
    front = None                       # union de tout ce qui est DEVANT
    for m in reversed(ordered):        # du premier plan vers le fond
        vis = m if front is None else (m & ~front)
        front = m.copy() if front is None else (front | m)
        out.append(vis)
    out.reverse()
    kept = [m for m in out if m.any()]
    if len(kept) != len(out):
        log(f"[calques] {len(out) - len(kept)} zone(s) entièrement masquée(s) "
            "par l'avant-plan, retirée(s).")
    return kept


def _auto_masks(model_dir, img, points_per_side, batch, log):
    """Masques SAM sans clic : grille de points sur toute l'image."""
    import numpy as np
    import torch
    from transformers import SamModel, SamProcessor

    device = pick_device(torch)
    log(f"[calques] chargement de SAM sur {label(device)}…")
    model = SamModel.from_pretrained(model_dir).to(device).eval()
    processor = SamProcessor.from_pretrained(model_dir)

    W, H = img.size
    # Grille en coordonnées PIXEL : le processeur attend des points image.
    step_x, step_y = W / (points_per_side + 1), H / (points_per_side + 1)
    grid = [[int(step_x * (i + 1)), int(step_y * (j + 1))]
            for j in range(points_per_side) for i in range(points_per_side)]
    log(f"[calques] {len(grid)} points de sondage sur {W}×{H}…")

    out: list = []
    for start in range(0, len(grid), batch):
        chunk = grid[start:start + batch]
        # [image][point][x, y] — un point par masque candidat.
        pts = [[[p] for p in chunk]]
        inputs = processor(img, input_points=pts, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs, multimask_output=True)
        masks = processor.image_processor.post_process_masks(
            res.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]
        scores = res.iou_scores.cpu()[0]            # (points, 3)
        for i in range(masks.shape[0]):
            best = int(scores[i].argmax())
            if float(scores[i][best]) < 0.80:
                continue                            # candidat peu sûr
            out.append(masks[i][best].numpy().astype(bool))
        log(f"[calques]   {min(start + batch, len(grid))}/{len(grid)} points…")
    return out


def _depth_order(depth_dir, img, masks, log):
    """Indices des masques triés du PLUS LOIN au plus près, ou None."""
    import numpy as np
    if not depth_dir or not Path(depth_dir).is_dir():
        log("[calques] profondeur non installée → tri par surface "
            "(approximation : les grandes zones passent derrière).")
        return None
    try:
        import torch
        from transformers import pipeline
        device = pick_device(torch)
        idx = int(device.split(":")[-1]) if device.startswith("cuda") else (
            -1 if device == "cpu" else device)
        pipe = pipeline("depth-estimation", model=str(depth_dir), device=idx)
        depth = np.asarray(pipe(img)["depth"], dtype="float32")
    except Exception as exc:  # noqa: BLE001
        log(f"[calques] profondeur indisponible ({exc}) → tri par surface.")
        return None
    # Depth Anything : valeur ÉLEVÉE = proche. On veut le fond d'abord.
    medians = [float(np.median(depth[m])) if m.any() else 0.0 for m in masks]
    log("[calques] ordre d'empilement déduit de la carte de profondeur.")
    return sorted(range(len(masks)), key=lambda i: medians[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-dir", required=True)
    ap.add_argument("--depth-dir", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--points-per-side", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-area", type=float, default=0.004,
                    help="surface minimale d'un calque, en fraction de l'image")
    ap.add_argument("--max-area", type=float, default=0.85)
    ap.add_argument("--iou-max", type=float, default=0.75)
    ap.add_argument("--max-layers", type=int, default=24)
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    import numpy as np
    from PIL import Image

    img = Image.open(args.input).convert("RGB")
    total = img.width * img.height
    masks = _auto_masks(args.sam_dir, img, args.points_per_side, args.batch, log)
    log(f"[calques] {len(masks)} masque(s) bruts.")
    kept = _filter_masks(masks, args.min_area * total, args.max_area * total,
                         args.iou_max, log)
    if not kept:
        log("[calques] aucune zone exploitable — image trop uniforme ?")
    order = _depth_order(args.depth_dir, img, kept, log)
    if order is None:
        order = sorted(range(len(kept)), key=lambda i: int(kept[i].sum()),
                       reverse=True)
    kept = [kept[i] for i in order][:args.max_layers]
    # Disjonction APRÈS le tri : elle dépend de qui est devant qui.
    kept = _partition(kept, log)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, m in enumerate(kept):
        name = f"mask_{i:02d}.png"
        Image.fromarray((m.astype("uint8") * 255), "L").save(out / name)
        ys, xs = np.nonzero(m)
        entries.append({
            "file": name,
            "area_pct": round(100.0 * int(m.sum()) / total, 2),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1],
        })
    (out / "layers.json").write_text(
        json.dumps({"width": img.width, "height": img.height,
                    "masks": entries}, indent=2), encoding="utf-8")
    log(f"[calques] {len(entries)} calque(s) écrits dans {out}")


if __name__ == "__main__":
    main()
