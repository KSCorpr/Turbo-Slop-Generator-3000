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


def _clip_labels(clip_dir, img, masks, log):
    """Étiquette chaque zone par CLIP en zéro-shot. [(étiquette, score)] ou None.

    Le découpage envoyé à CLIP mérite une explication : ni la boîte englobante
    brute, ni la découpe sur fond noir.

    · La boîte brute noie un objet fin dans son décor — un mât au milieu d'un
      ciel se fait étiqueter « ciel ».
    · La découpe sur fond noir supprime tout contexte et déroute CLIP, qui a été
      entraîné sur des photos entières, pas sur des silhouettes.

    On garde donc la boîte (avec une marge) en ATTÉNUANT l'extérieur du masque
    vers un gris neutre : l'objet ressort sans que sa scène disparaisse.
    """
    if not clip_dir or not Path(clip_dir).is_dir():
        log("[calques] CLIP non installé → pas d'étiquetage sémantique "
            "(les calques seront nommés par position et couleur).")
        return None
    import numpy as np
    from PIL import Image
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from atelier.engine import vocab
    except Exception as exc:  # noqa: BLE001
        log(f"[calques] CLIP indisponible ({exc}) → pas d'étiquetage.")
        return None

    device = pick_device(torch)
    log(f"[calques] étiquetage CLIP de {len(masks)} zone(s) sur {label(device)}…")
    model = CLIPModel.from_pretrained(clip_dir).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_dir)
    texts, owner = vocab.prompts()
    names = [n for n, _v, _d in vocab.entries()]

    with torch.no_grad():
        tin = processor(text=texts, return_tensors="pt", padding=True).to(device)
        temb = model.get_text_features(**tin)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    crops = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = max(8, int(0.12 * max(y1 - y0, x1 - x0)))
        y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
        sub = rgb[y0:y1, x0:x1].astype(np.float32)
        keep = m[y0:y1, x0:x1][..., None].astype(np.float32)
        # Extérieur du masque atténué vers un gris neutre (60 % de fondu).
        blended = sub * keep + (sub * 0.4 + 128.0 * 0.6) * (1.0 - keep)
        crops.append(Image.fromarray(blended.clip(0, 255).astype("uint8")))

    out = []
    BATCH = 16
    for start in range(0, len(crops), BATCH):
        chunk = crops[start:start + BATCH]
        with torch.no_grad():
            iin = processor(images=chunk, return_tensors="pt").to(device)
            iemb = model.get_image_features(**iin)
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            sims = (iemb @ temb.T).cpu().numpy()
        for row in sims:
            # Score d'une CATÉGORIE = meilleure de ses formulations.
            per = {}
            for j, o in enumerate(owner):
                per[o] = max(per.get(o, -9.9), float(row[j]))
            best = max(per, key=per.get)
            # Marge sur le second : une étiquette qui ne gagne que d'un cheveu
            # n'est pas une information, c'est un tirage au sort.
            ordered = sorted(per.values(), reverse=True)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            out.append((names[best], margin))
    return out


def _merge_by_label(masks, labels, log, gap=12):
    """Fusionne les zones VOISINES portant la même étiquette.

    C'est ici que l'étiquetage sert vraiment à la segmentation, et pas seulement
    à l'affichage : SAM rend « carrosserie », « portière », « roue » comme trois
    masques distincts. Étiquetés « véhicule » tous les trois et adjacents, ils
    redeviennent UN calque — ce qu'un humain appelle une voiture.

    L'adjacence est exigée : deux voitures aux extrémités de l'image partagent
    l'étiquette mais ne sont pas le même objet.
    """
    import numpy as np
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1))

    def near(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 + gap < bx0 or bx1 + gap < ax0
                    or ay1 + gap < by0 or by1 + gap < ay0)

    parent = list(range(len(masks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if labels[i][0] == labels[j][0] and near(boxes[i], boxes[j]):
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == len(masks):
        log("[calques] aucune fusion sémantique (aucune zone voisine de même "
            "nature).")
        return masks, labels

    out_m, out_l = [], []
    for members in groups.values():
        m = masks[members[0]].copy()
        for k in members[1:]:
            m |= masks[k]
        out_m.append(m)
        # On garde la meilleure marge du groupe : c'est le membre le plus sûr
        # qui répond de l'étiquette commune.
        out_l.append(max((labels[k] for k in members), key=lambda x: x[1]))
    log(f"[calques] fusion sémantique : {len(masks)} zone(s) -> {len(out_m)} "
        "(morceaux d'un même objet regroupés).")
    return out_m, out_l


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
    ap.add_argument("--clip-dir", default="",
                    help="modèle CLIP : étiquetage sémantique + fusion des "
                         "morceaux d'un même objet. Facultatif.")
    ap.add_argument("--junk-margin", type=float, default=0.012,
                    help="marge minimale entre la 1re et la 2e étiquette : "
                         "en dessous, l'étiquette est un tirage au sort")
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
    # --- Sémantique (facultative) : étiqueter, écarter le vide, fusionner ---
    labels = _clip_labels(args.clip_dir, img, kept, log) if kept else None
    if labels:
        from atelier.engine import vocab
        keep_idx = []
        for i, (name, margin) in enumerate(labels):
            if name in vocab.JUNK_LABELS:
                continue                     # flou, texture plate, fragment
            if margin < args.junk_margin:
                continue                     # aucune étiquette ne se détache
            keep_idx.append(i)
        dropped = len(kept) - len(keep_idx)
        if dropped:
            log(f"[calques] {dropped} zone(s) écartée(s) : ne correspondent à "
                "rien d'identifiable (flou, aplat, fragment).")
        # Garde-fou : si TOUT est écarté, l'étiquetage s'est trompé, pas
        # l'image. Mieux vaut des calques sans nom que pas de calques.
        if keep_idx:
            kept = [kept[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
        else:
            log("[calques] toutes les zones jugées non identifiables — "
                "étiquetage ignoré.")
            labels = None
    if labels:
        kept, labels = _merge_by_label(kept, labels, log)

    order = _depth_order(args.depth_dir, img, kept, log)
    if order is None and labels:
        # Sans carte de profondeur, la sémantique fait un bien meilleur juge que
        # la surface : le ciel va derrière parce que c'est le ciel, pas parce
        # qu'il est grand.
        from atelier.engine import vocab
        order = sorted(range(len(kept)),
                       key=lambda i: vocab.typical_depth(labels[i][0]))
        log("[calques] ordre d'empilement déduit des étiquettes "
            "(ciel et sol derrière, sujets devant).")
    if order is None:
        order = sorted(range(len(kept)), key=lambda i: int(kept[i].sum()),
                       reverse=True)
    kept = [kept[i] for i in order][:args.max_layers]
    if labels:
        labels = [labels[i] for i in order][:args.max_layers]
    # Disjonction APRÈS le tri : elle dépend de qui est devant qui.
    before = len(kept)
    kept = _partition(kept, log)
    if labels and len(kept) != before:
        labels = labels[:len(kept)]

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
            "label": labels[i][0] if labels else "",
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1],
        })
    (out / "layers.json").write_text(
        json.dumps({"width": img.width, "height": img.height,
                    "masks": entries}, indent=2), encoding="utf-8")
    log(f"[calques] {len(entries)} calque(s) écrits dans {out}")


if __name__ == "__main__":
    main()
