"""Paramètres façon Midjourney tapés directement dans le prompt.

Midjourney se pilote avec des suffixes (`--ar 16:9`, `--stylize 500`, `--no cars`)
plutôt qu'avec des curseurs. On accepte la même syntaxe : elle est retirée du
texte envoyé au modèle et convertie en réglages de l'application.

Reconnus et APPLIQUÉS :
  --ar / --aspect W:H     -> largeur/hauteur (à surface constante)
  --stylize / -s N        -> intensité stylistique (0–1000)
  --chaos / -c N          -> diversité entre les propositions (0–100)
  --no a, b, c            -> prompt négatif

Reconnus et IGNORÉS (propres à Midjourney, sans équivalent ici) : --v, --version,
--niji, --q, --quality, --style, --weird, --w, --tile, --iw, --repeat, --sref,
--cref, --seed, --sameseed, --stop. Ils sont retirés du prompt et signalés.
"""
from __future__ import annotations

import math
import re

# Paramètres consommés par l'app.
_APPLIED = {
    "ar": "ar", "aspect": "ar",
    "stylize": "stylize", "s": "stylize",
    "chaos": "chaos", "c": "chaos",
    "no": "no",
}
# Paramètres Midjourney sans équivalent : on les retire proprement du prompt.
_IGNORED = {"v", "version", "niji", "q", "quality", "style", "weird", "w",
            "tile", "iw", "repeat", "r", "sref", "cref", "cw", "seed",
            "sameseed", "stop", "turbo", "relax", "fast", "raw"}

# « --nom » suivi de sa valeur (tout jusqu'au prochain « -- » ou la fin).
_TOKEN = re.compile(r"--([A-Za-z]+)\s*([^-]*(?:-(?!-)[^-]*)*)")


def parse(text: str) -> tuple[str, dict]:
    """(prompt nettoyé, paramètres). Jamais d'exception : un paramètre mal
    formé est simplement ignoré."""
    raw = text or ""
    out: dict = {"ignored": []}
    if "--" not in raw:
        return raw.strip(), out

    def _take(m: "re.Match") -> str:
        name = m.group(1).lower()
        val = (m.group(2) or "").strip()
        key = _APPLIED.get(name)
        if key == "ar":
            mm = re.match(r"^(\d{1,3})\s*[:x/]\s*(\d{1,3})", val)
            if mm and int(mm.group(1)) and int(mm.group(2)):
                out["ar"] = (int(mm.group(1)), int(mm.group(2)))
                return ""
        elif key in ("stylize", "chaos"):
            mm = re.match(r"^(\d{1,4})", val)
            if mm:
                out[key] = int(mm.group(1))
                return ""
        elif key == "no":
            if val:
                out["no"] = ", ".join(p.strip() for p in val.split(",")
                                      if p.strip())
                return ""
        elif name in _IGNORED:
            out["ignored"].append("--" + name)
            return ""
        return m.group(0)          # inconnu : on laisse le texte intact

    cleaned = _TOKEN.sub(_take, raw)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().strip(",").strip()
    return cleaned, out


def aspect_size(ratio: tuple[int, int], base_w: int, base_h: int,
                multiple: int = 16, min_side: int = 256,
                max_side: int = 2048) -> tuple[int, int]:
    """Dimensions au ratio demandé, à **surface constante** — on garde le nombre
    de pixels natif du modèle (c'est là qu'il rend le mieux), on ne fait que
    changer la forme."""
    rw, rh = ratio
    area = max(1, int(base_w) * int(base_h))
    r = rw / rh
    w = math.sqrt(area * r)
    h = w / r
    def _snap(v: float) -> int:
        v = int(round(v / multiple)) * multiple
        return max(min_side, min(max_side, v))
    return _snap(w), _snap(h)


def describe(p: dict) -> str:
    """Résumé lisible des paramètres reconnus (vide si aucun)."""
    bits = []
    if p.get("ar"):
        bits.append(f"format {p['ar'][0]}:{p['ar'][1]}")
    if p.get("stylize") is not None:
        bits.append(f"stylize {p['stylize']}")
    if p.get("chaos") is not None:
        bits.append(f"chaos {p['chaos']}")
    if p.get("no"):
        bits.append(f"sans « {p['no']} »")
    if p.get("ignored"):
        bits.append("ignoré : " + " ".join(sorted(set(p["ignored"]))))
    return " · ".join(bits)
