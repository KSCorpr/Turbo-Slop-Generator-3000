#!/usr/bin/env python3
"""Télécharge un binaire pré-compilé de stable-diffusion.cpp (sd-cli) dans ./bin.

Choisit l'archive correspondant à la plateforme (Windows CUDA 12 par défaut,
idéal pour les cartes RTX). `--variant cpu` force une build CPU/AVX2.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "atelier"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _score(name: str, variant: str) -> int:
    n = name.lower()
    sysname = platform.system().lower()
    score = 0
    if sysname == "windows" and "win" in n:
        score += 10
    if sysname == "linux" and ("linux" in n or "ubuntu" in n):
        score += 10
    if variant == "cuda" and ("cuda12" in n or "cu12" in n):
        score += 8
    elif variant == "cuda" and "cuda" in n:
        score += 6
    if variant == "cpu" and "cuda" not in n:
        score += 4
    if "avx2" in n and variant == "cpu":
        score += 5
    if any(t in n for t in ("x64", "amd64", "x86_64")):
        score += 1
    if not n.endswith((".zip", ".tar.gz", ".tgz")):
        score -= 100
    return score


def _extract(blob: bytes, name: str) -> None:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(BIN_DIR)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            t.extractall(BIN_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    print("Recherche de la dernière release stable-diffusion.cpp…")
    assets = _fetch_json(API).get("assets", [])
    if not assets:
        sys.exit("Aucune archive trouvée.")
    if args.list:
        for a in assets:
            print(" ", a["name"])
        return

    best = max(assets, key=lambda a: _score(a["name"], args.variant))
    if _score(best["name"], args.variant) <= 0:
        print("Aucune archive ne correspond. Disponibles :")
        for a in assets:
            print(" ", a["name"])
        sys.exit("Téléchargez-en une manuellement dans ./bin.")

    name, url = best["name"], best["browser_download_url"]
    print(f"Téléchargement : {name}")
    req = urllib.request.Request(url, headers={"User-Agent": "atelier"})
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    _extract(blob, name)
    print(f"Décompressé dans {BIN_DIR}. Binaire sd-cli prêt.")


if __name__ == "__main__":
    main()
