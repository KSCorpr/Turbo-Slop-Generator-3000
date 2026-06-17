#!/usr/bin/env python3
"""Télécharge un binaire pré-compilé de stable-diffusion.cpp (sd-cli) dans ./bin.

Pour Windows + CUDA, récupère DEUX archives :
  - la build principale  : sd-master-*-bin-win-cuda12-x64.zip  (contient sd-cli)
  - le runtime CUDA      : cudart-sd-bin-win-cu12-x64.zip       (DLLs CUDA)
…et les décompresse côte à côte (indispensable sans CUDA Toolkit installé).

Usage :
    python scripts/get_sdcpp.py                 # auto (CUDA)
    python scripts/get_sdcpp.py --variant cpu   # build CPU/AVX2
    python scripts/get_sdcpp.py --list          # liste les archives dispo
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import socket
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

try:
    import requests  # plus robuste qu'urllib sur les redirections GitHub (Windows)
except ImportError:
    requests = None

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=10"
_UA = {"User-Agent": "atelier"}


def _force_ipv4():
    """Force la résolution IPv4 uniquement.

    Sur beaucoup de réseaux Windows, IPv6 est annoncé mais non routable : Python
    tente l'IPv6 du CDN GitHub et reste bloqué (pas de Happy-Eyeballs comme les
    navigateurs). On filtre getaddrinfo pour ne garder que l'IPv4.
    """
    _orig = socket.getaddrinfo

    def _ipv4_only(host, *args, **kwargs):
        res = _orig(host, *args, **kwargs)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 or res

    socket.getaddrinfo = _ipv4_only


def _fetch_json(url: str):
    if requests is not None:
        r = requests.get(url, headers=_UA, timeout=60)
        r.raise_for_status()
        return r.json()
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _platform_tokens() -> list[str]:
    s = platform.system().lower()
    return {"windows": ["win"], "linux": ["linux", "ubuntu"],
            "darwin": ["darwin", "macos"]}.get(s, [])


def _is_archive(n: str) -> bool:
    return n.endswith((".zip", ".tar.gz", ".tgz"))


def _score_main(name: str, variant: str) -> int:
    """Note l'archive PRINCIPALE (qui contient sd-cli). Exclut le cudart."""
    n = name.lower()
    if not _is_archive(n) or "cudart" in n:
        return -1000
    toks = _platform_tokens()
    if toks and not any(t in n for t in toks):
        return -1000
    score = 10
    if variant == "cuda":
        if "cuda12" in n or "cu12" in n:
            score += 10
        elif "cuda" in n:
            score += 8
        else:
            score -= 4              # pas une build CUDA
        if "rocm" in n or "vulkan" in n:
            score -= 20
    else:  # cpu
        if any(x in n for x in ("cuda", "rocm", "vulkan")):
            score -= 20
        if "avx2" in n:
            score += 6
        elif "avx512" in n:
            score += 3
        elif "avx" in n:
            score += 4
        elif "noavx" in n:
            score += 1
    if any(t in n for t in ("x64", "amd64", "x86_64")):
        score += 1
    return score


def _find_cudart(assets: list[dict]) -> dict | None:
    for a in assets:
        n = a["name"].lower()
        if "cudart" in n and "win" in n and _is_archive(n):
            return a
    return None


def _latest_release_with_assets() -> dict:
    data = _fetch_json(RELEASES)
    if isinstance(data, dict):  # message d'erreur (rate limit, etc.)
        sys.exit(f"API GitHub : {data.get('message', data)}")
    for rel in data:
        if rel.get("assets"):
            return rel
    sys.exit("Aucune release avec archives trouvée.")


def _progress(got: int, total: int, last: int) -> int:
    step = (total / 20) if total else 5_000_000
    if got - last >= step:
        if total:
            print(f"     {got/1e6:6.1f} / {total/1e6:.1f} Mo ({got*100//total}%)",
                  flush=True)
        else:
            print(f"     {got/1e6:6.1f} Mo téléchargés…", flush=True)
        return got
    return last


def _download(url: str) -> bytes:
    """Télécharge en streaming avec progression. Utilise requests si dispo
    (gère proprement les redirections du CDN GitHub, contrairement à urllib)."""
    buf = bytearray()
    got = last = 0
    if requests is not None:
        # timeout=(connexion 15 s, lecture 120 s) : un blocage lève une erreur.
        with requests.get(url, headers=_UA, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0) or 0)
            for chunk in r.iter_content(262144):
                if not chunk:
                    continue
                buf += chunk
                got += len(chunk)
                last = _progress(got, total, last)
    else:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0) or 0)
            while True:
                block = r.read(262144)
                if not block:
                    break
                buf += block
                got += len(block)
                last = _progress(got, total, last)
    print(f"     terminé ({got/1e6:.1f} Mo).", flush=True)
    return bytes(buf)


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
    ap.add_argument("--allow-ipv6", action="store_true",
                    help="ne pas forcer l'IPv4 (par défaut on force l'IPv4)")
    args = ap.parse_args()

    if not args.allow_ipv6:
        _force_ipv4()

    print("Recherche de la dernière release stable-diffusion.cpp…")
    rel = _latest_release_with_assets()
    assets = rel["assets"]
    print(f"Release : {rel.get('tag_name')}")
    if args.list:
        for a in assets:
            print(" ", a["name"])
        return

    best = max(assets, key=lambda a: _score_main(a["name"], args.variant))
    if _score_main(best["name"], args.variant) <= 0:
        print("Aucune archive principale ne correspond. Disponibles :")
        for a in assets:
            print(" ", a["name"])
        sys.exit("Téléchargez-en une manuellement dans ./bin.")

    print(f"Téléchargement (binaire) : {best['name']}")
    _extract(_download(best["browser_download_url"]), best["name"])

    # Runtime CUDA (Windows) : nécessaire à côté du binaire.
    if args.variant == "cuda" and platform.system().lower() == "windows":
        cudart = _find_cudart(assets)
        if cudart:
            print(f"Téléchargement (runtime CUDA) : {cudart['name']}")
            _extract(_download(cudart["browser_download_url"]), cudart["name"])
        else:
            print("⚠️ Runtime CUDA (cudart) introuvable dans la release ; "
                  "si sd-cli ne démarre pas, installez le CUDA Toolkit 12.")

    print(f"Décompressé dans {BIN_DIR}. Binaire sd-cli prêt.")


if __name__ == "__main__":
    main()
