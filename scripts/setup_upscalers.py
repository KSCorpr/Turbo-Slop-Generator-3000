#!/usr/bin/env python3
"""Installe un upscaler PyTorch : SeedVR2 ou NVIDIA PiD.

Conçu pour fonctionner avec le Python embarqué (aucun git requis) :
  1. télécharge le code du dépôt en ZIP (codeload GitHub) -> upscalers_repo/<id>
  2. installe PyTorch CUDA 12.1 (compatible RTX Turing→Blackwell)
  3. installe les dépendances du dépôt
  4. télécharge les poids

Peut être lancé depuis l'interface (bouton « Installer ») ou en ligne :
    python scripts/setup_upscalers.py seedvr2
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atelier import registry, settings  # noqa: E402


def _owner_repo(code_repo: str) -> tuple[str, str]:
    s = code_repo.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    parts = s.split("/")
    return parts[-2], parts[-1]


def _download_zip(owner: str, repo: str, dest: Path) -> None:
    """Télécharge et extrait le dépôt (branche main puis master) sans git."""
    last = None
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        try:
            print(f"Téléchargement du code ({branch}) : {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "atelier"})
            with urllib.request.urlopen(req, timeout=300) as r:
                blob = r.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(dest.parent / f"_tmp_{repo}")
            # Le ZIP extrait dans <repo>-<branch>/ : on aplatit dans dest.
            tmp = dest.parent / f"_tmp_{repo}"
            sub = next((p for p in tmp.iterdir() if p.is_dir()), tmp)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(sub), str(dest))
            shutil.rmtree(tmp, ignore_errors=True)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"Échec du téléchargement du code : {last}")


def sh(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upscaler_id", choices=["seedvr2", "nvidia-pid"])
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    up = next((u for u in registry.load_upscalers() if u.id == args.upscaler_id), None)
    if up is None:
        sys.exit(f"Upscaler inconnu : {args.upscaler_id}")

    settings.UPSCALERS_REPO_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = settings.UPSCALERS_REPO_DIR / up.id
    ckpt_dir = repo_dir / "ckpts"

    # 1. Code (ZIP, sans git) -------------------------------------------------
    if not repo_dir.is_dir():
        owner, repo = _owner_repo(up.code_repo)
        _download_zip(owner, repo, repo_dir)
    else:
        print(f"Code déjà présent : {repo_dir}")

    # 2. PyTorch CUDA 12.1 ----------------------------------------------------
    print("\nInstallation de PyTorch (CUDA 12.1)… (volumineux)")
    sh([sys.executable, "-m", "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121"])

    # 3. Dépendances du dépôt -------------------------------------------------
    req = repo_dir / "requirements.txt"
    if req.is_file():
        try:
            sh([sys.executable, "-m", "pip", "install", "-r", str(req)])
        except subprocess.CalledProcessError:
            print("⚠️  Certaines dépendances ont échoué (flash-attn/apex parfois "
                  "pénibles sous Windows). L'inférence peut tout de même marcher.")

    # 4. Poids ----------------------------------------------------------------
    print(f"\nTéléchargement des poids {up.weights_repo}…")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=up.weights_repo, local_dir=str(ckpt_dir))
        print(f"Poids dans {ckpt_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Échec du téléchargement des poids : {exc}")

    print(f"\n« {up.name} » installé. Disponible dans l'onglet Upscale.")


if __name__ == "__main__":
    main()
