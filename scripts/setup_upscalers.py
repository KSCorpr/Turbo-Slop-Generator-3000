#!/usr/bin/env python3
"""Installe un upscaler PyTorch à la demande : SeedVR2 ou NVIDIA PiD.

Clone le dépôt de code, installe PyTorch CUDA 12.1 (compatible RTX Turing→Blackwell),
installe les dépendances du dépôt et télécharge les poids.

Usage :
    python scripts/setup_upscalers.py seedvr2
    python scripts/setup_upscalers.py nvidia-pid
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atelier import registry, settings  # noqa: E402


def sh(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upscaler_id", choices=["seedvr2", "nvidia-pid"])
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    up = next((u for u in registry.load_upscalers() if u.id == args.upscaler_id), None)
    if up is None:
        sys.exit(f"Upscaler inconnu : {args.upscaler_id}")

    repo_dir = settings.UPSCALERS_REPO_DIR / up.id
    ckpt_dir = repo_dir / "ckpts"
    settings.UPSCALERS_REPO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Code -----------------------------------------------------------------
    if not repo_dir.is_dir():
        sh(["git", "clone", "--depth", "1", up.code_repo, str(repo_dir)])
    else:
        print(f"Dépôt déjà présent : {repo_dir}")

    # 2. PyTorch CUDA 12.1 ----------------------------------------------------
    print("\nInstallation de PyTorch (CUDA 12.1)…")
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
        print("    Téléchargez-les manuellement dans", ckpt_dir)

    print(f"\n« {up.name} » installé. Disponible dans l'onglet Upscale.")


if __name__ == "__main__":
    main()
