#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  upscale -> SDXL + ControlNet Tile : upscale créatif (façon Magnific).

Réutilise les helpers torch CUDA de _torch_setup (build adaptée au GPU,
sans verrouiller de DLL). Lançable depuis l'interface ou en ligne :
    python scripts/setup_tools.py depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atelier import settings  # noqa: E402
from _torch_setup import ensure_torch_cuda, pin_numpy, sh  # noqa: E402

# Modèle léger (~100 Mo) : rapide, tourne même sur Pascal (GTX 10xx) et en CPU.
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# RMBG-1.4 (~176 Mo) : code self-contained (pur torch), poids sur HF.
BG_REPO = "briaai/RMBG-1.4"
# Upscale créatif : SDXL base (1 fichier) + ControlNet Tile + VAE fp16-fix.
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0.safetensors"
CN_TILE_REPO = "xinsir/controlnet-tile-sdxl-1.0"
VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"


def install_depth():
    model_dir = settings.ROOT / "tools_repo" / "depth" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5", "pillow"])
    print(f"\nTéléchargement du modèle de profondeur ({DEPTH_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DEPTH_REPO, local_dir=str(model_dir))
    pin_numpy()  # transformers peut réintroduire NumPy 2 -> on re-fige
    print("\n[OK] Depth Anything V2 installé (profondeur + normales).")


def install_bg():
    model_dir = settings.ROOT / "tools_repo" / "bg" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5",
        "scikit-image", "pillow"])
    print(f"\nTéléchargement du modèle de suppression d'arrière-plan ({BG_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=BG_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] RMBG-1.4 installé. Disponible dans l'onglet Toolkit.")


def install_upscale():
    base = settings.ROOT / "tools_repo" / "upscale"
    ensure_torch_cuda()
    print("Installation de diffusers + accelerate…")
    sh([sys.executable, "-m", "pip", "install", "diffusers>=0.30,<0.32",
        "transformers>=4.45,<5", "accelerate", "safetensors", "omegaconf", "pillow"])
    from huggingface_hub import hf_hub_download, snapshot_download
    print(f"\nTéléchargement du checkpoint SDXL ({SDXL_REPO}/{SDXL_FILE}, ~6,6 Go)…")
    hf_hub_download(repo_id=SDXL_REPO, filename=SDXL_FILE, local_dir=str(base))
    print(f"\nTéléchargement du ControlNet Tile ({CN_TILE_REPO}, ~2,5 Go)…")
    snapshot_download(repo_id=CN_TILE_REPO, local_dir=str(base / "controlnet"),
                      allow_patterns=["*.json", "*.safetensors"])
    print(f"\nTéléchargement de la VAE fp16-fix ({VAE_FIX_REPO})…")
    snapshot_download(repo_id=VAE_FIX_REPO, local_dir=str(base / "vae"),
                      allow_patterns=["*.json", "*.safetensors"])
    pin_numpy()
    print("\n[OK] Upscale créatif SDXL installé. Disponible dans l'onglet ✨.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "upscale"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()
    elif args.tool == "bg":
        install_bg()
    elif args.tool == "upscale":
        install_upscale()


if __name__ == "__main__":
    main()
