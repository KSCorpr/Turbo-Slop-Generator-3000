#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  sam     -> Segment Anything (facebook/sam-vit-base) : extraction d'objet au clic.
  enhance -> Qwen2.5-3B-Instruct : améliore un prompt brut (LLM).

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
# Segment Anything (base, ~375 Mo) via transformers, depuis HF.
SAM_REPO = "facebook/sam-vit-base"
# Améliorateur de prompt : petit LLM instruct (~6 Go fp16), tourne en sous-process.
ENHANCE_REPO = "Qwen/Qwen2.5-3B-Instruct"


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


def install_sam():
    model_dir = settings.ROOT / "tools_repo" / "sam" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5", "pillow"])
    print(f"\nTéléchargement de Segment Anything ({SAM_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=SAM_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] Segment Anything installé. Disponible dans l'onglet Toolkit.")


def install_enhance():
    model_dir = settings.ROOT / "tools_repo" / "enhance" / "model"
    ensure_torch_cuda()
    print("Installation de transformers + accelerate…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5",
        "accelerate", "safetensors"])
    print(f"\nTéléchargement de l'améliorateur de prompt ({ENHANCE_REPO}, ~6 Go)…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=ENHANCE_REPO, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "tokenizer*", "vocab*", "merges*"])
    pin_numpy()
    print("\n[OK] Améliorateur de prompt installé. Bouton « ✨ Améliorer ».")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "sam", "enhance"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()
    elif args.tool == "bg":
        install_bg()
    elif args.tool == "sam":
        install_sam()
    elif args.tool == "enhance":
        install_enhance()


if __name__ == "__main__":
    main()
