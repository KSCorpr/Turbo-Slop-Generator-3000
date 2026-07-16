#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  sam     -> Segment Anything (facebook/sam-vit-base) : extraction d'objet au clic.
  enhance -> Qwen2.5-3B-Instruct : améliore un prompt brut (LLM).
  upscale -> SDXL base + VAE fp16-fix : upscale créatif tuilé (Ultimate SD Upscale).

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
# Upscale créatif tuilé : SDXL base (1 fichier) + VAE fp16-fix + ControlNet Tile
# (optionnel, verrouille la structure pour pousser la créativité sans dériver).
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0.safetensors"
VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"
CN_TILE_REPO = "xinsir/controlnet-tile-sdxl-1.0"


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


def _hf_fetch(fn, desc: str, manual_url: str, dest) -> None:
    """Téléchargement HF avec 3 tentatives + diagnostic réseau actionnable.

    Sur un poste d'entreprise, l'erreur huggingface_hub « cannot find the
    requested files / check your connection » cache presque toujours un proxy
    obligatoire, une inspection SSL ou un huggingface.co filtré — on l'explique
    au lieu de laisser le traceback brut."""
    import time
    for attempt in range(3):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  [!] échec ({type(exc).__name__}) — nouvel essai "
                      f"dans {wait} s…")
                time.sleep(wait)
            else:
                print(f"\n[X] Téléchargement impossible : {desc}")
                print("    Causes fréquentes sur un réseau d'entreprise :")
                print("    • proxy obligatoire → définissez HTTPS_PROXY="
                      "http://proxy:port avant de lancer run.bat ;")
                print("    • inspection SSL → REQUESTS_CA_BUNDLE="
                      "chemin\\vers\\ca-entreprise.pem ;")
                print("    • huggingface.co filtré → téléchargez à la main :")
                print(f"      {manual_url}")
                print(f"      et placez le fichier dans : {dest}")
                raise


def install_upscale():
    base = settings.ROOT / "tools_repo" / "upscale"
    ensure_torch_cuda()
    print("Installation de diffusers + accelerate…")
    sh([sys.executable, "-m", "pip", "install", "diffusers>=0.30,<0.32",
        "transformers>=4.45,<5", "accelerate", "safetensors", "omegaconf", "pillow"])
    from huggingface_hub import hf_hub_download, snapshot_download
    print(f"\nTéléchargement du checkpoint SDXL ({SDXL_REPO}/{SDXL_FILE}, ~6,6 Go)…")
    _hf_fetch(lambda: hf_hub_download(repo_id=SDXL_REPO, filename=SDXL_FILE,
                                      local_dir=str(base)),
              f"checkpoint SDXL ({SDXL_FILE})",
              f"https://huggingface.co/{SDXL_REPO}/resolve/main/{SDXL_FILE}",
              base)
    print(f"\nTéléchargement de la VAE fp16-fix ({VAE_FIX_REPO})…")
    _hf_fetch(lambda: snapshot_download(repo_id=VAE_FIX_REPO,
                                        local_dir=str(base / "vae"),
                                        allow_patterns=["*.json", "*.safetensors"]),
              "VAE fp16-fix",
              f"https://huggingface.co/{VAE_FIX_REPO}/tree/main",
              base / "vae")
    print(f"\nTéléchargement du ControlNet Tile ({CN_TILE_REPO}, ~2,5 Go)…")
    _hf_fetch(lambda: snapshot_download(repo_id=CN_TILE_REPO,
                                        local_dir=str(base / "controlnet"),
                                        allow_patterns=["*.json", "*.safetensors"]),
              "ControlNet Tile",
              f"https://huggingface.co/{CN_TILE_REPO}/tree/main",
              base / "controlnet")
    # Dossier où déposer des checkpoints SDXL perso (sélectionnables dans l'UI).
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    pin_numpy()
    print("\n[OK] Upscale créatif SDXL + ControlNet Tile installé "
          "(onglet Toolkit → Upscale créatif).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "sam", "enhance", "upscale"])
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
    elif args.tool == "upscale":
        install_upscale()


if __name__ == "__main__":
    main()
