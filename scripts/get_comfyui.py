#!/usr/bin/env python3
"""Installe un backend ComfyUI (PyTorch) piloté par API, à la demande.

EXPÉRIMENTAL. Contrairement à sd.cpp (binaire natif, GGUF, léger), ComfyUI est
un backend PyTorch complet : plus lourd (~4–6 Go), mais il fait tourner les
modèles GGUF Flux.2/Krea EXACTS de ce projet avec un support LoRA PyTorch natif.

Étapes :
  1. clone ComfyUI dans ./comfyui
  2. clone le custom node ComfyUI-GGUF (chargement des .gguf)
  3. PyTorch CUDA (réutilise le helper partagé du Toolkit) + requirements
  4. génère comfyui/extra_model_paths.yaml pour que ComfyUI voie models/ et loras/

Usage :
    python scripts/get_comfyui.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMFY_DIR = ROOT / "comfyui"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
GGUF_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _clone(repo: str, dest: Path) -> None:
    if (dest / ".git").is_dir():
        print(f"{dest.name} déjà cloné — mise à jour (git pull)…")
        try:
            sh(["git", "-C", str(dest), "pull", "--ff-only"])
        except subprocess.CalledProcessError:
            print(f"   (git pull a échoué pour {dest.name}, on garde l'existant)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "--depth", "1", repo, str(dest)])


def write_extra_model_paths() -> None:
    """Indique à ComfyUI où trouver les modèles/LoRA DÉJÀ téléchargés du projet.

    On pointe toutes les catégories sur models/ (ComfyUI scanne récursivement) et
    les LoRA sur loras/. Les fichiers sont ainsi référencés par leur chemin
    relatif (ex. « leejet__FLUX.2-klein-9B-GGUF/flux-2-klein-9b-Q4_0.gguf »)."""
    models = (ROOT / "models").as_posix()
    loras = (ROOT / "loras").as_posix()
    base = ROOT.as_posix()
    content = f"""# Généré par get_comfyui.py — ne pas éditer à la main.
# Fait pointer ComfyUI sur les modèles/LoRA de Turbo Slop Generator.
turbo_slop:
    base_path: {base}
    is_default: false
    checkpoints: models
    unet: models
    diffusion_models: models
    clip: models
    text_encoders: models
    clip_vision: models
    vae: models
    loras: loras
    upscale_models: models
"""
    (COMFY_DIR / "extra_model_paths.yaml").write_text(content, encoding="utf-8")
    print("extra_model_paths.yaml écrit (models/ + loras/).")


def main() -> None:
    print("=== Installation du backend ComfyUI (expérimental) ===", flush=True)
    _clone(COMFY_REPO, COMFY_DIR)
    _clone(GGUF_REPO, GGUF_DIR)

    # PyTorch CUDA : on réutilise exactement le même helper que le Toolkit, pour
    # une build GPU cohérente et sans verrou de DLL sous Windows.
    try:
        from scripts._torch_setup import ensure_torch_cuda, pin_numpy
        ensure_torch_cuda()
        pin_numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Mise en place de PyTorch CUDA impossible ({exc}). "
              "ComfyUI risque de tourner sur CPU.")

    # Dépendances ComfyUI + node GGUF (torch est déjà satisfait -> non réinstallé).
    req = COMFY_DIR / "requirements.txt"
    if req.is_file():
        sh([sys.executable, "-m", "pip", "install", "-r", str(req)])
    gguf_req = GGUF_DIR / "requirements.txt"
    if gguf_req.is_file():
        sh([sys.executable, "-m", "pip", "install", "-r", str(gguf_req)])
    else:
        sh([sys.executable, "-m", "pip", "install", "gguf"])

    write_extra_model_paths()
    print("\n✅ ComfyUI installé dans ./comfyui")
    print("   Activez le moteur « ComfyUI » dans l'onglet Réglages, puis générez.")


if __name__ == "__main__":
    main()
