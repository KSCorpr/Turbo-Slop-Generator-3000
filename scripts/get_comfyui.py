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

# Console Windows = cp1252 : forcer UTF-8 évite un UnicodeEncodeError sur les
# accents/emojis (sinon le vrai message d'erreur est masqué par ce crash).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

COMFY_DIR = ROOT / "comfyui"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
GGUF_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
# Nœud INT8 (Krea 2 ConvRot : OTUNetLoaderW8A8). Nécessite Triton — fragile sous
# Windows, installé en best-effort (le niveau INT8 « simple » n'en a pas besoin).
INT8_REPO = "https://github.com/BobJohnson24/ComfyUI-INT8-Fast"
INT8_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-INT8-Fast"


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


def _fix_torch_stack() -> None:
    """Réaligne torch/torchvision/torchaudio APRÈS les requirements ComfyUI.

    Le requirements.txt de ComfyUI liste `torchaudio` : pip l'installe depuis
    PyPI en DERNIÈRE version (build CPU), qui ne correspond pas au torch CUDA
    épinglé par ensure_torch_cuda. Au lancement, la DLL de torchaudio ne trouve
    pas les symboles de torch → « WinError 127 : procédure introuvable » et
    ComfyUI meurt à l'import. On répare : torch CUDA d'abord (au cas où les
    requirements l'auraient remplacé), puis torchaudio à la MÊME version depuis
    le MÊME index CUDA."""
    from _torch_setup import _torch_install_args, ensure_torch_cuda, pin_numpy
    print("\n• Réalignement torch / torchaudio (compat ComfyUI)…")
    ensure_torch_cuda()   # no-op si torch CUDA est déjà bon
    args = _torch_install_args()
    pinned = next((a.split("==", 1)[1] for a in args if a.startswith("torch==")),
                  None)
    audio = f"torchaudio=={pinned}" if pinned else "torchaudio"
    index = args[args.index("--index-url") + 1] if "--index-url" in args else None
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                     "torchaudio"])
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", audio]
    if index:
        cmd += ["--index-url", index]
    try:
        sh(cmd)
        print(f"  [OK] {audio} aligné sur torch.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] torchaudio non réaligné ({exc}) — ComfyUI risque de "
              "replanter à l'import.")
    pin_numpy()   # les requirements peuvent réintroduire NumPy 2 → on re-fige


def _install_int8_node() -> None:
    """Nœud ComfyUI-INT8-Fast (Krea 2 ConvRot) + Triton. Best-effort : un échec
    ne bloque PAS l'installation — le niveau INT8 « simple » (loader standard)
    fonctionne sans. Triton sous Windows = paquet `triton-windows` (fragile)."""
    import platform
    print("\n• Nœud INT8 ConvRot (ComfyUI-INT8-Fast) + Triton [optionnel]…")
    try:
        _clone(INT8_REPO, INT8_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] clone du nœud INT8 impossible ({exc}). "
              "Le niveau INT8 simple reste disponible.")
        return
    int8_req = INT8_DIR / "requirements.txt"
    if int8_req.is_file():
        try:
            sh([sys.executable, "-m", "pip", "install", "-r", str(int8_req)])
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] requirements du nœud INT8 : {exc}")
    triton = "triton-windows" if platform.system() == "Windows" else "triton"
    try:
        sh([sys.executable, "-m", "pip", "install", "-U", triton])
        print(f"  [OK] {triton} installé (ConvRot activable).")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] {triton} non installé ({exc}). Le ConvRot ne marchera pas, "
              "mais l'INT8 simple oui. C'est le point fragile sous Windows.")


def _verify() -> None:
    """Go/no-go clair : ComfyUI s'importe-t-il vraiment (comfy + torch) ? Utilise
    la MÊME injection de sys.path que le lancement (indispensable en Python
    embarqué), donc reflète le comportement réel au démarrage du moteur."""
    print("\n• Vérification (import comfy + torch + torchaudio)…")
    boot = ("import sys; sys.path.insert(0, r'{d}'); "
            "import comfy.options; import torch; import torchaudio; "
            "print('  [OK] comfy OK, torch', torch.__version__, "
            "'/ torchaudio', torchaudio.__version__, "
            "('CUDA' if torch.cuda.is_available() else 'CPU (pas de GPU !)'))"
            ).format(d=str(COMFY_DIR))
    try:
        subprocess.check_call([sys.executable, "-c", boot], cwd=str(COMFY_DIR))
    except subprocess.CalledProcessError:
        print("  [!] ComfyUI ne s'importe PAS encore (voir l'erreur au-dessus). "
              "Causes fréquentes : dépendances non installées (relancez), ou "
              "clone incomplet (supprimez le dossier comfyui/ puis relancez).")


def main() -> None:
    print("=== Installation du backend ComfyUI (expérimental) ===", flush=True)
    _clone(COMFY_REPO, COMFY_DIR)
    _clone(GGUF_REPO, GGUF_DIR)

    # PyTorch CUDA : on réutilise exactement le même helper que le Toolkit, pour
    # une build GPU cohérente et sans verrou de DLL sous Windows.
    try:
        from _torch_setup import ensure_torch_cuda, pin_numpy
        ensure_torch_cuda()
        pin_numpy()
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"  [!] Mise en place de PyTorch CUDA impossible ({exc}). "
              "ComfyUI risque de tourner sur CPU.")
        traceback.print_exc()

    # Dépendances ComfyUI + node GGUF (torch est déjà satisfait -> non réinstallé).
    req = COMFY_DIR / "requirements.txt"
    if req.is_file():
        sh([sys.executable, "-m", "pip", "install", "-r", str(req)])
    gguf_req = GGUF_DIR / "requirements.txt"
    if gguf_req.is_file():
        sh([sys.executable, "-m", "pip", "install", "-r", str(gguf_req)])
    else:
        sh([sys.executable, "-m", "pip", "install", "gguf"])

    _install_int8_node()

    # Toujours en DERNIER : les requirements (ComfyUI, GGUF, INT8) peuvent avoir
    # désaccordé torch/torchaudio — cette étape a le dernier mot sur la pile torch.
    _fix_torch_stack()

    write_extra_model_paths()
    _verify()
    print("\n[OK] Installation terminée (voir la vérification ci-dessus).")
    print("   Activez le moteur « ComfyUI » dans l'onglet Réglages, puis générez.")


if __name__ == "__main__":
    main()
