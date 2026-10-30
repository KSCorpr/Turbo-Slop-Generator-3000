#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  sam     -> Segment Anything (facebook/sam-vit-base) : extraction d'objet au clic.
  describe -> Qwen2.5-VL-3B : image → prompt.

Réutilise les helpers torch CUDA de _torch_setup (build adaptée au GPU,
sans verrouiller de DLL). Lançable depuis l'interface ou en ligne :
    python scripts/setup_tools.py depth
"""
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
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
DESCRIBE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
# Épinglage PROPRE à cet outil, plus serré que le pin global : 4.45 suffirait
# aux autres add-ons mais ne connaît pas ce modèle, et l'échec serait un
# « KeyError: qwen2_5_vl » incompréhensible au premier clic.
DESCRIBE_TRANSFORMERS_PIN = "transformers>=4.49,<4.50"
# Compatibility constraint for older shared Python installations: recent
# diffusers imports can break Transformers with the Pascal-compatible Torch.
# Isolated restoration and 3D runtimes manage their own dependencies.
DIFFUSERS_PIN = "diffusers==0.33.1"

# NumPy < 2 : impose par notre socle torch 2.4.1 / torchvision 0.19.
# Le passer DANS la commande pip (au lieu de le re-figer apres coup avec
# pin_numpy) change tout : le resolveur choisit alors lui-meme une version
# d'opencv-python compatible avec numpy 1.x, au lieu qu'on installe la derniere
# puis qu'on casse sa dependance en redescendant numpy. On ne devine plus la
# borne d'opencv — pip la trouve.
NUMPY_PIN = "numpy>=1.24,<2"
SPANDREL_PIN = "spandrel>=0.4,<0.5"

# transformers : borne BASSE pour les quatre add-ons qui l'utilisent
# (profondeur, detourage, SAM, description), borne HAUTE dictee par torch.
#
# « <5 » ne suffisait pas : les 4.5x recents importent
# « from torch.distributed.tensor import DTensor », or ce module PUBLIC n'existe
# qu'a partir de torch 2.5 (en 2.4 c'est torch.distributed._tensor). Sur notre
# torch 2.4.1 ca donne un ImportError en cascade des que diffusers touche a
# transformers. On reste donc dans la generation contemporaine de torch 2.4 /
# diffusers 0.33.
# ⚠️ Cette borne est liee a _torch_setup : si torch passe un jour en >= 2.5
# (carte Blackwell, ou abandon de Pascal), elle peut etre relevee.
TRANSFORMERS_PIN = "transformers>=4.45,<4.50"

# Paquets dont on IMPOSE la version, quoi qu'en disent les requirements amont.
# Tous les add-ons partagent un seul Python : ce qui n'est pas borne ici finit
# par etre decide par le dernier « pip install » lance.
_PINS = {"diffusers": DIFFUSERS_PIN, "numpy": NUMPY_PIN,
         "transformers": TRANSFORMERS_PIN}


def install_depth():
    model_dir = settings.ROOT / "tools_repo" / "depth" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "pillow"])
    print(f"\nDownloading the depth model ({DEPTH_REPO})…")
    from atelier.transfers import model_snapshot_download
    model_snapshot_download(repo_id=DEPTH_REPO, local_dir=str(model_dir))
    pin_numpy()  # transformers peut réintroduire NumPy 2 -> on re-fige
    print("\n[OK] Depth Anything V2 installed (depth + normals).")


def install_bg():
    model_dir = settings.ROOT / "tools_repo" / "bg" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "scikit-image", "pillow"])
    print(f"\nDownloading the background-removal model ({BG_REPO})…")
    from atelier.transfers import model_snapshot_download
    model_snapshot_download(repo_id=BG_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] RMBG-1.4 installed. Available in the Toolkit tab.")


def install_sam():
    model_dir = settings.ROOT / "tools_repo" / "sam" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "pillow"])
    print(f"\nDownloading Segment Anything ({SAM_REPO})…")
    from atelier.transfers import model_snapshot_download
    model_snapshot_download(repo_id=SAM_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] Segment Anything installed. Available in the Toolkit tab.")


def install_describe():
    model_dir = settings.ROOT / "tools_repo" / "describe" / "model"
    ensure_torch_cuda()
    print("Installation de transformers + accelerate…")
    sh([sys.executable, "-m", "pip", "install",
        DESCRIBE_TRANSFORMERS_PIN, NUMPY_PIN, "accelerate", "safetensors",
        "pillow"])
    print(f"\nDownloading the image → prompt model ({DESCRIBE_REPO}, "
          f"~7.5 GB)…")
    from atelier.transfers import snapshot_download
    snapshot_download(repo_id=DESCRIBE_REPO, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "tokenizer*", "vocab*", "merges*",
                                      "preprocessor*", "chat_template*"])
    pin_numpy()
    print("\n[OK] Image → prompt installed. Tools tab → “📝 Image → prompt”.")


def install_spandrel():
    """Install the modern upscaler runtime without downloading face models."""
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=120)
    if torch_probe.returncode != 0:
        ensure_torch_cuda()
    # Keep the working GPU build: torchvision/timm resolution must not replace
    # Torch with a newer build that drops support for the user's card.
    with tempfile.TemporaryDirectory(prefix="spandrel setup ") as temporary:
        constraints = Path(temporary) / "constraints.txt"
        pins = []
        for package in ("torch", "torchvision"):
            try:
                pins.append(f"{package}=={importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                pass
        constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
        print("Installing the modern upscaler runtime (Spandrel)…", flush=True)
        sh([sys.executable, "-m", "pip", "install", "-c", str(constraints),
            SPANDREL_PIN, NUMPY_PIN, "pillow"])
    sh([sys.executable, "-c",
        "from spandrel import ImageModelDescriptor, ModelLoader; "
        "import torch,numpy as np; torch.from_numpy(np.zeros((1,),dtype=np.float32)); "
        "print('Spandrel ready; Torch '+torch.__version__)"])


def ensure_spandrel():
    """Probe in a child before installation so Torch DLLs stay unlocked."""
    probe = subprocess.run(
        [sys.executable, "-c", "from spandrel import ImageModelDescriptor, ModelLoader; "
         "import torch,numpy as np; torch.from_numpy(np.zeros((1,),dtype=np.float32))"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", timeout=120)
    if probe.returncode == 0:
        return
    print("Modern upscaler dependencies missing or unavailable; preparing Spandrel…", flush=True)
    print(probe.stdout.strip(), flush=True)
    install_spandrel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "sam", "describe", "spandrel"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()
    elif args.tool == "bg":
        install_bg()
    elif args.tool == "sam":
        install_sam()
    elif args.tool == "describe":
        install_describe()
    elif args.tool == "spandrel":
        install_spandrel()


if __name__ == "__main__":
    main()
