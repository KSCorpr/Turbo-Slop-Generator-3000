#!/usr/bin/env python3
"""Helpers d'installation PyTorch CUDA, partagés par les scripts de setup.

Conçu pour le Python embarqué (Windows) : choisit la build CUDA selon le GPU,
évite les verrous de DLL, et fige NumPy < 2 (compat torch).
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CU121 = "https://download.pytorch.org/whl/cu121"
CU128 = "https://download.pytorch.org/whl/cu128"


def sh(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _gpu_arch() -> str:
    try:
        from atelier import hardware
        gpus = hardware.detect_gpus()
        if gpus:
            return max(gpus, key=lambda g: g.vram_gb).arch
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _torch_install_args():
    """Choisit la build PyTorch/CUDA selon le GPU détecté.

    - Blackwell (RTX 50xx, sm_120) : nécessite CUDA 12.8+ et un torch récent.
    - Turing/Ampere/Ada/Pascal : cu121 + torch 2.4.1 (supporte sm_61→sm_89,
      et satisfait transformers récent qui exige torch >= 2.4).
    """
    if is_macos():
        # Sur Mac, l'accélération passe par MPS (Metal), inclus dans les roues
        # PyTorch standard : pas d'index CUDA, et pas d'épinglage 2.4.1 — cette
        # version-là existe pour couvrir les vieilles cartes NVIDIA, contrainte
        # qui n'a aucun sens ici.
        print("macOS -> standard PyTorch (Metal/MPS acceleration).")
        return ["torch", "torchvision"]
    arch = _gpu_arch()
    if arch == "blackwell":
        print("Blackwell GPU detected -> PyTorch CUDA 12.8 (recent).")
        return ["torch", "torchvision", "--index-url", CU128]
    print(f"GPU {arch} -> PyTorch 2.4.1 CUDA 12.1.")
    return ["torch==2.4.1", "torchvision==0.19.1", "--index-url", CU121]


def _torch_gpu_ok() -> bool:
    """Accélération GPU disponible ET torch >= 2.4 (exigé par transformers).

    « GPU » veut dire CUDA sur PC et MPS sur Mac : c'est le même besoin (ne pas
    tomber sur CPU sans le dire), avec deux back-ends. Testé dans un SOUS-PROCESS
    pour ne pas charger torch ici — ses DLL (tbb/mkl) seraient verrouillées et la
    réinstallation échouerait."""
    probe = ("import torch,sys;"
             "v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]);"
             "ok=torch.backends.mps.is_available() if sys.platform=='darwin' "
             "else torch.cuda.is_available();"
             "sys.exit(0 if ok and v>=(2,4) else 3)")
    try:
        r = subprocess.run([sys.executable, "-c", probe], timeout=240)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# Ancien nom, conservé pour les appels existants.
_torch_cuda_ok = _torch_gpu_ok


def _clean_broken_dists():
    """Supprime les paquets à moitié désinstallés (dossiers « ~* ») laissés par
    une install pip interrompue (ex. « ~bb » après un échec sur tbb)."""
    sp = Path(sys.executable).parent / "Lib" / "site-packages"
    if sp.is_dir():
        for p in sp.glob("~*"):
            shutil.rmtree(p, ignore_errors=True)


def ensure_torch_cuda():
    """Garantit un PyTorch CUDA fonctionnel sans verrouiller de DLL.

    On évite --force-reinstall (qui voudrait remplacer tbb/mkl, souvent
    verrouillés -> « Accès refusé »). On désinstalle seulement torch/torchvision
    puis on réinstalle la build CUDA ; les libs annexes (tbb/mkl/numpy) restent.
    """
    _clean_broken_dists()
    backend = "Metal (MPS)" if is_macos() else "CUDA"
    if _torch_gpu_ok():
        print(f"PyTorch {backend} already working.")
        return
    print(f"Mise en place de PyTorch {backend} (volumineux)…")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                     "torch", "torchvision"])
    sh([sys.executable, "-m", "pip", "install", "--no-cache-dir",
        *_torch_install_args()])
    # Vérification finale : si torch ne voit toujours pas le GPU, on le DIT fort
    # (sinon les outils tourneraient sur CPU = extrêmement lent, sans prévenir).
    if _torch_gpu_ok():
        print(f"✅ PyTorch {backend} working.")
    elif is_macos():
        print("⚠️  WARNING: PyTorch does not see Metal (MPS unavailable). The "
              "tools would run on the CPU (very slow). Check that you are on "
              "an Apple Silicon Mac with macOS 12.3 or newer.")
    else:
        print("⚠️  WARNING: PyTorch does NOT see the GPU (CUDA unavailable). "
              "The tools would run on the CPU (very slow). Check your NVIDIA "
              "drivers (nvidia-smi), then run the installation again.")


def pin_numpy():
    """On fige NumPy < 2 (compatibilité torch/torchvision la plus large)."""
    print("Verrouillage de NumPy < 2…")
    sh([sys.executable, "-m", "pip", "install", "numpy>=1.24,<2"])
