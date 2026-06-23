#!/usr/bin/env python3
"""Helpers d'installation PyTorch CUDA, partagés par les scripts de setup.

Conçu pour le Python embarqué (Windows) : choisit la build CUDA selon le GPU,
évite les verrous de DLL, et fige NumPy < 2 (compat torch).
"""
from __future__ import annotations

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
    arch = _gpu_arch()
    if arch == "blackwell":
        print("GPU Blackwell détecté -> PyTorch CUDA 12.8 (récent).")
        return ["torch", "torchvision", "--index-url", CU128]
    print(f"GPU {arch} -> PyTorch 2.4.1 CUDA 12.1.")
    return ["torch==2.4.1", "torchvision==0.19.1", "--index-url", CU121]


def _torch_cuda_ok() -> bool:
    """CUDA dispo ET torch >= 2.4 (requis par transformers récent). Testé dans un
    SOUS-PROCESS (sans charger torch ici, sinon ses DLL — tbb/mkl — seraient
    verrouillées et la réinstall échouerait)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch,sys;"
             "v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]);"
             "sys.exit(0 if torch.cuda.is_available() and v>=(2,4) else 3)"],
            timeout=240)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


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
    if _torch_cuda_ok():
        print("PyTorch CUDA déjà opérationnel.")
        return
    print("Mise en place de PyTorch CUDA (build GPU, volumineux)…")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                     "torch", "torchvision"])
    sh([sys.executable, "-m", "pip", "install", "--no-cache-dir",
        *_torch_install_args()])


def pin_numpy():
    """On fige NumPy < 2 (compatibilité torch/torchvision la plus large)."""
    print("Verrouillage de NumPy < 2…")
    sh([sys.executable, "-m", "pip", "install", "numpy>=1.24,<2"])
