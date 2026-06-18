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
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# Console Windows : forcer l'UTF-8 pour éviter les UnicodeEncodeError (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

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


CU121 = "https://download.pytorch.org/whl/cu121"


def _torch_cuda_ok() -> bool:
    """Teste la dispo CUDA dans un SOUS-PROCESS (sans charger torch ici, sinon
    ses DLL — tbb/mkl — seraient verrouillées et la réinstall échouerait)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
            timeout=240)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_torch_cuda():
    """Garantit un PyTorch CUDA fonctionnel sans verrouiller de DLL.

    On évite --force-reinstall (qui voudrait remplacer tbb/mkl, souvent
    verrouillés -> « Accès refusé »). On désinstalle seulement torch/torchvision
    puis on réinstalle la build CUDA ; les libs annexes (tbb/mkl/numpy) restent.
    """
    if _torch_cuda_ok():
        print("PyTorch CUDA déjà opérationnel.")
        return
    print("Mise en place de PyTorch CUDA 12.1 (build GPU, volumineux)…")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                     "torch", "torchvision"])
    sh([sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.3.0", "torchvision==0.18.0", "--index-url", CU121])


def pin_numpy():
    """torch/torchvision 2.3 sont compilés pour NumPy 1.x -> on fige < 2."""
    print("Verrouillage de NumPy < 2 (compatibilité torch 2.3)…")
    sh([sys.executable, "-m", "pip", "install", "numpy>=1.24,<2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upscaler_id",
                    choices=[u.id for u in registry.load_upscalers()])
    args = ap.parse_args()

    settings.configure_hf_env()
    up = next((u for u in registry.load_upscalers() if u.id == args.upscaler_id), None)
    if up is None:
        sys.exit(f"Upscaler inconnu : {args.upscaler_id}")

    settings.UPSCALERS_REPO_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = settings.UPSCALERS_REPO_DIR / up.id
    ckpt_dir = repo_dir / "ckpts"

    # --- Cas paquet pip (ex. AuraSR) : pas de git, poids auto au 1er run -----
    if up.pip_package:
        ensure_torch_cuda()
        print(f"Installation de {up.pip_package}…")
        sh([sys.executable, "-m", "pip", "install", up.pip_package])
        pin_numpy()  # aura-sr peut retirer numpy<2 -> on re-fige en dernier
        repo_dir.mkdir(parents=True, exist_ok=True)  # marqueur « installé »
        print(f"\n[OK] {up.name} installe. Disponible dans l'onglet Upscale.")
        return

    # 1. Code (ZIP, sans git) -------------------------------------------------
    if not repo_dir.is_dir():
        owner, repo = _owner_repo(up.code_repo)
        _download_zip(owner, repo, repo_dir)
    else:
        print(f"Code déjà présent : {repo_dir}")

    # 2. PyTorch CUDA 12.1 ----------------------------------------------------
    ensure_torch_cuda()

    # 3. Dépendances du dépôt -------------------------------------------------
    req = repo_dir / "requirements.txt"
    if req.is_file():
        try:
            sh([sys.executable, "-m", "pip", "install", "-r", str(req)])
        except subprocess.CalledProcessError:
            print("[ATTENTION] Certaines dependances ont echoue (flash-attn/apex "
                  "parfois penibles sous Windows). L'inference peut tout de meme "
                  "fonctionner.")

    # 4. Poids ----------------------------------------------------------------
    print(f"\nTéléchargement des poids {up.weights_repo}…")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=up.weights_repo, local_dir=str(ckpt_dir))
        print(f"Poids dans {ckpt_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERREUR] Echec du telechargement des poids : {exc}")
        raise SystemExit(1)

    pin_numpy()  # les deps du dépôt peuvent réintroduire NumPy 2
    print(f"\n[OK] {up.name} installe. Disponible dans l'onglet Upscale.")


if __name__ == "__main__":
    main()
