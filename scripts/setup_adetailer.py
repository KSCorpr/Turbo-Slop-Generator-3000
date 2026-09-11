#!/usr/bin/env python3
"""Installe les détecteurs ADetailer (YOLOv8) pour sd.cpp.

Deux difficultés, et la seconde décide de toute la forme de ce script.

1. **sd.cpp n'accepte pas les .pt d'Ultralytics tels quels.** Il lui faut un
   safetensors aux noms de tenseurs de son implémentation GGML, BatchNorm déjà
   fusionnée dans les convolutions. La conversion est donc obligatoire, et
   aucune version convertie n'est publiée : il faut la faire ici.

2. **Le convertisseur exige `ultralytics`**, parce qu'un `.pt` est un pickle
   qui contient les CLASSES d'Ultralytics — sans le paquet, il ne se dépickle
   pas. Or ultralytics est en AGPL-3.0, tire opencv/pandas/scipy, et veut un
   NumPy que nos add-ons n'ont pas. L'installer à côté des autres casserait
   exactement ce que le ménage d'`update.bat` passe son temps à signaler.

D'où le choix : un environnement **jetable**. On crée un venv isolé, on
convertit, **on le supprime**. Il reste 6 Mo de safetensors et aucune
dépendance permanente — ni dans l'app, ni en AGPL.

Le `.pt` est dépicklé, donc exécuté. On ne lit que `Bingsu/adetailer`, le
dépôt de référence de l'extension ADetailer, et jamais un fichier fourni par
l'utilisateur.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atelier import settings  # noqa: E402

DETECTOR_REPO = "Bingsu/adetailer"
WORK = ROOT / "tools_repo" / "adetailer-build"

# sd.cpp ne gère QUE la détection YOLOv8 : ni la segmentation (-seg), ni
# YOLOv9. Les proposer conduirait à un échec au premier clic, alors qu'ils
# existent bel et bien dans le dépôt d'en face.
DETECTORS = [
    ("face_yolov8n.pt", "face_yolov8n.safetensors", "faces, fast (6 MB)"),
    ("face_yolov8s.pt", "face_yolov8s.safetensors", "faces, more accurate"),
    ("hand_yolov8n.pt", "hand_yolov8n.safetensors", "hands, fast (6 MB)"),
    ("hand_yolov8s.pt", "hand_yolov8s.safetensors", "hands, more accurate"),
]


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32"
                   else "bin/python")


def _converter() -> Path:
    """Le convertisseur officiel, récupéré à la version du moteur installé.

    On ne le recopie pas dans le dépôt : les noms de tenseurs qu'il produit
    doivent correspondre à CE que le binaire attend, et une copie figée
    dériverait en silence à la première mise à jour de moteur.
    """
    import urllib.request
    url = ("https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/"
           "master/scripts/convert_yolov8_to_safetensors.py")
    dest = WORK / "convert_yolov8_to_safetensors.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching the official converter…", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        blob = response.read()
    if b"safetensors" not in blob or b"ultralytics" not in blob:
        raise SystemExit("the downloaded converter does not look right — "
                         "a proxy may have replaced it with an error page.")
    dest.write_bytes(blob)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-venv", action="store_true",
                    help="keep the throwaway environment (for debugging)")
    args = ap.parse_args()

    settings.configure_hf_env()
    out_dir = settings.MODELS_DIR / "adetailer"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [(src, dst, note) for src, dst, note in DETECTORS
            if not (out_dir / dst).is_file()]
    if not todo:
        print("[OK] Every detector is already converted.")
        return 0

    from huggingface_hub import hf_hub_download
    venv = WORK / "venv"
    try:
        try:
            import uv  # noqa: F401
        except ImportError:
            print("Installing the uv environment manager…", flush=True)
            _run([sys.executable, "-m", "pip", "install", "uv>=0.8,<1"])
        py = _venv_python(venv)
        if not py.is_file():
            print("Creating a throwaway environment for the conversion…",
                  flush=True)
            _run([sys.executable, "-m", "uv", "venv", "--python", "3.12",
                  "--seed", str(venv)])
            # CPU torch : la conversion ne calcule rien, elle relit des poids.
            # Tirer une roue CUDA de 2,5 Go pour ça serait absurde.
            _run([sys.executable, "-m", "uv", "pip", "install", "--python",
                  str(py), "--index-url",
                  "https://download.pytorch.org/whl/cpu", "torch"])
            _run([sys.executable, "-m", "uv", "pip", "install", "--python",
                  str(py), "ultralytics", "safetensors"])
        converter = _converter()

        for src, dst, note in todo:
            print(f"\nDownloading {src} ({note})…", flush=True)
            pt = hf_hub_download(repo_id=DETECTOR_REPO, filename=src,
                                 local_dir=str(WORK / "pt"))
            _run([str(py), str(converter), str(pt), str(out_dir / dst)])
            print(f"  [OK] {dst}", flush=True)
    finally:
        if not args.keep_venv and WORK.exists():
            # C'est le point de tout l'exercice : l'AGPL et les 2 Go de
            # dépendances repartent, les 6 Mo utiles restent.
            print("\nRemoving the throwaway environment…", flush=True)
            shutil.rmtree(WORK, ignore_errors=True)

    print(f"\n[OK] Detectors ready in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
