#!/usr/bin/env python3
"""Runner SeedVR2 : appelle le script d'inférence officiel du dépôt cloné.

Appelé par atelier/engine/upscalers.py. SeedVR2-3B traite un dossier d'images ;
on y dépose l'image d'entrée, on récupère le résultat dans --output-dir.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    "projects/inference_seedvr2_3b.py",
    "projects/inference_seedvr2.py",
    "inference_seedvr2_3b.py",
]


def find_script(repo: Path) -> Path:
    for rel in CANDIDATES:
        if (repo / rel).is_file():
            return repo / rel
    for p in repo.rglob("inference_seedvr2*.py"):
        return p
    sys.exit(f"Script d'inférence SeedVR2 introuvable dans {repo}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    repo = Path(args.repo_dir).resolve()
    script = find_script(repo)

    in_dir = Path(args.output_dir).resolve().parent / "seedvr2_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.input, in_dir / Path(args.input).name)

    # SeedVR2-3B : entrée = dossier, sp_size=1 = mono-GPU.
    cmd = [sys.executable, str(script),
           "--video_path", str(in_dir),
           "--output_dir", str(Path(args.output_dir).resolve()),
           "--seed", "-1", "--sp_size", "1"]
    print("$", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(repo)))


if __name__ == "__main__":
    main()
