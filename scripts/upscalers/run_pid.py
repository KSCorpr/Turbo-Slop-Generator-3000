#!/usr/bin/env python3
"""Runner NVIDIA PiD : appelle le script d'inférence du dépôt nv-tlabs/PiD.

Appelé par atelier/engine/upscalers.py. PiD (Pixel Diffusion Decoder) fait du
super-resolution / décodage haute résolution. L'API du dépôt évoluant, on
localise un script d'inférence et on tente un jeu d'arguments courant ; en cas
de différence, adaptez CANDIDATES / les arguments ci-dessous.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    "inference.py",
    "scripts/inference.py",
    "demo.py",
    "scripts/demo.py",
    "sample.py",
]


def find_script(repo: Path) -> Path:
    for rel in CANDIDATES:
        if (repo / rel).is_file():
            return repo / rel
    for name in ("inference", "demo", "sample", "upscale"):
        for p in repo.rglob(f"*{name}*.py"):
            if "test" not in p.name.lower():
                return p
    sys.exit(
        f"Script d'inférence PiD introuvable dans {repo}.\n"
        "Consultez le README du dépôt et ajustez scripts/upscalers/run_pid.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    repo = Path(args.repo_dir).resolve()
    script = find_script(repo)
    ckpt = repo / "ckpts"

    # Jeu d'arguments « probable » pour un script de super-resolution.
    cmd = [sys.executable, str(script),
           "--input", str(Path(args.input).resolve()),
           "--output", str(Path(args.output_dir).resolve()),
           "--scale", str(args.scale)]
    if ckpt.is_dir():
        cmd += ["--ckpt", str(ckpt)]
    print("$", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(repo))
    if code != 0:
        print("\n[NVIDIA PiD] Le script s'est terminé en erreur.\n"
              "L'API du dépôt diffère peut-être : ouvrez le README de "
              f"{repo} et ajustez les arguments dans run_pid.py.", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
