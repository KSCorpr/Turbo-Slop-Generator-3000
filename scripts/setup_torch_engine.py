#!/usr/bin/env python3
"""Installe la pile PyTorch de la branche Test7000, dans le Python embarqué.

Ce script fait ce que `setup_tools.py` fait pour les add-ons du Toolkit, mais
pour le MOTEUR DE GÉNÉRATION lui-même — et il assume une conséquence que
l'autre n'avait pas : il déplace la pile commune.

Pourquoi il faut la déplacer. Le Toolkit tourne sur `torch 2.4.1` et
`diffusers 0.33.1`, et ces deux épingles se tiennent l'une l'autre : torch
2.4.1 a été choisi pour couvrir Pascal → Ada, et son `infer_schema` ne sait
pas lire les annotations « X | None » qu'utilise diffusers ≥ 0.35. Or les trois
modèles du catalogue réclament `diffusers ≥ 0.36` — c'est écrit dans le
`model_index.json` de chacun. Il n'y a donc pas de version qui satisfasse les
deux mondes : cette branche monte la pile entière, et la contrainte qui
justifiait l'ancienne épingle disparaît d'elle-même en montant torch.

Ce qui survit et ce qui ne survit pas :

* **Pascal survit.** Les roues `cu126` embarquent encore `5.0;6.0;7.0;…`
  (vérifié dans `.ci/manywheel/build_cuda.sh` jusqu'à la 2.12) et un cubin
  `sm_60` se charge sur une carte `sm_61` — la GTX 1080 Ti garde donc son rôle.
* **Blackwell change d'index.** Les roues `cu128` commencent à `7.0` : les
  RTX 50xx en ont besoin, et les cartes antérieures à Volta n'y sont plus.
  L'application choisit selon la carte détectée, comme le fait déjà
  `_torch_setup`.
* **Les add-ons du Toolkit changent de socle.** C'est le seul vrai risque de
  cette branche, et il est écrit ici plutôt que découvert au premier clic.

Lançable depuis l'interface ou en ligne :
    python scripts/setup_torch_engine.py
    python scripts/setup_torch_engine.py --check
"""
from __future__ import annotations

import argparse
import subprocess
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

CU126 = "https://download.pytorch.org/whl/cu126"
CU128 = "https://download.pytorch.org/whl/cu128"

#  Les couples torch / torchvision sont ceux publiés ensemble ; les mélanger
#  donne un ImportError sur une extension C, pas un message lisible. Les deux
#  versions ci-dessous ont été vérifiées présentes en `cp311-win_amd64` sur
#  leur index respectif.
TORCH_CU126 = ("torch==2.12.1", "torchvision==0.27.1")
#  L'index cu128 s'arrête plus tôt : c'est la contrainte, pas un choix.
TORCH_CU128 = ("torch==2.11.0", "torchvision==0.26.0")

#  diffusers : 0.36 est le PLANCHER (les trois `model_index.json` le disent) ;
#  on installe la 0.40.0, dont on a vérifié qu'elle publie bien les onze
#  classes de pipeline que le catalogue nomme.
DIFFUSERS = "diffusers==0.40.0"

#  Le reste du socle. Aucune de ces lignes n'est décorative :
#   · gguf          -> LA pièce qui fait tout l'intérêt de cette branche :
#                      sans elle, diffusers refuse `GGUFQuantizationConfig` et
#                      il faudrait retélécharger chaque modèle en dépôt
#                      complet. Le plancher 0.10.0 est celui que diffusers
#                      exige (`is_gguf_version("<", "0.10.0")`) ;
#   · transformers  -> les encodeurs de texte (Qwen3), y compris depuis un
#                      GGUF : « qwen3 » figure dans sa table de conversion ;
#   · accelerate    -> `enable_model_cpu_offload` et la décharge séquentielle,
#                      et le quantiseur GGUF l'exige explicitement (>= 0.26) ;
#   · peft          -> les LoRA (`load_lora_weights` en dépend) ;
#   · bitsandbytes  -> les crans int8 et NF4 du chemin dépôt complet, qui ne
#                      sert plus qu'à Krea 2 ; SEUL des deux candidats à
#                      publier une roue `win_amd64` (torchao n'en publie pas) ;
#   · scipy         -> le scheduler « beta » lève un ImportError sans elle,
#                      et c'est une entrée du menu, donc un clic possible ;
#   · sentencepiece -> tokeniseurs de la famille Qwen ;
#   · safetensors   -> format des poids non quantifiés (les VAE).
#   · kernels       -> le noyau CUDA de déquantification GGUF. Installé mais
#                      INACTIF : diffusers ne s'en sert que si
#                      `DIFFUSERS_GGUF_CUDA_KERNELS` est posée, ce que fait la
#                      case des réglages. Sans lui, la case ne pourrait rien
#                      activer ; avec lui et sans la case, rien ne change.
STACK = ("gguf>=0.10.0", "kernels>=0.9", "transformers>=4.51",
         "accelerate>=1.0", "peft>=0.14", "bitsandbytes>=0.47", "scipy>=1.11",
         "sentencepiece", "safetensors>=0.4", "protobuf")


def sh(cmd: list[str]) -> None:
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


def torch_args() -> tuple[list[str], str]:
    """Les paquets torch à installer, et l'index d'où les prendre."""
    arch = _gpu_arch()
    if arch == "blackwell":
        return list(TORCH_CU128), CU128
    return list(TORCH_CU126), CU126


def installed() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("torch", "diffusers", "gguf", "transformers", "accelerate",
                "peft", "bitsandbytes", "scipy", "safetensors"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = ""
    return out


def report() -> str:
    lines = ["Installed:"]
    for pkg, ver in installed().items():
        lines.append(f"  {pkg:14s} {ver or '— absent'}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="show what is installed and what would change")
    args = ap.parse_args()

    packages, index = torch_args()
    print("=" * 62)
    print("  PyTorch engine (branch Test7000)")
    print("=" * 62)
    print(f"GPU family detected: {_gpu_arch()}")
    print(f"torch index       : {index}")
    print(f"torch packages    : {', '.join(packages)}")
    print(f"diffusers         : {DIFFUSERS}")
    print()
    print(report())

    if args.check:
        print("\n--check: nothing was installed.")
        return 0

    print("\nThis replaces the Toolkit's torch 2.4.1 / diffusers 0.33.1 with a "
          "\nnewer stack. That is deliberate — the models in the catalog need "
          "diffusers >= 0.36,\nand no single version satisfies both.")
    print("It does NOT re-download any model: diffusers reads the same GGUF "
          "files\nstable-diffusion.cpp already uses.\n")
    py = [sys.executable, "-m", "pip", "install", "--upgrade"]
    sh([*py, *packages, "--index-url", index])
    sh([*py, DIFFUSERS, *STACK])

    print()
    print(report())
    print("\n[OK] PyTorch engine installed. Start the app again with run.bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
