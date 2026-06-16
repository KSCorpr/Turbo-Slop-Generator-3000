"""Chemins du projet, préférences utilisateur persistées et localisation de sd-cli."""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Dossiers (créés au besoin). Tout est local au projet -> portable.
MODELS_DIR = ROOT / "models"
LORA_DIR = ROOT / "loras"
BIN_DIR = ROOT / "bin"
OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
USERDATA_DIR = ROOT / "userdata"
UPSCALERS_REPO_DIR = ROOT / "upscalers_repo"

CONFIG_DIR = ROOT / "config"
PREFS_FILE = USERDATA_DIR / "preferences.json"

# Préférences par défaut (surchargées par l'onglet Réglages, persistées en JSON).
DEFAULT_PREFS: dict[str, Any] = {
    "gpu_index": None,          # None = auto (meilleure carte détectée)
    "auto_optimize": True,      # déduire les flags du matériel
    "quant": None,              # None = recommandé selon VRAM
    "enc_quant": None,          # None = recommandé selon RAM
    # Surcharges manuelles (utilisées seulement si auto_optimize = False)
    "flags": {
        "diffusion_fa": True,
        "offload_to_cpu": True,
        "vae_tiling": True,
        "clip_on_cpu": False,
        "vae_on_cpu": False,
    },
    "hf_endpoint": "https://huggingface.co",
}


def ensure_dirs() -> None:
    for d in (MODELS_DIR, LORA_DIR, BIN_DIR, OUTPUT_DIR, TMP_DIR, USERDATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_prefs() -> dict[str, Any]:
    ensure_dirs()
    prefs = json.loads(json.dumps(DEFAULT_PREFS))  # copie profonde
    if PREFS_FILE.is_file():
        try:
            saved = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            prefs.update({k: v for k, v in saved.items() if k != "flags"})
            if isinstance(saved.get("flags"), dict):
                prefs["flags"].update(saved["flags"])
        except (json.JSONDecodeError, OSError):
            pass
    return prefs


def save_prefs(prefs: dict[str, Any]) -> None:
    ensure_dirs()
    PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


# --- localisation du binaire stable-diffusion.cpp --------------------------
def sd_cli_name() -> str:
    return "sd-cli.exe" if platform.system() == "Windows" else "sd-cli"


def find_sd_cli() -> Path | None:
    name = sd_cli_name()
    for candidate in BIN_DIR.rglob(name):
        if candidate.is_file():
            return candidate
    found = shutil.which(name) or shutil.which("sd")
    return Path(found) if found else None


def model_repo_dir(repo: str) -> Path:
    """Emplacement local d'un dépôt HF : models/<owner>__<name>/."""
    return MODELS_DIR / repo.replace("/", "__")
