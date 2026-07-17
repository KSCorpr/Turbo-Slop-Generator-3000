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
CUSTOM_DIR = MODELS_DIR / "custom"   # modèles téléchargés manuellement ailleurs
LORA_DIR = ROOT / "loras"
BIN_DIR = ROOT / "bin"
OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
USERDATA_DIR = ROOT / "userdata"

CONFIG_DIR = ROOT / "config"
PREFS_FILE = USERDATA_DIR / "preferences.json"

# Préférences par défaut (surchargées par l'onglet Réglages, persistées en JSON).
DEFAULT_PREFS: dict[str, Any] = {
    "lang": "fr",               # langue de l'interface : "fr" | "en"
    "theme": "light",           # thème de l'interface : "light" | "dark"
    "gpu_index": None,          # None = auto (meilleure carte détectée, mono-GPU)
    # GPU pour l'améliorateur de prompt. None = même GPU que la génération.
    # (Conservé pour compat avec le Toolkit ; mono-GPU sur ce setup.)
    "text_gpu_index": None,
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
    "civitai_token": "",        # jeton Civitai (optionnel, pour les LoRA protégés)
    # Accélération par cache (sd.cpp docs/caching.md). "" = désactivé.
    # Modes DiT (Flux/Krea) : easycache | dbcache | taylorseer | cache-dit | spectrum
    "cache_mode": "",
    "cache_option": "",         # ex. "threshold=0.2" (easycache) — vide = défauts
}


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CUSTOM_DIR, LORA_DIR, BIN_DIR, OUTPUT_DIR, TMP_DIR,
              USERDATA_DIR):
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
def sd_cli_names() -> list[str]:
    """Noms possibles du binaire selon la version de stable-diffusion.cpp."""
    if platform.system() == "Windows":
        return ["sd-cli.exe", "sd.exe"]
    return ["sd-cli", "sd"]


def find_sd_cli() -> Path | None:
    for name in sd_cli_names():
        for candidate in BIN_DIR.rglob(name):
            if candidate.is_file():
                return candidate
    for name in sd_cli_names():
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def model_repo_dir(repo: str) -> Path:
    """Emplacement local d'un dépôt HF : models/<owner>__<name>/."""
    return MODELS_DIR / repo.replace("/", "__")


def configure_hf_env() -> None:
    """Configure l'environnement Hugging Face pour des téléchargements fiables.

    hf_transfer est désactivé : sous Windows il provoque des verrous de fichier
    (os error 32). Le téléchargement HTTP standard est plus lent mais robuste.
    """
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    prefs = load_prefs()
    os.environ.setdefault("HF_ENDPOINT",
                          prefs.get("hf_endpoint", "https://huggingface.co"))
