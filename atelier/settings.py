"""Chemins du projet, préférences utilisateur persistées et localisation de sd-cli."""
from __future__ import annotations

import json
import platform
import shutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .fileio import atomic_write_text

ROOT = Path(__file__).resolve().parent.parent

# Dossiers (créés au besoin). Tout est local au projet -> portable.
LORA_DIR = ROOT / "loras"
BIN_DIR = ROOT / "bin"
OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
CACHE_DIR = ROOT / ".cache"
RUNTIME_DIR = ROOT / "runtime"
USERDATA_DIR = ROOT / "userdata"

CONFIG_DIR = ROOT / "config"
PREFS_FILE = USERDATA_DIR / "preferences.json"

DEFAULT_MODELS_DIR = ROOT / "models"


def _read_models_dir() -> Path:
    """Dossier des modèles : `models/` du projet, ou un dossier EXTERNE choisi
    dans les Réglages (ex. un NVMe rapide).

    Lu au CHARGEMENT du module (donc changement = redémarrage requis) : des
    modules capturent ce chemin à l'import. Lecture volontairement minimale —
    pas d'appel à load_prefs()/ensure_dirs() ici, qui dépendent de ce chemin.
    """
    try:
        if PREFS_FILE.is_file():
            data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            raw = (data.get("models_dir") or "").strip()
            if raw:
                p = Path(raw).expanduser()
                if p.is_absolute():
                    return p
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return DEFAULT_MODELS_DIR


MODELS_DIR = _read_models_dir()
CUSTOM_DIR = MODELS_DIR / "custom"   # modèles téléchargés manuellement ailleurs

# Préférences par défaut (surchargées par l'onglet Réglages, persistées en JSON).
DEFAULT_PREFS: dict[str, Any] = {
    "theme": "light",           # thème de l'interface : "light" | "dark"
    # Dossier des modèles. Vide/None = `models/` dans le projet. Un chemin
    # ABSOLU le déplace ailleurs (ex. "D:\\IA\\models" sur un NVMe).
    # Pris en compte au REDÉMARRAGE de l'application.
    "models_dir": None,
    "gpu_index": None,          # None = auto (meilleure carte détectée)
    "auto_gpu_index": None,     # choix facultatif, indépendant des anciens presets manuels
    # GPU secondaire dédié au TEXTE (améliorateur de prompt). None = même GPU
    # que la génération. Ex. : mettre la 1080 Ti ici. La génération d'images et
    # l'upscale SDXL restent TOUJOURS sur le GPU de génération.
    "text_gpu_index": None,
    # Auto = profil système recalculé pour chaque modèle et ses poids installés
    # au moment de la génération. Manual = réglages experts et profils mesurés.
    # Les anciennes préférences restent intactes pour pouvoir revenir au manuel.
    "system_mode": "auto",
    "hardware_bias": "balanced",  # mémoire | balanced | qualité, même en auto
    # Moteur d'inférence pour la génération d'images : "sdcpp" (intégré) ou
    # "comfyui" (API HTTP locale, ex. http://127.0.0.1:8188).
    "generation_backend": "sdcpp",
    "comfyui_url": "http://127.0.0.1:8188",
    # EXPÉRIMENTAL : GPU dédié à l'encodeur de texte dans sd.cpp (--backend te=).
    # None = comportement normal (encodeur sur le GPU principal / déchargé RAM).
    "encoder_gpu_index": None,
    # Résidence explicite des POIDS sd.cpp (`--params-backend`). Le calcul est
    # réglé séparément par `--backend`. Cette séparation évite qu'un encodeur
    # annoncé « sur la 2e carte » soit en réalité relu depuis la RAM à travers
    # un port PCIe lent. Vide = laisser le moteur / le profil décider.
    "params_backend": "",
    # EXPÉRIMENTAL : répartit AUTOMATIQUEMENT le modèle (diffusion/encodeur/VAE)
    # sur TOUS les GPU visibles selon leur VRAM (sd.cpp --auto-fit). Permet
    # d'utiliser la VRAM d'une 2e carte (ex. 1080 Ti) pour la DIFFUSION elle-même,
    # pas seulement l'encodeur. Remplace le split d'encodeur ci-dessus.
    "auto_fit": False,
    # Mode de découpe entre GPU : "layer" (blocs entiers, défaut) ou "row"
    # (lignes de matmul réparties, CUDA seulement).
    "split_mode": "layer",
    # Explicit opt-in survives automatic memory profiles and machine presets.
    "sage_attention": False,
    "auto_optimize": True,      # déduire les flags du matériel
    "quant": None,              # None = recommandé selon VRAM
    "enc_quant": None,          # None = recommandé selon RAM
    # Fichiers GGUF choisis explicitement dans le catalogue, par modèle/rôle.
    # L'absence de choix conserve la quantification automatique.
    "model_files": {},
    # Surcharges manuelles (utilisées seulement si auto_optimize = False)
    "flags": {
        "diffusion_fa": True,
        "offload_to_cpu": True,
        "vae_tiling": True,
        "clip_on_cpu": False,
        "vae_on_cpu": False,
    },
    # Convolution DIRECTE au lieu d'im2col (sd.cpp --diffusion-conv-direct /
    # --vae-conv-direct). Volontairement HORS du dictionnaire « flags » : ce
    # n'est pas une déduction matérielle, c'est un compromis à essayer, et le
    # mettre dans flags le ferait écraser par le profil auto et par les presets
    # « 1 clic ». Détecté sur le binaire : ignoré s'il ne connaît pas l'option.
    "conv_direct_diffusion": False,
    "conv_direct_vae": False,
    "hf_endpoint": "https://huggingface.co",
    # Mo/s décimaux : 0 = sans plafond. Lu pendant le transfert.
    "download_limit_mbps": 5.0,
    # Accélération par cache (sd.cpp docs/caching.md). "" = désactivé.
    # Modes DiT (Flux/Krea) : easycache | dbcache | taylorseer | cache-dit | spectrum
    "cache_mode": "",
    "cache_option": "",         # ex. "threshold=0.2" (easycache) — vide = défauts
    # Presets ciblés : contrairement au cache global ci-dessus, ils ne touchent
    # que le modèle nommé — { "<id>": {"mode": ..., "option": ...} }. Plus
    # exposé dans l'interface : sur les modèles distillés en 4 à 8 pas le
    # cache aide peu ; sur Qwen (plus de pas), il peut changer le rendu.
    # Le choix ciblé par modèle reste modifiable manuellement.
    "cache_by_model": {},
    # Exécution SEGMENTÉE (sd.cpp --max-vram) : autorise le moteur à découper
    # son graphe de calcul pour tenir dans un budget, au lieu d'allouer d'un
    # bloc et d'échouer si ça ne rentre pas.
    #   ""     -> désactivé pour la génération ordinaire (comportement d'avant)
    #   "auto" -> VRAM libre détectée moins une marge
    #   "6" / "cuda0=6,cuda1=4" -> plafond ferme
    # La passe HD l'utilise de toute façon : c'est là que le tout-ou-rien casse.
    "max_vram": "",
    # Streaming des couches depuis leur backend de paramètres. Requiert les
    # poids de diffusion en RAM (`--offload-to-cpu` ou params diffusion=cpu),
    # mais PAS `--max-vram`. Coûteux en bande passante PCIe : opt-in.
    "stream_layers": False,
}


def served_paths() -> list[str]:
    """Dossiers que le serveur web a le droit de servir, en plus du cache Gradio.

    Sans cette liste, une image désignée par son chemin D'ORIGINE — et non par
    une copie de cache — reçoit un 403, donc une icône cassée côté navigateur.
    C'est le cas de tout ce que l'application produit elle-même.

    Volontairement limité aux deux dossiers qui contiennent des IMAGES : c'est
    ce que le navigateur a le droit de lire par chemin. Y ajouter models/ ou la
    racine du projet donnerait la même permission à tout le reste.
    """
    return [str(OUTPUT_DIR), str(TMP_DIR)]


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CUSTOM_DIR, LORA_DIR, BIN_DIR, OUTPUT_DIR, TMP_DIR,
              CACHE_DIR, USERDATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def local_cache_env(root: Path | None = None) -> dict[str, str]:
    """Return every cache and temp env var anchored inside the app directory.

    Keeping uv, pip, Hugging Face, PyTorch, Triton, CUDA, XDG, Gradio and OS
    temp folders on the same filesystem as the app avoids cross-drive hardlink
    fallbacks (e.g. C: vs D:/X:), speeds up installs/loads, and leaves nothing
    behind outside the app folder.
    """
    if root is None:
        cache = CACHE_DIR
        tmp = TMP_DIR
        runtime = RUNTIME_DIR
    else:
        base = Path(root)
        cache = base / ".cache"
        tmp = base / "tmp"
        runtime = base / "runtime"
    hf = cache / "huggingface"
    hf_hub = hf / "hub"
    sys_tmp = cache / "tmp"
    return {
        "UV_CACHE_DIR": str(cache / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(runtime / "uv-python"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf_hub),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCH_HOME": str(cache / "torch"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "CUDA_CACHE_PATH": str(cache / "nv"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "GRADIO_TEMP_DIR": str(tmp / "gradio"),
        "TMP": str(sys_tmp),
        "TEMP": str(sys_tmp),
        "TMPDIR": str(sys_tmp),
    }


def configure_local_caches(
    env: dict[str, str] | None = None,
    *,
    root: Path | None = None,
    create: bool = True,
) -> dict[str, str]:
    """Force all cache and temp variables inside the app directory."""
    import os
    target = os.environ if env is None else env
    mapping = local_cache_env(root)
    for key, value in mapping.items():
        target[key] = value
    if create:
        for path in set(mapping.values()):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
    return mapping


def sync_local_pip_configs(root: Path | None = None) -> list[Path]:
    """Pin pip's cache-dir inside every local Python environment under the app."""
    import os
    base = ROOT if root is None else Path(root)
    cache_dir = (CACHE_DIR if root is None else base / ".cache") / "pip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = "pip.ini" if os.name == "nt" else "pip.conf"
    content = f"[global]\ncache-dir = {cache_dir}\n"
    candidates: list[Path] = [base / "python", base / "venv", base / ".venv"]
    tools_repo = base / "tools_repo"
    if tools_repo.is_dir():
        for entry in sorted(tools_repo.iterdir()):
            if entry.is_dir():
                candidates.append(entry / ".venv")
    written: list[Path] = []
    for env_dir in candidates:
        if not env_dir.is_dir():
            continue
        cfg = env_dir / cfg_name
        try:
            if cfg.is_file() and cfg.read_text(encoding="utf-8") == content:
                written.append(cfg)
                continue
            cfg.write_text(content, encoding="utf-8")
            written.append(cfg)
        except OSError:
            pass
    return written


@lru_cache(maxsize=4)
def _read_preferences(path: Path, mtime_ns: int, size: int) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_PREFS)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            merged.update({k: v for k, v in saved.items() if k != "flags"})
            if isinstance(saved.get("flags"), dict):
                merged["flags"].update(saved["flags"])
    except (ValueError, OSError, UnicodeError):
        pass
    return merged


def load_prefs() -> dict[str, Any]:
    try:
        st = PREFS_FILE.stat()
        return deepcopy(_read_preferences(PREFS_FILE, st.st_mtime_ns, st.st_size))
    except OSError:
        return deepcopy(DEFAULT_PREFS)


def save_prefs(prefs: dict[str, Any]) -> None:
    ensure_dirs()
    atomic_write_text(PREFS_FILE, json.dumps(prefs, indent=2))
    _read_preferences.cache_clear()


def generation_gpu_index(prefs: dict[str, Any]) -> int | None:
    """Choix de carte actif ; l'ancien choix manuel ne fige pas l'auto."""
    return (prefs.get("auto_gpu_index") if prefs.get("system_mode") == "auto"
            else prefs.get("gpu_index"))


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


def child_env(gpu_index: "int | None" = None) -> dict:
    """Environnement des sous-process Python (outils, installeurs).

    Force l'UTF-8 côté ENFANT. Sans ça, sous Windows la sortie standard d'un
    sous-process hérite de la page de codes de la console (cp1252) : le moindre
    emoji affiché par du code tiers lève un UnicodeEncodeError et tue le
    process — souvent à l'import, avant même d'avoir commencé à travailler.
    On ne peut pas corriger le code tiers, mais on peut lui donner un stdout
    capable d'encoder ce qu'il écrit. Nos propres lecteurs décodent déjà en
    UTF-8, donc les deux bouts sont cohérents.
    """
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # stdout/stderr de l'enfant en UTF-8
    env["PYTHONUTF8"] = "1"             # mode UTF-8 global (PEP 540)
    mapping = configure_local_caches(env, create=False)
    try:
        Path(mapping["TMP"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def configure_hf_env() -> None:
    """Configure l'environnement Hugging Face pour des téléchargements fiables.

    HTTP séquentiel : les téléchargements contrôlés n'utilisent pas Xet.
    """
    import os
    mapping = configure_local_caches(create=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Gradio peut avoir importé le Hub avant cet appel : ses constantes sont
    # figées à l'import, modifier seulement os.environ ne suffit alors pas.
    import sys
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HOME = mapping["HF_HOME"]
        constants.HF_HUB_CACHE = mapping["HF_HUB_CACHE"]
        constants.HF_HUB_DISABLE_XET = True
        constants.HF_HUB_ENABLE_HF_TRANSFER = False
        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    prefs = load_prefs()
    os.environ.setdefault("HF_ENDPOINT",
                          prefs.get("hf_endpoint", "https://huggingface.co"))
