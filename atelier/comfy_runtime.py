"""Portable ComfyUI installation and model paths; no Torch in the UI process."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from . import hardware, settings
from .fileio import atomic_write_text

VERSION = "v0.39.0"
COMMIT = "b0b743566f65daafc423b4fea8a2fbda94b3384a"
SCHEMA = 1
MODEL_IDS = frozenset({"qwen-image-2.1-turbo", "sdxl-base"})


def root() -> Path:
    return settings.ROOT / "tools_repo" / "comfy"


def python() -> Path:
    return root() / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def manifest() -> dict:
    try:
        value = json.loads((root() / "installed.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def ready() -> bool:
    info = manifest()
    return (info.get("commit") == COMMIT and info.get("schema") == SCHEMA
            and python().is_file() and (root() / "source" / "main.py").is_file())


def enabled(model_id: str, prefs: dict | None = None) -> bool:
    prefs = settings.load_prefs() if prefs is None else prefs
    return model_id in MODEL_IDS and prefs.get("image_backend", "comfy") == "comfy"


def selected_gpu(prefs: dict | None = None):
    prefs = settings.load_prefs() if prefs is None else prefs
    return hardware.auto_profile(settings.generation_gpu_index(prefs)).gpu


def cuda_flavor(gpu) -> str:
    if gpu is None:
        return "cpu"
    if gpu.is_apple:
        return "mps"
    try:
        recent_driver = int(gpu.driver.split(".")[0]) >= 580
    except (ValueError, AttributeError):
        recent_driver = False
    return "cu130" if gpu.tensor_cores and recent_driver else "cu126"


def gpu_env(index: int | None) -> dict:
    env = settings.child_env(index)
    # NVIDIA-SMI indices need not match CUDA's enumeration of mixed cards.
    if index is not None:
        rows = hardware._nvidia_smi("index,uuid") or ""
        for row in rows.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 2 and parts[0] == str(index) and parts[1].startswith("GPU-"):
                env["CUDA_VISIBLE_DEVICES"] = parts[1]
                break
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
    return env


def model_roots(extra: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    candidates = [settings.MODELS_DIR, settings.CUSTOM_DIR,
                  settings.ROOT / "tools_repo" / "upscale", settings.LORA_DIR]
    # A user can explicitly select a checkpoint outside the portable folder.
    candidates += [Path(p).resolve().parent for p in extra]
    return tuple(dict.fromkeys(p.resolve() for p in candidates))


def model_name(path: Path, roots: tuple[Path, ...]) -> str:
    path = Path(path).resolve()
    for directory in roots:
        if path.is_relative_to(directory):
            return path.relative_to(directory).as_posix()
    raise ValueError(f"Model is outside the configured local folders: {path.name}")


def write_paths(roots: tuple[Path, ...]) -> Path:
    # Comfy recursively discovers weights; no copies or Windows junctions.
    folders = "\n".join(str(p) for p in roots)
    config = {"turboslop": {key: folders for key in
              ("checkpoints", "diffusion_models", "text_encoders", "vae",
               "loras", "controlnet", "upscale_models")}}
    destination = root() / "model-paths.yaml"
    atomic_write_text(destination, yaml.safe_dump(config, allow_unicode=True))
    return destination


def command(port: int, config: Path, *, gpu=None) -> list[str]:
    folder = root()
    cmd = [str(python()), "-u", str(folder / "source" / "main.py"),
           "--listen", "127.0.0.1", "--port", str(port),
           "--disable-auto-launch", "--offline", "--disable-all-custom-nodes",
           "--whitelist-custom-nodes", "turboslop",
           "--extra-model-paths-config", str(config),
           "--input-directory", str(folder / "input"),
           "--output-directory", str(folder / "output"),
           "--temp-directory", str(settings.TMP_DIR / "comfy"),
           "--user-directory", str(folder / "user"),
           "--preview-method", "latent2rgb", "--cache-lru", "12"]
    flavor = manifest().get("flavor", "cpu")
    if flavor == "cpu" or (gpu is None and flavor != "mps"):
        cmd += ["--cpu"]
    elif flavor == "cu126":
        # Current optimized Kitchen CUDA kernels require cu130. Keep the
        # compatible runtime usable with older drivers, without claiming it
        # has native INT8 acceleration. Standard SDPA and offload still work.
        cmd += ["--disable-dynamic-vram", "--lowvram"]
    return cmd


def status() -> str:
    if not ready():
        return "Comfy local engine: installation required."
    info = manifest()
    flavor = info.get("flavor", "unknown")
    detail = ("compatible CUDA 12.6; optimized INT8 CUDA kernels unavailable"
              if flavor == "cu126" else flavor)
    return f"Comfy local engine ready · {VERSION} · {detail}."
