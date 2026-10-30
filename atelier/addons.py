"""Paths and pinned versions for optional, isolated media tools."""
from __future__ import annotations

import os
from pathlib import Path
from . import settings

SEED_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEED_CONVROT_COMMIT = "68191c40d03566608b65d24e97d11dc4389f3a62"
COMFY_CONVROT_COMMIT = "7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217"
BRUSH_VERSION = "v0.3.0"
ADDON_NAMES = ("seedvr2", "seedvr2-convrot", "vosr", "splat", "svg", "hivg", "omnisvg", "sam2", "qwen-layered", "image-splats")
SAM2_REPO = "facebook/sam2.1-hiera-small"
SAM2_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
QWEN_LAYERED_REPO = "Qwen/Qwen-Image-Layered"
QWEN_LAYERED_REVISION = "8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf"
QWEN_LAYERED_GGUF_REPO = "unsloth/Qwen-Image-Layered-GGUF"
QWEN_LAYERED_GGUF_REVISION = "c83e10072f369c67db4a88c71ee63871b6b4ad90"
QWEN_LAYERED_GGUF = "qwen-image-layered-Q4_K_M.gguf"
OMNISVG_COMMIT = "308b49b3e3df29c8b5424bbaa7f1f7c6ff218e8d"
OMNISVG_REVISION = "117d4c4541839e02943554fe71c0a0d54fcef819"
OMNISVG_PROCESSOR_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
HIVG_COMMIT = "2ada95d057cb5d04fdbad4686258e174c8280c5f"
HIVG_REVISION = "259cba9684201bbde830a59ca7543345bb333519"
LAYERTRACER_COMMIT = "8e089f92795b4f32dcae8f7dbbd5ced20395dad7"
SEED_MODELS = {
    "3b": "seedvr2_ema_3b-Q8_0.gguf",
    "7b": "seedvr2_ema_7b-Q4_K_M.gguf",
}
SEED_CONVROT_MODELS = {
    "3b-w4a8": "seedvr2_3b_convrot_w4a8.safetensors",
    "3b-int8": "seedvr2_3b_convrot_int8.safetensors",
    "7b-w4a8": "seedvr2_7b_convrot_w4a8.safetensors",
    "7b-int8": "seedvr2_7b_convrot_int8.safetensors",
    "7b-sharp-w4a8": "seedvr2_7b_sharp_convrot_w4a8.safetensors",
    "7b-sharp-int8": "seedvr2_7b_sharp_convrot_int8.safetensors",
}


def seed_addon(model: str) -> str:
    if model in SEED_CONVROT_MODELS:
        return "seedvr2-convrot"
    if model in SEED_MODELS:
        return "seedvr2"
    raise ValueError("Choose a listed SeedVR2 model.")


def root(name: str) -> Path:
    if name not in ADDON_NAMES:
        raise ValueError("Unknown add-on")
    return settings.ROOT / "tools_repo" / name


def python(name: str) -> Path:
    return root(name) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def brush() -> Path | None:
    folder = root("splat") / "brush"
    # The pinned v0.3.0 archive uses an underscore, despite its hyphenated
    # package/archive name. Keep the other spellings for existing installs.
    names = (("brush_app.exe", "brush-app.exe", "brush.exe") if os.name == "nt"
             else ("brush_app", "brush-app", "brush"))
    for name in names:
        found = sorted(p for p in folder.rglob(name) if p.is_file()) if folder.exists() else []
        if found:
            return found[0]
    return None


def ready(name: str) -> bool:
    return (root(name) / "installed.json").is_file() and python(name).is_file()
