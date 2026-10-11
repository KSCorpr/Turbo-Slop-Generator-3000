"""Paths and pinned versions for optional, isolated media tools."""
from __future__ import annotations

import os
from pathlib import Path
from . import settings

SEED_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEED_CONVROT_COMMIT = "68191c40d03566608b65d24e97d11dc4389f3a62"
COMFY_CONVROT_COMMIT = "7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217"
BRUSH_VERSION = "v0.3.0"
ADDON_NAMES = ("seedvr2", "seedvr2-convrot", "splat")
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
