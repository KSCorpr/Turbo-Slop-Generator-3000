"""Paths and pinned versions for optional, isolated media tools."""
from __future__ import annotations

import os
from pathlib import Path
from . import settings

SEED_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
BRUSH_VERSION = "v0.3.0"
SEED_MODELS = {
    "3b": "seedvr2_ema_3b-Q8_0.gguf",
    "7b": "seedvr2_ema_7b-Q4_K_M.gguf",
}


def root(name: str) -> Path:
    if name not in ("seedvr2", "splat"):
        raise ValueError("Unknown add-on")
    return settings.ROOT / "tools_repo" / name


def python(name: str) -> Path:
    return root(name) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def brush() -> Path | None:
    folder = root("splat") / "brush"
    names = ("brush-app.exe", "brush.exe") if os.name == "nt" else ("brush-app", "brush")
    for name in names:
        found = sorted(folder.rglob(name)) if folder.exists() else []
        if found:
            return found[0]
    return None


def ready(name: str) -> bool:
    return (root(name) / "installed.json").is_file() and python(name).is_file()
