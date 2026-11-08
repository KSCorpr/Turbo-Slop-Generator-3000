"""Pinned Kromoss DMD2 adapter for native, unmerged MOSS inference."""
from __future__ import annotations

from . import moss_dmd, settings, transfers

REPO = "lodestones/Kromoss"
REVISION = "0c5417ec89301cb82da9ee164f8ed696523910a9"
FILENAME = "8-steps-dmd2-lora.safetensors"
FILE_SIZE = 1_189_176_800
SHA256 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
LORA_NAME = "kromoss-dmd2/8-steps-dmd2-lora"
STRENGTH = 1.0


def path():
    return settings.LORA_DIR / "kromoss-dmd2" / FILENAME


def strip_tag(prompt):
    return moss_dmd.strip_tag(prompt, LORA_NAME)


def ready():
    try:
        return path().is_file() and path().stat().st_size == FILE_SIZE
    except OSError:
        return False


def install(log=print):
    log("Downloading / verifying the Kromoss DMD2 adapter (1.19 GB)…")
    transfers.hf_hub_download(repo_id=REPO, filename=FILENAME,
                              revision=REVISION, local_dir=str(path().parent))
    require()
    log("Installed. In Images → Kromoss, select DMD2 fast (8 steps).")


def require():
    if not ready():
        raise ValueError("Download the Kromoss · DMD2 8-step adapter in Models first.")
    moss_dmd.validate(path())
