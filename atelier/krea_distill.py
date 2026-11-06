"""Pinned four-step Krea 2 Turbo adapter from lvladikov."""
from __future__ import annotations

import re

from . import model_store, settings, transfers

REPO = "lvladikov/Krea2-Turbo-Distill-4step-LoRA"
REVISION = "d4d6c8e9edb60587f9f6531b89995e668d5982dc"
FILENAME = "krea2_turbo_4step_rank_64_lora_comfyui.safetensors"
FILE_SIZE = 438_141_176
SHA256 = "27a3d1644fa1a8c137b8549591e50acc4746f8e9bc736fccff80e506a8ee8ac7"
LORA_NAME = "krea2-turbo-distill/" + FILENAME.removesuffix(".safetensors")
SIGMAS = "1.0,0.90453,0.75951,0.51284,0.0"


def path():
    return settings.LORA_DIR / (LORA_NAME + ".safetensors")


def strip_tag(prompt):
    return re.sub(r"\s*<lora:" + re.escape(LORA_NAME) + r"(?:\.safetensors)?:[^>]+>", "", prompt)


def ready():
    try:
        return path().is_file() and path().stat().st_size == FILE_SIZE
    except OSError:
        return False


def install(log=print):
    log("Downloading / verifying lvladikov's Krea 2 Turbo four-step LoRA (438 MB)…")
    transfers.hf_hub_download(repo_id=REPO, filename=FILENAME, revision=REVISION,
                              local_dir=str(path().parent))
    if not ready() or model_store.digest(path(), transfers._check_cancelled) != SHA256:
        raise RuntimeError("Krea 2 Turbo LoRA failed verification. Run Install / repair again.")
    log("Installed. In Images → Krea 2 Turbo, choose 4 steps + distillation LoRA.")


def require():
    if not ready():
        raise ValueError("Download Krea 2 Turbo · 4-step distillation LoRA in Models first.")
