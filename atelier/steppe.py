"""Pinned TensorForger/Steppe-Wan2.1 latent upscaler artifacts (no Torch import)."""
from __future__ import annotations

import json
from pathlib import Path

from . import addons, transfers

SOURCE_REPO = "TensorForger/ComfyUI-Steppe"
SOURCE_COMMIT = "ad9eee37ae2bfbf83893016b94930334f96e6f0a"
MODEL_REPO = "TensorForger/Steppe-Wan2.1"
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

MODELS: dict[str, tuple[str, str]] = {
    "200m-1step": (
        "steppe_wan_2_1_200M_1step.safetensors",
        "Steppe 200M · 1-step (445 MB · default · 99% of base quality)",
    ),
    "200m-base": (
        "steppe_wan_2_1_200M_base.safetensors",
        "Steppe 200M · Base multi-step (444 MB · 1-50 steps · CFG on HFF)",
    ),
    "20m-1step": (
        "steppe_wan_2_1_20M_1step.safetensors",
        "Steppe 20M · 1-step lite (39 MB · very large 2K→4K & fast video)",
    ),
}

DEFAULT_VARIANT = "200m-1step"

VAE_FILES: tuple[str, ...] = (
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


def model_dir() -> Path:
    return addons.root("steppe") / "models"


def vae_dir() -> Path:
    return model_dir() / "wan21_vae" / "vae"


def checkpoint_path(variant: str = DEFAULT_VARIANT) -> Path:
    if variant not in MODELS:
        raise ValueError("Choose a listed Steppe-Wan2.1 model variant.")
    return model_dir() / MODELS[variant][0]


def vae_ready() -> bool:
    root = model_dir() / "wan21_vae"
    return all((root / rel).is_file() and (root / rel).stat().st_size > 0 for rel in VAE_FILES)


def weights_ready(variant: str = DEFAULT_VARIANT) -> bool:
    if variant not in MODELS:
        return False
    ckpt = checkpoint_path(variant)
    return vae_ready() and ckpt.is_file() and ckpt.stat().st_size > 0


def ready(variant: str = DEFAULT_VARIANT) -> bool:
    try:
        marker = json.loads((addons.root("steppe") / "installed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        addons.ready("steppe")
        and marker.get("commit") == SOURCE_COMMIT
        and (addons.root("steppe") / "source" / "steppe" / "__init__.py").is_file()
        and weights_ready(variant)
    )


def any_ready() -> bool:
    return any(ready(v) for v in MODELS)


def download_models(variant: str = DEFAULT_VARIANT, log=print) -> Path:
    if variant not in MODELS:
        raise ValueError("Choose a listed Steppe-Wan2.1 model variant.")
    vae_root = model_dir() / "wan21_vae"
    vae_root.mkdir(parents=True, exist_ok=True)
    for rel in VAE_FILES:
        dest = vae_root / rel
        if dest.is_file() and dest.stat().st_size > 0:
            log(f"Steppe Wan 2.1 VAE: already installed — {rel}")
            continue
        log(f"Downloading Steppe Wan 2.1 VAE ({VAE_REPO}): {rel}")
        transfers.hf_hub_download(
            repo_id=VAE_REPO,
            filename=rel,
            local_dir=str(vae_root),
        )

    filename = MODELS[variant][0]
    ckpt_dest = checkpoint_path(variant)
    if ckpt_dest.is_file() and ckpt_dest.stat().st_size > 0:
        log(f"Steppe-Wan2.1 checkpoint: already installed — {filename}")
        return ckpt_dest
    model_dir().mkdir(parents=True, exist_ok=True)
    log(f"Downloading Steppe-Wan2.1 ({MODEL_REPO}): {filename}")
    return Path(
        transfers.hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(model_dir()),
        )
    )
