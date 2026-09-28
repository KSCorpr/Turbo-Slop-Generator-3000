"""MiniMax H3 Turbo: three video-only weights, shared across all video modes."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from . import downloader, settings
from .registry import Component

DIFFUSION_REPO = "molbal/MiniMax-H3-Turbo-GGUF"
ENCODER_REPO = "leejet/MiniMax-H3-GGUF"
VAE_REPO = "Comfy-Org/MiniMax-H3"

# Bytes from the public repository listings; the scan button reads exact sizes.
DIFFUSION = {
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q4_0.gguf": 11.4e9,
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_0.gguf": 21.4e9,
}
ENCODERS = {
    "qwen3vl_32b_minimax_h3-Q2_K_M.gguf": 13.1e9,
    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf": 18.2e9,
}
VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
VAE_SIZE = 5.21e9
DEFAULT_DIFFUSION = next(iter(DIFFUSION))
DEFAULT_ENCODER = next(iter(ENCODERS))


def selected(prefs: dict | None = None) -> tuple[str, str]:
    prefs = prefs if prefs is not None else settings.load_prefs()
    choices = prefs.get("video_model_files") or {}
    if not isinstance(choices, dict):
        choices = {}
    diffusion = choices.get("diffusion")
    encoder = choices.get("text_encoder")
    return (diffusion if diffusion in DIFFUSION else DEFAULT_DIFFUSION,
            encoder if encoder in ENCODERS else DEFAULT_ENCODER)


def weights(diffusion: str, encoder: str) -> tuple[Path, Path, Path]:
    if diffusion not in DIFFUSION or encoder not in ENCODERS:
        raise ValueError("Choose MiniMax H3 weights from the Model Catalog.")
    return (settings.model_repo_dir(DIFFUSION_REPO) / diffusion,
            settings.model_repo_dir(ENCODER_REPO) / encoder,
            settings.model_repo_dir(VAE_REPO) / VAE)


def ready(diffusion: str, encoder: str) -> bool:
    return all(p.is_file() and p.stat().st_size > 0
               for p in weights(diffusion, encoder))


def choices(role: str, *, exact: bool = False) -> list[tuple[str, str]]:
    """Offer only files that are known to run as standalone FL2VA GGUF weights."""
    repo, names = ((DIFFUSION_REPO, DIFFUSION) if role == "diffusion"
                   else (ENCODER_REPO, ENCODERS))
    sizes = dict(names)
    if exact:
        settings.configure_hf_env()
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        available = {s.rfilename: s for s in info.siblings}
        missing = set(names) - set(available)
        if missing:
            raise RuntimeError(f"Weights missing from {repo}: {', '.join(sorted(missing))}")
        for name in names:
            sibling = available[name]
            size = sibling.size or getattr(sibling.lfs, "size", None)
            if size:
                sizes[name] = size
    rows = []
    for name, size in sizes.items():
        here = settings.model_repo_dir(repo) / name
        installed = " · installed" if here.is_file() and here.stat().st_size else ""
        rows.append((f"{name} — {size / 1e9:.2f} GB{installed}", name))
    return rows


def download(diffusion: str, encoder: str,
             log: Callable[[str], None] | None = None) -> Iterator[str]:
    weights(diffusion, encoder)  # Reject arbitrary paths before downloading.
    specs = (("diffusion", DIFFUSION_REPO, diffusion),
             ("text_encoder", ENCODER_REPO, encoder),
             ("vae", VAE_REPO, VAE))
    yield "Downloading MiniMax H3 Turbo video weights (no audio VAE)…"
    for role, repo, name in specs:
        yield f"  ↓ {role}: {repo}/{name}"
        comp = Component(role, repo, name, None)
        downloader.download_component(comp, log=log)
        yield f"  ✓ {role}: {name}"


def delete(diffusion: str, encoder: str) -> int:
    """Remove the chosen H3 weights only; never remove other models' files."""
    count = 0
    dit, te, vae = weights(diffusion, encoder)
    other_diffusion = any(
        (settings.model_repo_dir(DIFFUSION_REPO) / name).is_file()
        for name in DIFFUSION if name != diffusion)
    # All H3 diffusion quantizations share the same encoder and video VAE.
    # Leave those components in place while another quantization is installed.
    paths = (dit,) if other_diffusion else (dit, te, vae)
    for path in paths:
        if path.is_file():
            path.unlink()
            count += 1
    return count
