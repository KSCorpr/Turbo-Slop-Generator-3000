"""MiniMax H3 video weights: separate Turbo and Ref2VA diffusion checkpoints."""
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
REF_DIFFUSION = {
    "minimax_h3_ref2va_pruned-Q2_K_M.gguf": 6.72e9,
    "minimax_h3_ref2va_pruned-Q4_K_M.gguf": 11.4e9,
}
ENCODERS = {
    "qwen3vl_32b_minimax_h3-Q2_K_M.gguf": 13.1e9,
    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf": 18.2e9,
}
VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
VAE_SIZE = 5.21e9
DEFAULT_DIFFUSION = next(iter(DIFFUSION))
DEFAULT_REF_DIFFUSION = next(iter(REF_DIFFUSION))
DEFAULT_ENCODER = next(iter(ENCODERS))


def _diffusions(mode: str) -> tuple[str, dict[str, float], str, str]:
    if mode == "turbo":
        return DIFFUSION_REPO, DIFFUSION, "diffusion", DEFAULT_DIFFUSION
    if mode == "refs":
        return ENCODER_REPO, REF_DIFFUSION, "ref_diffusion", DEFAULT_REF_DIFFUSION
    raise ValueError("Unknown MiniMax H3 video mode.")


def selected(prefs: dict | None = None, *, mode: str = "turbo") -> tuple[str, str]:
    prefs = prefs if prefs is not None else settings.load_prefs()
    choices = prefs.get("video_model_files") or {}
    if not isinstance(choices, dict):
        choices = {}
    _, diffusions, key, default = _diffusions(mode)
    diffusion = choices.get(key)
    encoder = choices.get("text_encoder")
    return (diffusion if diffusion in diffusions else default,
            encoder if encoder in ENCODERS else DEFAULT_ENCODER)


def weights(diffusion: str, encoder: str, *, mode: str = "turbo") -> tuple[Path, Path, Path]:
    repo, diffusions, _, _ = _diffusions(mode)
    if diffusion not in diffusions or encoder not in ENCODERS:
        raise ValueError("Choose MiniMax H3 weights from the Model Catalog.")
    return (settings.model_repo_dir(repo) / diffusion,
            settings.model_repo_dir(ENCODER_REPO) / encoder,
            settings.model_repo_dir(VAE_REPO) / VAE)


def ready(diffusion: str, encoder: str, *, mode: str = "turbo") -> bool:
    return all(p.is_file() and p.stat().st_size > 0
               for p in weights(diffusion, encoder, mode=mode))


def choices(role: str, *, exact: bool = False) -> list[tuple[str, str]]:
    """Offer verified H3 files, showing their sizes and installed status."""
    if role == "text_encoder":
        repo, names = ENCODER_REPO, ENCODERS
    elif role in ("diffusion", "ref_diffusion"):
        repo, names, _, _ = _diffusions("turbo" if role == "diffusion" else "refs")
    else:
        raise ValueError("Unknown MiniMax H3 weight type.")
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
             log: Callable[[str], None] | None = None, *,
             mode: str = "turbo") -> Iterator[str]:
    weights(diffusion, encoder, mode=mode)  # Reject arbitrary paths.
    repo, _, _, _ = _diffusions(mode)
    specs = (("diffusion", repo, diffusion),
             ("text_encoder", ENCODER_REPO, encoder),
             ("vae", VAE_REPO, VAE))
    yield f"Downloading MiniMax H3 {mode} video weights (no audio VAE)…"
    for role, repo, name in specs:
        yield f"  ↓ {role}: {repo}/{name}"
        comp = Component(role, repo, name, None)
        downloader.download_component(comp, log=log)
        yield f"  ✓ {role}: {name}"


def delete(diffusion: str, encoder: str, *, mode: str = "turbo") -> int:
    """Remove the chosen H3 weights only; never remove other models' files."""
    count = 0
    dit, te, vae = weights(diffusion, encoder, mode=mode)
    repo, diffusions, _, _ = _diffusions(mode)
    other_diffusion = any(
        (settings.model_repo_dir(other_repo) / name).is_file()
        for other_repo, names in ((DIFFUSION_REPO, DIFFUSION),
                                  (ENCODER_REPO, REF_DIFFUSION))
        for name in names if (other_repo, name) != (repo, diffusion))
    # Both modes and all quantizations use the same encoder and video VAE.
    paths = (dit,) if other_diffusion else (dit, te, vae)
    for path in paths:
        if path.is_file():
            path.unlink()
            count += 1
    return count
