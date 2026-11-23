"""MiniMax H3 Video & HyperFlow H3 Flash weight catalog and shared-path resolver.

Reuses the exact same `models/` repository layout as MiniMax H3 Image so
shared text encoders (Qwen3-VL-32B GGUF or P-H3 Qwen3-VL-4B) are never
downloaded twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import settings, transfers


@dataclass(frozen=True)
class WeightEntry:
    key: str
    label: str
    repo: str
    filename: str
    size_gb: float
    kind: str = ""          # "fl2va" | "ref2va" | ""
    hyperflow: bool = False
    min_build: int = 1088


DIFFUSION_VARIANTS: dict[str, WeightEntry] = {
    "hyperflow-fl2va-q4": WeightEntry(
        key="hyperflow-fl2va-q4",
        label="HyperFlow H3 Flash FL2VA · Q4_K_M (9.05 GB · 4-8 steps · 11-12 GB GPU)",
        repo="VideoRebirthAI/hyperflow-h3-flash",
        filename="fl2va_gguf/minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf",
        size_gb=9.05,
        kind="fl2va",
        hyperflow=True,
        min_build=1088,
    ),
    "hyperflow-fl2va-q5": WeightEntry(
        key="hyperflow-fl2va-q5",
        label="HyperFlow H3 Flash FL2VA · Q5_K_M (10.3 GB · 4-8 steps)",
        repo="VideoRebirthAI/hyperflow-h3-flash",
        filename="fl2va_gguf/minimax_h3_fl2va_hyperflow_flash_14b-Q5_K_M.gguf",
        size_gb=10.3,
        kind="fl2va",
        hyperflow=True,
        min_build=1088,
    ),
    "hyperflow-fl2va-int8": WeightEntry(
        key="hyperflow-fl2va-int8",
        label="HyperFlow H3 Flash FL2VA · INT8 ConvRot (16.0 GB · 64 GB RAM offload)",
        repo="VideoRebirthAI/hyperflow-h3-flash",
        filename="fl2va_int8_convrot_comfyui/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors",
        size_gb=16.0,
        kind="fl2va",
        hyperflow=True,
        min_build=1088,
    ),
    "hyperflow-ref2va-q4": WeightEntry(
        key="hyperflow-ref2va-q4",
        label="HyperFlow H3 Flash Ref2VA · Q4_K_M (9.22 GB · reference image → video)",
        repo="VideoRebirthAI/hyperflow-h3-flash",
        filename="ref2va_gguf/minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf",
        size_gb=9.22,
        kind="ref2va",
        hyperflow=True,
        min_build=1088,
    ),
    "hyperflow-ref2va-int8": WeightEntry(
        key="hyperflow-ref2va-int8",
        label="HyperFlow H3 Flash Ref2VA · INT8 ConvRot (16.4 GB · reference image → video)",
        repo="VideoRebirthAI/hyperflow-h3-flash",
        filename="ref2va_int8_convrot_comfyui/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
        size_gb=16.4,
        kind="ref2va",
        hyperflow=True,
        min_build=1088,
    ),
    "h3-fl2va-q4": WeightEntry(
        key="h3-fl2va-q4",
        label="MiniMax H3 Base FL2VA · Pruned Q4_K_M (11.4 GB · 20-30 steps)",
        repo="leejet/MiniMax-H3-GGUF",
        filename="minimax_h3_fl2va_pruned-Q4_K_M.gguf",
        size_gb=11.4,
        kind="fl2va",
        hyperflow=False,
        min_build=1088,
    ),
    "h3-ref2va-q4": WeightEntry(
        key="h3-ref2va-q4",
        label="MiniMax H3 Base Ref2VA · Pruned Q4_K_M (11.4 GB · 20-30 steps)",
        repo="leejet/MiniMax-H3-GGUF",
        filename="minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        size_gb=11.4,
        kind="ref2va",
        hyperflow=False,
        min_build=1088,
    ),
}

ENCODER_VARIANTS: dict[str, WeightEntry] = {
    "p-h3-int8": WeightEntry(
        key="p-h3-int8",
        label="P-H3 Qwen3-VL-4B INT8 ConvRot (5.12 GB · recommended for 11-12 GB GPU)",
        repo="PrunaAI/P-H3-Image-Comfy",
        filename="text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
        size_gb=5.12,
        min_build=1090,
    ),
    "p-h3-w4a8": WeightEntry(
        key="p-h3-w4a8",
        label="P-H3 Qwen3-VL-4B W4A8 (3.58 GB · lightest · sd.cpp build 1110+)",
        repo="PrunaAI/P-H3-Image-Comfy",
        filename="text_encoders/p-h3-image_text_encoder_w4a8.safetensors",
        size_gb=3.58,
        min_build=1110,
    ),
    "qwen3vl-32b-q4": WeightEntry(
        key="qwen3vl-32b-q4",
        label="Qwen3-VL-32B GGUF Q4_K_M (18.2 GB · shared with H3 Image · 64 GB RAM)",
        repo="leejet/MiniMax-H3-GGUF",
        filename="qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        size_gb=18.22,
        min_build=1088,
    ),
}

VIDEO_VAE_VARIANTS: dict[str, WeightEntry] = {
    "sol-int8": WeightEntry(
        key="sol-int8",
        label="Sol-H3-VAE INT8 ConvRot (1.99 GB · fast distilled decoder · recommended)",
        repo="Comfy-Org/MiniMax-H3",
        filename="vae/minimax_h3_sol_vae_int8_convrot.safetensors",
        size_gb=1.99,
        min_build=1088,
    ),
    "sol-fp16": WeightEntry(
        key="sol-fp16",
        label="Sol-H3-VAE FP16 (3.60 GB · fast distilled decoder)",
        repo="Comfy-Org/MiniMax-H3",
        filename="vae/minimax_h3_sol_vae_fp16.safetensors",
        size_gb=3.60,
        min_build=1088,
    ),
    "video-int8": WeightEntry(
        key="video-int8",
        label="Original H3 Video VAE INT8 ConvRot (2.81 GB)",
        repo="Comfy-Org/MiniMax-H3",
        filename="vae/minimax_h3_video_vae_int8_convrot.safetensors",
        size_gb=2.81,
        min_build=1088,
    ),
    "video-fp16": WeightEntry(
        key="video-fp16",
        label="Original H3 Video VAE FP16 (5.21 GB)",
        repo="Comfy-Org/MiniMax-H3",
        filename="vae/minimax_h3_video_vae_fp16.safetensors",
        size_gb=5.21,
        min_build=1088,
    ),
}

AUDIO_VAE = WeightEntry(
    key="audio-fp32",
    label="MiniMax H3 Audio VAE FP32 (0.61 GB · stereo soundtrack)",
    repo="Comfy-Org/MiniMax-H3",
    filename="vae/minimax_h3_audio_vae_fp32.safetensors",
    size_gb=0.61,
    min_build=1088,
)

DEFAULT_DIFFUSION = "hyperflow-fl2va-q4"
DEFAULT_ENCODER = "p-h3-int8"
DEFAULT_VAE = "sol-int8"


def entry_path(entry: WeightEntry) -> Path:
    return settings.model_repo_dir(entry.repo) / entry.filename


def is_installed(entry: WeightEntry) -> bool:
    p = entry_path(entry)
    return p.is_file() and p.stat().st_size > 0


def catalog_variants() -> tuple[tuple[str, str], ...]:
    """Variant list exposed in Models -> Video & 3D."""
    return tuple((e.label, e.key) for e in DIFFUSION_VARIANTS.values())


def resolve_files(
    diffusion_key: str = DEFAULT_DIFFUSION,
    encoder_key: str = DEFAULT_ENCODER,
    vae_key: str = DEFAULT_VAE,
) -> dict[str, Path | None]:
    diff = DIFFUSION_VARIANTS.get(diffusion_key)
    enc = ENCODER_VARIANTS.get(encoder_key)
    vae = VIDEO_VAE_VARIANTS.get(vae_key)
    if diff is None or enc is None or vae is None:
        raise ValueError("Choose listed MiniMax H3 Video weights.")
    return {
        "diffusion": entry_path(diff) if is_installed(diff) else None,
        "encoder": entry_path(enc) if is_installed(enc) else None,
        "vae": entry_path(vae) if is_installed(vae) else None,
        "audio_vae": entry_path(AUDIO_VAE) if is_installed(AUDIO_VAE) else None,
    }


def first_installed_encoder() -> str:
    """Prefer an already-installed shared H3 encoder (e.g. from H3 Image)."""
    for key in (DEFAULT_ENCODER, "p-h3-w4a8", "qwen3vl-32b-q4"):
        if is_installed(ENCODER_VARIANTS[key]):
            return key
    return DEFAULT_ENCODER


def first_installed_vae() -> str:
    for key in (DEFAULT_VAE, "sol-fp16", "video-int8", "video-fp16"):
        if is_installed(VIDEO_VAE_VARIANTS[key]):
            return key
    return DEFAULT_VAE


def ready(
    diffusion_key: str = DEFAULT_DIFFUSION,
    encoder_key: str | None = None,
    vae_key: str | None = None,
) -> bool:
    diff = DIFFUSION_VARIANTS.get(diffusion_key)
    if diff is None or not is_installed(diff):
        return False
    if not is_installed(AUDIO_VAE):
        return False
    enc_ok = (
        is_installed(ENCODER_VARIANTS[encoder_key])
        if encoder_key and encoder_key in ENCODER_VARIANTS
        else any(is_installed(e) for e in ENCODER_VARIANTS.values())
    )
    vae_ok = (
        is_installed(VIDEO_VAE_VARIANTS[vae_key])
        if vae_key and vae_key in VIDEO_VAE_VARIANTS
        else any(is_installed(v) for v in VIDEO_VAE_VARIANTS.values())
    )
    return enc_ok and vae_ok


def download_pack(
    diffusion_key: str = DEFAULT_DIFFUSION,
    encoder_key: str | None = None,
    vae_key: str | None = None,
    log=print,
) -> dict[str, Path]:
    """Download missing H3 Video components; reuses already-installed shared weights."""
    diff = DIFFUSION_VARIANTS.get(diffusion_key)
    if diff is None:
        raise ValueError(f"Unknown MiniMax H3 Video checkpoint: {diffusion_key}")
    enc_k = encoder_key or first_installed_encoder()
    vae_k = vae_key or first_installed_vae()
    enc = ENCODER_VARIANTS.get(enc_k)
    vae = VIDEO_VAE_VARIANTS.get(vae_k)
    if enc is None or vae is None:
        raise ValueError("Choose listed MiniMax H3 Video encoder and VAE weights.")

    out: dict[str, Path] = {}
    for role, entry in (
        ("diffusion", diff),
        ("encoder", enc),
        ("vae", vae),
        ("audio_vae", AUDIO_VAE),
    ):
        dest = entry_path(entry)
        if is_installed(entry):
            log(f"MiniMax H3 Video ({role}): already installed — {dest.name}")
            out[role] = dest
            continue
        log(f"Downloading MiniMax H3 Video ({role}, ~{entry.size_gb:.2f} GB): {entry.repo} / {entry.filename}")
        target_dir = settings.model_repo_dir(entry.repo)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            transfers.hf_hub_download(
                repo_id=entry.repo,
                filename=entry.filename,
                local_dir=str(target_dir),
            )
        )
        out[role] = path
    return out
