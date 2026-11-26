"""Pinned Kandinsky ImageSR 2.0 artifacts and download helpers (no Torch import)."""
from __future__ import annotations

import json
from pathlib import Path

from . import addons, transfers

SOURCE_REPO = "kandinskylab/kandinsky-image-sr"
SOURCE_COMMIT = "6c2405796cf7576b0cffff27acd103d667b91ca8"
MODEL_REPO = "kandinskylab/Kandinsky-ImageSR-2.0"
MODEL_REVISION = "7d5b5ed7def0b45b98f398c0b295a6b1658863db"

# Exact Diffusers bundle files (10.42 GB total: 5B baked Kandinsky 5 transformer
# + Apache-2.0 FLUX.1-schnell AutoencoderKL). Public MIT + Apache-2.0 release.
FILES: dict[str, int] = {
    "model_index.json": 455,
    "transformer/config.json": 628,
    "transformer/diffusion_pytorch_model.safetensors.index.json": 90_362,
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": 4_988_839_016,
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": 4_993_423_840,
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": 268_045_424,
    "vae/config.json": 824,
    "vae/diffusion_pytorch_model.safetensors": 167_666_902,
}

# Calibrated flow timesteps from kandinsky_image_sr.pipeline.CREATIVITY_TIMESTEPS.
# Creativity level 0..8; level 4 is the default per factor (x2=0.15, x4=0.244).
CREATIVITY_TIMESTEPS: dict[int, tuple[float, ...]] = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}
DEFAULT_CREATIVITY = 4


def model_dir() -> Path:
    return addons.root("kandinsky-sr") / "models"


def weights_ready() -> bool:
    root = model_dir()
    return all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in FILES.items()
    )


def ready() -> bool:
    try:
        marker = json.loads(
            (addons.root("kandinsky-sr") / "installed.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return (
        addons.ready("kandinsky-sr")
        and marker.get("commit") == SOURCE_COMMIT
        and marker.get("revision") == MODEL_REVISION
        and (addons.root("kandinsky-sr") / "source" / "src" / "kandinsky_image_sr" / "pipeline.py").is_file()
        and weights_ready()
    )


def download_models(log=print) -> None:
    for filename, size in FILES.items():
        log("Kandinsky ImageSR 2.0: " + filename)
        path = Path(
            transfers.hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                revision=MODEL_REVISION,
                local_dir=str(model_dir()),
            )
        )
        if path.stat().st_size != size:
            raise RuntimeError("Incomplete Kandinsky ImageSR 2.0 file: " + filename)
