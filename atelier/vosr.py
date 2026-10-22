"""Pinned VOSR 2.0 artifacts and standalone source adaptation (no Torch import)."""
from __future__ import annotations

import json
from pathlib import Path

from . import addons, transfers

SOURCE_REPO = "ylchen333/ComfyUI-VOSR2"
SOURCE_COMMIT = "6a21810b3f4f1a0ffc477c55330db0ac03c671f0"
MODEL_REPO = "CSWRY/VOSR"
MODEL_REVISION = "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3"
# Exactly one upstream copy of each component. DINO is loaded with weights_only,
# not converted into another 1.2 GB checkpoint. The 2D VAE is specific to VOSR.
FILES = {
    "VOSR2/args.json": 816,
    "VOSR2/checkpoints/ema_model.safetensors": 5_576_385_424,
    "Qwen-Image-vae-2d/config.json": 811,
    "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors": 178_418_892,
    "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth": 1_217_586_395,
}
REQUIRED_ARGS = {
    "ae_type": "qwen", "dim": 1536, "depth": 36, "num_heads": 24,
    "patch_size": 2, "enc_type": "dinov2l", "enc_dim": 1024,
    "layer_dinov2b_list": [17], "auxiliary_time_cond": False,
    "distill_type": "onestep",
}
DIT_ARG_KEYS = ("mlp_ratio", "use_qknorm", "use_swiglu", "use_rope",
                "use_rmsnorm", "encdim_ratio", "resolution")


def model_dir():
    return addons.root("vosr") / "models"


def weights_ready():
    root = model_dir()
    return all((root / name).is_file() and (root / name).stat().st_size == size
               for name, size in FILES.items())


def ready():
    try:
        marker = json.loads((addons.root("vosr") / "installed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (addons.ready("vosr") and marker.get("commit") == SOURCE_COMMIT
            and marker.get("revision") == MODEL_REVISION
            and marker.get("adapter") == 1
            and (addons.root("vosr") / "source/progress.py").is_file() and weights_ready())


def download_models(log=print):
    for filename, size in FILES.items():
        log("VOSR 2.0: " + filename)
        path = Path(transfers.hf_hub_download(repo_id=MODEL_REPO, filename=filename,
                    revision=MODEL_REVISION, local_dir=str(model_dir())))
        if path.stat().st_size != size:
            raise RuntimeError("Incomplete VOSR model: " + filename)


def load_args(root):
    args = json.loads((Path(root) / "VOSR2/args.json").read_text(encoding="utf-8"))
    for key, expected in REQUIRED_ARGS.items():
        if args.get(key) != expected:
            raise ValueError(f"Incompatible VOSR 2.0 config: {key} must be {expected!r}.")
    missing = [key for key in DIT_ARG_KEYS if key not in args]
    if missing:
        raise ValueError("Incomplete VOSR 2.0 config: " + ", ".join(missing))
    return args


def adapt_source(source):
    """Use actual PyTorch layers in this standalone process; never fake ComfyUI.

    The upstream explicitly documents these ops as equivalent torch.nn layers.
    Meta-device construction + strict assign loading replaces Comfy's lazy init.
    Check expected source text so an upstream API change cannot pass unnoticed.
    Neural operations, normalization, color correction and tiling are preserved.
    """
    source = Path(source)
    for name in ("lightningdit", "dinov2", "qwenimage_vae2d", "swiglu_ffn"):
        path = source / "models" / (name + ".py")
        text = path.read_text(encoding="utf-8")
        for old, new in (("import comfy.ops\n", ""),
                         ("ops = comfy.ops.disable_weight_init", "ops = nn")):
            if text.count(old) != 1:
                raise RuntimeError("Unexpected pinned VOSR source: " + str(path))
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    path = source / "inference.py"
    text = path.read_text(encoding="utf-8")
    for old, new in (("import comfy.utils", "from .progress import ProgressBar"),
                     ("comfy.utils.ProgressBar", "ProgressBar"),
                     ("model.dit_patcher.load_device", "model.device")):
        if old not in text:
            raise RuntimeError("Unexpected pinned VOSR inference source.")
        text = text.replace(old, new)
    # Reflect padding in upstream rejects very narrow images. Replicate padding
    # has no such dimensional constraint and is used only for the added border.
    text = text.replace('mode="reflect"', 'mode="replicate"')
    path.write_text(text, encoding="utf-8")
    # Import this package without the Comfy node registration side effects.
    (source / "__init__.py").write_text('"""Standalone VOSR 2.0 inference package."""\n', encoding="utf-8")
    (source / "progress.py").write_text(
        'class ProgressBar:\n'
        '    def __init__(self, total):\n'
        '        self.total, self.current = total, 0\n'
        '    def update_absolute(self, value):\n'
        '        self.current = value\n'
        '        print(f"VOSR DiT tile {value}/{self.total}", flush=True)\n', encoding="utf-8")
