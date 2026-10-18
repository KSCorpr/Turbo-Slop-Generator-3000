"""Pinned MOSS-SFT DMD adapter, applied unmerged by sd.cpp at runtime."""
from __future__ import annotations

import json
import math
import struct
import re

from . import settings, transfers

REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
REVISION = "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f"
FILENAME = "lora_latest.safetensors"
FILE_SIZE = 1_189_176_800
SHA256 = "be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452"
LORA_NAME = "moss-image-sft-dmd/lora_latest"
STRENGTH = 0.7


def strip_tag(prompt):
    return re.sub(r"\s*<lora:" + re.escape(LORA_NAME) + r"(?:\.safetensors)?:[^>]+>", "", prompt)


def path():
    return settings.LORA_DIR / "moss-image-sft-dmd" / FILENAME


def ready():
    try:
        return path().is_file() and path().stat().st_size == FILE_SIZE
    except OSError:
        return False


def validate(filename):
    """Validate all 252 target linears without loading the 1.19 GB payload."""
    with filename.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated DMD adapter header.")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= 1024 * 1024:
            raise ValueError("Invalid DMD adapter header size.")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid DMD adapter header.")
    header.pop("__metadata__", None)
    shapes = {}
    dimensions = {"q": (3072, 3072), "k": (3072, 3072), "v": (3072, 3072),
                  "o": (3072, 3072), "ffn_up_gate": (3072, 16384), "ffn_down": (8192, 3072)}
    for layer in range(42):
        for name, (inputs, outputs) in dimensions.items():
            prefix = f"diffusion_model.layers.{layer}.proj.{name}"
            shapes[prefix + ".alpha"] = []
            shapes[prefix + ".lora_down.weight"] = [128, inputs]
            shapes[prefix + ".lora_up.weight"] = [outputs, 128]
    if header.keys() != shapes.keys():
        raise ValueError("DMD adapter targets do not match MOSS-Image SFT (756 tensors required).")
    ranges = []
    for name, shape in shapes.items():
        tensor = header[name]
        if not isinstance(tensor, dict) or tensor.get("dtype") != "F32" or tensor.get("shape") != shape:
            raise ValueError(f"Invalid DMD tensor type/shape: {name}")
        offsets = tensor.get("data_offsets", [])
        if (len(offsets) != 2 or not all(isinstance(v, int) and v >= 0 for v in offsets)
                or offsets[1] - offsets[0] != 4 * math.prod(shape)):
            raise ValueError(f"Invalid DMD tensor offsets: {name}")
        ranges.append(offsets)
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError("Invalid DMD adapter tensor layout.")
        end = stop
    if filename.stat().st_size != length + 8 + end:
        raise ValueError("Incomplete DMD adapter payload.")


def install(log=print):
    log("Downloading / verifying the MOSS-SFT DMD 8-step adapter (1.19 GB)…")
    transfers.hf_hub_download(repo_id=REPO, filename=FILENAME,
                              revision=REVISION, local_dir=str(path().parent))
    validate(path())
    if not ready():
        raise RuntimeError("The DMD adapter is incomplete. Retry the download to resume.")
    log("DMD adapter installed. In Images → MOSS-Image SFT, choose DMD fast (8 steps).")


def require():
    if not ready():
        raise ValueError("Download the MOSS SFT · DMD 8-step adapter in Models first.")
    validate(path())
