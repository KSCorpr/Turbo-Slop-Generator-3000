"""Automatic sd.cpp placement based on the selected model's installed weights.

The old hardware profile was global: on an 11 GB card it staged *every*
model in RAM, including small ones that fit completely in VRAM. This module
decides after the actual GGUF/safetensors paths have been resolved. It never
changes saved preferences or the weight selected in the Model Catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import hardware, registry, settings
from .engine import sdcpp

_GIB = 1024 ** 3


@dataclass(frozen=True)
class SystemPlan:
    gpu_index: int | None
    flags: dict[str, bool]
    params_backend: str
    auto_fit: bool
    max_vram: str
    stream_layers: bool
    placement: str
    image_gib: float | None
    encoder_gib: float | None
    available_gib: float | None


def _size_gib(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        size = Path(path).stat().st_size
        return size / _GIB if size > 0 else None
    except OSError:
        return None


def plan_for_model(model: registry.BaseModel, files: dict, prefs: dict,
                   sd_cli: Path | None, width: int = 1024,
                   height: int = 1024, has_reference: bool = False,
                   available_gib: float | None = None) -> SystemPlan:
    """Choose residency, GPU and compute budget for this model and image.

    Weight size is known from installed files, not a guessed parameter count.
    The reserve is deliberately conservative and grows with output size and
    reference images. sd.cpp's OOM retry remains a final safety net.
    """
    profile = hardware.biased_profile(prefs.get("hardware_bias", "balanced"),
                                      settings.generation_gpu_index(prefs))
    gpu = profile.gpu
    flags = profile.flags()
    flags.update(conv_direct_diffusion=False, conv_direct_vae=False)
    flags["vae_tiling"] = flags["vae_tiling"] or width * height > 1024 ** 2
    image_path = files.get("diffusion") or files.get("model_path")
    image_gib = _size_gib(image_path)
    vae_gib = _size_gib(files.get("vae")) or 0.0
    if image_gib is not None:
        image_gib += vae_gib
    encoder_gib = _size_gib(files.get("enc"))
    vision_gib = _size_gib(files.get("llm_vision")) if has_reference else None

    if gpu is None:
        return SystemPlan(None, flags, "", False, "", False,
                          "CPU (no GPU detected)", image_gib, encoder_gib, None)
    if gpu.is_apple:
        flags.update(offload_to_cpu=False, clip_on_cpu=False,
                     vae_on_cpu=False, vae_tiling=True)
        return SystemPlan(gpu.index, flags, "", False, "", False,
                          "Metal (unified memory)", image_gib, encoder_gib,
                          gpu.vram_gb)

    known = sdcpp.supported_options(sd_cli) if sd_cli else frozenset()
    free = (available_gib if available_gib is not None else
            hardware.free_vram_gb(gpu.index))
    if free <= 0:
        free = max(0.0, gpu.vram_gb - 1.0)  # nvidia-smi unavailable
    free = min(free, gpu.vram_gb)
    pixels = max(1, width * height) / (1024 ** 2)
    reserve = (1.5 + max(0.0, pixels - 1.0) * 0.8
               + (0.75 if has_reference else 0.0)
               + (vision_gib or 0.0)
               + (0.5 if model.family == "qwen21" else 0.0))
    fits = image_gib is not None and image_gib + reserve <= free
    # The encoder's weights are stored in RAM; its math runs on the main RTX.
    # This avoids wasting VRAM while also avoiding fp16 compute on a Pascal.
    flags.update(offload_to_cpu=False, clip_on_cpu=False, vae_on_cpu=False)
    if fits and "--params-backend" in known:
        params = "diffusion=cuda0,vae=cuda0,te=cpu"
        budget = (sdcpp.max_vram_arg("auto") if
                  "--max-vram" in known and image_gib + reserve + 0.5 > free
                  else "")
        return SystemPlan(gpu.index, flags, params, False, budget, False,
                          "image on GPU · text weights in RAM",
                          image_gib, encoder_gib, free)

    if fits and "--params-backend" not in known:
        # An older engine cannot stage only the encoder. Keep everything on
        # GPU if it fits; otherwise use its legacy RAM offload for the whole
        # model. Never pretend a nonexistent flag has been applied.
        if encoder_gib is not None and image_gib + encoder_gib + reserve > free:
            flags["offload_to_cpu"] = True
            placement = "weights in RAM (update sd.cpp for split residency)"
        else:
            placement = "image on GPU (older engine)"
        return SystemPlan(gpu.index, flags, "", False, "", False,
                          placement, image_gib, encoder_gib, free)

    budget = (sdcpp.max_vram_arg("auto")
              if "--max-vram" in known else "")
    if "--auto-fit" in known:
        # The current sd.cpp stages individual modules between GPU and RAM.
        # Explicit backend assignments would silently disable auto-fit.
        return SystemPlan(gpu.index, flags, "", True, budget, False,
                          "sd.cpp automatic placement (weight exceeds VRAM budget)",
                          image_gib, encoder_gib, free)
    if "--params-backend" in known:
        stream = "--stream-layers" in known
        return SystemPlan(gpu.index, flags,
                          "diffusion=cpu,vae=cuda0,te=cpu", False, budget,
                          stream, "image streamed from RAM" if stream else
                          "image weights in RAM (older engine)",
                          image_gib, encoder_gib, free)
    flags["offload_to_cpu"] = True
    return SystemPlan(gpu.index, flags, "", False, budget, False,
                      "weights in RAM (older engine)",
                      image_gib, encoder_gib, free)
