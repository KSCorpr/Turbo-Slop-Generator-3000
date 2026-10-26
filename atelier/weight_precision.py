"""Storage precision advice; file format is not a promise of compute kernels.

No Torch import or model download is needed to display the catalog. CUDA
capability reported by the driver takes precedence over architecture names.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
import re

from . import hardware, quant, settings


def storage_type(filename: str, label: str = "") -> str:
    name = Path(filename).name.upper()
    for pattern, value in (
        (r"W4A8", "INT4 / W4A8"), (r"INT8", "INT8"),
        (r"BF16|BFLOAT16", "BF16"), (r"FP32|FLOAT32|(?<![A-Z0-9])F32(?![A-Z0-9])", "FP32"),
        (r"FP16|FLOAT16|(?<![A-Z0-9])F16(?![A-Z0-9])", "FP16"),
        (r"FP8|MXFP8", "FP8"), (r"NVFP4|FP4", "FP4"),
    ):
        if re.search(pattern, name):
            return value
    if filename.lower().endswith(".gguf"):
        return quant.find_quant(name) or "GGUF (precision unspecified)"
    # Explicit catalog labels can describe files named simply model.safetensors.
    if label:
        return storage_type(label)
    return "unspecified"


def profile(prefs=None):
    return hardware.auto_profile(settings.generation_gpu_index(
        settings.load_prefs() if prefs is None else prefs))


def native_bf16(gpu) -> bool | None:
    if gpu is None or gpu.is_apple:
        return None
    if gpu.compute_cap:
        try:
            major, minor = map(int, gpu.compute_cap.split(".", 1))
            return (major, minor) >= (8, 0)
        except ValueError:
            pass
    if gpu.arch in {"ampere", "ada", "hopper", "blackwell"}:
        return True
    if gpu.arch in {"maxwell", "pascal", "volta", "turing"}:
        return False
    return None


def gpu_advice(dtype: str, prof) -> str:
    gpu = prof.gpu
    if gpu is None:
        return "GPU not detected; CPU / engine support must be checked"
    if gpu.is_apple:
        return "Metal backend dependent; shared memory budget"
    if dtype == "BF16":
        support = native_bf16(gpu)
        if support is True:
            return "Native BF16 hardware; 16-bit memory cost"
        if support is False:
            return "No native BF16; sd.cpp conversion possible, prefer lighter weights"
        return "BF16 hardware support unknown; check the backend"
    if dtype == "FP16":
        if gpu.arch in {"maxwell", "pascal"} and not gpu.tensor_cores:
            return "Loadable with sd.cpp; FP16 speed depends on the GPU / kernel"
        return "16-bit option; backend decides arithmetic and accumulation"
    if dtype == "FP32":
        return "32-bit option; twice the weight memory of FP16 / BF16"
    if dtype in {"INT8", "INT4 / W4A8"}:
        return "Compact weights; native integer compute is not guaranteed"
    if dtype in {"FP8", "FP4"}:
        return "Requires an explicitly supported model format and backend"
    return "Quantized weights; backend handles dequantization / compute"


def describe(filename: str, prof, *, label: str = "", size=None,
             role="diffusion", recommended=False) -> str:
    dtype = storage_type(filename, label)
    fmt = "GGUF" if filename.lower().endswith(".gguf") else "Safetensors"
    bits = [dtype, fmt]
    if recommended:
        bits.append("current target")
    if dtype == "BF16" and native_bf16(prof.gpu) is False:
        bits.append("no native BF16 · sd.cpp conversion")
    if size is not None:
        # A weight-file lower bound, deliberately not an inference VRAM estimate.
        gib = size / 2**30
        if prof.gpu and (role != "text_encoder" or prof.gpu.is_apple):
            if gib >= prof.gpu.vram_gb:
                bits.append("weights exceed GPU memory · offload needed")
            elif gib >= prof.gpu.vram_gb * .7:
                bits.append("limited GPU headroom · offload likely")
        elif role == "text_encoder" and prof.ram_gb and gib >= prof.ram_gb * .7:
            bits.append("large system RAM requirement")
    return " · ".join(bits)


def suggest(comp, candidates, prof) -> str | None:
    """Memory heuristic among approved, sized files; no quality/speed claim.

    candidates: (filename, byte size, catalog label). Reserve 30% for working
    memory; the generation planner still makes the actual placement decision.
    Encoders keep their existing RAM-aware target rather than assuming that
    both networks will fit on the GPU simultaneously.
    """
    if not prof.gpu:
        return None
    if prof.gpu.arch == "unknown" and native_bf16(prof.gpu) is None:
        return None
    if comp.role == "text_encoder":
        return next((name for name, _, _ in candidates if is_target(comp, name)), None)
    budget = prof.gpu.vram_gb * 2**30 * .7
    sized = [(name, size, storage_type(name, label)) for name, size, label in candidates
             if size is not None and size > 0]
    fits = [(name, size, dtype) for name, size, dtype in sized if size <= budget]
    full = [(name, size) for name, size, dtype in fits
            if dtype == "FP16" or (dtype == "BF16" and native_bf16(prof.gpu) is True)]
    if full:
        return min(full, key=lambda row: row[1])[0]
    integer = [(name, size) for name, size, dtype in fits if dtype == "INT8"]
    if integer:
        return min(integer, key=lambda row: row[1])[0]
    target = next((name for name, _, _ in fits if is_target(comp, name)), None)
    if target:
        return target
    quants = [(name, size) for name, size, dtype in fits if dtype.startswith(("Q", "IQ"))]
    if quants:
        return max(quants, key=lambda row: row[1])[0]
    return None


def is_target(comp, filename: str) -> bool:
    return fnmatch.fnmatch(filename, comp.requested()) or fnmatch.fnmatch(
        Path(filename).name, comp.requested())


def model_advice(model, prefs=None, variants=None) -> str:
    """Small capability table, with only catalog-approved downloadable types."""
    prof = profile(prefs)
    device = prof.gpu.label() if prof.gpu else "No GPU detected"
    available = {}
    has_gguf = False
    for comp in model.components:
        if comp.role not in {"diffusion", "text_encoder"}:
            continue
        files = [(comp.requested(), "")]
        files += [(name, "") for name in comp.selectable]
        files += [(v.template, v.label) for v in comp.alternatives]
        if variants and comp.role in variants:
            files += [(name, "") for _, name in variants[comp.role]]
        for name, label in files:
            dtype = storage_type(name, label)
            if name.lower().endswith(".gguf"):
                has_gguf = True
            if dtype != "unspecified":
                available.setdefault(dtype, set()).add(
                    "image" if comp.role == "diffusion" else "encoder")
    rows = [f"**GPU & weight precision — {device}**", "",
            "GGUF / Safetensors are file formats. INT8, BF16, FP16 and FP32 describe "
            "weight storage; actual arithmetic depends on the engine and kernels.", "",
            "| Weights | Advice for this device | Available for this model |",
            "|---|---|---|"]
    for dtype in ("INT8", "BF16", "FP16", "FP32"):
        roles = ", ".join(sorted(available.get(dtype, ())))
        availability = roles or ("Check GGUF files via Load weights and sizes"
                                 if has_gguf and dtype in {"FP16", "FP32", "BF16"}
                                 else "No approved file in this catalog")
        rows.append(f"| {dtype} | {gpu_advice(dtype, prof)} | {availability} |")
    ggufs = sorted(k for k in available if k.startswith(("Q", "IQ")))
    if has_gguf:
        rows.append("| GGUF quantizations | Compact alternative; choose by memory budget | "
                    + (", ".join(ggufs) or "Load weights and sizes") + " |")
    rows += ["", "**Current weight targets:** " + "; ".join(
        f"{c.role.replace('_', ' ')}: {storage_type(c.requested())}"
        for c in model.components if c.role in {"diffusion", "text_encoder"}) + ". "
        "Load sizes for a GPU memory suggestion: 16-bit weights when there is headroom, "
        "otherwise INT8 or GGUF. This is a memory heuristic, not a speed / quality benchmark. "
        "Explicit selections stay selected.",
        "File sizes are download sizes, not total VRAM use: activations, VAE, encoder "
        "and buffers need additional memory. No automatic conversion or extra copy is created."]
    return "\n".join(rows)
