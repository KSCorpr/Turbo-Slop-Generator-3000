"""Optional native sd.cpp SageAttention; no Python attention package required."""
from __future__ import annotations

import json
from pathlib import Path

from . import hardware, settings


def gpu_supported(gpu):
    if gpu is None or gpu.is_apple:
        return False
    if gpu.compute_cap:
        try:
            major, minor = map(int, gpu.compute_cap.split('.'))
            return (major, minor) >= (8, 0)
        except (ValueError, TypeError):
            pass
    return gpu.arch in {'ampere', 'ada', 'hopper', 'blackwell'}


def availability(cli, gpu_index=None, *, all_gpus=False):
    """GPU/CLI eligibility, not a benchmark or proof of compiled CUDA kernels."""
    from .engine import sdcpp
    gpu = hardware.auto_profile(gpu_index).gpu
    if not gpu_supported(gpu):
        name = gpu.name if gpu else 'No NVIDIA GPU detected'
        return False, f'{name}: SageAttention needs NVIDIA compute capability 8.0+ (RTX 30 or newer).'
    if all_gpus and any(not gpu_supported(g) for g in hardware.detect_gpus()):
        return False, 'Mixed-GPU auto-fit: an incompatible visible GPU may run diffusion. Keeping standard attention.'
    if cli is None or '--sage-attn' not in sdcpp.supported_options(cli):
        return False, 'The installed engine does not expose --sage-attn. Run update.bat, then refresh support.'
    # A CPU/Vulkan build can list the common CLI flag without CUDA kernels.
    candidates = {Path(cli).parent / 'engine-manifest.json'}
    if Path(cli).resolve().is_relative_to(settings.BIN_DIR.resolve()):
        candidates.add(settings.BIN_DIR / 'engine-manifest.json')
    for path in candidates:
        try:
            archive = json.loads(path.read_text(encoding='utf-8')).get('archive', '').lower()
        except (OSError, ValueError, AttributeError):
            continue
        if any(part in archive for part in ('vulkan', 'metal', 'opencl', 'rocm', '-cpu', '-avx')):
            return False, 'SageAttention requires a CUDA engine. Install the CUDA variant and refresh support.'
    return True, (f'{gpu.name}: GPU and CLI support detected. Requires a CUDA build with SageAttention kernels. '
                  'Changes numerical results; compare quality and total generation time at a fixed seed.')


def requested(cli, gpu_index=None, prefs=None, log=None, *, all_gpus=False):
    prefs = settings.load_prefs() if prefs is None else prefs
    if not prefs.get('sage_attention', False):
        return False
    supported, reason = availability(cli, gpu_index, all_gpus=all_gpus)
    if log:
        log('SageAttention: ' + ('enabled (--sage-attn).' if supported else 'inactive. ' + reason))
    return supported
