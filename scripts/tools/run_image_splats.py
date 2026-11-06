#!/usr/bin/env python3
"""Run pinned upstream 3D models with local files and native PyTorch SDPA."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


def encoder_precision(capability):
    # BF16 emulation is not native BF16 support on Pascal / Turing.
    return "bf16" if capability[0] >= 8 else "fp32"


def configure_gpu():
    import torch
    import torch.nn.functional as F
    from torchvision.ops import deform_conv2d
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required. Select the generation GPU in System and install a compatible driver.")
    major, minor = torch.cuda.get_device_capability(0)
    if major < 6:
        raise RuntimeError("This runtime requires an NVIDIA Pascal or newer GPU.")
    # Exercise native kernels, catching missing GPU architectures / DLLs at setup.
    q = torch.randn(1, 1, 8, 32, device="cuda", dtype=torch.float16)
    F.scaled_dot_product_attention(q, q, q)
    x = torch.zeros(1, 1, 8, 8, device="cuda")
    deform_conv2d(x, torch.zeros(1, 18, 8, 8, device="cuda"), torch.ones(1, 1, 3, 3, device="cuda"), padding=(1, 1))
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}; sm_{major}{minor}; image encoders {encoder_precision((major, minor))}.", flush=True)
    return torch, torch.bfloat16 if major >= 8 else torch.float32


def release(torch, *modules):
    for module in modules:
        module.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def prepare_tripo_export(gaussian, torch):
    # The upstream PLY writer inverts sigmoid without clamping. Saturated
    # half-precision alpha can be exactly 1, yielding +inf in a PLY logit.
    gaussian._inverse_opacity_activation = lambda value: torch.logit(value.float().clamp(1e-6, 1 - 1e-6))
    return gaussian


def run_triposplat(request, torch, dtype):
    import triposplat as api
    paths = request["paths"]
    output = Path(request["output"])
    generator = torch.Generator(device="cuda").manual_seed(request["seed"])
    print("Removing background and preparing the object…", flush=True)
    bg = api.load_rmbg(paths["bg"], device="cuda", dtype=torch.float16)
    prepared = api.preprocess_image(request["source"], bg, erode_radius=1)
    prepared.save(output / "input.png")
    release(torch, bg)
    del bg
    print("Encoding the source image…", flush=True)
    dino = api.load_dinov3(paths["dino"], device="cuda", dtype=dtype)
    vae = api.load_vae_encoder(paths["flux"], device="cuda", dtype=dtype)
    condition = api.encode_image(prepared, dino, vae, generator=generator)
    release(torch, dino, vae)
    del dino, vae
    print("Generating the 3D latent…", flush=True)
    flow = api.load_flow_model(paths["denoiser"], device="cuda", dtype=torch.float16)
    sampled = api.sample_latent(flow, condition, steps=request["steps"], guidance_scale=request["cfg"],
                               shift=request["shift"], generator=generator, show_progress=True)
    del condition
    release(torch, flow)
    del flow
    print("Decoding Gaussian splats…", flush=True)
    decoder = api.load_decoder(paths["decoder"], device="cuda", dtype=torch.float16)
    gaussian = decoder.decode(sampled["latent"], num_gaussians=request["count"])
    prepare_tripo_export(gaussian, torch)
    gaussian.save_ply(str(output / "object.ply"))
    gaussian.save_splat(str(output / "object.splat"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triposplat",), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    torch, dtype = configure_gpu()
    from triposplat import load_decoder, load_flow_model  # noqa: F401
    if args.probe:
        print("Native SDPA, deformable convolution and model imports verified.", flush=True)
        return
    if not args.request:
        parser.error("--request is required for generation")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request["backend"] != args.backend:
        raise ValueError("Backend does not match the request.")
    with torch.inference_mode():
        run_triposplat(request, torch, dtype)
    print("Gaussian object exported.", flush=True)


if __name__ == "__main__":
    main()
