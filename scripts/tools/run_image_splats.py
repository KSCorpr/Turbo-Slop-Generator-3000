#!/usr/bin/env python3
"""Run pinned upstream 3D models with local files and native PyTorch SDPA."""
# Idol VAE token packing is adapted from speridlabs/idol-sdk's conditioning.py
# under Apache-2.0; see LICENSES/Idol-Apache-2.0.txt. Precision selection and
# explicit local checkpoint paths are the TurboSlop adaptations.
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
    # Match Idol's export epsilon; the generated opacity / SPLAT stay intact.
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


def idol_encoder_class(request, torch, dtype):
    """Use the SDK's token processing with explicit local encoder directories.

    Only the VAE input cast is generalized: native BF16 on Ampere+, FP32 on
    Pascal/Turing. Packing, batch normalization and token padding stay upstream.
    """
    from idol.conditioning import ImageEncoder

    class LocalImageEncoder(ImageEncoder):
        def __init__(self, _dino_repo, _flux_repo, dinov3_dtype="bf16", device="cuda"):
            from diffusers.models import AutoencoderKLFlux2
            from transformers import AutoModel
            self.device = torch.device(device)
            self.dinov3 = AutoModel.from_pretrained(str(Path(request["paths"]["dino"]).parent),
                                                     torch_dtype=dtype, local_files_only=True)
            self.dinov3 = self.dinov3.eval().requires_grad_(False).to(self.device)
            self.flux2_vae = AutoencoderKLFlux2.from_pretrained(
                str(Path(request["paths"]["flux"]).parents[1]), subfolder="vae",
                torch_dtype=dtype, local_files_only=True)
            self.flux2_vae = self.flux2_vae.eval().requires_grad_(False).to(self.device)

        def _flux2_tokens(self, images):
            vae = self.flux2_vae
            latents = vae.encode(images.to(next(vae.parameters()).dtype) * 2 - 1).latent_dist.sample()
            batch, channels, height, width = latents.shape
            latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
            latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
            tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
            padding = torch.zeros(batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
            return torch.cat([padding, tokens], dim=1)

    return LocalImageEncoder


def apply_idol_turbo(pipeline, filename):
    """Phase-LoRA adapter for akhaliq/idol-turbo, pinned v5 checkpoint.

    Adapted from turbo.py at HF revision 45cdd41f1f4d6d70496daa7d619f57ad79307a37
    (Apache-2.0). Local-only loading, strict adapter shapes, resident experts,
    and exception-safe restoration of the SDK's attention settings are ours.
    """
    import numpy as np
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from tensordict import TensorDict
    import idol.pipeline as api
    import idol.models.layers as layers

    class PhaseLinear(nn.Module):
        def __init__(self, base, tensors):
            super().__init__()
            self.base = base
            self.active = "slow"
            for key, tensor in tensors.items():
                self.register_buffer(key, tensor.to(base.weight.device))

        def forward(self, x):
            out = self.base(x)
            a = getattr(self, self.active + "_A")
            b = getattr(self, self.active + "_B")
            return out + (x.to(a.dtype) @ a.t() @ b.t()).to(out.dtype) * 4.0  # alpha 128 / rank 32

    targets = []
    for group in ("noise_refiner", "context_refiner", "blocks"):
        for index, block in enumerate(getattr(pipeline.denoiser, group)):
            for suffix, parent, key in (("attn.qkv", block.attn, "qkv"),
                                         ("attn.out", block.attn, "out"),
                                         ("mlp.mlp.0", block.mlp.mlp, "0"),
                                         ("mlp.mlp.2", block.mlp.mlp, "2")):
                base = getattr(parent, key)
                if not isinstance(base, nn.Linear):
                    raise ValueError("Idol Turbo requires the original, unpatched Idol denoiser.")
                targets.append((f"{group}.{index}.{suffix}", parent, key, base))
    state = load_file(str(filename), device="cpu")
    expected = {}
    for name, _, _, base in targets:
        for phase in ("slow", "fast"):
            expected[name + "." + phase + "_A"] = (32, base.in_features)
            expected[name + "." + phase + "_B"] = (base.out_features, 32)
    if set(state) != set(expected):
        raise ValueError("Idol Turbo adapter tensor names do not match the installed denoiser.")
    for key, shape in expected.items():
        value = state[key]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid Idol Turbo adapter tensor: {key}.")
    experts = []
    for name, parent, key, base in targets:
        tensors = {suffix: state[name + "." + suffix] for suffix in ("slow_A", "slow_B", "fast_A", "fast_B")}
        expert = PhaseLinear(base, tensors)
        setattr(parent, key, expert)
        experts.append(expert)
    original_sampler = api.sample_flow

    @torch.no_grad()
    def turbo_sample_flow(model, noise, cond, neg_cond, steps=5, rescale_t=3.0,
                          cfg_strength=0.0, verbose=True, batch_cfg=False):
        if model is not pipeline.denoiser:
            return original_sampler(model, noise, cond=cond, neg_cond=neg_cond, steps=steps,
                                    rescale_t=rescale_t, cfg_strength=cfg_strength,
                                    verbose=verbose, batch_cfg=batch_cfg)
        if steps not in range(4, 9) or cfg_strength != 0 or rescale_t != 3:
            raise ValueError("Idol Turbo uses 4–8 steps, guidance 0 and time shift 3.")
        from tqdm import tqdm
        previous_backend = layers.ATTN_BACKEND
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            layers.ATTN_BACKEND = "sdpa"
            torch.backends.cuda.matmul.allow_tf32 = True
            sequence = np.linspace(1, 0, steps + 1)
            sequence = rescale_t * sequence / (1 + (rescale_t - 1) * sequence)
            sample = noise
            for index in tqdm(range(steps), desc="Idol Turbo sampling", disable=not verbose):
                t, previous_t = sequence[index], sequence[index + 1]
                for expert in experts:
                    expert.active = "slow" if t >= 0.5 else "fast"
                timestep = torch.tensor([1000 * t] * sample.shape[0], device=sample.device, dtype=torch.float32)
                velocity = model(sample, timestep, cond)
                sample = TensorDict({key: sample[key] - velocity[key] * (t - previous_t)
                                     for key in sample.keys()}, batch_size=sample.shape[0], device=sample.device)
            return sample
        finally:
            layers.ATTN_BACKEND = previous_backend
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    api.sample_flow = turbo_sample_flow
    return pipeline


def run_idol(request, torch, dtype):
    import idol.pipeline as api
    # The parent has verified/downloaded all files. Inference never contacts HF.
    api.ImageEncoder = idol_encoder_class(request, torch, dtype)
    print("Loading Idol and its local image encoders (compilation disabled)…", flush=True)
    pipeline = api.IdolPipeline(str(Path(request["paths"]["config"]).parent), device="cuda",
                               dinov3_dtype="bf16" if dtype == torch.bfloat16 else "fp32", optimize=False)
    if request["backend"] == "idol-turbo":
        apply_idol_turbo(pipeline, request["paths"]["turbo_lora"])
        print("Idol Turbo: phase experts active; guidance-free, time shift 3.", flush=True)
    output = pipeline(request["source"], mode=request["mode"], seed=request["seed"], steps=request["steps"],
                      cfg_strength=request["cfg"], rescale_t=request["shift"],
                      num_gaussians=request["count"], verbose=True)
    folder = Path(request["output"])
    output.gaussian.save_ply(str(folder / "object.ply"))
    output.preprocessed_image.save(folder / "input.png")
    (folder / "camera.json").write_text(json.dumps(output.camera, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triposplat", "idol", "idol-turbo"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    torch, dtype = configure_gpu()
    if args.backend == "triposplat":
        from triposplat import load_decoder, load_flow_model  # noqa: F401
    else:
        from idol.pipeline import IdolPipeline  # noqa: F401
        from diffusers.models import AutoencoderKLFlux2  # noqa: F401
        from transformers import DINOv3ViTModel  # noqa: F401
    if args.probe:
        print("Native SDPA, deformable convolution and model imports verified.", flush=True)
        return
    if not args.request:
        parser.error("--request is required for generation")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request["backend"] != args.backend:
        raise ValueError("Backend does not match the request.")
    with torch.inference_mode():
        (run_triposplat if args.backend == "triposplat" else run_idol)(request, torch, dtype)
    print("Gaussian object exported.", flush=True)


if __name__ == "__main__":
    main()
