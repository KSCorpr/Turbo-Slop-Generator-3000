#!/usr/bin/env python3
"""Standalone VOSR 2.0, using pinned upstream neural modules and tiled inference."""
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atelier import vosr


def source_modules(source):
    """The installer removes Comfy registration and maps only ops to torch.nn."""
    name = "_turboslop_vosr"
    spec = importlib.util.spec_from_file_location(name, Path(source) / "__init__.py",
                                                 submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:
        raise RuntimeError("Install / repair VOSR 2.0 from Models first.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return [importlib.import_module(name + "." + part) for part in
            ("models.lightningdit", "models.dinov2", "models.qwenimage_vae2d", "tiled_vae", "inference")]


def require_runtime(dtype="fp16"):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("VOSR requires an NVIDIA CUDA GPU. Select it in System and repair VOSR if needed.")
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on the selected GPU. Use FP16.")
    device = torch.device("cuda:0")  # physical selection is inherited via CUDA_VISIBLE_DEVICES
    print("VOSR GPU: " + torch.cuda.get_device_name(0), flush=True)
    return torch, device, torch.float16 if dtype == "fp16" else torch.bfloat16


def load_weights(module, path, dtype):
    """Cast one mmap tensor at a time, then strictly assign to a meta-built model."""
    import torch
    from safetensors import safe_open
    path = Path(path)
    state = {}
    def add(key, tensor):
        for prefix in ("module.", "_orig_mod.", "ema_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in {"n_averaged", "step_count", "decay"}:
            return
        if key in state:
            raise RuntimeError("Duplicate VOSR checkpoint key: " + key)
        state[key] = tensor.to(dtype=dtype)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                add(key, handle.get_tensor(key))
    else:
        # No unrestricted pickle or torch.hub execution. The pinned Meta DINO
        # release is one flat state dict; retain its single original file.
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(loaded, dict) or not all(isinstance(t, torch.Tensor) for t in loaded.values()):
            raise RuntimeError("Unexpected DINOv2 checkpoint format.")
        for key, tensor in loaded.items():
            add(key, tensor)
    module.load_state_dict(state, strict=True, assign=True)
    module.eval().requires_grad_(False)
    del state
    gc.collect()


def make_dit(architecture, args):
    return architecture.LightningDiT(
        input_size=args["resolution"] // 8, patch_size=args["patch_size"],
        in_channels=32, out_channels=16, hidden_size=args["dim"], depth=args["depth"],
        num_heads=args["num_heads"], mlp_ratio=args["mlp_ratio"], z_dims=args["enc_dim"],
        encdim_ratio=args["encdim_ratio"], auxiliary_time_cond=args["auxiliary_time_cond"],
        use_qknorm=args["use_qknorm"], use_swiglu=args["use_swiglu"],
        use_rope=args["use_rope"], use_rmsnorm=args["use_rmsnorm"],
        num_fused_layers=len(args["layer_dinov2b_list"]))


class Model:
    def __init__(self, modules, root, device, dtype, offload=True):
        import torch
        dit_arch, dino_arch, vae_arch, self.tiled, _ = modules
        self.device, self.dtype, self.offload = device, dtype, offload
        self.active = None
        self.args = vosr.load_args(root)
        config = json.loads((root / "Qwen-Image-vae-2d/config.json").read_text(encoding="utf-8"))
        # Construction on meta avoids allocating and randomly initializing 7 GB
        # before loading another complete checkpoint copy in system RAM.
        with torch.device("meta"):
            self.dit = make_dit(dit_arch, self.args)
            self.vision = dino_arch.build_dinov2_vitl14()
            self.vae = vae_arch.AutoencoderKLQwenImage2D.from_config_dict(config)
        for label, model, path, precision in (
            ("DiT", self.dit, "VOSR2/checkpoints/ema_model.safetensors", dtype),
            ("DINOv2", self.vision, "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth", dtype),
            ("VAE", self.vae, "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors", torch.float32)):
            print("Loading VOSR " + label + "...", flush=True)
            load_weights(model, root / path, precision)
        if not offload:
            for model in (self.dit, self.vision, self.vae):
                model.to(device)

    def activate(self, model):
        import torch
        if self.active is not model:
            label = "VAE" if model is self.vae else "DINOv2" if model is self.vision else "DiT"
            print("VOSR stage: " + label, flush=True)
            if self.offload and self.active is not None:
                self.active.cpu()
                torch.cuda.empty_cache()
            model.to(self.device)
            self.active = model
        return model

    def vision_features(self, image):
        import torch
        import torch.nn.functional as F
        model = self.activate(self.vision)
        size = self.args.get("dinov2_size", 448)
        x = F.interpolate(image.to(self.device), size=(size, size), mode="bicubic").clamp(0, 1)
        mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        features = model.forward_intermediate_layer(((x - mean) / std).to(self.dtype),
                                                   self.args["layer_dinov2b_list"][0])
        # The upstream caches vision features for all DiT tiles. Keep the cache
        # in RAM so a large image does not retain them all in VRAM.
        return [features.cpu()]

    def encode_tiled(self, image, tile_size, overlap):
        return self.tiled.encode_dispatch(self.activate(self.vae), image.to(self.device).float(), tile_size, overlap)

    def decode_tiled(self, latent, mean, std, tile_size, overlap):
        return self.tiled.decode_dispatch(self.activate(self.vae), latent.to(self.device).float(),
                                         mean, std, tile_size, overlap)

    def dit_velocity(self, image, t, next_t, features):
        import torch
        model = self.activate(self.dit)
        x = image.to(device=self.device, dtype=self.dtype)
        current = torch.full((x.shape[0],), t, device=self.device, dtype=self.dtype)
        following = torch.full_like(current, next_t)
        return model.forward_flexible(x, current, following,
                                      [feature.to(device=self.device, dtype=self.dtype) for feature in features])

    def denoise_one_step(self, latent, noise, features):
        import torch
        return noise - self.dit_velocity(torch.cat([latent, noise], dim=1), 1.0, 0.0, features)


def generate(args):
    import numpy as np
    from PIL import Image, ImageOps, PngImagePlugin
    from atelier.engine.vosr import validate_controls, target_size
    validate_controls(args.upscale, args.dtype, args.color, args.tile_size, args.tile_overlap,
                      args.vae_tile_size, args.vae_tile_overlap, args.seed)
    if not vosr.weights_ready():
        raise RuntimeError("VOSR weights are missing or incomplete. Install / repair VOSR 2.0 in Models.")
    # Read and close the upload before model loading. EXIF orientation and alpha
    # are preserved. Alpha is resized conventionally; the neural model sees RGB.
    with Image.open(args.input) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA" if
                "A" in original.getbands() or "transparency" in original.info else "RGB")
    width, height = target_size(image.size, args.upscale)
    torch, device, dtype = require_runtime(args.dtype)
    free, _ = torch.cuda.mem_get_info(device)
    if free < 3.2 * 1024**3:
        raise RuntimeError("VOSR needs at least 3.2 GiB of free VRAM to load its DiT. Close other GPU tasks.")
    print(f"Target: {width} x {height}; DiT tiles {args.tile_size}, VAE tiles {args.vae_tile_size}; "
          f"stage offload: {not args.keep_on_gpu}.", flush=True)
    modules = source_modules(args.addon_root / "source")
    try:
        model = Model(modules, vosr.model_dir(), device, dtype, offload=not args.keep_on_gpu)
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0)[None]
        with torch.inference_mode():
            result = modules[-1].run_vosr2(model, pixels, args.upscale, args.seed, args.color,
                                         args.tile_size, args.tile_overlap,
                                         args.vae_tile_size, args.vae_tile_overlap)
        if result.shape != (1, height, width, 3) or not torch.isfinite(result).all().item():
            raise RuntimeError("VOSR produced invalid pixels or dimensions; no output was exported.")
        array = (result[0].cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        output = Image.fromarray(array)
        if image.mode == "RGBA":
            output.putalpha(image.getchannel("A").resize((width, height), Image.Resampling.LANCZOS))
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Software", "TurboSlop / VOSR 2.0")
        metadata.add_text("VOSR", json.dumps({key: str(value) if isinstance(value, Path) else value
                                            for key, value in vars(args).items()}))
        temporary = args.output.with_name(args.output.name + ".part")
        try:
            output.save(temporary, format="PNG", pnginfo=metadata)
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print("VOSR finished: " + str(args.output), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("VOSR ran out of VRAM. Use stage offload and smaller VAE / DiT tiles "
                           "(256), or reduce the output factor. No partial image was exported.") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upscale", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--color", choices=("wavelet", "adain", "none"), default="wavelet")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--vae-tile-size", type=int, default=512)
    parser.add_argument("--vae-tile-overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-on-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.probe:
        torch, device, dtype = require_runtime(args.dtype)
        source_modules(args.addon_root / "source")
        with torch.inference_mode():
            x = torch.randn((1, 2, 32, 32), dtype=dtype, device=device)
            y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
            if not torch.isfinite(y).all().item():
                raise RuntimeError("VOSR SDPA probe failed on the selected GPU.")
        print(f"VOSR runtime ready: Torch {torch.__version__}, CUDA {torch.version.cuda}, {args.dtype}.", flush=True)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required")
        generate(args)


if __name__ == "__main__":
    main()
