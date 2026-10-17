#!/usr/bin/env python3
"""Qwen-Image-Layered Q4_K_M, isolated and local, with explicit memory admission."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atelier import addons
from atelier.engine.layer_decompose import DEFAULTS, validate, export_layers

GIB = 1024**3
GGUF_BYTES = 13_244_770_944
OTHER_BYTES = 16_838_231_880
TEXT_ENCODER_BYTES = 16_584_414_544


def memory_plan(torch, available_ram, options):
    validate(options)
    # Includes CPU-held weights, a temporary GGUF-sized loading allowance and
    # runtime space. This is admission guidance, not an OS-level RAM quota.
    required = GGUF_BYTES*2 + OTHER_BYTES*(2 if options["memory"] == "cpu" else 1) + 3*GIB
    available = max(0, int(available_ram-4*GIB))
    ram = min(available, int(options["ram"]*GIB)) if options["ram"] else available
    if ram < required:
        raise RuntimeError(f"Qwen layers need a RAM budget of at least {required/GIB:.1f} GiB "
                           f"for this loading path; only {ram/GIB:.1f} GiB is available. Close other models/apps or raise the budget.")
    plan = {"mode": options["memory"], "ram_budget": ram, "ram_required": required,
            "gpu_budget": 0, "dtype": "float32"}
    if options["memory"] == "cpu":
        return plan
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select CPU explicitly or repair the selected GPU runtime.")
    free, total = torch.cuda.mem_get_info(0)
    budget = max(0, int(free-1.5*GIB))
    if options["vram"]:
        budget = min(budget, int(options["vram"]*GIB))
    minimum = max(GGUF_BYTES,TEXT_ENCODER_BYTES)+3*GIB if options["memory"] == "model" else 4*GIB
    if budget < minimum:
        raise RuntimeError(f"This Qwen memory mode needs a VRAM budget of at least {minimum/GIB:.1f} GiB. "
                           "Choose sequential CPU offload, lower the layer count/resolution, or free VRAM.")
    plan.update(gpu_budget=budget, gpu_total=total, gpu_name=torch.cuda.get_device_name(0),
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    return plan


def describe_plan(plan):
    memory = (f"RAM budget {plan['ram_budget']/GIB:.1f} GiB; "
              f"loading estimate {plan['ram_required']/GIB:.1f} GiB.")
    if plan["mode"] == "cpu":
        return "CPU only (FP32). "+memory
    mode = "Layer-by-layer RAM offload" if plan["mode"] == "sequential" else "Whole-model RAM offload"
    return f"{plan['gpu_name']} · {mode} · VRAM budget {plan['gpu_budget']/GIB:.1f} GiB. "+memory


def apply_leaf_offload(pipe, device):
    from diffusers.hooks import apply_group_offloading
    # Accelerate's sequential meta-parameter re-wrap loses GGUF quant_type in
    # this pinned runtime. Native Diffusers leaf hooks move original parameters
    # and retain Q4_K metadata (upstream diffusers issue #14691).
    # Offload every learned component, including the 16.6 GB vision/text
    # encoder. Streaming/pinned-RAM duplication is deliberately disabled.
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        apply_group_offloading(model, onload_device=device, offload_device="cpu",
                               offload_type="leaf_level", use_stream=False)


def load_pipeline(root, plan):
    import torch
    from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    dtype = getattr(torch, plan["dtype"])
    root = Path(root)
    config = json.loads((root / "model/transformer/config.json").read_text())
    if not config.get("use_layer3d_rope") or not config.get("use_additional_t_cond"):
        raise RuntimeError("This is not the Qwen-Image-Layered transformer configuration.")
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(root / "quantized" / addons.QWEN_LAYERED_GGUF), config=str(root / "model"),
        subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype, local_files_only=True)
    pipe = QwenImageLayeredPipeline.from_pretrained(str(root / "model"), transformer=transformer,
        torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    if plan["mode"] == "cpu":
        pipe.to("cpu")
    else:
        torch.cuda.set_per_process_memory_fraction(plan["gpu_budget"]/plan["gpu_total"], 0)
        # Never call pipe.to('cuda') before applying offload hooks.
        if plan["mode"] == "sequential":
            apply_leaf_offload(pipe, torch.device("cuda:0"))
        else:
            pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe


def generate(args, plan):
    import torch
    from PIL import Image, ImageOps
    started = time.monotonic()
    pipe = load_pipeline(args.addon_root, plan)
    with Image.open(args.input) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else ""

    def progress(pipeline, step, timestep, callback_kwargs):
        print(f"Qwen layers: step {step+1}/{args.steps}; {time.monotonic()-started:.1f}s including loading", flush=True)
        return callback_kwargs

    with torch.inference_mode():
        output = pipe(image=image, prompt=prompt or None, negative_prompt=" ", layers=args.layers,
            resolution=args.resolution, num_inference_steps=args.steps, true_cfg_scale=args.cfg,
            cfg_normalize=True, use_en_prompt=True, num_images_per_prompt=1,
            generator=torch.Generator(device="cpu").manual_seed(args.seed), callback_on_step_end=progress)
    images = output.images[0]
    if len(images) != args.layers:
        raise RuntimeError(f"Qwen returned {len(images)} layers instead of {args.layers}.")
    args.output.mkdir(parents=True, exist_ok=True)
    image.save(args.output / "source.png")
    export_layers(images, args.output, {"engine": "Qwen-Image-Layered", "quantization": "Q4_K_M",
        "revision": addons.QWEN_LAYERED_REVISION, "gguf_revision": addons.QWEN_LAYERED_GGUF_REVISION,
        "controls": {key: getattr(args, key) for key in DEFAULTS}, "description": prompt,
        "memory_plan": plan, "elapsed_seconds": round(time.monotonic()-started, 2),
        "source_size": list(image.size)})
    print(f"Exported {len(images)} RGBA layers, PSD, composite preview and ZIP. Compare with source.png.", flush=True)


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("--addon-root", type=Path)
    cli.add_argument("--input", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--prompt-file", type=Path)
    cli.add_argument("--probe", action="store_true")
    cli.add_argument("--check-imports", action="store_true")
    for name in ("layers", "steps", "seed", "resolution"):
        cli.add_argument("--"+name, type=int, default=DEFAULTS[name])
    for name in ("cfg", "vram", "ram"):
        cli.add_argument("--"+name, type=float, default=DEFAULTS[name])
    cli.add_argument("--memory", choices=["sequential", "model", "cpu"], default="sequential")
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    if args.check_imports:
        from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
        from diffusers.hooks import apply_group_offloading
        import gguf, accelerate
        print("Qwen-Image-Layered and GGUF imports ready.", flush=True)
    else:
        import torch
        import psutil
        options = {key: getattr(args, key) for key in DEFAULTS}
        plan = memory_plan(torch, psutil.virtual_memory().available, options)
        print("Memory admission passed. "+describe_plan(plan), flush=True)
        if not args.probe:
            if not args.addon_root or not args.input or not args.output:
                raise ValueError("Choose the installed model, source image and output folder.")
            try:
                generate(args, plan)
            except torch.OutOfMemoryError as exc:
                raise RuntimeError("Qwen exceeded the available memory. Try 2 layers at resolution 640 with sequential offload; close other models.") from exc
