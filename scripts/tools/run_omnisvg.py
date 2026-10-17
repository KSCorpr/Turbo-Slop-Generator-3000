#!/usr/bin/env python3
"""OmniSVG 1.1 4B: strict local checkpoint loading, bounded memory and serial variants."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

try:
    from .run_svg import export, parse_svg
except ImportError:
    from run_svg import export, parse_svg

GIB = 1024**3
WEIGHT_BYTES = 7_694_002_862
SYSTEM_PROMPT = ("You are an expert SVG code generator. \n"
    "Generate precise, valid SVG path commands that accurately represent the described scene or object.\n"
    "Focus on capturing key shapes, spatial relationships, and visual composition.")


def validate_options(options):
    limits = {"tokens": (256, 4096), "seed": (0, 2**32-1), "variants": (1, 4),
              "temperature": (0.05, 1.5), "top_p": (0.05, 1), "top_k": (1, 200),
              "repetition": (1, 1.5), "vram": (0, 128), "ram": (0, 1024), "timeout": (30, 1800)}
    for key, (lower, upper) in limits.items():
        try:
            value = float(getattr(options, key))
        except (TypeError, ValueError):
            raise ValueError(f"OmniSVG {key} must be a number.") from None
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"OmniSVG {key} must be between {lower} and {upper}.")
        if key in {"tokens", "seed", "variants", "top_k", "timeout"} and value != int(value):
            raise ValueError(f"OmniSVG {key} must be a whole number.")
    if options.memory not in {"auto", "offload", "cpu"}:
        raise ValueError("Choose Auto, CPU offload or CPU only.")


def memory_plan(torch, available_ram, options):
    validate_options(options)
    available = max(0, int(available_ram - 4*GIB))
    ram = min(available, int(options.ram*GIB)) if options.ram else available
    # Covers checkpoint pages, a dtype conversion and runtime headroom. This is
    # a conservative admission check, not an OS-enforced resident-RAM quota.
    required = WEIGHT_BYTES * (3 if options.memory == "cpu" else 2) + 2*GIB
    if ram < required:
        raise RuntimeError(f"OmniSVG needs a RAM budget of at least {required/GIB:.1f} GiB "
                           "for loading. Raise the RAM budget or close other applications.")
    plan = {"mode": options.memory, "ram_budget": ram, "ram_required": required,
            "gpu_budget": 0, "gpu_weights": 0, "dtype": "float32" if options.memory == "cpu" else "bfloat16"}
    if options.memory == "cpu":
        return plan
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected OmniSVG environment. "
                           "Check the driver or choose CPU only explicitly.")
    free, total = torch.cuda.mem_get_info(0)
    budget = max(0, int(free - 1.5*GIB))
    if options.vram:
        budget = min(budget, int(options.vram*GIB))
    if budget < 4*GIB:
        raise RuntimeError("OmniSVG requires at least 4 GiB of GPU budget after headroom. "
                           "Unload other models, raise the limit, or choose CPU only.")
    plan.update(gpu_budget=budget, gpu_weights=budget-2*GIB, gpu_total=total,
                gpu_free=free, gpu_name=torch.cuda.get_device_name(0),
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    return plan


def describe_plan(plan):
    ram = f"RAM budget: {plan['ram_budget']/GIB:.1f} GiB; minimum loading estimate: {plan['ram_required']/GIB:.1f} GiB."
    if plan["mode"] == "cpu":
        return "CPU only · FP32. " + ram
    return (f"GPU: {plan['gpu_name']} · free {plan['gpu_free']/GIB:.1f} GiB · "
            f"Torch limit {plan['gpu_budget']/GIB:.1f} GiB · weights budget {plan['gpu_weights']/GIB:.1f} GiB. " + ram)


def load_model(model_dir, plan):
    import torch
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, cpu_offload
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen2_5_vl" or config.vocab_size != 197000:
        raise RuntimeError("Unexpected OmniSVG 4B model configuration.")
    config._attn_implementation = "sdpa"
    config.pad_token_id = 151643
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
    # Construct the extended model on meta first. Do not download/load Qwen's
    # original weights and then resize live GPU embeddings as upstream does.
    state = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    if not state or any(not key.startswith("transformer.") for key in state):
        raise RuntimeError("Unexpected OmniSVG checkpoint keys.")
    model.load_state_dict({key.removeprefix("transformer."): value for key, value in state.items()},
                          strict=True, assign=True)
    del state
    model.tie_weights()
    dtype = getattr(torch, plan["dtype"])
    model = model.eval().to(dtype=dtype)
    if plan["mode"] == "cpu":
        return model
    torch.cuda.set_per_process_memory_fraction(plan["gpu_budget"] / plan["gpu_total"], 0)
    if plan["mode"] == "offload":
        return cpu_offload(model, execution_device=torch.device("cuda:0"))
    device_map = infer_auto_device_map(model, max_memory={0: plan["gpu_weights"], "cpu": plan["ram_budget"]},
        dtype=dtype, no_split_module_classes=["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock",
                                             "Qwen2_5_VisionTransformerPretrainedModel"])
    if "disk" in device_map.values():
        raise RuntimeError("The memory budgets are too small. Disk offload is disabled.")
    if not any(value in (0, "cuda:0") for value in device_map.values()):
        return cpu_offload(model, execution_device=torch.device("cuda:0"))
    print("Model placement: " + json.dumps(device_map), flush=True)
    return dispatch_model(model, device_map=device_map, main_device=0)


def load_tokenizer(source):
    import importlib.util
    # Only the official token decoding and DeepSVG serialization are used.
    # setup_svg removes eager Cairo/notebook/video imports from its pinned copy.
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("omnisvg_tokenizer", source / "tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SVGTokenizer(str(source / "config.yaml"), model_size="4B")


def prepare_inputs(processor, mode, source, prompt, keep_aspect=True):
    reference = None
    if mode == "omnisvg-image":
        with Image.open(source) as image:
            image = image.convert("RGB")
            reference = (ImageOps.pad(image, (448, 448), color="white", method=Image.Resampling.LANCZOS)
                         if keep_aspect else image.resize((448, 448), Image.Resampling.LANCZOS))
        content = [{"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                   {"type": "image", "image": reference}]
    else:
        content = [{"type": "text", "text": f"Generate an SVG illustration for: {prompt}\n        \n"
            "Requirements:\n- Create complete SVG path commands\n- Include proper coordinates and colors\n"
            "- Maintain visual clarity and composition"}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if reference is not None:
        kwargs["images"] = [reference]
    return processor(**kwargs), reference


def decode_svg(tokenizer, generated):
    import torch
    ids = generated.detach().cpu().reshape(-1)
    eos = (ids == 196999).nonzero().reshape(-1)
    if not len(eos):
        raise ValueError("The token/time budget ended before the SVG was complete.")
    ids = ids[:int(eos[0])]
    wrapped = torch.cat([torch.tensor([196998]), ids, torch.tensor([196999])]).unsqueeze(0)
    xy = tokenizer.process_generated_tokens(wrapped)
    tensors, colors = tokenizer.raster_svg(xy)
    if not tensors or not tensors[0]:
        raise ValueError("OmniSVG produced no vector paths.")
    colors += [40012] * max(0, len(tensors[0]) - len(colors))
    svg = tokenizer.apply_colors_to_svg(tensors[0], colors).to_str()
    root = parse_svg(svg)
    paths = sum(node.tag.split("}")[-1] == "path" for node in root.iter())
    if not paths:
        raise ValueError("OmniSVG produced no drawable paths.")
    return svg, paths


def render_candidate(svg):
    import io
    import resvg_py
    parse_svg(svg)
    with Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg, width=1024, height=1024,
                                                    skip_system_fonts=True))) as image:
        rgba = image.convert("RGBA")
        result = Image.new("RGB", rgba.size, "white")
        result.paste(rgba, mask=rgba.getchannel("A"))
    if not ImageChops.difference(result, Image.new("RGB", result.size, "white")).getbbox():
        raise ValueError("OmniSVG produced an empty white render.")
    return result


def generate(args, plan):
    import torch
    from transformers import AutoProcessor
    tokenizer = load_tokenizer(args.addon_root / "source")
    processor = AutoProcessor.from_pretrained(args.addon_root / "processor", local_files_only=True,
        trust_remote_code=False, padding_side="left", max_pixels=448*448)
    inputs, reference = prepare_inputs(processor, args.mode, args.source, args.prompt, args.keep_aspect)
    print("Memory admission check passed. " + describe_plan(plan), flush=True)
    started = time.monotonic()
    model = load_model(args.addon_root / "model", plan)
    device = "cpu" if plan["mode"] == "cpu" else "cuda:0"
    inputs = {key: value.to(device=device, dtype=getattr(torch, plan["dtype"])) if key == "pixel_values"
              else value.to(device) for key, value in inputs.items()}
    folder = args.output / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    successful, files = [], []
    generation_start = time.monotonic()
    for attempt in range(args.variants):
        remaining = args.timeout - (time.monotonic() - generation_start)
        if remaining <= 0:
            print("Generation time budget reached; keeping completed variants.", flush=True)
            break
        torch.manual_seed((args.seed + attempt) % 2**32)
        print(f"Generating variant {attempt+1}/{args.variants} (one at a time)…", flush=True)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.tokens, max_time=remaining,
                num_return_sequences=1, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition,
                use_cache=args.cache, eos_token_id=196999, bos_token_id=196998, pad_token_id=151643)
        try:
            svg, count = decode_svg(tokenizer, generated[0, inputs["input_ids"].shape[1]:])
            preview = render_candidate(svg)
            number = len(successful) + 1
            path = folder / f"{number:02d}.svg"
            path.write_text(svg, encoding="utf-8")
            preview.save(path.with_suffix(".png"))
            files += [path, path.with_suffix(".png")]
            score = None
            if reference is not None:
                diff = ImageChops.difference(reference.resize(preview.size), preview)
                score = round(sum(ImageStat.Stat(diff).mean) / (3*255)*100, 2)
            successful.append({"number": number, "seed": (args.seed+attempt) % 2**32,
                               "paths": count, "pixel_difference_percent": score})
            print(f"Variant {number}: {count} paths; pixel difference {score} percent.", flush=True)
        except (ValueError, RuntimeError) as exc:
            print(f"Variant {attempt+1} rejected: {exc}", flush=True)
        del generated
        gc.collect()
    if not successful:
        raise RuntimeError("No complete, nonempty SVG was generated. Increase the token/time budget "
                           "or simplify the source/description. No partial SVG was exported.")
    if reference is not None:
        reference.save(args.output / "reference.png")
        files.append(args.output / "reference.png")
    metadata = {"engine": "OmniSVG1.1_4B", "revision": "117d4c4541839e02943554fe71c0a0d54fcef819",
        "mode": args.mode, "variants": successful, "memory_plan": plan,
        "controls": {key: getattr(args, key) for key in ("tokens", "seed", "temperature", "top_p", "top_k",
                     "repetition", "variants", "timeout", "cache", "keep_aspect")},
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "prompt": args.prompt if args.mode == "omnisvg-text" else None,
        "peak_torch_vram": torch.cuda.max_memory_allocated(0) if device != "cpu" else 0}
    export((folder / "01.svg").read_text(encoding="utf-8"), args.output, metadata, files)
    print(f"Exported {len(successful)} complete variant(s). Compare them with the source.", flush=True)


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("--addon-root", type=Path, required=True)
    cli.add_argument("--check-imports", action="store_true")
    cli.add_argument("--probe", action="store_true")
    cli.add_argument("--mode", choices=["omnisvg-image", "omnisvg-text"])
    cli.add_argument("--source", type=Path)
    cli.add_argument("--prompt", default="")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--memory", choices=["auto", "offload", "cpu"], default="auto")
    for name, default, kind in (("tokens", 2048, int), ("seed", 0, int), ("variants", 1, int),
        ("temperature", .3, float), ("top-p", .9, float), ("top-k", 50, int), ("repetition", 1.05, float),
        ("vram", 0, float), ("ram", 0, float), ("timeout", 300, int)):
        cli.add_argument("--" + name, type=kind, default=default)
    cli.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--keep-aspect", action=argparse.BooleanOptionalAction, default=True)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    if args.check_imports:
        import torch, accelerate, transformers, resvg_py
        load_tokenizer(args.addon_root / "source")
        print("OmniSVG inference and official SVG decoder imports ready.", flush=True)
    else:
        import torch
        import psutil
        plan = memory_plan(torch, psutil.virtual_memory().available, args)
        if args.probe:
            print("OmniSVG diagnostic: " + describe_plan(plan), flush=True)
        else:
            if not args.mode or not args.output or (args.mode == "omnisvg-image" and not args.source) or (
                    args.mode == "omnisvg-text" and not args.prompt.strip()):
                raise ValueError("Select a mode and provide its source image or description and an output folder.")
            args.output.mkdir(parents=True, exist_ok=True)
            try:
                generate(args, plan)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("OmniSVG reached the GPU allocator budget. Choose CPU offload, "
                                   "disable KV cache or reduce tokens. Completed temporary variants are discarded.") from exc
