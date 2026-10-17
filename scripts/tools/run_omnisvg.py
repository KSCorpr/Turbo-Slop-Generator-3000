#!/usr/bin/env python3
"""OmniSVG 1.1 4B: strict local checkpoint loading, bounded memory and serial variants."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
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
    limits = {"tokens": (256, 16384), "seed": (0, 2**32-1), "variants": (1, 4),
              "temperature": (0.05, 1.5), "top_p": (0.05, 1), "top_k": (1, 200),
              "repetition": (1, 1.5), "vram": (0, 128), "ram": (0, 1024), "timeout": (0, 7200)}
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


class GenerationWatch:
    """StoppingCriteria callable without importing Torch into the main UI process."""
    def __init__(self, input_length, token_budget, seconds=0, log=print, clock=None):
        self.input_length = input_length
        self.token_budget = token_budget
        self.seconds = seconds
        self.clock = clock or time.monotonic
        self.started = self.last_log = self.clock()
        self.log = log
        self.timed_out = False

    def __call__(self, input_ids, scores=None, **kwargs):
        now = self.clock()
        elapsed = now - self.started
        count = input_ids.shape[-1] - self.input_length
        if now-self.last_log >= 10:
            self.log(f"OmniSVG progress: {count}/{self.token_budget} tokens · "
                     f"{elapsed:.1f}s · {count/max(elapsed, .001):.2f} tokens/s")
            self.last_log = now
        self.timed_out = self.seconds > 0 and elapsed >= self.seconds
        return self.timed_out


def stop_statistics(ids, budget, elapsed, timed_out=False):
    ids = list(ids)
    limits = (["token_limit"] if len(ids) >= budget else []) + (["time_limit"] if timed_out else [])
    return {"generated_tokens": len(ids), "token_budget": budget,
            "elapsed_seconds": round(elapsed, 2), "last_token_id": ids[-1] if ids else None,
            "limits_reached": limits,
            "stop_reason": "eos" if 196999 in ids else "token_and_time_limit" if len(limits) == 2 else "time_limit" if timed_out else
                           "token_limit" if len(ids) >= budget else "unknown"}


def incomplete_message(stats):
    if stats["stop_reason"] == "token_and_time_limit":
        return (f"OmniSVG reached both its {stats['token_budget']}-token limit and generation time limit "
                f"after {stats['elapsed_seconds']}s, without EOS 196999. Raise the token budget and "
                "increase the time budget or set time to 0.")
    if stats["stop_reason"] == "time_limit":
        return (f"OmniSVG generation time limit reached after {stats['elapsed_seconds']}s "
                f"and {stats['generated_tokens']}/{stats['token_budget']} tokens, without EOS 196999. "
                "Increase the time budget or set it to 0 (no time limit).")
    if stats["stop_reason"] == "token_limit":
        return (f"OmniSVG reached its {stats['token_budget']}-token limit without EOS 196999. "
                "Increase the SVG token budget or simplify the source.")
    return (f"OmniSVG stopped without EOS 196999 after {stats['generated_tokens']} tokens "
            f"and {stats['elapsed_seconds']}s. See the saved token IDs for diagnosis.")


def save_diagnostics(args, attempts, raw_outputs, error):
    directory = getattr(args, "diagnostics_dir", None) or args.output / "diagnostic"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"engine": "OmniSVG1.1_4B", "failed": True, "error": str(error), "attempts": attempts,
              "time_budget_seconds": args.timeout, "token_budget": args.tokens}
    (directory / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(directory / "diagnostics.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(directory / "diagnostic.json", "diagnostic.json")
        for index, ids in enumerate(raw_outputs, 1):
            archive.writestr(f"generated-token-ids-{index:02d}.json", json.dumps(ids))
    print("Diagnostics saved to: " + str(directory / "diagnostics.zip"), flush=True)


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
    previous_path = list(sys.path)
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location("omnisvg_tokenizer", source / "tokenizer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SVGTokenizer(str(source / "config.yaml"), model_size="4B")
    finally:
        sys.path[:] = previous_path


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


def decode_svg(tokenizer, generated, stats=None, *, allow_partial=False):
    import torch
    ids = generated.detach().cpu().reshape(-1)
    eos = (ids == 196999).nonzero().reshape(-1)
    complete = bool(len(eos))
    if not len(eos):
        message = incomplete_message(stats or stop_statistics(ids.tolist(), len(ids)+1, 0))
        if not allow_partial:
            raise ValueError(message)
        # Retain only a prefix ending at an actual path-color token. The
        # official decoder can serialize truncated output, but an unfinished
        # tail must not acquire invented geometry or a default fill here.
        start = tokenizer.COLOR_COORD_BOUNDARY
        end = start + tokenizer.MAX_COLOR_TOKENS
        positions = ((ids >= start) & (ids < end)).nonzero().reshape(-1)
        if not len(positions):
            raise ValueError(message + " No finished color-delimited path is available for a partial SVG.")
        retained = int(positions[-1])+1
        if stats is not None:
            stats.update(complete=False, retained_tokens=retained, discarded_tail_tokens=len(ids)-retained,
                         warning=message)
        ids = ids[:retained]
    else:
        ids = ids[:int(eos[0])]
        if stats is not None:
            stats["complete"] = True
    wrapped = torch.cat([torch.tensor([196998]), ids, torch.tensor([196999])]).unsqueeze(0)
    xy = tokenizer.process_generated_tokens(wrapped)
    tensors, colors = tokenizer.raster_svg(xy)
    if not tensors or not tensors[0]:
        raise ValueError("OmniSVG produced no vector paths.")
    if not complete and len(colors) != len(tensors[0]):
        raise ValueError("No finished color-delimited path is available for a partial SVG.")
    colors += [40012] * max(0, len(tensors[0]) - len(colors))
    svg = tokenizer.apply_colors_to_svg(tensors[0], colors).to_str()
    root = parse_svg(svg)
    paths = sum(node.tag.split("}")[-1] == "path" for node in root.iter())
    if not paths:
        raise ValueError("OmniSVG produced no drawable paths.")
    if not complete:
        description = ET.Element("{http://www.w3.org/2000/svg}desc")
        description.text = "PARTIAL SVG: generation ended without EOS. Only finished color-delimited paths were retained; the drawing may be incomplete."
        root.insert(0, description)
        svg = ET.tostring(root, encoding="unicode")
    return svg, paths


def export_variants(args, variants, files, plan, attempts, started, raw_outputs, *, recovered=False):
    import torch
    selected = next((item for item in variants if item["complete"]), variants[0])
    for index, ids in enumerate(raw_outputs, 1):
        path = args.output / f"generated-token-ids-{index:02d}.json"
        path.write_text(json.dumps(ids), encoding="utf-8")
        files.append(path)
    metadata = {"engine": "OmniSVG1.1_4B", "revision": "117d4c4541839e02943554fe71c0a0d54fcef819",
        "mode": args.mode, "variants": variants, "memory_plan": plan, "attempts": attempts,
        "complete": selected["complete"], "default_variant": selected["number"],
        "recovered_from_diagnostics": recovered,
        "controls": {key: getattr(args, key, None) for key in ("tokens", "seed", "temperature", "top_p", "top_k",
                     "repetition", "variants", "timeout", "cache", "keep_aspect")},
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "prompt": args.prompt if args.mode == "omnisvg-text" else None,
        "peak_torch_vram": torch.cuda.max_memory_allocated(0) if plan and plan["mode"] != "cpu" else 0}
    export((args.output / "candidates" / (selected["file_stem"]+".svg")).read_text(encoding="utf-8"),
           args.output, metadata, files)
    partial = sum(not item["complete"] for item in variants)
    print(f"Exported {len(variants)} SVG variant(s), including {partial} explicitly marked partial drawing(s).", flush=True)


def recover_diagnostics(args):
    """Decode saved IDs on CPU; never instantiate or load the vision model."""
    import torch
    import re
    started = time.monotonic()
    with zipfile.ZipFile(args.recover_diagnostics) as archive:
        if len(archive.infolist()) > 12 or sum(item.file_size for item in archive.infolist()) > 2_000_000:
            raise ValueError("The OmniSVG diagnostic ZIP is too large.")
        report = json.loads(archive.read("diagnostic.json"))
        if not isinstance(report, dict) or report.get("engine") != "OmniSVG1.1_4B":
            raise ValueError("Import an OmniSVG 4B diagnostic ZIP containing generated token IDs.")
        budget = report.get("token_budget")
        if type(budget) is not int or not 256 <= budget <= 16384:
            raise ValueError("Invalid token budget in OmniSVG diagnostic ZIP.")
        names = sorted(name for name in archive.namelist() if re.fullmatch(r"generated-token-ids-\d{2}\.json", name))
        if not 1 <= len(names) <= 4 or len(set(names)) != len(names):
            raise ValueError("The ZIP must contain one to four OmniSVG token sequences.")
        raw_outputs = [json.loads(archive.read(name)) for name in names]
    for ids in raw_outputs:
        if not isinstance(ids, list) or not 1 <= len(ids) <= 16384 or any(type(token) is not int or not 0 <= token < 197000 for token in ids):
            raise ValueError("Invalid OmniSVG token sequence in diagnostic ZIP.")
    tokenizer = load_tokenizer(args.addon_root / "source")
    folder = args.output / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    variants, files, attempts = [], [], []
    for index, ids in enumerate(raw_outputs):
        stats = stop_statistics(ids, budget, 0)
        stats.update(attempt=index+1, seed=None)
        try:
            svg, count = decode_svg(tokenizer, torch.tensor(ids), stats, allow_partial=True)
            preview = render_candidate(svg)
            number = len(variants)+1
            stem = f"{number:02d}" + ("" if stats["complete"] else ".partial")
            path = folder / (stem+".svg")
            path.write_text(svg, encoding="utf-8")
            preview.save(path.with_suffix(".png"))
            # with_suffix replaces '.svg', preserving the '.partial' marker.
            files += [path, path.with_suffix(".png")]
            variants.append({"number": number, "seed": None, "paths": count, "pixel_difference_percent": None,
                             "complete": stats["complete"], "file_stem": stem, "generation": stats.copy()})
        except (ValueError, RuntimeError) as exc:
            stats["error"] = str(exc)
            print(f"Saved attempt {index+1} cannot be recovered: {exc}", flush=True)
        attempts.append(stats)
    if not variants:
        raise RuntimeError("The saved tokens contain no recoverable, nonempty vector drawing. No model was loaded.")
    print("Recovered saved tokens on CPU. No inference model or GPU weights were loaded.", flush=True)
    args.mode = "omnisvg-recovery"
    export_variants(args, variants, files, None, attempts, started, raw_outputs, recovered=True)


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
    from transformers import AutoProcessor, StoppingCriteriaList
    tokenizer = load_tokenizer(args.addon_root / "source")
    processor = AutoProcessor.from_pretrained(args.addon_root / "processor", local_files_only=True,
        trust_remote_code=False, use_fast=False, padding_side="left", max_pixels=448*448)
    inputs, reference = prepare_inputs(processor, args.mode, args.source, args.prompt, args.keep_aspect)
    print("Memory admission check passed. " + describe_plan(plan), flush=True)
    started = time.monotonic()
    model = load_model(args.addon_root / "model", plan)
    device = "cpu" if plan["mode"] == "cpu" else "cuda:0"
    inputs = {key: value.to(device=device, dtype=getattr(torch, plan["dtype"])) if key == "pixel_values"
              else value.to(device) for key, value in inputs.items()}
    folder = args.output / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    successful, files, attempts, raw_outputs = [], [], [], []
    generation_start = time.monotonic()
    for attempt in range(args.variants):
        remaining = max(0, args.timeout - (time.monotonic() - generation_start)) if args.timeout else 0
        if args.timeout and remaining <= 0:
            print("Generation time budget reached; keeping completed variants.", flush=True)
            break
        torch.manual_seed((args.seed + attempt) % 2**32)
        limit_label = f"{remaining:.0f}s remaining" if args.timeout else "no time limit"
        print(f"Generating variant {attempt+1}/{args.variants}: up to {args.tokens} tokens, {limit_label}…", flush=True)
        watch = GenerationWatch(inputs["input_ids"].shape[1], args.tokens, remaining)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.tokens,
                stopping_criteria=StoppingCriteriaList([watch]),
                num_return_sequences=1, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition,
                use_cache=args.cache, eos_token_id=196999, bos_token_id=196998, pad_token_id=151643)
        ids = generated[0, inputs["input_ids"].shape[1]:].detach().cpu()
        raw_outputs.append(ids.tolist())
        stats = stop_statistics(raw_outputs[-1], args.tokens, time.monotonic()-watch.started, watch.timed_out)
        stats.update(attempt=attempt+1, seed=(args.seed+attempt) % 2**32)
        print(f"Variant {attempt+1} stopped: {stats['stop_reason']} · "
              f"{stats['generated_tokens']}/{args.tokens} tokens · {stats['elapsed_seconds']}s.", flush=True)
        try:
            svg, count = decode_svg(tokenizer, ids, stats, allow_partial=True)
            preview = render_candidate(svg)
            number = len(successful) + 1
            stem = f"{number:02d}" + ("" if stats["complete"] else ".partial")
            path = folder / (stem+".svg")
            path.write_text(svg, encoding="utf-8")
            preview.save(path.with_suffix(".png"))
            files += [path, path.with_suffix(".png")]
            score = None
            if reference is not None:
                diff = ImageChops.difference(reference.resize(preview.size), preview)
                score = round(sum(ImageStat.Stat(diff).mean) / (3*255)*100, 2)
            successful.append({"number": number, "seed": (args.seed+attempt) % 2**32,
                               "paths": count, "pixel_difference_percent": score, "generation": stats.copy(),
                               "complete": stats["complete"], "file_stem": stem})
            label = "complete sequence" if stats["complete"] else "PARTIAL drawing (no EOS)"
            print(f"Variant {number}: {label}; {count} paths; pixel difference {score} percent.", flush=True)
        except (ValueError, RuntimeError) as exc:
            stats["error"] = str(exc)
            print(f"Variant {attempt+1} rejected: {exc}", flush=True)
        attempts.append(stats)
        del generated
        gc.collect()
    if not successful:
        reason = next((item["error"] for item in reversed(attempts) if item.get("error")),
                      "The shared generation time budget was exhausted.")
        error = "No complete SVG was generated. " + reason + " No partial SVG was exported."
        save_diagnostics(args, attempts, raw_outputs, error)
        raise RuntimeError(error)
    if reference is not None:
        reference.save(args.output / "reference.png")
        files.append(args.output / "reference.png")
    export_variants(args, successful, files, plan, attempts, started, raw_outputs)


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("--addon-root", type=Path, required=True)
    cli.add_argument("--check-imports", action="store_true")
    cli.add_argument("--probe", action="store_true")
    cli.add_argument("--mode", choices=["omnisvg-image", "omnisvg-text"])
    cli.add_argument("--source", type=Path)
    cli.add_argument("--prompt", default="")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--diagnostics-dir", type=Path)
    cli.add_argument("--recover-diagnostics", type=Path)
    cli.add_argument("--memory", choices=["auto", "offload", "cpu"], default="auto")
    for name, default, kind in (("tokens", 4096, int), ("seed", 0, int), ("variants", 1, int),
        ("temperature", .3, float), ("top-p", .9, float), ("top-k", 50, int), ("repetition", 1.05, float),
        ("vram", 0, float), ("ram", 0, float), ("timeout", 0, int)):
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
    elif args.recover_diagnostics:
        if not args.output:
            raise ValueError("Choose an output folder for recovered SVG files.")
        args.output.mkdir(parents=True, exist_ok=True)
        recover_diagnostics(args)
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
