#!/usr/bin/env python3
"""HiVG's official prompts and detokenizer with local weights and bounded SDPA loading."""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

try:
    from .run_svg import export, parse_svg
except ImportError:
    from run_svg import export, parse_svg


def generation_stats(ids, limit, eos_ids, svg_end_id):
    ids = list(ids)
    reason = ("svg_end" if ids and ids[-1] == svg_end_id else
              "eos" if ids and ids[-1] in eos_ids else
              "token_limit" if len(ids) >= limit else "unknown")
    return {"generated_tokens": len(ids), "token_budget": limit, "stop_reason": reason}


def validate_result(result, stats):
    raw = result.get("raw_output") or ""
    segments = sorted(set(re.findall(r"<SEG_\d+>", raw)))
    if segments:
        raise ValueError("HiVG generated compressed SEG tokens, but this public release has no "
                         "matching segment-to-SVG dictionary. These shapes cannot be decoded faithfully. "
                         "Use OmniSVG or VTracer. Upstream: https://github.com/ximinng/HiVG/issues/5")
    if "</svg>" not in raw:
        count = stats.get("generated_tokens", "unknown")
        budget = stats.get("token_budget", "unknown")
        if stats.get("stop_reason") == "token_limit":
            raise ValueError(f"HiVG reached its {budget}-token budget before completing </svg> "
                             f"({count} tokens generated). Raise the HiVG token budget or simplify the source.")
        raise ValueError(f"HiVG stopped before completing </svg> ({count}/{budget} tokens; "
                         f"stop: {stats.get('stop_reason', 'unknown')}). "
                         "The output is incomplete; adding a closing tag would not restore missing shapes.")
    if result.get("success") and result.get("svg"):
        try:
            parse_svg(result["svg"])
            return result["svg"], "official_atomic_tokens"
        except (ValueError, ET.ParseError):
            pass
    # Some outputs contain complete XML instead of the training token format.
    # Parse it as XML without feeding XML attributes/path data to the atomic decoder.
    match = re.search(r"<svg(?:\s[^>]*|)>.*?</svg>", raw, re.S)
    if match:
        try:
            candidate = match.group(0)
            if re.search(r"<!\s*(DOCTYPE|ENTITY)", candidate, re.I):
                raise ValueError("Unsupported SVG declarations.")
            root = ET.fromstring(candidate)
            if root.tag == "svg":
                root.set("xmlns", "http://www.w3.org/2000/svg")
            candidate = ET.tostring(root, encoding="unicode")
            parse_svg(candidate)
            return candidate, "native_svg_xml"
        except (ValueError, ET.ParseError):
            pass
    raise ValueError("HiVG output could not be decoded into valid vector shapes: " +
                     str(result.get("error") or "malformed token sequence"))


def save_diagnostic(directory, result, stats, error):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = result.get("raw_output") or ""
    report = {"engine": "HiVG-3B-Base", "failed": True, "error": str(error), **stats,
              "raw_output_characters": len(raw), "decoder_error": result.get("error"),
              "contains_segment_tokens": bool(re.search(r"<SEG_\d+>", raw))}
    (directory / "raw-output.txt").write_text(raw, encoding="utf-8")
    (directory / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(directory / "diagnostics.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("raw-output.txt", "diagnostic.json"):
            archive.write(directory / name, name)
    print(f"HiVG output: {report.get('generated_tokens', 'unknown')}/{report.get('token_budget', 'unknown')} "
          f"tokens; stop: {report.get('stop_reason', 'unknown')}; "
          f"compressed segments: {report['contains_segment_tokens']}.", flush=True)
    print("Diagnostics saved to: " + str(directory / "diagnostics.zip"), flush=True)


def memory_plan(torch, available_ram):
    gib = 1024 ** 3
    cpu = int(available_ram - 2 * gib)
    if cpu < 4 * gib:
        raise RuntimeError("HiVG needs more free system RAM. Close other applications first.")
    if not torch.cuda.is_available():
        if cpu < 17 * gib:
            raise RuntimeError("CPU inference needs at least 19 GB free system RAM including headroom.")
        return "cpu", torch.float32, None
    free, _ = torch.cuda.mem_get_info(0)
    gpu = int(free - 2 * gib)
    if gpu < 2 * gib:
        raise RuntimeError("Not enough free GPU memory for HiVG. Unload other models first.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return "auto", dtype, {0: gpu, "cpu": cpu}


def generate(args):
    import torch
    import psutil
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from hivg_infer import HiSVGInferencePipeline

    device, dtype, budget = memory_plan(torch, psutil.virtual_memory().available)
    torch.manual_seed(args.seed)

    class LocalPipeline(HiSVGInferencePipeline):
        def _load_model(self, model_path, adapter_path, device_map, torch_dtype, trust_remote_code):
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=False, max_pixels=512 * 512)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                local_files_only=True, trust_remote_code=False, device_map=device,
                dtype=dtype, attn_implementation="sdpa", max_memory=budget,
                offload_folder=str(args.output / "offload"))
            self.model.eval()

        def _decode_output(self, output_ids, input_len):
            ids = output_ids[input_len:].tolist()
            self.output_stats = generation_stats(ids, args.tokens, eos_ids, svg_end)
            return super()._decode_output(output_ids, input_len)

    print("Loading local HiVG weights with SDPA; CPU offload may reduce speed.", flush=True)
    pipeline = LocalPipeline(str(args.addon_root / "model"), trust_remote_code=False,
        max_new_tokens=args.tokens, do_sample=False, temperature=1.0, top_p=1.0,
        repetition_penalty=1.05)
    vocab = pipeline.processor.tokenizer.get_vocab()
    svg_end = vocab.get("</svg>")
    # Keep the checkpoint's chat EOS tokens and also stop at a complete SVG.
    configured_eos = pipeline.model.generation_config.eos_token_id
    eos_ids = configured_eos if isinstance(configured_eos, list) else [configured_eos]
    eos_ids = [value for value in eos_ids if isinstance(value, int)]
    stops = list(dict.fromkeys(eos_ids + ([svg_end] if isinstance(svg_end, int) else [])))
    generation_options = {"use_cache": True}
    if stops:
        generation_options["eos_token_id"] = stops
    with torch.inference_mode():
        result = (pipeline.img2svg(str(args.source), **generation_options) if args.mode == "hivg-image"
                  else pipeline.text2svg(args.prompt, **generation_options))
    stats = getattr(pipeline, "output_stats", {"token_budget": args.tokens, "stop_reason": "unknown"})
    try:
        svg, decoding = validate_result(result, stats)
    except ValueError as exc:
        diagnostic_dir = getattr(args, "diagnostics_dir", None) or args.output / "diagnostic"
        save_diagnostic(diagnostic_dir, result, stats, exc)
        raise RuntimeError(str(exc)) from exc
    raw = result.get("raw_output", "")
    export(svg, args.output, {"engine": "HiVG-3B-Base", "mode": args.mode,
        "seed": args.seed, "max_new_tokens": args.tokens,
        "revision": "259cba9684201bbde830a59ca7543345bb333519",
        "raw_output_characters": len(raw), "generation": stats, "decoding": decoding,
        "sampling": "greedy", "prompt": args.prompt if args.mode == "hivg-text" else None})
    print("HiVG SVG exported. Compare the result with the source; details may be reconstructed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hivg-image", "hivg-text"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096, choices=range(256, 16385))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "hivg-image" and not args.source:
        parser.error("An image is required.")
    if args.mode == "hivg-text" and not args.prompt.strip():
        parser.error("A description is required.")
    args.output.mkdir(parents=True, exist_ok=True)
    generate(args)
