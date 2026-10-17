#!/usr/bin/env python3
"""HiVG's official prompts and detokenizer with local weights and bounded SDPA loading."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_svg import export


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
                torch_dtype=dtype, attn_implementation="sdpa", max_memory=budget,
                offload_folder=str(args.output / "offload"))
            self.model.eval()

    print("Loading local HiVG weights with SDPA; CPU offload may reduce speed.", flush=True)
    pipeline = LocalPipeline(str(args.addon_root / "model"), trust_remote_code=False,
        max_new_tokens=args.tokens, do_sample=True, temperature=0.7, top_p=0.9)
    with torch.inference_mode():
        result = pipeline.img2svg(str(args.source)) if args.mode == "hivg-image" else pipeline.text2svg(args.prompt)
    if not result.get("success") or not result.get("svg"):
        raise RuntimeError("HiVG did not produce a valid SVG: " + str(result.get("error", "incomplete output")))
    raw = result.get("raw_output", "")
    # Upstream detokenization repairs unclosed sequences, even if generation hit its cap.
    if "</svg>" not in raw:
        raise RuntimeError("HiVG stopped before completing the SVG. Increase the token budget or simplify the input.")
    export(result["svg"], args.output, {"engine": "HiVG-3B-Base", "mode": args.mode,
        "seed": args.seed, "max_new_tokens": args.tokens,
        "revision": "259cba9684201bbde830a59ca7543345bb333519",
        "raw_output_characters": len(raw), "prompt": args.prompt if args.mode == "hivg-text" else None})
    print("HiVG SVG exported. Compare the result with the source; details may be reconstructed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hivg-image", "hivg-text"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048, choices=range(256, 4097))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "hivg-image" and not args.source:
        parser.error("An image is required.")
    if args.mode == "hivg-text" and not args.prompt.strip():
        parser.error("A description is required.")
    args.output.mkdir(parents=True, exist_ok=True)
    generate(args)
