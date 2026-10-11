#!/usr/bin/env python3
"""Install the tested local Comfy edition in its own portable Python runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atelier import comfy_runtime as runtime, settings
from scripts.setup_media import _install_source, environment, pip, run


def verify_local_api(log=print):
    """Validate native workflows and exercise the real owned loopback server.

    No inference weights are downloaded. This verifies installation, node
    contracts, HTTP/WS submission and RGBA roundtrip; not GPU model quality.
    """
    from PIL import Image
    from atelier.engine import comfy, comfy_workflows as workflows
    with tempfile.TemporaryDirectory(dir=runtime.root()) as temporary:
        folder = Path(temporary)
        paths = {}
        for kind in ("diffusion", "encoder", "vae", "checkpoint", "lora", "controlnet"):
            path = folder / ("contract_" + kind + ".safetensors")
            path.touch()  # only used by validate_prompt's filename checks
            paths[kind] = path
        roots = runtime.model_roots(tuple(paths.values()))
        source = folder / "alpha.png"
        Image.new("RGBA", (64, 64), (20, 40, 60, 100)).save(source)
        old_output = settings.OUTPUT_DIR
        settings.OUTPUT_DIR = folder / "roundtrip"
        try:
            with comfy.ENGINE.session():
                comfy.ENGINE.ensure(roots, log=log)
                uploaded = comfy.ENGINE.upload(source, rgba=True)
                info = comfy.ENGINE.request("/object_info").json()
                required = {"UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21",
                            "ManualSigmas", "SamplerCustom", "KSamplerSelect", "VAEDecodeTiled",
                            "VAEEncodeTiled", "SetLatentNoiseMask", "CheckpointLoaderSimple",
                            "ControlNetApplyAdvanced", "ControlNetLoader", "TurboSlopLoadRGBA",
                            "TurboSlopEmptyQwen21", "SaveImage"}
                missing = required - info.keys()
                if missing:
                    raise RuntimeError("Comfy nodes missing: " + ", ".join(sorted(missing)))
                names = {k: runtime.model_name(p, roots) for k, p in paths.items()}
                graphs = [workflows.qwen(diffusion=names["diffusion"], encoder=names["encoder"],
                           vae=names["vae"], prompt="test", width=64, height=64, seed=1, prefix="test"),
                          workflows.qwen(diffusion=names["diffusion"], encoder=names["encoder"],
                           vae=names["vae"], prompt="test", width=64, height=64, seed=1, prefix="test",
                           references=[uploaded] * 3, init=uploaded, mask=uploaded,
                           loras=[(names["lora"], 0.5)]),
                          workflows.sdxl(checkpoint=names["checkpoint"], vae=names["vae"],
                           prompt="test", negative="", width=64, height=64, seed=1, steps=2, cfg=6,
                           prefix="test", init=uploaded, mask=uploaded,
                           controlnet=names["controlnet"], loras=[(names["lora"], 0.5)])]
                for graph in graphs:
                    result = comfy.ENGINE.request("/turboslop/validate", payload={"prompt": graph.nodes}).json()
                    if not result.get("valid"):
                        raise RuntimeError("Native workflow contract failed: " + json.dumps(result))
                roundtrip = workflows.Workflow()
                image = roundtrip.add("TurboSlopLoadRGBA", image=uploaded)
                roundtrip.finish(image, "turboslop-install-check")
                out = comfy.ENGINE.execute(roundtrip, log=log)[0]
                with Image.open(out) as checked:
                    if checked.mode != "RGBA" or checked.getpixel((0, 0)) != (20, 40, 60, 100):
                        raise RuntimeError("Local RGBA roundtrip failed.")
                log("Native Qwen/SDXL workflows, loopback API and RGBA roundtrip passed.")
        finally:
            comfy.ENGINE.close()
            settings.OUTPUT_DIR = old_output


def install(flavor="auto"):
    target = runtime.root()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "installed.json"
    marker.unlink(missing_ok=True)
    gpu = runtime.selected_gpu()
    flavor = runtime.cuda_flavor(gpu) if flavor == "auto" else flavor
    if flavor not in {"cu126", "cu130", "cpu", "mps"}:
        raise ValueError("Unknown Torch runtime.")
    # Provision full Python: the Windows UI's embedded Python cannot make venvs.
    py = environment("comfy")
    _install_source("Comfy-Org/ComfyUI", runtime.COMMIT, target / "source")
    constraints = target / "constraints.txt"
    constraints.write_text("torch==2.13.0\ntorchvision==0.28.0\ncomfy-kitchen==0.2.37\n", encoding="utf-8")
    torch_args = ["torch==2.13.0", "torchvision==0.28.0"]
    if flavor != "mps":
        torch_args += ["--index-url", "https://download.pytorch.org/whl/" + flavor]
    pip(py, *torch_args)
    pip(py, "-c", constraints, "-r", target / "source" / "requirements.txt",
        "requests>=2.31", "websocket-client==1.8.0")
    if flavor.startswith("cu"):
        run(py, "-c", "import torch; assert torch.cuda.is_available(), "
            "'CUDA unavailable: check NVIDIA drivers and the selected runtime'; "
            "torch.ones(16, device='cuda').add_(1); torch.cuda.synchronize(); "
            "print('Comfy GPU:', torch.cuda.get_device_name(0))")
    marker.write_text(json.dumps({"schema": runtime.SCHEMA, "commit": runtime.COMMIT,
                                 "version": runtime.VERSION, "flavor": flavor}), encoding="utf-8")
    try:
        # Run inside the isolated runtime, not Gradio's dependency environment.
        run(py, Path(__file__), "--verify")
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    print(runtime.status(), flush=True)
    if flavor == "cu126":
        print("This driver-compatible runtime does not enable Comfy's latest native INT8 CUDA kernels. "
              "Use CUDA 13 with a compatible driver to enable them.", flush=True)
    print("Installation API verified. Image inference and GPU speed still depend on the selected model and hardware.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("auto", "cu126", "cu130", "cpu", "mps"), default="auto")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    args = parser.parse_args()
    settings.configure_hf_env()
    # Same GPU choice as the app; index-to-UUID mapping also applies to Torch's probe.
    gpu = runtime.selected_gpu()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = runtime.gpu_env(gpu.index)["CUDA_VISIBLE_DEVICES"]
    if args.check_deps:
        run(sys.executable, "-m", "uv", "pip", "check", "--python", runtime.python())
    elif args.verify:
        verify_local_api()
    else:
        install(args.runtime)


if __name__ == "__main__":
    main()
