#!/usr/bin/env python3
"""Explicit installs for SVG tools; never change the application's Torch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atelier import addons, transfers, hardware
from scripts.setup_media import environment, pip, run, _install_source


def prepare_omnisvg_source(source):
    # DeepSVG imports notebook/video/Cairo renderers eagerly. SVG serialization
    # does not use them; Turbo Slop uses its validated resvg PNG renderer instead.
    path = Path(source) / "deepsvg/svglib/svg.py"
    text = path.read_text(encoding="utf-8")
    for line in ("import IPython.display as ipd\n", "import cairosvg\n",
                 "from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display\n"):
        if line not in text:
            raise RuntimeError("The pinned DeepSVG rendering imports have changed.")
        text = text.replace(line, "")
    path.write_text(text, encoding="utf-8")


def install_omnisvg(target, py, common):
    gpus = hardware.detect_gpus()
    # CUDA_VISIBLE_DEVICES is set by the shared installer job, so honor it here too.
    import os
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu = next((g for g in gpus if str(g.index) == selected), gpus[0] if gpus else None)
    blackwell = gpu is not None and gpu.arch == "blackwell"
    torch_version, vision_version, cuda = (("2.8.0", "0.23.0", "cu128") if blackwell
                                           else ("2.7.1", "0.22.1", "cu126"))
    constraints = target / "constraints.txt"
    constraints.write_text(f"torch=={torch_version}+{cuda}\ntorchvision=={vision_version}+{cuda}\n", encoding="utf-8")
    pip(py, f"torch=={torch_version}+{cuda}", f"torchvision=={vision_version}+{cuda}",
        "--index-url", "https://download.pytorch.org/whl/" + cuda)
    pip(py, "-c", constraints, *common, "transformers==4.51.3", "accelerate>=1.2,<2",
        "safetensors>=0.5,<1", "PyYAML>=6,<7", "psutil>=6,<8", "shapely>=2,<3", "networkx>=3,<4")
    _install_source("OmniSVG/OmniSVG", addons.OMNISVG_COMMIT, target / "source")
    prepare_omnisvg_source(target / "source")
    transfers.snapshot_download(repo_id="OmniSVG/OmniSVG1.1_4B", revision=addons.OMNISVG_REVISION,
        local_dir=str(target / "model"), allow_patterns=["config.json", "pytorch_model.bin", "README.md"])
    # Only processor/tokenizer assets, never another copy of the Qwen base weights.
    transfers.snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision=addons.OMNISVG_PROCESSOR_REVISION, local_dir=str(target / "processor"),
        allow_patterns=["config.json", "preprocessor_config.json", "tokenizer*.json", "vocab.json",
                        "merges.txt", "chat_template.json", "LICENSE"])
    if (target / "model/pytorch_model.bin").stat().st_size != 7_694_002_862:
        raise RuntimeError("Incomplete OmniSVG checkpoint. Run Install / repair again.")
    run(py, ROOT / "scripts/tools/run_omnisvg.py", "--addon-root", target, "--check-imports")
    return {"commit": addons.OMNISVG_COMMIT, "revision": addons.OMNISVG_REVISION,
            "processor_revision": addons.OMNISVG_PROCESSOR_REVISION, "cuda": cuda}


def install(name):
    target = addons.root(name)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "installed.json"
    marker.unlink(missing_ok=True)
    py = environment(name)
    common = ["Pillow>=10.2,<13", "numpy>=1.26,<3", "resvg-py==0.5.0"]
    if name == "svg":
        pip(py, *common, "vtracer==0.6.11", "opencv-python-headless==4.11.0.86")
        _install_source("showlab/LayerTracer", addons.LAYERTRACER_COMMIT, target / "source")
        run(py, "-c", "import vtracer, cv2, resvg_py; print('CPU tracing ready')")
        metadata = {"commit": addons.LAYERTRACER_COMMIT}
    elif name == "omnisvg":
        metadata = install_omnisvg(target, py, common)
    else:
        # SDPA requires no FlashAttention, vLLM or compiler. cu128 supports Blackwell too.
        constraints = target / "constraints.txt"
        constraints.write_text("torch==2.8.0\ntorchvision==0.23.0\n", encoding="utf-8")
        pip(py, "torch==2.8.0", "torchvision==0.23.0",
            "--index-url", "https://download.pytorch.org/whl/cu128")
        pip(py, "-c", constraints, *common, "transformers==4.57.6", "accelerate>=1.2,<2",
            "safetensors>=0.5,<1", "PyYAML>=6,<7", "psutil>=6,<8")
        _install_source("ximinng/HiVG", addons.HIVG_COMMIT, target / "source")
        # Official inference/tokenizer only; evaluation and training dependencies are not needed.
        pip(py, "--no-deps", target / "source")
        transfers.snapshot_download(repo_id="xingxm/HiVG-3B-Base",
            revision=addons.HIVG_REVISION, local_dir=str(target / "model"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        index = json.loads((target / "model/model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            path = target / "model" / shard
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Incomplete HiVG weights. Run Install / repair again.")
        run(py, "-c", "from hivg_infer import HiSVGInferencePipeline; import resvg_py; "
            "from transformers import Qwen2_5_VLForConditionalGeneration; print('HiVG imports ready')")
        metadata = {"commit": addons.HIVG_COMMIT, "revision": addons.HIVG_REVISION}
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    print(name + " installed. No model was loaded into the interface.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["svg", "hivg", "omnisvg"])
    install(parser.parse_args().engine)
