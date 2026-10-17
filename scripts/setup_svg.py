#!/usr/bin/env python3
"""Explicit installs for SVG tools; never change the application's Torch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atelier import addons, transfers
from scripts.setup_media import environment, pip, run, _install_source


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
    parser.add_argument("engine", choices=["svg", "hivg"])
    install(parser.parse_args().engine)
