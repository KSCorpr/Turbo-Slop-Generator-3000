#!/usr/bin/env python3
"""Explicit, isolated installs for SAM 2 and Qwen-Image-Layered."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atelier import addons, hardware, transfers
from scripts.setup_media import environment, pip, run


def install(name):
    if name not in {"sam2", "qwen-layered"}:
        raise ValueError("Choose SAM 2 or Qwen-Image-Layered.")
    target = addons.root(name)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "installed.json"
    marker.unlink(missing_ok=True)
    py = environment(name)
    gpus = hardware.detect_gpus()
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu = next((g for g in gpus if str(g.index) == selected), gpus[0] if gpus else None)
    blackwell = gpu is not None and gpu.arch == "blackwell"
    torch_version, vision, cuda = ("2.8.0", "0.23.0", "cu128") if blackwell else ("2.7.1", "0.22.1", "cu126")
    constraints = target / "constraints.txt"
    constraints.write_text(f"torch=={torch_version}+{cuda}\ntorchvision=={vision}+{cuda}\n", encoding="utf-8")
    pip(py, f"torch=={torch_version}+{cuda}", f"torchvision=={vision}+{cuda}",
        "--index-url", "https://download.pytorch.org/whl/"+cuda)
    pip(py, "-c", constraints, "transformers==4.57.6", "accelerate>=1.2,<2", "Pillow>=10.2,<13",
        "numpy>=1.26,<3", "scipy>=1.11,<2", "safetensors>=0.5,<1", "psutil>=6,<8")
    if name == "sam2":
        transfers.snapshot_download(repo_id=addons.SAM2_REPO, revision=addons.SAM2_REVISION,
            local_dir=str(target / "model"),
            allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json", "README.md"])
        if (target / "model/model.safetensors").stat().st_size != 184_305_280:
            raise RuntimeError("Incomplete SAM 2 checkpoint. Run Install / repair again.")
        run(py, "-c", "from transformers import Sam2Model, Sam2Processor; print('SAM 2 imports ready')")
        metadata = {"repo": addons.SAM2_REPO, "revision": addons.SAM2_REVISION}
    else:
        pip(py, "-c", constraints, "diffusers==0.37.1", "gguf==0.17.1", "sentencepiece>=0.2,<1")
        # No BF16 transformer shards: load only this selected Q4_K_M conversion.
        transfers.snapshot_download(repo_id=addons.QWEN_LAYERED_REPO, revision=addons.QWEN_LAYERED_REVISION,
            local_dir=str(target / "model"), allow_patterns=["model_index.json", "README.md",
                "transformer/config.json", "scheduler/*", "tokenizer/*", "processor/*", "text_encoder/*", "vae/*"])
        transfers.snapshot_download(repo_id=addons.QWEN_LAYERED_GGUF_REPO,
            revision=addons.QWEN_LAYERED_GGUF_REVISION, local_dir=str(target / "quantized"),
            allow_patterns=[addons.QWEN_LAYERED_GGUF, "README.md"])
        if (target / "quantized" / addons.QWEN_LAYERED_GGUF).stat().st_size != 13_244_770_944:
            raise RuntimeError("Incomplete Qwen-Image-Layered GGUF. Run Install / repair again.")
        index = json.loads((target / "model/text_encoder/model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            if not (target / "model/text_encoder" / shard).is_file():
                raise RuntimeError("Incomplete Qwen text encoder. Run Install / repair again.")
        run(py, ROOT / "scripts/tools/run_qwen_layered.py", "--check-imports")
        metadata = {"revision": addons.QWEN_LAYERED_REVISION,
                    "gguf_revision": addons.QWEN_LAYERED_GGUF_REVISION, "quantization": "Q4_K_M"}
    marker.write_text(json.dumps({**metadata, "cuda": cuda}), encoding="utf-8")
    print(name+" installed. No model was loaded into the interface.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["sam2", "qwen-layered"])
    install(parser.parse_args().engine)
