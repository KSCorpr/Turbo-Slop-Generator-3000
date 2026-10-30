#!/usr/bin/env python3
"""One isolated CUDA runtime, only the selected splat model's pinned artifacts."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atelier import addons, hardware, image_splats as models, settings
from scripts.setup_media import environment, pip, run, _install_source


def torch_packages():
    index = models.selected_gpu_index()
    gpu = next((g for g in hardware.detect_gpus() if g.index == index), None)
    if gpu and gpu.arch == "blackwell":
        return "2.7.1+cu128", "0.22.1+cu128", "https://download.pytorch.org/whl/cu128"
    # CUDA 12.4 includes sm_61 and sm_75; no CUDA 13 driver or compiler required.
    return "2.6.0+cu124", "0.21.0+cu124", "https://download.pytorch.org/whl/cu124"


def install(backend):
    models.artifacts(backend)
    if backend == "idol":
        models.check_encoder_access()
    selected = models.selected_gpu_index()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if selected is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected)
    target = models.root()
    target.mkdir(parents=True, exist_ok=True)
    models.marker(backend).unlink(missing_ok=True)
    py = environment("image-splats")
    torch, vision, index = torch_packages()
    constraints = target / "constraints.txt"
    constraints.write_text(f"torch=={torch}\ntorchvision=={vision}\n", encoding="utf-8")
    pip(py, f"torch=={torch}", f"torchvision=={vision}", "--index-url", index)
    pip(py, "-c", constraints, "numpy>=1.26,<3", "Pillow>=10.2,<13", "safetensors==0.6.2", "tqdm>=4.66,<5")
    if backend == "idol":
        pip(py, "-c", constraints, "transformers==4.57.6", "diffusers==0.36.0", "tensordict==0.10.0",
            "plyfile==1.1.3", "huggingface_hub==0.36.2")
    (target / "sources").mkdir(exist_ok=True)
    _install_source(*models.SOURCES[backend], models.source(backend))
    run(py, ROOT / "scripts/tools/run_image_splats.py", "--probe", "--backend", backend,
        "--source", models.source(backend))
    run(sys.executable, "-m", "uv", "pip", "check", "--python", py)
    models.download_models(backend)
    (target / "installed.json").write_text(json.dumps({"torch": torch, "torchvision": vision}), encoding="utf-8")
    models.marker(backend).write_text(json.dumps(models.identity(backend)), encoding="utf-8")
    print(f"{models.BACKENDS[backend]} installed. Use Tools & 3D → Image → splats.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=models.BACKENDS)
    install(parser.parse_args().backend)
