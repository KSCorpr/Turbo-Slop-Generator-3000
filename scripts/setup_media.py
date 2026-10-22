#!/usr/bin/env python3
"""Install optional media engines without changing the main Torch environment."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlsplit
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atelier import addons, settings, seedvr2_acceleration as acceleration


def run(*args):
    print("$ " + subprocess.list2cmdline([str(a) for a in args]), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def download(url, destination, *, attempts=4):
    """Retry transient failures; keep the destination until a complete transfer.

    Source archives use GitHub's archive host directly. The original archive
    endpoint is retained as a fallback, at the same pinned commit.
    """
    if attempts < 1:
        raise ValueError("At least one download attempt is required.")
    destination = Path(destination)
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    urls = [url]
    if (parsed.scheme == "https" and parsed.hostname == "github.com"
            and len(parts) == 4 and parts[2] == "archive"
            and parts[3].endswith(".zip")):
        urls.insert(0, f"https://codeload.github.com/{parts[0]}/{parts[1]}"
                       f"/zip/{parts[3][:-4]}")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        request = urllib.request.Request(urls[attempt % len(urls)],
                                         headers={"User-Agent": "TurboSlop-media/1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
                length = response.headers.get("Content-Length")
                if length is not None and stream.tell() != int(length):
                    raise http.client.IncompleteRead(b"", max(0, int(length) - stream.tell()))
            partial.replace(destination)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = f"HTTP {exc.code}"
            exc.close()
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
            failure = type(exc).__name__
            last_error = exc
        finally:
            partial.unlink(missing_ok=True)
        if attempt + 1 < attempts:
            delay = 2 ** attempt
            print(f"Download interrupted ({failure}). Retrying "
                  f"{attempt + 2}/{attempts} in {delay}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Download failed after {attempts} attempts ({failure}). "
                       "Retry Install / repair when the connection is available.") from last_error


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                path = (destination / entry.filename).resolve()
                if not path.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")


def environment(name):
    # uv provisions its own Python: Windows embeddable Python has no venv module.
    try:
        import uv  # noqa: F401
    except ImportError:
        run(sys.executable, "-m", "pip", "install", "uv>=0.8,<1")
    py = addons.python(name)
    if not py.is_file():
        run(sys.executable, "-m", "uv", "venv", "--python", "3.12",
            "--managed-python", py.parents[1])
    return py


def pip(py, *args):
    run(sys.executable, "-m", "uv", "pip", "install", "--python", py, *args)


def install_seed():
    target = addons.root("seedvr2")
    target.mkdir(parents=True, exist_ok=True)
    (target / "installed.json").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as temp:
        stage = Path(temp)
        archive = stage / "source.zip"
        download("https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/archive/"
                 + addons.SEED_COMMIT + ".zip", archive)
        unpack(archive, stage / "unpacked")
        source = next((stage / "unpacked").iterdir())
        if (target / "source").exists():
            shutil.rmtree(target / "source")
        shutil.move(str(source), target / "source")
    py = environment("seedvr2")
    # CUDA 12.6 keeps Turing/Ampere support and does not require CUDA 13 drivers.
    constraints = target / "torch-constraints.txt"
    constraints.write_text("torch==2.7.1\ntorchvision==0.22.1\n", encoding="utf-8")
    pip(py, "torch==2.7.1", "torchvision==0.22.1",
        "--index-url", "https://download.pytorch.org/whl/cu126")
    pip(py, "-c", constraints, "-r", target / "source" / "requirements.txt")
    run(py, target / "source" / "inference_cli.py", "--help")
    (target / "installed.json").write_text(json.dumps({"commit": addons.SEED_COMMIT}))
    print("SeedVR2 installed. The selected DiT and VAE download on first use.", flush=True)


def install_seed_optimizations():
    if not addons.ready("seedvr2"):
        raise RuntimeError("Use Install / repair SeedVR2 first.")
    target = addons.root("seedvr2")
    py = addons.python("seedvr2")
    with tempfile.TemporaryDirectory(dir=target) as temporary:
        stage = Path(temporary)
        report_path = stage / "probe.json"
        command = [py, "-u", Path(acceleration.__file__), "--output", report_path]
        run(*command, "--mode", "info")
        info = json.loads(report_path.read_text(encoding="utf-8"))
        wheels, message = acceleration.wheel_plan(info)
        print("Selected GPU: " + info["gpu"]["name"], flush=True)
        if info.get("compiler"):
            print("SeedVR2 compiler: " + info["compiler"], flush=True)
        print(message, flush=True)
        failures = []
        for wheel in wheels:
            try:
                filename = unquote(Path(urlsplit(wheel.url).path).name)
                archive = stage / filename
                print("Installing " + wheel.name, flush=True)
                download(wheel.url, archive)
                with archive.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != wheel.sha256:
                    raise RuntimeError(wheel.name + " wheel checksum mismatch.")
                # No compilation, dependency resolution or Torch replacement.
                pip(py, "--no-deps", "--only-binary", ":all:", archive)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                failure = f"{wheel.name}: {exc}"
                failures.append(failure)
                print("Optional installation failed: " + failure, flush=True)
        run(*command, "--mode", "all")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["installation_failures"] = failures
        marker = target / acceleration.MARKER
        partial = marker.with_suffix(".json.part")
        partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        partial.replace(marker)
        selected = acceleration.choose_mode("auto", report)
        if selected == "sdpa":
            print("Optional acceleration unavailable: no attention backend passed the GPU test. "
                  "SeedVR2 remains ready with SDPA. See the diagnostics above; "
                  "use Attention > SDPA to skip optional checks.", flush=True)
            return
        print(f"SeedVR2 optimizations ready: {selected}. Auto rechecks the selected GPU before use.",
              flush=True)


def _install_source(repo, commit, target):
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary)
        archive = stage / "source.zip"
        download(f"https://github.com/{repo}/archive/{commit}.zip", archive)
        unpack(archive, stage / "unpacked")
        source = next((stage / "unpacked").iterdir())
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), target)


def install_vosr():
    from atelier import vosr
    target = addons.root("vosr")
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "installed.json"
    marker.unlink(missing_ok=True)
    py = environment("vosr")
    # Plain PyTorch SDPA: no Triton, xFormers, Diffusers or CUDA 13 requirement.
    constraints = target / "constraints.txt"
    constraints.write_text("torch==2.7.1+cu126\n", encoding="utf-8")
    pip(py, "torch==2.7.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126")
    pip(py, "-c", constraints, "safetensors>=0.5,<0.8", "einops>=0.8,<0.9",
        "numpy>=1.26,<3", "Pillow>=10.2,<13", "huggingface_hub>=0.26,<1",
        "requests>=2.31,<3", "filelock>=3.12", "tqdm>=4.42", "PyYAML>=6")
    _install_source(vosr.SOURCE_REPO, vosr.SOURCE_COMMIT, target / "source")
    vosr.adapt_source(target / "source")
    wrapper = ROOT / "scripts/tools/run_vosr.py"
    # Check the selected GPU and native SDPA before the 6.97 GB download.
    run(py, wrapper, "--addon-root", target, "--probe")
    vosr.download_models()
    vosr.load_args(vosr.model_dir())
    marker.write_text(json.dumps({"commit": vosr.SOURCE_COMMIT,
                                 "revision": vosr.MODEL_REVISION,
                                 "adapter": 1}), encoding="utf-8")
    print("VOSR 2.0 installed. Use Tools & 3D → VOSR 2.0 upscale. "
          "All inference runs offline; models are stored once.", flush=True)


def install_seed_convrot(precision="w4a8"):
    target = addons.root("seedvr2-convrot")
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "installed.json"
    marker.unlink(missing_ok=True)
    py = environment("seedvr2-convrot")
    wrapper = ROOT / "scripts" / "tools" / "run_seedvr2_convrot.py"
    # Comfy's pinned quant_ops disables native CUDA kernels on Torch cu126.
    # Check the driver before downloading packages or replacing the sources.
    run(py, wrapper, "--addon-root", target, "--check-driver")
    constraints = target / "torch-constraints.txt"
    # Pin the local CUDA version too: 2.13.0 alone accepts an installed cu126
    # build, so changing only --index-url can silently leave the wrong runtime.
    constraints.write_text("torch==2.13.0+cu130\ntorchvision==0.28.0+cu130\n"
                           "comfy-kitchen==0.2.37\n", encoding="utf-8")
    pip(py, "torch==2.13.0+cu130", "torchvision==0.28.0+cu130",
        "--index-url", "https://download.pytorch.org/whl/cu130")
    run(py, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "assert torch.cuda.get_device_capability(0) >= (7, 5), 'Select an RTX GPU'; "
        "print('Selected GPU:', torch.cuda.get_device_name(0))")
    _install_source("ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT",
                    addons.SEED_CONVROT_COMMIT, target / "source")
    _install_source("Comfy-Org/ComfyUI", addons.COMFY_CONVROT_COMMIT, target / "comfy")
    cli = target / "source" / "inference_cli.py"
    cli.write_text(cli.read_text(encoding="utf-8").replace(
        "script_dir + ':'", "script_dir + os.pathsep"), encoding="utf-8")
    pip(py, "-c", constraints, "-r", target / "comfy" / "requirements.txt",
        "-r", target / "source" / "requirements.txt", "imageio-ffmpeg>=0.6,<1")
    run(py, wrapper, "--addon-root", target, "--probe", "--precision", precision)
    run(py, wrapper, "--addon-root", target, "--help")
    marker.write_text(json.dumps({"commit": addons.SEED_CONVROT_COMMIT,
                                 "comfy_commit": addons.COMFY_CONVROT_COMMIT,
                                 "torch_cuda": "13.0",
                                 "gpu_tested_precision": precision}), encoding="utf-8")
    print("SeedVR2 ConvRot installed; the selected model and shared VAE download on first use. "
          "No node editor or TensorRT VAE engine is required.", flush=True)


def install_splat():
    target = addons.root("splat")
    target.mkdir(parents=True, exist_ok=True)
    (target / "installed.json").unlink(missing_ok=True)
    py = environment("splat")
    pip(py, "pycolmap==3.12.0", "Pillow>=10,<13", "plyfile>=1.1,<2")
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows" and machine in ("amd64", "x86_64"):
        asset = "brush-app-x86_64-pc-windows-msvc.zip"
    elif system == "Linux" and machine in ("amd64", "x86_64"):
        asset = "brush-app-x86_64-unknown-linux-gnu.tar.xz"
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        asset = "brush-app-aarch64-apple-darwin.tar.xz"
    else:
        raise RuntimeError("No bundled Brush release for this operating system/architecture.")
    base = f"https://github.com/ArthurBrussee/brush/releases/download/{addons.BRUSH_VERSION}/"
    with tempfile.TemporaryDirectory(dir=target) as temp:
        archive = Path(temp) / asset
        download(base + asset, archive)
        checksum = Path(temp) / "checksum.txt"
        download(base + asset + ".sha256", checksum)
        expected = checksum.read_text().split()[0].lower()
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected != actual:
            raise RuntimeError("Brush archive checksum mismatch.")
        unpack(archive, target / "brush")
    binary = addons.brush()
    if binary is None:
        raise RuntimeError(f"Brush executable missing after extraction in {target / 'brush'}. "
                           "Expected brush_app.exe on Windows or brush_app on Linux/macOS. "
                           "Run Install / repair COLMAP + Brush again.")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)
    run(binary, "--help")
    run(py, "-c", "import pycolmap; print('COLMAP', pycolmap.__version__)")
    (target / "installed.json").write_text(json.dumps({"brush": addons.BRUSH_VERSION,
                                                       "pycolmap": "3.12.0"}))
    print("COLMAP + Brush installed.", flush=True)


def install_ltx(quant):
    from atelier.engine import ltx25, sdcpp
    from atelier import downloader
    from atelier.registry import Component
    cli = settings.find_sd_cli()
    if cli is None:
        raise RuntimeError("Install sd.cpp first from System.")
    diffusion, encoder, vae = ltx25.weights(quant)
    for role, repo, filename in [("diffusion", ltx25.REPO, ltx25.DIFFUSIONS[quant]),
                                  ("vae", ltx25.OFFICIAL, ltx25.VAE)]:
        downloader.download_component(Component(role, repo, filename, None), log=print)
    if not encoder.is_file() or not encoder.stat().st_size:
        original = downloader.download_component(
            Component("text_encoder", ltx25.OFFICIAL, ltx25.ENCODER, None), log=print)
        encoder.parent.mkdir(parents=True, exist_ok=True)
        temporary = encoder.with_name(encoder.stem + ".partial.gguf")
        try:
            run(cli, "-M", "convert", "-m", original, "--type", "q4_0", "-o", temporary)
            if not temporary.is_file() or temporary.stat().st_size < 1024:
                raise RuntimeError("The text encoder conversion produced no valid file.")
            with temporary.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise RuntimeError("The converted encoder is not GGUF.")
            temporary.replace(encoder)
        finally:
            temporary.unlink(missing_ok=True)
    print("LTX 2.5 weights ready. The original encoder is kept for other precisions.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=["seedvr2", "seedvr2-convrot", "vosr", "splat", "ltx25"])
    parser.add_argument("--precision", choices=["int8", "w4a8"], default="w4a8")
    parser.add_argument("--quant", choices=["Q3_K_M", "Q4_K_M"], default="Q3_K_M")
    parser.add_argument("--optimizations", action="store_true",
                        help="Install and test optional SeedVR2 Windows attention wheels.")
    args = parser.parse_args(argv)
    if args.optimizations and args.tool != "seedvr2":
        parser.error("--optimizations is available for seedvr2 only")
    {"seedvr2": install_seed_optimizations if args.optimizations else install_seed, "splat": install_splat,
     "vosr": install_vosr,
     "seedvr2-convrot": lambda: install_seed_convrot(args.precision),
     "ltx25": lambda: install_ltx(args.quant)}[args.tool]()


if __name__ == "__main__":
    main()
