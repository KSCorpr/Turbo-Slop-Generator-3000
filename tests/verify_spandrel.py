"""Disposable CPU check: missing backend -> real HAT x4 -> offline reuse.

Run in an environment with Torch/torchvision, but initially no Spandrel.
Uses a small random HAT checkpoint to verify execution, not image quality.
"""
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts import setup_tools

    before = {p: importlib.metadata.version(p) for p in ("torch", "torchvision")}
    try:
        importlib.metadata.version("spandrel")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Use a disposable environment with no Spandrel installed.")
    setup_tools.ensure_spandrel()

    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader
    from spandrel.architectures.HAT import HAT

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = HAT(img_size=16, embed_dim=32, depths=(2,), num_heads=(4,),
                window_size=4, squeeze_factor=8, upscale=4, num_feat=16).eval()
    with tempfile.TemporaryDirectory(prefix="Spandrel HAT ") as temporary:
        stage = Path(temporary)
        checkpoint = stage / "4x HAT fixture.pth"
        torch.save(model.state_dict(), checkpoint)
        descriptor = ModelLoader().load_from_file(checkpoint).eval()
        assert descriptor.architecture.id == "HAT" and descriptor.scale == 4
        rgb = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(8, 12, 3)
        source = Image.fromarray(rgb).convert("RGBA")
        source.putalpha(Image.fromarray(np.arange(12 * 8, dtype=np.uint8).reshape(8, 12)))
        profile = b"preserved ICC fixture"
        original = stage / "input with spaces.png"
        source.save(original, icc_profile=profile)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255
        with torch.inference_mode():
            result = descriptor(tensor)[0].numpy().transpose(1, 2, 0)
        expected = (result * 255).round().astype(np.uint8)

        # Exercise the runner's automatic installation, not only the helper.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "spandrel"], check=True)
        command = [sys.executable, str(ROOT / "scripts" / "tools" / "run_spandrel.py"),
                   "--model", str(checkpoint), "--input", str(original),
                   "--output", str(stage / "output.png"), "--full-precision"]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        # Neutral cwd also verifies the script imports its bootstrap correctly.
        for offline in (False, True):
            if offline:
                env.update(PIP_NO_INDEX="1", HF_HUB_OFFLINE="1")
            completed = subprocess.run(command, cwd=stage, env=env, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=180)
            print(completed.stdout, flush=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            if offline:
                assert "Installing the modern upscaler runtime" not in completed.stdout
            else:
                assert "Installing the modern upscaler runtime" in completed.stdout
            with Image.open(stage / "output.png") as image:
                assert image.size == (48, 32) and image.mode == "RGBA"
                assert image.info["icc_profile"] == profile
                np.testing.assert_allclose(np.asarray(image.convert("RGB")), expected, atol=1)
                assert image.getchannel("A").tobytes() == source.getchannel("A").resize(
                    image.size, Image.Resampling.LANCZOS).tobytes()
    assert {p: importlib.metadata.version(p) for p in before} == before
    print("Spandrel checks passed: automatic installation, real HAT x4 inference, "
          "48x32 RGBA/ICC output, unchanged Torch and offline reuse.")


if __name__ == "__main__":
    main()
