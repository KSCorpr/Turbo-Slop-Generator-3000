"""Verify actual pinned VOSR architectures on CPU, without pretrained payloads.

Run in a Torch environment after downloading/adapting the pinned source:
    python tests/verify_vosr_runtime.py --source tools_repo/vosr/source

Checks every real DiT/VAE tensor shape against upstream checkpoint headers;
then runs actual tiny neural modules through the upstream tiled pipeline and
our strict checkpoint loader. Random tiny weights test execution, not quality.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "tools_repo/vosr/source")
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    from atelier import vosr
    from scripts.tools import run_vosr as runner
    torch.set_num_threads(2)
    torch.manual_seed(123)
    modules = runner.source_modules(args.source)
    with gzip.open(ROOT / "tests/fixtures/vosr_headers.json.gz", "rt") as stream:
        fixture = json.load(stream)
    assert fixture["source_commit"] == vosr.SOURCE_COMMIT
    assert fixture["revision"] == vosr.MODEL_REVISION
    with torch.device("meta"):
        dit = runner.make_dit(modules[0], fixture["args"])
        vae = modules[2].AutoencoderKLQwenImage2D.from_config_dict(fixture["vae_config"])
    for name, model in (("dit", dit), ("vae", vae)):
        expected = {k: v["shape"] for k, v in fixture[name].items() if k != "__metadata__"}
        actual = {k: list(v.shape) for k, v in model.state_dict().items()}
        assert actual == expected, name + " checkpoint architecture mismatch"
        print(f"{name}: all {len(actual)} checkpoint tensor names/shapes match.")
    del dit, vae

    model = runner.Model.__new__(runner.Model)
    model.device, model.dtype, model.offload, model.active = torch.device("cpu"), torch.float32, True, None
    model.tiled, model.args = modules[3], {"dinov2_size": 28, "layer_dinov2b_list": [0]}
    def make_dit():
        return modules[0].LightningDiT(input_size=4, patch_size=2, in_channels=32,
            out_channels=16, hidden_size=32, depth=1, num_heads=4, z_dims=16,
            encdim_ratio=1, use_qknorm=True, use_swiglu=True, use_rope=True, use_rmsnorm=True)
    def make_dino():
        return modules[1].DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=16, depth=2, num_heads=2)
    def make_vae():
        return modules[2].AutoencoderKLQwenImage2D(base_dim=8, z_dim=16, dim_mult=(1, 2, 2, 2), num_res_blocks=1)
    model.dit, model.vision, model.vae = make_dit(), make_dino(), make_vae()
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        # Actual strict assignment from both upstream checkpoint formats.
        for name, original, constructor, kind in (("dit", model.dit, make_dit, "safe"),
                ("vae", model.vae, make_vae, "safe"), ("dino", model.vision, make_dino, "pth")):
            file = folder / (name + (".safetensors" if kind == "safe" else ".pth"))
            state = original.state_dict()
            if kind == "safe":
                save_file({k: v.contiguous() for k, v in state.items()}, str(file))
            else:
                torch.save(state, file)
            with torch.device("meta"):
                restored = constructor()
            runner.load_weights(restored, file, torch.float32)
            assert all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items())
            if name == "dit":
                model.dit = restored
            elif name == "vae":
                model.vae = restored
            else:
                model.vision = restored
        broken = folder / "missing.safetensors"
        save_file({"wrong_key": torch.zeros(1)}, str(broken))
        try:
            runner.load_weights(model.dit, broken, torch.float32)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing checkpoint keys were accepted")
    print("Strict meta assignment: DiT/VAE safetensors and DINO weights-only mmap passed.")
    with torch.inference_mode():
        for shape, color in (((17, 25), "wavelet"), ((32, 8), "none"), ((9, 9), "adain")):
            pixels = torch.rand(1, *shape, 3)
            state = torch.random.get_rng_state().clone()
            # Real VAE encode/decode, DINO features, DiT SDPA/RoPE, overlapping
            # latent/pixel tiles and color correction. No fake neural forwards.
            with redirect_stdout(io.StringIO()):
                result = modules[-1].run_vosr2(model, pixels, 2, 42, color, 32, 16, 32, 16)
                repeated = modules[-1].run_vosr2(model, pixels, 2, 42, color, 32, 16, 32, 16)
            assert result.shape == (1, shape[0] * 2, shape[1] * 2, 3)
            assert torch.isfinite(result).all().item()
            assert 0 <= result.min() <= result.max() <= 1
            assert torch.equal(result, repeated), "fixed seed is not reproducible"
            assert torch.equal(state, torch.random.get_rng_state()), "global RNG state changed"
    print("Actual tiny tiled pipeline: rectangular/narrow inputs, all color modes, finite output, "
          "exact dimensions, repeatable local RNG passed. No pretrained GPU quality/speed test.")
    # Run the real PNG read/EXIF/alpha/export code around that same neural
    # pipeline, replacing only the absent GPU and full pretrained model load.
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        source, output = folder / "entrée…é.png", folder / "result.png"
        exif = Image.Exif(); exif[274] = 6
        Image.new("RGBA", (25, 17), (60, 120, 180, 80)).save(source, exif=exif)
        options = argparse.Namespace(addon_root=args.source.parent, input=source, output=output,
            upscale=2, dtype="fp16", color="wavelet", tile_size=128, tile_overlap=16,
            vae_tile_size=128, vae_tile_overlap=16, seed=42, keep_on_gpu=False)
        with patch.object(vosr, "weights_ready", return_value=True), \
             patch.object(runner, "require_runtime", return_value=(torch, torch.device("cpu"), torch.float32)), \
             patch.object(runner, "source_modules", return_value=modules), \
             patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)), \
             patch.object(runner, "Model", return_value=model), redirect_stdout(io.StringIO()):
            runner.generate(options)
        with Image.open(output) as restored:
            assert restored.size == (34, 50) and restored.mode == "RGBA"
            assert restored.getpixel((0, 0))[3] == 80
            assert "VOSR" in restored.info and restored.info["Software"] == "TurboSlop / VOSR 2.0"
        assert not output.with_name(output.name + ".part").exists()
    print("Real PNG export: Unicode paths, EXIF orientation, alpha, metadata and atomic output passed.")


if __name__ == "__main__":
    main()
