"""Verify the MiniMax H3 Image package and its native sd-cli requests."""
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from atelier import model_catalog, model_store, registry, settings
from atelier.engine import generate, sdcpp

MODEL = "minimax-h3-image"


def _pack_gguf_str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_abiray_like_gguf(path: Path) -> tuple[int, bytes]:
    kvs = [
        _pack_gguf_str("general.architecture") + struct.pack("<I", 8) + _pack_gguf_str("wan"),
        _pack_gguf_str("general.quantization_version") + struct.pack("<II", 4, 2),
        _pack_gguf_str("general.file_type") + struct.pack("<II", 4, 15),
    ]
    orig_shapes = {
        "adaln_t_table": (1025, 8),
        "condition_proj.weight": (5376, 5120),
        "final_layer.adaln_proj.linear.weight": (10752, 8),
        "final_layer.video_out.weight": (576, 5376),
        "video_patch_proj.weight": (5376, 96),
        "blocks.0.adaln_proj.linear.weight": (96768, 8),
    }
    for name, dims in orig_shapes.items():
        kvs.append(
            _pack_gguf_str(f"comfy.gguf.orig_shape.{name}")
            + struct.pack("<IIQ", 9, 5, len(dims))
            + struct.pack(f"<{len(dims)}i", *dims)
        )
    kvs.append(_pack_gguf_str("config") + struct.pack("<I", 8) + _pack_gguf_str('{"pdd_steps":6}'))

    tensors = []
    offset = 0
    for name, dims in orig_shapes.items():
        flattened = dims[0] * dims[1]
        tensors.append(_pack_gguf_str(name) + struct.pack("<IQIQ", 1, flattened, 0, offset))
        offset += 64
    # A quantized 2D tensor must remain untouched.
    tensors.append(_pack_gguf_str("blocks.0.attn.qkv_proj.weight")
                   + struct.pack("<I2QIQ", 2, 5376, 21504, 12, offset))

    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    data_offset = ((len(header) + 31) // 32) * 32
    payload = bytes(range(256)) * 4
    path.write_bytes(header + b"\x00" * (data_offset - len(header)) + payload)
    return data_offset, payload


class MiniMaxH3ImageTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.manifest = settings.BIN_DIR / "engine-manifest.json"
        self.manifest.write_text('{"tag":"master-1088-32d516c"}', encoding="utf-8")
        self.enterContext(patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"))
        self.enterContext(patch.object(sdcpp, "binary_help", return_value="--sigmas minimax-h3-image --rng"))
        self.model = registry.get_base_model(MODEL, {})
        self.files = {}
        for component in self.model.components:
            path = settings.model_repo_dir(component.repo) / component.requested()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.files[component.role] = path
        self.request = dict(model_id=MODEL, prompt="A studio portrait", negative="", steps=6,
                            cfg_scale=1, width=1024, height=1024, seed=42, batch_count=1,
                            prefs_override={"auto_optimize": False}, save_prompt=False)

    def test_package_reuses_video_encoder_and_uses_its_own_image_vae(self):
        self.assertIsNotNone(model_catalog.get("image:" + MODEL))
        self.assertTrue(registry.model_is_ready(self.model))
        self.assertEqual(set(self.files), {"diffusion", "text_encoder", "vae"})
        self.assertEqual(self.files["diffusion"].parent.name, "Abiray__MiniMax-H3-Image-GGUF")
        self.assertRegex(self.files["diffusion"].name, r"^MiniMax-H3-Image-Pruned-Q[2345]_K(?:_[SM])?\.gguf$")
        self.assertEqual(self.files["vae"].name, "minimax_h3_image_vae_fp16.safetensors")
        self.assertEqual(self.files["text_encoder"].relative_to(settings.MODELS_DIR).as_posix(),
                         "leejet__MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q2_K_M.gguf")

        # Hardware profile quantizations map cleanly onto Abiray/MiniMax-H3-Image-GGUF files.
        expected_map = {
            "Q4_K_S": "MiniMax-H3-Image-Pruned-Q2_K.gguf",
            "Q4_K_M": "MiniMax-H3-Image-Pruned-Q3_K_M.gguf",
            "Q5_K_M": "MiniMax-H3-Image-Pruned-Q4_K_S.gguf",
            "Q6_K": "MiniMax-H3-Image-Pruned-Q4_K_M.gguf",
            "Q8_0": "MiniMax-H3-Image-Pruned-Q5_K_M.gguf",
        }
        for raw_quant, expected_file in expected_map.items():
            with patch.object(registry, "effective_quants", return_value=(raw_quant, "Q8_0")):
                comp = next(c for c in registry.get_base_model(MODEL, {}).components if c.role == "diffusion")
            self.assertEqual(comp.repo, "Abiray/MiniMax-H3-Image-GGUF")
            self.assertEqual(comp.requested(), expected_file)

        # Explicit GGUF quantizations and Comfy-Org Safetensors alternatives resolve to their repositories.
        for selected, expected_repo in (
            ("MiniMax-H3-Image-Pruned-Q2_K.gguf", "Abiray/MiniMax-H3-Image-GGUF"),
            ("MiniMax-H3-Image-Pruned-Q5_K_M.gguf", "Abiray/MiniMax-H3-Image-GGUF"),
            ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3-Image"),
            ("diffusion_models/minimax_h3_image_pruned_bf16.safetensors", "Comfy-Org/MiniMax-H3-Image"),
        ):
            prefs = {"model_files": {MODEL: {"diffusion": selected}}}
            comp = next(c for c in registry.get_base_model(MODEL, prefs).components if c.role == "diffusion")
            self.assertEqual((comp.repo, comp.requested()), (expected_repo, selected))

        # A retained Q4 video encoder is compatible with the automatic Q2 target.
        installed_encoder = self.files["text_encoder"]
        legacy_encoder = installed_encoder.with_name(installed_encoder.name.replace("Q2_K_M", "Q4_K_M"))
        installed_encoder.rename(legacy_encoder)
        self.assertEqual(generate.resolve_model_files(self.model, want_vision=True)["enc"], legacy_encoder)

    def test_native_generation_and_ordered_edit_use_the_fixed_pdd_recipe(self):
        refs = [self.root / f"reference-{i}.png" for i in range(16)]
        for ref in refs:
            ref.touch()
        for ordered in ([], refs[::-1]):
            prompt = "Edit <Picture 1> using <Picture 2>." if ordered else self.request["prompt"]
            with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prompt": prompt, "width": 1025}, ref_image=ordered)
            command = run.call_args.args[0]
            for flag, value in (
                ("--diffusion-model", self.files["diffusion"]), ("--vae", self.files["vae"]),
                ("--llm", self.files["text_encoder"]), ("--sigmas", MODEL), ("--rng", "cpu"),
                ("--steps", 6), ("--sampling-method", "euler"), ("-W", 1056), ("-p", prompt),
            ):
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertIn("--vae-tiling", command)
            self.assertNotIn("--llm_vision", command)
            self.assertEqual([command[i+1] for i, arg in enumerate(command) if arg == "-r"],
                             [str(path) for path in ordered])

    def test_restores_flattened_comfy_orig_shapes_in_place_before_running_sd_cli(self):
        diffusion = self.files["diffusion"]
        data_offset, payload = _build_abiray_like_gguf(diffusion)
        os.utime(diffusion, ns=(1_700_000_000_000_000_000, 1_700_000_000_123_456_789))
        fp_before = model_store.fingerprint(diffusion)

        with patch.object(sdcpp, "run"), patch.object(sdcpp, "collect_outputs", return_value=[]):
            generate.generate(**self.request)

        # File size, mtime, inode, data_offset, and tensor data payload are preserved byte-for-byte.
        self.assertEqual(model_store.fingerprint(diffusion), fp_before)
        raw = diffusion.read_bytes()
        self.assertEqual(raw[data_offset:], payload)
        # Subsequent calls are a no-op.
        self.assertEqual(sdcpp.ensure_gguf_orig_shapes(diffusion), 0)

        # Parse the patched GGUF header and verify GGML 2D shapes match stable-diffusion.cpp expectations.
        with diffusion.open("rb") as f:
            magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", f.read(24))
            self.assertEqual((magic, version, tensor_count), (b"GGUF", 3, 7))
            for _ in range(kv_count):
                klen, = struct.unpack("<Q", f.read(8))
                key = f.read(klen).decode("utf-8")
                self.assertFalse(key.startswith("comfy.gguf.orig_shape."))
                vtype, = struct.unpack("<I", f.read(4))
                self.assertTrue(sdcpp._skip_gguf_value(f, vtype))
            parsed_shapes = {}
            for _ in range(tensor_count):
                nlen, = struct.unpack("<Q", f.read(8))
                name = f.read(nlen).decode("utf-8")
                ndims, = struct.unpack("<I", f.read(4))
                ne = struct.unpack(f"<{ndims}Q", f.read(8 * ndims))
                ttype, _offset = struct.unpack("<IQ", f.read(12))
                parsed_shapes[name] = (ne, ttype)
            aligned_end = ((f.tell() + 31) // 32) * 32
            self.assertEqual(aligned_end, data_offset)

        self.assertEqual(parsed_shapes["video_patch_proj.weight"], ((96, 5376), 0))
        self.assertEqual(parsed_shapes["final_layer.video_out.weight"], ((5376, 576), 0))
        self.assertEqual(parsed_shapes["condition_proj.weight"], ((5120, 5376), 0))
        self.assertEqual(parsed_shapes["adaln_t_table"], ((8, 1025), 0))
        self.assertEqual(parsed_shapes["final_layer.adaln_proj.linear.weight"], ((8, 10752), 0))
        self.assertEqual(parsed_shapes["blocks.0.adaln_proj.linear.weight"], ((8, 96768), 0))
        self.assertEqual(parsed_shapes["blocks.0.attn.qkv_proj.weight"], ((5376, 21504), 12))

    def test_rejects_an_old_engine_and_unsupported_requests_before_inference(self):
        with patch.object(sdcpp, "run") as run:
            for changes, message in (
                ({"ref_image": [self.root / "ref.png"] * 17}, "at most 16"),
                ({"steps": 8}, "official PDD sigmas"),
                ({"init_image": self.root / "ref.png"}, "reference-image editor"),
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(sdcpp.EngineError, message):
                    generate.generate(**{**self.request, **changes})
            self.manifest.write_text('{"tag":"master-1087-f8417d0"}', encoding="utf-8")
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1088"):
                generate.generate(**self.request)
        run.assert_not_called()
