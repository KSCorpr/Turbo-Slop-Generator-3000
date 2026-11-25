"""Verify the MiniMax H3 Image package and its native sd-cli requests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image, PngImagePlugin

from atelier import downloader, generation_weights, hardware, model_catalog, registry, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab

MODEL = "minimax-h3-image"


class MiniMaxH3ImageTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        for name in ("MODELS_DIR", "BIN_DIR", "OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR", "CUSTOM_DIR", "LORA_DIR"):
            path = self.root / name
            path.mkdir()
            self.enterContext(patch.object(settings, name, path))
        self.enterContext(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        self.manifest = settings.BIN_DIR / "engine-manifest.json"
        self.manifest.write_text('{"tag":"master-1117-0d71f96"}', encoding="utf-8")
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
                            cfg_scale=1, width=2048, height=2048, seed=42, batch_count=1,
                            prefs_override={"auto_optimize": False}, save_prompt=False)

    def test_package_reuses_video_encoder_and_uses_its_own_image_vae(self):
        self.assertIsNotNone(model_catalog.get("image:" + MODEL))
        self.assertTrue(registry.model_is_ready(self.model))
        self.assertEqual(self.model.defaults["width"], 2048)
        self.assertEqual(self.model.defaults["height"], 2048)
        self.assertEqual(self.model.defaults["sampler"], "lcm")
        self.assertEqual(self.model.defaults["min_generation_area"], 4194304)
        self.assertEqual(set(self.files), {"diffusion", "text_encoder", "vae"})
        self.assertEqual(self.files["diffusion"].relative_to(settings.MODELS_DIR).as_posix(),
                         "Comfy-Org__MiniMax-H3-Image/diffusion_models/minimax_h3_image_pruned_w4a8.safetensors")
        self.assertEqual(self.files["vae"].name, "minimax_h3_image_vae_fp16.safetensors")
        self.assertEqual(self.files["text_encoder"].relative_to(settings.MODELS_DIR).as_posix(),
                         "PrunaAI__P-H3-Image-Comfy/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors")

        # Clean 2-tier Comfy-Org Safetensors diffusion choices (W4A8 fast default + INT8 ConvRot max quality).
        for selected in (
            "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
            "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        ):
            prefs = {"model_files": {MODEL: {"diffusion": selected}}}
            comp = next(c for c in registry.get_base_model(MODEL, prefs).components if c.role == "diffusion")
            self.assertEqual((comp.repo, comp.requested()), ("Comfy-Org/MiniMax-H3-Image", selected))

        # Clean 2-tier text encoder choices (P-H3 4B INT8 fast default + Qwen3-VL-32B Q4_K_M max precision).
        for selected, expected_repo in (
            ("text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors", "PrunaAI/P-H3-Image-Comfy"),
            ("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "leejet/MiniMax-H3-GGUF"),
        ):
            prefs = {"model_files": {MODEL: {"text_encoder": selected}}}
            comp = next(c for c in registry.get_base_model(MODEL, prefs).components if c.role == "text_encoder")
            self.assertEqual((comp.repo, comp.requested()), (expected_repo, selected))

    def test_native_generation_and_ordered_edit_use_sd_cpp_recipe_and_unlocked_steps(self):
        refs = [self.root / f"reference-{i}.png" for i in range(16)]
        for ref in refs:
            ref.touch()
        for ordered in ([], refs[::-1]):
            prompt = "Edit <Picture 1> using <Picture 2>." if ordered else self.request["prompt"]
            with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prompt": prompt, "width": 1025, "height": 1024,
                                     "min_generation_area": 0}, ref_image=ordered)
            command = run.call_args.args[0]
            for flag, value in (
                ("--diffusion-model", self.files["diffusion"]), ("--vae", self.files["vae"]),
                ("--llm", self.files["text_encoder"]), ("--sigmas", MODEL), ("--rng", "cpu"),
                ("--steps", 6), ("--sampling-method", "lcm"), ("-W", 1056), ("-p", prompt),
            ):
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertIn("--vae-tiling", command)
            self.assertNotIn("--llm_vision", command)
            self.assertEqual([command[i+1] for i, arg in enumerate(command) if arg == "-r"],
                             [str(path) for path in ordered])

        # Custom step counts omit --sigmas minimax-h3-image so sd-cli interpolates the PDD schedule across N steps.
        with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
            generate.generate(**{**self.request, "steps": 12})
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--steps") + 1], "12")
        self.assertEqual(command[command.index("-W") + 1], "2048")
        self.assertEqual(command[command.index("-H") + 1], "2048")
        self.assertNotIn("--sigmas", command)

    def test_below_4mp_requests_generate_on_4mp_canvas_and_lanczos_downscale(self):
        for target_size, expected_canvas in (((1024, 1024), (2048, 2048)), ((1248, 832), (2496, 1664))):
            with self.subTest(target_size=target_size):
                out_file = settings.OUTPUT_DIR / f"h3-{target_size[0]}x{target_size[1]}.png"
                info = PngImagePlugin.PngInfo()
                info.add_text("parameters", "minimax-h3 test metadata")
                Image.new("RGB", expected_canvas, (120, 140, 160)).save(out_file, pnginfo=info)
                logs = []
                with patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[out_file]):
                    paths = generate.generate(**{**self.request, "width": target_size[0], "height": target_size[1],
                                                 "save_prompt": True}, log=logs.append)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-W") + 1], str(expected_canvas[0]))
                self.assertEqual(command[command.index("-H") + 1], str(expected_canvas[1]))
                self.assertTrue(any(f"{expected_canvas[0]}x{expected_canvas[1]} (4 MP minimum canvas)" in line
                                    for line in logs), logs)
                self.assertEqual(paths, [out_file])
                with Image.open(out_file) as final_img:
                    self.assertEqual(final_img.size, target_size)
                    self.assertEqual(final_img.info.get("parameters"), "minimax-h3 test metadata")
                txt = out_file.with_suffix(".txt").read_text(encoding="utf-8")
                self.assertIn(f"Size: {target_size[0]}x{target_size[1]} "
                              f"(canvas {expected_canvas[0]}x{expected_canvas[1]}, Lanczos downscale)", txt)
                meta = json.loads(out_file.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual((meta["width"], meta["height"]), target_size)
                self.assertEqual((meta["canvas_width"], meta["canvas_height"]), expected_canvas)

    def test_q2_text_encoder_warns_and_ui_exposes_4mp_16mp_and_bypass_toggle(self):
        # Custom Q2_K_M file still triggers the low-precision warning if manually loaded.
        q2_custom = settings.CUSTOM_DIR / "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
        q2_custom.write_bytes(b"q2")
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False,
                             "generation_custom_files": {MODEL: {"text_encoder": "__local__:qwen3vl_32b_minimax_h3-Q2_K_M.gguf"}}})

        logs = []
        with patch.object(sdcpp, "run"), patch.object(sdcpp, "collect_outputs", return_value=[]):
            generate.generate(**self.request, encoder_override=q2_custom, log=logs.append)
        self.assertTrue(any("Q2_K_M quantization" in line and "Upgrade to Q4_K_M" in line for line in logs), logs)

        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(MODEL, self.model.name)
        self.addCleanup(demo.close)
        summaries = [b.value for b in demo.blocks.values()
                     if isinstance(b, gr.Markdown) and "model-summary" in (b.elem_classes or [])]
        self.assertIn("Q2_K_M text encoder active", summaries[0])
        width = next(b for b in demo.blocks.values() if isinstance(b, gr.Slider) and b.label == "Width")
        height = next(b for b in demo.blocks.values() if isinstance(b, gr.Slider) and b.label == "Height")
        ratio = next(b for b in demo.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Aspect ratio")
        mp_preset = next(b for b in demo.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Image size (MP)")
        sampler = next(b for b in demo.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Sampler")
        self.assertEqual(sampler.value, "lcm")
        self.assertEqual((width.value, width.maximum, width.step), (2048, 6272, 32))
        self.assertEqual((height.value, height.maximum, height.step), (2048, 6272, 32))
        self.assertEqual(ratio.value, "Square 1:1 — 2048×2048 (4 MP · default)")
        self.assertEqual(mp_preset.value, 4.0)
        mp_values = [val for _, val in mp_preset.choices]
        for expected_mp in (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0):
            self.assertIn(expected_mp, mp_values)
        on_mp = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "on_mp_preset")
        w_upd, h_upd, r_upd, mp_upd = on_mp.fn(16.0, 2720, 1536)
        self.assertEqual((w_upd["value"], h_upd["value"]), (5472, 3072))
        self.assertEqual(r_upd["value"], "Wide 16:9 — 5472×3072 (16 MP)")
        self.assertIn("5472×3072", mp_upd["info"])
        w8, h8, _, mp8 = on_mp.fn(8.0, 2048, 2048)
        self.assertEqual((w8["value"], h8["value"]), (2912, 2912))
        self.assertIn("2912×2912", mp8["info"])
        w20, h20, _, _ = on_mp.fn(20.0, 2048, 2048)
        self.assertEqual((w20["value"], h20["value"]), (4576, 4576))
        labels = [lbl for lbl, _ in ratio.choices]
        self.assertIn("Square 1:1 — 4096×4096 (16 MP · best)", labels)
        self.assertIn("Cinema 21:9 — 6272×2688 (16 MP)", labels)

        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        inputs = [c.value for c in binding.inputs]
        inputs[2] = "A studio portrait"
        toggle_idx = next(i for i, c in enumerate(binding.inputs)
                          if isinstance(c, gr.Checkbox) and "4 MP minimum canvas" in (c.label or ""))
        inputs[toggle_idx] = False
        with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
            list(binding.fn(*inputs))
        self.assertEqual(run.call_args.kwargs.get("min_generation_area"), 0)
        self.assertEqual(run.call_args.kwargs.get("sampler"), "lcm")

    def test_rejects_an_old_engine_and_unsupported_requests_before_inference(self):
        with patch.object(sdcpp, "run") as run:
            for changes, message in (
                ({"ref_image": [self.root / "ref.png"] * 17}, "at most 16"),
                ({"steps": 0}, "must be positive"),
                ({"init_image": self.root / "ref.png"}, "reference-image editor"),
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(sdcpp.EngineError, message):
                    generate.generate(**{**self.request, **changes})
            self.manifest.write_text('{"tag":"master-1089-f8417d0"}', encoding="utf-8")
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1090"):
                generate.generate(**self.request)
            self.manifest.write_text('{"tag":"master-1106-532d2c1"}', encoding="utf-8")
            with self.assertRaisesRegex(sdcpp.EngineError, "build 1110"):
                generate.generate(**self.request)
        run.assert_not_called()

    def _modern_sd_help(self) -> str:
        return (
            "--sigmas minimax-h3-image --rng --flash-attn --diffusion-fa "
            "--offload-to-cpu --max-vram --disable-prefetch --model-args "
            "--auto-fit on|off --params-backend --backend --vae-tiling "
            "--clip-on-cpu --vae-on-cpu"
        )

    def _make_realistic_h3_weights(self) -> None:
        # Sparse files matching the real W4A8 diffusion (~12.5 GiB) + FP16 VAE (~1.17 GiB).
        with open(self.files["diffusion"], "wb") as f:
            f.truncate(int(12.5 * 1024 ** 3))
        with open(self.files["vae"], "wb") as f:
            f.truncate(int(1.17 * 1024 ** 3))

    def test_4mp_and_16mp_vram_guard_on_11gb_and_12gb_gpus(self):
        self._make_realistic_h3_weights()
        rtx_2080ti = hardware.Gpu(index=0, name="NVIDIA GeForce RTX 2080 Ti", vram_gb=11.0,
                                  arch="turing", tensor_cores=True)
        rtx_3060 = hardware.Gpu(index=0, name="NVIDIA GeForce RTX 3060", vram_gb=12.0,
                                arch="ampere", tensor_cores=True)
        gtx_1080ti = hardware.Gpu(index=1, name="NVIDIA GeForce GTX 1080 Ti", vram_gb=11.0,
                                  arch="pascal", tensor_cores=False)

        with patch.object(sdcpp, "binary_help", return_value=self._modern_sd_help()):
            # 1) Office 11 GB card with manual preset that hard-pins diffusion=cuda0:
            #    must override diffusion=cuda0 -> --offload-to-cpu + --flash-attn on +
            #    --disable-prefetch + --max-vram -1.5 + attn_chunk_budget=0.0625.
            office_prefs = {
                "system_mode": "manual",
                "auto_optimize": False,
                "gpu_index": 0,
                "params_backend": "diffusion=cuda0,vae=cuda0,te=cpu",
                "flags": {"diffusion_fa": True, "offload_to_cpu": False, "vae_tiling": True},
            }
            settings.save_prefs(office_prefs)
            logs = []
            with patch.object(hardware, "detect_gpus", return_value=[rtx_2080ti]), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prefs_override": None}, log=logs.append)
            cmd = run.call_args.args[0]
            self.assertIn("--offload-to-cpu", cmd)
            self.assertIn("--vae-tiling", cmd)
            self.assertIn("--disable-prefetch", cmd)
            self.assertNotIn("--params-backend", cmd)
            self.assertNotIn("--diffusion-fa", cmd)
            self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
            self.assertEqual(cmd[cmd.index("--auto-fit") + 1], "off")
            self.assertEqual(cmd[cmd.index("--max-vram") + 1], "-1.5")
            self.assertEqual(cmd[cmd.index("--model-args") + 1], "minimax_h3_attn_chunk_budget=0.0625")
            self.assertFalse(run.call_args.kwargs.get("all_gpus"))
            self.assertEqual(run.call_args.kwargs.get("gpu_index"), 0)
            self.assertTrue(any("MiniMax H3 Image VRAM guard (2048x2048)" in line for line in logs), logs)

            # 2) Home dual-GPU (RTX 3060 12 GB + GTX 1080 Ti 11 GB) in Auto mode at 16 MP:
            #    must keep offload pinned to the 3060 (all_gpus=False), raise spare to -2 GiB.
            settings.save_prefs({"system_mode": "auto", "gpu_index": 0})
            with patch.object(hardware, "detect_gpus", return_value=[rtx_3060, gtx_1080ti]), \
                 patch.object(hardware, "free_vram_gb", return_value=11.0), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prefs_override": None, "width": 4096, "height": 4096})
            cmd = run.call_args.args[0]
            self.assertIn("--offload-to-cpu", cmd)
            self.assertIn("--disable-prefetch", cmd)
            self.assertEqual(cmd[cmd.index("--auto-fit") + 1], "off")
            self.assertEqual(cmd[cmd.index("--max-vram") + 1], "-2")
            self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
            self.assertNotIn("--diffusion-fa", cmd)
            self.assertEqual(cmd[cmd.index("--model-args") + 1], "minimax_h3_attn_chunk_budget=0.0625")
            self.assertFalse(run.call_args.kwargs.get("all_gpus"))
            self.assertEqual(run.call_args.kwargs.get("gpu_index"), 0)

            # 3) 24 GB GPU at 4 MP: fits in VRAM, does NOT force offload/disable-prefetch,
            #    still enables --flash-attn on and 0.5 GiB attention chunk budget.
            rtx_4090 = hardware.Gpu(index=0, name="NVIDIA GeForce RTX 4090", vram_gb=24.0,
                                    arch="ada", tensor_cores=True)
            settings.save_prefs({"system_mode": "manual", "auto_optimize": False, "gpu_index": 0,
                                 "flags": {"diffusion_fa": True, "offload_to_cpu": False}})
            with patch.object(hardware, "detect_gpus", return_value=[rtx_4090]), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(**{**self.request, "prefs_override": None})
            cmd = run.call_args.args[0]
            self.assertNotIn("--offload-to-cpu", cmd)
            self.assertNotIn("--disable-prefetch", cmd)
            self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
            self.assertNotIn("--diffusion-fa", cmd)
            self.assertEqual(cmd[cmd.index("--model-args") + 1], "minimax_h3_attn_chunk_budget=0.5")

    def test_h3_image_oom_rescue_retries_with_full_ram_offload_and_tiny_chunk_budget(self):
        self._make_realistic_h3_weights()
        rtx_3060 = hardware.Gpu(index=0, name="NVIDIA GeForce RTX 3060", vram_gb=12.0,
                                arch="ampere", tensor_cores=True)
        out_file = settings.OUTPUT_DIR / "h3-rescued.png"
        Image.new("RGB", (2048, 2048), (90, 100, 110)).save(out_file)

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            if len(calls) == 1:
                raise sdcpp.VramError("CUDA error: out of memory")

        logs = []
        with patch.object(sdcpp, "binary_help", return_value=self._modern_sd_help()), \
             patch.object(hardware, "detect_gpus", return_value=[rtx_3060]), \
             patch.object(sdcpp, "run", side_effect=fake_run), \
             patch.object(sdcpp, "collect_outputs", return_value=[out_file]):
            paths = generate.generate(**{**self.request, "prefs_override": None}, log=logs.append)

        self.assertEqual(paths, [out_file])
        self.assertEqual(len(calls), 2)
        retry_cmd, retry_kw = calls[1]
        self.assertIn("--offload-to-cpu", retry_cmd)
        self.assertIn("--disable-prefetch", retry_cmd)
        self.assertIn("--clip-on-cpu", retry_cmd)
        self.assertIn("--vae-on-cpu", retry_cmd)
        self.assertIn("--vae-tiling", retry_cmd)
        self.assertEqual(retry_cmd[retry_cmd.index("--max-vram") + 1], "-3")
        self.assertEqual(retry_cmd[retry_cmd.index("--model-args") + 1], "minimax_h3_attn_chunk_budget=0.03125")
        self.assertEqual(retry_cmd[retry_cmd.index("--auto-fit") + 1], "off")
        self.assertNotIn("--backend", retry_cmd)
        self.assertFalse(retry_kw.get("all_gpus"))
        self.assertTrue(any("full RAM offload" in line for line in logs), logs)

