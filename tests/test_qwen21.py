"""Qwen 2.1 catalog and CLI contract; no model download or GPU required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier import benchmark, registry, sampling
from atelier.engine import generate, sdcpp
from atelier.ui.preview import step_frames


class QwenCatalogTests(unittest.TestCase):
    def test_turbo_uses_verified_weights_and_reuses_vae_encoder(self):
        model = registry.get_base_model("qwen-image-2.1-turbo", {
            "model_files": {"qwen-image-2.1": {"diffusion": "qwen_image_2.1-Q6_K.gguf"}}})
        self.assertEqual((model.defaults["steps"], model.defaults["cfg_scale"]), (8, 1.0))
        self.assertFalse(model.defaults["supports_negative"])
        components = {c.role: c for c in model.components}
        diffusion = components["diffusion"]
        self.assertEqual(diffusion.repo, "Comfy-Org/Qwen-Image-2.1")
        self.assertEqual(diffusion.requested(), "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors")
        self.assertEqual(len(diffusion.selectable), 2)
        self.assertNotIn("qwen_image_2.1_int8_convrot.safetensors", diffusion.requested())
        self.assertEqual(components["vae"].requested(), "vae/qwen_image_2.1_vae_bf16.safetensors")
        self.assertTrue(components["text_encoder_vision"].optional)
        self.assertEqual(model.defaults["extra_flags"], ["--sigmas", "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"])
        bf16 = registry.get_base_model(model.id, {"model_files": {model.id: {
            "diffusion": "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors"}}})
        self.assertTrue(next(c for c in bf16.components if c.role == "diffusion").requested().endswith("turbo_bf16.safetensors"))

    def test_qwen_guidance_is_not_marked_like_distilled_flux(self):
        self.assertIn("CFG 1.0", sampling.rationale("qwen21"))
        self.assertIn("8 steps", sampling.rationale("qwen21"))
        self.assertEqual(sampling.level("sampler", "euler", "qwen21"),
                         sampling.BEST)
        self.assertEqual(sampling.level("schedule", "auto", "qwen21"),
                         sampling.BEST)
        self.assertEqual(sampling.level("sampler", "euler_cfg_pp", "qwen21"),
                         sampling.OK)


class QwenEngineTests(unittest.TestCase):
    def test_turbo_rejects_old_engine_and_incorrect_sampling_before_loading_weights(self):
        with patch.object(generate.settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(generate, "_check_qwen_engine"), \
             patch.object(generate, "resolve_model_files") as resolve:
            with patch.object(generate.sdcpp, "supported_options", return_value=frozenset()):
                with self.assertRaisesRegex(sdcpp.EngineError, "--sigmas"):
                    generate.generate("qwen-image-2.1-turbo", "cat", "", 8, 1, 1024, 1024, 42, 1)
            with patch.object(generate.sdcpp, "supported_options", return_value={"--sigmas"}):
                for options in ({"steps": 40}, {"cfg_scale": 6}, {"sampler": "heun"},
                                {"schedule": "karras"}, {"flow_shift": 1.15}):
                    args = dict(model_id="qwen-image-2.1-turbo", prompt="cat", negative="",
                                steps=8, cfg_scale=1, width=1024, height=1024, seed=42, batch_count=1)
                    args.update(options)
                    with self.subTest(options=options), self.assertRaisesRegex(sdcpp.EngineError, "official Qwen Turbo"):
                        generate.generate(**args)
            resolve.assert_not_called()

    def test_benchmark_preserves_all_eight_distilled_steps(self):
        model = registry.get_base_model("qwen-image-2.1-turbo", {})
        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark, "_monitor_peak"), \
             patch.object(benchmark.generate, "generate", return_value=[]) as run:
            benchmark._one_run(model.id, {}, model.defaults, None)
        self.assertEqual(run.call_args.kwargs["steps"], 8)
        self.assertEqual(run.call_args.kwargs["cfg_scale"], 1.0)

    def test_adetailer_includes_turbo_sigmas_and_rejects_longer_steps(self):
        files = dict(model_path=None, diffusion=Path("turbo.safetensors"), vae=None,
                     enc=None, uncond=None, t5xxl=None, clip_l=None)
        with patch.object(generate.settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(generate.settings, "load_prefs", return_value={"system_mode": "manual"}), \
             patch.object(generate, "resolve_model_files", return_value=files), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(generate.sdcpp, "adetailer_supported", return_value=True), \
             patch.object(generate.sdcpp, "supported_options", return_value={"--sigmas"}), \
             patch.object(generate.sdcpp, "memory_args", return_value=[]):
            command = generate.adetailer_command("qwen-image-2.1-turbo", Path("in.png"), Path("out.png"), Path("detector"))
            self.assertIn("--sigmas", command)
            self.assertTrue(command[command.index("--sigmas") + 1].endswith(",0.0"))
            with self.assertRaisesRegex(sdcpp.EngineError, "requires 8 steps"):
                generate.adetailer_command("qwen-image-2.1-turbo", Path("in.png"), Path("out.png"), Path("detector"), steps=20)

    def test_numbered_previews_recover_every_step_after_a_polling_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "preview_%03d.png"
            for step in (0, 1, 2):
                Image.new("RGBA", (2, 2), (step, 0, 0, 255)).save(
                    str(template).replace("%03d", f"{step:03d}"))
            frames = list(step_frames(template, 0))
            self.assertEqual([idx for idx, _ in frames], [0, 1, 2])
            self.assertEqual([frame.getpixel((0, 0))[0] for _, frame in frames],
                             [0, 1, 2])
            self.assertEqual(list(step_frames(template, 3)), [])
            Image.new("RGBA", (2, 2), (3, 0, 0, 255)).save(
                str(template).replace("%03d", "003"))
            self.assertEqual([i for i, _ in step_frames(template, 3)],
                             [3])

    def test_native_edit_command_uses_vision_and_all_references(self):
        with patch.object(sdcpp, "_require"), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()):
            req = sdcpp.GenRequest(
                diffusion_model=Path("qwen_image_2.1-Q4_K.gguf"),
                vae=Path("qwen_image_2.1_vae_bf16.safetensors"),
                text_encoder=Path("Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                llm_vision=Path("mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
                ref_image=[Path("one.png"), Path("two.png")],
                prompt="Change the lettering", negative="blurred",
                cfg_scale=1.0, steps=8, width=1024, height=1024,
            )
            cmd = sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))
        self.assertEqual([cmd[i + 1] for i, v in enumerate(cmd) if v == "-r"],
                         ["one.png", "two.png"])
        for flag, value in (("--diffusion-model", "qwen_image_2.1-Q4_K.gguf"),
                            ("--vae", "qwen_image_2.1_vae_bf16.safetensors"),
                            ("--llm", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                            ("--llm_vision", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
                            ("--cfg-scale", "1.0"), ("--steps", "8")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn("-i", cmd)
        self.assertNotIn("--scheduler", cmd)
        self.assertNotIn("--strength", cmd)

    def test_preview_cli_uses_model_method_and_interval(self):
        with patch.object(sdcpp, "_require"), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()):
            for method, interval in (("vae", 5), ("proj", 1)):
                with self.subTest(method=method):
                    preview_options = ({"preview_method": method,
                                        "preview_interval": interval}
                                       if method == "vae" else {})
                    req = sdcpp.GenRequest(
                        diffusion_model=Path("model.gguf"),
                        preview_path=Path("preview_%03d.png"),
                        **preview_options)
                    cmd = sdcpp.build_gen_cmd(Path("sd-cli"), req,
                                              Path("output.png"))
                    self.assertEqual(cmd[cmd.index("--preview") + 1], method)
                    self.assertEqual(cmd[cmd.index("--preview-path") + 1],
                                     "preview_%03d.png")
                    self.assertEqual(cmd[cmd.index("--preview-interval") + 1],
                                     str(interval))

    def test_edit_without_projector_fails_but_text_generation_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = registry.get_base_model("qwen-image-2.1-turbo", {})
            paths = {}
            for role in ("diffusion", "vae", "text_encoder"):
                paths[role] = root / role
                paths[role].write_bytes(b"model")

            def resolve(component):
                return paths.get(component.role)

            prefs = {"auto_optimize": False, "flags": {}}
            with patch.object(generate.settings, "find_sd_cli", return_value=root / "sd-cli"), \
                 patch.object(generate.registry, "get_base_model", return_value=model), \
                 patch.object(generate.registry, "resolve_component_path", side_effect=resolve), \
                 patch.object(generate.settings, "BIN_DIR", root), \
                 patch.object(generate.sdcpp, "build_gen_cmd", return_value=["sd-cli"]) as build, \
                 patch.object(generate.sdcpp, "run"), \
                 patch.object(generate.sdcpp, "supported_options", return_value=frozenset({"--sigmas"})), \
                 patch.object(generate.sdcpp, "collect_outputs", return_value=[]):
                generate.generate("qwen-image-2.1-turbo", "a cat", "", 8, 1.0,
                                  1024, 1024, 1, 1, prefs_override=prefs,
                                  preview_path=root / "preview.png",
                                  save_prompt=False)
                request = build.call_args.args[1]
                self.assertEqual(request.extra_flags[0], "--sigmas")
                self.assertIsNone(request.llm_vision)
                self.assertEqual(request.preview_method, "proj")
                self.assertEqual(request.preview_interval, 1)
                generate.generate("qwen-image-2.1-turbo", "a cat", "", 8, 1.0,
                                  1024, 1024, 1, 1, prefs_override=prefs,
                                  preview_path=root / "preview.png",
                                  save_prompt=False)
                self.assertEqual(build.call_args.args[1].preview_interval, 1)
                with self.assertRaisesRegex(sdcpp.EngineError,
                                            "vision projector is missing"):
                    generate.generate("qwen-image-2.1-turbo", "edit", "", 8, 1.0,
                                      1024, 1024, 1, 1,
                                      ref_image=[root / "reference.png"],
                                      prefs_override=prefs, save_prompt=False)

    def test_outdated_official_binary_has_actionable_error(self):
        model = registry.get_base_model("qwen-image-2.1-turbo", {})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(generate.settings, "BIN_DIR", Path(tmp)):
            manifest = Path(tmp) / "engine-manifest.json"
            manifest.write_text(json.dumps({"tag": "master-895-e112ab5"}))
            with self.assertRaisesRegex(sdcpp.EngineError,
                                        "update.bat"):
                generate._check_qwen_engine(model)
            manifest.write_text(json.dumps({"tag": "master-896-e112ab5"}))
            generate._check_qwen_engine(model)
            with self.assertRaisesRegex(sdcpp.EngineError, "build 901"):
                generate._check_qwen_engine(model, Path(tmp) / "preview.png")
            manifest.write_text(json.dumps({"tag": "master-901-e112ab5"}))
            generate._check_qwen_engine(model, Path(tmp) / "preview.png")


if __name__ == "__main__":
    unittest.main()
