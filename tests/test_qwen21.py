"""Qwen 2.1 catalog and CLI contract; no model download or GPU required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier import downloader, registry, sampling
from atelier.engine import generate, sdcpp
from atelier.ui.preview import step_frames


class QwenCatalogTests(unittest.TestCase):
    def test_real_repository_filenames_and_hardware_quants(self):
        for requested, expected in (("Q4_K_M", "Q4_K"),
                                    ("Q5_K_M", "Q5_0"),
                                    ("Q6_K", "Q6_K")):
            with self.subTest(requested=requested), \
                 patch.object(registry, "effective_quants",
                              return_value=(requested, "Q8_0")):
                model = registry.get_base_model("qwen-image-2.1", {})
            self.assertEqual(model.family, "qwen21")
            self.assertEqual(model.defaults["edit"], "full")
            components = {c.role: c for c in model.components}
            diff = components["diffusion"]
            self.assertEqual(diff.requested(), f"qwen_image_2.1-{expected}.gguf")
            self.assertEqual(downloader._pick_file(diff, [
                "qwen_image_2.1-Q4_0.gguf", "qwen_image_2.1-Q4_K.gguf",
                "qwen_image_2.1-Q5_0.gguf", "qwen_image_2.1-Q6_K.gguf",
            ]), diff.requested())
            self.assertEqual(components["vae"].requested(),
                             "vae/qwen_image_2.1_vae_bf16.safetensors")
            self.assertEqual(components["text_encoder"].requested(),
                             "Qwen3VL-8B-Instruct-Q8_0.gguf")
            self.assertEqual(components["text_encoder_vision"].requested(),
                             "mmproj-Qwen3VL-8B-Instruct-F16.gguf")
            self.assertTrue(components["text_encoder_vision"].optional)

    def test_qwen_guidance_is_not_marked_like_distilled_flux(self):
        self.assertIn("CFG 6.0", sampling.rationale("qwen21"))
        self.assertIn("40 steps", sampling.rationale("qwen21"))
        self.assertEqual(sampling.level("sampler", "euler", "qwen21"),
                         sampling.BEST)
        self.assertEqual(sampling.level("schedule", "auto", "qwen21"),
                         sampling.BEST)
        self.assertEqual(sampling.level("sampler", "euler_cfg_pp", "qwen21"),
                         sampling.OK)


class QwenEngineTests(unittest.TestCase):
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
                cfg_scale=6.0, steps=40, width=1024, height=1024,
            )
            cmd = sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))
        self.assertEqual([cmd[i + 1] for i, v in enumerate(cmd) if v == "-r"],
                         ["one.png", "two.png"])
        for flag, value in (("--diffusion-model", "qwen_image_2.1-Q4_K.gguf"),
                            ("--vae", "qwen_image_2.1_vae_bf16.safetensors"),
                            ("--llm", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                            ("--llm_vision", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
                            ("--cfg-scale", "6.0"), ("--steps", "40"),
                            ("-n", "blurred")):
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
            model = registry.get_base_model("qwen-image-2.1", {})
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
                 patch.object(generate.sdcpp, "collect_outputs", return_value=[]):
                generate.generate("qwen-image-2.1", "a cat", "", 40, 6.0,
                                  1024, 1024, 1, 1, prefs_override=prefs,
                                  preview_path=root / "preview.png",
                                  save_prompt=False)
                request = build.call_args.args[1]
                self.assertIsNone(request.llm_vision)
                self.assertEqual(request.preview_method, "proj")
                self.assertEqual(request.preview_interval, 1)
                generate.generate("qwen-image-2.1", "a cat", "", 3, 6.0,
                                  1024, 1024, 1, 1, prefs_override=prefs,
                                  preview_path=root / "preview.png",
                                  save_prompt=False)
                self.assertEqual(build.call_args.args[1].preview_interval, 1)
                with self.assertRaisesRegex(sdcpp.EngineError,
                                            "vision projector is missing"):
                    generate.generate("qwen-image-2.1", "edit", "", 40, 6.0,
                                      1024, 1024, 1, 1,
                                      ref_image=[root / "reference.png"],
                                      prefs_override=prefs, save_prompt=False)

    def test_outdated_official_binary_has_actionable_error(self):
        model = registry.get_base_model("qwen-image-2.1", {})
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
