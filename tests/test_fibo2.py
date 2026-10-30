"""Fibo Dev/Edit Turbo: shared weights, native CLI, reference/JSON UI and memory."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import benchmark, downloader, hardware, registry, sampling, settings, system_profile
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab

DEV = "fibo2-dev"
TURBO = "fibo2-edit-turbo"
OPTIONS = frozenset({"--llm_vision", "--params-backend", "--auto-fit", "--max-vram", "--stream-layers"})


class FiboTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name in ("MODELS_DIR", "BIN_DIR", "TMP_DIR", "USERDATA_DIR"):
            path = self.root / name; path.mkdir()
            self.stack.enter_context(patch.object(settings, name, path))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        self.dev = registry.get_base_model(DEV, {})
        self.turbo = registry.get_base_model(TURBO, {})
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-985-2e2ba3f"}')

    def install(self, component, name=None, size=None):
        path = settings.model_repo_dir(component.repo) / (name or component.requested())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(b"test weights")
            if size:
                stream.truncate(size)
        return path

    def install_model(self, model):
        return {component.role: self.install(component) for component in model.components}

    def test_defaults_and_shared_encoder_projector_vae_with_no_extra_runtime(self):
        for model, steps, cfg in ((self.dev, 20, 5), (self.turbo, 4, 1)):
            self.assertEqual((model.defaults["steps"], model.defaults["cfg_scale"]), (steps, cfg))
            self.assertEqual(model.defaults["min_engine_build"], 985)
            self.assertEqual(model.defaults["max_references"], 10)
            self.assertEqual(model.defaults["flow_shift"], 0)
            self.assertFalse(model.defaults["supports_adetailer"])
            components = {c.role: c for c in model.components}
            self.assertEqual(set(components), {"diffusion", "vae", "text_encoder", "text_encoder_vision"})
            for role, shared in (("vae", "flux2-klein-9b"), ("text_encoder", "krea2-turbo"),
                                 ("text_encoder_vision", "krea2-turbo")):
                other = next(c for c in registry.get_base_model(shared, {}).components if c.role == role)
                self.assertEqual((components[role].repo, components[role].requested()),
                                 (other.repo, other.requested()))
        self.assertTrue(self.dev.defaults["supports_negative"])
        self.assertFalse(self.turbo.defaults["supports_negative"])
        self.assertTrue(self.turbo.defaults["requires_reference"])

    def test_bf16_selection_is_explicit_and_cannot_silently_select_other_checkpoint(self):
        filename = "diffusion_models/fibo2_dev_bf16.safetensors"
        prefs = {"model_files": {DEV: {"diffusion": filename}}}
        model = registry.get_base_model(DEV, prefs)
        self.assertEqual(model.components[0].requested(), filename)
        prefs["model_files"][DEV]["diffusion"] = "diffusion_models/fibo2_edit_turbo_bf16.safetensors"
        self.assertEqual(registry.get_base_model(DEV, prefs).components[0].requested(),
                         "diffusion_models/fibo2_dev_int8_convrot.safetensors")

    def test_download_only_fetches_new_diffusion_when_shared_files_exist(self):
        for component in self.dev.components:
            if component.role != "diffusion":
                self.install(component)
        for model in (self.dev, self.turbo):
            diffusion = model.components[0]
            with patch("huggingface_hub.list_repo_files", return_value=[diffusion.requested()]), \
                 patch("atelier.transfers.hf_hub_download",
                       side_effect=lambda **kwargs: str(self.install(diffusion))) as download:
                list(downloader.download_model(model))
            download.assert_called_once()
            self.assertEqual(download.call_args.kwargs["filename"], diffusion.requested())
            self.assertTrue(registry.model_is_ready(model))

    def test_dev_text_generation_does_not_require_or_load_vision_but_turbo_does(self):
        for model in (self.dev, self.turbo):
            files = self.install_model(model)
            files["text_encoder_vision"].unlink()
            self.assertEqual(registry.model_is_ready(model), model.id == DEV)
            self.assertIsNone(generate.resolve_model_files(model)["llm_vision"])
            with self.assertRaisesRegex(sdcpp.EngineError, "vision projector"):
                generate.resolve_model_files(model, want_vision=True)

    def test_cli_for_dev_generation_and_ordered_multi_reference_edits(self):
        refs = [self.root / f"référence-{i}.png" for i in range(10)]
        for path in refs:
            Image.new("RGB", (16, 16)).save(path)
        for model, counts in ((self.dev, (0, 1, 10)), (self.turbo, (1, 10))):
            files = self.install_model(model)
            for count in counts:
                with self.subTest(model=model.id, count=count), \
                     patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                     patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
                     patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                     patch.object(sdcpp, "run") as run, patch.object(sdcpp, "collect_outputs", return_value=[]):
                    generate.generate(model.id, "Edit <image_1>" if count else "A poster", "blur",
                        model.defaults["steps"], model.defaults["cfg_scale"], 1024, 1024, 42, 1,
                        ref_image=refs[:count] or None, prefs_override={"auto_optimize": False}, save_prompt=False)
                command = run.call_args.args[0]
                for flag, value in (("--diffusion-model", files["diffusion"]), ("--vae", files["vae"]),
                                    ("--llm", files["text_encoder"]), ("--steps", model.defaults["steps"]),
                                    ("--cfg-scale", model.defaults["cfg_scale"])):
                    self.assertEqual(command[command.index(flag) + 1], str(value))
                self.assertEqual([command[i+1] for i, arg in enumerate(command) if arg == "-r"],
                                 [str(path) for path in refs[:count]])
                self.assertEqual("--llm_vision" in command, bool(count))
                if count:
                    self.assertEqual(command[command.index("--llm_vision")+1], str(files["text_encoder_vision"]))
                for option in ("--tokenizer", "--sigmas", "--scheduler", "--flow-shift", "-i"):
                    self.assertNotIn(option, command)

    def test_old_engine_missing_references_dimensions_and_masks_fail_before_resolving_weights(self):
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "resolve_model_files") as resolve:
            (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-984-old"}')
            with self.assertRaisesRegex(sdcpp.EngineError, "985.*update.bat"):
                generate.generate(DEV, "cat", "", 20, 5, 1024, 1024, 42, 1)
            (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-985-new"}')
            for model, width, kwargs, error in (
                (TURBO, 1024, {}, "at least one reference"),
                (TURBO, 1024, {"init_image": Path("input.png")}, "at least one reference"),
                (DEV, 1025, {}, "multiples of 16"),
                (DEV, 1024, {"mask_image": Path("mask.png")}, "mask preprocessing"),
                (DEV, 1024, {"ref_image": [Path("r.png")] * 11}, "at most 10")):
                with self.subTest(error=error), self.assertRaisesRegex(sdcpp.EngineError, error):
                    generate.generate(model, "cat", "", 4, 1, width, 1024, 42, 1, **kwargs)
            resolve.assert_not_called()

    def test_text_only_benchmark_rejects_turbo_without_starting(self):
        with patch.object(registry, "model_is_ready", return_value=True), patch.object(benchmark, "_one_run") as run:
            self.assertNotEqual(benchmark._pick_model({}), TURBO)
            result = benchmark._run_case(TURBO, {}, "test")
            self.assertFalse(result["ok"])
            run.assert_not_called()

    def binding(self, model):
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(model.id, model.name)
        return demo, next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")

    def test_ui_rejects_turbo_without_reference_and_preserves_json_without_styles(self):
        for model in (self.dev, self.turbo):
            demo, binding = self.binding(model)
            inputs = [component.value for component in binding.inputs]
            caption = json.dumps({"structured_caption": {"short_description": "A poster", "text_render": []},
                                  "edit_instruction": "Make the title blue"})
            inputs[1:6] = ["watercolor", caption, "blur", ["photo"], ["art"]]
            with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
                if model.id == TURBO:
                    with self.assertRaisesRegex(gr.Error, "at least one reference"):
                        list(binding.fn(*inputs))
                    run.assert_not_called()
                inputs[6] = Image.new("RGB", (32, 32))
                with patch.object(generate_tab.styles, "apply_photo_styles", side_effect=AssertionError("mutated JSON")), \
                     patch.object(generate_tab.styles, "apply_style_bank", side_effect=AssertionError("mutated JSON")):
                    list(binding.fn(*inputs))
                self.assertEqual(run.call_args.kwargs["prompt"], caption)
                inputs[2] = "{invalid}"
                run.reset_mock()
                with self.assertRaisesRegex(gr.Error, "Fibo caption JSON"):
                    list(binding.fn(*inputs))
                run.assert_not_called()
            summaries = [block.value for block in demo.blocks.values() if isinstance(block, gr.Markdown)
                         and isinstance(block.value, str) and "model-summary" in (block.elem_classes or [])]
            if model.id == TURBO:
                self.assertNotIn("text to image", summaries[0])

    def test_ui_sends_ten_references_in_order_and_restores_native_presets(self):
        for model in (self.dev, self.turbo):
            demo, binding = self.binding(model)
            inputs = [component.value for component in binding.inputs]
            inputs[2] = "Combine <image_1> and <image_2>"
            paths = []
            for i in range(10):
                path = self.root / f"input-{i}.png"; Image.new("RGB", (16, 16), (i, 0, 0)).save(path)
                paths.append(str(path))
            self.assertIsInstance(binding.inputs[-1], gr.File)
            inputs[-1] = paths
            captured = []
            def run(**kwargs):
                captured.extend(kwargs["ref_image"])
                self.assertEqual(kwargs["steps"], model.defaults["steps"])
                self.assertEqual(kwargs["cfg_scale"], model.defaults["cfg_scale"])
                self.assertEqual(len(kwargs["ref_image"]), 10)
                for i, path in enumerate(kwargs["ref_image"]):
                    with Image.open(path) as image:
                        self.assertEqual(image.getpixel((0, 0)), (i, 0, 0))
                return []
            with patch.object(generate_tab.gen_engine, "generate", side_effect=run):
                list(binding.fn(*inputs))
            self.assertTrue(all(not path.exists() for path in captured))
            preset = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "apply_preset")
            values = preset.fn(model.presets[0]["name"])
            self.assertEqual(values[2]["value"], model.defaults["steps"])
            self.assertEqual(values[3]["value"], model.defaults["cfg_scale"])
            self.assertEqual(values[4]["value"], 0)

    def test_12gb_placement_accounts_for_projector_and_streams_bf16_from_ram(self):
        sizes = {"diffusion": 8_408_562_474, "vae": 336_211_292,
                 "text_encoder": 4_280_406_144, "text_encoder_vision": 453_974_304}
        for component in self.dev.components:
            self.install(component, size=sizes[component.role])
        files = generate.resolve_model_files(self.dev, want_vision=True)
        gpu = hardware.Gpu(1, "RTX 3060", 12.0, "ampere", True)
        with patch.object(hardware, "detect_gpus", return_value=(gpu,)), \
             patch.object(hardware, "detect_ram_gb", return_value=64), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS):
            plan = system_profile.plan_for_model(self.dev, files, {"system_mode": "auto"},
                                                self.root / "sd-cli", available_gib=10, has_reference=True)
            bf16 = self.install(self.dev.components[0],
                                "diffusion_models/fibo2_dev_bf16.safetensors", size=16_452_934_194)
            bf16_plan = system_profile.plan_for_model(self.dev, {**files, "diffusion": bf16},
                {"system_mode": "auto"}, self.root / "sd-cli", available_gib=11, has_reference=True)
        self.assertEqual(plan.gpu_index, 1)
        self.assertTrue(plan.auto_fit)
        self.assertFalse(plan.flags["clip_on_cpu"])
        self.assertFalse(plan.params_backend)
        self.assertTrue(bf16_plan.auto_fit)
        self.assertGreater(bf16_plan.image_gib, 15)
        self.assertIn("20 steps", sampling.rationale("fibo2"))
        self.assertIn("4 steps", sampling.rationale("fibo2_turbo"))
        self.assertEqual(sampling.level("sampler", "euler", "fibo2"), sampling.BEST)


if __name__ == "__main__":
    unittest.main()
