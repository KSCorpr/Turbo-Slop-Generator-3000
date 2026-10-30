"""MOSS downloads, prompt preservation, placement and native CLI contracts."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import benchmark, downloader, hardware, registry, sampling, settings, system_profile
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab


MODEL = "moss-image-sft"
INT8 = "diffusion_models/moss_image_sft_int8_convrot.safetensors"
BF16 = "diffusion_models/moss_image_sft_bf16.safetensors"
OPTIONS = frozenset({"--tokenizer", "--params-backend", "--max-vram"})


class MossTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key in ("MODELS_DIR", "BIN_DIR", "TMP_DIR", "USERDATA_DIR"):
            folder = self.root / key
            folder.mkdir()
            self.stack.enter_context(patch.object(settings, key, folder))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        self.model = registry.get_base_model(MODEL, {})
        self.components = {c.role: c for c in self.model.components}
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-975-95b3090"}')

    def install(self, component, name=None, size=None):
        path = settings.model_repo_dir(component.repo) / (name or component.requested())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(b"test weight")
            if size:
                stream.truncate(size)  # sparse files, no downloaded multi-GB weights
        return path

    def install_all(self):
        return {c.role: self.install(c) for c in self.model.components}

    def test_sources_default_encoder_and_exact_shared_vae(self):
        self.assertEqual(self.model.family, "moss_image")
        self.assertEqual(self.components["diffusion"].requested(), INT8)
        self.assertEqual(self.components["diffusion"].selectable, (INT8, BF16))
        self.assertEqual(self.components["text_encoder"].requested(), "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(self.components["tokenizer"].repo, "OpenMOSS-Team/MOSS-Image-Base")
        self.assertEqual(self.components["tokenizer"].requested(), "tokenizer/tokenizer.json")
        self.assertFalse(self.components["tokenizer"].optional)
        flux = registry.get_base_model("flux2-klein-9b", {})
        vae = next(c for c in flux.components if c.role == "vae")
        self.assertEqual((vae.repo, vae.requested()),
                         (self.components["vae"].repo, self.components["vae"].requested()))
        self.assertEqual((self.model.defaults["steps"], self.model.defaults["cfg_scale"]), (50, 3))
        self.assertTrue(self.model.defaults["supports_negative"])
        self.assertEqual([p["steps"] for p in self.model.presets[:2]], [50, 8])
        self.assertTrue(self.model.presets[1]["moss_dmd"])

    def test_manual_and_model_specific_weight_choices_override_encoder_default(self):
        with patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q6_K")):
            manual = registry.get_base_model(MODEL, {"system_mode": "manual", "enc_quant": "Q6_K"})
            auto = registry.get_base_model(MODEL, {"system_mode": "auto", "enc_quant": "Q6_K"})
        self.assertEqual(next(c for c in manual.components if c.role == "text_encoder").quant, "Q6_K")
        self.assertEqual(next(c for c in auto.components if c.role == "text_encoder").quant, "Q4_K_M")
        selected = registry.get_base_model(MODEL, {"model_files": {MODEL: {
            "diffusion": BF16, "text_encoder": "Qwen3.5-9B-Q8_0.gguf"}}})
        comp = {c.role: c for c in selected.components}
        self.assertEqual(comp["diffusion"].requested(), BF16)
        self.assertEqual(comp["text_encoder"].requested(), "Qwen3.5-9B-Q8_0.gguf")
        # Do not substitute the Qwen3 / Qwen3-VL encoders used by other models.
        self.assertNotEqual(comp["text_encoder"].repo,
                            next(c for c in registry.get_base_model("qwen-image-2.1-turbo", {}).components
                                 if c.role == "text_encoder").repo)

    def test_download_fetches_three_new_components_and_reuses_flux_vae(self):
        self.install(self.components["vae"])
        seen = []

        def listing(repo):
            return [c.requested() for c in self.components.values() if c.repo == repo]

        def transfer(repo_id, filename, local_dir):
            seen.append((repo_id, filename))
            component = next(c for c in self.components.values() if c.repo == repo_id and c.requested() == filename)
            return str(self.install(component))

        with patch.object(settings, "ensure_dirs"), \
             patch("huggingface_hub.list_repo_files", side_effect=listing), \
             patch("atelier.transfers.hf_hub_download", side_effect=transfer):
            messages = list(downloader.download_model(self.model))
        self.assertTrue(registry.model_is_ready(self.model), messages)
        self.assertEqual(len(seen), 3)
        self.assertIn((self.components["tokenizer"].repo, "tokenizer/tokenizer.json"), seen)
        self.assertFalse(any(repo == self.components["vae"].repo for repo, _ in seen))

    def test_missing_tokenizer_is_required_before_any_inference(self):
        files = self.install_all()
        files["tokenizer"].unlink()
        self.assertFalse(registry.model_is_ready(self.model))
        with self.assertRaisesRegex(sdcpp.EngineError, "tokenizer"):
            generate.resolve_model_files(self.model)

    def test_native_generation_has_external_tokenizer_and_official_sampling(self):
        files = self.install_all()
        source = self.root / "reference.png"
        Image.new("RGB", (16, 16)).save(source)
        for init in (None, source):
            with self.subTest(img2img=bool(init)), \
                 patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                 patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
                 patch.object(sdcpp, "_require"), \
                 patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                 patch.object(sdcpp, "unique_output", return_value=self.root / "out.png"), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(MODEL, "A poster", "blur", 50, 3, 1024, 1024, 42, 1,
                                  init_image=init, sampler="euler", schedule="auto", flow_shift=0,
                                  prefs_override={"auto_optimize": False}, save_prompt=False)
            command = run.call_args.args[0]
            for flag, expected in (("--diffusion-model", files["diffusion"]), ("--vae", files["vae"]),
                                   ("--llm", files["text_encoder"]), ("--tokenizer", files["tokenizer"]),
                                   ("--steps", 50), ("--cfg-scale", 3), ("-n", "blur")):
                self.assertEqual(command[command.index(flag) + 1], str(expected))
            for flag in ("--llm_vision", "--scheduler", "--flow-shift", "-r", "--sigmas"):
                self.assertNotIn(flag, command)
            self.assertEqual("-i" in command, bool(init))

    def test_old_build_and_missing_tokenizer_option_fail_before_loading(self):
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-974-abcdef0"}')
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "975.*update.bat"):
                generate.generate(MODEL, "cat", "", 50, 3, 1024, 1024, 42, 1)
            resolve.assert_not_called()
        (settings.BIN_DIR / "engine-manifest.json").unlink()
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "--tokenizer.*update.bat"):
                generate.generate(MODEL, "cat", "", 50, 3, 1024, 1024, 42, 1)
            resolve.assert_not_called()


    def test_3060_placement_stages_text_weights_and_keeps_image_on_gpu(self):
        for role, size in (("diffusion", 5_413_698_958), ("text_encoder", 5_680_522_464), ("vae", 336_211_292)):
            self.install(self.components[role], size=size)
        self.install(self.components["tokenizer"])
        files = generate.resolve_model_files(self.model)
        gpu = hardware.Gpu(1, "RTX 3060", 12.0, "ampere", True)
        with patch.object(hardware, "detect_gpus", return_value=(gpu,)), \
             patch.object(hardware, "detect_ram_gb", return_value=64), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS):
            plan = system_profile.plan_for_model(self.model, files, {"system_mode": "auto"},
                                                  self.root / "sd-cli", available_gib=10)
        self.assertEqual(plan.gpu_index, 1)
        self.assertEqual(plan.params_backend, "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertFalse(plan.flags["clip_on_cpu"])

    def test_ui_preserves_layout_and_rejects_invalid_json_without_running(self):
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(MODEL, "MOSS-Image SFT")
        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        inputs = [component.value for component in binding.inputs]
        prompt = json.dumps({"summary": {"caption": "A poster"}, "items": [
            {"label": "text", "text": "NICE", "bbox": [-400, -400, -200, 400]}],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"})
        inputs[1:6] = ["watercolor", prompt, "blur", ["photo style"], ["art style"]]
        with patch.object(generate_tab.styles, "apply_photo_styles", side_effect=AssertionError("changed JSON")), \
             patch.object(generate_tab.styles, "apply_style_bank", side_effect=AssertionError("changed JSON")), \
             patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
            list(binding.fn(*inputs))
        self.assertEqual(run.call_args.kwargs["prompt"], prompt)
        self.assertEqual(run.call_args.kwargs["steps"], 50)
        self.assertEqual(run.call_args.kwargs["cfg_scale"], 3)
        for invalid in ('{"items":', '{"items": []}'):
            inputs[2] = invalid
            with patch.object(generate_tab.gen_engine, "generate") as run:
                with self.assertRaisesRegex(gr.Error, "MOSS layout JSON"):
                    list(binding.fn(*inputs))
            run.assert_not_called()

    def test_benchmark_and_sampling_help_use_the_non_distilled_preset(self):
        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark, "_monitor_peak"), \
             patch.object(benchmark.generate, "generate", return_value=[]) as run:
            benchmark._one_run(MODEL, {}, self.model.defaults, None)
        self.assertEqual(run.call_args.kwargs["steps"], 50)
        text = sampling.rationale("moss_image")
        for expected in ("50 steps", "CFG 3", "22", "Qwen3.5"):
            self.assertIn(expected, text)
        self.assertEqual(sampling.level("sampler", "euler", "moss_image"), sampling.BEST)
        self.assertEqual(sampling.level("schedule", "auto", "moss_image"), sampling.BEST)


if __name__ == "__main__":
    unittest.main()
