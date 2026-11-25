"""Juggernaut Moss Turbo checkpoint selection, shared MOSS reuse, CLI and UI contracts."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from atelier import benchmark, downloader, model_catalog, registry, sampling, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab


MODEL = "moss-image-juggernaut-turbo"
REPO = "Kandoo/Juggernaut-Moss-Turbo-Beta"
INT8 = "juggernaut_moss_turbo_v1_int8_convrot.safetensors"
FP8 = "juggernaut_moss_turbo_v1_fp8_scaled.safetensors"
BF16 = "juggernaut_moss_turbo_v1_bf16.safetensors"
W6A8 = "juggernaut_moss_turbo_v1_w6a8.safetensors"
NVFP4 = "juggernaut_moss_turbo_v1_nvfp4.safetensors"
GGUF_Q8 = "juggernaut_moss_turbo_v1-Q8_0.gguf"
SIZES = {
    INT8: 5_413_698_958,
    FP8: 5_408_351_984,
    BF16: 10_164_656_056,
    W6A8: 4_521_893_744,
    NVFP4: 3_327_470_008,
    GGUF_Q8: 5_706_346_688,
}
OPTIONS = frozenset({"--tokenizer", "--params-backend", "--max-vram"})


class JuggernautMossTurboTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key in ("MODELS_DIR", "BIN_DIR", "TMP_DIR", "USERDATA_DIR", "OUTPUT_DIR", "CUSTOM_DIR", "LORA_DIR"):
            folder = self.root / key
            folder.mkdir()
            self.stack.enter_context(patch.object(settings, key, folder))
        self.stack.enter_context(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-1117-0d71f96"}')
        self.model = registry.get_base_model(MODEL, {})
        self.components = {c.role: c for c in self.model.components}

    def install(self, component, name=None):
        path = settings.model_repo_dir(component.repo) / (name or component.requested())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weight")
        return path

    def install_all(self):
        return {c.role: self.install(c) for c in self.model.components}

    def test_catalog_shares_moss_dependencies_and_uses_author_defaults(self):
        self.assertEqual(self.model.family, "moss_image_turbo")
        self.assertIsNotNone(model_catalog.get("image:" + MODEL))
        d = self.model.defaults
        self.assertEqual((d["steps"], d["cfg_scale"], d["sampler"], d["scheduler"]),
                         (6, 1.0, "euler_a", "simple"))
        self.assertEqual((d["width"], d["height"]), (1344, 1792))
        self.assertFalse(d["supports_negative"])

        sft = {c.role: c for c in registry.get_base_model("moss-image-sft", {}).components}
        kromoss = {c.role: c for c in registry.get_base_model("moss-image-kromoss", {}).components}
        for role in ("text_encoder", "tokenizer"):
            self.assertEqual((self.components[role].repo, self.components[role].requested()),
                             (sft[role].repo, sft[role].requested()))
        self.assertEqual((self.components["vae"].repo, self.components["vae"].requested()),
                         (kromoss["vae"].repo, kromoss["vae"].requested()))
        self.assertEqual(self.components["vae"].alternatives[1].repo, sft["vae"].repo)

        self.assertEqual(self.components["diffusion"].repo, REPO)
        self.assertEqual(self.components["diffusion"].requested(), INT8)
        self.assertEqual(self.components["diffusion"].selectable, (INT8, FP8, BF16))

        for filename in (INT8, FP8, BF16, W6A8, NVFP4, GGUF_Q8, "../evil.safetensors"):
            chosen = registry.get_base_model(MODEL, {"model_files": {MODEL: {"diffusion": filename}}})
            comp = next(c for c in chosen.components if c.role == "diffusion")
            expected = filename if filename in (INT8, FP8, BF16) else INT8
            self.assertEqual(comp.requested(), expected)

    def test_menu_lists_real_metadata_and_excludes_unsupported_formats(self):
        def info(repo_id, files_metadata):
            self.assertTrue(files_metadata)
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=n, size=s, lfs=None) for n, s in SIZES.items()
            ] if repo_id == REPO else [])

        with patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = info
            menu = downloader.weight_choices(self.model)["diffusion"]
        self.assertEqual({v for _, v in menu}, {INT8, FP8, BF16})
        labels = {v: lbl for lbl, v in menu}
        self.assertIn("5.41 GB", labels[INT8])
        self.assertIn("10.16 GB", labels[BF16])

    def test_download_reuses_installed_moss_encoder_tokenizer_and_small_vae(self):
        for role in ("text_encoder", "tokenizer", "vae"):
            self.install(self.components[role])

        def transfer(repo_id, filename, local_dir):
            self.assertEqual((repo_id, filename), (REPO, INT8))
            return str(self.install(self.components["diffusion"], filename))

        with patch("huggingface_hub.list_repo_files", return_value=list(SIZES)), \
             patch("atelier.transfers.hf_hub_download", side_effect=transfer) as download:
            list(downloader.download_model(self.model))
        download.assert_called_once()
        self.assertTrue(registry.model_is_ready(self.model))

    def test_native_generation_uses_euler_a_simple_and_external_tokenizer(self):
        files = self.install_all()
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
             patch.object(sdcpp, "run") as run, \
             patch.object(sdcpp, "collect_outputs", return_value=[]):
            generate.generate(MODEL, "A cinematic street photograph", "blur", 6, 1.0,
                              1344, 1792, 42, 1, sampler="euler_a", schedule="simple",
                              prefs_override={"auto_optimize": False}, save_prompt=False)
        cmd = run.call_args.args[0]
        for flag, expected in (
            ("--diffusion-model", files["diffusion"]),
            ("--vae", files["vae"]),
            ("--llm", files["text_encoder"]),
            ("--tokenizer", files["tokenizer"]),
            ("--steps", "6"),
            ("--sampling-method", "euler_a"),
            ("--scheduler", "simple"),
            ("-W", "1344"),
            ("-H", "1792"),
        ):
            self.assertEqual(cmd[cmd.index(flag) + 1], str(expected))
        self.assertEqual(float(cmd[cmd.index("--cfg-scale") + 1]), 1.0)
        self.assertNotIn("-n", cmd)
        self.assertNotIn("--lora-apply-mode", cmd)

    def test_sampling_verdicts_benchmark_and_ui_mp_presets_and_layout_json(self):
        self.assertEqual(sampling.level("sampler", "euler_a", "moss_image_turbo"), sampling.BEST)
        self.assertEqual(sampling.level("schedule", "simple", "moss_image_turbo"), sampling.BEST)
        self.assertEqual(sampling.level("schedule", "karras", "moss_image_turbo"), sampling.BAD)
        self.assertIn("Euler Ancestral", sampling.rationale("moss_image_turbo"))

        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark, "_monitor_peak"), \
             patch.object(benchmark.generate, "generate", return_value=[]) as run:
            benchmark._one_run(MODEL, {}, self.model.defaults, None)
        self.assertEqual(run.call_args.kwargs["steps"], 6)
        self.assertEqual(run.call_args.kwargs["sampler"], "euler_a")
        self.assertEqual(run.call_args.kwargs["schedule"], "simple")

        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(MODEL, self.model.name)
        self.addCleanup(demo.close)
        controls = {b.label: b for b in demo.blocks.values()
                    if isinstance(b, (gr.Slider, gr.Dropdown, gr.Textbox))}
        self.assertEqual(controls["Sampler"].value, "euler_a")
        self.assertEqual(controls["Scheduler (sigmas)"].value, "simple")
        self.assertEqual(controls["Steps"].value, 6)
        self.assertEqual(controls["CFG"].value, 1.0)
        self.assertEqual((controls["Width"].value, controls["Height"].value), (1344, 1792))
        self.assertFalse(controls["Negative prompt"].visible)
        self.assertEqual(controls["Image size (MP)"].value, 2.3)

        on_mp = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "on_mp_preset")
        w4, h4, _, mp4 = on_mp.fn(4.0, 1344, 1792)
        self.assertEqual((w4["value"], h4["value"]), (1776, 2368))
        self.assertIn("1776×2368", mp4["info"])

        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        inputs = [c.value for c in binding.inputs]
        layout = json.dumps({"summary": {"caption": "A poster"}, "items": []})
        inputs[2] = layout
        with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
            list(binding.fn(*inputs))
        self.assertEqual(run.call_args.kwargs["prompt"], layout)

        inputs[2] = '{"items": []}'
        with patch.object(generate_tab.gen_engine, "generate") as run:
            with self.assertRaisesRegex(gr.Error, "MOSS layout JSON"):
                list(binding.fn(*inputs))
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
