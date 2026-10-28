"""MOSS DMD: pinned real checkpoint header; sparse payload, no GPU/weights."""
import gzip
import json
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from atelier import model_catalog, moss_dmd, registry, settings, system_profile, hardware
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab
from tests import test_moss

MODEL, OPTIONS = test_moss.MODEL, test_moss.OPTIONS


class MossDMDTests(unittest.TestCase):
    install = test_moss.MossTests.install
    install_all = test_moss.MossTests.install_all

    def setUp(self):
        test_moss.MossTests.setUp(self)
        self.stack.enter_context(patch.object(settings, "LORA_DIR", self.root / "loras"))

    def adapter(self, mutate=None):
        source = Path(__file__).parent / "fixtures/moss_dmd_header.json.gz"
        header = json.loads(gzip.decompress(source.read_bytes()))
        if mutate:
            mutate(header)
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        payload = max(v["data_offsets"][1] for v in header.values())
        path = moss_dmd.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(raw)) + raw)
            stream.truncate(len(raw) + 8 + payload)
        self.stack.enter_context(patch.object(moss_dmd, "FILE_SIZE", path.stat().st_size))
        return path

    def test_real_checkpoint_header_and_corrupt_adapter_are_checked(self):
        path = self.adapter()
        moss_dmd.require()
        self.assertTrue(moss_dmd.ready())
        with path.open("r+b") as stream:
            stream.truncate(50)
        self.assertFalse(moss_dmd.ready())
        with self.assertRaisesRegex(ValueError, "Download"):
            moss_dmd.require()
        self.adapter(lambda h: h.pop("diffusion_model.layers.41.proj.q.alpha"))
        with self.assertRaisesRegex(ValueError, "756 tensors"):
            moss_dmd.require()
        self.adapter(lambda h: h["diffusion_model.layers.0.proj.q.alpha"].update(dtype="BF16"))
        with self.assertRaisesRegex(ValueError, "type/shape"):
            moss_dmd.require()

    def test_catalog_installs_only_the_pinned_adapter_and_uses_existing_paths(self):
        path = self.adapter()
        entry = model_catalog.get("moss:dmd")
        self.assertIn("1.19 GB", entry.note)
        with patch.object(moss_dmd.transfers, "hf_hub_download", return_value=str(path)) as fetch:
            model_catalog.install("moss:dmd", log=lambda _: None)
        fetch.assert_called_once_with(repo_id=moss_dmd.REPO, filename="lora_latest.safetensors",
                                      revision=moss_dmd.REVISION, local_dir=str(path.parent))
        self.assertTrue(model_catalog.ready("moss:dmd"))
        self.assertTrue(any(value == "moss:dmd" for _, value in model_catalog.choices(query="DMD")))

    def test_missing_adapter_and_unsupported_apply_mode_stop_before_loading(self):
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS | {"--lora-apply-mode"}), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "Download.*Models"):
                generate.generate(MODEL, "cat", "", 8, 1, 1024, 1024, 42, 1, moss_dmd=True)
            resolve.assert_not_called()
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "at_runtime.*update.bat"):
                generate.generate(MODEL, "cat", "", 8, 1, 1024, 1024, 42, 1, moss_dmd=True)
            resolve.assert_not_called()

    def test_native_command_forces_unmerged_lora_and_switching_back_removes_it(self):
        self.install_all()
        self.adapter()
        commands = []
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS | {"--lora-apply-mode"}), \
             patch.object(sdcpp, "unique_output", return_value=self.root / "out.png"), \
             patch.object(sdcpp, "run", side_effect=lambda cmd, **_: commands.append(cmd)), \
             patch.object(sdcpp, "collect_outputs", return_value=[]):
            for fast, steps, cfg in ((True, 8, 1), (False, 50, 3)):
                generate.generate(MODEL, "A poster", "blur", steps, cfg, 1024, 1024, 42, 1,
                                  moss_dmd=fast, sampler="euler", schedule="auto",
                                  prefs_override={"auto_optimize": False}, save_prompt=False)
        fast, standard = commands
        self.assertEqual(fast[fast.index("--lora-apply-mode") + 1], "at_runtime")
        self.assertEqual(fast[fast.index("-p") + 1],
                         "A poster <lora:moss-image-sft-dmd/lora_latest:0.7>")
        self.assertEqual(fast[fast.index("--lora-model-dir") + 1], str(settings.LORA_DIR))
        self.assertNotIn("-n", fast)
        self.assertNotIn("--lora-apply-mode", standard)
        self.assertNotIn("--lora-model-dir", standard)
        self.assertEqual(standard[standard.index("-p") + 1], "A poster")

    def test_wrong_settings_or_model_never_launch_a_fake_eight_step_mode(self):
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS | {"--lora-apply-mode"}), \
             patch.object(generate, "resolve_model_files") as resolve:
            for kwargs in ({"steps": 50}, {"cfg_scale": 3}, {"sampler": "lcm"},
                           {"schedule": "karras"}, {"flow_shift": 2}):
                params = dict(model_id=MODEL, prompt="cat", negative="", steps=8, cfg_scale=1,
                              width=1024, height=1024, seed=42, batch_count=1, moss_dmd=True)
                params.update(kwargs)
                with self.assertRaisesRegex(sdcpp.EngineError, "Use DMD fast"):
                    generate.generate(**params)
            with self.assertRaisesRegex(sdcpp.EngineError, "only for MOSS"):
                generate.generate("z-image-turbo", "cat", "", 8, 1, 1024, 1024, 42, 1, moss_dmd=True)
            resolve.assert_not_called()

    def test_ui_preset_selects_real_dmd_mode_and_standard_preset_clears_it(self):
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(MODEL, "MOSS-Image SFT")
        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        apply = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "apply_preset")
        inputs = [component.value for component in binding.inputs]
        inputs[2] = '{"summary":{"caption":"A poster"},"items":[]}'
        for preset, fast in (("DMD fast (8 steps)", True), ("Recommended (50 steps)", False)):
            sampler, schedule, steps, cfg, shift, note = apply(preset)
            inputs[14:18] = [steps["value"], cfg["value"], sampler["value"], schedule["value"]]
            inputs[-1] = preset
            inputs[18] = shift["value"]
            self.assertEqual(shift["value"], 0)
            with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
                list(binding.fn(*inputs))
            self.assertEqual(run.call_args.kwargs.get("moss_dmd", False), fast)
            self.assertEqual(run.call_args.kwargs["steps"], 8 if fast else 50)
            self.assertEqual(run.call_args.kwargs["prompt"], inputs[2])

    def test_adapter_size_is_included_in_memory_plan(self):
        for role, size in (("diffusion", 5_413_698_958), ("text_encoder", 5_680_522_464), ("vae", 336_211_292)):
            self.install(self.components[role], size=size)
        self.install(self.components["tokenizer"])
        adapter = self.adapter()
        files = generate.resolve_model_files(self.model)
        gpu = hardware.Gpu(1, "RTX 3060", 12, "ampere", True)
        with patch.object(hardware, "detect_gpus", return_value=(gpu,)), \
             patch.object(hardware, "detect_ram_gb", return_value=64), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS | {"--auto-fit"}):
            standard = system_profile.plan_for_model(self.model, files, {"system_mode": "auto"},
                                                     self.root / "sd-cli", available_gib=7.5)
            fast = system_profile.plan_for_model(self.model, {**files, "image_adapter": adapter},
                                                 {"system_mode": "auto"}, self.root / "sd-cli", available_gib=7.5)
        self.assertFalse(standard.auto_fit)
        self.assertTrue(fast.auto_fit)

    def test_metadata_keeps_clean_json_and_records_the_adapter(self):
        prompt = '{"summary":{"caption":"A poster"},"items":[]}'
        output = self.root / "out.png"
        req = sdcpp.GenRequest(prompt=prompt + f" <lora:{moss_dmd.LORA_NAME}:0.7>", steps=8, cfg_scale=1)
        generate._write_prompt_sidecars([output], req, self.model, 42)
        data = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(data["prompt"], prompt)
        self.assertTrue(data["moss_dmd"])
        self.assertEqual(data["dmd_lora_strength"], 0.7)
