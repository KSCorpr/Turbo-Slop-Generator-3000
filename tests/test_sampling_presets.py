"""Catalog presets reach the UI and native command without losing model constraints."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import registry, sampling, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab


class SamplingPresetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key in ("MODELS_DIR", "USERDATA_DIR", "TMP_DIR", "BIN_DIR", "CUSTOM_DIR", "LORA_DIR", "OUTPUT_DIR"):
            path = self.root / key
            path.mkdir()
            self.stack.enter_context(patch.object(settings, key, path))
        self.stack.enter_context(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-1017-be48649"}')
        self.models = registry.load_base_models({})

    def test_every_model_has_distinct_explained_native_alternatives_and_its_original_default(self):
        for model in self.models:
            with self.subTest(model=model.id):
                first = model.presets[0]
                self.assertFalse(first.get("experimental"))
                for key, default in (("sampler", "euler"), ("scheduler", "auto"), ("steps", 8), ("cfg_scale", 1)):
                    self.assertEqual(first[key], model.defaults.get(key, default))
                extras = [p for p in model.presets if p.get("experimental")]
                self.assertGreaterEqual(len(extras), 3)
                pairs = {(p["sampler"], p["scheduler"]) for p in extras}
                self.assertEqual(len(pairs), len(extras))
                self.assertNotIn((first["sampler"], first["scheduler"]), pairs)
                self.assertEqual(len({p["name"] for p in model.presets}), len(model.presets))
                for preset in extras:
                    self.assertIn(preset["sampler"], sampling.SAMPLERS)
                    self.assertIn(preset["scheduler"], sampling.SCHEDULES)
                    self.assertNotIn(preset["sampler"], {"lcm", "tcd", "euler_cfg_pp", "euler_a_cfg_pp"})
                    self.assertGreater(len(preset["note"]), 60)
                    self.assertIn("Experimental preset", sampling.preset_description(preset))
                    self.assertEqual(preset["cfg_scale"], model.defaults["cfg_scale"])
                    self.assertFalse(preset.get("moss_dmd"))
        moss = next(m for m in self.models if m.family == "moss_image")
        self.assertTrue(moss.presets[1]["moss_dmd"])

    def test_all_alternatives_build_real_native_commands_with_fixed_qwen_sigmas(self):
        options = {"--sigmas", "--tokenizer", "--params-backend", "--max-vram"}
        reference = self.root / "reference.png"
        Image.new("RGB", (32, 32)).save(reference)
        for model in self.models:
            for comp in model.components:
                path = settings.model_repo_dir(comp.repo) / comp.requested().replace("*", "fixture")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            for preset in model.presets:
                if not preset.get("experimental"):
                    continue
                with self.subTest(model=model.id, preset=preset["name"]), \
                     patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                     patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
                     patch.object(sdcpp, "supported_options", return_value=options), \
                     patch.object(sdcpp, "run") as run, \
                     patch.object(sdcpp, "collect_outputs", return_value=[]):
                    generate.generate(model.id, "A detailed poster", "", preset["steps"], preset["cfg_scale"],
                                      1024, 1024, 42, 1, sampler=preset["sampler"], schedule=preset["scheduler"],
                                      ref_image=[reference] if model.defaults.get("requires_reference") else None,
                                      prefs_override={"auto_optimize": False}, save_prompt=False)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--sampling-method") + 1], preset["sampler"])
                    self.assertEqual(command[command.index("--steps") + 1], str(preset["steps"]))
                    self.assertEqual(float(command[command.index("--cfg-scale") + 1]), preset["cfg_scale"])
                    if preset["scheduler"] == "auto":
                        self.assertNotIn("--scheduler", command)
                    else:
                        self.assertEqual(command[command.index("--scheduler") + 1], preset["scheduler"])
                    self.assertNotIn("--flow-shift", command)
                    if model.family == "qwen21":
                        sigmas = command[command.index("--sigmas") + 1]
                        self.assertEqual(sigmas, model.defaults["extra_flags"][1])
                        self.assertEqual(len(sigmas.split(",")), 9)
                    self.assertNotIn("--lora-apply-mode", command)

    def test_preset_selection_updates_controls_and_the_actual_ui_request_for_every_model(self):
        reference = Image.new("RGB", (32, 32))
        for model in self.models:
            with gr.Blocks() as demo:
                generate_tab.build_generative_tab(model.id, model.name)
            binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
            apply = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "apply_preset")
            if model.family == "qwen21":
                self.assertTrue(binding.inputs[16].interactive)
                self.assertFalse(binding.inputs[17].interactive)
                self.assertEqual({value for _, value in binding.inputs[16].choices},
                                 {p["sampler"] for p in model.presets})
            for preset in model.presets:
                if not preset.get("experimental"):
                    continue
                with self.subTest(model=model.id, preset=preset["name"]):
                    values = {c._id: c.value for c in binding.inputs}
                    values[binding.inputs[2]._id] = "A detailed poster"
                    values[binding.inputs[18]._id] = 7.5  # stale manual shift must be cleared
                    if model.defaults.get("requires_reference"):
                        values[binding.inputs[6]._id] = reference
                    for component, update in zip(apply.outputs, apply.fn(preset["name"])):
                        if isinstance(update, dict) and "value" in update:
                            values[component._id] = update["value"]
                    self.assertIn(preset["note"], apply.fn(preset["name"])[-1]["info"])
                    values[apply.inputs[0]._id] = preset["name"]
                    with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
                        list(binding.fn(*(values[c._id] for c in binding.inputs)))
                    run.assert_called_once()
                    request = run.call_args.kwargs
                    for field, expected in (("sampler", preset["sampler"]), ("schedule", preset["scheduler"]),
                                            ("steps", preset["steps"]), ("cfg_scale", preset["cfg_scale"]),
                                            ("flow_shift", 0)):
                        self.assertEqual(request[field], expected)
                    self.assertFalse(request.get("moss_dmd"))


if __name__ == "__main__":
    unittest.main()
