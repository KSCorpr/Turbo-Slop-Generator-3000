"""Installed variant selection reaches the real native command without fallback."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr

from atelier import generation_weights as weights, registry, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab


class GenerationWeightTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("MODELS_DIR", "USERDATA_DIR", "TMP_DIR", "BIN_DIR", "CUSTOM_DIR"):
            path = self.root / name
            path.mkdir()
            self.stack.enter_context(patch.object(settings, name, path))
        self.stack.enter_context(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-985-2e2ba3f"}')

    def install(self, repo, filename):
        path = settings.model_repo_dir(repo) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def component(self, model_id, role):
        return next(c for c in registry.get_base_model(model_id, {}).components if c.role == role)

    def binding(self, model_id):
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(model_id, model_id)
        self.addCleanup(demo.close)
        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        return demo, binding

    def test_every_catalog_model_has_installed_choices_and_no_network_queries(self):
        for model in registry.load_base_models({}):
            main = next(c for c in model.components if c.role == "diffusion")
            filename = main.requested().replace("*", "fixture")
            self.install(main.repo, filename)
            # A partial transfer and Hub cache blob must not appear as variants.
            self.install(main.repo, filename + ".download")
            self.install(main.repo, ".cache/" + filename)
            with patch("huggingface_hub.HfApi", side_effect=AssertionError("network query")):
                menu = weights.menus(model.id)[0]
            self.assertIn(filename, [value for _, value in menu["choices"]], model.id)
            self.assertEqual(len(menu["choices"]), 2, model.id)
            _, binding = self.binding(model.id)
            self.assertTrue(any(c.label == "Model variant (installed weights)" for c in binding.inputs))

    def test_cross_repository_gguf_int8_bf16_and_model_isolation(self):
        comp = self.component("krea2-turbo", "diffusion")
        gguf = self.install(comp.repo, "krea2_turbo-Q5_K_M.gguf")
        alternatives = comp.alternatives[1:]
        for variant in alternatives:
            self.install(variant.repo, variant.template)
        self.install(alternatives[0].repo, "diffusion_models/krea2_raw_bf16.safetensors")
        rows = weights.installed(registry.get_base_model("krea2-turbo", {}))["diffusion"]
        self.assertEqual(len(rows), 3)
        for filename, path, _ in rows:
            result = weights.resolve("krea2-turbo", (filename, weights.AUTO, weights.AUTO))
            self.assertEqual(result["diffusion"], path)
        self.assertIn(gguf, [p for _, p, _ in rows])
        weights.save_selection("krea2-turbo", "diffusion", alternatives[-1].template)
        self.assertEqual(weights.menus("krea2-turbo")[0]["value"], alternatives[-1].template)
        self.assertIn(gguf.name, weights.menus("krea2-turbo")[0]["choices"][0][0])

    def test_saved_encoder_and_vae_choices_survive_refresh_and_other_model_changes(self):
        model_id = "qwen-image-2.1-turbo"
        enc = self.component(model_id, "text_encoder")
        enc_name = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        enc_path = self.install(enc.repo, enc_name)
        vae = self.component(model_id, "vae").alternatives[1]
        vae_path = self.install(vae.repo, vae.template)
        weights.save_selection(model_id, "text_encoder", enc_name)
        weights.save_selection(model_id, "vae", vae.template)
        menu = weights.menus(model_id)
        self.assertEqual([row["value"] for row in menu][1:], [enc_name, vae.template])
        resolved = weights.resolve(model_id, (weights.AUTO, enc_name, vae.template))
        self.assertEqual(resolved["text_encoder"], enc_path)
        self.assertEqual(resolved["vae"], vae_path)
        weights.save_selection("moss-image-sft", "diffusion", weights.AUTO)
        self.assertEqual(weights.menus(model_id)[2]["value"], vae.template)
        weights.save_selection(model_id, "text_encoder", weights.AUTO)
        self.assertNotIn("text_encoder", settings.load_prefs()["model_files"][model_id])
        self.assertEqual(settings.load_prefs()["model_files"][model_id]["vae"], vae.template)

    def test_missing_or_foreign_explicit_weight_never_falls_back(self):
        model_id = "moss-image-sft"
        comp = self.component(model_id, "diffusion")
        self.install(comp.repo, comp.requested())
        bf16 = comp.selectable[1]
        for name in (bf16, "../secret.safetensors", "diffusion_models/fibo2_dev_bf16.safetensors"):
            with self.assertRaisesRegex(ValueError, "not installed"):
                weights.resolve(model_id, (name, weights.AUTO, weights.AUTO))
        path = self.install(comp.repo, bf16)
        weights.save_selection(model_id, "diffusion", bf16)
        path.unlink()
        menu = weights.menus(model_id)[0]
        self.assertEqual(menu["value"], bf16)
        self.assertIn("Not installed", str(menu["choices"]))

    def test_automatic_and_custom_components_do_not_inherit_another_request_selection(self):
        model_id = "moss-image-sft"
        comp = self.component(model_id, "diffusion")
        original = self.install(comp.repo, comp.requested())
        bf16 = comp.selectable[1]
        self.install(comp.repo, bf16)
        weights.save_selection(model_id, "diffusion", bf16)
        self.assertEqual(weights.resolve(model_id, (weights.AUTO,) * 3)["diffusion"], original)
        custom = settings.CUSTOM_DIR / "custom-moss.gguf"
        custom.write_bytes(b"fixture")
        value = weights.LOCAL + custom.name
        weights.save_selection(model_id, "diffusion", value)
        self.assertEqual(weights.menus(model_id)[0]["value"], value)
        self.assertEqual(weights.resolve(model_id, (value, weights.AUTO, weights.AUTO))["diffusion"], custom)
        self.assertEqual(settings.load_prefs()["model_files"][model_id]["diffusion"], bf16)
        custom.unlink()
        with self.assertRaisesRegex(ValueError, "not installed"):
            weights.resolve(model_id, (value, weights.AUTO, weights.AUTO))
        weights.save_selection(model_id, "diffusion", weights.AUTO)
        self.assertNotIn(model_id, settings.load_prefs()["generation_custom_files"])

    def test_ui_snapshot_selects_exact_native_paths_and_rejects_deleted_selection(self):
        model_id = "moss-image-sft"
        model = registry.get_base_model(model_id, {})
        for comp in model.components:
            self.install(comp.repo, comp.requested())
        comp = self.component(model_id, "diffusion")
        bf16 = comp.selectable[1]
        selected_path = self.install(comp.repo, bf16)
        enc = self.component(model_id, "text_encoder")
        enc_name = "Qwen3.5-9B-Q8_0.gguf"
        enc_path = self.install(enc.repo, enc_name)
        settings.save_prefs({"system_mode": "auto", "auto_optimize": False})
        _, binding = self.binding(model_id)
        inputs = [c.value for c in binding.inputs]
        inputs[2] = "A poster"
        inputs[next(i for i, c in enumerate(binding.inputs) if c.label == "Model variant (installed weights)")] = bf16
        inputs[next(i for i, c in enumerate(binding.inputs) if c.label == "Encoder variant (installed)")] = enc_name
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(generate.system_profile, "plan_for_model", return_value=None) as plan, \
             patch.object(sdcpp, "supported_options", return_value={"--tokenizer", "--params-backend", "--max-vram"}), \
             patch.object(sdcpp, "unique_output", return_value=self.root / "out.png"), \
             patch.object(sdcpp, "run") as run, \
             patch.object(sdcpp, "collect_outputs", return_value=[]):
            messages = list(binding.fn(*inputs))
            self.assertNotIn("Error", messages[-1][0], messages[-1])
            command = run.call_args.args[0]
            self.assertEqual(plan.call_args.args[1]["diffusion"], selected_path)
            self.assertEqual(plan.call_args.args[2]["system_mode"], "auto")
        self.assertEqual(command[command.index("--diffusion-model") + 1], str(selected_path))
        self.assertEqual(command[command.index("--llm") + 1], str(enc_path))
        selected_path.unlink()
        with patch.object(generate, "generate") as launch:
            with self.assertRaisesRegex(gr.Error, "not installed"):
                list(binding.fn(*inputs))
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
