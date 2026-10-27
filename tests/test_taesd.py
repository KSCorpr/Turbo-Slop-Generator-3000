"""Tiny autoencoder routing, exact native commands, integrity and UI snapshots."""
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr

from atelier import hardware, inventory, model_catalog, model_store, model_updates, registry, settings, system_profile, taesd
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab

OPTIONS = {"--taesd", "--taesd-preview-only", "--tokenizer", "--sigmas", "--params-backend", "--max-vram"}


class TinyAutoencoderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key in ("MODELS_DIR", "USERDATA_DIR", "TMP_DIR", "BIN_DIR", "CUSTOM_DIR"):
            path = self.root / key
            path.mkdir()
            self.stack.enter_context(patch.object(settings, key, path))
        self.stack.enter_context(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "prefs.json"))
        self.stack.enter_context(patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")))
        settings.save_prefs({"system_mode": "manual", "auto_optimize": False})
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-1017-be48649"}')

    def model(self, key="moss-image-sft"):
        return registry.get_base_model(key, {})

    def image_files(self, model):
        for comp in model.components:
            path = settings.model_repo_dir(comp.repo) / comp.requested().replace("*", "fixture")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model fixture")

    def tiny(self, key="taef2"):
        data = (key + " fixture weights").encode()
        spec = replace(taesd.SPECS[key], size=len(data), sha256=sha256(data).hexdigest())
        self.stack.enter_context(patch.dict(taesd.SPECS, {key: spec}))
        spec.path().parent.mkdir(parents=True, exist_ok=True)
        spec.path().write_bytes(data)
        return spec

    def command(self, model, mode, preview=True, **kwargs):
        output = self.root / "output.png"
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
             patch.object(sdcpp, "unique_output", return_value=output), \
             patch.object(sdcpp, "run") as run, \
             patch.object(sdcpp, "collect_outputs", return_value=[output]):
            generate.generate(model.id, "test", "", model.defaults["steps"], model.defaults["cfg_scale"],
                              1024, 1024, 42, 1, taesd_mode=mode,
                              preview_path=self.root / "preview_%03d.png" if preview else None, **kwargs)
        return run.call_args.args[0], output

    def test_all_nine_models_select_only_their_vae_architecture_and_share_weights(self):
        expected = {"flux2-klein-9b": "taef2", "flux3-klein": "taef2", "moss-image-sft": "taef2",
                    "fibo2-dev": "taef2", "fibo2-edit-turbo": "taef2", "z-image-turbo": "taef1",
                    "qwen-image-2.1-turbo": "taeqi2_1", "krea2-turbo": "taew2_1", "krea2-realism": "taew2_1"}
        for model in registry.load_base_models({}):
            self.assertEqual(taesd.for_model(model).key, expected[model.id])
        self.assertIs(taesd.for_model(self.model("flux2-klein-9b")), taesd.for_model(self.model()))
        self.assertIsNone(taesd.for_model(replace(self.model(), family="unverified")))

    def test_catalog_download_is_one_pinned_file_and_integrity_is_checked(self):
        spec = self.tiny()
        self.assertIn("taesd:taef2", dict((value, label) for label, value in model_catalog.choices(query="TAESD")))
        with patch.object(taesd.transfers, "hf_hub_download", return_value=str(spec.path())) as download:
            model_catalog.install("taesd:taef2", log=lambda _: None)
        download.assert_called_once_with(repo_id=spec.repo, filename=spec.filename,
                                         revision=spec.revision, local_dir=str(spec.path().parent))
        self.assertTrue(model_store.known(spec.path(), spec.sha256))
        spec.path().write_bytes(b"x" * spec.size)
        with self.assertRaisesRegex(ValueError, "checksum"):
            taesd.validate(spec)
        spec.path().unlink()
        with self.assertRaisesRegex(ValueError, "Download TAESD"):
            taesd.validate(spec)

    def test_preview_full_and_off_generate_different_correct_commands_and_metadata(self):
        model = self.model()
        self.image_files(model)
        spec = self.tiny()
        for mode in ("preview", "full", "off"):
            cmd, output = self.command(model, mode)
            if mode != "off":
                self.assertEqual(cmd[cmd.index("--taesd") + 1], str(spec.path()))
                self.assertEqual(cmd[cmd.index("--preview") + 1], "tae")
                self.assertEqual(json.loads(output.with_suffix(".json").read_text())["taesd_mode"], mode)
            else:
                self.assertNotIn("--taesd", cmd)
                self.assertEqual(cmd[cmd.index("--preview") + 1], "proj")
                self.assertNotIn("taesd_mode", json.loads(output.with_suffix(".json").read_text()))
            self.assertEqual("--taesd-preview-only" in cmd, mode == "preview")
            self.assertEqual("--vae" in cmd, mode != "full")
            self.assertEqual(cmd[cmd.index("--steps") + 1], "50")

    def test_full_mode_replaces_missing_vae_and_preview_only_still_requires_it(self):
        model = self.model()
        self.image_files(model)
        self.tiny()
        vae = registry.resolve_component_path(next(c for c in model.components if c.role == "vae"))
        vae.unlink()
        taesd.save_mode(model.id, "full")
        self.assertTrue(taesd.model_ready(model))
        cmd, _ = self.command(model, "full")
        self.assertNotIn("--vae", cmd)
        taesd.save_mode(model.id, "preview")
        self.assertFalse(taesd.model_ready(model))
        with self.assertRaisesRegex(sdcpp.EngineError, "missing files.*vae"):
            self.command(model, "preview")

    def test_missing_weight_old_engine_and_custom_vae_fail_before_inference(self):
        model = self.model()
        with patch.object(sdcpp, "supported_options", return_value=set()):
            with self.assertRaisesRegex(sdcpp.EngineError, "update.bat"):
                taesd.require(model, "preview", self.root / "sd-cli")
        with patch.object(sdcpp, "supported_options", return_value=OPTIONS):
            with self.assertRaisesRegex(sdcpp.EngineError, "Download TAESD"):
                taesd.require(model, "full", self.root / "sd-cli")
            self.tiny()
            with self.assertRaisesRegex(sdcpp.EngineError, "catalog VAE architecture"):
                taesd.require(model, "full", self.root / "sd-cli", self.root / "custom-vae.safetensors")
            with self.assertRaisesRegex(sdcpp.EngineError, "listed TAE mode"):
                taesd.require(model, "arbitrary", self.root / "sd-cli")

    def test_qwen_preview_retains_texture_fix_vae_and_uses_dedicated_rgba_tae(self):
        model = self.model("qwen-image-2.1-turbo")
        self.image_files(model)
        spec = self.tiny("taeqi2_1")
        cmd, _ = self.command(model, "preview")
        self.assertEqual(cmd[cmd.index("--taesd") + 1], str(spec.path()))
        self.assertIn("texture_fix", cmd[cmd.index("--vae") + 1])
        self.assertIn("--sigmas", cmd)
        self.assertIn("--taesd-preview-only", cmd)

    def test_preview_only_has_no_extra_load_for_jobs_without_a_preview(self):
        model = self.model()
        self.image_files(model)
        cmd, _ = self.command(model, "preview", preview=False)
        self.assertNotIn("--taesd", cmd)
        self.assertIn("--vae", cmd)

    def test_memory_plan_counts_both_networks_for_preview_but_only_tae_for_full(self):
        model = self.model()
        self.image_files(model)
        spec = self.tiny()
        files = generate.resolve_model_files(model)
        with patch.object(hardware, "biased_profile") as profile:
            profile.return_value.gpu = None
            profile.return_value.flags.return_value = {"vae_tiling": False}
            normal = system_profile.plan_for_model(model, files, {}, None)
            preview = system_profile.plan_for_model(model, {**files, "taesd": spec.path()}, {}, None)
            full = system_profile.plan_for_model(model, {**files, "taesd": spec.path(), "vae": None}, {}, None)
        self.assertAlmostEqual(preview.image_gib - normal.image_gib, spec.size / 2**30)
        self.assertAlmostEqual(full.image_gib, (files["diffusion"].stat().st_size + spec.size) / 2**30)

    def test_ui_captures_mode_and_persists_it_separately_for_each_model(self):
        model = self.model()
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(model.id, model.name)
        self.addCleanup(demo.close)
        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        save = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "save_generation_variants")
        inputs = [c.value for c in binding.inputs]
        inputs[2] = "An image"
        index = next(i for i, c in enumerate(binding.inputs) if c.label == "Tiny autoencoder (TAESD)")
        inputs[index] = "preview"
        save("__auto__", "__auto__", "__auto__", "preview")
        self.assertEqual(taesd.selected_mode(model.id), "preview")
        self.assertEqual(taesd.selected_mode("z-image-turbo"), "off")
        taesd.save_mode(model.id, "off")
        with patch.object(generate_tab.gen_engine, "generate", return_value=[]) as run:
            list(binding.fn(*inputs))
        self.assertEqual(run.call_args.kwargs["taesd_mode"], "preview")

    def test_inventory_and_update_check_keep_shared_pinned_tae_visible(self):
        spec = self.tiny()
        self.assertTrue(any(item.key == "taesd_taef2" and spec.path().parent in item.paths
                            for item in inventory.items()))
        row = next(row for row in model_updates.discover() if row["path"] == str(spec.path().resolve()))
        self.assertEqual(row["repo"], spec.repo)
        self.assertEqual(row["revision"], spec.revision)


if __name__ == "__main__":
    unittest.main()
