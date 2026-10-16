"""Krea Realism checkpoint selection, shared downloads and CLI/UI contracts."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from atelier import benchmark, downloader, hardware, registry, sampling, settings, system_profile
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab, outpaint_tab


MODEL = "krea2-realism"
REPO = "RudySen/Krea2-Realism-Checkpoint"
INT8 = "Krea2Realism_V1_INT8_ConvRot.safetensors"
FP8 = "Krea2Realism_V1.safetensors"
W4A8 = "Krea2Realism_V1_W4A8.safetensors"
SIZES = {INT8: 13_320_477_960, FP8: 13_141_769_680, W4A8: 7_927_584_736}


def preferences(filename=INT8):
    return {"system_mode": "manual", "auto_optimize": False,
            "model_files": {MODEL: {"diffusion": filename}}}


class KreaRealismTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("MODELS_DIR", "BIN_DIR", "TMP_DIR"):
            path = self.root / name
            path.mkdir()
            self.stack.enter_context(patch.object(settings, name, path))
        self.stack.enter_context(patch.object(registry, "effective_quants",
                                              return_value=("Q4_K_M", "Q8_0")))
        self.model = registry.get_base_model(MODEL, {})

    def install(self, comp, filename=None):
        filename = filename or comp.requested()
        if comp.role == "vae":
            filename = "split_files/vae/wan_2.1_vae.safetensors"
        path = settings.model_repo_dir(comp.repo) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weight")
        return path

    def test_full_checkpoint_has_author_defaults_and_shared_krea_components(self):
        self.assertEqual(self.model.family, "krea2_realism")
        d = self.model.defaults
        self.assertEqual((d["steps"], d["cfg_scale"], d["sampler"], d["scheduler"]),
                         (6, 1, "euler", "simple"))
        self.assertEqual((d["width"], d["height"]), (1152, 1792))
        self.assertFalse(d["supports_negative"])
        base = registry.get_base_model("krea2-turbo", {})
        for comp in self.model.components:
            if comp.role != "diffusion":
                shared = next(c for c in base.components if c.role == comp.role)
                self.assertEqual((comp.repo, comp.requested()),
                                 (shared.repo, shared.requested()))
        for filename in (INT8, FP8, W4A8, "../Krea2Realism_V1.safetensors"):
            model = registry.get_base_model(MODEL, preferences(filename))
            comp = next(c for c in model.components if c.role == "diffusion")
            self.assertEqual(comp.repo, REPO)
            self.assertEqual(comp.requested(), FP8 if filename == FP8 else INT8)
            self.assertEqual(set(comp.selectable), {FP8, INT8})

    def test_downloading_each_variant_reuses_all_installed_krea_dependencies(self):
        for comp in self.model.components:
            if comp.role != "diffusion":
                self.install(comp)
        for filename in (INT8, FP8):
            model = registry.get_base_model(MODEL, preferences(filename))
            comp = next(c for c in model.components if c.role == "diffusion")

            def transfer(repo_id, filename, local_dir):
                self.assertEqual(repo_id, REPO)
                return str(self.install(comp, filename))

            with patch.object(settings, "ensure_dirs"), \
                 patch("huggingface_hub.list_repo_files", return_value=list(SIZES)) as listing, \
                 patch("atelier.transfers.hf_hub_download", side_effect=transfer) as download:
                messages = list(downloader.download_model(model))
            listing.assert_called_once_with(REPO)
            self.assertEqual(download.call_args.kwargs["filename"], filename)
            download.assert_called_once()
            self.assertTrue(registry.model_is_ready(model), messages)
            self.assertEqual(generate.resolve_model_files(model)["diffusion"].name, filename)

    def test_menu_uses_real_metadata_and_excludes_w4a8(self):
        def info(repo_id, files_metadata):
            self.assertTrue(files_metadata)
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=s, lfs=None)
                                              for n, s in SIZES.items()] if repo_id == REPO else [])

        with patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = info
            menu = downloader.weight_choices(self.model)["diffusion"]
        self.assertEqual({v for _, v in menu}, {FP8, INT8})
        labels = {v: label for label, v in menu}
        self.assertIn("13.32 GB", labels[INT8])
        self.assertIn("13.14 GB", labels[FP8])

    def test_both_selected_checkpoints_reach_native_generation_with_simple_scheduler(self):
        for comp in self.model.components:
            self.install(comp)
        diffusion = next(c for c in self.model.components if c.role == "diffusion")
        self.install(diffusion, FP8)
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-968-8cd731e"}')
        for filename in (INT8, FP8):
            with self.subTest(filename=filename), \
                 patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                 patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
                 patch.object(sdcpp, "_require"), \
                 patch.object(sdcpp, "supported_options", return_value=frozenset()), \
                 patch.object(sdcpp, "unique_output", return_value=self.root / "out.png"), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(MODEL, "A person in a blue jacket", "blur", 6, 1,
                                  1152, 1792, 42, 1, sampler="euler", schedule="simple",
                                  prefs_override=preferences(filename), save_prompt=False)
            command = run.call_args.args[0]
            self.assertEqual(Path(command[command.index("--diffusion-model") + 1]).name, filename)
            for flag, expected in (("--steps", "6"), ("--scheduler", "simple"),
                                    ("--sampling-method", "euler"), ("-W", "1152"),
                                    ("-H", "1792")):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertNotIn("-n", command)
            self.assertNotIn("--lora-model-dir", command)
            self.assertIn("--vae", command)
            self.assertIn("--llm", command)

    def test_adetailer_preserves_six_steps_and_simple_schedule(self):
        for comp in self.model.components:
            self.install(comp)
        with patch.object(settings, "load_prefs", return_value=preferences()), \
             patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(sdcpp, "adetailer_supported", return_value=True), \
             patch.object(sdcpp, "memory_args", return_value=[]):
            command = generate.adetailer_command(MODEL, Path("in.png"), Path("out.png"), Path("detector"))
        self.assertEqual(command[command.index("--steps") + 1], "6")
        self.assertEqual(command[command.index("--scheduler") + 1], "simple")

    def test_old_engine_is_rejected_before_loading_thirteen_gb_checkpoint(self):
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-900-abcdef0"}')
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "Krea 2 Realism.*968.*update.bat"):
                generate.generate(MODEL, "cat", "", 6, 1, 1152, 1792, 42, 1)
        resolve.assert_not_called()

    def test_benchmark_outpaint_and_sampling_help_use_realism_defaults(self):
        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark, "_monitor_peak"), \
             patch.object(benchmark.generate, "generate", return_value=[]) as run:
            benchmark._one_run(MODEL, {}, self.model.defaults, None)
        self.assertEqual(run.call_args.kwargs["steps"], 6)
        self.assertEqual(run.call_args.kwargs["schedule"], "simple")
        self.assertEqual(outpaint_tab._defaults(MODEL)["schedule"], "simple")
        self.assertEqual(outpaint_tab._defaults(MODEL)["steps"], 6)
        self.assertEqual(sampling.level("schedule", "simple", "krea2_realism"), sampling.BEST)
        self.assertNotEqual(sampling.level("schedule", "auto", "krea2_realism"), sampling.BEST)
        self.assertIn("6 steps", sampling.rationale("krea2_realism"))

    def test_twelve_gb_card_uses_ram_and_tiling_for_large_checkpoint(self):
        files = {c.role: self.install(c) for c in self.model.components}
        with files["diffusion"].open("wb") as stream:
            stream.truncate(SIZES[INT8])  # fichier creux, aucun téléchargement
        resolved = generate.resolve_model_files(self.model)
        with patch.object(hardware, "detect_gpus", return_value=(
                hardware.Gpu(0, "RTX 3060", 12, "ampere", True),)), \
             patch.object(hardware, "detect_ram_gb", return_value=64), \
             patch.object(sdcpp, "supported_options", return_value={"--auto-fit", "--max-vram"}):
            plan = system_profile.plan_for_model(self.model, resolved,
                {**settings.DEFAULT_PREFS, "system_mode": "auto"}, Path("sd-cli"),
                width=1152, height=1792, available_gib=10.5)
        self.assertTrue(plan.auto_fit)
        self.assertTrue(plan.flags["vae_tiling"])
        self.assertEqual(plan.max_vram, "-1")

    def test_ui_has_separate_checkpoint_and_author_defaults(self):
        import app

        demo = app.build_app()
        tab = next(b for b in demo.blocks.values() if isinstance(b, gr.Tab) and b.id == MODEL)

        def inside(block):
            parent = block.parent
            while parent is not None:
                if parent is tab:
                    return True
                parent = parent.parent
            return False

        controls = {b.label: b for b in demo.blocks.values()
                    if inside(b) and isinstance(b, (gr.Slider, gr.Dropdown, gr.Textbox))}
        for label, expected in (("Steps", 6), ("CFG", 1), ("Scheduler (sigmas)", "simple"),
                                ("Width", 1152), ("Height", 1792)):
            self.assertEqual(controls[label].value, expected)
        ratio = controls["Aspect ratio"]
        self.assertIn("1152×1792", ratio.value)
        self.assertFalse(controls["Negative prompt"].visible)
        self.assertIn((1152, 1792), generate_tab._ratios_for("krea2_realism").values())


if __name__ == "__main__":
    unittest.main()
