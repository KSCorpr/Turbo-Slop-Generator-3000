"""Flux.3 catalog, weight reuse, UI and CLI contracts; no GPU inference."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import benchmark, downloader, registry, sampling, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab, outpaint_tab


MODEL = "flux3-klein"
WEIGHT = "flux3_image_klein_sd-Q5_K_M.gguf"


class Flux3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("MODELS_DIR", "BIN_DIR", "TMP_DIR"):
            directory = self.root / name
            directory.mkdir()
            self.stack.enter_context(patch.object(settings, name, directory))
        self.stack.enter_context(patch.object(registry, "effective_quants",
                                              return_value=("Q4_K_M", "Q8_0")))
        self.model = registry.get_base_model(MODEL, {})

    def install(self, component, filename=None):
        path = settings.model_repo_dir(component.repo) / (filename or component.requested())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weight")
        return path

    def test_correct_checkpoint_defaults_and_shared_components(self):
        components = {c.role: c for c in self.model.components}
        self.assertEqual(self.model.family, "flux3")
        self.assertEqual(components["diffusion"].repo, "akhaliq/flux-3-image-klein-sd-gguf")
        self.assertEqual(components["diffusion"].requested(), WEIGHT)
        self.assertEqual((self.model.defaults["steps"], self.model.defaults["cfg_scale"]), (16, 1))
        self.assertEqual(self.model.defaults["flow_shift"], 0)
        self.assertFalse(self.model.defaults["supports_negative"])
        self.assertNotIn("text_encoder_vision", components)
        for role, shared_model in (("vae", "flux2-klein-9b"), ("text_encoder", "krea2-turbo")):
            shared = next(c for c in registry.get_base_model(shared_model, {}).components
                          if c.role == role)
            self.assertEqual((components[role].repo, components[role].requested()),
                             (shared.repo, shared.requested()))
        with patch.object(registry, "effective_quants", return_value=("Q5_K_M", "Q8_0")):
            model = registry.get_base_model(MODEL, {})
            self.assertEqual(next(c for c in model.components if c.role == "diffusion").requested(),
                             "flux3_image_klein_sd-Q8_0.gguf")

    def test_download_only_fetches_new_diffusion_and_reuses_existing_encoder_quant(self):
        components = {c.role: c for c in self.model.components}
        self.install(components["vae"])
        encoder = self.install(components["text_encoder"], "Qwen3VL-4B-Instruct-Q4_K_M.gguf")

        def transfer(repo_id, filename, local_dir):
            self.assertEqual((repo_id, filename), (components["diffusion"].repo, WEIGHT))
            return str(self.install(components["diffusion"]))

        with patch.object(settings, "ensure_dirs"), \
             patch("huggingface_hub.list_repo_files", return_value=[WEIGHT]) as listing, \
             patch("atelier.transfers.hf_hub_download", side_effect=transfer) as download:
            messages = list(downloader.download_model(self.model))
        listing.assert_called_once_with(components["diffusion"].repo)
        download.assert_called_once()
        self.assertTrue(registry.model_is_ready(self.model), messages)
        self.assertEqual(generate.resolve_model_files(self.model, want_vision=True)["enc"], encoder)

    def test_generation_and_ten_reference_editing_build_correct_native_command(self):
        files = {c.role: self.install(c) for c in self.model.components}
        refs = []
        for i in range(10):
            path = self.root / f"ref-{i}.png"
            Image.new("RGB", (16, 16), (i, 0, 0)).save(path)
            refs.append(path)
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-968-8cd731e"}')
        for reference_count in (0, 1, 10):
            with self.subTest(reference_count=reference_count), \
                 patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
                 patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
                 patch.object(sdcpp, "_require"), \
                 patch.object(sdcpp, "supported_options", return_value=frozenset()), \
                 patch.object(sdcpp, "unique_output", return_value=self.root / "out.png"), \
                 patch.object(sdcpp, "run") as run, \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(MODEL, "A glass boat", "blur", 16, 1, 1024, 1024, 42, 1,
                                  sampler="euler", schedule="auto", flow_shift=0,
                                  ref_image=refs[:reference_count] or None,
                                  prefs_override={"auto_optimize": False}, save_prompt=False)
            command = run.call_args.args[0]
            for flag, expected in (("--diffusion-model", files["diffusion"]),
                                    ("--vae", files["vae"]), ("--llm", files["text_encoder"]),
                                    ("--steps", 16), ("--cfg-scale", 1)):
                self.assertEqual(command[command.index(flag) + 1], str(expected))
            self.assertEqual([command[i + 1] for i, v in enumerate(command) if v == "-r"],
                             [str(p) for p in refs[:reference_count]])
            for flag in ("--llm_vision", "--scheduler", "--flow-shift", "-i", "-n"):
                self.assertNotIn(flag, command)

    def test_old_engine_fails_before_loading_weights_in_generation_and_adetailer(self):
        (settings.BIN_DIR / "engine-manifest.json").write_text('{"tag":"master-967-abcdef0"}')
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "adetailer_supported", return_value=True), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "968.*update.bat"):
                generate.generate(MODEL, "cat", "", 16, 1, 1024, 1024, 42, 1)
            with self.assertRaisesRegex(sdcpp.EngineError, "968.*update.bat"):
                generate.adetailer_command(MODEL, Path("in.png"), Path("out.png"), Path("detector"))
        resolve.assert_not_called()
        for tag in ("master-968-8cd731e", "master-969-abcdef0"):
            (settings.BIN_DIR / "engine-manifest.json").write_text(json.dumps({"tag": tag}))
            generate._check_flux3_engine(self.model)

    def test_eleven_references_fail_before_loading_weights(self):
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "at most 10"):
                generate.generate(MODEL, "cat", "", 16, 1, 1024, 1024, 42, 1,
                                  ref_image=[Path("reference.png")] * 11)
        resolve.assert_not_called()

    def test_benchmark_and_outpaint_keep_flux3_defaults(self):
        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark, "_monitor_peak"), \
             patch.object(benchmark.generate, "generate", return_value=[]) as run:
            benchmark._one_run(MODEL, {}, self.model.defaults, None)
        self.assertEqual(run.call_args.kwargs["steps"], 16)
        defaults = outpaint_tab._defaults(MODEL)
        self.assertEqual((defaults["steps"], defaults["cfg_scale"], defaults["edit"]), (16, 1, True))
        text = sampling.rationale("flux3")
        for expected in ("16 steps", "6.93", "Qwen3-VL 4B"):
            self.assertIn(expected, text)
        self.assertEqual(sampling.level("sampler", "euler", "flux3"), sampling.BEST)
        self.assertEqual(sampling.level("schedule", "auto", "flux3"), sampling.BEST)

    def test_reference_preparation_keeps_order_and_rejects_excess_before_writing(self):
        images = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(3)]
        extra = self.root / "extra.png"
        Image.new("RGBA", (16, 16), (3, 0, 0, 255)).save(extra)
        paths = generate_tab._save_edit_references(images, [str(extra)], 123, 10)
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (i, 0, 0))
        self.assertTrue(extra.is_file())
        with self.assertRaisesRegex(gr.Error, "at most 10"):
            generate_tab._save_edit_references(images, [str(extra)] * 8, 456, 10)
        self.assertEqual(list(settings.TMP_DIR.glob("edit_ref_456*")), [])

    def test_ui_exposes_flux3_and_sends_all_uploaded_references(self):
        import app

        demo = app.build_app()
        tab = next(b for b in demo.blocks.values() if isinstance(b, gr.Tab) and b.id == MODEL)
        self.assertEqual(tab.label, "Flux.3 Klein")
        binding = next(f for f in demo.fns.values()
                       if getattr(f.fn, "__name__", "") == "do_generate" and any(
                           k == "family" and c.cell_contents == "flux3"
                           for k, c in zip(f.fn.__code__.co_freevars, f.fn.__closure__)))
        inputs = [component.value for component in binding.inputs]
        inputs[2] = "Combine these references"
        extra = []
        for i in range(10):
            path = self.root / f"upload-{i}.png"
            Image.new("RGB", (16, 16), (i, 0, 0)).save(path)
            extra.append(str(path))
        self.assertIsInstance(binding.inputs[-1], gr.File)
        inputs[-1] = extra
        captured = []

        def generate_images(**kwargs):
            paths = kwargs["ref_image"]
            captured.extend(paths)
            self.assertEqual(kwargs["steps"], 16)
            self.assertEqual(len(paths), 10)
            for i, path in enumerate(paths):
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), (i, 0, 0))
            return []

        with patch.object(generate_tab.gen_engine, "generate", side_effect=generate_images) as run:
            list(binding.fn(*inputs))
        run.assert_called_once()
        self.assertTrue(all(not path.exists() for path in captured))
        self.assertTrue(all(Path(path).is_file() for path in extra))


if __name__ == "__main__":
    unittest.main()
