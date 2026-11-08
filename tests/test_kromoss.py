"""Kromoss components, adapter isolation and UI-to-native generation contracts."""
import gzip
import json
from pathlib import Path
import struct
from unittest import TestCase
from unittest.mock import patch

import gradio as gr

from atelier import downloader, kromoss, model_catalog, registry, settings
from atelier.engine import generate, sdcpp
from atelier.ui import generate_tab
from tests import test_moss


class KromossTests(TestCase):
    install = test_moss.MossTests.install
    install_all = test_moss.MossTests.install_all

    def setUp(self):
        test_moss.MossTests.setUp(self)
        self.stack.enter_context(patch.object(settings, "LORA_DIR", self.root / "loras"))
        self.stack.enter_context(patch.object(settings, "OUTPUT_DIR", self.root / "outputs"))
        self.stack.enter_context(patch.object(settings, "CUSTOM_DIR", self.root / "custom"))
        self.stack.enter_context(patch.object(settings, "PREFS_FILE", self.root / "prefs.json"))
        self.model = registry.get_base_model("moss-image-kromoss", {})
        self.components = {c.role: c for c in self.model.components}
        self.options = {"--tokenizer", "--lora-apply-mode"}

    def adapter(self):
        # DMD2 has the same 756 F32 rank-128 tensor layout as the SFT adapter.
        header = json.loads(gzip.decompress(
            (Path(__file__).parent / "fixtures/moss_dmd_header.json.gz").read_bytes()))
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        size = len(raw) + 8 + max(value["data_offsets"][1] for value in header.values())
        path = kromoss.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(raw)) + raw)
            stream.truncate(size)
        self.stack.enter_context(patch.object(kromoss, "FILE_SIZE", size))
        return path

    def test_catalog_has_shared_moss_encoder_and_tokenizer_but_its_own_weights_and_small_vae(self):
        sft = {c.role: c for c in registry.get_base_model("moss-image-sft", {}).components}
        for role in ("text_encoder", "tokenizer"):
            self.assertEqual((self.components[role].repo, self.components[role].requested()),
                             (sft[role].repo, sft[role].requested()))
        self.assertEqual(self.components["diffusion"].repo, "lodestones/Kromoss")
        self.assertEqual(self.components["diffusion"].requested(),
                         "diffusion_models/kromoss_int8_convrot.safetensors")
        self.assertIn("diffusion_models/kromoss_bf16.safetensors", self.components["diffusion"].selectable)
        self.assertEqual(self.components["vae"].requested(), "vae/flux2-vae_small.safetensors")
        self.assertEqual(self.components["vae"].alternatives[1].repo, sft["vae"].repo)
        self.assertEqual(self.model.family, "moss_image")
        self.assertIsNotNone(model_catalog.get("image:moss-image-kromoss"))

    def test_download_reuses_both_moss_text_components_and_fetches_only_diffusion_and_vae(self):
        for role in ("text_encoder", "tokenizer"):
            self.install(self.components[role])

        def listing(repo):
            return [c.requested() for c in self.components.values() if c.repo == repo]

        def transfer(repo_id, filename, local_dir):
            component = next(c for c in self.components.values()
                             if c.repo == repo_id and c.requested() == filename)
            return str(self.install(component))

        with patch("huggingface_hub.list_repo_files", side_effect=listing), \
             patch("atelier.transfers.hf_hub_download", side_effect=transfer) as download:
            list(downloader.download_model(self.model))
        self.assertEqual({c.kwargs["filename"] for c in download.call_args_list},
                         {self.components["diffusion"].requested(), self.components["vae"].requested()})
        self.assertTrue(registry.model_is_ready(self.model))

    def test_dmd2_package_downloads_one_pinned_adapter_and_checks_its_layout(self):
        path = self.adapter()
        with patch.object(kromoss.transfers, "hf_hub_download", return_value=str(path)) as download:
            model_catalog.install("kromoss:dmd2", log=lambda _: None)
        download.assert_called_once_with(repo_id=kromoss.REPO, filename=kromoss.FILENAME,
                                         revision=kromoss.REVISION, local_dir=str(path.parent))
        self.assertTrue(model_catalog.ready("kromoss:dmd2"))
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Download.*Models"):
            kromoss.require()

    def test_missing_dmd2_stops_before_model_loading(self):
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(sdcpp, "supported_options", return_value=self.options), \
             patch.object(generate, "resolve_model_files") as resolve:
            with self.assertRaisesRegex(sdcpp.EngineError, "Kromoss.*Models"):
                generate.generate(self.model.id, "cat", "", 8, 1, 1024, 1024, 42, 1, moss_dmd=True)
        resolve.assert_not_called()

    def test_ui_selects_dmd2_native_command_and_metadata_then_returns_to_full_sampling(self):
        files = self.install_all()
        self.adapter()
        with gr.Blocks() as demo:
            generate_tab.build_generative_tab(self.model.id, "Kromoss")
        binding = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "do_generate")
        preset = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "apply_preset")
        prompt = '{"summary":{"caption":"A poster"},"items":[]}'
        commands = []
        output = self.root / "out.png"
        with patch.object(settings, "find_sd_cli", return_value=self.root / "sd-cli"), \
             patch.object(generate, "_resolved_flags", return_value=({}, 0)), \
             patch.object(sdcpp, "supported_options", return_value=self.options), \
             patch.object(sdcpp, "run", side_effect=lambda cmd, **_: commands.append(cmd)), \
             patch.object(sdcpp, "collect_outputs", return_value=[output]):
            for label, fast in (("DMD2 fast (8 steps)", True), ("Recommended (50 steps)", False)):
                values = {c._id: c.value for c in binding.inputs}
                values[binding.inputs[2]._id] = prompt
                values[preset.inputs[0]._id] = label
                for component, update in zip(preset.outputs, preset.fn(label)):
                    if isinstance(update, dict) and "value" in update:
                        values[component._id] = update["value"]
                list(binding.fn(*(values[c._id] for c in binding.inputs)))
                command = commands[-1]
                self.assertEqual(command[command.index("--diffusion-model") + 1], str(files["diffusion"]))
                self.assertEqual(command[command.index("--vae") + 1], str(files["vae"]))
                self.assertEqual(int(command[command.index("--steps") + 1]), 8 if fast else 50)
                self.assertEqual(float(command[command.index("--cfg-scale") + 1]), 1 if fast else 3)
                if fast:
                    self.assertEqual(command[command.index("--lora-apply-mode") + 1], "at_runtime")
                    self.assertEqual(command[command.index("-p") + 1],
                                     prompt + " <lora:" + kromoss.LORA_NAME + ":1>")
                    metadata = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(metadata["prompt"], prompt)
                    self.assertEqual(metadata["dmd_lora_strength"], 1)
                    self.assertEqual(metadata["dmd_lora_file"], kromoss.FILENAME)
                    self.assertEqual(metadata["dmd_lora_revision"], kromoss.REVISION)
                else:
                    self.assertNotIn("--lora-apply-mode", command)
                    self.assertEqual(command[command.index("-p") + 1], prompt)
                    self.assertNotIn("moss_dmd", json.loads(output.with_suffix(".json").read_text()))
        self.assertEqual(len(commands), 2)
