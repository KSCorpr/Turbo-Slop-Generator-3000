"""Admission controls, real local Qwen loading and official OmniSVG SVG decoding."""
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from atelier import addons, inventory, settings
from atelier.engine import vectorize
from atelier.ui.vector_tab import variant_result
from scripts.tools import run_omnisvg as omni
from scripts import setup_svg

SOURCE = os.environ.get("TURBOSLOP_OMNISVG_SOURCE")
TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers") and importlib.util.find_spec("accelerate")


def options(**changes):
    return SimpleNamespace(**{**vectorize.OMNI_DEFAULTS, "tokens": 2048, "seed": 0, **changes})


def fake_cuda(free=12*omni.GIB):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        mem_get_info=lambda index: (free, 12*omni.GIB), get_device_name=lambda index: "RTX 3060",
        is_bf16_supported=lambda: True))


class ControlTests(unittest.TestCase):
    def test_gpu_and_ram_limits_are_clamped_to_actual_free_memory(self):
        plan = omni.memory_plan(fake_cuda(), 64*omni.GIB, options(vram=8, ram=32))
        self.assertEqual(plan["gpu_budget"], 8*omni.GIB)
        self.assertEqual(plan["gpu_weights"], 6*omni.GIB)
        self.assertEqual(plan["ram_budget"], 32*omni.GIB)
        self.assertEqual(plan["dtype"], "bfloat16")
        plan = omni.memory_plan(fake_cuda(), 32*omni.GIB, options(vram=128, ram=128))
        self.assertEqual(plan["gpu_budget"], 10.5*omni.GIB)
        self.assertEqual(plan["ram_budget"], 28*omni.GIB)

    def test_rejects_insufficient_memory_and_does_not_silently_fall_back_to_cpu(self):
        for gpu, ram, opts in ((fake_cuda(), 12*omni.GIB, options()),
                               (fake_cuda(), 64*omni.GIB, options(ram=8)),
                               (fake_cuda(5*omni.GIB), 64*omni.GIB, options()),
                               (fake_cuda(), 64*omni.GIB, options(vram=3))):
            with self.subTest(opts=opts), self.assertRaises(RuntimeError):
                omni.memory_plan(gpu, ram, opts)
        torch = fake_cuda()
        torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            omni.memory_plan(torch, 64*omni.GIB, options())
        plan = omni.memory_plan(torch, 64*omni.GIB, options(memory="cpu"))
        self.assertEqual(plan["dtype"], "float32")
        self.assertEqual(plan["gpu_budget"], 0)

    def test_rejects_invalid_or_nonfinite_controls(self):
        for values in ({"tokens": 100000}, {"ram": float("nan")}, {"temperature": 0},
                       {"variants": 8}, {"variants": 1.5}, {"vram": -1}, {"top_p": 1.1},
                       {"timeout": 0}, {"seed": -1}, {"memory": "magic"}, {"ram": None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                omni.validate_options(options(**values))
        with self.assertRaises(ValueError):
            vectorize.omni_arguments({"unknown": True})

    def test_dispatch_passes_controls_and_checks_memory_in_the_same_selected_gpu_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def child(command, log, cwd=None, env=None):
                calls.append((list(map(str, command)), env))
                if "--probe" in command:
                    return
                folder = Path(cwd) / "export"
                folder.mkdir()
                for name in ("result.svg", "preview.png", "vectors.zip", "manifest.json"):
                    (folder / name).write_bytes(b"complete")

            values = {"memory": "offload", "vram": 7.5, "ram": 32, "variants": 3.0,
                      "cache": False, "keep_aspect": True, "temperature": .5, "top_k": 80}
            with patch.object(addons, "ready", return_value=True), patch.object(vectorize, "_gen_gpu_index", return_value=1), \
                 patch.object(vectorize.JOB, "run", side_effect=child), patch.object(settings, "OUTPUT_DIR", root):
                vectorize.diagnose(values)
                result = vectorize.run("omnisvg-text", None, "red square", 1024, 8, omni=values)
            self.assertTrue((result / "result.svg").exists())
            for command, env in calls:
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
                self.assertIn("run_omnisvg.py", command[2])
                self.assertIn("--no-cache", command)
                self.assertEqual(command[command.index("--memory")+1], "offload")
                self.assertEqual(command[command.index("--variants")+1], "3")

    def test_variant_selection_is_limited_to_generated_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "svg-test"
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"variants": [
                {"number": 1, "seed": 7, "paths": 2, "pixel_difference_percent": 4.25}]}))
            with patch.object(settings, "OUTPUT_DIR", root):
                preview, svg, info = variant_result(1, folder)
                self.assertTrue(svg.endswith("candidates/01.svg"))
                self.assertIn("4.25%", info)
                with self.assertRaises(ValueError):
                    variant_result(2, folder)
                with self.assertRaises(ValueError):
                    variant_result(1, root.parent / "svg-test")

    def test_inventory_and_maintenance_keep_omnisvg_weights_and_runtime(self):
        items = {item.key: item for item in inventory.items()}
        self.assertIn("engine_omnisvg", items)
        self.assertEqual(items["omnisvg_models"].paths,
                         [addons.root("omnisvg") / "model", addons.root("omnisvg") / "processor"])
        self.assertIn("omnisvg", addons.ADDON_NAMES)

    def test_installer_uses_selected_gpu_build_and_never_downloads_qwen_base_weights(self):
        original_stat = Path.stat

        def stat(path, *args, **kwargs):
            return SimpleNamespace(st_size=omni.WEIGHT_BYTES) if path.name == "pytorch_model.bin" else original_stat(path, *args, **kwargs)

        for selected, cuda in (("0", "cu126"), ("1", "cu128")):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                gpus = [SimpleNamespace(index=0, arch="ampere"), SimpleNamespace(index=1, arch="blackwell")]
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": selected}), \
                     patch.object(setup_svg.hardware, "detect_gpus", return_value=gpus), \
                     patch.object(setup_svg, "pip") as pip, patch.object(setup_svg, "run") as run, \
                     patch.object(setup_svg, "_install_source"), patch.object(setup_svg, "prepare_omnisvg_source"), \
                     patch.object(setup_svg.transfers, "snapshot_download") as download, patch.object(Path, "stat", stat):
                    metadata = setup_svg.install_omnisvg(target, target / "python", [])
                self.assertEqual(metadata["cuda"], cuda)
                self.assertIn(cuda, " ".join(map(str, pip.call_args_list[0].args)))
                self.assertEqual(download.call_count, 2)
                processor = download.call_args_list[1].kwargs
                self.assertEqual(processor["repo_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
                self.assertNotIn("*.safetensors", processor["allow_patterns"])
                self.assertNotIn("pytorch_model.bin", processor["allow_patterns"])
                self.assertIn("--check-imports", run.call_args.args)


@unittest.skipUnless(TORCH, "Optional local Torch/Transformers validation dependencies")
class RealTorchTests(unittest.TestCase):
    def test_real_meta_checkpoint_loading_and_two_token_inference_without_base_weights(self):
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Qwen2_5_VLConfig(vocab_size=197000, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
                vision_end_token_id=151653, vision_token_id=151654,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
                vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                               "num_heads": 2, "out_hidden_size": 16})
            original = Qwen2_5_VLForConditionalGeneration(config).eval().bfloat16()
            config.save_pretrained(root)
            torch.save({"transformer."+key: value for key, value in original.state_dict().items()},
                       root / "pytorch_model.bin")
            original.float()
            expected = original(input_ids=torch.tensor([[10, 11]])).logits.detach()
            model = omni.load_model(root, {"mode": "cpu", "dtype": "float32"})
            with torch.inference_mode():
                actual = model(input_ids=torch.tensor([[10, 11]])).logits
                result = model.generate(input_ids=torch.tensor([[10, 11]]), max_new_tokens=2,
                    do_sample=False, eos_token_id=196999, pad_token_id=151643)
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
            self.assertEqual(tuple(result.shape), (1, 4))
            state = torch.load(root / "pytorch_model.bin", weights_only=True, mmap=True)
            state["transformer.invalid.weight"] = torch.zeros(1)
            torch.save(state, root / "bad.bin")
            (root / "bad.bin").replace(root / "pytorch_model.bin")
            with self.assertRaises(RuntimeError):
                omni.load_model(root, {"mode": "cpu", "dtype": "float32"})

    def test_unfinished_generation_is_not_repaired_into_a_successful_svg(self):
        import torch
        with self.assertRaisesRegex(ValueError, "before the SVG was complete"):
            omni.decode_svg(Mock(), torch.tensor([151939, 151944]))

    @unittest.skipUnless(SOURCE, "Set TURBOSLOP_OMNISVG_SOURCE to the prepared pinned source")
    def test_official_decoder_colors_and_full_serial_variant_export(self):
        import torch
        from transformers import AutoProcessor
        token = lambda x, y: 151944+y*200+x
        ids = [196998, 151939, token(20, 20), token(20, 20), 151940, token(180, 20),
               151940, token(180, 180), 151940, token(20, 180), 151943, token(20, 20), 195789, 196999]
        tokenizer = omni.load_tokenizer(Path(SOURCE))
        svg, count = omni.decode_svg(tokenizer, torch.tensor(ids))
        self.assertEqual(count, 1)
        self.assertEqual(omni.render_candidate(svg).getpixel((512, 512)), (255, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "export"
            output.mkdir()
            source = root / "source.png"
            Image.new("RGB", (128, 64), "red").save(source)
            processor = Mock()
            processor.apply_chat_template.return_value = "chat template"
            processor.return_value = {"input_ids": torch.tensor([[10, 11]]),
                                      "attention_mask": torch.ones((1, 2), dtype=torch.long)}
            model = Mock()
            model.generate.return_value = torch.tensor([[10, 11, *ids]])
            args = options(variants=2, mode="omnisvg-image", addon_root=root, output=output,
                           source=source, prompt="")
            plan = omni.memory_plan(fake_cuda(), 64*omni.GIB, options(memory="cpu"))
            with patch.object(omni, "load_tokenizer", return_value=tokenizer), \
                 patch.object(omni, "load_model", return_value=model), \
                 patch.object(AutoProcessor, "from_pretrained", return_value=processor) as load:
                omni.generate(args, plan)
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertFalse(load.call_args.kwargs["trust_remote_code"])
            self.assertEqual(model.generate.call_count, 2)
            for call in model.generate.call_args_list:
                self.assertEqual(call.kwargs["num_return_sequences"], 1)
                self.assertEqual(call.kwargs["max_new_tokens"], 2048)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([item["seed"] for item in manifest["variants"]], [0, 1])
            self.assertIsNotNone(manifest["variants"][0]["pixel_difference_percent"])
            with Image.open(output / "reference.png") as reference:
                self.assertEqual(reference.size, (448, 448))
                self.assertEqual(reference.getpixel((0, 0)), (255, 255, 255))
            with zipfile.ZipFile(output / "vectors.zip") as archive:
                self.assertTrue({"candidates/01.svg", "candidates/02.svg", "reference.png"} <= set(archive.namelist()))


if __name__ == "__main__":
    unittest.main()
