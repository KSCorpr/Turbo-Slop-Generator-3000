"""Keep honest partial drawings and recover prior numeric diagnostics without model inference."""
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atelier import addons, settings
from atelier.engine import vectorize
from atelier.ui.vector_tab import variant_result
from scripts.tools import run_omnisvg as omni

SOURCE = os.environ.get("TURBOSLOP_OMNISVG_SOURCE")
TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")


def square_tokens():
    xy = lambda x, y: 151944+y*200+x
    return [196998, 151939, xy(20,20), xy(20,20), 151940, xy(180,20), 151940, xy(180,180),
            151940, xy(20,180), 151943, xy(20,20), 195789]


class RecoveryDispatchTests(unittest.TestCase):
    def test_recovery_dispatch_uses_only_the_cpu_decode_entry_and_keeps_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "saved.zip"
            source.write_bytes(b"saved tokens")
            calls = []

            def child(command, log, cwd, env):
                calls.append(list(map(str, command)))
                out = Path(cwd) / "export"
                out.mkdir()
                for name in ("result.svg", "preview.png", "manifest.json", "vectors.zip"):
                    (out / name).write_text("export")

            with patch.object(addons, "ready", return_value=True), patch.object(settings, "OUTPUT_DIR", root), \
                 patch.object(vectorize.JOB, "run", side_effect=child):
                result = vectorize.recover(source)
            self.assertTrue((result / "vectors.zip").exists())
            self.assertIn("--recover-diagnostics", calls[0])
            self.assertNotIn("--mode", calls[0])
            self.assertNotIn("--memory", calls[0])

    def test_partial_variant_download_keeps_the_partial_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "svg-recovered-test"
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"recovered_from_diagnostics": True,
                "variants": [{"number": 1, "seed": None, "paths": 1, "complete": False,
                              "file_stem": "01.partial", "pixel_difference_percent": None}]}))
            with patch.object(settings, "OUTPUT_DIR", root):
                preview, svg, info = variant_result(1, folder)
            self.assertTrue(svg.endswith("01.partial.svg"))
            self.assertTrue(preview.endswith("01.partial.png"))
            self.assertIn("PARTIAL SVG", info)
            self.assertIn("no model inference", info)


@unittest.skipUnless(SOURCE and TORCH, "Pinned OmniSVG source and optional CPU validation dependencies")
class OfficialPartialDecodeTests(unittest.TestCase):
    def test_truncated_tail_is_dropped_and_actual_colored_paths_are_retained(self):
        import torch
        tokenizer = omni.load_tokenizer(Path(SOURCE))
        ids = square_tokens()+[151939, 151944, 151945]
        stats = omni.stop_statistics(ids, len(ids), 1)
        with self.assertRaises(ValueError):
            omni.decode_svg(tokenizer, torch.tensor(ids))
        svg, count = omni.decode_svg(tokenizer, torch.tensor(ids), stats, allow_partial=True)
        self.assertEqual(count, 1)
        self.assertFalse(stats["complete"])
        self.assertEqual(stats["discarded_tail_tokens"], 3)
        self.assertIn("PARTIAL SVG", svg)
        self.assertEqual(omni.render_candidate(svg).getpixel((512,512)), (255,0,0))
        complete = {}
        omni.decode_svg(tokenizer, torch.tensor(square_tokens()+[196999]), complete, allow_partial=True)
        self.assertTrue(complete["complete"])

    def test_recovery_of_existing_zip_renders_without_loading_model_or_processor(self):
        import torch
        from transformers import AutoProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "old-failed.zip"
            with zipfile.ZipFile(archive, "w") as saved:
                saved.writestr("diagnostic.json", json.dumps({"engine": "OmniSVG1.1_4B", "token_budget": 4096}))
                saved.writestr("generated-token-ids-01.json", json.dumps(square_tokens()+[151939,151944]))
            args = SimpleNamespace(addon_root=Path(SOURCE).parent, output=root / "out", recover_diagnostics=archive)
            tokenizer = self.tokenizer()
            with patch.object(omni, "load_tokenizer", return_value=tokenizer), \
                 patch.object(omni, "load_model", side_effect=AssertionError("Do not load inference weights")), \
                 patch.object(AutoProcessor, "from_pretrained", side_effect=AssertionError("Do not load processor")), \
                 patch.object(torch.cuda, "mem_get_info", side_effect=AssertionError("No GPU probe")):
                omni.recover_diagnostics(args)
            metadata = json.loads((args.output / "manifest.json").read_text())
            self.assertFalse(metadata["complete"])
            self.assertTrue(metadata["recovered_from_diagnostics"])
            self.assertIsNone(metadata["memory_plan"])
            self.assertTrue((args.output / "candidates/01.partial.svg").exists())
            self.assertTrue((args.output / "candidates/01.partial.png").exists())
            with zipfile.ZipFile(args.output / "vectors.zip") as saved:
                self.assertIn("candidates/01.partial.svg", saved.namelist())
                self.assertIn("generated-token-ids-01.json", saved.namelist())

    @staticmethod
    def tokenizer():
        return omni.load_tokenizer(Path(SOURCE))

    def test_real_runner_prefers_a_finished_variant_over_an_earlier_partial(self):
        import torch
        from transformers import AutoProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            processor = Mock()
            processor.apply_chat_template.return_value = "template"
            processor.return_value = {"input_ids": torch.tensor([[10,11]])}
            model = Mock()
            first = square_tokens()+[0]*(256-len(square_tokens()))
            model.generate.side_effect = [torch.tensor([[10,11,*first]]), torch.tensor([[10,11,*square_tokens(),196999]])]
            args = SimpleNamespace(**{**vectorize.OMNI_DEFAULTS, "tokens":256, "seed":0, "variants":2,
                "mode":"omnisvg-text", "addon_root":root, "output":out, "source":None, "prompt":"square"})
            plan = {"mode":"cpu", "dtype":"float32", "ram_budget":32*omni.GIB, "ram_required":24*omni.GIB}
            tokenizer = self.tokenizer()
            with patch.object(omni, "load_tokenizer", return_value=tokenizer), \
                 patch.object(omni, "load_model", return_value=model), patch.object(AutoProcessor,"from_pretrained", return_value=processor):
                omni.generate(args, plan)
            metadata = json.loads((out / "manifest.json").read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["default_variant"], 2)
            self.assertFalse(metadata["variants"][0]["complete"])
            self.assertIn("PARTIAL SVG", (out / "candidates/01.partial.svg").read_text())
            self.assertNotIn("PARTIAL SVG", (out / "result.svg").read_text())

    def test_unusable_or_malformed_tokens_do_not_become_a_fake_partial_svg(self):
        import torch
        tokenizer = self.tokenizer()
        with self.assertRaisesRegex(ValueError, "No finished color-delimited path"):
            omni.decode_svg(tokenizer, torch.tensor([151939,151944]), {}, allow_partial=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ids in ([True], [197000], "not a list"):
                path = root / "bad.zip"
                with zipfile.ZipFile(path,"w") as archive:
                    archive.writestr("diagnostic.json",json.dumps({"engine":"OmniSVG1.1_4B","token_budget":4096}))
                    archive.writestr("generated-token-ids-01.json",json.dumps(ids))
                args=SimpleNamespace(addon_root=root,output=root/"out",recover_diagnostics=path)
                with self.assertRaisesRegex(ValueError,"Invalid OmniSVG token sequence"):
                    omni.recover_diagnostics(args)


if __name__ == "__main__":
    unittest.main()
