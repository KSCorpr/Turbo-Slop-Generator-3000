"""HiVG failures must distinguish truncation, native XML and missing segment decoding."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atelier import addons, settings
from atelier.engine import vectorize
from atelier.ui.vector_tab import mode_help
from scripts.tools import run_hivg as hivg
from scripts.tools.run_svg import parse_svg

SOURCE = os.environ.get("TURBOSLOP_HIVG_SOURCE")
TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><path d="M0 0H64V64H0Z" fill="red"/></svg>'
ATOMIC = '<svg>width=234 height=234 viewBox=0 0 234 234<path>fill=#f00<cmd_M><P_20><P_20><cmd_l><d_100><d_0><cmd_l><d_0><d_100><cmd_l><d_-100><d_0><cmd_z></path></svg>'


class HiVGOutputTests(unittest.TestCase):
    def test_stop_statistics_distinguish_svg_end_chat_eos_and_hard_budget(self):
        self.assertEqual(hivg.generation_stats([10, 11, 151689], 256, [151645], 151689)["stop_reason"], "svg_end")
        self.assertEqual(hivg.generation_stats([10, 151645], 256, [151645], 151689)["stop_reason"], "eos")
        result = hivg.generation_stats([10]*256, 256, [151645], 151689)
        self.assertEqual(result["stop_reason"], "token_limit")
        self.assertEqual(result["generated_tokens"], 256)

    def test_original_traceback_is_classified_as_truncation_when_the_budget_was_reached(self):
        result = {"success": False, "error": "Invalid SVG: missing </svg> closing tag", "raw_output": ATOMIC[:-15]}
        with self.assertRaisesRegex(ValueError, "4096-token budget"):
            hivg.validate_result(result, {"stop_reason": "token_limit", "token_budget": 4096, "generated_tokens": 4096})
        with self.assertRaisesRegex(ValueError, "stop: eos"):
            hivg.validate_result(result, {"stop_reason": "eos", "token_budget": 4096, "generated_tokens": 300})

    def test_compressed_segments_are_rejected_even_if_the_official_decoder_reports_success(self):
        result = {"success": True, "svg": SVG, "raw_output": "<svg><SEG_42></svg>"}
        with self.assertRaisesRegex(ValueError, "no matching segment-to-SVG dictionary"):
            hivg.validate_result(result, {})

    def test_complete_native_xml_is_decoded_without_requiring_atomic_path_tokens(self):
        for raw in (SVG, SVG.replace(' xmlns="http://www.w3.org/2000/svg"', "")):
            result = {"success": False, "error": "Invalid SVG: missing <svg> tag", "raw_output": raw}
            decoded, method = hivg.validate_result(result, {})
            self.assertEqual(method, "native_svg_xml")
            parse_svg(decoded)

    def test_does_not_close_an_incomplete_drawing_or_export_external_content(self):
        for raw in ('<svg><path d="M0 0H10"/>', '<svg><path><cmd_M></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/x.png"/></svg>'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                hivg.validate_result({"success": False, "raw_output": raw}, {})

    def test_failed_outputs_preserve_raw_text_and_stop_reason_as_downloadable_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = "<svg><path>"
            hivg.save_diagnostic(root, {"raw_output": raw, "error": "missing closing tag"},
                {"token_budget": 4096, "generated_tokens": 4096, "stop_reason": "token_limit"}, "Incomplete output")
            with zipfile.ZipFile(root / "diagnostics.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"diagnostic.json", "raw-output.txt"})
                self.assertEqual(archive.read("raw-output.txt").decode(), raw)
                report = json.loads(archive.read("diagnostic.json"))
                self.assertTrue(report["failed"])
                self.assertEqual(report["stop_reason"], "token_limit")
            self.assertFalse((root / "result.svg").exists())

    def test_ui_gives_both_svg_models_room_to_complete(self):
        self.assertEqual(mode_help("hivg-image")[3]["maximum"], 16384)
        self.assertEqual(mode_help("hivg-image")[3]["value"], 4096)
        self.assertEqual(mode_help("omnisvg-image")[3]["maximum"], 16384)
        self.assertEqual(mode_help("omnisvg-image")[3]["value"], 4096)

    def test_failure_survives_child_cleanup_without_pretending_to_be_a_completed_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def child(command, log, cwd, env):
                diag = Path(command[command.index("--diagnostics-dir")+1])
                hivg.save_diagnostic(diag, {"raw_output": "<svg>"}, {"stop_reason": "token_limit"}, "Incomplete output")
                raise RuntimeError("Child exited with code 1")

            with patch.object(addons, "ready", return_value=True), patch.object(settings, "OUTPUT_DIR", root), \
                 patch.object(vectorize.JOB, "run", side_effect=child):
                with self.assertRaises(vectorize.SVGDiagnosticError) as caught:
                    vectorize.run("hivg-text", None, "a simple square", tokens=8192)
                self.assertTrue((caught.exception.directory / "diagnostics.zip").is_file())
                self.assertFalse((caught.exception.directory / "result.svg").exists())
                self.assertEqual(len(list(root.iterdir())), 1)
                self.assertIn("Incomplete output", str(caught.exception))


@unittest.skipUnless(SOURCE and TORCH, "Set TURBOSLOP_HIVG_SOURCE to pinned HiVG source with local Torch installed")
class OfficialHiVGPipelineTests(unittest.TestCase):
    def test_real_official_decoder_reproduces_missing_close_and_accepts_complete_atomic_tokens(self):
        with patch.object(sys, "path", [SOURCE, *sys.path]):
            from hivg_infer.postprocess import postprocess_output
            complete = postprocess_output(ATOMIC)
            self.assertTrue(complete["success"])
            decoded, method = hivg.validate_result(complete, {})
            parse_svg(decoded)
            self.assertEqual(method, "official_atomic_tokens")
            broken = postprocess_output(ATOMIC.replace("</path>", ""))
            self.assertFalse(broken["success"])
            self.assertIn("missing </svg>", broken["error"])
            with self.assertRaises(ValueError):
                hivg.validate_result(broken, {})

    def test_actual_pipeline_preserves_svg_end_and_logs_the_generation_stop(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        with patch.object(sys, "path", [SOURCE, *sys.path]):
            from hivg_infer import HiSVGInferencePipeline
        processor = Mock()
        processor.tokenizer.get_vocab.return_value = {"</svg>": 151689}
        processor.decode.return_value = ATOMIC
        model = Mock()
        model.generation_config.eos_token_id = [151645, 151643]
        model.generate.return_value = torch.tensor([[42, 152443, 152444, 151689]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            args = SimpleNamespace(seed=0, tokens=4096, mode="hivg-text", source=None,
                prompt="square", addon_root=root, output=output, diagnostics_dir=root / "diagnostics")
            with patch.object(AutoProcessor, "from_pretrained", return_value=processor), \
                 patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", return_value=model), \
                 patch.object(HiSVGInferencePipeline, "_prepare_inputs", return_value={"input_ids": torch.tensor([[42]])}), \
                 patch.object(hivg, "memory_plan", return_value=("cpu", torch.float32, None)):
                hivg.generate(args)
            call = model.generate.call_args.kwargs
            self.assertFalse(call["do_sample"])
            self.assertIn(151689, call["eos_token_id"])
            self.assertEqual(call["max_new_tokens"], 4096)
            metadata = json.loads((output / "manifest.json").read_text())
            self.assertEqual(metadata["generation"]["stop_reason"], "svg_end")
            self.assertEqual(metadata["generation"]["generated_tokens"], 3)
            self.assertTrue((output / "preview.png").is_file())


if __name__ == "__main__":
    unittest.main()
