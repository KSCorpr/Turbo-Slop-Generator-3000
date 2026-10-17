"""Regression tests for the 2048-token / 300-second rejection on the user's RTX 3060."""
import importlib.util
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atelier import addons, settings
from atelier.engine import vectorize
from atelier.ui.vector_tab import mode_help
from scripts.tools import run_omnisvg as omni

TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")


class CompletionControlsTests(unittest.TestCase):
    def test_explicit_zero_timeout_is_unlimited_and_larger_token_budgets_are_accepted(self):
        values = {**vectorize.OMNI_DEFAULTS, "tokens": 8192, "seed": 0}
        omni.validate_options(SimpleNamespace(**values))
        self.assertEqual(values["timeout"], 0)
        args = omni.parser().parse_args(["--addon-root", "addon", "--mode", "omnisvg-text", "--output", "out"])
        self.assertEqual((args.tokens, args.timeout), (4096, 0))
        self.assertEqual(mode_help("omnisvg-image")[3]["value"], 4096)
        self.assertEqual(mode_help("omnisvg-image")[3]["maximum"], 16384)

    def test_time_limit_stops_after_a_token_step_and_progress_is_throttled(self):
        clock = Mock(side_effect=[0, 1, 11, 15])
        log = Mock()
        watch = omni.GenerationWatch(10, 4096, 15, log=log, clock=clock)
        self.assertFalse(watch(SimpleNamespace(shape=(1, 11))))
        self.assertFalse(watch(SimpleNamespace(shape=(1, 60))))
        log.assert_called_once()
        self.assertIn("50/4096", log.call_args.args[0])
        self.assertTrue(watch(SimpleNamespace(shape=(1, 70))))
        self.assertTrue(watch.timed_out)
        unlimited = omni.GenerationWatch(10, 4096, 0, log=Mock(), clock=Mock(side_effect=[0, 100000]))
        self.assertFalse(unlimited(SimpleNamespace(shape=(1, 11))))

    def test_eos_token_time_and_simultaneous_limits_are_not_confused(self):
        cases = [([1]*2048, 2048, False, "token_limit", "2048-token limit"),
                 ([1]*200, 2048, True, "time_limit", "time limit reached"),
                 ([1]*2048, 2048, True, "token_and_time_limit", "both"),
                 ([1]*200, 2048, False, "unknown", "stopped without EOS")]
        for ids, budget, timed_out, reason, text in cases:
            with self.subTest(reason=reason):
                stats = omni.stop_statistics(ids, budget, 300, timed_out)
                self.assertEqual(stats["stop_reason"], reason)
                self.assertIn(text, omni.incomplete_message(stats))
        self.assertEqual(omni.stop_statistics([1, 196999], 2, 300, True)["stop_reason"], "eos")

    def test_failed_omnisvg_uses_the_existing_diagnostic_download_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def child(command, log, cwd, env):
                folder = Path(command[command.index("--diagnostics-dir")+1])
                args = SimpleNamespace(output=Path(cwd) / "export", diagnostics_dir=folder, timeout=300, tokens=2048)
                stats = omni.stop_statistics([151939]*200, 2048, 300, True)
                omni.save_diagnostics(args, [stats], [[151939]*200], omni.incomplete_message(stats))
                raise RuntimeError("Child exited with code 1")

            with patch.object(addons, "ready", return_value=True), patch.object(settings, "OUTPUT_DIR", root), \
                 patch.object(vectorize.JOB, "run", side_effect=child):
                with self.assertRaises(vectorize.SVGDiagnosticError) as caught:
                    vectorize.run("omnisvg-text", None, "square")
            self.assertIn("time limit", str(caught.exception))
            with zipfile.ZipFile(caught.exception.directory / "diagnostics.zip") as archive:
                report = json.loads(archive.read("diagnostic.json"))
                self.assertEqual(report["attempts"][0]["stop_reason"], "time_limit")
                self.assertEqual(len(json.loads(archive.read("generated-token-ids-01.json"))), 200)
            self.assertFalse((caught.exception.directory / "result.svg").exists())


@unittest.skipUnless(TORCH, "Optional Torch/Transformers validation environment")
class RealGenerationTests(unittest.TestCase):
    def test_time_criterion_runs_in_actual_transformers_generation(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteriaList
        config = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
                            bos_token_id=1, eos_token_id=None, pad_token_id=0)
        model = GPT2LMHeadModel(config).eval()
        watch = omni.GenerationWatch(2, 10, seconds=1, log=Mock(), clock=Mock(side_effect=[0, 2]))
        with torch.inference_mode():
            result = model.generate(input_ids=torch.tensor([[1, 2]]), max_new_tokens=10, do_sample=False,
                                    stopping_criteria=StoppingCriteriaList([watch]))
        self.assertEqual(result.shape[1], 3)
        self.assertTrue(watch.timed_out)

    def test_actual_runner_records_budget_failure_instead_of_a_generic_token_time_message(self):
        import torch
        from transformers import AutoProcessor
        for reason in ("token_limit", "time_limit"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "export"
                output.mkdir()
                processor = Mock()
                processor.apply_chat_template.return_value = "template"
                processor.return_value = {"input_ids": torch.tensor([[10, 11]]),
                                          "attention_mask": torch.ones((1, 2), dtype=torch.long)}
                model = Mock()

                def generation(**kwargs):
                    watch = kwargs["stopping_criteria"][0]
                    count = 256 if reason == "token_limit" else 10
                    if reason == "time_limit":
                        watch.started = time.monotonic()-300
                        watch.timed_out = True
                    return torch.tensor([[10, 11, *([151939]*count)]])

                model.generate.side_effect = generation
                args = SimpleNamespace(**{**vectorize.OMNI_DEFAULTS, "tokens": 256, "seed": 0,
                    "mode": "omnisvg-text", "addon_root": root, "output": output,
                    "source": None, "prompt": "square", "timeout": 300, "diagnostics_dir": root / "diagnostic"})
                plan = {"mode": "cpu", "dtype": "float32", "ram_budget": 32*omni.GIB, "ram_required": 24*omni.GIB}
                with patch.object(omni, "load_tokenizer", return_value=Mock(COLOR_COORD_BOUNDARY=191947, MAX_COLOR_TOKENS=4098)), \
                     patch.object(omni, "load_model", return_value=model), \
                     patch.object(AutoProcessor, "from_pretrained", return_value=processor) as load:
                    with self.assertRaisesRegex(RuntimeError, "No complete SVG"):
                        omni.generate(args, plan)
                self.assertFalse(load.call_args.kwargs["use_fast"])
                report = json.loads((root / "diagnostic/diagnostic.json").read_text())
                self.assertEqual(report["attempts"][0]["stop_reason"], reason)
                self.assertFalse((output / "result.svg").exists())
                self.assertNotIn("max_time", model.generate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
