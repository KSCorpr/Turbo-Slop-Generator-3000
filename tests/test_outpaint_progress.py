"""Outpaint remains responsive during segmented inference and preserves images."""
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import settings
from atelier.engine import outpaint
from atelier.ui import outpaint_tab


class StreamTests(unittest.TestCase):
    def test_tensor_log_storm_does_not_emit_thousands_of_ui_updates(self):
        messages = queue.Queue()
        messages.put("[V] qwen_image_2_1 using 34 segments --- ggml_runner.cpp")
        for _ in range(5000):
            messages.put("[V] loading 14/457 tensors from weights.gguf")
            messages.put("[I] loading tensors completed, taking 0.20s")
        messages.put("|========>| 2/8 - 6.99s/it\x1b[K")
        messages.put("[W] useful warning")
        messages.put(None)
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            result = list(outpaint_tab._stream_outpaint(
                messages, Path(tmp) / "preview_%03d.png", 8, logs))
        self.assertLess(len(result), 10)
        self.assertLess(len(logs), 10)
        self.assertTrue(any("34 segments" in line for line in logs))
        self.assertIn("[W] useful warning", logs)
        self.assertTrue(any("Step 2/8" in status for status, _, _ in result))

    def test_all_step_previews_written_in_a_burst_are_displayed(self):
        messages = queue.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "preview_%03d.png"
            for step in range(3):
                Image.new("RGB", (4, 4), (step + 1, 0, 0)).save(
                    str(template).replace("%03d", f"{step:03d}"))
                messages.put(f"|==>| {step+1}/8 - 1.0s/it")
            messages.put(None)
            result = list(outpaint_tab._stream_outpaint(messages, template, 8, []))
        frames = [update["value"] for _, update, _ in result
                  if isinstance(update.get("value"), Image.Image)]
        self.assertEqual([image.getpixel((0, 0))[0] for image in frames], [1, 2, 3])

    def test_loading_bar_is_not_confused_with_a_sampling_step(self):
        messages = queue.Queue()
        messages.put("|########| 8/8 - 900MB/s\x1b[K")
        messages.put(None)
        with tempfile.TemporaryDirectory() as tmp:
            result = list(outpaint_tab._stream_outpaint(
                messages, Path(tmp) / "preview_%03d.png", 8, []))
        self.assertFalse(any("Step 8/8" in status for status, _, _ in result))

    def test_silent_engine_still_gets_a_heartbeat(self):
        messages = queue.Queue()
        timer = threading.Timer(1.1, lambda: messages.put(None))
        timer.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = list(outpaint_tab._stream_outpaint(
                    messages, Path(tmp) / "preview_%03d.png", 8, []))
            self.assertTrue(any("1 s elapsed" in status for status, _, _ in result))
        finally:
            timer.cancel()
            timer.join()


class WorkingResolutionTests(unittest.TestCase):
    def test_full_resolution_remains_the_default(self):
        self.assertEqual(outpaint.working_size((1536, 1216)), (1536, 1216))
        self.assertEqual(outpaint.working_size((640, 480), 1024, 32), (640, 480))

    def test_fast_resolution_is_bounded_and_aligned_for_qwen(self):
        w, h = outpaint.working_size((1536, 1216), 1024, 32)
        self.assertEqual(w, 1024)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertLess(w * h, 1536 * 1216)


class OutpaintWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gr.Blocks() as demo:
            outpaint_tab.build_outpaint_tab()
        cls.handler = staticmethod(next(f.fn for f in demo.fns.values()
                                      if getattr(f.fn, "__name__", "") == "do_outpaint"))

    def test_fast_generation_gets_preview_and_returns_original_canvas_size(self):
        source = Image.new("RGB", (1000, 800), (73, 109, 151))
        plan = outpaint.plan(source.size, outpaint.DIRECTIONS, .25, multiple=32)
        captured = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "TMP_DIR", Path(tmp) / "tmp"), \
             patch.object(settings, "OUTPUT_DIR", Path(tmp) / "outputs"), \
             patch.object(settings, "ensure_dirs", side_effect=lambda: (
                 settings.TMP_DIR.mkdir(exist_ok=True), settings.OUTPUT_DIR.mkdir(exist_ok=True))):
            def generate(**kw):
                with Image.open(kw["ref_image"][0]) as reference:
                    self.assertEqual(reference.size, (kw["width"], kw["height"]))
                captured.append(kw)
                for step in range(3):
                    name = str(kw["preview_path"]).replace("%03d", f"{step:03d}")
                    Image.new("RGB", (2, 2), (step + 1, 0, 0)).save(name)
                    kw["log"](f"|==>| {step+1}/8 - 1.0s/it")
                result = settings.OUTPUT_DIR / "engine.png"
                Image.new("RGB", (kw["width"], kw["height"]), (10, 20, 30)).save(result)
                return [result]
            with patch.object(outpaint_tab.gen_engine, "generate", side_effect=generate):
                messages = list(self.handler(source, "left|right|top|bottom", .25,
                    "qwen-image-2.1-turbo", "", "neutral", .85, 0, 0, 8, 123, 1024))
            kw = captured[0]
            self.assertLessEqual(max(kw["width"], kw["height"]), 1024)
            self.assertEqual(kw["steps"], 8)
            frames = [item[1]["value"] for item in messages
                      if isinstance(item[1], dict) and isinstance(item[1].get("value"), Image.Image)]
            self.assertEqual([image.getpixel((0, 0))[0] for image in frames], [1, 2, 3])
            result = Path(messages[-1][1]["value"])
            with Image.open(result) as image:
                self.assertEqual(image.size, (plan["width"], plan["height"]))
                centre = image.crop((plan["left"], plan["top"],
                                    plan["left"] + source.width, plan["top"] + source.height))
                self.assertEqual(centre.tobytes(), source.tobytes())
            self.assertEqual(list(settings.TMP_DIR.iterdir()), [])
            self.assertIn("Processing size:", result.with_suffix(".txt").read_text())

    def test_failure_cleans_temporary_files_and_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "TMP_DIR", Path(tmp)), \
             patch.object(settings, "ensure_dirs"), \
             patch.object(outpaint_tab.gen_engine, "generate", side_effect=RuntimeError("CUDA test failure")):
            result = list(self.handler(Image.new("RGB", (64, 64)), "right", .25,
                "qwen-image-2.1-turbo", "", "neutral", .85, 0, 0, 8, 123))
            self.assertIn("CUDA test failure", result[-1][2])
            self.assertIn("Failed", result[-1][0])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_closing_ui_keeps_reference_until_engine_finishes(self):
        entered = threading.Event()
        release = threading.Event()
        paths = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "TMP_DIR", Path(tmp)), \
             patch.object(settings, "ensure_dirs"):
            def generate(**kw):
                reference = kw["ref_image"][0]
                paths.append(reference)
                Image.new("RGB", (2, 2)).save(str(kw["preview_path"]).replace("%03d", "000"))
                entered.set()
                release.wait(2)
                return []
            with patch.object(outpaint_tab.gen_engine, "generate", side_effect=generate):
                stream = self.handler(Image.new("RGB", (64, 64)), "right", .25,
                    "qwen-image-2.1-turbo", "", "neutral", .85, 0, 0, 8, 123)
                try:
                    next(stream)
                    self.assertTrue(entered.wait(1))
                    stream.close()
                    self.assertTrue(paths[0].is_file())
                finally:
                    release.set()
                    stream.close()
                deadline = time.monotonic() + 1
                while list(Path(tmp).iterdir()) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
