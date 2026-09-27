"""Gradio consumes every sd.cpp preview, including frames written in a burst."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class LivePreviewTests(unittest.TestCase):
    def test_qwen_generator_emits_each_step_to_gradio(self):
        import app
        from atelier import settings
        from atelier.ui import generate_tab

        demo = app.build_app()
        handler = next(f.fn for f in demo.fns.values()
                       if getattr(f.fn, "__name__", "") == "do_generate"
                       and any(k == "model_id" and c.cell_contents ==
                               "qwen-image-2.1" for k, c in zip(
                                   f.fn.__code__.co_freevars,
                                   f.fn.__closure__)))
        event = next(f for f in demo.fns.values() if f.fn is handler)
        inputs = [getattr(component, "value", None) for component in event.inputs]
        inputs[2] = "a portrait"
        inputs[14] = 3
        inputs[19] = 123

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "TMP_DIR", Path(tmp)):
            def fake_generate(**kw):
                for step in range(3):
                    name = str(kw["preview_path"]).replace("%03d",
                                                          f"{step:03d}")
                    Image.new("RGB", (2, 2), (step + 1, 0, 0)).save(name)
                    kw["log"](f"{step + 1}/3")
                out = Path(tmp) / "result.png"
                Image.new("RGB", (2, 2)).save(out)
                return [out]

            with patch.object(generate_tab.gen_engine, "generate",
                              side_effect=fake_generate):
                messages = list(handler(*inputs))
            frames = [item[1]["value"] for item in messages
                      if isinstance(item[1], dict)
                      and isinstance(item[1].get("value"), Image.Image)]
            self.assertEqual([frame.getpixel((0, 0))[0] for frame in frames],
                             [1, 2, 3])
            self.assertFalse(list(Path(tmp).glob("preview_*.png")))


if __name__ == "__main__":
    unittest.main()
