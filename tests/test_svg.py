"""SVG contracts, actual optional CPU engines, and bounded HiVG dispatch."""
import importlib.util
from contextlib import nullcontext
import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from atelier import addons, inventory, settings
from atelier.engine import vectorize
from scripts.tools import run_svg as runner

with patch.dict(sys.modules, {"run_svg": runner}):
    from scripts.tools import run_hivg as hivg

ROOT = Path(__file__).resolve().parents[1]
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><path d="M0 0H64V64H0Z" fill="red"/></svg>'
CPU_AVAILABLE = all(importlib.util.find_spec(name) for name in ("vtracer", "cv2", "resvg_py"))
LAYER_SOURCE = os.environ.get("TURBOSLOP_LAYERTRACER_SOURCE")


class SVGValidationTests(unittest.TestCase):
    def test_real_shapes_required_and_external_or_active_content_rejected(self):
        runner.parse_svg(SVG)
        for content in ('<image href="data:image/png;base64,AA"/>', '<script/>',
                        '<foreignObject/>', '<rect onclick="x()"/>',
                        '<rect fill="url(https://example.com/a)"/>',
                        '<rect style="fill:red"/>', '<g/>'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                runner.parse_svg(f'<svg xmlns="{runner.NS}">{content}</svg>')
        with self.assertRaises(ValueError):
            runner.parse_svg('<!DOCTYPE svg [<!ENTITY x "a">]>' + SVG)

    def test_hiVG_memory_plan_preserves_headroom_and_selected_gpu_is_local_zero(self):
        gib = 1024**3
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
            mem_get_info=lambda i: (12*gib, 12*gib), is_bf16_supported=lambda: True),
            float32="fp32", bfloat16="bf16", float16="fp16")
        self.assertEqual(hivg.memory_plan(torch, 24*gib),
                         ("auto", "bf16", {0: 10*gib, "cpu": 22*gib}))
        torch.cuda.is_available = lambda: False
        self.assertEqual(hivg.memory_plan(torch, 24*gib), ("cpu", "fp32", None))
        with self.assertRaises(RuntimeError):
            hivg.memory_plan(torch, 8*gib)

    def test_hiVG_uses_official_modes_local_qwen_sdpa_and_rejects_unfinished_tokens(self):
        from unittest.mock import Mock
        calls = []
        result = {"success": True, "svg": SVG, "raw_output": "<svg><path></path></svg>"}
        processor = Mock()
        model = Mock()
        torch = SimpleNamespace(manual_seed=Mock(), inference_mode=nullcontext)

        class OfficialPipelineContract:
            def __init__(self, path, **kwargs):
                calls.append(("constructor", kwargs))
                self._load_model(path, None, None, None, None)

            def img2svg(self, path):
                calls.append(("image", path))
                return result

            def text2svg(self, prompt):
                calls.append(("text", prompt))
                return result

        modules = {"torch": torch, "psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=32*1024**3)),
                   "transformers": SimpleNamespace(AutoProcessor=processor, Qwen2_5_VLForConditionalGeneration=model),
                   "hivg_infer": SimpleNamespace(HiSVGInferencePipeline=OfficialPipelineContract)}
        args = SimpleNamespace(seed=7, output=Path("export"), addon_root=Path("addon"),
                               tokens=1024, source=Path("source.png"), mode="hivg-image", prompt="red square")
        with patch.dict(sys.modules, modules), patch.object(hivg, "memory_plan", return_value=("auto", "bf16", {0: 10})), \
             patch.object(hivg, "export") as export:
            hivg.generate(args)
            args.mode = "hivg-text"
            hivg.generate(args)
            self.assertIn(("image", "source.png"), calls)
            self.assertIn(("text", "red square"), calls)
            self.assertEqual(export.call_count, 2)
            load = model.from_pretrained.call_args.kwargs
            self.assertTrue(load["local_files_only"])
            self.assertFalse(load["trust_remote_code"])
            self.assertEqual(load["attn_implementation"], "sdpa")
            self.assertEqual(calls[0][1]["max_new_tokens"], 1024)
            result["raw_output"] = "<svg><path>"
            with self.assertRaisesRegex(RuntimeError, "before completing"):
                hivg.generate(args)
            self.assertEqual(export.call_count, 2)

    def test_inventory_separates_weights_and_runtimes(self):
        items = {item.key: item for item in inventory.items()}
        self.assertIn("engine_svg", items)
        self.assertIn("engine_hivg", items)
        self.assertEqual(items["hivg_models"].paths, [addons.root("hivg") / "model"])
        self.assertTrue(set(("svg", "hivg")).issubset(addons.ADDON_NAMES))

    def test_invalid_job_never_starts_a_process(self):
        with patch.object(addons, "ready", return_value=True), patch.object(vectorize.JOB, "run") as run:
            for values in (("bad", None, "", 2048, 0, 5),
                           ("hivg-text", None, "", 2048, 0, 5),
                           ("hivg-text", None, "circle", 99999, 0, 5),
                           ("hivg-text", None, "circle", 2048, -1, 5),
                           ("hivg-text", None, "circle", 2048, 0, float("nan"))):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    vectorize.run(*values)
            run.assert_not_called()

    def test_dispatch_exports_only_complete_results_and_flattens_alpha_on_white(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(image)
            calls = []

            def child(command, log, cwd, env):
                calls.append((command, env))
                with Image.open(Path(cwd) / "source.png") as source:
                    self.assertEqual(source.getpixel((0, 0)), (255, 255, 255))
                export = Path(cwd) / "export"
                export.mkdir()
                for name in ("result.svg", "preview.png", "vectors.zip", "manifest.json"):
                    (export / name).write_bytes(b"complete")

            with patch.object(addons, "ready", return_value=True), \
                 patch.object(vectorize.JOB, "run", side_effect=child), \
                 patch.object(vectorize, "_gen_gpu_index", return_value=1), \
                 patch.object(settings, "OUTPUT_DIR", root / "outputs"):
                result = vectorize.run("hivg-image", image, "")
                self.assertTrue((result / "result.svg").is_file())
                self.assertEqual(calls[0][1]["CUDA_VISIBLE_DEVICES"], "1")
                self.assertIn("run_hivg.py", str(calls[0][0][2]))
            with patch.object(addons, "ready", return_value=True), \
                 patch.object(vectorize.JOB, "run"), \
                 patch.object(settings, "OUTPUT_DIR", root / "failed"):
                with self.assertRaisesRegex(RuntimeError, "complete SVG export"):
                    vectorize.run("hivg-text", None, "circle")
                self.assertFalse((root / "failed").exists())


@unittest.skipUnless(CPU_AVAILABLE, "Optional SVG engine dependencies not installed")
class CPUExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def invoke(self, mode, image, addon):
        image.save(self.root / "source.png")
        subprocess.run([sys.executable, str(ROOT / "scripts/tools/run_svg.py"),
            "--mode", mode, "--source", "source.png", "--output", "export",
            "--addon-root", str(addon)], cwd=self.root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return self.root / "export"

    def test_actual_vtracer_svg_and_png_agree_with_flat_source(self):
        output = self.invoke("trace", Image.new("RGB", (64, 64), "red"), self.root)
        svg = runner.parse_svg((output / "result.svg").read_text())
        self.assertTrue(any(node.tag.endswith("path") for node in svg.iter()))
        with Image.open(output / "preview.png") as preview:
            self.assertEqual(preview.size, (1024, 1024))
            self.assertEqual(preview.convert("RGB").getpixel((512, 512)), (255, 0, 0))
        with zipfile.ZipFile(output / "vectors.zip") as archive:
            self.assertEqual(set(archive.namelist()), {"result.svg", "preview.png", "manifest.json"})

    @unittest.skipUnless(LAYER_SOURCE, "Set TURBOSLOP_LAYERTRACER_SOURCE to the pinned source checkout")
    def test_actual_layertracer_four_and_nine_panel_order_and_last_step(self):
        size = 128
        for count in (4, 9):
            with self.subTest(count=count):
                side = 2 if count == 4 else 3
                grid = Image.new("RGB", (size*side, size*side), "white")
                frame = Image.new("RGB", (size, size), "white")
                draw = ImageDraw.Draw(frame)
                positions = [(0, 0), (1, 0), (1, 1), (0, 1)] if count == 4 else [
                    (i % 3, i // 3) for i in range(9)]
                for index, position in enumerate(positions):
                    # Distinct large additions survive upstream speckle/morphology thresholds.
                    x, y = 5 + (index % 3)*40, 5 + (index // 3)*40
                    draw.rectangle((x, y, x+28, y+28), fill="black" if index == 0 else "blue")
                    grid.paste(frame, (position[0]*size, position[1]*size))
                output = self.invoke(f"layer{count}", grid, Path(LAYER_SOURCE).parent)
                root = runner.parse_svg((output / "result.svg").read_text())
                groups = [node for node in root if node.tag.endswith("g")]
                self.assertEqual(len(groups), count)
                last = next(group for group in groups if group.get("id", "").endswith(f"step-{count}"))
                self.assertTrue(any(node.tag.endswith("path") for node in last.iter()))
                with zipfile.ZipFile(output / "vectors.zip") as archive:
                    layers = [name for name in archive.namelist() if name.startswith("layers/")]
                    self.assertEqual(len(layers), count)
                metadata = json.loads((output / "manifest.json").read_text())
                self.assertEqual(metadata["stage"], "prepared-sequence-to-svg")
                # Each run uses fresh intermediates just like the application.
                import shutil
                shutil.rmtree(self.root / "traced")
                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
