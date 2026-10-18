"""Release gates for history, startup work and the shared inference queue."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from PIL import Image
import gradio as gr

from atelier import history, inventory, registry, settings
from atelier.ui import widgets


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs"
        self.output.mkdir()
        for key, value in {"OUTPUT_DIR": self.output, "TMP_DIR": self.root / "tmp"}.items():
            mock = patch.object(settings, key, value)
            mock.start()
            self.addCleanup(mock.stop)

    def image(self, name="image.png", size=(20, 10)):
        path = self.output / name
        Image.new("RGBA", size, (100, 40, 20, 128)).save(path)
        return path

    def test_pagination_is_bounded_and_newest_first(self):
        for i in range(31):
            path = self.image(f"image-{i:02d}.png")
            os.utime(path, ns=(1000 + i, 1000 + i))
        entries, total, page = history.page()
        self.assertEqual((len(entries), total, page), (24, 31, 1))
        self.assertEqual(entries[0].name, "image-30.png")
        self.assertEqual(len(history.page(number=2)[0]), 7)
        self.assertEqual(history.page(number=200)[2], 2)
        self.assertEqual(history.page("IMAGE-30")[1], 1)
        self.assertEqual(history.page("absent")[1:], (0, 1))

    def test_new_and_deleted_files_invalidate_the_snapshot(self):
        # Windows can report an unchanged directory timestamp after a write.
        # Reproduce that deterministically without waiting for a cache TTL.
        stamp = self.output.stat()
        with patch("time.monotonic", return_value=100):
            self.assertEqual(history.page()[1], 0)
            path = self.image()
            os.utime(self.output, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(history.page()[1], 1)
            path.unlink()
            os.utime(self.output, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(history.page()[1], 0)

    def test_replaced_image_invalidates_its_thumbnail(self):
        path = self.image()
        with patch("time.monotonic", return_value=100):
            original = history.page()[0][0]
            old_thumb = history.thumbnail(original)
            self.image(size=(80, 40))
            os.utime(path, ns=(original.modified_ns, original.modified_ns))
            replacement = history.page()[0][0]
            new_thumb = history.thumbnail(replacement)
        self.assertNotEqual(old_thumb, new_thumb)
        with Image.open(new_thumb) as image:
            self.assertEqual(image.size, (80, 40))

    def test_thumbnail_keeps_alpha_and_does_not_modify_original(self):
        path = self.image("été & portrait.png", (1600, 800))
        before = hashlib.sha256(path.read_bytes()).digest()
        entry = history.page()[0][0]
        thumb = history.thumbnail(entry)
        with Image.open(thumb) as im:
            self.assertEqual(im.size, (448, 224))
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.getpixel((10, 10))[3], 128)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)
        with patch.object(Image, "open", side_effect=AssertionError("cached thumbnail decoded again")):
            self.assertEqual(history.thumbnail(entry), thumb)

    def test_corrupt_image_is_skipped_without_breaking_the_page(self):
        (self.output / "broken.png").write_bytes(b"not an image")
        self.assertIsNone(history.thumbnail(history.page()[0][0]))

    def test_paths_cannot_escape_outputs(self):
        for name in ("../private.png", "/tmp/image.png", "sub/image.png", "", "image.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                history.resolve(name)

    def test_symlink_cannot_expose_external_image_or_metadata(self):
        outside = self.root / "private.png"
        Image.new("RGB", (5, 5)).save(outside)
        try:
            (self.output / "link.png").symlink_to(outside)
        except OSError:
            self.skipTest("Creating symlinks needs a Windows privilege")
        self.assertEqual(history.page()[1], 0)
        with self.assertRaises(ValueError):
            history.resolve("link.png")
        path = self.image()
        secret = self.root / "secret.txt"
        secret.write_text("secret")
        path.with_suffix(".txt").symlink_to(secret)
        self.assertNotIn("text", history.details(path.name))

    def test_legacy_sidecars_recover_multiline_prompt_and_model(self):
        path = self.image()
        path.with_suffix(".txt").write_text(
            "First line\nSecond line\nNegative prompt: blur\nModel: Flux (flux2-klein-9b)\nSteps: 4", encoding="utf-8")
        data = history.details(path.name)
        self.assertEqual(data["prompt"], "First line\nSecond line")
        self.assertEqual(data["model_id"], "flux2-klein-9b")

    def test_json_metadata_takes_precedence_over_legacy_text(self):
        path = self.image()
        path.with_suffix(".txt").write_text("Old prompt", encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps({"prompt": "New prompt", "seed": 42}))
        self.assertEqual(history.details(path.name)["prompt"], "New prompt")
        path.with_suffix(".json").write_text("invalid")
        self.assertEqual(history.details(path.name)["prompt"], "Old prompt")

    def test_pruning_only_deletes_disposable_thumbnails(self):
        original = self.image()
        cache = settings.TMP_DIR / "history-thumbnails"
        cache.mkdir(parents=True)
        for i in range(6):
            (cache / f"{i}.png").write_bytes(b"cache")
        history.prune_thumbnails(keep=2)
        self.assertEqual(len(list(cache.glob("*.png"))), 2)
        self.assertTrue(original.is_file())


class PreferencesTests(unittest.TestCase):
    def test_cached_reads_are_isolated_and_notice_external_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            with patch.object(settings, "PREFS_FILE", path):
                path.write_text('{"flags":{"vae_tiling":false}}')
                first = settings.load_prefs()
                first["flags"]["vae_tiling"] = True
                self.assertFalse(settings.load_prefs()["flags"]["vae_tiling"])
                path.write_text('{"theme":"dark"}')
                self.assertEqual(settings.load_prefs()["theme"], "dark")

    def test_malformed_prefs_fall_back_without_writing_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            with patch.object(settings, "PREFS_FILE", path), patch.object(settings, "ensure_dirs") as ensure:
                for raw in ("[]", "null", "1", "bad json"):
                    path.write_text(raw)
                    self.assertEqual(settings.load_prefs(), settings.DEFAULT_PREFS)
                ensure.assert_not_called()


class InventoryTests(unittest.TestCase):
    def test_scandir_counts_nested_files_without_following_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "one").write_bytes(b"123")
            (root / "nested" / "two").write_bytes(b"45")
            try:
                (root / "nested" / "loop").symlink_to(root, target_is_directory=True)
            except OSError:
                pass
            self.assertEqual(inventory.path_size(root), 5)
            self.assertEqual(inventory.path_size(root / "gone"), 0)


class WorkerCleanupTests(unittest.TestCase):
    def test_a_failing_log_consumer_does_not_leave_an_engine_process_running(self):
        from atelier.engine import sdcpp, tools
        real_popen = subprocess.Popen
        cmd = [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"]
        for runner in (sdcpp, tools):
            with self.subTest(runner=runner.__name__):
                children = []

                def launch(*args, **kwargs):
                    proc = real_popen(*args, **kwargs)
                    children.append(proc)
                    return proc

                def log(line):
                    if line == "ready":
                        raise RuntimeError("output consumer failed")

                with patch.object(subprocess, "Popen", side_effect=launch):
                    with self.assertRaisesRegex(RuntimeError, "output consumer failed"):
                        if runner is sdcpp:
                            sdcpp.run(cmd, log=log)
                        else:
                            tools._run_tool(cmd, log=log, err_msg="test")
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(children[0].stdout.closed)
                self.assertNotIn(children[0], runner._ACTIVE)


class StudioWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        with patch("atelier.ui.manage_tab._choices_and_summary", side_effect=AssertionError("startup disk scan")), \
             patch("atelier.ui.manage_tab._move_choices", side_effect=AssertionError("startup move scan")), \
             patch("atelier.history.scan", side_effect=AssertionError("startup history scan")):
            cls.demo = app.build_app()

    def test_root_navigation_is_six_visible_workspaces(self):
        nav = next(b for b in self.demo.blocks.values() if b.elem_id == "studio-nav")
        self.assertEqual([b.id for b in nav.children],
                         ["create", "video", "tools", "history", "models", "system"])

    def test_retired_models_and_toolkit_enlargement_tabs_are_absent(self):
        tabs = [b for b in self.demo.blocks.values() if isinstance(b, gr.Tab)]
        self.assertFalse(any("Ming" in tab.label or "H3" in tab.label for tab in tabs))
        toolkit = next(tab for tab in tabs if tab.id == "toolkit")
        children = next(block for block in toolkit.children if isinstance(block, gr.Tabs))
        self.assertEqual([tab.id for tab in children.children],
                         ["describe", "depth", "bg", "sam", "layers", "face", "adetailer", "creative"])
        seed_model = next(b for b in self.demo.blocks.values()
                          if isinstance(b, gr.Dropdown) and b.label == "Restoration model")
        self.assertEqual(seed_model.value, "3b-w4a8")

    def test_editing_controls_are_connected_to_generation(self):
        def generation_inputs(model_id):
            tab = next(b for b in self.demo.blocks.values() if isinstance(b, gr.Tab) and b.id == model_id)

            def belongs(block):
                while block is not None:
                    if block is tab:
                        return True
                    block = block.parent
                return False

            function = next(fn for fn in self.demo.fns.values()
                            if getattr(fn.fn, "__name__", "") == "do_generate"
                            and any(belongs(block) for block in fn.inputs))
            return function.inputs

        for model in ("qwen-image-2.1-turbo", "flux2-klein-9b"):
            with self.subTest(model=model):
                inputs = generation_inputs(model)
                reference_labels = {block.label for block in inputs if isinstance(block, gr.Image)}
                self.assertTrue({"Image to edit", "Reference 2 (optional)", "Reference 3 (optional)"}.issubset(reference_labels))
                self.assertTrue(any(isinstance(block, gr.Slider) and "Centred outpaint" in block.label for block in inputs))
        krea_inputs = generation_inputs("krea2-turbo")
        self.assertTrue(any(isinstance(block, gr.Checkbox) and "Edit mode" in block.label for block in krea_inputs))

    def test_every_inference_entry_uses_the_same_gpu_slot(self):
        names = {"do_generate", "do_xanax", "run", "do_outpaint", "do_generate3d", "do_depth", "do_bg",
                 "do_describe", "do_upscale", "do_layers", "do_hd", "do_highres", "do_face",
                 "do_adetailer", "do_creative", "_enhance", "_do_bench", "_on_click", "_lay_click",
                 "run_seed", "run_ltx", "run_splat", "install_seed", "install_ltx", "install_splat",
                 "run_views", "convert_panorama", "install_view_caption", "run_svg",
                 "install_svg_cpu", "install_svg_ai", "install_svg_omni", "check_svg_memory", "recover_svg",
                 "install_layer_model", "check_layer_memory", "handler", "deleter",
                 "install_catalog_package", "consolidate_catalog_weights"}
        found = [fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", "") in names]
        self.assertGreaterEqual(len(found), 29)
        for fn in found:
            self.assertEqual(fn.concurrency_id, widgets.GPU_QUEUE["concurrency_id"], fn.name)
            self.assertEqual(fn.concurrency_limit, 1, fn.name)

    def test_style_choices_are_lazy_but_available_on_open(self):
        functions = [fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", "") == "load_style_choices"]
        self.assertEqual(len(functions), len(registry.load_base_models(settings.load_prefs())))
        for fn in functions:
            self.assertEqual(fn.outputs[0].choices, [])
            self.assertEqual(fn.outputs[1].choices, [])
            photo, art, loaded = fn.fn(False)
            self.assertGreater(len(photo["choices"]) + len(art["choices"]), 500)
            self.assertTrue(loaded)
            self.assertNotIn("choices", fn.fn(True)[0])

    def test_unified_catalog_switches_image_weights_and_tool_controls_without_network(self):
        from atelier import model_store
        select = next(fn.fn for fn in self.demo.fns.values()
                      if getattr(fn.fn,"__name__","") == "select_package")
        with patch.object(model_store,"digest",side_effect=AssertionError("startup weight hash")), \
             patch("huggingface_hub.HfApi.model_info",side_effect=AssertionError("automatic network request")):
            flux=select("image:flux2-klein-9b")
            qwen=select("image:qwen-image-2.1-turbo")
            layers=select("addon:qwen-layered")
        self.assertTrue(flux[0]["visible"])
        self.assertNotEqual(flux[4]["choices"],qwen[4]["choices"])
        self.assertFalse(layers[0]["visible"])
        self.assertTrue(layers[1]["visible"])
        self.assertFalse(layers[3]["visible"])
        self.assertIn("Qwen-Image-Layered",layers[6])
        self.assertFalse(layers[7]["visible"])

    def test_transfers_are_direct_and_do_not_wait_for_the_gpu_queue(self):
        functions = [fn for fn in self.demo.fns.values()
                     if getattr(fn.fn, "__name__", "") == "transfer_image"]
        self.assertEqual(len(functions),
                         3 * len(registry.load_base_models(settings.load_prefs())) + 1)
        for fn in functions:
            self.assertFalse(fn.queue)
            self.assertTrue(any(event in ("click", "input") for _, event in fn.targets))

    def test_payload_stays_below_the_original_1_9_mb(self):
        # Measure the UTF-8 payload served to the browser, including schemas.
        # json.dumps() defaults add whitespace and escape every Unicode glyph,
        # overstating actual download size as more model workspaces are added.
        from gradio.routes import App
        from starlette.testclient import TestClient

        with TestClient(App.create_app(self.demo)) as client:
            response = client.get("/config")
        self.assertEqual(response.status_code, 200)
        self.assertLess(len(response.content), 1_520_000)

    def test_failed_hivg_exposes_diagnostics_and_never_a_completed_svg(self):
        from atelier.engine import vectorize
        from atelier.ui import vector_tab
        function = next(fn.fn for fn in self.demo.fns.values()
                        if getattr(fn.fn, "__name__", "") == "run_svg")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "diagnostic.json").write_text(json.dumps({"failed": True, "error": "Incomplete SVG"}))
            (folder / "diagnostics.zip").write_bytes(b"diagnostic bundle")

            def stream(action):
                path = action(lambda line: None)
                yield "Completed.", str(path), "output log"

            with patch.object(vectorize, "run", side_effect=vectorize.SVGDiagnosticError("Incomplete SVG", folder)), \
                 patch.object(vector_tab, "stream_task", side_effect=stream):
                result = list(function("hivg-text", None, "square", 4096, 0, 5))[-1]
            self.assertEqual(result[0], "Failed: Incomplete SVG")
            self.assertIsNone(result[1])
            self.assertIsNone(result[2])
            self.assertEqual(result[3], str(folder / "diagnostics.zip"))
            self.assertIsNone(result[6])

    def test_recovered_partial_is_labeled_and_downloads_its_marked_filename(self):
        from atelier.engine import vectorize
        from atelier.ui import vector_tab
        function = next(fn.fn for fn in self.demo.fns.values()
                        if getattr(fn.fn, "__name__", "") == "recover_svg")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "svg-recovered-test"
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"complete": False, "default_variant": 1,
                "recovered_from_diagnostics": True, "variants": [{"number": 1, "seed": None, "paths": 2,
                "complete": False, "file_stem": "01.partial"}]}))

            def stream(action):
                yield "Completed.", str(action(lambda line: None)), "recovery log"

            with patch.object(settings, "OUTPUT_DIR", Path(directory)), \
                 patch.object(vectorize, "recover", return_value=folder), \
                 patch.object(vector_tab, "stream_task", side_effect=stream):
                result = list(function("old-diagnostics.zip"))[-1]
            self.assertIn("Partial SVG exported", result[0])
            self.assertTrue(result[1].endswith("01.partial.png"))
            self.assertTrue(result[2].endswith("01.partial.svg"))
            self.assertIn("PARTIAL SVG", result[7])
            self.assertIn("no model inference", result[7])

    def test_qwen_layer_mode_dispatches_all_controls_and_returns_real_exports(self):
        from atelier.engine import layer_decompose
        from atelier.ui import toolkit_tab
        function = next(fn.fn for fn in self.demo.fns.values()
                        if getattr(fn.fn, "__name__", "") == "do_layers")
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'manifest.json').write_text(json.dumps({'layers':[{'file':'layer-01.png','name':'Layer 01'}]}))
            def stream(action):
                yield 'Completed.',str(action(lambda line:None)),'step 2/2'
            options={**layer_decompose.DEFAULTS,'layers':2,'steps':2,'seed':42}
            with patch.object(layer_decompose,'run',return_value=folder) as run, \
                 patch.object(toolkit_tab,'stream_task',side_effect=stream):
                result=list(function(Image.new('RGB',(32,32)),'qwen',[],12,.4,24,False,False,'sam2',
                                     'a poster',*options.values()))[-1]
            self.assertEqual(run.call_args.args[1],options)
            self.assertEqual(run.call_args.args[2],'a poster')
            self.assertEqual(result[0],[str(folder/'layers.psd'),str(folder/'layers.zip')])
            self.assertEqual(result[2],str(folder/'composite.png'))
            self.assertEqual(result[3],[(str(folder/'layer-01.png'),'Layer 01')])


if __name__ == "__main__":
    unittest.main()
