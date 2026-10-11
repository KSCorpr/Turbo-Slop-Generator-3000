"""Local client, workflow routing and cache-sensitive comparisons without weights.

The installer additionally runs contracts against a real pinned Comfy server.
These tests exercise cancellation, protocol races and image boundaries.
"""
from contextlib import ExitStack, nullcontext
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import websocket

from atelier import benchmark, comfy_runtime, registry, settings
from atelier.engine import comfy, comfy_workflows as workflows, generate, sdcpp


def png(mode="RGBA", color=(20, 40, 60, 100), size=(8, 8)):
    data = io.BytesIO()
    Image.new(mode, size, color).save(data, format="PNG")
    return data.getvalue()


class WorkflowsTests(unittest.TestCase):
    def qwen(self, **kw):
        return workflows.qwen(diffusion="turbo.safetensors", encoder="encoder.safetensors",
            vae="vae.safetensors", prompt="a transparent object", width=1024,
            height=1024, seed=42, prefix="test", **kw)

    @staticmethod
    def inputs(graph, kind):
        return [n["inputs"] for n in graph.nodes.values() if n["class_type"] == kind]

    def test_turbo_uses_the_full_distilled_schedule_and_cfg_one(self):
        graph = self.qwen()
        sigmas = self.inputs(graph, "ManualSigmas")[0]["sigmas"].split(",")
        self.assertEqual(len(sigmas), 9)
        self.assertEqual((sigmas[0], sigmas[-1]), ("1.0", "0.0"))
        sampler = self.inputs(graph, "SamplerCustom")[0]
        self.assertEqual((sampler["cfg"], sampler["noise_seed"]), (1.0, 42))
        self.assertEqual(self.inputs(graph, "KSamplerSelect")[0]["sampler_name"], "euler")
        self.assertEqual(graph.nodes[graph.output]["class_type"], "SaveImage")

    def test_three_references_use_rgba_loader_and_native_dynamic_inputs(self):
        graph = self.qwen(references=["one.png", "two.png", "three.png"])
        self.assertEqual([v["image"] for v in self.inputs(graph, "TurboSlopLoadRGBA")],
                         ["one.png", "two.png", "three.png"])
        encoded = self.inputs(graph, "TextEncodeQwenImage21")[0]
        self.assertTrue(all(f"images.image_{i}" in encoded for i in range(1, 4)))
        self.assertFalse(self.inputs(graph, "LoadImage"))
        self.assertFalse(self.inputs(graph, "TurboSlopEmptyQwen21"))

    def test_mask_applies_to_encoded_source_not_an_empty_latent(self):
        graph = self.qwen(init="source.png", mask="paint.png")
        self.assertTrue(self.inputs(graph, "VAEEncodeTiled"))
        noise = self.inputs(graph, "SetLatentNoiseMask")[0]
        self.assertEqual(graph.nodes[noise["samples"][0]]["class_type"], "VAEEncodeTiled")
        self.assertEqual(self.inputs(graph, "LoadImageMask")[0]["channel"], "red")

    def test_sdxl_preserves_negative_guidance_sampler_and_denoise(self):
        graph = workflows.sdxl(checkpoint="base.safetensors", prompt="sharp", negative="blur",
            width=1024, height=768, seed=12, steps=30, cfg=6, prefix="test",
            init="tile.png", mask="paint.png", strength=0.35,
            sampler="dpm++2m", scheduler="karras", controlnet="control.safetensors")
        sample = self.inputs(graph, "KSampler")[0]
        self.assertEqual((sample["steps"], sample["cfg"], sample["denoise"]), (30, 6, 0.35))
        self.assertEqual((sample["sampler_name"], sample["scheduler"]), ("dpmpp_2m", "karras"))
        self.assertEqual([v["text"] for v in self.inputs(graph, "CLIPTextEncode")], ["sharp", "blur"])
        self.assertTrue(self.inputs(graph, "ControlNetApplyAdvanced"))

    def test_comfy_catalog_uses_native_encoder_and_reuses_diffusion_vae(self):
        native = registry.get_base_model("qwen-image-2.1-turbo", {"image_backend": "comfy"})
        classic = registry.get_base_model(native.id, {"image_backend": "sdcpp"})
        a, b = ({c.role: c for c in model.components} for model in (native, classic))
        self.assertTrue(a["text_encoder"].requested().endswith(".safetensors"))
        self.assertTrue(b["text_encoder"].requested().endswith(".gguf"))
        self.assertNotIn("text_encoder_vision", a)
        for role in ("diffusion", "vae"):
            self.assertEqual((a[role].repo, a[role].requested()), (b[role].repo, b[role].requested()))


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("OUTPUT_DIR", "TMP_DIR"):
            directory = self.folder / name
            directory.mkdir()
            self.stack.enter_context(patch.object(settings, name, directory))
        self.stack.enter_context(patch.object(comfy_runtime, "root", return_value=self.folder / "runtime"))
        self.engine = comfy.Engine()
        self.engine.port = 8099
        self.socket = Mock()
        self.socket.recv.side_effect = websocket.WebSocketTimeoutException()
        self.connect = self.stack.enter_context(patch.object(websocket, "create_connection", return_value=self.socket))
        self.engine._http.get = Mock(return_value=Mock(content=png()))
        self.graph = workflows.Workflow()
        self.graph.finish(self.graph.add("TurboSlopLoadRGBA", image="source.png"), "test")
        self.history = {"outputs": {self.graph.output: {"images": [
            {"filename": "test.png", "subfolder": "turboslop", "type": "output"}]}},
            "status": {"status_str": "success"}}

    def protocol(self, histories=None):
        iterator = iter(histories if histories is not None else [self.history])
        order = []
        self.connect.side_effect = lambda *a, **k: (order.append("subscribe") or self.socket)
        def request(path, *, payload=None, **unused):
            order.append(path)
            if path == "/prompt":
                return Mock(**{"json.return_value": {"prompt_id": "job", "node_errors": {}}})
            if path == "/history/job":
                value = next(iterator, self.history)
                return Mock(**{"json.return_value": {"job": value} if value else {}})
            return Mock(**{"json.return_value": {"queue_running": [], "queue_pending": []}})
        self.engine.request = Mock(side_effect=request)
        return order

    def test_immediate_cached_completion_is_subscribed_before_submission(self):
        order = self.protocol()
        outputs = self.engine.execute(self.graph)
        self.assertLess(order.index("subscribe"), order.index("/prompt"))
        with Image.open(outputs[0]) as im:
            self.assertEqual((im.mode, im.getpixel((0, 0))), ("RGBA", (20, 40, 60, 100)))
        self.assertIn("/history", order)
        self.socket.close.assert_called_once()

    def test_dropped_websocket_recovers_final_image_from_history(self):
        self.protocol()
        self.socket.recv.side_effect = websocket.WebSocketConnectionClosedException()
        self.assertEqual(len(self.engine.execute(self.graph)), 1)
        self.socket.close.assert_called_once()

    def test_live_previews_do_not_skip_indices_for_other_binary_events(self):
        self.protocol([None, None, self.history])
        blob = struct.pack(">II", 1, 2) + png("RGB", "red")
        self.socket.recv.side_effect = [struct.pack(">II", 9, 0), blob, blob]
        template = settings.TMP_DIR / "preview_%03d.png"
        with patch.object(comfy.time, "monotonic", side_effect=range(0, 200)):
            self.engine.execute(self.graph, preview_path=template, preview_offset=8)
        self.assertTrue((settings.TMP_DIR / "preview_008.png").is_file())
        self.assertTrue((settings.TMP_DIR / "preview_009.png").is_file())
        self.assertFalse((settings.TMP_DIR / "preview_010.png").exists())

    def test_foreign_progress_and_errors_are_ignored(self):
        self.protocol()
        self.socket.recv.side_effect = [json.dumps({"type": "execution_error", "data": {
            "prompt_id": "someone-else", "exception_message": "out of memory"}})]
        self.assertEqual(len(self.engine.execute(self.graph)), 1)

    def test_out_of_memory_cleans_up_and_targets_the_own_job(self):
        self.protocol()
        self.socket.recv.side_effect = [json.dumps({"type": "execution_error", "data": {
            "prompt_id": "job", "exception_message": "CUDA out of memory"}})]
        with self.assertRaises(sdcpp.VramError):
            self.engine.execute(self.graph)
        calls = [(c.args[0], c.kwargs.get("payload")) for c in self.engine.request.call_args_list]
        self.assertIn(("/interrupt", {"prompt_id": "job"}), calls)
        self.assertIn(("/queue", {"delete": ["job"]}), calls)
        self.assertIn(("/history", {"delete": ["job"]}), calls)
        self.assertIsNone(self.engine._prompt_id)

    def test_history_errors_are_not_mistaken_for_completed_images(self):
        self.protocol([{"status": {"status_str": "error", "messages": [["execution_error", {
            "exception_message": "CUDA out of memory"}]]}}])
        with self.assertRaises(sdcpp.VramError):
            self.engine.execute(self.graph)

    def test_cancel_sets_flag_and_interrupts_only_active_prompt(self):
        self.protocol()
        with self.engine.session():
            self.engine._prompt_id = "job"
            self.assertIn("Stopping", self.engine.cancel())
            with self.assertRaisesRegex(sdcpp.EngineError, "cancelled"):
                self.engine.check()
        self.engine.request.assert_any_call("/interrupt", payload={"prompt_id": "job"}, timeout=2)
        self.assertFalse(self.engine._busy)

    def test_handoff_rejects_a_still_active_sampler(self):
        with self.engine.session(), self.assertRaisesRegex(sdcpp.EngineError, "still running"):
            self.engine.release()

    def test_request_stays_on_loopback_and_never_follows_redirects(self):
        self.engine._http.request = Mock(return_value=Mock(status_code=200, text="redirect"))
        self.engine.request("/prompt", payload={"prompt": "private"})
        self.assertFalse(self.engine._http.trust_env)
        args = self.engine._http.request.call_args
        self.assertEqual(args.args[:2], ("POST", "http://127.0.0.1:8099/prompt"))
        self.assertFalse(args.kwargs["allow_redirects"])
        for route in ("https://remote/prompt", "//remote/prompt"):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.engine.request(route)
        self.engine._http.request.return_value.status_code = 302
        with self.assertRaises(sdcpp.EngineError):
            self.engine.request("/view")

    def test_upload_keeps_alpha_and_deduplicates_owned_copies(self):
        source = self.folder / "input.png"
        source.write_bytes(png())
        captured = []
        def request(path, *, files, **unused):
            name, contents, mime = files["image"]
            captured.append(contents)
            return Mock(**{"json.return_value": {"name": name, "subfolder": ""}})
        self.engine.request = Mock(side_effect=request)
        self.assertEqual(self.engine.upload(source, rgba=True), self.engine.upload(source, rgba=True))
        self.assertEqual(len(captured), 1)
        with Image.open(io.BytesIO(captured[0])) as im:
            self.assertEqual(im.getpixel((0, 0)), (20, 40, 60, 100))

    def test_runtime_only_whitelists_our_nodes_and_maps_gpu_by_uuid(self):
        with patch.object(settings, "child_env", return_value={}), \
             patch.object(comfy_runtime.hardware, "_nvidia_smi", return_value="0, GPU-first\n1, GPU-second"):
            self.assertEqual(comfy_runtime.gpu_env(1)["CUDA_VISIBLE_DEVICES"], "GPU-second")
        command = comfy_runtime.command(8099, Path("paths.yaml"))
        self.assertEqual(command[command.index("--listen") + 1], "127.0.0.1")
        self.assertIn("--offline", command)
        self.assertIn("--disable-all-custom-nodes", command)
        self.assertEqual(command[command.index("--whitelist-custom-nodes") + 1], "turboslop")

    def test_explicit_cpu_runtime_still_uses_cpu_on_a_machine_with_nvidia(self):
        with patch.object(comfy_runtime, "manifest", return_value={"flavor": "cpu"}):
            self.assertIn("--cpu", comfy_runtime.command(8099, Path("paths.yaml"), gpu=Mock()))

    def test_compatible_images_keep_the_same_process_then_handoff_releases_it(self):
        from atelier.engine import trellis
        proc = Mock(stdout=io.StringIO("ready\n"))
        proc.poll.return_value = None
        def request(*args, **kw):
            return Mock(**{"json.return_value": {"token": self.engine._token, "schema": comfy_runtime.SCHEMA}})
        with patch.object(comfy_runtime, "ready", return_value=True), \
             patch.object(comfy_runtime, "selected_gpu", return_value=None), \
             patch.object(comfy_runtime, "manifest", return_value={"flavor": "cpu"}), \
             patch.object(comfy_runtime, "gpu_env", return_value={}), \
             patch.object(trellis, "resident_is_running", return_value=False), \
             patch.object(comfy.subprocess, "run"), \
             patch.object(comfy.subprocess, "Popen", return_value=proc) as spawn, \
             patch.object(self.engine, "request", side_effect=request):
            roots = (self.folder,)
            self.engine.ensure(roots)
            self.engine.ensure(roots)
            spawn.assert_called_once()
            self.engine.release()
        self.assertIsNone(self.engine._proc)
        self.assertIsNone(self.engine.port)
        if comfy.os.name == "nt":
            # Windows uses taskkill on the process tree instead of terminate.
            proc.wait.assert_called()
        else:
            proc.terminate.assert_called_once()


class IntegrationTests(unittest.TestCase):
    def test_comfy_route_needs_no_sd_cli_and_keeps_completed_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "completed.png"
            output.write_bytes(png())
            def infer(**kw):
                kw["on_output"]([output], 987)
                raise sdcpp.EngineError("Generation cancelled. Completed images have been kept.")
            with patch.object(generate, "resolve_model_files", return_value={}), \
                 patch.object(comfy, "generate", side_effect=infer), \
                 patch.object(settings, "find_sd_cli") as cli, \
                 patch.object(generate, "_check_qwen_engine") as legacy:
                with self.assertRaisesRegex(sdcpp.EngineError, "cancelled"):
                    generate.generate("qwen-image-2.1-turbo", "a glass", "", 8, 1,
                                      1024, 1024, -1, 2, prefs_override={"image_backend": "comfy"})
            cli.assert_not_called()
            legacy.assert_not_called()
            data = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual((data["seed"], data["prompt"]), (987, "a glass"))

    def test_catalogue_sdxl_reuses_the_toolkit_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(settings, "ROOT", Path(temporary)):
            path = settings.ROOT / "tools_repo/upscale/sd_xl_base_1.0.safetensors"
            path.parent.mkdir(parents=True)
            path.touch()
            model = registry.get_base_model("sdxl-base", {"image_backend": "comfy"})
            component = next(c for c in model.components if c.role == "model")
            self.assertEqual(registry.resolve_component_path(component), path)

    def test_tile_blending_keeps_all_four_outside_edges_and_corners(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            for name in ("OUTPUT_DIR", "TMP_DIR"):
                path = folder / name
                path.mkdir()
                stack.enter_context(patch.object(settings, name, path))
            source, checkpoint = folder / "source.png", folder / "base.safetensors"
            Image.new("RGB", (640, 640), "white").save(source)
            checkpoint.touch()
            engine = Mock()
            engine.session.return_value = nullcontext()
            crops = []
            def upload(path):
                with Image.open(path) as image:
                    crops.append(image.size)
                return "tile.png"
            def execute(*args, **kw):
                path = settings.OUTPUT_DIR / "tile.png"
                Image.new("RGB", crops[-1], (200, 120, 80)).save(path)
                return [path]
            engine.upload.side_effect = upload
            engine.execute.side_effect = execute
            stack.enter_context(patch.object(comfy, "ENGINE", engine))
            output = comfy.upscale(source, checkpoint=checkpoint, width=640, height=640,
                                   tile=512, overlap=128, steps=2)
            with Image.open(output) as im:
                pixels = np.asarray(im)
            self.assertGreaterEqual(int(pixels[:, :, 0].min()), 199)
            self.assertEqual(tuple(pixels[0, 0]), (200, 120, 80))
            self.assertEqual(tuple(pixels[-1, -1]), (200, 120, 80))
            self.assertEqual(len(crops), 4)
            self.assertFalse((settings.OUTPUT_DIR / "tile.png").exists())

    def test_rejected_options_fail_before_starting_runtime(self):
        model = registry.get_base_model("qwen-image-2.1-turbo", {"image_backend": "comfy"})
        args = dict(model=model, files={}, prompt="test", negative="", steps=8,
                    cfg_scale=1, width=1024, height=1024, seed=1, batch_count=1)
        with patch.object(comfy.ENGINE, "ensure") as start:
            for change in ({"steps": 40}, {"cfg_scale": 6}, {"sampler": "heun"},
                           {"schedule": "karras"}, {"flow_shift": 1}, {"circular": True},
                           {"width": 1030}, {"batch_count": 0}):
                with self.subTest(change=change), self.assertRaises(sdcpp.EngineError):
                    comfy.generate(**{**args, **change})
        start.assert_not_called()

    def test_comparison_varies_seeds_uses_full_preset_and_does_not_save_preferences(self):
        seen = []
        def measure(mid, prefs, defaults, log, **kw):
            seen.append((prefs["image_backend"], kw["seed"], kw["full_steps"], kw["size"]))
            return (10 if prefs["image_backend"] == "sdcpp" else 5), [], {}
        with patch.object(benchmark.registry, "model_is_ready", return_value=True), \
             patch.object(benchmark, "_one_run", side_effect=measure), \
             patch.object(benchmark, "_write_report", side_effect=lambda prefix, payload: payload), \
             patch.object(benchmark.diagnostics, "system_report", return_value={}), \
             patch.object(comfy.ENGINE, "close"), \
             patch.object(settings, "save_prefs") as save:
            data = benchmark.run_engine_benchmark()
        save.assert_not_called()
        for backend in ("sdcpp", "comfy"):
            rows = [r for r in seen if r[0] == backend]
            self.assertEqual([r[1] for r in rows], [424242, 424243, 424244, 424245])
            self.assertTrue(all(r[2:] == (True, 1024) for r in rows))
        self.assertEqual(data["comfy_speedup"], 2.0)
        self.assertNotIn("recommended_prefs_patch", data)

    def test_classic_placement_measurement_cannot_silently_route_to_comfy(self):
        prefs = benchmark._benchmark_prefs({"image_backend": "comfy"}, {})
        self.assertEqual(prefs["image_backend"], "sdcpp")


if __name__ == "__main__":
    unittest.main()
