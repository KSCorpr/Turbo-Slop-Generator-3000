"""H3 Turbo and Ref2VA: selected weights, safe memory, playable output."""
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atelier import downloader, settings, video
from atelier.engine import sdcpp
from atelier.engine import video as engine
from scripts import maintenance


OPTIONS = frozenset({
    "--video-frames", "--diffusion-model", "--vae", "--llm", "--auto-fit",
    "--max-vram", "--flow-shift", "--end-img", "--diffusion-fa",
    "--vae-tiling", "--rng", "--params-backend", "--offload-to-cpu",
    "--ref-image",
})


class MiniMaxH3Tests(unittest.TestCase):
    def test_missing_converter_repairs_the_running_portable_python(self):
        calls = []

        def pip_install(command, **kwargs):
            calls.append(command)
            sys.modules["imageio_ffmpeg"] = SimpleNamespace(
                get_ffmpeg_exe=lambda: "ffmpeg-installed.exe")
            return SimpleNamespace(returncode=0, stdout="installed", stderr="")

        with patch.dict(sys.modules, {"imageio_ffmpeg": None}), \
             patch.object(engine.shutil, "which", return_value=None), \
             patch.object(engine.subprocess, "run", side_effect=pip_install):
            self.assertEqual(engine._ffmpeg_exe(), "ffmpeg-installed.exe")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("imageio-ffmpeg>=0.6,<1", calls[0])

    def test_can_recover_previous_avi_and_keeps_it_when_converter_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "OUTPUT_DIR", Path(tmp)):
            avi = Path(tmp) / "minimax-h3-turbo-previous.avi"
            avi.write_bytes(b"saved video")
            self.assertIn(avi.name, engine.saved_avis())
            with patch.object(engine, "_ffmpeg_exe",
                              side_effect=sdcpp.EngineError("offline")):
                with self.assertRaisesRegex(sdcpp.EngineError, "offline"):
                    engine.convert_saved_avi(avi.name)
            self.assertTrue(avi.exists())
            with self.assertRaises(sdcpp.EngineError):
                engine.convert_saved_avi("../other.avi")

            def encoded(command, **kwargs):
                Path(command[-1]).write_bytes(b"playable mp4")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(engine, "_ffmpeg_exe", return_value="ffmpeg.exe"), \
                 patch.object(engine.subprocess, "run", side_effect=encoded):
                mp4 = engine.convert_saved_avi(avi.name)
            self.assertEqual(mp4.read_bytes(), b"playable mp4")
            self.assertFalse(avi.exists())

    def test_reference_avi_is_recoverable_too(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "OUTPUT_DIR", Path(tmp)):
            avi = Path(tmp) / "minimax-h3-ref2va-previous.avi"
            avi.write_bytes(b"saved video")
            self.assertIn(avi.name, engine.saved_avis())
            with patch.object(engine, "encode_mp4", return_value=avi.with_suffix(".mp4")):
                self.assertEqual(engine.convert_saved_avi(avi.name).suffix, ".mp4")
            with self.assertRaises(sdcpp.EngineError):
                engine.convert_saved_avi("minimax-h3-ref2va-../other.avi")

    def test_maintenance_never_marks_downloaded_h3_repositories_as_orphans(self):
        expected = maintenance._expected_model_dirs()
        for repo in (video.DIFFUSION_REPO, video.ENCODER_REPO, video.VAE_REPO):
            self.assertIn(settings.model_repo_dir(repo).name, expected)

    def test_downloads_only_three_selected_weights_and_no_audio(self):
        requested = []
        with patch.object(downloader, "download_component",
                          side_effect=lambda comp, log=None: requested.append(comp)):
            events = list(video.download(video.DEFAULT_DIFFUSION,
                                         video.DEFAULT_ENCODER))
        self.assertEqual([c.role for c in requested],
                         ["diffusion", "text_encoder", "vae"])
        self.assertEqual(requested[0].repo, video.DIFFUSION_REPO)
        self.assertEqual(requested[1].requested(), video.DEFAULT_ENCODER)
        self.assertEqual(requested[2].requested(), video.VAE)
        self.assertNotIn("audio-vae", " ".join(events))
        with self.assertRaises(ValueError):
            list(video.download("../malicious.gguf", video.DEFAULT_ENCODER))

    def test_reference_weights_reuse_the_h3_encoder_and_vae(self):
        prefs = {"video_model_files": {
            "diffusion": video.DEFAULT_DIFFUSION,
            "ref_diffusion": "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
            "text_encoder": video.DEFAULT_ENCODER}}
        ref, encoder = video.selected(prefs, mode="refs")
        self.assertEqual(ref, "minimax_h3_ref2va_pruned-Q4_K_M.gguf")
        self.assertEqual(video.selected(prefs)[0], video.DEFAULT_DIFFUSION)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)):
            turbo = video.weights(video.DEFAULT_DIFFUSION, encoder)
            reference = video.weights(ref, encoder, mode="refs")
            self.assertNotEqual(turbo[0], reference[0])
            self.assertEqual(turbo[1:], reference[1:])
            for path in (*turbo, reference[0]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            self.assertTrue(video.ready(ref, encoder, mode="refs"))
            self.assertEqual(video.delete(ref, encoder, mode="refs"), 1)
            self.assertTrue(all(p.is_file() for p in turbo))
            self.assertFalse(reference[0].exists())

    def test_reference_download_requests_only_its_diffusion_and_shared_files(self):
        requested = []
        with patch.object(downloader, "download_component",
                          side_effect=lambda comp, log=None: requested.append(comp)):
            list(video.download(video.DEFAULT_REF_DIFFUSION,
                                video.DEFAULT_ENCODER, mode="refs"))
        self.assertEqual([(c.repo, c.requested()) for c in requested], [
            (video.ENCODER_REPO, video.DEFAULT_REF_DIFFUSION),
            (video.ENCODER_REPO, video.DEFAULT_ENCODER),
            (video.VAE_REPO, video.VAE)])
        with self.assertRaises(ValueError):
            list(video.download(video.DEFAULT_DIFFUSION,
                                video.DEFAULT_ENCODER, mode="refs"))

    def test_deleting_one_quant_keeps_shared_encoder_and_vae(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(settings, "MODELS_DIR", Path(tmp)):
            chosen, encoder, vae = video.weights(video.DEFAULT_DIFFUSION,
                                                  video.DEFAULT_ENCODER)
            alternate = settings.model_repo_dir(video.DIFFUSION_REPO) / \
                next(name for name in video.DIFFUSION
                     if name != video.DEFAULT_DIFFUSION)
            for path in (chosen, alternate, encoder, vae):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            self.assertEqual(video.delete(video.DEFAULT_DIFFUSION,
                                          video.DEFAULT_ENCODER), 1)
            self.assertFalse(chosen.exists())
            self.assertTrue(alternate.is_file())
            self.assertTrue(encoder.is_file())
            self.assertTrue(vae.is_file())

    def test_video_cli_fixes_turbo_schedule_and_has_no_audio_argument(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            files = [root / n for n in ("sd-cli", "dit.gguf", "te.gguf",
                                       "vae.safetensors", "first.png", "last.png")]
            for file in files:
                file.write_bytes(b"fixture")
            with patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                 patch.object(sdcpp, "binary_help", return_value="--auto-fit on|off"):
                cmd = engine.build_command(*files[:4], "a tracking shot",
                                           root / "clip.avi", first=files[4],
                                           last=files[5], frames=39)
            def value(flag):
                return cmd[cmd.index(flag) + 1]
            self.assertEqual(value("-M"), "vid_gen")
            self.assertEqual(value("--steps"), "4")
            self.assertEqual(value("--flow-shift"), "6")
            self.assertEqual(value("--cfg-scale"), "1.0")
            self.assertEqual(value("--video-frames"), "39")
            self.assertEqual(value("--fps"), "24")
            self.assertEqual(value("--auto-fit"), "on")
            self.assertEqual(value("--max-vram"), "-1")
            self.assertIn("--vae-tiling", cmd)
            self.assertEqual(value("--end-img"), str(files[5]))
            self.assertNotIn("--audio-vae", cmd)

    def test_reference_cli_passes_three_images_in_order_without_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / n for n in ("sd-cli", "ref.gguf", "te.gguf",
                                       "vae.safetensors", "one.png", "two.png",
                                       "three.png")]
            for file in files:
                file.write_bytes(b"fixture")
            with patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                 patch.object(sdcpp, "binary_help", return_value="--auto-fit on|off"):
                cmd = engine.build_command(*files[:4],
                    "Use <Picture 1>, <Picture 2>, <Picture 3>", root / "clip.avi",
                    mode="refs", refs=files[4:])
                self.assertEqual(cmd.count("--ref-image"), 3)
                self.assertEqual([cmd[i + 1] for i, v in enumerate(cmd)
                                  if v == "--ref-image"], [str(p) for p in files[4:]])
                self.assertEqual(cmd[cmd.index("--steps") + 1], "50")
                self.assertEqual(cmd[cmd.index("--flow-shift") + 1], "12")
                self.assertNotIn("-i", cmd)
                self.assertNotIn("--end-img", cmd)
                self.assertNotIn("--audio-vae", cmd)
                with self.assertRaisesRegex(sdcpp.EngineError, "cannot use first/last"):
                    engine.build_command(*files[:4], "prompt", root / "clip.avi",
                                         mode="refs", refs=files[4:5], first=files[4])
                with self.assertRaisesRegex(sdcpp.EngineError, "1–3 images"):
                    engine.build_command(*files[:4], "prompt", root / "clip.avi",
                                         mode="refs")
                with self.assertRaisesRegex(sdcpp.EngineError, "need the H3 Ref2VA"):
                    engine.build_command(*files[:4], "prompt", root / "clip.avi",
                                         refs=files[4:])
                with patch.object(sdcpp, "supported_options", return_value=OPTIONS - {"--ref-image"}):
                    with self.assertRaisesRegex(sdcpp.EngineError, "update.bat"):
                        engine.build_command(*files[:4], "prompt", root / "clip.avi",
                                             mode="refs", refs=files[4:5])

    def test_out_of_memory_retries_smaller_without_losing_requested_frames(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            files = [root / n for n in ("sd-cli", "dit.gguf", "te.gguf",
                                       "vae.safetensors")]
            for file in files:
                file.write_bytes(b"fixture")
            commands = []

            def run(cmd, **kwargs):
                commands.append(cmd)
                if len(commands) == 1:
                    raise sdcpp.VramError("out of memory")
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"AVI fixture")

            def fake_encode(path, log=None):
                return path.with_suffix(".mp4")

            with patch.object(settings, "find_sd_cli", return_value=files[0]), \
                 patch.object(sdcpp, "unique_output", return_value=root / "clip.avi"), \
                 patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                 patch.object(sdcpp, "binary_help", return_value="--auto-fit on|off"), \
                 patch.object(sdcpp, "run", side_effect=run), \
                 patch.object(engine, "encode_mp4", side_effect=fake_encode):
                result = engine.generate(*files[1:], "moving camera", frames=56)
            self.assertEqual(result, root / "clip.mp4")
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[1][commands[1].index("-W") + 1], "640")
            self.assertEqual(commands[1][commands[1].index("--video-frames") + 1],
                             "56")

    def test_reference_retry_keeps_three_images_in_the_same_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / n for n in ("sd-cli", "ref.gguf", "te.gguf",
                                       "vae.safetensors", "one.png", "two.png",
                                       "three.png")]
            for file in files:
                file.write_bytes(b"fixture")
            commands = []

            def run(cmd, **kwargs):
                commands.append(cmd)
                if len(commands) == 1:
                    raise sdcpp.VramError("out of memory")
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"AVI fixture")

            with patch.object(settings, "find_sd_cli", return_value=files[0]), \
                 patch.object(sdcpp, "unique_output", return_value=root / "clip.avi"), \
                 patch.object(sdcpp, "supported_options", return_value=OPTIONS), \
                 patch.object(sdcpp, "binary_help", return_value="--auto-fit on|off"), \
                 patch.object(sdcpp, "run", side_effect=run), \
                 patch.object(engine, "encode_mp4", return_value=root / "clip.mp4"):
                engine.generate(*files[1:4], "moving camera", mode="refs",
                                refs=files[4:])
            self.assertEqual(len(commands), 2)
            for cmd in commands:
                self.assertEqual([cmd[i + 1] for i, v in enumerate(cmd)
                                  if v == "--ref-image"], [str(p) for p in files[4:]])
            self.assertEqual(commands[1][commands[1].index("-W") + 1], "640")


if __name__ == "__main__":
    unittest.main()
