import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier.engine import tools


class SeedVr2CommandTests(unittest.TestCase):
    def test_secondary_gpu_is_mapped_as_cuda_one(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            fake_python = base / "python.exe"
            fake_python.touch()
            image = Image.new("RGB", (32, 24), "navy")

            def fake_run(cmd, log, err_msg, gpu_index=None, cwd=None, env=None):
                captured.update(cmd=cmd, cwd=cwd, env=env)
                output = Path(cmd[cmd.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), "navy").save(output)

            with (
                patch.object(tools, "seedvr2_is_installed", return_value=True),
                patch.object(tools, "_seedvr2_python", return_value=fake_python),
                patch.object(tools, "SEEDVR2_SOURCE_DIR", source),
                patch.object(tools, "SEEDVR2_MODEL_DIR", base / "models"),
                patch.object(tools.settings, "OUTPUT_DIR", base / "outputs"),
                patch.object(tools.settings, "load_prefs", return_value={
                    "encoder_gpu_index": 5, "text_gpu_index": 5,
                }),
                patch.object(tools.settings, "child_env", return_value=os.environ.copy()),
                patch.object(tools, "_gen_gpu_index", return_value=2),
                patch.object(tools, "_run_tool", side_effect=fake_run),
            ):
                output = tools.seedvr2_upscale(
                    image, resolution=2048, blocks_to_swap=16,
                    offload="secondary")

        self.assertTrue(output.name.startswith("seedvr2-"))
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertEqual(captured["cwd"], source)
        for option in ("--dit_offload_device", "--vae_offload_device",
                       "--tensor_offload_device"):
            self.assertEqual(captured["cmd"][captured["cmd"].index(option) + 1], "1")
        self.assertIn("--swap_io_components", captured["cmd"])


if __name__ == "__main__":
    unittest.main()
