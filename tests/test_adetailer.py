"""ADetailer : la commande envoyée au moteur, et les refus avant de l'envoyer.

Le binaire ne tourne pas ici (il lui faut le moteur et 8 Go de poids), mais
tout ce qui DÉCIDE à sa place se teste : les options exactes que le mode
attend, le fait que le modèle et le placement mémoire viennent du même chemin
que la génération normale, et les trois raisons possibles de ne pas pouvoir
lancer — nommées plutôt que masquées.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import registry, settings
from atelier.engine import generate, sdcpp, tools


def _model(tmp: Path) -> registry.BaseModel:
    diffusion = tmp / "krea2.gguf"
    diffusion.write_bytes(b"weights")
    return registry.BaseModel(
        id="krea2-turbo", name="Krea 2 Turbo", family="krea2", tags=[],
        description="", components=[
            registry.Component("diffusion", "repo", "krea2.gguf", None)],
        defaults={"steps": 8, "cfg_scale": 1.0, "sampler": "euler"},
        vram_min_gb=12, presets=[])


class CommandTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _model(root)
            with patch.object(settings, "find_sd_cli",
                              return_value=Path("sd-cli")), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset({"--ad-model"})), \
                 patch.object(registry, "get_base_model", return_value=model), \
                 patch.object(generate, "resolve_model_files",
                              return_value={"model_path": None,
                                            "diffusion": root / "krea2.gguf",
                                            "vae": None, "enc": None,
                                            "uncond": None, "t5xxl": None,
                                            "clip_l": None,
                                            "llm_vision": None}), \
                 patch.object(generate, "_resolved_flags",
                              return_value=({}, 0)):
                return generate.adetailer_command(
                    "krea2-turbo", root / "in.png", root / "out.png",
                    root / "hand_yolov8n.safetensors", **kwargs)

    def test_it_runs_the_dedicated_mode_on_an_existing_image(self):
        cmd = self._cmd()
        self.assertEqual(cmd[cmd.index("-M") + 1], "adetailer")
        self.assertIn("-i", cmd)          # image d'entrée, pas du text2img
        self.assertIn("--ad-model", cmd)

    def test_the_detector_is_passed_as_the_ad_model(self):
        cmd = self._cmd()
        self.assertTrue(
            cmd[cmd.index("--ad-model") + 1].endswith("hand_yolov8n.safetensors"))

    def test_the_redraw_amount_travels_as_strength(self):
        """`denoising_strength` est hérité de --strength dans ce mode."""
        cmd = self._cmd(denoise=0.55)
        self.assertEqual(cmd[cmd.index("--strength") + 1], "0.55")

    def test_detection_settings_go_into_one_comma_separated_argument(self):
        cmd = self._cmd(extra={"confidence": 0.25, "mask_blur": 8})
        args = cmd[cmd.index("--extra-ad-args") + 1]
        self.assertEqual(sorted(args.split(",")),
                         ["confidence=0.25", "mask_blur=8"])

    def test_an_empty_extra_value_is_not_sent(self):
        """« key= » sans valeur ferait échouer le parseur du moteur."""
        cmd = self._cmd(extra={"confidence": 0.3, "sample_method": ""})
        self.assertEqual(cmd[cmd.index("--extra-ad-args") + 1],
                         "confidence=0.3")

    def test_the_negative_prompt_follows_the_same_rule_as_generation(self):
        """Sous CFG 1, sd.cpp ne calcule pas la branche non conditionnée."""
        self.assertNotIn("-n", self._cmd(negative="blurry"))


class RefusalTests(unittest.TestCase):
    """Ne pas pouvoir lancer est normal ; ne pas dire pourquoi ne l'est pas."""

    def test_an_engine_without_adetailer_says_to_update_it(self):
        with patch.object(settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()):
            reason = tools.adetailer_reason()
        self.assertIn("update-engine.bat", reason)

    def test_a_missing_binary_is_reported_first(self):
        with patch.object(settings, "find_sd_cli", return_value=None):
            self.assertIn("install.bat", tools.adetailer_reason())

    def test_a_recent_engine_without_detectors_points_at_the_button(self):
        with patch.object(settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--ad-model"})), \
             patch.object(tools, "adetailer_is_installed", return_value=False):
            self.assertIn("No detector installed", tools.adetailer_reason())

    def test_everything_ready_gives_no_reason_at_all(self):
        with patch.object(settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--ad-model"})), \
             patch.object(tools, "adetailer_is_installed", return_value=True):
            self.assertEqual(tools.adetailer_reason(), "")


class DetectorSetTests(unittest.TestCase):
    def test_only_yolov8_detection_models_are_offered(self):
        """sd.cpp refuse la segmentation et YOLOv9 : les proposer serait un
        échec garanti au premier clic, alors qu'ils existent en face."""
        import scripts.setup_adetailer as setup
        for source, _dest, _note in setup.DETECTORS:
            self.assertTrue(source.startswith(("face_yolov8", "hand_yolov8")),
                            source)
            self.assertNotIn("-seg", source)
            self.assertNotIn("yolov9", source)

    def test_the_ui_list_matches_what_the_installer_produces(self):
        import scripts.setup_adetailer as setup
        produced = {dest for _src, dest, _note in setup.DETECTORS}
        offered = {name for name, _label in tools.ADETAILER_MODELS}
        self.assertEqual(offered, produced)


if __name__ == "__main__":
    unittest.main()
