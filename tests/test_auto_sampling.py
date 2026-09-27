"""The generation tab applies each model's catalog recipe at click time."""
import unittest

from atelier import registry, settings
from atelier.ui.generate_tab import _sampling_values


class AutoSamplingTests(unittest.TestCase):
    def test_every_model_uses_its_own_defaults_even_with_stale_sliders(self):
        for model in registry.load_base_models(settings.DEFAULT_PREFS):
            with self.subTest(model=model.id):
                recipe = _sampling_values(
                    model.id, "auto", 1, 0.0, "heun", "karras", 10.0)
                self.assertEqual(recipe, (
                    model.defaults["steps"], model.defaults["cfg_scale"],
                    model.defaults["sampler"], model.defaults["scheduler"],
                    model.defaults.get("flow_shift", 0.0)))

    def test_custom_sampling_keeps_manual_values(self):
        self.assertEqual(
            _sampling_values("qwen-image-2.1", "custom", 3, 4.5,
                             "euler", "karras", 2.0),
            (3, 4.5, "euler", "karras", 2.0))


if __name__ == "__main__":
    unittest.main()
