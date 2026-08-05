import unittest
from unittest.mock import patch

from atelier import hardware


class Rtx3060ComboTests(unittest.TestCase):
    def test_detects_3060_12gb_and_1080ti_in_any_order(self):
        cards = (
            hardware.Gpu(4, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal", False),
            hardware.Gpu(1, "NVIDIA GeForce RTX 3060", 12.0, "ampere", True),
        )
        with patch.object(hardware, "detect_gpus", return_value=cards):
            main, secondary = hardware.rtx3060_1080ti_combo()
            self.assertEqual(main.index, 1)
            self.assertEqual(secondary.index, 4)
            prefs = hardware.rtx3060_1080ti_prefs()

        self.assertEqual(prefs["gpu_index"], 1)
        self.assertEqual(prefs["encoder_gpu_index"], 4)
        self.assertEqual(prefs["text_gpu_index"], 4)
        self.assertFalse(prefs["auto_fit"])
        self.assertEqual(prefs["split_mode"], "layer")

    def test_does_not_confuse_3060_ti_with_12gb_3060(self):
        cards = (
            hardware.Gpu(0, "NVIDIA GeForce RTX 3060 Ti", 12.0, "ampere", True),
            hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal", False),
        )
        with patch.object(hardware, "detect_gpus", return_value=cards):
            self.assertIsNone(hardware.rtx3060_1080ti_combo())


if __name__ == "__main__":
    unittest.main()
