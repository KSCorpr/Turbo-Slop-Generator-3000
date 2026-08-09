"""Tuilage de l'upscaler ESRGAN (sd.cpp).

Le défaut de sd.cpp est une tuile de 128 px : le réseau RRDB ne voit alors
presque aucun contexte et prend des décisions d'accentuation différentes d'une
tuile à l'autre — d'où les coutures, le grain qui change de carré en carré et
l'aliasing sur les diagonales. Ces tests verrouillent les deux garde-fous :
la tuile choisie, et le fait qu'on n'envoie jamais une option que le binaire
installé ne connaît pas.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier.engine import sdcpp


class UpscaleTileSizeTests(unittest.TestCase):
    def test_whole_image_when_it_fits_under_the_vram_cap(self):
        # 1024x768 sur 24 Go : la tuile couvre l'image -> sd.cpp ne découpe pas.
        tile = sdcpp.upscale_tile_size(1024, 768, 24.0)
        self.assertEqual(tile, 1024)
        self.assertGreaterEqual(tile, 1024)
        self.assertGreaterEqual(tile, 768)

    def test_capped_by_vram_when_image_is_larger(self):
        self.assertEqual(sdcpp.upscale_tile_size(4096, 4096, 24.0), 1024)
        self.assertEqual(sdcpp.upscale_tile_size(2048, 1536, 11.0), 832)
        self.assertEqual(sdcpp.upscale_tile_size(2048, 1536, 8.0), 640)

    def test_unknown_vram_falls_back_to_the_prudent_cap(self):
        self.assertEqual(sdcpp.upscale_tile_size(4096, 4096, None), 512)

    def test_always_beats_the_sdcpp_default(self):
        # Quel que soit le cas, on ne fait jamais PIRE que les 128 px d'origine.
        for w, h, vram in ((640, 480, None), (8000, 6000, 8.0),
                           (1152, 896, 11.0), (16, 16, 24.0)):
            self.assertGreaterEqual(
                sdcpp.upscale_tile_size(w, h, vram),
                min(sdcpp.SDCPP_DEFAULT_UPSCALE_TILE, max(w, h)),
                f"{w}x{h} / {vram} Go")


class BuildUpscaleCmdTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.object(sdcpp, "_require", lambda *a, **k: None)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.args = (Path("sd-cli"), Path("in.png"), Path("model.gguf"),
                     Path("out.png"))

    def test_tile_option_sent_when_the_binary_knows_it(self):
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--upscale-tile-size"})):
            cmd = sdcpp.build_upscale_cmd(*self.args, tile_size=832)
        self.assertIn("--upscale-tile-size", cmd)
        self.assertEqual(cmd[cmd.index("--upscale-tile-size") + 1], "832")

    def test_tile_option_omitted_on_an_older_binary(self):
        # Un sd-cli plus ancien s'arrête sur un argument inconnu : mieux vaut
        # des coutures qu'une erreur illisible à la place de l'image.
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--upscale-model"})):
            cmd = sdcpp.build_upscale_cmd(*self.args, tile_size=832)
        self.assertNotIn("--upscale-tile-size", cmd)

    def test_no_tile_option_when_disabled(self):
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--upscale-tile-size"})):
            cmd = sdcpp.build_upscale_cmd(*self.args, tile_size=0)
        self.assertNotIn("--upscale-tile-size", cmd)

    def test_repeats_still_works(self):
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset()):
            cmd = sdcpp.build_upscale_cmd(*self.args, repeats=2)
        self.assertEqual(cmd[cmd.index("--upscale-repeats") + 1], "2")


if __name__ == "__main__":
    unittest.main()
