"""Le pavage ne doit RIEN changer à l'image, sinon il fabrique des coutures.

Ces tests utilisent un « modèle » exact (une réplication de pixels) : avec lui,
la sortie pavée et la sortie non pavée doivent être identiques au bit près. Un
vrai réseau n'est pas exact, mais tout écart mesuré ici serait un défaut de
NOTRE découpe, pas du modèle — et c'est précisément ce qu'on veut isoler.
"""
import unittest

import numpy as np

from atelier.engine import tiling


def _exact(scale: int):
    """Agrandissement exact par réplication : reproductible et sans invention."""
    def predict(block: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(block, scale, axis=0), scale, axis=1)
    return predict


def _image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


class PlanTests(unittest.TestCase):
    def test_an_image_smaller_than_a_tile_is_one_segment(self):
        self.assertEqual(tiling.plan_tiles(300, 512, 32), [(0, 300)])

    def test_segments_cover_everything_and_overlap(self):
        segments = tiling.plan_tiles(1000, 256, 32)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], 1000)
        for (_a0, a1), (b0, _b1) in zip(segments, segments[1:]):
            self.assertLess(b0, a1, "a gap between tiles would be a black band")

    def test_the_last_segment_is_flush_with_the_end(self):
        """Un dernier segment de 6 px ne donne rien d'exploitable au réseau."""
        segments = tiling.plan_tiles(1030, 512, 32)
        self.assertEqual(segments[-1], (518, 1030))
        self.assertTrue(all(b - a == 512 for a, b in segments))

    def test_a_tile_smaller_than_its_overlap_is_refused(self):
        with self.assertRaises(ValueError):
            tiling.plan_tiles(1000, 32, 64)


class SeamTests(unittest.TestCase):
    """Le cœur : pavé ou non, le résultat doit être le même."""

    def test_a_single_pass_reproduces_the_model_exactly(self):
        src = _image(64, 96)
        got = tiling.upscale_tiled(src, _exact(2), scale=2, tile=0)
        np.testing.assert_allclose(got, _exact(2)(src), atol=1e-6)

    def test_tiling_reproduces_the_untiled_result(self):
        """S'il y a un écart ici, c'est notre découpe qui l'a créé."""
        src = _image(200, 260, seed=3)
        untiled = tiling.upscale_tiled(src, _exact(2), scale=2, tile=0)
        tiled = tiling.upscale_tiled(src, _exact(2), scale=2,
                                     tile=64, overlap=16, halo=8)
        self.assertEqual(tiled.shape, untiled.shape)
        np.testing.assert_allclose(tiled, untiled, atol=1e-5)

    def test_no_column_is_left_unwritten(self):
        """Un trou de couverture passerait pour une bande noire dans l'image."""
        src = _image(150, 150, seed=7)
        out = tiling.upscale_tiled(src, _exact(4), scale=4,
                                   tile=64, overlap=16, halo=8)
        self.assertEqual(out.shape, (600, 600, 3))
        self.assertGreater(out.min(), 0.0)

    def test_the_halo_is_given_to_the_model_and_then_discarded(self):
        """Le réseau doit VOIR plus large que ce qu'on lui garde."""
        seen = []

        def spy(block):
            seen.append(block.shape[:2])
            return _exact(2)(block)

        tiling.upscale_tiled(_image(160, 160), spy, scale=2,
                             tile=64, overlap=16, halo=8)
        # La tuile centrale reçoit 64 + 8 de chaque côté.
        self.assertIn((80, 80), seen)


class ContractTests(unittest.TestCase):
    def test_a_wrong_output_size_is_refused_rather_than_resized(self):
        """Redimensionner pour rattraper masquerait un mauvais modèle."""
        with self.assertRaises(ValueError) as caught:
            tiling.upscale_tiled(_image(32, 32), _exact(2), scale=4, tile=0)
        self.assertIn("Nothing was resized", str(caught.exception))

    def test_non_finite_pixels_are_reported_not_written(self):
        """fp16 sur une architecture qui ne le supporte pas : NaN silencieux."""
        def broken(block):
            out = _exact(2)(block)
            out[0, 0, 0] = np.nan
            return out

        with self.assertRaises(ArithmeticError):
            tiling.upscale_tiled(_image(32, 32), broken, scale=2, tile=0)

    def test_a_non_rgb_input_is_refused(self):
        with self.assertRaises(ValueError):
            tiling.upscale_tiled(np.zeros((8, 8), np.float32), _exact(2),
                                 scale=2, tile=0)


class VramPlanTests(unittest.TestCase):
    def test_plenty_of_memory_means_no_tiling_at_all(self):
        self.assertEqual(tiling.tile_for_vram(1024, 1024, 4, free_gb=24.0), 0)

    def test_a_tight_card_gets_a_tile_it_can_hold(self):
        tile = tiling.tile_for_vram(4096, 4096, 4, free_gb=6.0)
        self.assertGreaterEqual(tile, 128)
        self.assertLessEqual(tile, 1024)
        self.assertEqual(tile % 64, 0, "tiles stay on a 64 px grid")

    def test_an_unknown_vram_falls_back_to_a_safe_tile(self):
        self.assertEqual(tiling.tile_for_vram(4096, 4096, 4, free_gb=None), 1024)


if __name__ == "__main__":
    unittest.main()
