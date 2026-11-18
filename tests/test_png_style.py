"""Unit tests for the 3D Render & PNG Texture Studio workflow."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from atelier.engine import png_style as ps


class PngStyleEngineTests(unittest.TestCase):
    def test_closest_aspect_ratio(self):
        self.assertEqual(ps.get_closest_aspect_ratio(1024, 1024), "1:1")
        self.assertEqual(ps.get_closest_aspect_ratio(1920, 1080), "16:9")
        self.assertEqual(ps.get_closest_aspect_ratio(1080, 1920), "9:16")
        self.assertEqual(ps.get_closest_aspect_ratio(1600, 1200), "4:3")
        self.assertEqual(ps.get_closest_aspect_ratio(1200, 1600), "3:4")

    def test_analyze_alpha_detects_transparent_and_opaque_pngs(self):
        opaque = Image.new("RGBA", (128, 128), (180, 180, 180, 255))
        self.assertFalse(ps.analyze_alpha(opaque))

        transparent = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        for y in range(32, 96):
            for x in range(32, 96):
                transparent.putpixel((x, y), (200, 200, 200, 255))
        info = ps.analyze_image(transparent)
        self.assertTrue(info["has_alpha"])
        self.assertEqual((info["width"], info["height"]), (128, 128))
        self.assertEqual(info["aspect_ratio_str"], "1:1")

    def test_generation_size_for_model_preserves_aspect_ratio_and_alignment(self):
        w, h = ps.generation_size_for_model(1920, 1080, base_size=1024, multiple=32)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertGreater(w, h)
        self.assertAlmostEqual(w / h, 16 / 9, delta=0.08)

        w4m, h4m = ps.generation_size_for_model(1200, 1600, base_size=2048, multiple=32)
        self.assertEqual(w4m % 32, 0)
        self.assertEqual(h4m % 32, 0)
        self.assertLess(w4m, h4m)
        self.assertAlmostEqual(w4m / h4m, 3 / 4, delta=0.05)

    def test_prepare_reference_image_flattens_transparent_void_to_neutral_backdrop(self):
        # Raw 3D render with pitch-black RGB in transparent background
        src = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        src.putpixel((32, 32), (220, 210, 200, 255))

        ref_rgb = ps.prepare_reference_image(src, bg_color=(128, 128, 128), keep_rgba=False)
        self.assertEqual(ref_rgb.mode, "RGB")
        self.assertEqual(ref_rgb.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(ref_rgb.getpixel((32, 32)), (220, 210, 200))

        ref_rgba = ps.prepare_reference_image(src, keep_rgba=True)
        self.assertEqual(ref_rgba.mode, "RGBA")
        self.assertEqual(ref_rgba.getpixel((0, 0))[3], 0)

    def test_build_texture_prompt_adds_silhouette_coverage_and_model_tokens(self):
        p_flux = ps.build_texture_prompt("weathered bronze", model_family="flux2", lock_silhouette=True)
        self.assertIn("weathered bronze", p_flux)
        self.assertIn("complete full coverage artwork", p_flux)
        self.assertIn("edge-to-edge dense composition", p_flux)

        p_h3 = ps.build_texture_prompt("cyberpunk neon", model_family="minimax_h3_image", lock_silhouette=True)
        self.assertTrue(p_h3.startswith("In <Picture 1>, "))

        p_fibo = ps.build_texture_prompt("clay figurine", model_family="fibo2", lock_silhouette=True)
        self.assertTrue(p_fibo.startswith("Using <image_1>"))

    def test_composite_and_transfer_alpha_exact_source_dimensions_and_subpixel_alpha(self):
        # Source 3D render: 300x200 with transparent background (A=0),
        # anti-aliased edge (A=128), and opaque center (A=255).
        source = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        for y in range(50, 150):
            for x in range(50, 250):
                source.putpixel((x, y), (180, 180, 180, 255))
        source.putpixel((49, 100), (180, 180, 180, 128))

        # Generated texture at 600x400 with bright orange everywhere
        generated = Image.new("RGB", (600, 400), (255, 120, 40))

        out = ps.composite_and_transfer_alpha(
            generated,
            source,
            preserve_alpha=True,
            preserve_proportions=False,
        )
        self.assertEqual(out.size, (300, 200))
        self.assertEqual(out.mode, "RGBA")
        # Transparent pixel must have R=0, G=0, B=0, A=0
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 0, 0))
        # Anti-aliased edge pixel preserves exact alpha=128 and textured RGB
        edge_r, edge_g, edge_b, edge_a = out.getpixel((49, 100))
        self.assertEqual(edge_a, 128)
        self.assertGreater(edge_r, 200)
        # Opaque interior pixel has alpha=255 and generated texture color
        self.assertEqual(out.getpixel((150, 100)), (255, 120, 40, 255))

    def test_composite_and_transfer_alpha_native_resolution_exact_aspect_ratio(self):
        # Source is 500x250 (ratio 2.0), generated is 1024x576 (ratio 1.777)
        source = Image.new("RGBA", (500, 250), (200, 200, 200, 255))
        source.putpixel((0, 0), (0, 0, 0, 0))
        generated = Image.new("RGB", (1024, 576), (80, 160, 240))

        out = ps.composite_and_transfer_alpha(
            generated,
            source,
            preserve_alpha=True,
            preserve_proportions=True,
        )
        # gen_ratio (1.777) < target_ratio (2.0) -> final_w = 1024, final_h = round(1024 / 2.0) = 512
        self.assertEqual(out.size, (1024, 512))
        self.assertAlmostEqual(out.width / out.height, 2.0, places=4)
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(out.getpixel((512, 256))[3], 255)

    def test_model_choices_include_minimax_h3_image_and_editing_models(self):
        from atelier.ui import png_style_tab

        choices, _ = png_style_tab._model_choices()
        ids = {mid for _, mid in choices}
        self.assertTrue(
            {
                "minimax-h3-image",
                "flux2-klein-9b",
                "qwen-image-2.1-turbo",
                "fibo2-dev",
                "fibo2-edit-turbo",
            }.issubset(ids)
        )


if __name__ == "__main__":
    unittest.main()
