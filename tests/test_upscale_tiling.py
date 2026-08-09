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


class HiresArgsTests(unittest.TestCase):
    """Passe HD native (`--hires`) : ce qu'on envoie réellement à sd-cli."""

    def test_builtin_upscaler_needs_no_directory(self):
        args = sdcpp.hires_args(sdcpp.HiresParams(
            upscaler="Latent", upscalers_dir=Path("/ups"), tile_size=832))
        self.assertIn("--hires", args)
        self.assertNotIn("--hires-upscalers-dir", args)
        # La tuile ne concerne que les agrandisseurs à MODÈLE.
        self.assertNotIn("--hires-upscale-tile-size", args)

    def test_model_upscaler_gets_directory_and_tile(self):
        args = sdcpp.hires_args(sdcpp.HiresParams(
            upscaler="4x_anime.gguf", upscalers_dir=Path("/ups"),
            tile_size=832))
        self.assertEqual(args[args.index("--hires-upscaler") + 1],
                         "4x_anime.gguf")
        self.assertEqual(args[args.index("--hires-upscalers-dir") + 1], "/ups")
        self.assertEqual(args[args.index("--hires-upscale-tile-size") + 1],
                         "832")

    def test_explicit_target_wins_over_scale(self):
        # Une taille explicite garantit que ce qu'on annonce est ce qu'on demande.
        args = sdcpp.hires_args(sdcpp.HiresParams(
            scale=2.0, target_width=2304, target_height=1792))
        self.assertNotIn("--hires-scale", args)
        self.assertEqual(args[args.index("--hires-width") + 1], "2304")
        self.assertEqual(args[args.index("--hires-height") + 1], "1792")

    def test_scale_used_when_no_target(self):
        args = sdcpp.hires_args(sdcpp.HiresParams(scale=2.5))
        self.assertEqual(args[args.index("--hires-scale") + 1], "2.5")
        self.assertNotIn("--hires-width", args)

    def test_gen_cmd_omits_hires_on_an_older_binary(self):
        req = sdcpp.GenRequest(diffusion_model=Path("d.gguf"),
                               hires=sdcpp.HiresParams())
        with patch.object(sdcpp, "_require", lambda *a, **k: None), \
             patch.object(sdcpp, "supported_options", return_value=frozenset()):
            cmd = sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("o.png"))
        self.assertNotIn("--hires", cmd)


class OomDiagnosisTests(unittest.TestCase):
    """Un OOM doit être RECONNU comme tel : c'est le seul échec qu'on retente."""

    REAL_LOG = [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4731.82 MiB "
        "on device 0: cudaMalloc failed: out of memory",
        "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size "
        "4961670528",
        "krea2: failed to allocate the compute buffer",
        "hires sampling for image 1/1 failed after 1.63s",
    ]

    def test_recognised_and_typed(self):
        from collections import deque
        err = sdcpp._failure_error(1, ["sd-cli", "--hires"],
                                   deque(self.REAL_LOG))
        self.assertIsInstance(err, sdcpp.VramError)
        # La taille manquante est citée : c'est ce qui rend le message utile.
        self.assertIn("4.6 Go", str(err))

    def test_hires_gets_its_own_advice(self):
        from collections import deque
        hires = str(sdcpp._failure_error(1, ["sd-cli", "--hires"],
                                        deque(self.REAL_LOG)))
        plain = str(sdcpp._failure_error(1, ["sd-cli"], deque(self.REAL_LOG)))
        self.assertIn("facteur d'agrandissement", hires)
        self.assertIn("quantification", plain)

    def test_other_failures_stay_plain_engine_errors(self):
        from collections import deque
        err = sdcpp._failure_error(1, ["sd-cli"], deque(["something else"]))
        self.assertIsInstance(err, sdcpp.EngineError)
        self.assertNotIsInstance(err, sdcpp.VramError)


class HdBudgetTests(unittest.TestCase):
    """Le budget en pixels de la passe HD : VRAM moins les poids du modèle."""

    def _model(self, weights_gb):
        from atelier.engine import generate as gen
        m = object()
        self._patch = patch.object(
            gen, "_weights_bytes",
            lambda _m, gb=weights_gb: int(gb * 1024 ** 3))
        return m

    def test_lighter_weights_buy_more_pixels(self):
        from atelier.engine import generate as gen
        m = self._model(8.2)
        with self._patch:
            tight = gen.hd_pixel_budget(m, 12.0)
        self._model(6.3)
        with self._patch:
            roomy = gen.hd_pixel_budget(m, 12.0)
        self.assertGreater(roomy, tight)

    def test_unknown_vram_means_no_cap(self):
        from atelier.engine import generate as gen
        self.assertEqual(gen.hd_pixel_budget(object(), None), 0)

    def test_weights_bigger_than_vram_means_no_cap(self):
        # Rien à budgéter : on laisse la reprise automatique trancher plutôt
        # que d'inventer une taille.
        from atelier.engine import generate as gen
        m = self._model(20.0)
        with self._patch:
            self.assertEqual(gen.hd_pixel_budget(m, 11.0), 0)


class HdAlignTests(unittest.TestCase):
    def test_aligns_up_never_down(self):
        from atelier.engine import generate as gen
        self.assertEqual(gen._align_up(1000), 1008)
        self.assertEqual(gen._align_up(1152), 1152)
        # Les tailles produites par l'application ne bougent pas.
        for v in (1184, 880, 1152, 896, 1248, 832, 752, 1024):
            self.assertEqual(gen._align_up(v), v, v)

    def test_never_returns_zero(self):
        from atelier.engine import generate as gen
        self.assertEqual(gen._align_up(1), gen.HD_ALIGN)


if __name__ == "__main__":
    unittest.main()
