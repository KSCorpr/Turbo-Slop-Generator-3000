"""Choix du paquet trellis : quelle archive pour quelle carte.

Longtemps la réponse était « Vulkan, sauf RTX 30xx/50xx » : le CMakeLists amont
ÉCRASAIT la liste d'architectures de sa propre CI et ne compilait ses noyaux
que pour 86 et 120. v0.6.0 (19 août 2026) en a fait un simple défaut, et publie
deux archives CUDA — vérifié dans .github/workflows/release.yml au tag :

    cuda   (CUDA 13.1) -> 75;80;86;89;90;120   Turing et plus récent
    cuda12 (CUDA 12.9) -> 60;61;70             Pascal et Volta

Ces tests verrouillent la correspondance carte → archive, et surtout les deux
façons de se tromper : livrer « cuda » à une Pascal, ou « cuda12 » à une
Turing. Les deux se téléchargent, s'installent, et échouent au premier noyau.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import get_trellis  # noqa: E402


class BackendChoiceTests(unittest.TestCase):
    def test_turing_and_newer_get_the_cuda_package(self):
        for cap, card in (("7.5", "RTX 2080 Ti"), ("8.0", "A100"),
                          ("8.6", "RTX 3060"), ("8.9", "RTX 4070"),
                          ("9.0", "H100"), ("12.0", "RTX 50xx")):
            self.assertEqual(get_trellis.preferred_backend(cap), "cuda", card)

    def test_pascal_and_volta_get_the_legacy_package(self):
        for cap, card in (("6.0", "Tesla P100"), ("6.1", "GTX 1080 Ti"),
                          ("7.0", "Tesla V100")):
            self.assertEqual(get_trellis.preferred_backend(cap), "cuda12", card)

    def test_anything_else_gets_vulkan(self):
        # Sans information, on ne parie pas : Vulkan ne compile rien par
        # architecture, il marche partout.
        for unknown in ("", None, "   ", "bogus", "5.2", "10.0"):
            self.assertEqual(get_trellis.preferred_backend(unknown), "vulkan")


class AssetPickTests(unittest.TestCase):
    # Les noms réels de la release v0.6.0.
    ASSETS = [
        {"name": "trellis-cuda-linux-x64.tar.gz"},
        {"name": "trellis-cuda-windows-x64.zip"},
        {"name": "trellis-cuda12-linux-x64.tar.gz"},
        {"name": "trellis-cuda12-windows-x64.zip"},
        {"name": "trellis-rocm-windows-x64.zip"},
        {"name": "trellis-studio-windows-x64-portable.zip"},
        {"name": "trellis-vulkan-windows-x64.zip"},
        {"name": "trellis-vulkan-linux-x64.tar.gz"},
    ]

    def _pick(self, backend, assets=None):
        return get_trellis._pick_asset(assets or self.ASSETS, backend)["name"]

    def test_cuda_is_not_confused_with_cuda12(self):
        # « cuda » est un préfixe de « cuda12 » : un simple `in` donnerait
        # l'archive Pascal à une carte Turing.
        self.assertEqual(self._pick("cuda"), "trellis-cuda-windows-x64.zip")
        self.assertEqual(self._pick("cuda12"), "trellis-cuda12-windows-x64.zip")

    def test_it_never_picks_rocm_the_studio_app_or_linux(self):
        for backend in ("cuda", "cuda12", "vulkan"):
            name = self._pick(backend)
            self.assertNotIn("rocm", name)
            self.assertNotIn("studio", name)
            self.assertIn("windows", name)

    def test_the_only_fallback_is_vulkan(self):
        # Un job de CI peut échouer. Mais remplacer « cuda » par « cuda12 »
        # (ou l'inverse) livrerait un binaire compilé pour d'autres
        # architectures que celles de la carte.
        without_cuda = [a for a in self.ASSETS if "cuda-" not in a["name"]]
        self.assertEqual(get_trellis._pick_asset(without_cuda, "cuda")["name"],
                         "trellis-vulkan-windows-x64.zip")
        without_legacy = [a for a in self.ASSETS if "cuda12" not in a["name"]]
        self.assertEqual(get_trellis._pick_asset(without_legacy, "cuda12")["name"],
                         "trellis-vulkan-windows-x64.zip")

    def test_no_windows_asset_at_all(self):
        self.assertIsNone(get_trellis._pick_asset(
            [{"name": "trellis-vulkan-linux-x64.tar.gz"}], "vulkan"))

    def test_the_studio_app_is_never_mistaken_for_the_engine(self):
        only_studio = [{"name": "trellis-studio-windows-x64-portable.zip"}]
        self.assertIsNone(get_trellis._pick_asset(only_studio, "cuda"))


class DiagnosisTests(unittest.TestCase):
    def test_kernel_image_error_stays_actionable(self):
        """L'ancien message conseillait une mise à jour — qui ne pouvait rien.

        Depuis v0.6.0 elle le peut, justement : c'est le conseil à donner.
        """
        from atelier.engine import trellis
        trellis._CRASH.clear()
        trellis._CRASH.append(
            "CUDA error: no kernel image is available for execution on the device")
        msg = trellis._diagnose_crash(None)
        self.assertTrue("update-trellis" in msg.lower()
                        or "vulkan" in msg.lower(),
                        "le message n'indique aucune action")


if __name__ == "__main__":
    unittest.main()


class BackgroundRemovalTests(unittest.TestCase):
    """L'auto est une ABSENCE de drapeau, pas une valeur — et c'est piégeux.

    `trellis_args.h` déclare trois états : 1 BiRefNet, 0 seuil, **-1 auto**.
    Le parseur, lui, écrit :

        p.birefnet = (strcmp(v, "birefnet") == 0) ? 1 : 0;

    Aucune chaîne ne produit -1. Écrire « --bg-removal auto » donne donc le
    SEUIL — exactement l'inverse de ce qu'on demande, et le seuil est le mode
    que l'amont documente comme perceur de trous : il lit les hautes lumières
    spéculaires (min(RGB) ≥ 232) comme du fond, et le flow génère des trous
    là où il y avait un reflet.
    """

    def _args(self, **kw):
        from atelier.engine import trellis
        return trellis.build_server_args(512, **kw)

    def test_the_flag_is_never_sent_for_auto(self):
        """Il n'est d'ailleurs jamais sent AU LANCEMENT : le détourage est un
        réglage par requête. Ce test fixe surtout qu'on n'invente pas une
        valeur « auto » quelque part."""
        self.assertNotIn("--bg-removal", self._args())

    def test_auto_is_the_default_of_the_pipeline(self):
        import inspect
        from atelier.engine import trellis
        sig = inspect.signature(trellis.generate)
        self.assertEqual(sig.parameters["bg_removal"].default,
                         trellis.BG_AUTO)

    def test_the_request_omits_the_field_when_auto(self):
        """Côté serveur la règle est la même (`trellis-server.cpp:99`) :
        envoyer « auto » dans le champ donnerait le seuil. On l'omet."""
        import inspect
        from atelier.engine import trellis
        src = inspect.getsource(trellis._post_generate)
        self.assertIn("if bg_removal in (BG_BIREFNET, BG_THRESHOLD)", src)

    def test_the_ui_offers_auto_first_and_warns_about_the_threshold(self):
        import inspect
        from atelier.ui import threed_tab
        src = inspect.getsource(threed_tab.build_threed_tab)
        block = src[src.index("bg = gr.Dropdown("):]
        block = block[:block.index("label=\"Background removal\"")]
        self.assertLess(block.index("BG_AUTO"), block.index("BG_BIREFNET"),
                        "l'automatique doit être proposé en premier")
        self.assertIn("punches holes", block)


class GlbTextureTests(unittest.TestCase):
    """Les textures WebP passent par une EXTENSION glTF.

    `EXT_texture_webp` n'est pas lu par tous les visualiseurs ni par toutes
    les places de marché. Le défaut amont est WebP (fichier plus léger) ; on
    veut pouvoir retomber sur du PNG pour un GLB qui doit sortir d'ici.
    """

    def _args(self, **kw):
        from atelier.engine import trellis
        return trellis.build_server_args(512, **kw)

    def test_webp_stays_the_default_and_sends_nothing(self):
        self.assertNotIn("--webp", self._args())

    def test_asking_for_png_sends_the_flag(self):
        args = self._args(webp=False)
        self.assertIn("--webp", args)
        self.assertEqual(args[args.index("--webp") + 1], "off")


class DecimationTests(unittest.TestCase):
    """`--decim` est une GRILLE, pas un nombre de faces, et elle est legacy."""

    def test_zero_sends_nothing_so_the_engine_default_applies(self):
        """Le défaut du moteur (`decim = -1`) est une simplification quadrique
        à 150 000 faces en 512. Envoyer « --decim 0 » demanderait au contraire
        de GARDER le maillage brut — plusieurs millions de faces."""
        from atelier.engine import trellis
        self.assertNotIn("--decim", trellis.build_server_args(512, decim=0))

    def test_the_label_no_longer_promises_a_face_count(self):
        import inspect
        from atelier.ui import threed_tab
        src = inspect.getsource(threed_tab.build_threed_tab)
        block = src[src.index("decim = gr.Number("):]
        block = block[:block.index("atlas = gr.Dropdown(")]
        self.assertNotIn("target faces", block)
        self.assertIn("legacy", block)
