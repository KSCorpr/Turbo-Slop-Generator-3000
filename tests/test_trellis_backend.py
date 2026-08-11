"""Choix du backend trellis : CUDA n'est PAS le bon défaut.

Le `CMakeLists.txt` amont épingle les noyaux CUDA de trellis à deux
architectures — `set_target_properties(trellis_core PROPERTIES
CUDA_ARCHITECTURES "86;120")` — ce qui écrase la liste complète passée par sa
propre CI (`75;80;86;89;90;120`). Résultat : seules les RTX 30xx et RTX 50xx
reçoivent du code machine. Ces tests verrouillent le fait qu'on n'installe la
build CUDA que sur ces cartes-là, et Vulkan partout ailleurs.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import get_trellis  # noqa: E402


class BackendChoiceTests(unittest.TestCase):
    def test_covered_cards_get_cuda(self):
        self.assertEqual(get_trellis.preferred_backend("8.6"), "cuda")   # RTX 30xx
        self.assertEqual(get_trellis.preferred_backend("12.0"), "cuda")  # RTX 50xx

    def test_uncovered_nvidia_cards_get_vulkan(self):
        # Les cartes que la build CUDA amont laisse sur le carreau, alors même
        # que la CI croit les compiler.
        for cap, card in (("7.5", "RTX 2080 Ti"), ("8.9", "RTX 4070"),
                          ("8.0", "A100"), ("9.0", "H100")):
            self.assertEqual(get_trellis.preferred_backend(cap), "vulkan", card)

    def test_unknown_card_falls_back_to_vulkan(self):
        # Mieux vaut le backend qui marche partout que celui qui marche sur
        # deux modèles : sans information, on ne parie pas.
        for unknown in ("", None, "   ", "bogus"):
            self.assertEqual(get_trellis.preferred_backend(unknown), "vulkan")


class AssetPickTests(unittest.TestCase):
    ASSETS = [
        {"name": "trellis-cuda-windows-x64.zip"},
        {"name": "trellis-vulkan-windows-x64.zip"},
        {"name": "trellis-rocm-windows-x64.zip"},
        {"name": "trellis-vulkan-linux-x64.zip"},
    ]

    def test_picks_the_requested_backend(self):
        self.assertEqual(get_trellis._pick_asset(self.ASSETS, "vulkan")["name"],
                         "trellis-vulkan-windows-x64.zip")
        self.assertEqual(get_trellis._pick_asset(self.ASSETS, "cuda")["name"],
                         "trellis-cuda-windows-x64.zip")

    def test_never_picks_rocm_or_linux(self):
        for backend in ("cuda", "vulkan"):
            name = get_trellis._pick_asset(self.ASSETS, backend)["name"]
            self.assertNotIn("rocm", name)
            self.assertIn("windows", name)

    def test_falls_back_when_the_wanted_backend_is_missing(self):
        # Le job CUDA de la CI est marqué « experimental » : une release où il
        # a échoué ne doit pas bloquer l'installation.
        only_vulkan = [{"name": "trellis-vulkan-windows-x64.zip"}]
        self.assertEqual(get_trellis._pick_asset(only_vulkan, "cuda")["name"],
                         "trellis-vulkan-windows-x64.zip")

    def test_no_windows_asset_at_all(self):
        self.assertIsNone(get_trellis._pick_asset(
            [{"name": "trellis-vulkan-linux-x64.zip"}], "vulkan"))


class DiagnosisTests(unittest.TestCase):
    def test_kernel_image_error_points_at_vulkan(self):
        """L'ancien message conseillait une mise à jour — qui ne peut rien y faire."""
        from atelier.engine import trellis
        trellis._CRASH.clear()
        trellis._CRASH.append(
            "CUDA error: no kernel image is available for execution on the device")
        msg = trellis._diagnose_crash(None)
        self.assertIn("VULKAN", msg.upper())
        self.assertIn("86;120", msg)


if __name__ == "__main__":
    unittest.main()
