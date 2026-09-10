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
        # L'encodeur de texte ne va PLUS sur la 1080 Ti : mesuré à 38 s par
        # image, parce que le fp16 d'une Pascal tourne à 1/64 de sa vitesse.
        # Le LLM d'amélioration de prompt, lui, y reste : il tourne seul.
        self.assertEqual(prefs["encoder_gpu_index"], 1)
        self.assertEqual(prefs["text_gpu_index"], 4)
        self.assertFalse(prefs["auto_fit"])
        self.assertEqual(prefs["split_mode"], "layer")
        # Mono-GPU : la carte choisie est remappée en cuda0. Les poids de
        # l'encodeur restent en RAM, son calcul se fait sur la carte.
        self.assertEqual(prefs["params_backend"],
                         "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertFalse(prefs["flags"]["offload_to_cpu"])

    def test_does_not_confuse_3060_ti_with_12gb_3060(self):
        cards = (
            hardware.Gpu(0, "NVIDIA GeForce RTX 3060 Ti", 12.0, "ampere", True),
            hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal", False),
        )
        with patch.object(hardware, "detect_gpus", return_value=cards):
            self.assertIsNone(hardware.rtx3060_1080ti_combo())


if __name__ == "__main__":
    unittest.main()


class MachinePresetTests(unittest.TestCase):
    """Deux tours connues, deux boutons — et rien qui mente sur les autres.

    Le profil automatique déduit tout de la VRAM et il a raison la plupart du
    temps. Ce qu'il ne peut pas savoir tient en deux faits mesurés : une Pascal
    ne doit jamais encoder le texte (fp16 à 1/64 de sa vitesse fp32), et une
    carte de 11 Go manque de place pour CALCULER une fois les poids posés.
    """

    HOME = (hardware.Gpu(0, "NVIDIA GeForce RTX 3060", 12.0, "ampere", True),
            hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal",
                         False))
    OFFICE = (hardware.Gpu(0, "NVIDIA GeForce RTX 2080 Ti", 11.0, "turing",
                           True),)

    def _keys(self, gpus):
        return [p.key for p, _cards in hardware.available_presets(gpus)]

    def test_each_machine_sees_only_its_own_profile(self):
        self.assertEqual(self._keys(self.HOME), ["home"])
        self.assertEqual(self._keys(self.OFFICE), ["office"])

    def test_an_unknown_machine_is_offered_nothing(self):
        """Un bouton « profil bureau » sur une machine qui n'a pas la carte
        est un bouton qui ment."""
        other = (hardware.Gpu(0, "NVIDIA GeForce RTX 4070", 12.0, "ada",
                              True),)
        self.assertEqual(self._keys(other), [])

    def test_a_3060_ti_is_not_a_3060_12gb(self):
        """Les deux noms se ressemblent, les profils mémoire non : 8 Go contre
        12. C'est la VRAM qui tranche, pas le nom."""
        pair = (hardware.Gpu(0, "NVIDIA GeForce RTX 3060 Ti", 8.0, "ampere",
                             True),
                hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal",
                             False))
        self.assertEqual(self._keys(pair), [])

    def test_the_office_profile_keeps_the_encoder_weights_in_ram(self):
        """11 Go doivent loger la diffusion ET de quoi calculer. Les 4 Go de
        l'encodeur y tiennent mal — et `--params-backend` décide de la
        résidence, pas du lieu d'exécution : le calcul reste sur la carte."""
        prefs = hardware.preset_prefs("office", self.OFFICE)
        self.assertEqual(prefs["params_backend"],
                         "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertFalse(prefs["flags"]["offload_to_cpu"],
                         "the explicit placement above must not be undone by "
                         "the blanket RAM shortcut")

    def test_the_office_profile_gives_the_engine_a_budget(self):
        """LE réglage qui distingue ce profil de l'automatique, et il vient
        d'une panne réelle : sans `--max-vram`, le moteur découpe son graphe
        sans cible et découvre au troisième segment qu'il ne reste que 622 Mo
        pour un besoin de 1049."""
        self.assertEqual(hardware.preset_prefs("office", self.OFFICE)["max_vram"],
                         "auto")

    def test_the_home_profile_never_encodes_on_the_pascal(self):
        prefs = hardware.preset_prefs("home", self.HOME)
        self.assertEqual(prefs["encoder_gpu_index"], 0)
        self.assertEqual(prefs["text_gpu_index"], 1,
                         "the prompt improver runs alone, so the Pascal is "
                         "free real estate for it")

    def test_asking_for_a_profile_this_machine_cannot_run_says_why(self):
        with self.assertRaises(ValueError) as caught:
            hardware.preset_prefs("office", self.HOME)
        self.assertIn("cards this machine does not have", str(caught.exception))

    def test_an_unknown_key_is_told_apart_from_a_missing_card(self):
        with self.assertRaises(ValueError) as caught:
            hardware.preset_prefs("laptop", self.HOME)
        self.assertIn("Unknown", str(caught.exception))

    def test_a_card_is_never_counted_twice(self):
        """Sur une machine à deux 1080 Ti, un profil qui en demande deux doit
        en trouver deux — pas compter la même fois deux."""
        twins = (hardware.Gpu(0, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal",
                              False),)
        self.assertIsNone(hardware._match_cards(
            ((r"GTX\s*1080\s*TI", 10.0), (r"GTX\s*1080\s*TI", 10.0)), twins))
