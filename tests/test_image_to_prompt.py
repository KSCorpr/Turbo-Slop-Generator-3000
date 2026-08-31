"""« Image → prompt » : ce qui distingue un prompt d'une légende.

Le modèle n'est pas testable ici (7,5 Go de poids, un GPU, et une sortie non
déterministe). Ce qui EST testable, et qui décide de la qualité du résultat,
c'est tout ce qu'il y a autour : les consignes envoyées au modèle, le nettoyage
de sa réponse, et le câblage qui ramène le texte dans le champ Prompt.

Ces trois choses ont chacune un mode de panne silencieux :
- une consigne qui n'interdit pas les formules de légende → « This image shows
  a cat », collé tel quel, donne une image plate ;
- un nettoyage qui laisse passer l'amorce → même résultat, malgré la consigne ;
- un câblage qui n'écrit nulle part → le bouton semble marcher et ne fait rien
  (c'est arrivé : la première version passait par un State consommé au
  changement d'onglet, et une sélection programmatique ne déclenche pas
  `Tabs.select`).
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

ROOT = Path(__file__).resolve().parent.parent


def _runner():
    """Charge run_describe.py SANS importer torch (absent des tests)."""
    spec = importlib.util.spec_from_file_location(
        "run_describe", ROOT / "scripts" / "tools" / "run_describe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # les imports lourds sont dans main()
    return mod


class CleaningTests(unittest.TestCase):
    """La consigne réduit les amorces de légende ; elle ne les supprime pas."""

    def setUp(self):
        self.run = _runner()

    def test_caption_lead_ins_are_stripped(self):
        cases = {
            "This image shows a red fox in the snow":
                "A red fox in the snow",
            "The picture depicts an old man reading":
                "An old man reading",
            "Here is a prompt: cinematic portrait, soft light":
                "Cinematic portrait, soft light",
            "Sure, a moody landscape at dusk": "A moody landscape at dusk",
        }
        for raw, expected in cases.items():
            self.assertEqual(self.run._clean(raw), expected, raw)

    def test_stacked_lead_ins_are_all_removed(self):
        """Les modèles en empilent deux : « Sure, here is a prompt: … »."""
        got = self.run._clean("Sure, here is a prompt: golden hour photograph")
        self.assertEqual(got, "Golden hour photograph")

    def test_code_fences_and_quotes_go_away(self):
        self.assertEqual(
            self.run._clean('```\n"oil painting, thick impasto"\n```'),
            "Oil painting, thick impasto")

    def test_a_legitimate_prompt_is_left_alone(self):
        """Le nettoyage ne doit pas mordre sur un texte déjà correct."""
        good = ("Cinematic photograph of a lighthouse, 85mm lens, shallow "
                "depth of field, cold blue dusk light")
        self.assertEqual(self.run._clean(good), good)

    def test_empty_stays_empty(self):
        self.assertEqual(self.run._clean(""), "")
        self.assertEqual(self.run._clean("   "), "")


class ModePromptTests(unittest.TestCase):
    """Les trois modes doivent demander trois choses DIFFÉRENTES."""

    def setUp(self):
        self.run = _runner()

    def test_the_three_modes_exist_and_differ(self):
        self.assertEqual(set(self.run.MODES), {"full", "style", "plain"})
        self.assertEqual(len(set(self.run.MODES.values())), 3)

    def test_english_is_demanded_everywhere(self):
        """L'interface est en français ; les modèles, non. Sans cette consigne
        le VLM répond dans la langue de la question."""
        for name, text in self.run.MODES.items():
            self.assertIn("ENGLISH", text, name)

    def test_caption_formulas_are_forbidden_everywhere(self):
        for name, text in self.run.MODES.items():
            self.assertIn("NEVER start with caption formulas", text, name)

    def test_style_mode_forbids_naming_the_subject(self):
        """C'est TOUT l'intérêt du mode : un style réutilisable ailleurs. S'il
        nomme le sujet, il n'est pas transposable et le mode ne sert à rien."""
        style = self.run.MODES["style"]
        self.assertIn("Say NOTHING about what the picture is of", style)
        self.assertIn("ONLY THE STYLE", style)

    def test_full_mode_asks_for_what_actually_steers_diffusion(self):
        full = self.run.MODES["full"]
        for axis in ("LIGHT", "COMPOSITION", "COLOURS", "MEDIUM"):
            self.assertIn(axis, full, axis)

    def test_plain_mode_does_not_ask_for_prompt_jargon(self):
        plain = self.run.MODES["plain"]
        self.assertIn("No prompt vocabulary", plain)

    def test_medium_coherence_is_required(self):
        """Mélanger « oil painting » et « 85mm lens » brouille le rendu : c'est
        la faute la plus commune d'un prompt écrit par une machine."""
        for name, text in self.run.MODES.items():
            self.assertIn("ONLY that medium's vocabulary", text, name)


class WiringTests(unittest.TestCase):
    """Le bouton « → Krea 2 » doit VRAIMENT écrire dans le champ Prompt."""

    def test_generation_tabs_return_their_prompt_box(self):
        """C'est ce que l'application collecte pour câbler l'envoi. Si un jour
        `build_generative_tab` cesse de le renvoyer, le bouton d'envoi
        disparaîtrait en silence au lieu d'échouer."""
        import gradio as gr

        import app
        demo = app.build_app()
        boxes = [b for b in demo.blocks.values()
                 if isinstance(b, gr.Textbox) and b.label == "Prompt"]
        self.assertGreaterEqual(len(boxes), 2)

    def test_the_send_buttons_exist_and_are_visible(self):
        import gradio as gr

        import app
        demo = app.build_app()
        labels = [(b.value or "") for b in demo.blocks.values()
                  if isinstance(b, gr.Button) and getattr(b, "visible", True)]
        self.assertTrue(any("Krea 2 Turbo" in v and v.startswith("→")
                            for v in labels), labels[:5])
        self.assertTrue(any("Flux.2 Klein" in v and v.startswith("→")
                            for v in labels))

    def test_the_tool_is_reachable_from_the_toolkit(self):
        import gradio as gr

        import app
        demo = app.build_app()
        ids = [getattr(b, "id", None) for b in demo.blocks.values()
               if isinstance(b, gr.Tab)]
        self.assertIn("describe", ids)
        # Les onglets de génération ont besoin d'un id STABLE : l'envoi les
        # vise par identifiant, pas par libellé (qui change avec la langue).
        self.assertIn("krea2-turbo", ids)
        self.assertIn("flux2-klein-9b", ids)


class EngineLayerTests(unittest.TestCase):
    def test_modes_agree_between_the_app_and_the_runner(self):
        """Deux listes de modes qui divergent = un mode qui échoue à l'appel,
        avec un message d'argparse que personne ne comprendra."""
        from atelier.engine import tools
        self.assertEqual(set(tools.DESCRIBE_MODES), set(_runner().MODES))

    def test_it_refuses_politely_when_not_installed(self):
        from atelier.engine import tools
        if tools.describe_is_installed():
            self.skipTest("modèle installé sur cette machine")
        with self.assertRaises(tools.ToolError) as ctx:
            tools.image_to_prompt(ROOT / "README.md")
        self.assertIn("pas installé", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
