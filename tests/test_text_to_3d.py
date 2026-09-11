"""Texte → 3D : un ENCHAÎNEMENT, et c'est le sujet des tests.

TRELLIS ne lit que des images. « Texte → 3D » veut donc dire : fabriquer
l'image avec le modèle du catalogue, puis la lui donner — ce que la doc de
trellis.cpp appelle elle-même « driven end-to-end from a text prompt with
stable-diffusion.cpp producing the input image ».

Trois choses peuvent mal tourner, et aucune ne se voit à la lecture :

1. **le prompt.** Un prompt de belle photo donne un mauvais maillage. TRELLIS
   reconstruit UN objet : il détoure, traite en carré, et suppose que ce qu'il
   voit est le sujet entier. Cadrage serré → il invente ce qui dépasse ; ombre
   portée → géométrie ; décor → bruit. La réécriture n'est pas cosmétique ;
2. **la forme de l'image.** Non carrée, elle sera rognée ou déformée — autant
   la produire carrée plutôt que laisser TRELLIS choisir ce qu'il coupe ;
3. **la carte.** Le serveur 3D résident garde son modèle en VRAM. Générer une
   image par-dessus tombe sur une carte pleine, et la panne ne ressemble pas
   à sa cause.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from atelier.ui import threed_tab as T  # noqa: E402


class PromptRewriteTests(unittest.TestCase):
    def test_the_subject_survives_the_rewrite(self):
        out = T.studio_prompt("a weathered bronze dragon")
        self.assertIn("a weathered bronze dragon", out)

    def test_each_term_answers_a_known_failure(self):
        """Le gabarit n'est pas une liste de mots qui font joli.

        Chaque terme vise un mode d'échec précis de TRELLIS. Ce test est ce
        qui empêche de les remplacer un jour par « masterpiece, 8k, trending
        on artstation », qui ne veulent rien dire pour un reconstructeur 3D.
        """
        tpl = T.STUDIO_STYLE
        for need in ("one object only",       # il en reconstruit un seul
                     "entire object visible",  # sinon il invente le hors-champ
                     "plain flat neutral",     # le décor devient du bruit
                     "no cast shadow",         # l'ombre devient de la géométrie
                     "centred"):
            self.assertIn(need, tpl, need)

    def test_no_article_is_doubled(self):
        """Première version : « a single {subject} » + « a red dragon »
        donnait « a single a red dragon »."""
        out = T.studio_prompt("a red dragon")
        self.assertNotIn("a single a ", out)
        self.assertFalse(out.startswith("a single"))

    def test_an_empty_template_sends_the_prompt_untouched(self):
        """L'échappatoire : quelqu'un qui sait ce qu'il fait doit pouvoir
        court-circuiter la réécriture, et le champ le dit."""
        self.assertEqual(T.studio_prompt("  a red dragon  ", ""),
                         "a red dragon")

    def test_a_template_without_the_placeholder_still_works(self):
        """Un gabarit édité à la main peut perdre `{subject}`. On l'ajoute
        plutôt que de lever une KeyError au milieu d'une génération."""
        self.assertEqual(T.studio_prompt("a dragon", "low poly"),
                         "a dragon, low poly")


class WiringTests(unittest.TestCase):
    """Ce que fait le chemin texte, lu dans le source plutôt que deviné."""

    @staticmethod
    def _source(name: str) -> str:
        import inspect
        src = inspect.getsource(T.build_threed_tab)
        start = src.index(f"def {name}(")
        return src[start:]

    def test_the_image_is_generated_square(self):
        """TRELLIS traite son entrée en carré de toute façon : lui donner du
        16:9 revient à choisir nous-mêmes ce qui sera rogné."""
        src = self._source("_image_from_prompt")
        self.assertIn("width=1024, height=1024", src)

    def test_the_resident_3d_server_is_stopped_first(self):
        """Il garde son modèle en VRAM. Sans cet arrêt, la génération d'image
        tombe sur une carte pleine — et la panne accuse le mauvais coupable."""
        src = self._source("_image_from_prompt")
        self.assertIn("resident_stop()", src)
        self.assertLess(src.index("resident_stop()"),
                        src.index("gen_engine.generate("),
                        "l'arrêt doit précéder la génération")

    def test_only_downloaded_models_are_offered(self):
        """Proposer un modèle absent est un piège : on clique, ça échoue."""
        import inspect
        self.assertIn("model_is_ready", inspect.getsource(T._ready_models))

    def test_either_a_prompt_or_an_image_is_required(self):
        src = self._source("do_generate3d")
        self.assertIn("Describe an object, or load an image.", src)

    def test_the_generated_image_is_shown_before_the_mesh(self):
        """Le point de conception : l'intermédiaire est MONTRÉ.

        C'est là que ça rate, et un maillage coûte des minutes. L'image
        atterrit dans le champ image avant que trellis ne démarre, donc on
        peut relancer sans payer la suite.
        """
        src = self._source("do_generate3d")
        shown = src.index("Image ready — now the mesh.")
        mesh = src.index("Generating 3D (")
        self.assertLess(shown, mesh)

    def test_the_image_component_is_an_output(self):
        """Sans ça, l'image serait fabriquée et jamais affichée."""
        import inspect
        src = inspect.getsource(T.build_threed_tab)
        wiring = src[src.index("gen_evt = run.click("):]
        outputs = wiring[wiring.index("outputs=["):]
        self.assertIn("image", outputs[:outputs.index("]")])


if __name__ == "__main__":
    unittest.main()
