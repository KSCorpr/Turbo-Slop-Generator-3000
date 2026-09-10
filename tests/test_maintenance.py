"""La maintenance ne doit pas supprimer du code vivant.

Ce test existe à cause d'un incident précis : `atelier/engine/sdserver.py`
était déclaré dans REMOVED_FEATURES (ancien backend serveur, retiré), un
nouveau module du même nom est arrivé des mois plus tard, et chaque passage de
la maintenance l'effaçait. L'application ne démarrait plus, avec un ImportError
que rien ne rattachait à la maintenance.

Une table de noms de fichiers ne peut pas savoir qu'un nom a été repris. Le
code actuel, lui, le sait.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "maintenance.py"
_SPEC = importlib.util.spec_from_file_location("maintenance_test", _PATH)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(M)


def _fake_project(root: Path, engine_init: str) -> None:
    """Un projet minuscule mais de la même FORME que le vrai."""
    (root / "atelier" / "engine").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "app.py").write_text("from atelier import engine\n", encoding="utf-8")
    (root / "atelier" / "__init__.py").write_text("", encoding="utf-8")
    (root / "atelier" / "engine" / "__init__.py").write_text(
        engine_init, encoding="utf-8")
    (root / "atelier" / "engine" / "sdserver.py").write_text(
        "SERVER = 1\n", encoding="utf-8")


class RemovedFeatureGuardTests(unittest.TestCase):
    def setUp(self):
        M._REACHABLE = None
        self._problems = M._problems

    def tearDown(self):
        M._REACHABLE = None
        M._problems = self._problems

    def _clean(self, root, engine_init):
        _fake_project(root, engine_init)
        table = [{"name": "Ancien backend serveur",
                  "files": ["atelier/engine/sdserver.py"], "dirs": []}]
        with patch.object(M, "ROOT", root), \
             patch.object(M, "REMOVED_FEATURES", table):
            M.clean_removed_features(purge=False)
        return (root / "atelier" / "engine" / "sdserver.py").exists()

    def test_a_module_the_code_imports_survives_the_table(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # L'import est PARESSEUX, à l'intérieur d'une fonction : c'est
            # exactement la forme qu'a le vrai code, et celle qu'une analyse
            # trop naïve rate.
            init = ("def resident_engine():\n"
                    "    from . import sdserver\n"
                    "    return sdserver\n")
            self.assertTrue(self._clean(Path(tmp), init),
                            "un module importé a été supprimé")

    def test_a_module_nobody_imports_is_still_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._clean(Path(tmp), "# rien\n"),
                             "le nettoyage ne fait plus son travail")


class RelativeImportTests(unittest.TestCase):
    """« from . import x » dans un __init__.py désigne le paquet lui-même."""

    def setUp(self):
        M._REACHABLE = None

    def tearDown(self):
        M._REACHABLE = None

    def test_a_package_init_resolves_its_own_package(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_project(root, "def f():\n    from . import sdserver\n")
            with patch.object(M, "ROOT", root):
                found = M._imports_of(root / "atelier" / "engine" / "__init__.py")
        self.assertIn("atelier.engine.sdserver", found)
        # Le paquet du dessus n'a rien à voir : c'était le bug.
        self.assertNotIn("atelier.sdserver", found)

    def test_a_plain_module_resolves_its_parent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_project(root, "")
            (root / "atelier" / "engine" / "generate.py").write_text(
                "from . import sdcpp\n", encoding="utf-8")
            with patch.object(M, "ROOT", root):
                found = M._imports_of(root / "atelier" / "engine" / "generate.py")
        self.assertIn("atelier.engine.sdcpp", found)


class RealProjectTests(unittest.TestCase):
    """Sur le vrai dépôt, aucune entrée de la table ne vise du code vivant."""

    def setUp(self):
        M._REACHABLE = None

    def tearDown(self):
        M._REACHABLE = None

    def test_no_declared_removal_targets_a_living_module(self):
        living = [rel for feat in M.REMOVED_FEATURES for rel in feat["files"]
                  if M._still_in_service(rel)]
        self.assertEqual(living, [], "REMOVED_FEATURES vise du code vivant")

    def test_every_engine_module_is_reachable(self):
        """Aucun module du paquet moteur ne doit passer pour orphelin.

        C'est le garde-fou qui manquait le jour où `atelier/engine/sdserver.py`
        — importé paresseusement depuis un `__init__.py` — a été jugé mort puis
        supprimé par la maintenance. Le module n'existe plus, le piège si.
        """
        reachable = M._reachable_modules()
        engine = sorted(
            "atelier.engine." + p.stem
            for p in (M.ROOT / "atelier" / "engine").glob("*.py")
            if p.stem != "__init__")
        missing = [m for m in engine if m not in reachable]
        self.assertEqual(missing, [], f"modules jugés orphelins : {missing}")


if __name__ == "__main__":
    unittest.main()


class MenuTests(unittest.TestCase):
    """Ce qui décide qu'on supprime — et si on le demande d'abord.

    Le menu et `--ask-purge` mènent au même endroit par deux chemins : l'un
    est un choix tapé devant l'écran, l'autre est `update.bat` qui enchaîne.
    Dans les deux cas plusieurs gigaoctets peuvent partir, dont certains ne se
    retéléchargent qu'à travers une acceptation de licence. Ce sont ces tests
    qui gardent la porte, pas la relecture du code.
    """

    def test_a_typed_purge_deletes_straight_away(self):
        """Celui qui a écrit `--purge` a déjà décidé : on ne redemande pas."""
        purge, engine, models, deferred = M.parse_args(
            ["maintenance.py", "--purge"])
        self.assertEqual((purge, models, deferred), (True, True, False))
        self.assertFalse(engine)

    def test_all_does_both_without_asking(self):
        purge, engine, _, deferred = M.parse_args(["maintenance.py", "--all"])
        self.assertEqual((purge, engine, deferred), (True, True, False))

    def test_ask_purge_measures_first_and_deletes_nothing_yet(self):
        """La première passe ne doit RIEN supprimer : elle chiffre.

        C'est ce qui rend le total affichable avant la question. Un `purge`
        laissé à True ici, et le message « voici ce qu'on peut libérer »
        arriverait après la suppression.
        """
        purge, engine, models, deferred = M.parse_args(
            ["maintenance.py", "--update-engine", "--ask-purge"])
        self.assertTrue(deferred)
        self.assertTrue(engine)
        self.assertFalse(purge, "la passe de mesure supprime déjà")
        self.assertFalse(models, "la passe de mesure supprime déjà")

    def test_the_menu_is_never_shown_when_arguments_were_given(self):
        """Un appel automatisé ne doit jamais rester bloqué sur une question."""
        def boom():
            raise AssertionError("le menu a été affiché malgré un argument")
        M.parse_args(["maintenance.py", "--purge"], ask=boom)

    def test_the_menu_is_not_shown_without_a_terminal(self):
        """`ask=None` est ce que passe `main` quand stdin n'est pas un tty."""
        self.assertEqual(M.parse_args(["maintenance.py"], ask=None),
                         (False, False, False, False))

    def test_the_menu_choices_mean_what_the_menu_says(self):
        for answer, expected in (("1", (False, True)), ("2", (True, False)),
                                 ("3", (True, True))):
            self.assertEqual(M.ask_choice(read=lambda: answer), expected,
                             f"choix {answer}")

    def test_anything_unexpected_falls_back_to_just_checking(self):
        """Taper au hasard, ou fermer la fenêtre, ne doit rien effacer."""
        for answer in ("", "oui", "42", "y"):
            self.assertEqual(M.ask_choice(read=lambda: answer), (False, False),
                             f"« {answer} » n'est pas neutre")

    def test_closing_the_window_on_the_menu_is_a_no(self):
        def eof():
            raise EOFError
        self.assertEqual(M.ask_choice(read=eof), (False, False))

    def test_the_confirmation_defaults_to_no(self):
        self.assertFalse(M.confirm_purge(5_000_000_000, read=lambda p: ""))

    def test_the_confirmation_accepts_both_languages(self):
        for answer in ("y", "yes", "o", "oui", "OUI"):
            self.assertTrue(M.confirm_purge(5_000_000_000,
                                            read=lambda p: answer), answer)

    def test_no_terminal_answers_no_rather_than_crashing(self):
        """Sortie redirigée, tâche planifiée : `input()` lève EOFError.

        C'est le cas d'`update.bat` lancé autrement qu'au double-clic. La bonne
        réponse est « on ne supprime pas », pas une trace d'exception au milieu
        d'une mise à jour qui s'est bien passée.
        """
        def eof(prompt):
            raise EOFError
        self.assertFalse(M.confirm_purge(5_000_000_000, read=eof))

    def test_nothing_to_reclaim_asks_nothing(self):
        def boom(prompt):
            raise AssertionError("question posée pour zéro octet")
        self.assertFalse(M.confirm_purge(0, read=boom))
