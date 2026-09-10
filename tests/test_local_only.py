"""L'application n'écoute que sur cette machine, et le moteur ne survit pas
à une image.

Deux options ont été retirées ensemble parce qu'elles avaient le même défaut :
elles transformaient un outil local en service, et ni l'une ni l'autre n'était
utilisée.

  • **Le partage réseau** (`run-lan.bat`, `--listen`, `--share`) exposait
    l'interface aux autres machines. Une interface qui accepte des connexions
    doit alors répondre à trois questions — un mot de passe qui vaille quelque
    chose, une règle de pare-feu, et ce que `allowed_paths` laisse lire à
    quiconque connaît l'URL — dont aucune n'a de bonne réponse par défaut.

  • **Le moteur résident** (`sd-server`) gardait le modèle en VRAM entre deux
    images. En échange, chaque outil du projet devait penser à lui faire rendre
    la carte avant de la prendre : sd-cli, la 3D, la boîte à outils, le banc de
    mesure. Un oubli ne se voyait qu'au moment d'un manque de mémoire, loin de
    sa cause.

Ces tests ne relisent pas la suppression : ils vérifient qu'elle ne revient pas
par un chemin détourné — un drapeau qu'on remet « au cas où », un import laissé
en place.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sources() -> list[Path]:
    files = [ROOT / "app.py"]
    for folder in ("atelier", "scripts"):
        files += sorted((ROOT / folder).rglob("*.py"))
    return files


class NoNetworkSharingTests(unittest.TestCase):
    def test_the_launch_host_is_the_loopback_and_nothing_else(self):
        """`server_name` décide seul qui peut se connecter.

        On lit l'ARBRE et pas le texte : une constante nommée est ce qu'on
        veut, une adresse construite à l'exécution (`"0.0.0.0" if …`) est
        exactement ce que ce test refuse.
        """
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        hosts = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 for tgt in n.targets
                 if isinstance(tgt, ast.Name) and tgt.id == "HOST"]
        self.assertEqual(len(hosts), 1, "app.py ne fixe pas un HOST unique")
        self.assertIsInstance(hosts[0], ast.Constant)
        self.assertEqual(hosts[0].value, "127.0.0.1")

    def test_the_command_line_offers_no_way_to_open_it_up(self):
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        flags = {n.args[0].value for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "add_argument"
                 and n.args and isinstance(n.args[0], ast.Constant)}
        for gone in ("--listen", "--share", "--auth", "--host"):
            self.assertNotIn(gone, flags, f"{gone} est revenu")

    def test_the_lan_launchers_are_gone(self):
        back = [p.name for p in ROOT.glob("run-lan.*")]
        self.assertEqual(back, [], f"lanceurs réseau revenus : {back}")

    def test_the_declared_removal_covers_them(self):
        """Une copie DÉJÀ installée ne perd pas un fichier toute seule.

        Dézipper par-dessus ajoute et remplace, mais n'efface jamais : sans
        cette déclaration, `run-lan.bat` resterait sur les machines des
        utilisateurs et continuerait de lancer une option qui n'existe plus.
        """
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import maintenance
        declared = {rel for feat in maintenance.REMOVED_FEATURES
                    for rel in feat["files"]}
        for rel in ("run-lan.bat", "run-lan.sh",
                    "atelier/engine/sdserver.py"):
            self.assertIn(rel, declared, f"{rel} n'est pas déclaré retiré")


class NoResidentEngineTests(unittest.TestCase):
    def test_nothing_imports_the_resident_engine_any_more(self):
        guilty = []
        for path in _sources():
            src = path.read_text(encoding="utf-8", errors="replace")
            for name in ("sdserver", "resident_engine",
                         "release_resident_engine"):
                # Les commentaires ont le droit de raconter l'histoire ; le
                # CODE, non. D'où la lecture de l'arbre plutôt que du texte.
                tree = ast.parse(src)
                hit = any(
                    (isinstance(n, ast.Name) and n.id == name)
                    or (isinstance(n, ast.Attribute) and n.attr == name)
                    or (isinstance(n, ast.alias) and n.name == name)
                    for n in ast.walk(tree))
                if hit:
                    guilty.append(f"{path.relative_to(ROOT)} → {name}")
        self.assertEqual(guilty, [], "\n".join(guilty))

    def test_the_preference_is_wiped_when_the_settings_are_touched(self):
        """Un réglage devenu sans objet reste dans prefs.json à vie.

        Il n'a plus d'effet, mais il se relit dans les diagnostics et il se
        raconte : « j'avais activé le moteur résident » alors qu'il n'existe
        plus. Le nettoyage des réglages périmés était déjà là — il fallait
        seulement y ajouter cette clé.
        """
        src = (ROOT / "atelier" / "ui" / "settings_tab.py").read_text(
            encoding="utf-8")
        self.assertIn('"resident_engine"', src)


if __name__ == "__main__":
    unittest.main()
