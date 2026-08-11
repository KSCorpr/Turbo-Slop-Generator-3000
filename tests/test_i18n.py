"""Santé du dictionnaire de traduction.

Le piège que ces tests attrapent : un dictionnaire Python accepte silencieusement
une clé écrite deux fois — la dernière gagne. Quand les deux valeurs diffèrent,
une traduction disparaît sans le moindre signe, et personne ne s'en aperçoit
avant de lire l'interface en anglais. Ça s'est produit en renommant des onglets.
"""
import ast
import unittest
from pathlib import Path

from atelier import i18n

_SOURCE = Path(i18n.__file__)


def _entries() -> list[tuple[str, str, int]]:
    """(clé, valeur, ligne) telles qu'ÉCRITES, avant que Python ne dédoublonne."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and getattr(target, "id", "") == "_EN":
            return [(ast.literal_eval(k), ast.literal_eval(v), k.lineno)
                    for k, v in zip(node.value.keys, node.value.values)]
    raise AssertionError("dictionnaire _EN introuvable dans i18n.py")


class TranslationTableTests(unittest.TestCase):
    def test_no_key_is_silently_overwritten(self):
        seen: dict[str, tuple[str, int]] = {}
        clashes = []
        for key, value, line in _entries():
            if key in seen and seen[key][0] != value:
                clashes.append(f"« {key} » ligne {seen[key][1]} ({seen[key][0]!r}) "
                               f"écrasée ligne {line} ({value!r})")
            seen[key] = (value, line)
        self.assertEqual(clashes, [], "\n".join(clashes))

    def test_no_empty_translation(self):
        empty = [k for k, v, _ in _entries() if not v.strip()]
        self.assertEqual(empty, [])

    def test_format_placeholders_are_preserved(self):
        """« {seed} » perdu à la traduction = KeyError à l'exécution."""
        import re
        holder = re.compile(r"\{(\w+)\}")
        broken = []
        for key, value, line in _entries():
            if set(holder.findall(key)) != set(holder.findall(value)):
                broken.append(f"ligne {line} : {key!r} -> {value!r}")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_reverse_lookup_is_usable(self):
        # to_source() sert aux menus dont le libellé est la clé : il doit
        # exister une inverse pour les entrées réellement distinctes.
        i18n.set_lang("en")
        try:
            self.assertEqual(i18n.to_source(i18n.t("🧰 Outils")), "🧰 Outils")
        finally:
            i18n.set_lang("fr")


if __name__ == "__main__":
    unittest.main()
