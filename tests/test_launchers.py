"""Les lanceurs `.bat` / `.sh` — le seul point d'entrée pour qui n'a pas Python.

Ce fichier existe parce que j'ai livré une sonde utilisable uniquement par
`python scripts/...`, sur un projet dont l'utilisateur n'a pas Python installé :
il y a un interpréteur PORTABLE dans `python\\`, et tout passe par un `.bat`.
Un script sans lanceur est un script inexistant.

Deux défauts se voient au double-clic et nulle part ailleurs : un `.bat` qui
appelle un fichier qui n'existe plus, et un `.bat` qui invoque `python` tout
court — ce qui marche sur la machine du développeur et échoue chez tout le
monde.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# L'exigence n'est pas une orthographe précise mais un FAIT : le lanceur doit
# viser un python.exe DU PROJET. run.bat l'écrit « %~dp0python\python.exe »,
# install.bat « %PYDIR%\python.exe » — il installe cet interpréteur, il ne peut
# pas encore s'y référer autrement. Les deux sont corrects.
_PORTABLE = re.compile(r'set\s+"PY=[^"]*python\.exe"', re.I)
# Script Python appelé par un lanceur : « scripts\x.py », « app.py »…
_CALLS = re.compile(r"([\w./\\-]+\.py)")


def _launchers(suffix: str) -> list[Path]:
    return sorted(p for p in ROOT.glob(f"*{suffix}"))


class LaunchersExistTests(unittest.TestCase):
    def test_there_are_launchers_to_check(self):
        self.assertTrue(_launchers(".bat"))
        self.assertTrue(_launchers(".sh"))

    def test_every_launcher_calls_a_script_that_exists(self):
        missing = []
        for path in _launchers(".bat") + _launchers(".sh"):
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                # Une URL ou un fichier de %TEMP% n'est pas un script du dépôt :
                # install.bat télécharge get-pip.py, ce n'est pas un oubli.
                if "http" in line.lower() or "%TEMP%" in line.upper():
                    continue
                for call in set(_CALLS.findall(line)):
                    target = ROOT / call.replace("\\", "/")
                    if not target.is_file():
                        missing.append(f"{path.name} -> {call}")
        self.assertEqual(missing, [], "\n".join(missing))


class PortablePythonTests(unittest.TestCase):
    """Un `.bat` qui écrit juste `python` marche chez le développeur et
    échoue chez l'utilisateur, qui n'a que l'interpréteur du dossier."""

    def test_every_bat_prefers_the_bundled_interpreter(self):
        wrong = []
        for path in _launchers(".bat"):
            src = path.read_text(encoding="utf-8", errors="replace")
            if ".py" not in src:
                continue            # lanceur qui n'exécute pas de Python
            if not _PORTABLE.search(src):
                wrong.append(path.name)
        self.assertEqual(wrong, [], f"sans Python portable : {wrong}")

    def test_every_bat_runs_from_its_own_folder(self):
        """Sans `cd /d %~dp0`, un double-clic depuis un raccourci démarre
        ailleurs et ne trouve plus rien."""
        wrong = [p.name for p in _launchers(".bat")
                 if "cd /d" not in p.read_text(encoding="utf-8",
                                               errors="replace").lower()]
        self.assertEqual(wrong, [], f"sans cd : {wrong}")


class EveryEntryPointIsReachableTests(unittest.TestCase):
    """Un script destiné à l'utilisateur DOIT avoir son lanceur."""

    # Scripts internes, appelés par le code et jamais à la main.
    INTERNAL = {"get_sdcpp.py", "get_trellis.py", "setup_tools.py",
                "setup_adetailer.py",  # idem, onglet « Détails »
                "_torch_setup.py", "convert_gguf.py"}

    @staticmethod
    def _basename(call: str) -> str:
        """Le nom du fichier, quel que soit le séparateur écrit dans le .bat.

        `Path("scripts\\x.py").name` rend la chaîne ENTIÈRE sous Linux : la
        barre inverse n'y est pas un séparateur. Le test ne passait donc que
        parce qu'un autre lanceur mentionnait par hasard le même script avec
        une barre normale — et il aurait rendu un verdict différent sous
        Windows, c'est-à-dire sur la seule machine qui exécute ces .bat.
        """
        return call.replace("\\", "/").rsplit("/", 1)[-1]

    def test_user_facing_scripts_have_a_launcher(self):
        launched = set()
        for path in _launchers(".bat") + _launchers(".sh"):
            launched |= {self._basename(c)
                         for c in _CALLS.findall(
                             path.read_text(encoding="utf-8",
                                            errors="replace"))}
        orphans = [p.name for p in sorted((ROOT / "scripts").glob("*.py"))
                   if p.name not in launched and p.name not in self.INTERNAL]
        self.assertEqual(orphans, [],
                         f"scripts sans lanceur : {orphans} — ajoutez un .bat "
                         "ou classez-les dans INTERNAL")


if __name__ == "__main__":
    unittest.main()


class UpdateBranchTests(unittest.TestCase):
    """`update.bat` doit ramener LA branche installée, pas « la principale ».

    Le piège était réel et destructeur : `BRANCH = "main"` en dur signifiait
    qu'une installation faite depuis une autre branche se faisait écraser au
    premier `update.bat`, sans un mot, et sans autre retour que le rollback.
    """

    @staticmethod
    def _module():
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import update_app
        return update_app

    def test_the_default_branch_is_the_one_this_code_lives_on(self):
        """LE test qui compte, et il ne sert qu'aux développeurs.

        La valeur voyage avec le code : chaque branche porte la sienne. Ce qui
        peut mal tourner, c'est une fusion — ramener « Test7000 » sur `main`
        enverrait tous les utilisateurs de `main` sur une branche
        expérimentale. Ce test l'attrape là où il faut : dans le dépôt git.

        Ignoré quand il n'y a pas de dépôt (une installation utilisateur est un
        zip déplié) ou en HEAD détachée, où la question n'a pas de réponse.
        """
        import subprocess
        if not (ROOT / ".git").exists():
            raise unittest.SkipTest("not a git checkout")
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            raise unittest.SkipTest("git unavailable")
        if not branch or branch == "HEAD":
            raise unittest.SkipTest("detached HEAD")
        self.assertEqual(
            self._module().DEFAULT_BRANCH, branch,
            "update.bat would send this install to another branch than the "
            "one this code is on")

    def test_the_recorded_branch_wins_over_the_shipped_default(self):
        """Une archive d'une autre branche dépliée par-dessus ne doit pas
        faire basculer l'installation en silence."""
        u = self._module()
        branch, warning = u.resolve_branch(None, {"branch": "Test7000"})
        self.assertEqual(branch, "Test7000")
        self.assertIn("--branch", warning)

    def test_an_explicit_request_wins_and_says_nothing(self):
        u = self._module()
        self.assertEqual(u.resolve_branch("Test7000", {"branch": "main"}),
                         ("Test7000", ""))

    def test_a_fresh_install_follows_the_code_it_came_with(self):
        u = self._module()
        self.assertEqual(u.resolve_branch(None, {}),
                         (u.DEFAULT_BRANCH, ""))

    def test_the_launcher_forwards_its_arguments(self):
        """Sans `%*`, `--branch` et `--rollback` n'arriveraient jamais."""
        text = (ROOT / "update.bat").read_text(encoding="utf-8",
                                               errors="replace")
        self.assertIn("%*", text)
