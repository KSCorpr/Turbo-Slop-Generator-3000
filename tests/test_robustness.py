"""Ce qui doit survivre à une coupure, un proxy ou un disque déjà rempli.

Aucun de ces cas n'est visible en usage normal — c'est précisément pourquoi ils
se testent ici plutôt que de se découvrir le jour où l'application ne redémarre
plus.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import downloader, registry, settings
from atelier.fileio import atomic_write_text


class AtomicWriteTests(unittest.TestCase):
    def test_the_replacement_leaves_no_temporary_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "preferences.json"
            atomic_write_text(target, json.dumps({"theme": "dark"}))
            self.assertEqual(json.loads(target.read_text()), {"theme": "dark"})
            self.assertEqual([p.name for p in Path(tmp).iterdir()],
                             ["preferences.json"])

    def test_a_failed_write_leaves_the_previous_file_intact(self):
        """Le point de tout l'exercice : ne jamais perdre le fichier existant.

        `write_text` tronque AVANT d'écrire ; une écriture interrompue laissait
        un JSON à moitié écrit, et l'application ne redémarrait plus.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "preferences.json"
            target.write_text('{"theme": "light"}', encoding="utf-8")
            with patch("os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, '{"theme": "dark"}')
            self.assertEqual(json.loads(target.read_text()),
                             {"theme": "light"})
            self.assertEqual([p.name for p in Path(tmp).iterdir()],
                             ["preferences.json"])

    def test_the_temporary_lands_next_to_its_target(self):
        """`os.replace` n'est atomique que sur un même volume.

        Un temporaire dans %TEMP% marcherait la plupart du temps et échouerait
        exactement sur les machines qui nous intéressent : projet sur D:,
        %TEMP% sur C:.
        """
        seen = {}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "style_presets.json"
            real = tempfile.NamedTemporaryFile

            def spy(*args, **kwargs):
                seen["dir"] = kwargs.get("dir")
                return real(*args, **kwargs)

            with patch("tempfile.NamedTemporaryFile", side_effect=spy):
                atomic_write_text(target, "{}")
        self.assertEqual(Path(seen["dir"]), target.parent)


class DownloadSkipsTheHubTests(unittest.TestCase):
    """Un composant déjà installé ne doit pas dépendre du réseau."""

    def _component(self, quant_token="Q5_K_M"):
        return registry.Component(
            role="diffusion", repo="acme/model-GGUF",
            template="model-{quant}.gguf", quant=quant_token)

    def _run(self, comp, on_disk, content=b"weights"):
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "acme_model-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / on_disk).write_bytes(content)
            with patch.object(settings, "model_repo_dir",
                              return_value=repo_dir), \
                 patch.object(settings, "configure_hf_env"), \
                 patch.dict("sys.modules", {"huggingface_hub": _FakeHub()}):
                return downloader.download_component(comp), _FakeHub.listed

    def setUp(self):
        _FakeHub.listed = []

    def test_the_requested_quant_on_disk_is_returned_without_listing(self):
        path, listed = self._run(self._component(), "model-Q5_K_M.gguf")
        self.assertEqual(path.name, "model-Q5_K_M.gguf")
        self.assertEqual(listed, [], "the Hub was queried for nothing")

    def test_a_lower_quant_on_disk_does_not_block_an_upgrade(self):
        """Le piège de l'optimisation : un Q4 installé masquant le Q6 demandé.

        `resolve_component_path` sait se rabattre sur un quant voisin — c'est
        utile pour LIRE, et faux pour décider qu'il n'y a rien à télécharger.
        """
        with self.assertRaises(_HubCalled):
            self._run(self._component("Q6_K"), "model-Q4_K_M.gguf")
        self.assertEqual(_FakeHub.listed, ["acme/model-GGUF"])

    def test_an_empty_file_is_not_mistaken_for_a_download(self):
        """Un téléchargement coupé laisse un fichier de 0 octet AU BON NOM.

        C'est le cas qui rend le raccourci dangereux : le nom correspond, donc
        tout a l'air installé, et le modèle échoue à l'usage sans rien dire.
        """
        with self.assertRaises(_HubCalled):
            self._run(self._component(), "model-Q5_K_M.gguf", content=b"")
        self.assertEqual(_FakeHub.listed, ["acme/model-GGUF"])


class _HubCalled(RuntimeError):
    """Levée à la place d'un vrai téléchargement, pour le rendre observable."""


class _FakeHub:
    listed: list = []

    def __init__(self):
        _FakeHub.listed = []

    @staticmethod
    def list_repo_files(repo):
        _FakeHub.listed.append(repo)
        raise _HubCalled(repo)

    @staticmethod
    def hf_hub_download(**kwargs):  # pragma: no cover - jamais atteint
        raise _HubCalled(kwargs)


if __name__ == "__main__":
    unittest.main()
