"""Génération de vidéo (Wan 2.2 TI2V 5B, sd.cpp `-M vid_gen`).

Trois choses peuvent mal tourner ici, et aucune ne se voit à la lecture :

1. **le nombre d'images.** Le VAE temporel de Wan travaille par groupes de
   quatre plus une. 50 images n'existent pas : sd.cpp réaligne en silence, et
   l'interface aurait annoncé une durée que la vidéo n'a pas ;
2. **le modèle vidéo proposé comme modèle d'image.** Une quinzaine d'écrans
   lisent le catalogue pour remplir une liste de modèles — passe HD, outpaint,
   ADetailer, upscale créatif, banc de mesure. Aucun ne sait quoi faire de
   Wan, et aucun ne le dirait : il échouerait à la génération suivante ;
3. **la reprise de nom.** `atelier/ui/video_tab.py` et
   `atelier/engine/video.py` ont déjà été déclarés « retirés » du temps de
   LTX-2.3. Les fichiers sont revenus. C'est exactement le piège documenté en
   haut de `scripts/maintenance.py`, et il s'est refermé ici.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

from atelier import registry, settings  # noqa: E402
from atelier.engine import sdcpp, video  # noqa: E402


class FrameAlignmentTests(unittest.TestCase):
    """4n+1 : la contrainte du VAE temporel, pas une préférence."""

    def test_the_documented_lengths_are_already_exact(self):
        for n in (1, 5, 33, 65, 81, 121):
            self.assertEqual(sdcpp.align_video_frames(n), n, n)

    def test_anything_else_is_rounded_down_not_up(self):
        """Vers le BAS : arrondir vers le haut ferait dépasser une durée
        choisie, et sur une carte juste, quatre images de plus peuvent être
        exactement ce qui ne tient pas."""
        self.assertEqual(sdcpp.align_video_frames(50), 49)
        self.assertEqual(sdcpp.align_video_frames(34), 33)
        self.assertEqual(sdcpp.align_video_frames(36), 33)

    def test_nothing_below_one_frame(self):
        for n in (0, -1, 1):
            self.assertEqual(sdcpp.align_video_frames(n), 1, n)

    def test_the_ui_only_offers_exact_lengths(self):
        """Une durée affichée doit être la vraie durée.

        Les longueurs proposées par l'onglet sont écrites à la main ; ce test
        est ce qui empêche d'en ajouter une qui « a l'air ronde ».
        """
        from atelier.ui import video_tab
        for n in video_tab.LENGTHS:
            self.assertEqual(sdcpp.align_video_frames(n), n,
                             f"{n} images n'est pas un 4n+1")

    def test_the_duration_shown_is_the_duration_produced(self):
        self.assertAlmostEqual(sdcpp.video_duration_s(121, 24), 121 / 24)
        # Une demande non alignée est mesurée sur ce qui sortira vraiment.
        self.assertAlmostEqual(sdcpp.video_duration_s(50, 24), 49 / 24)


class CatalogueTests(unittest.TestCase):
    """Ce qui produit une vidéo ne doit pas se proposer comme modèle d'image."""

    def setUp(self):
        self.prefs = settings.load_prefs()

    def test_there_is_a_video_model(self):
        self.assertTrue(registry.load_video_models(self.prefs))

    def test_the_image_lists_do_not_contain_it(self):
        image_ids = {m.id for m in registry.load_base_models(self.prefs)}
        video_ids = {m.id for m in registry.load_video_models(self.prefs)}
        self.assertTrue(video_ids)
        self.assertEqual(image_ids & video_ids, set())

    def test_the_default_is_the_narrow_one(self):
        """Le défaut de `load_base_models` doit être « image ».

        C'est tout le mécanisme : une quinzaine d'appelants ne passent pas
        d'argument, et c'est le défaut qui les protège. Inversé, chacun d'eux
        proposerait Wan comme modèle d'image sans que rien ne le signale.
        """
        import inspect
        sig = inspect.signature(registry.load_base_models)
        self.assertEqual(sig.parameters["kind"].default, registry.IMAGE)

    def test_the_download_catalogue_shows_everything(self):
        every = {m.id for m in registry.load_base_models(
            self.prefs, kind=registry.EVERYTHING)}
        for m in registry.load_video_models(self.prefs):
            self.assertIn(m.id, every)

    def test_asking_for_it_by_name_still_works(self):
        """`get_base_model` ne filtre pas : on nomme ce qu'on demande."""
        m = registry.get_base_model("wan22-ti2v-5b", self.prefs)
        self.assertIsNotNone(m)
        self.assertEqual(m.kind, registry.VIDEO)

    def test_the_video_model_declares_what_the_command_needs(self):
        m = registry.get_base_model("wan22-ti2v-5b", self.prefs)
        roles = {c.role for c in m.components}
        self.assertEqual(roles, {"diffusion", "vae", "t5xxl"})

    def test_the_orphan_check_knows_about_it(self):
        """Sinon la maintenance proposerait d'effacer ses 8,5 Go.

        Le contrôle des modèles orphelins compare les dossiers de `models/` au
        catalogue. S'il ne demandait que les modèles d'IMAGE, les dossiers de
        Wan n'y seraient pas — et il les signalerait comme récupérables.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        import maintenance
        expected = maintenance._expected_model_dirs()
        m = registry.get_base_model("wan22-ti2v-5b", self.prefs)
        for comp in m.components:
            self.assertIn(settings.model_repo_dir(comp.repo).name, expected,
                          f"{comp.repo} passerait pour un dossier orphelin")


class CommandTests(unittest.TestCase):
    """La ligne de commande, comparée à docs/wan.md."""

    def _cmd(self, **kw):
        fields = dict(diffusion_model=None, vae=None, t5xxl=None,
                      prompt="a lovely cat", cfg_scale=6.0, steps=20,
                      sampler="euler", width=704, height=1280,
                      video_frames=33, fps=24, flow_shift=3.0)
        fields.update(kw)
        req = sdcpp.VidRequest(**fields)
        with patch.object(sdcpp, "video_supported", return_value=True), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset()), \
             patch.object(sdcpp, "_require"):
            return sdcpp.build_vid_cmd(Path("sd-cli"), req, Path("out.webm"))

    def test_it_asks_for_the_video_mode(self):
        cmd = self._cmd()
        self.assertIn("--mode", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "vid_gen")

    def test_the_documented_video_options_are_there(self):
        cmd = self._cmd()
        for opt, value in (("--video-frames", "33"), ("--fps", "24"),
                           ("--flow-shift", "3.0")):
            self.assertIn(opt, cmd)
            self.assertEqual(cmd[cmd.index(opt) + 1], value, opt)

    def test_the_frame_count_is_aligned_before_it_is_sent(self):
        """Le moteur réaligne de toute façon — mais alors on ne le sait pas.

        Envoyer 50 et recevoir 49 images est un écart silencieux entre ce que
        l'interface a annoncé et ce que le fichier contient.
        """
        cmd = self._cmd(video_frames=50)
        self.assertEqual(cmd[cmd.index("--video-frames") + 1], "49")

    def test_a_starting_frame_carries_no_strength(self):
        """En vid_gen, `-i` est la PREMIÈRE IMAGE du film.

        En img2img, la même option veut dire « repeins par-dessus » et
        s'accompagne de `--strength`. La confondre reviendrait à demander à Wan
        de repeindre l'image de départ au lieu de l'animer.
        """
        cmd = self._cmd(start_image=Path("start.png"))
        self.assertIn("-i", cmd)
        self.assertNotIn("--strength", cmd)

    def test_text_to_video_sends_no_image_at_all(self):
        self.assertNotIn("-i", self._cmd())

    def test_the_negative_prompt_travels_when_guidance_is_on(self):
        cmd = self._cmd(negative="blurry")
        self.assertIn("-n", cmd)
        self.assertEqual(cmd[cmd.index("-n") + 1], "blurry")

    def test_an_engine_without_video_says_so_before_running(self):
        """Un argument inconnu fait sortir sd-cli sur un message illisible.

        On préfère la phrase qui nomme le geste : `update.bat` met le moteur à
        jour juste après le code.
        """
        req = sdcpp.VidRequest(prompt="x")
        with patch.object(sdcpp, "video_supported", return_value=False), \
             patch.object(sdcpp, "_require"):
            with self.assertRaises(sdcpp.EngineError) as caught:
                sdcpp.build_vid_cmd(Path("sd-cli"), req, Path("out.webm"))
        self.assertIn("update.bat", str(caught.exception))

    def test_the_capability_is_read_from_the_binary_not_a_version(self):
        """Une même version couvre des builds qui n'offrent pas la même chose."""
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--video-frames"})):
            self.assertTrue(sdcpp.video_supported(Path("sd-cli")))
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset({"--hires"})):
            self.assertFalse(sdcpp.video_supported(Path("sd-cli")))


class PipelineTests(unittest.TestCase):
    """Les deux décisions que `video.generate_video` prend d'office."""

    def _built(self, prefs: dict, **kw):
        """Lance la génération jusqu'à la commande, sans rien exécuter."""
        seen: dict = {}
        model = registry.get_base_model("wan22-ti2v-5b", prefs)

        def fake_files(m, **_):
            return {"diffusion": Path("d.gguf"), "vae": Path("v.safetensors"),
                    "t5xxl": Path("t.gguf"), "enc": None, "model_path": None,
                    "uncond": None, "clip_l": None, "llm_vision": None}

        def fake_build(sd_cli, req, output):
            seen["req"] = req
            return ["sd-cli"]

        with patch.object(settings, "load_prefs", return_value=prefs), \
             patch.object(settings, "find_sd_cli", return_value=Path("sd-cli")), \
             patch.object(registry, "get_base_model", return_value=model), \
             patch.object(video, "resolve_model_files", fake_files), \
             patch.object(sdcpp, "build_vid_cmd", fake_build), \
             patch.object(sdcpp, "run"), \
             patch.object(Path, "is_file", lambda self: True):
            video.generate_video(prompt="a cat", **kw)
        return seen["req"]

    def test_vae_tiling_is_forced_whatever_the_profile_says(self):
        """Le décodage du VAE est le seul moment où toutes les images
        existent en même temps : c'est là que la mémoire lâche, et ce coût
        n'apparaît dans la taille d'aucun fichier de poids."""
        req = self._built({"auto_optimize": False,
                           "flags": {"vae_tiling": False}})
        self.assertTrue(req.flags["vae_tiling"])

    def test_a_budget_is_supplied_when_none_was_chosen(self):
        """Sans `--max-vram`, le moteur découpe son graphe sans cible et
        découvre au milieu qu'il ne tient pas — le mode d'échec diagnostiqué
        sur les images, en pire, parce qu'une vidéo dure plus longtemps."""
        req = self._built({"auto_optimize": False, "flags": {}})
        self.assertTrue(req.max_vram)

    def test_a_chosen_budget_is_left_alone(self):
        """Un budget écrit à la main est une décision, pas un défaut."""
        req = self._built({"auto_optimize": False, "flags": {},
                           "max_vram": "6"})
        self.assertEqual(req.max_vram, "6")

    def test_an_image_model_is_refused_by_name(self):
        prefs = settings.load_prefs()
        with patch.object(settings, "load_prefs", return_value=prefs), \
             patch.object(settings, "find_sd_cli", return_value=Path("sd-cli")):
            with self.assertRaises(sdcpp.EngineError) as caught:
                video.generate_video(prompt="x", model_id="z-image-turbo")
        self.assertIn("not a video model", str(caught.exception))


class RenamedFileTests(unittest.TestCase):
    """Le piège de la reprise de nom, refermé pour de bon."""

    def test_the_old_removal_entry_is_gone(self):
        """`atelier/engine/video.py` a été déclaré retiré, puis est revenu.

        `_still_in_service` empêchait la suppression, donc rien n'aurait été
        perdu — mais l'entrée aurait imprimé à chaque maintenance un
        avertissement décrivant une erreur qui n'existe pas.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        import maintenance
        declared = {rel for feat in maintenance.REMOVED_FEATURES
                    for rel in feat["files"]}
        for rel in ("atelier/engine/video.py", "atelier/ui/video_tab.py"):
            self.assertNotIn(rel, declared)

    def test_no_declared_removal_targets_a_living_file(self):
        """La règle générale, revérifiée depuis ce fichier-ci.

        `test_maintenance.py` la vérifie déjà ; on la redit ici parce que
        c'est CE changement qui l'a mise en défaut, et qu'un test placé là où
        la faute se commet est celui qu'on relit.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        import maintenance
        living = [rel for feat in maintenance.REMOVED_FEATURES
                  for rel in feat["files"]
                  if maintenance._still_in_service(rel)]
        self.assertEqual(living, [])


if __name__ == "__main__":
    unittest.main()
