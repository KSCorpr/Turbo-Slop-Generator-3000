"""Le moteur PyTorch, vérifié SANS carte graphique.

C'est la contrainte qui a dessiné le module : tout ce qui DÉCIDE — précision,
placement, échantillonneur, mode, prompt négatif — est du Python sans torch, et
c'est donc ici que ça se vérifie. Sans cette séparation, ces règles ne seraient
contrôlables que sur une machine avec une carte, c'est-à-dire jamais en
intégration continue, c'est-à-dire jamais.

Ce que ces tests protègent n'est pas « le code tourne » mais « le code dit la
vérité » : un mode qui n'existe pas doit se voir, un échantillonneur sans
équivalent doit se dire, une taille inconnue ne doit pas acheter le chemin
rapide.
"""
from __future__ import annotations

import inspect
import pathlib
import unittest

from atelier.engine import backends, generate as engine_generate
from atelier.torchengine import backend, catalog, placement, schedulers


class SignatureTests(unittest.TestCase):
    def test_both_engines_take_exactly_the_same_arguments(self):
        """Toute l'interface passe par une seule fonction ; les deux moteurs
        doivent donc s'y substituer sans que le moindre appelant change.

        L'aiguille transmet `*args, **kwargs` : un paramètre en trop ou en
        moins d'un côté ne se verrait qu'au moment où quelqu'un l'utilise,
        c'est-à-dire sur une seule fonctionnalité, longtemps après.
        """
        native = inspect.signature(engine_generate.generate_sdcpp)
        torch = inspect.signature(backend.generate)
        self.assertEqual(list(native.parameters), list(torch.parameters))

    def test_the_dispatcher_still_answers_introspection(self):
        """La passe HD vérifie qu'un paramètre existe avant de s'en servir."""
        params = inspect.signature(engine_generate.generate).parameters
        for name in ("stream_layers", "max_vram", "prefs_override"):
            self.assertIn(name, params)


class BackendChoiceTests(unittest.TestCase):
    def test_this_branch_defaults_to_pytorch(self):
        self.assertEqual(backends.DEFAULT, backends.TORCH)
        self.assertEqual(backends.active({}), backends.TORCH)

    def test_a_saved_preference_wins_over_the_default(self):
        self.assertEqual(backends.active({"engine_backend": "sdcpp"}),
                         backends.SDCPP)

    def test_an_unknown_value_falls_back_instead_of_raising(self):
        """Un fichier de préférences hérité d'une autre branche ne doit pas
        empêcher l'application de démarrer."""
        self.assertEqual(backends.active({"engine_backend": "comfyui"}),
                         backends.DEFAULT)

    def test_the_environment_variable_wins_over_everything(self):
        """Deux applications côte à côte, une par moteur, sans rien modifier."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {backends.ENV_VAR: "sdcpp"}):
            self.assertEqual(backends.active({"engine_backend": "torch"}),
                             backends.SDCPP)


class CatalogTests(unittest.TestCase):
    def test_every_pytorch_entry_matches_a_catalogue_model(self):
        """Les identifiants sont le pont entre les deux moteurs : un id qui
        n'existe que d'un côté est un onglet qui ne générera jamais."""
        from atelier import registry
        known = {m.id for m in registry.load_base_models({})}
        for model in catalog.load():
            self.assertIn(model.id, known, model.id)

    def test_a_missing_mode_is_a_fact_and_not_an_oversight(self):
        """diffusers 0.40 ne publie PAS de classe img2img pour Krea 2.

        Le catalogue doit le dire, sinon l'interface propose une « image de
        départ » qui sera ignorée en silence.
        """
        krea = catalog.get("krea2-turbo")
        self.assertTrue(krea.can(catalog.TEXT_TO_IMAGE))
        self.assertFalse(krea.can(catalog.IMAGE_TO_IMAGE))
        self.assertEqual(krea.modes, [catalog.TEXT_TO_IMAGE])

    def test_flux_edits_with_the_same_class_it_generates_with(self):
        """`Flux2KleinPipeline.__call__` prend `image=` : l'édition
        multi-références n'est pas une seconde classe."""
        flux = catalog.get("flux2-klein-9b")
        self.assertEqual(flux.pipeline_for(catalog.EDIT),
                         flux.pipeline_for(catalog.TEXT_TO_IMAGE))
        self.assertTrue(flux.can(catalog.EDIT))

    def test_a_closed_repository_declares_no_size_rather_than_zero(self):
        """Zéro serait commode et faux : il se propagerait dans le placement
        comme un modèle qui ne pèse rien, donc qui tient partout."""
        for model in catalog.load():
            if model.gated:
                self.assertIsNone(model.resident_bf16_gb, model.id)

    def test_the_requested_mode_is_computed_before_it_is_granted(self):
        krea = catalog.get("krea2-turbo")
        self.assertEqual(
            catalog.mode_for(krea, init_image="a.png"),
            catalog.IMAGE_TO_IMAGE,
            "the request must be readable even when the model cannot serve it")


class PlacementTests(unittest.TestCase):
    Z = (20.5, 12.3)   # Z-Image Turbo en bf16 : pipeline entier, plus gros bloc

    def test_the_two_cards_this_project_targets_get_a_working_plan(self):
        """11 Go (2080 Ti) et 12 Go (3060) : ni l'un ni l'autre ne doit tomber
        sur le streaming couche par couche, qui coûte un facteur dix."""
        for vram, arch in ((11.0, "turing"), (12.0, "ampere")):
            p = placement.plan(vram, *self.Z, arch)
            self.assertEqual(p.mode, placement.MODEL_OFFLOAD, f"{vram} GB")
            self.assertEqual(p.quant, "int8", f"{vram} GB")

    def test_a_big_card_keeps_full_precision_and_stays_resident(self):
        p = placement.plan(24.0, *self.Z, "ada")
        self.assertEqual(p.mode, placement.FULL)
        self.assertEqual(p.quant, "none")

    def test_precision_is_only_traded_to_escape_layer_streaming(self):
        """16 Go tient en bf16 avec la décharge par module : descendre en int8
        pour gagner de la vitesse serait un choix qu'on n'a pas à faire à la
        place de l'utilisateur."""
        p = placement.plan(16.0, *self.Z, "ada")
        self.assertEqual(p.quant, "none")
        self.assertEqual(p.mode, placement.MODEL_OFFLOAD)

    def test_an_unknown_size_never_buys_the_fast_path(self):
        p = placement.plan(24.0, None, None, "ada")
        self.assertEqual(p.mode, placement.MODEL_OFFLOAD)

    def test_turing_and_older_stay_on_fp16(self):
        """bf16 n'est matériel qu'à partir d'Ampere ; l'émuler est lent."""
        self.assertEqual(placement.dtype_for("turing"), "float16")
        self.assertEqual(placement.dtype_for("pascal"), "float16")
        self.assertEqual(placement.dtype_for("ampere"), "bfloat16")

    def test_no_gpu_means_cpu_and_full_precision(self):
        p = placement.plan(None, *self.Z, "unknown")
        self.assertEqual(p.mode, placement.SEQUENTIAL_OFFLOAD)
        self.assertEqual(p.dtype, "float32")

    def test_the_reason_is_written_for_a_human(self):
        line = placement.describe(placement.plan(12.0, *self.Z, "ampere"))
        self.assertIn("int8", line)
        self.assertIn("GB", line)


class SchedulerTranslationTests(unittest.TestCase):
    def test_the_default_stays_the_default(self):
        choice = schedulers.resolve("euler", "auto")
        self.assertTrue(choice.is_default)
        self.assertEqual(choice.notes, ())

    def test_a_sampler_without_equivalent_falls_back_AND_says_so(self):
        """Le seul vrai défaut possible ici : retomber en silence. On croirait
        avoir essayé une méthode qu'on n'a jamais lancée."""
        choice = schedulers.resolve("res_multistep", "auto")
        self.assertEqual(choice.cls, "FlowMatchEulerDiscreteScheduler")
        self.assertTrue(any("no diffusers equivalent" in n
                            for n in choice.notes), choice.notes)

    def test_an_approximation_is_labelled_as_one(self):
        choice = schedulers.resolve("dpm++2m_sde_bt", "auto")
        self.assertTrue(any("seed" in n for n in choice.notes), choice.notes)

    def test_flow_shift_uses_the_name_the_class_expects(self):
        """Le même réglage porte deux orthographes selon la classe ; l'envoyer
        sous la mauvaise lève un TypeError au premier chargement."""
        euler = schedulers.resolve("euler", "auto", flow_shift=1.15)
        self.assertEqual(euler.kwargs.get("shift"), 1.15)
        dpm = schedulers.resolve("dpm++2m", "auto", flow_shift=1.15)
        self.assertEqual(dpm.kwargs.get("flow_shift"), 1.15)

    def test_flow_models_never_get_diffusion_sigmas_by_accident(self):
        """`use_flow_sigmas` est ce qui rend les solveurs DPM utilisables sur
        du flow matching ; sans lui le rendu part en bouillie."""
        for name in ("dpm++2m", "dpm++2m_sde", "dpm++2s_a"):
            self.assertTrue(schedulers.resolve(name, "auto")
                            .kwargs.get("use_flow_sigmas"), name)

    def test_options_the_class_refuses_are_dropped_not_forced(self):
        """Mesuré : 110 des 336 combinaisons du menu échouaient à la
        construction, presque toutes pour un argument que la classe visée
        n'accepte pas. Un TypeError n'arrive qu'au premier chargement — donc
        après vingt secondes d'attente et jamais en test."""
        kwargs, dropped = schedulers.accepted(
            {"shift": 1.0, "timestep_spacing": "trailing"},
            {"shift", "num_train_timesteps"})
        self.assertEqual(kwargs, {"shift": 1.0})
        self.assertEqual(dropped, ["timestep_spacing"])

    def test_an_unreadable_signature_filters_nothing(self):
        """`LMSDiscreteScheduler` n'expose qu'un `*args`. Mieux vaut laisser
        passer et échouer bruyamment que tout retirer en silence."""
        kwargs, dropped = schedulers.accepted({"a": 1}, None)
        self.assertEqual(kwargs, {"a": 1})
        self.assertEqual(dropped, [])


def _diffusers_or_skip():
    try:
        import diffusers
    except ImportError:
        raise unittest.SkipTest("diffusers is not installed here")
    return diffusers


class AgainstTheRealLibraryTests(unittest.TestCase):
    """Le seul test qui touche vraiment diffusers — et il en vaut la peine.

    Le tableau de traduction a l'air juste tant qu'on le LIT. Confronté aux
    vraies classes, 110 des 336 combinaisons du menu levaient une exception à
    la construction : arguments inconnus, options réservées aux schedulers de
    diffusion, une variante de DPM qui refuse son propre réglage par défaut.
    Aucune ne se serait vue avant le premier clic de l'utilisateur, chacune sur
    une combinaison précise.

    Ignoré quand diffusers n'est pas installé : cette branche n'oblige personne
    à porter deux gigaoctets de dépendances pour lancer la suite.
    """

    def test_every_menu_combination_builds(self):
        from atelier import sampling
        from atelier.torchengine import runtime
        diffusers = _diffusers_or_skip()
        broken = []
        for sampler in sampling.SAMPLERS:
            for schedule in sampling.SCHEDULES:
                choice = schedulers.resolve(sampler, schedule, flow_shift=1.15)
                cls = getattr(diffusers, choice.cls, None)
                if cls is None:
                    broken.append(f"{sampler}/{schedule}: no {choice.cls}")
                    continue
                kwargs, _ = schedulers.accepted(
                    choice.kwargs, runtime._signature_of(cls))
                try:
                    cls(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    broken.append(f"{sampler}/{schedule}: {exc}")
        self.assertEqual(broken, [], "\n".join(broken[:10]))

    def test_every_pipeline_class_the_catalogue_names_exists(self):
        diffusers = _diffusers_or_skip()
        missing = [f"{m.id}/{mode}: {cls}"
                   for m in catalog.load()
                   for mode, cls in m.pipelines.items()
                   if not hasattr(diffusers, cls)]
        self.assertEqual(missing, [])


class CallArgumentTests(unittest.TestCase):
    """Ce qu'on passe au pipeline : c'est là que les familles divergent."""

    @classmethod
    def setUpClass(cls):
        """De VRAIES images sur le disque, pas des chemins factices.

        `_call_kwargs` ouvre ce qu'on lui donne — c'est son travail, et un
        double qui ne l'ouvrirait pas laisserait passer une inversion
        d'arguments entre l'image et le masque.
        """
        import tempfile
        from PIL import Image
        cls._tmp = tempfile.TemporaryDirectory()
        cls.images = []
        for name in ("a.png", "b.png", "mask.png"):
            path = pathlib.Path(cls._tmp.name) / name
            Image.new("RGB", (64, 64), "grey").save(path)
            cls.images.append(str(path))
        cls.a, cls.b, cls.mask = cls.images

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _kwargs(self, model_id, mode, **over):
        model = catalog.get(model_id)
        args = dict(prompt="p", negative="n", steps=8, cfg_scale=1.0,
                    width=1024, height=1024, init_image=None, ref_image=None,
                    mask_image=None, strength=0.6)
        args.update(over)
        return backend._call_kwargs(mode, model, **args)

    def test_a_negative_prompt_is_dropped_at_cfg_one(self):
        """À CFG 1.0 il n'y a pas de passe non conditionnée où l'appliquer :
        l'envoyer laisserait croire qu'il agit."""
        self.assertNotIn("negative_prompt",
                         self._kwargs("z-image-turbo", catalog.TEXT_TO_IMAGE))

    def test_a_negative_prompt_passes_when_guidance_is_on(self):
        k = self._kwargs("z-image-turbo", catalog.TEXT_TO_IMAGE, cfg_scale=4.0)
        self.assertEqual(k["negative_prompt"], "n")

    def test_a_model_without_negative_prompt_never_receives_one(self):
        """`Flux2KleinPipeline.__call__` ne l'expose pas : le passer lève un
        TypeError après le chargement des poids."""
        k = self._kwargs("flux2-klein-9b", catalog.TEXT_TO_IMAGE, cfg_scale=4.0)
        self.assertNotIn("negative_prompt", k)

    def test_editing_passes_references_without_a_strength(self):
        """Une référence Flux.2 n'est pas un point de départ bruité mais un
        conditionnement : `strength` n'existe pas sur cette classe."""
        k = self._kwargs("flux2-klein-9b", catalog.EDIT,
                         ref_image=[self.a, self.b], strength=0.6)
        self.assertNotIn("strength", k)
        self.assertEqual(len(k["image"]), 2)

    def test_image_to_image_carries_its_strength(self):
        k = self._kwargs("z-image-turbo", catalog.IMAGE_TO_IMAGE,
                         init_image=self.a, strength=0.35)
        self.assertEqual(k["strength"], 0.35)
        self.assertIsNotNone(k["image"])


class ModeFallbackTests(unittest.TestCase):
    def test_an_unsupported_mode_is_announced_before_it_is_dropped(self):
        said = []
        mode = backend._resolve_mode(catalog.get("krea2-turbo"),
                                     init_image="a.png", ref_image=None,
                                     mask_image=None, log=said.append)
        self.assertEqual(mode, catalog.TEXT_TO_IMAGE)
        self.assertTrue(any("ignored" in m for m in said), said)


if __name__ == "__main__":
    unittest.main()


class RoutingTests(unittest.TestCase):
    """Les fonctions qui NE passaient PAS par `generate()`.

    L'outpaint n'est pas testé ici parce qu'il n'a rien à router : il appelle
    déjà `generate()` avec une image de départ et un masque, donc il a changé
    de moteur en même temps que l'aiguille. C'est exactement ce que vaut un
    point de passage unique, et c'est ce qui rendait les trois autres visibles.
    """

    def test_the_mask_question_is_asked_of_the_engine_not_the_binary(self):
        """L'outpaint demandait à `sd-cli` s'il connaissait l'option masque.

        Sur ce moteur il n'y a pas de binaire, et la réponse dépend du MODÈLE :
        diffusers publie une classe d'inpainting pour Z-Image et Flux.2, aucune
        pour Krea 2. Sans cette aiguille l'outpaint retombait en img2img
        partout, y compris là où il pouvait faire mieux.
        """
        from atelier.engine import generate as gen
        self.assertTrue(gen.mask_supported("z-image-turbo"))
        self.assertTrue(gen.mask_supported("flux2-klein-9b"))
        self.assertFalse(gen.mask_supported("krea2-turbo"))

    def test_a_gguf_upscaler_refuses_and_names_the_alternative(self):
        """Refuser est correct — les poids sont GGUF, illisibles hors sd.cpp.
        Refuser sans dire vers quoi aller ne l'est pas."""
        from atelier.engine import generate as gen
        with self.assertRaises(Exception) as caught:
            gen.upscale_image("a.png", "RealESRGAN_x4plus.gguf")
        self.assertIn("modern upscaler", str(caught.exception))

    def test_the_hd_pass_refuses_on_a_model_without_img2img(self):
        """Krea 2 n'a pas de classe img2img : la passe HD ne peut pas exister
        pour lui, et le dire vaut mieux qu'une image inchangée."""
        from atelier.torchengine import ops
        with self.assertRaises(RuntimeError) as caught:
            ops.hd_upscale("krea2-turbo", "a.png")
        self.assertIn("image-to-image", str(caught.exception))

    def test_the_detector_changes_extension_but_not_name(self):
        """Les deux moteurs ne peuvent pas partager le fichier : sd.cpp exige
        un safetensors converti (il refuse d'exécuter un pickle), Ultralytics
        ne lit que le .pt. Mêmes noms, pour que l'interface ne bouge pas."""
        from atelier.torchengine import ops
        self.assertEqual(ops.detector_for("face_yolov8n.safetensors").name,
                         "face_yolov8n.pt")

    def test_a_missing_detector_package_is_named_not_hidden(self):
        from atelier.torchengine import ops
        reason = ops.adetailer_reason()
        self.assertTrue(reason == "" or "ultralytics" in reason
                        or "detector" in reason, reason)


class InstallerTests(unittest.TestCase):
    """Le script d'installation : ses choix se vérifient sans rien installer."""

    def test_blackwell_and_the_rest_do_not_share_an_index(self):
        """cu128 commence à sm_70 : les RTX 50xx en ont besoin et les cartes
        antérieures y disparaissent. cu126 garde 5.0;6.0;7.0 — c'est ce qui
        laisse la GTX 1080 Ti (sm_61, servie par un cubin sm_60) au travail."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                               / "scripts"))
        import setup_torch_engine as ste
        from unittest.mock import patch
        with patch.object(ste, "_gpu_arch", return_value="blackwell"):
            _pkgs, index = ste.torch_args()
            self.assertTrue(index.endswith("cu128"))
        for arch in ("turing", "ampere", "ada", "pascal", "unknown"):
            with patch.object(ste, "_gpu_arch", return_value=arch):
                _pkgs, index = ste.torch_args()
                self.assertTrue(index.endswith("cu126"), arch)

    def test_torch_and_torchvision_are_pinned_together(self):
        """Les mélanger donne un ImportError sur une extension C, pas un
        message lisible."""
        import setup_torch_engine as ste
        for pair in (ste.TORCH_CU126, ste.TORCH_CU128):
            self.assertEqual(len(pair), 2)
            self.assertTrue(all("==" in p for p in pair), pair)

    def test_the_stack_carries_what_the_engine_actually_calls(self):
        """Chaque ligne répond à un appel précis du moteur : peft pour les
        LoRA, bitsandbytes pour les crans int8/NF4, scipy pour le scheduler
        « beta » — qui est une entrée du menu, donc un clic possible."""
        import setup_torch_engine as ste
        joined = " ".join(ste.STACK)
        for pkg in ("transformers", "accelerate", "peft", "bitsandbytes",
                    "scipy"):
            self.assertIn(pkg, joined)


class BenchmarkTests(unittest.TestCase):
    """Le banc d'essai est le seul endroit où « ne rien mesurer » est pire que
    « ne rien proposer » : un rapport a l'air d'avoir mesuré quelque chose."""

    def _gpu(self, vram=12.0, arch="ampere"):
        from atelier import hardware
        return (hardware.Gpu(0, "RTX 3060", vram, arch, True),)

    def _modes(self, vram=12.0, arch="ampere"):
        from unittest.mock import patch
        from atelier import benchmark, hardware
        gpus = self._gpu(vram, arch)
        with patch.object(hardware, "detect_gpus", return_value=gpus):
            return benchmark.placement_candidates(
                {"gpu_index": 0, "engine_backend": "torch"}, gpus)

    def test_the_sdcpp_profiles_are_not_offered_on_the_torch_engine(self):
        """`params_backend`, `split_mode`, auto-fit : des options d'un binaire
        qui ne tourne pas. Les proposer aurait donné trois tirs identiques et
        un classement tiré au sort dans le bruit."""
        keys = [m.key for m in self._modes()]
        self.assertTrue(all(k.startswith("torch-") for k in keys), keys)

    def test_the_second_profile_exists_only_when_it_differs(self):
        """Sur une carte assez grande pour tout tenir en bf16, le plan
        quantifié et le plan brut sont le MÊME plan : deux tirs identiques ne
        départagent rien, donc il n'y a qu'un profil."""
        big = [m.key for m in self._modes(vram=48.0, arch="ada")]
        self.assertEqual(big, ["torch-planned"])
        small = [m.key for m in self._modes(vram=12.0)]
        self.assertEqual(len(small), 2, small)

    def test_the_measured_profile_actually_changes_the_plan(self):
        """Un profil qui n'influence pas l'exécution mesurerait deux fois la
        même chose en affirmant le contraire."""
        from atelier import hardware
        from atelier.torchengine import backend, catalog
        gpu = self._gpu()[0]
        model = catalog.get("z-image-turbo")
        quantized = backend.plan_for(model, gpu, {"torch_allow_quant": True})
        plain = backend.plan_for(model, gpu, {"torch_allow_quant": False})
        self.assertNotEqual((quantized.quant, quantized.mode),
                            (plain.quant, plain.mode))

    def test_the_benchmark_cleanup_does_not_erase_the_measured_setting(self):
        """`_benchmark_prefs` efface le cache, le budget VRAM et le moteur
        résident — c'est voulu. Effacer aussi ce qu'on mesure ne le serait
        pas, et rien dans le rapport ne le dirait."""
        from atelier import benchmark
        for mode in self._modes():
            merged = benchmark._benchmark_prefs({"engine_backend": "torch"},
                                                mode.prefs_patch)
            self.assertIn("torch_allow_quant", merged)


class CancelTests(unittest.TestCase):
    def test_stop_reaches_both_engines(self):
        """Le bouton est appuyé PENDANT une génération. Demander alors quel
        moteur est censé tourner ferait dépendre l'arrêt d'une préférence
        qu'on vient peut-être de changer, et le vrai travail continuerait."""
        from atelier.engine import generate as gen
        from atelier.torchengine import backend as torch_backend
        torch_backend._CANCEL.clear()
        gen.cancel()
        self.assertTrue(torch_backend._CANCEL.is_set())
        torch_backend._CANCEL.clear()


class ReadinessTests(unittest.TestCase):
    def test_ready_means_what_the_active_engine_needs(self):
        """Le pire résultat serait un bouton « Générer » actif qui échoue au
        clic, ou un « à télécharger » sur des fichiers déjà là."""
        from unittest.mock import patch
        from atelier import registry
        model = next(m for m in registry.load_base_models({})
                     if m.id == "z-image-turbo")
        with patch.object(registry, "_torch_engine_active",
                          return_value=True), \
             patch.object(registry, "_torch_repo_present", return_value=True):
            self.assertTrue(registry.model_is_ready(model))
        with patch.object(registry, "_torch_engine_active",
                          return_value=True), \
             patch.object(registry, "_torch_repo_present", return_value=False):
            self.assertFalse(registry.model_is_ready(model))

    def test_a_bare_folder_is_not_a_downloaded_model(self):
        """diffusers écrit l'arborescence d'abord et les poids ensuite : un
        dossier existant est l'état exact d'un téléchargement interrompu."""
        import tempfile
        from unittest.mock import patch
        from atelier import registry, settings
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "Tongyi-MAI__Z-Image-Turbo").mkdir()
            with patch.object(settings, "MODELS_DIR", root), \
                 patch.object(settings, "model_repo_dir",
                              side_effect=lambda r: root / r.replace("/", "__")):
                self.assertFalse(registry._torch_repo_present("z-image-turbo"))
                (root / "Tongyi-MAI__Z-Image-Turbo"
                 / "model_index.json").write_text("{}")
                self.assertTrue(registry._torch_repo_present("z-image-turbo"))
