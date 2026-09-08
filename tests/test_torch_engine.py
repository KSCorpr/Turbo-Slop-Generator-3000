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
            if model.gated and not model.from_gguf:
                self.assertIsNone(model.full_repo_gb, model.id)

    def test_the_normal_path_reuses_the_files_already_installed(self):
        """LE point de la correction. Passer par PyTorch ne doit rien coûter
        de plus sur le disque : diffusers lit le GGUF, et lit celui que
        l'application a déjà téléchargé pour stable-diffusion.cpp.

        Le lien entre les deux catalogues est le RÔLE du composant. S'il ne
        correspond plus, le montage échoue au chargement — après vingt
        secondes — au lieu d'ici.
        """
        from atelier import registry
        for model in catalog.load():
            if not model.from_gguf:
                continue
            base = registry.get_base_model(model.id, {})
            declared = {c.role for c in base.components}
            for part in model.parts.values():
                self.assertIn(part.role, declared,
                              f"{model.id}: role “{part.role}” is not in "
                              "models.yaml")

    def test_every_transformer_now_comes_from_a_file_on_disk(self):
        """Y COMPRIS Krea 2, dont diffusers n'enregistre pas le chargeur : la
        correspondance de noms s'écrit, et elle est exacte. Ce qui bloque
        encore ce modèle est ailleurs, et nommé pièce par pièce."""
        self.assertEqual([m.id for m in catalog.load() if not m.from_gguf],
                         [])

    def test_a_token_is_asked_for_the_right_reason(self):
        """Quelques kilo-octets de configuration et trente gigaoctets de poids
        ne se demandent pas de la même façon."""
        z = catalog.get("z-image-turbo")
        self.assertEqual(z.needs_token, "", "Z-Image needs nothing")
        flux = catalog.get("flux2-klein-9b")
        self.assertIn("config", flux.needs_token)
        krea = catalog.get("krea2-turbo")
        self.assertIn("pipeline settings", krea.needs_token)
        self.assertNotIn("weights", krea.needs_token,
                         "saying “weights” would send someone to re-download "
                         "13 GB they already have")

    def test_the_requested_mode_is_computed_before_it_is_granted(self):
        krea = catalog.get("krea2-turbo")
        self.assertEqual(
            catalog.mode_for(krea, init_image="a.png"),
            catalog.IMAGE_TO_IMAGE,
            "the request must be readable even when the model cannot serve it")


class PlacementFromFilesTests(unittest.TestCase):
    """Le chemin normal : les poids arrivent quantifiés, du disque.

    Les tailles ci-dessous sont celles des vrais fichiers que l'application
    installe pour Z-Image Turbo sur une carte de 12 Go — diffusion Q8_0,
    encodeur Q4_K_M, VAE en safetensors. Ce sont des mesures, pas un calcul en
    octets par paramètre : un GGUF « _K_M » mélange les précisions par couche,
    donc l'estimation se trompe précisément là où elle compte.
    """
    Z = {"transformer": 6.58, "text_encoder": 2.50, "vae": 0.34}

    def test_a_twelve_gigabyte_card_keeps_everything_resident(self):
        p = placement.plan_from_files(12.0, self.Z, "ampere")
        self.assertEqual(p.mode, placement.FULL)
        self.assertEqual(p.quant, "gguf")

    def test_eleven_gigabytes_falls_back_to_taking_turns_not_to_streaming(self):
        """La 2080 Ti ne tient pas les 9,4 Go d'un bloc avec la réserve, mais
        son plus gros morceau (6,6) passe largement. Le streaming couche par
        couche coûterait un facteur dix pour rien."""
        p = placement.plan_from_files(11.0, self.Z, "turing")
        self.assertEqual(p.mode, placement.MODEL_OFFLOAD)

    def test_a_small_card_says_which_knob_to_turn(self):
        """Sur 8 Go rien ne passe. Dire « c'est lent » sans dire qu'un cran de
        quantification en dessous existe, c'est laisser l'utilisateur dans le
        pire réglage en croyant qu'il n'y en a pas d'autre."""
        p = placement.plan_from_files(8.0, self.Z, "ampere")
        self.assertEqual(p.mode, placement.SEQUENTIAL_OFFLOAD)
        self.assertIn("quantization rung", p.reason)

    def test_a_missing_file_never_buys_the_fast_path(self):
        """Un composant pas encore téléchargé pèse 0 : calculer dessus ferait
        tenir n'importe quoi n'importe où."""
        holes = {**self.Z, "text_encoder": 0.0}
        p = placement.plan_from_files(12.0, holes, "ampere")
        self.assertEqual(p.mode, placement.MODEL_OFFLOAD)

    def test_the_reason_names_the_parts_and_their_weight(self):
        line = placement.describe(
            placement.plan_from_files(11.0, self.Z, "turing"))
        self.assertIn("transformer 6.6", line)
        self.assertIn("GGUF", line)

    def test_forcing_a_placement_is_visible_in_the_reason(self):
        """Le banc d'essai contourne la réserve de calcul le temps d'une
        mesure ; un rapport qui ne le dirait pas serait faux."""
        chosen = placement.plan_from_files(11.0, self.Z, "turing")
        forced = placement.forced(chosen, placement.FULL)
        self.assertEqual(forced.mode, placement.FULL)
        self.assertIn("forced", forced.reason)
        self.assertIs(placement.forced(chosen, ""), chosen)


class PlacementFromRepoTests(unittest.TestCase):
    """L'ancien chemin, qui ne sert plus qu'à Krea 2."""
    Z = (20.5, 12.3)

    def test_precision_is_only_traded_to_escape_layer_streaming(self):
        p = placement.plan(16.0, *self.Z, "ada")
        self.assertEqual(p.quant, "none")
        self.assertEqual(p.mode, placement.MODEL_OFFLOAD)

    def test_a_big_card_keeps_full_precision_and_stays_resident(self):
        p = placement.plan(24.0, *self.Z, "ada")
        self.assertEqual(p.mode, placement.FULL)
        self.assertEqual(p.quant, "none")

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

    def _modes(self, vram=12.0, arch="ampere", sizes=None):
        from unittest.mock import patch
        from atelier import benchmark, hardware
        from atelier.torchengine import backend
        gpus = self._gpu(vram, arch)
        with patch.object(hardware, "detect_gpus", return_value=gpus), \
             patch.object(backend, "component_files", return_value={}), \
             patch.object(backend, "sizes_gb",
                          return_value=sizes if sizes is not None else
                          {"transformer": 6.58, "text_encoder": 2.50,
                           "vae": 0.34}):
            return benchmark.placement_candidates(
                {"gpu_index": 0, "engine_backend": "torch"}, gpus)

    def test_the_sdcpp_profiles_are_not_offered_on_the_torch_engine(self):
        """`params_backend`, `split_mode`, auto-fit : des options d'un binaire
        qui ne tourne pas. Les proposer aurait donné des tirs identiques et un
        classement tiré au sort dans le bruit."""
        keys = [m.key for m in self._modes()]
        self.assertTrue(all(k.startswith("torch-") for k in keys), keys)

    def test_precision_is_no_longer_a_question_to_measure(self):
        """Elle a été tranchée au TÉLÉCHARGEMENT, par l'échelle du catalogue
        principal. La proposer ici mesurerait un réglage qui n'existe plus."""
        for mode in self._modes():
            self.assertNotIn("torch_allow_quant", mode.prefs_patch)

    def test_the_second_profile_exists_only_when_it_differs(self):
        """Quand le plan tient déjà tout sur la carte, forcer « tout sur la
        carte » est le MÊME tir, et deux tirs identiques ne départagent rien."""
        resident = [m.key for m in self._modes(vram=24.0, arch="ada")]
        self.assertEqual(resident, ["torch-planned"])
        tight = [m.key for m in self._modes(vram=11.0, arch="turing")]
        self.assertEqual(tight, ["torch-planned", "torch-all-on-card"])

    def test_the_measured_profile_actually_changes_the_plan(self):
        """Un profil qui n'influence pas l'exécution mesurerait deux fois la
        même chose en affirmant le contraire.

        Ce qui reste ouvert sur ce moteur est la RÉSERVE DE CALCUL : 2,5 Gio
        estimés pour le contexte CUDA, les tampons d'attention et les latents.
        Elle décide seule entre « tout sur la carte » et « à tour de rôle », et
        elle n'a aucune raison d'être juste sur toutes les machines.
        """
        from atelier.torchengine import placement
        sizes = {"transformer": 6.58, "text_encoder": 2.50, "vae": 0.34}
        chosen = placement.plan_from_files(11.0, sizes, "turing")
        forced = placement.forced(chosen, placement.FULL)
        self.assertNotEqual(chosen.mode, forced.mode)

    def test_the_benchmark_cleanup_does_not_erase_the_measured_setting(self):
        """`_benchmark_prefs` efface le cache, le budget VRAM et le moteur
        résident — c'est voulu. Effacer aussi ce qu'on mesure ne le serait
        pas, et rien dans le rapport ne le dirait."""
        from atelier import benchmark
        for mode in self._modes(vram=11.0, arch="turing"):
            merged = benchmark._benchmark_prefs({"engine_backend": "torch"},
                                                mode.prefs_patch)
            self.assertIn("torch_force_placement", merged)


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
    def test_ready_means_the_same_thing_on_both_engines_for_a_gguf_model(self):
        """La conséquence la plus visible de la correction : un modèle installé
        l'est pour les deux moteurs. Répondre autrement donnerait un « à
        télécharger » sur des fichiers déjà là — 6,6 Go retéléchargés pour
        rien."""
        from unittest.mock import patch
        from atelier import registry
        model = next(m for m in registry.load_base_models({})
                     if m.id == "z-image-turbo")
        with patch.object(registry, "_torch_engine_active",
                          return_value=True), \
             patch.object(registry, "resolve_component_path",
                          return_value=pathlib.Path("x")):
            self.assertTrue(registry.model_is_ready(model))
        with patch.object(registry, "_torch_engine_active",
                          return_value=True), \
             patch.object(registry, "resolve_component_path",
                          return_value=None):
            self.assertFalse(registry.model_is_ready(model))

    def test_a_gguf_model_defers_to_the_native_answer(self):
        from atelier import registry
        self.assertIsNone(registry._torch_repo_present("z-image-turbo"),
                          "a GGUF model must defer to the native answer")

    def test_a_blocked_model_is_never_reported_ready(self):
        """Des fichiers présents ne font pas un modèle utilisable. Répondre
        « prêt » donnerait un bouton « Générer » actif qui refuse au clic —
        exactement ce qu'on reproche à une interface."""
        from atelier import registry
        self.assertIs(registry._torch_repo_present("krea2-turbo"), False)


class HuggingFaceAccessTests(unittest.TestCase):
    """« Qu'est-ce qu'il me manque exactement ? » — trois réponses, pas une.

    Un dépôt fermé échoue de trois façons qui se ressemblent à l'écran et
    n'appellent pas du tout la même action : pas de jeton, jeton valide mais
    licence non acceptée, ou tout va bien. Le message brut de `huggingface_hub`
    est un 401 dans les deux premiers cas.
    """

    def _probe(self, responses):
        """Remplace la couche HTTP par une table code-par-URL."""
        from unittest.mock import patch
        from atelier import hfaccess

        def fake(url, tok):
            for fragment, code in responses.items():
                if fragment in url:
                    return code, b"{}"
            return 404, b""
        return patch.object(hfaccess, "_get", side_effect=fake)

    def test_the_model_api_is_not_the_right_question(self):
        """Le piège qui a fait passer la première version du test.

        Sur un dépôt fermé, `/api/models/<id>` répond **200 sans jeton** — les
        métadonnées publiques sortent toujours. Seul `/raw/main/<fichier>`
        répond 401. Sonder l'API rendait donc « accès accordé » à quelqu'un qui
        n'a aucun compte, ce qui est le pire diagnostic possible : il envoie
        chercher la panne ailleurs.
        """
        from atelier import hfaccess
        with self._probe({"/api/models/": 200, "/raw/main/": 401}), \
             patch_token(""):
            access = hfaccess.check("acme/closed", "transformer/config.json")
        self.assertFalse(access.ok)
        self.assertEqual(access.state, "needs_token")

    def test_401_and_403_are_not_the_same_answer(self):
        """La distinction la plus utile du module, et celle que je n'avais pas
        faite. 401 : le Hub ne sait pas qui vous êtes. 403 : il le sait très
        bien et refuse quand même. Traiter les deux pareil envoie accepter une
        licence déjà acceptée — une heure perdue à chercher au mauvais
        endroit.
        """
        from atelier import hfaccess
        with self._probe({"/raw/main/": 401}), patch_token("hf_xxx"):
            self.assertEqual(hfaccess.check("acme/closed").state,
                             "needs_token")
        with self._probe({"/raw/main/": 403}), patch_token("hf_xxx"):
            self.assertEqual(hfaccess.check("acme/closed").state, "forbidden")

    def test_a_forbidden_answer_names_the_token_scope_too(self):
        """Le piège des jetons « fine-grained » : ils s'authentifient
        parfaitement — `whoami` répond — et il leur manque une case à cocher
        que rien sur la page du modèle ne concerne."""
        from atelier import hfaccess
        with self._probe({"/raw/main/": 403}), patch_token("hf_xxx"):
            detail = hfaccess.check("acme/closed").detail
        self.assertIn("licence", detail)
        self.assertIn("fine-grained", detail)

    def test_the_report_orders_the_two_causes(self):
        from atelier import hfaccess
        with self._probe({"/api/whoami-v2": 200, "/raw/main/": 403}), \
             patch_token("hf_xxx"):
            text = hfaccess.report()
        self.assertIn("token's scope", text)
        self.assertLess(text.index("1. the licence"), text.index("2. the"))

    def test_an_open_repository_says_there_is_nothing_to_do(self):
        from atelier import hfaccess
        with self._probe({"/raw/main/": 200}), patch_token(""):
            access = hfaccess.check("acme/open")
        self.assertTrue(access.ok)
        self.assertEqual(access.state, "open")

    def test_the_report_names_the_page_to_click(self):
        from atelier import hfaccess
        with self._probe({"/api/whoami-v2": 401, "/raw/main/": 401}), \
             patch_token(""):
            text = hfaccess.report()
        self.assertIn("huggingface.co/black-forest-labs/FLUX.2-klein-9B", text)
        self.assertIn("read", text)

    def test_it_says_the_weights_are_not_what_is_missing(self):
        """La distinction qui évite un retéléchargement de 6,6 Go par erreur."""
        from atelier import hfaccess
        entries = {m: why for m, _r, _p, why in hfaccess.gated_repos()}
        self.assertIn("already have", entries["flux2-klein-9b"])
        self.assertIn("transformer already loads", entries["krea2-turbo"])

    def test_the_settings_field_wins_over_a_stale_environment(self):
        """Coller un nouveau jeton doit agir tout de suite. Avec `setdefault`,
        une valeur laissée par la session précédente l'aurait emporté et le
        bouton de vérification aurait démenti ce qu'on venait de saisir."""
        import os
        from unittest.mock import patch
        from atelier import settings
        with patch.dict(os.environ, {"HF_TOKEN": "old"}), \
             patch.object(settings, "load_prefs",
                          return_value={"hf_token": "new"}):
            settings.configure_hf_env()
            self.assertEqual(os.environ["HF_TOKEN"], "new")

    def test_an_empty_field_leaves_an_external_token_alone(self):
        """Quelqu'un qui exporte HF_TOKEN lui-même garde la main."""
        import os
        from unittest.mock import patch
        from atelier import settings
        with patch.dict(os.environ, {"HF_TOKEN": "mine"}), \
             patch.object(settings, "load_prefs", return_value={"hf_token": ""}):
            settings.configure_hf_env()
            self.assertEqual(os.environ["HF_TOKEN"], "mine")


def patch_token(value: str):
    from unittest.mock import patch
    from atelier import hfaccess
    return patch.object(hfaccess, "token", return_value=value)


class Krea2GgufMappingTests(unittest.TestCase):
    """Le convertisseur écrit à la main — vérifié exhaustivement, hors ligne.

    C'est le seul endroit de cette branche où l'on écrit une correspondance de
    poids soi-même, et c'est aussi le seul type d'erreur qui ne se voit pas :
    un tenseur mal placé ne lève rien, il dégrade l'image. La seule parade est
    de tout vérifier — que chaque clé attendue reçoit exactement un tenseur, de
    la bonne taille, et qu'aucun tenseur ne disparaît en silence.

    Le relevé (`tests/fixtures/krea2_tensor_names.json`) vient du VRAI fichier :
    en-tête GGUF de `krea2_turbo-Q5_K_M.gguf` lu par requête partielle, et
    `state_dict` de la classe diffusers. Le conserver rend le test reproductible
    sans réseau et sans les treize gigaoctets.
    """

    @classmethod
    def setUpClass(cls):
        import json
        path = (pathlib.Path(__file__).parent / "fixtures"
                / "krea2_tensor_names.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        cls.gguf = data["gguf_tensors"]
        cls.want = data["diffusers_keys"]

    def _mapping(self):
        from atelier.torchengine import krea2_gguf as K
        mapped, dropped, unmapped = {}, [], []
        for name, dims in self.gguf:
            target = K.convert_key(name)
            if target is None:
                (dropped if name in K.EASTER_EGG else unmapped).append(name)
                continue
            # Les dimensions GGUF sont dans l'ordre inverse de torch.
            mapped.setdefault(target, []).append((name, list(reversed(dims))))
        return mapped, dropped, unmapped

    def test_every_expected_key_is_produced_exactly_once(self):
        mapped, _dropped, unmapped = self._mapping()
        self.assertEqual(unmapped, [], "tensors the converter does not know")
        self.assertEqual(sorted(k for k in self.want if k not in mapped), [],
                         "keys diffusers wants and the converter never makes")
        self.assertEqual(sorted(k for k in mapped if k not in self.want), [],
                         "keys the converter invents")
        self.assertEqual([k for k, v in mapped.items() if len(v) > 1], [],
                         "two tensors landing on the same key")

    def test_every_shape_matches(self):
        """Une correspondance de NOMS peut être juste et le contenu faux. Les
        formes sont la seule vérification disponible sans les poids."""
        import math
        mapped, _d, _u = self._mapping()
        wrong = [(v[0][0], v[0][1], k, self.want[k])
                 for k, v in mapped.items()
                 if math.prod(v[0][1]) != math.prod(self.want[k])]
        self.assertEqual(wrong, [])

    def test_the_two_orphans_are_dropped_on_purpose_and_named(self):
        """`last.up` / `last.down` : deux matrices 6144×6144 qu'aucun des deux
        moteurs ne contient — ni `Krea2FinalLayer` chez diffusers, ni
        `KreaLastLayer` chez stable-diffusion.cpp. Les métadonnées du fichier
        (`egg_w`, `egg_h`, `egg_c`, `egg_format`) disent ce que c'est : une
        image cachée par celui qui a empaqueté le GGUF."""
        from atelier.torchengine import krea2_gguf as K
        _m, dropped, _u = self._mapping()
        self.assertEqual(sorted(dropped), sorted(K.EASTER_EGG))

    def test_an_unknown_tensor_raises_instead_of_vanishing(self):
        """Le seul défaut qu'on ne verrait jamais : un modèle qui charge sans
        erreur et rend des images subtilement fausses."""
        from atelier.torchengine import krea2_gguf as K
        with self.assertRaises(ValueError) as caught:
            K.convert_state_dict({"blocks.0.attn.wq.weight": _Fake(),
                                  "something.unexpected": _Fake()})
        self.assertIn("unexpected", str(caught.exception))

    def test_the_flat_modulation_table_is_reshaped(self):
        """Le GGUF la stocke à plat (36864), diffusers l'attend en 6 × 6144."""
        from atelier.torchengine import krea2_gguf as K
        out = K.convert_state_dict({"blocks.0.mod.lin": _Fake((36864,))})
        table = out["transformer_blocks.0.scale_shift_table"]
        self.assertEqual(table.shape, (6, 6144))

    def test_registration_never_overwrites_an_upstream_loader(self):
        """Le jour où diffusers publie le sien, le nôtre doit s'effacer : deux
        correspondances pour un même modèle finiraient par diverger, et c'est
        la nôtre qui aurait tort."""
        _diffusers_or_skip()
        from unittest.mock import patch
        import diffusers
        from atelier.torchengine import krea2_gguf as K
        from diffusers.loaders import single_file_model as sfm
        with patch.dict(sfm.SINGLE_FILE_LOADABLE_CLASSES,
                        {"Krea2Transformer2DModel": {"checkpoint_mapping_fn":
                                                     lambda c, **k: c}}):
            self.assertFalse(K.register(diffusers))


class _Fake:
    """Un tenseur juste assez réel pour la conversion de noms."""

    def __init__(self, shape=(1,)):
        self.shape = shape
        self.ndim = len(shape)

    def reshape(self, *dims):
        return _Fake(tuple(dims))


class Krea2AvailabilityTests(unittest.TestCase):
    def test_the_blocker_is_named_per_part_not_per_model(self):
        """« Télécharge le modèle entier » envoyait chercher treize
        gigaoctets déjà présents. Ce qui bloque est ailleurs, et pèse
        autrement moins."""
        krea = catalog.get("krea2-turbo")
        self.assertTrue(krea.from_gguf, "its transformer does load from GGUF")
        self.assertFalse(krea.usable)
        parts = {b.part for b in krea.blocked_by}
        self.assertEqual(parts, {"text_encoder", "vae", "pipeline_config"})

    def test_generation_refuses_before_loading_anything(self):
        """Vingt secondes de chargement pour une pile d'appels, ce n'est pas
        un message d'erreur."""
        from atelier.torchengine import backend
        with self.assertRaises(Exception) as caught:
            backend.generate("krea2-turbo", "p", "", 8, 1.0, 1024, 1024, 1, 1,
                             prefs_override={})
        text = str(caught.exception)
        self.assertIn("text_encoder", text)
        self.assertIn("qwen3vl", text)

    def test_the_other_two_models_stay_usable(self):
        for model_id in ("z-image-turbo", "flux2-klein-9b"):
            self.assertTrue(catalog.get(model_id).usable, model_id)
