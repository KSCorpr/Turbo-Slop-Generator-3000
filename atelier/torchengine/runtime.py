"""Le seul module qui importe torch — et il le fait le plus tard possible.

Deux raisons, et aucune n'est esthétique.

**Le démarrage.** Importer torch coûte plusieurs secondes et quelques centaines
de mégaoctets. L'application ouvre son interface, liste des fichiers et affiche
un catalogue sans avoir besoin d'un seul tenseur ; payer ce prix au lancement le
ferait payer aussi à qui vient juste regarder ses images.

**Les tests.** Tout ce qui DÉCIDE (précision, placement, échantillonneur,
résolution) vit dans des modules sans torch. Ici on ne fait qu'exécuter la
décision. C'est ce qui permet de vérifier le raisonnement sur une machine sans
carte graphique, au lieu de le découvrir en panne.

Le pipeline chargé est GARDÉ en mémoire entre deux images, comme le moteur
résident du côté sd.cpp : recharger 20 Go pour changer une graine n'aurait
aucun sens. Il est libéré dès qu'on change de modèle, de mode ou de placement,
et par `release()` quand un autre outil réclame la carte.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .placement import FULL, MODEL_OFFLOAD, SEQUENTIAL_OFFLOAD, Plan


class TorchEngineError(RuntimeError):
    """Panne côté moteur PyTorch, formulée pour être lue par l'utilisateur."""


_INSTALL_HINT = (
    "The PyTorch engine is not installed. Run setup-torch-engine.bat — it "
    "fetches torch, diffusers and their dependencies into the app's own "
    "Python (several gigabytes, once)."
)


def _import(name: str):
    try:
        return __import__(name, fromlist=["_"])
    except ImportError as exc:  # noqa: BLE001
        raise TorchEngineError(f"{_INSTALL_HINT}\n({name}: {exc})") from exc


def torch():
    return _import("torch")


_KREA_REGISTERED = False


def diffusers():
    """diffusers, avec le chargeur Krea 2 branché au premier appel.

    Le branchement est fait ICI et pas à l'import du module : il touche une
    table interne de diffusers, donc il ne doit exister que dans les sessions
    qui se servent réellement du moteur. Il s'efface tout seul le jour où
    l'amont publie son propre chargeur — voir `krea2_gguf.register`.
    """
    global _KREA_REGISTERED
    module = _import("diffusers")
    if not _KREA_REGISTERED:
        _KREA_REGISTERED = True
        try:
            from . import krea2_gguf
            krea2_gguf.register(module)
        except Exception:  # noqa: BLE001
            # Un échec de branchement ne doit pas empêcher les deux autres
            # modèles de tourner : ils n'en dépendent pas.
            pass
    return module


#  Le socle MINIMAL pour qu'un pipeline se charge. Aucun de ces quatre n'est un
#  extra : les modèles ont un encodeur de texte Qwen (transformers), leurs
#  poids arrivent en GGUF (gguf — sans lui `GGUFQuantizationConfig` refuse), et
#  aucun ne tient sur 11-12 Go sans décharge (accelerate, que le quantiseur
#  GGUF exige d'ailleurs explicitement). Sans eux, `available()` dirait
#  « installé » et le premier clic rendrait un ImportError venu du fond de
#  diffusers.
REQUIRED = ("torch", "diffusers", "gguf", "transformers", "accelerate")


def available() -> bool:
    """Le socle est-il installé — sans l'importer pour de bon ?

    `find_spec` regarde les métadonnées d'installation ; c'est ce qui permet à
    l'interface de griser un bouton sans embarquer une seconde de chargement à
    chaque rafraîchissement.
    """
    from importlib.util import find_spec
    try:
        return all(find_spec(name) for name in REQUIRED)
    except (ImportError, ValueError):
        return False


def missing() -> list[str]:
    """Ce qui manque, nommé. « Ça ne marche pas » n'est pas un message."""
    from importlib.util import find_spec
    out = []
    for name in REQUIRED:
        try:
            if find_spec(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


def versions() -> dict[str, str]:
    """Ce qui est réellement installé — pour le diagnostic, pas pour décider."""
    out: dict[str, str] = {}
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("torch", "diffusers", "transformers", "accelerate",
                "bitsandbytes", "peft", "safetensors"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = ""
    return out


def cuda_report() -> str:
    """Une phrase sur ce que torch voit du matériel. Importe torch : à réserver
    au diagnostic, jamais au chemin d'affichage."""
    t = torch()
    if not t.cuda.is_available():
        return "torch sees no CUDA device — generation would run on the CPU."
    names = [t.cuda.get_device_name(i) for i in range(t.cuda.device_count())]
    return (f"torch {t.__version__} · CUDA {t.version.cuda} · "
            + ", ".join(names))


# --------------------------------------------------------------------------- #
#  Le pipeline résident
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoadKey:
    """Ce qui, en changeant, oblige à tout recharger.

    Les LoRA n'en font PAS partie : diffusers sait les greffer et les retirer
    sur un pipeline déjà chargé, et refaire 20 Go de chargement pour changer un
    poids serait absurde. Tout le reste touche à la construction elle-même.
    """
    model_id: str
    mode: str
    cls: str
    dtype: str
    quant: str
    placement: str
    gpu_index: int | None


class _Resident:
    """Le pipeline gardé chaud, et le verrou qui empêche deux générations.

    Le verrou n'est pas de la prudence : deux appels simultanés sur le même
    pipeline se marchent dessus au niveau du scheduler (il porte l'état des pas)
    et le second rend une image à moitié bruitée, sans erreur.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.key: LoadKey | None = None
        self.pipe: Any = None
        self.loras: tuple = ()

    def release(self, why: str = "", log: Callable | None = None) -> None:
        if self.pipe is None:
            return
        if log:
            log(f"[torch] releasing the pipeline{f' ({why})' if why else ''}.")
        self.pipe = None
        self.key = None
        self.loras = ()
        gc.collect()
        try:
            torch().cuda.empty_cache()
        except TorchEngineError:
            pass


_RESIDENT = _Resident()


def release(why: str = "", log: Callable | None = None) -> None:
    """Rend la VRAM. Appelé par les autres outils avant de prendre la carte."""
    with _RESIDENT.lock:
        _RESIDENT.release(why, log)


def is_loaded() -> bool:
    return _RESIDENT.pipe is not None


# --------------------------------------------------------------------------- #
#  Construction
# --------------------------------------------------------------------------- #
def _quant_config(quant: str, dtype):
    """La configuration bitsandbytes correspondant au cran choisi.

    `PipelineQuantizationConfig` applique la quantification AU CHARGEMENT,
    module par module. C'est l'équivalent le plus proche du GGUF : les poids
    n'existent jamais en pleine précision en mémoire, donc un modèle trop gros
    pour la carte le reste pas.

    Le VAE est laissé INTACT. Il est minuscule (200 Mo) et c'est lui qui décide
    du grain final : le quantifier ne libère rien et se voit tout de suite.
    """
    if quant == "none":
        return None
    d = diffusers()
    if quant == "nf4":
        kwargs = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": dtype,
                  "bnb_4bit_use_double_quant": True}
    elif quant == "int8":
        kwargs = {"load_in_8bit": True}
    else:
        raise TorchEngineError(f"Unknown quantization tier: {quant}")
    return d.PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit" if quant == "nf4"
        else "bitsandbytes_8bit",
        quant_kwargs=kwargs,
        components_to_quantize=["transformer", "text_encoder"])


def _apply_placement(pipe, plan: Plan, device: str, log=None) -> None:
    """Traduit le plan en appels diffusers, dans l'ordre qui compte.

    L'ordre EST le piège : `enable_*_cpu_offload` installe des points d'entrée
    sur les modules et doit être appelé APRÈS toute quantification et AVANT
    tout `.to(device)`. Un `.to("cuda")` posé ensuite annule la décharge en
    remontant tout sur la carte — le symptôme étant un OOM sur une
    configuration dont le journal vient d'affirmer qu'elle tenait.
    """
    if plan.mode == SEQUENTIAL_OFFLOAD:
        pipe.enable_sequential_cpu_offload()
    elif plan.mode == MODEL_OFFLOAD:
        pipe.enable_model_cpu_offload()
    elif plan.mode == FULL:
        pipe.to(device)

    if plan.vae_tiling and hasattr(pipe, "vae"):
        for enable in ("enable_tiling", "enable_slicing"):
            fn = getattr(pipe.vae, enable, None)
            if callable(fn):
                fn()
    if plan.attention_slicing:
        fn = getattr(pipe, "enable_attention_slicing", None)
        if callable(fn):
            fn()
    if log:
        log(f"[torch] placement applied: {plan.mode}.")


def _pipeline_class(name: str):
    d = diffusers()
    cls = getattr(d, name, None)
    if cls is None:
        raise TorchEngineError(
            f"The installed diffusers ({getattr(d, '__version__', '?')}) has "
            f"no “{name}”. Run setup-torch-engine.bat again to update it.")
    return cls


def _model_class(name: str):
    """La classe d'un composant, chez diffusers OU chez transformers.

    Un pipeline mélange les deux — le transformer et le VAE viennent de
    diffusers, l'encodeur de texte de transformers — et rien dans le nom ne le
    dit. On regarde donc les deux plutôt que de coder la répartition, qui
    changerait au premier modèle dont l'encodeur n'est pas un Qwen.
    """
    d = diffusers()
    cls = getattr(d, name, None)
    if cls is not None:
        return cls
    tr = _import("transformers")
    cls = getattr(tr, name, None)
    if cls is None:
        raise TorchEngineError(
            f"Neither diffusers nor transformers provides “{name}”. Run "
            "setup-torch-engine.bat again to update them.")
    return cls


def build_hybrid(model, pipeline_cls: str, dtype, files: dict,
                 log: Callable | None = None):
    """Transformer depuis le GGUF, le reste depuis le complément téléchargé.

    Le cas de Krea 2, et le seul. `from_pretrained` sait qu'un composant passé
    en argument ne doit pas être relu du disque : c'est ce qui permet de ne
    JAMAIS télécharger `transformer/` (26,3 Go) tout en gardant son
    `config.json` (588 octets), que le chargeur fichier-unique réclame pour
    savoir quelle forme donner aux poids.
    """
    d = diffusers()
    local = model.supplement.local_dir
    if not model.supplement.present:
        raise TorchEngineError(
            f"“{model.id}” also needs its text encoder and VAE, which cannot "
            "come from GGUF. Download it from the Model catalog tab "
            f"(~{model.supplement.download_gb:.0f} GB — the transformer is "
            "skipped, it is already on your disk).")

    part = model.parts["transformer"]
    path = files.get(part.role)
    if path is None:
        raise TorchEngineError(
            f"“{model.id}”: the {part.role} file is missing. Download the "
            "model from the Model catalog tab.")
    if log:
        log(f"[torch] transformer: {Path(path).name} (GGUF) · everything else "
            f"from {local.name}")
    transformer = _model_class(part.cls).from_single_file(
        str(path), quantization_config=d.GGUFQuantizationConfig(
            compute_dtype=dtype),
        torch_dtype=dtype, config=str(local))
    return _pipeline_class(pipeline_cls).from_pretrained(
        str(local), transformer=transformer, torch_dtype=dtype)


def build_from_files(model, pipeline_cls: str, dtype, files: dict,
                     log: Callable | None = None):
    """Monte le pipeline à partir des fichiers DÉJÀ INSTALLÉS.

    C'est le chemin normal de cette branche, et c'est celui qui la justifie :
    les poids sont ceux que l'application a téléchargés pour
    stable-diffusion.cpp — mêmes fichiers, même échelle de quantification,
    aucune place supplémentaire sur le disque.

    Ce qui ne vient PAS du disque tient en quelques mégaoctets : la
    configuration d'architecture, le tokeniseur et le scheduler, pris dans le
    dépôt de référence du modèle. On aurait pu reconstruire le tokeniseur
    depuis les métadonnées du GGUF — transformers sait le faire — mais un
    tokeniseur reconstruit peut différer sur des détails qu'on ne verrait
    qu'au rendu ; celui d'origine ne peut pas.
    """
    d = diffusers()
    gguf_cfg = d.GGUFQuantizationConfig(compute_dtype=dtype)
    components: dict[str, Any] = {}

    for name, part in model.parts.items():
        path = files.get(part.role)
        if path is None:
            raise TorchEngineError(
                f"“{model.id}”: the {part.role} file is missing. Download the "
                "model from the Model catalog tab.")
        cls = _model_class(part.cls)
        is_gguf = str(path).lower().endswith(".gguf")
        if log:
            log(f"[torch] {name}: {Path(path).name}"
                + (" (GGUF, dequantized on the fly)" if is_gguf else ""))
        if name == "text_encoder":
            # transformers, pas diffusers : l'encodeur se charge depuis le
            # DOSSIER du dépôt plus le nom du fichier, et transformers
            # reconstruit la configuration à partir des métadonnées du GGUF
            # (`general.architecture = qwen3`, le nombre de couches, la
            # largeur…). C'est pour ça qu'aucun `config.json` n'est requis ici.
            components[name] = _load_text_encoder(cls, path, dtype, log)
        elif is_gguf:
            components[name] = cls.from_single_file(
                str(path), quantization_config=gguf_cfg, torch_dtype=dtype,
                **_config_kwargs(model))
        else:
            components[name] = cls.from_single_file(
                str(path), torch_dtype=dtype, **_config_kwargs(model))

    meta = model.config_repo or model.repo
    tr = _import("transformers")
    try:
        components["tokenizer"] = tr.AutoTokenizer.from_pretrained(
            meta, subfolder="tokenizer")
    except Exception as exc:  # noqa: BLE001
        raise _metadata_error(model, meta, exc) from exc
    components["scheduler"] = _load_scheduler(meta, log)
    return _pipeline_class(pipeline_cls)(**components)


def _metadata_error(model, repo: str, exc: Exception) -> TorchEngineError:
    """Traduire un refus du Hub en quelque chose d'actionnable.

    L'erreur brute est un 401 sur une URL, ce qui laisse croire que le modèle
    est absent. Il ne l'est pas : ses POIDS sont là, sur le disque. Ce qui
    manque tient en quelques kilo-octets, et il y a deux raisons possibles —
    pas de jeton, ou licence non acceptée — qui n'appellent pas la même action.
    """
    text = str(exc)
    if not any(m in text for m in ("401", "403", "gated", "Unauthorized",
                                   "restricted")):
        return TorchEngineError(
            f"Could not read the metadata of “{model.id}” from {repo}: {exc}")
    return TorchEngineError(
        f"“{model.id}”: its weights are on your disk, but its architecture "
        f"metadata lives in a gated repository ({repo}).\n"
        "→ Settings → Theme and accounts: paste a Hugging Face **read** "
        "token, accept the licence on that model's page, then use "
        "“🔍 Check what is still missing” to confirm.")


def _config_kwargs(model) -> dict:
    """`config=` seulement quand le repli automatique de diffusers est faux.

    Pour Z-Image, diffusers retrouve tout seul `Tongyi-MAI/Z-Image-Turbo`, qui
    est le bon dépôt et il est ouvert. Pour Flux.2 Klein son repli pointe vers
    `black-forest-labs/FLUX.2-dev` — un AUTRE modèle — donc il faut le nommer.
    """
    return {"config": model.config_repo} if model.config_repo else {}


def _load_text_encoder(cls, path: Path, dtype, log=None):
    from pathlib import Path as _P
    p = _P(path)
    if p.suffix.lower() != ".gguf":
        return cls.from_pretrained(str(p.parent), torch_dtype=dtype)
    return cls.from_pretrained(str(p.parent), gguf_file=p.name,
                               torch_dtype=dtype)


def _load_scheduler(meta: str, log=None):
    d = diffusers()
    try:
        return d.FlowMatchEulerDiscreteScheduler.from_pretrained(
            meta, subfolder="scheduler")
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"[torch] scheduler config unavailable ({exc}) — using the "
                "library default, which may not carry this model's shift.")
        return d.FlowMatchEulerDiscreteScheduler()


def load_pipeline(key: LoadKey, plan: Plan, source: str | Path,
                  log: Callable | None = None, model=None, files=None):
    """Charge (ou réutilise) le pipeline correspondant à cette clé.

    Appelé sous le verrou du module. Renvoie l'objet diffusers prêt à tourner.
    """
    if _RESIDENT.key == key and _RESIDENT.pipe is not None:
        return _RESIDENT.pipe

    _RESIDENT.release("switching model or placement", log)
    t = torch()
    dtype = getattr(t, key.dtype)

    if model is not None and model.needs_supplement:
        pipe = build_hybrid(model, key.cls, dtype, files or {}, log)
    elif model is not None and model.from_gguf:
        pipe = build_from_files(model, key.cls, dtype, files or {}, log)
    else:
        cls = _pipeline_class(key.cls)
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        quant = _quant_config(key.quant, dtype)
        if quant is not None:
            kwargs["quantization_config"] = quant
        if log:
            log(f"[torch] loading {key.cls} from {source} …")
        pipe = cls.from_pretrained(str(source), **kwargs)

    device = "cuda" if key.gpu_index is None else f"cuda:{key.gpu_index}"
    if not t.cuda.is_available():
        device = "cpu"
    _apply_placement(pipe, plan, device, log)

    _RESIDENT.key = key
    _RESIDENT.pipe = pipe
    _RESIDENT.loras = ()
    return pipe


def set_loras(pipe, loras: tuple, log: Callable | None = None) -> None:
    """Greffe exactement ce jeu de LoRA — ni plus, ni moins.

    On COMPARE avec ce qui est déjà en place avant de toucher à quoi que ce
    soit. Sans ça, deux générations d'affilée avec le même LoRA l'empilent deux
    fois et doublent son effet en silence : l'utilisateur voit un rendu qui
    dérive image après image sans avoir rien changé.
    """
    if _RESIDENT.loras == loras:
        return
    if _RESIDENT.loras:
        try:
            pipe.unload_lora_weights()
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"[torch] could not unload the previous LoRAs ({exc}) — "
                    "reloading the pipeline instead.")
            raise
    names, weights = [], []
    for i, (path, weight) in enumerate(loras):
        adapter = f"lora{i}"
        pipe.load_lora_weights(str(path), adapter_name=adapter)
        names.append(adapter)
        weights.append(float(weight))
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        if log:
            log(f"[torch] {len(names)} LoRA(s) applied: "
                + ", ".join(f"{Path(p).name}×{w:g}" for p, w in loras))
    _RESIDENT.loras = loras


def generator_for(seed: int, gpu_index: int | None):
    """Le générateur aléatoire — sur le CPU, volontairement.

    Un générateur CUDA ne donne pas la même suite qu'un générateur CPU, et pas
    forcément la même d'une carte à l'autre. Tirer le bruit initial sur le CPU
    est ce qui rend une graine REJOUABLE d'une machine à l'autre, ce que
    l'interface promet depuis toujours.
    """
    t = torch()
    return t.Generator(device="cpu").manual_seed(int(seed))


def lock():
    return _RESIDENT.lock


# --------------------------------------------------------------------------- #
#  L'échantillonneur
# --------------------------------------------------------------------------- #
def _signature_of(cls) -> set[str] | None:
    """Les arguments que le constructeur de CETTE classe accepte vraiment.

    None quand la signature est illisible — `LMSDiscreteScheduler` n'expose
    qu'un `*args`, par exemple. Voir `schedulers.accepted` : dans ce cas on ne
    filtre rien plutôt que de tout retirer.
    """
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None
    names = {n for n, p in params.items()
             if n not in ("self",)
             and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return names or None


def build_scheduler(pipe, choice, log: Callable | None = None):
    """Le scheduler demandé, greffé sur la CONFIGURATION du modèle.

    `from_config(pipe.scheduler.config, **overrides)` et non une construction
    neuve : la configuration publiée par le modèle porte ses propres constantes
    d'entraînement — `shift`, `base_shift`, `num_train_timesteps`. Repartir des
    valeurs par défaut de la classe les remplacerait par celles de quelqu'un
    d'autre, et le rendu se dégraderait sans qu'aucune erreur ne le dise.

    Toute panne de construction retombe sur le scheduler d'origine du modèle,
    EN LE DISANT. Changer d'échantillonneur est un réglage ; échouer à générer
    parce qu'on a changé d'échantillonneur n'en est pas un.
    """
    from . import schedulers as sched_map
    if choice.is_default:
        return pipe.scheduler
    d = diffusers()
    cls = getattr(d, choice.cls, None)
    if cls is None:
        if log:
            log(f"[torch] this diffusers has no “{choice.cls}” — keeping the "
                "model's own sampler.")
        return pipe.scheduler
    kwargs, dropped = sched_map.accepted(choice.kwargs, _signature_of(cls))
    if dropped and log:
        log(f"[torch] “{choice.cls}” does not accept "
            + ", ".join(dropped) + " — those options are ignored.")
    try:
        return cls.from_config(pipe.scheduler.config, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"[torch] “{choice.cls}” could not be built ({exc}) — keeping "
                "the model's own sampler.")
        return pipe.scheduler
