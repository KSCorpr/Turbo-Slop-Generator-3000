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


def diffusers():
    return _import("diffusers")


def available() -> bool:
    """torch ET diffusers importables — sans les importer pour de bon.

    `find_spec` regarde les métadonnées d'installation ; c'est ce qui permet à
    l'interface de griser un bouton sans embarquer une seconde de chargement à
    chaque rafraîchissement.
    """
    from importlib.util import find_spec
    try:
        return bool(find_spec("torch") and find_spec("diffusers"))
    except (ImportError, ValueError):
        return False


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


def load_pipeline(key: LoadKey, plan: Plan, source: str | Path,
                  log: Callable | None = None):
    """Charge (ou réutilise) le pipeline correspondant à cette clé.

    Appelé sous le verrou du module. Renvoie l'objet diffusers prêt à tourner.
    """
    if _RESIDENT.key == key and _RESIDENT.pipe is not None:
        return _RESIDENT.pipe

    _RESIDENT.release("switching model or placement", log)
    d = diffusers()
    cls = getattr(d, key.cls, None)
    if cls is None:
        raise TorchEngineError(
            f"The installed diffusers ({getattr(d, '__version__', '?')}) has "
            f"no “{key.cls}”. Run setup-torch-engine.bat again to update it.")
    t = torch()
    dtype = getattr(t, key.dtype)
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
