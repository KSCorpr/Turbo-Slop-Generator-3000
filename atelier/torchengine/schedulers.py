"""Traduire le vocabulaire sd.cpp (21 samplers × 16 schedulers) chez diffusers.

Les deux moteurs ne découpent pas le problème pareil. sd.cpp expose un
ÉCHANTILLONNEUR et un SCHEDULER comme deux menus indépendants ; diffusers
expose des classes de scheduler dont les options couvrent parfois l'un,
parfois l'autre, parfois les deux. La traduction est donc partielle par nature,
et la seule façon honnête de la faire est de le dire.

Trois catégories, et l'interface ne ment sur aucune :

* **exacte** — la même méthode existe, avec les mêmes réglages ;
* **approchée** — une classe voisine la couvre, à une nuance près qui est
  écrite ;
* **absente** — rien chez diffusers ne fait ça. On retombe alors sur Euler
  (le défaut de tous nos modèles) EN LE DISANT dans le journal, plutôt que de
  laisser croire que le menu a été respecté.

Ce dernier point est le seul qui compte vraiment. Un menu qui accepte tout et
applique autre chose est pire qu'un menu réduit : on croit avoir essayé.

Toutes les signatures ci-dessous ont été relevées dans les sources de diffusers
0.40.0, pas de mémoire — notamment `use_flow_sigmas` / `flow_shift` sur les
solveurs DPM et UniPC, qui sont ce qui les rend utilisables sur des modèles de
flow matching, et `stochastic_sampling` sur le scheduler Euler à flux, qui est
l'équivalent des méthodes ancestrales.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EXACT = "exact"
APPROX = "approx"
MISSING = "missing"


@dataclass(frozen=True)
class Mapping:
    """La classe diffusers à instancier, et ce qu'on en pense."""
    cls: str
    kwargs: dict = field(default_factory=dict)
    fidelity: str = EXACT
    note: str = ""


_FLOW_EULER = "FlowMatchEulerDiscreteScheduler"
_DPM = "DPMSolverMultistepScheduler"
_DPM_1S = "DPMSolverSinglestepScheduler"

#  Les modèles du catalogue sont TOUS des modèles de flow matching : c'est ce
#  qui impose `use_flow_sigmas` sur les solveurs DPM/UniPC, sans quoi ils
#  calculent des sigmas de diffusion EDM et le rendu part en bouillie.
_FLOW = {"use_flow_sigmas": True, "prediction_type": "flow_prediction"}

SAMPLERS: dict[str, Mapping] = {
    "euler": Mapping(_FLOW_EULER),
    "euler_a": Mapping(
        _FLOW_EULER, {"stochastic_sampling": True}, APPROX,
        "diffusers expresses the ancestral idea as stochastic sampling on the "
        "flow scheduler; the noise schedule is not identical to sd.cpp's."),
    "heun": Mapping("FlowMatchHeunDiscreteScheduler"),
    "dpm++2m": Mapping(_DPM, {**_FLOW, "solver_order": 2,
                              "algorithm_type": "dpmsolver++"}),
    "dpm++2mv2": Mapping(
        _DPM, {**_FLOW, "solver_order": 2, "algorithm_type": "dpmsolver++",
               "solver_type": "heun"}, APPROX,
        "sd.cpp's “v2” is a revised step computation; the closest knob here is "
        "the Heun solver type."),
    "dpm++2m_sde": Mapping(_DPM, {**_FLOW, "solver_order": 2,
                                  "algorithm_type": "sde-dpmsolver++"}),
    "dpm++2m_sde_bt": Mapping(
        _DPM, {**_FLOW, "solver_order": 2,
               "algorithm_type": "sde-dpmsolver++"}, APPROX,
        "the Brownian-tree noise that makes sd.cpp's variant reproducible has "
        "no equivalent here: this one is stochastic, so the same seed will not "
        "give the same image twice."),
    "dpm++2s_a": Mapping(_DPM_1S, {**_FLOW, "solver_order": 2,
                                   "algorithm_type": "sde-dpmsolver++"}, APPROX,
                         "single-step solver with stochastic noise — the "
                         "closest thing to an ancestral second-order method."),
    "dpm2": Mapping(_DPM, {**_FLOW, "solver_order": 2,
                           "algorithm_type": "dpmsolver",
                           # `dpmsolver` (sans ++) REFUSE `final_sigmas_type`
                           # à zéro : la construction lève une ValueError, et
                           # ce n'est pas un détail de signature qu'un filtre
                           # rattrape. Vérifié en instanciant les 336
                           # combinaisons du menu.
                           "final_sigmas_type": "sigma_min"}, APPROX,
                    "sd.cpp's DPM2 is a Karras-style second-order method; this "
                    "is the DPM-Solver of the same order — and upstream has "
                    "deprecated that variant, so it may disappear."),
    "ipndm": Mapping("IPNDMScheduler", {}, APPROX,
                     "no flow-matching variant: the sigmas are the diffusion "
                     "ones, which suits these models poorly."),
    "lcm": Mapping("FlowMatchLCMScheduler"),
    "tcd": Mapping("TCDScheduler", {}, APPROX,
                   "TCD is defined for epsilon-prediction models; there is no "
                   "flow-matching variant."),
    "lms": Mapping("LMSDiscreteScheduler", {}, APPROX,
                   "linear multistep on diffusion sigmas, not flow ones."),
    "ddim_trailing": Mapping("DDIMScheduler", {"timestep_spacing": "trailing"},
                             APPROX,
                             "DDIM is a diffusion sampler; only the trailing "
                             "timestep spacing carries over."),
    # Ce que sd.cpp a et diffusers n'a pas. Aucune n'est un oubli : ce sont des
    # méthodes venues de k-diffusion ou propres au moteur, sans classe
    # correspondante en amont.
    "ipndm_v": Mapping("", {}, MISSING),
    "res_multistep": Mapping("", {}, MISSING),
    "res_2s": Mapping("", {}, MISSING),
    "er_sde": Mapping("", {}, MISSING),
    "euler_cfg_pp": Mapping("", {}, MISSING),
    "euler_a_cfg_pp": Mapping("", {}, MISSING),
    "euler_ge": Mapping("", {}, MISSING),
}

#  Les schedulers, eux, deviennent des OPTIONS de la classe choisie ci-dessus.
SCHEDULES: dict[str, tuple[dict, str, str]] = {
    "auto": ({}, EXACT, ""),
    "discrete": ({}, EXACT, ""),
    "karras": ({"use_karras_sigmas": True}, EXACT, ""),
    "exponential": ({"use_exponential_sigmas": True}, EXACT, ""),
    # `use_beta_sigmas` lève un ImportError si scipy manque — d'où scipy
    # dans les dépendances du moteur, plutôt qu'un menu qui plante au clic.
    "beta": ({"use_beta_sigmas": True}, EXACT, ""),
    "simple": ({}, MISSING, ""),
    "sgm_uniform": ({"timestep_spacing": "trailing"}, APPROX,
                    "SGM uniform is trailing timestep spacing, which only "
                    "exists on the diffusion schedulers: on a flow-matching "
                    "sampler it is dropped."),
    "ays": ({}, MISSING, ""),
    "gits": ({}, MISSING, ""),
    "kl_optimal": ({}, MISSING, ""),
    "smoothstep": ({}, MISSING, ""),
    "bong_tangent": ({}, MISSING, ""),
    "lcm": ({}, MISSING, ""),
    "flux": ({}, MISSING, ""),
    "flux2": ({}, MISSING, ""),
    "logit_normal": ({}, MISSING, ""),
}


@dataclass(frozen=True)
class Choice:
    """Le résultat d'une traduction : quoi construire, et quoi dire."""
    cls: str
    kwargs: dict
    notes: tuple[str, ...]

    @property
    def is_default(self) -> bool:
        return self.cls == _FLOW_EULER and not self.kwargs


def accepted(kwargs: dict, allowed: set[str] | None) -> tuple[dict, list[str]]:
    """Retire les options que CETTE classe n'accepte pas, et dit lesquelles.

    Le menu est celui de sd.cpp ; les classes diffusers, elles, n'ont pas
    toutes les mêmes réglages, et la liste bouge d'une version à l'autre.
    Coder les exceptions à la main revenait à réécrire ce tableau à chaque
    montée de version — et à découvrir les oublis en panne, puisqu'un argument
    inconnu lève un TypeError au premier chargement seulement.

    Mesuré sur les 336 combinaisons du menu : 110 échouaient à la
    construction. Presque toutes pour cette raison — `FlowMatchHeun` n'accepte
    ni Karras ni Exponential, `FlowMatchEuler` ignore `timestep_spacing`.

    `allowed` à None (signature illisible) ne filtre rien : mieux vaut laisser
    passer et échouer bruyamment que retirer en silence ce qui marchait.
    """
    if allowed is None:
        return dict(kwargs), []
    kept = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(k for k in kwargs if k not in allowed)
    return kept, dropped


def resolve(sampler: str | None, schedule: str | None,
            flow_shift: float = 0.0) -> Choice:
    """Traduit un couple (échantillonneur, scheduler) en scheduler diffusers.

    Ce qui n'existe pas retombe sur Euler à flux — le défaut de tous nos
    modèles — et la raison part dans `notes` pour être écrite dans le journal.
    Silencieusement retomber serait le seul vrai défaut possible ici.
    """
    notes: list[str] = []
    name = (sampler or "euler").strip() or "euler"
    mapping = SAMPLERS.get(name)
    if mapping is None:
        notes.append(f"Unknown sampler “{name}” — using Euler.")
        mapping = SAMPLERS["euler"]
    elif mapping.fidelity == MISSING:
        notes.append(f"“{name}” has no diffusers equivalent — using Euler "
                     "instead.")
        mapping = SAMPLERS["euler"]
    elif mapping.fidelity == APPROX:
        notes.append(f"“{name}”: {mapping.note}")

    kwargs = dict(mapping.kwargs)

    sched = (schedule or "auto").strip() or "auto"
    extra, fidelity, note = SCHEDULES.get(sched, ({}, MISSING, ""))
    if fidelity == MISSING and sched not in ("auto", "discrete"):
        notes.append(f"Scheduler “{sched}” has no diffusers equivalent — "
                     "keeping the model's own schedule.")
    else:
        if note:
            notes.append(f"“{sched}”: {note}")
        kwargs.update(extra)

    # `flow_shift` porte deux noms selon la classe : `shift` sur le scheduler
    # Euler à flux, `flow_shift` sur les solveurs DPM/UniPC. Le même réglage,
    # deux orthographes — et l'envoyer sous le mauvais nom lève un TypeError au
    # tout premier chargement.
    if flow_shift and flow_shift > 0:
        key = "shift" if mapping.cls.startswith("FlowMatch") else "flow_shift"
        kwargs[key] = float(flow_shift)

    return Choice(mapping.cls, kwargs, tuple(notes))
