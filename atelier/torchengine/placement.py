"""Faire tenir le modèle sur la carte — et par où il arrive.

**Corrigé après coup, et la correction change tout.** La première version de ce
module partait du principe qu'un modèle PyTorch arrive en pleine précision et
qu'il faut le quantifier soi-même au chargement (bitsandbytes, int8 ou NF4).
C'est vrai pour un dépôt diffusers complet. Ça ne l'est pas pour le chemin
normal de cette branche : diffusers lit le GGUF, donc le fichier est DÉJÀ
quantifié, par la même échelle que côté stable-diffusion.cpp, et il pèse ce
qu'il pèse sur le disque.

D'où deux plans possibles, et pas un :

* **depuis le GGUF** — la précision est déjà décidée, il ne reste que le
  placement. Les tailles ne sont pas estimées mais LUES sur le disque, ce qui
  est autrement plus fiable qu'un calcul en octets par paramètre ;
* **depuis un dépôt complet** (Krea 2, faute de chargement fichier-unique en
  amont) — là, et là seulement, l'ancienne échelle s'applique.

Le placement lui-même a trois réponses, et une seule est la bonne selon la
carte :

1. **Tout sur le GPU** — la plus rapide, et la seule sans allers-retours. Elle
   demande de la place pour les poids ET pour le calcul.
2. **Décharge par MODULE** (`enable_model_cpu_offload`) — chaque sous-modèle
   monte sur la carte quand c'est son tour, puis redescend. Un seul est présent
   à la fois : le pic est celui du plus gros, pas de la somme.
3. **Décharge par COUCHE** (`enable_sequential_cpu_offload`) — les poids
   restent en RAM et traversent le PCIe couche par couche, à chaque pas. Ça
   tient dans presque rien et c'est lent d'un ordre de grandeur ; c'est le
   filet, pas un mode de travail.

Sur le chemin dépôt complet, une règle relie précision et placement : **on garde
la meilleure précision qui évite la décharge par couche.**

Tout est décidé ICI, en Python pur. Aucun import de torch, donc le raisonnement
se teste sur une machine sans GPU — ce qui est précisément le cas où une erreur
de placement passerait inaperçue jusqu'à la première panne.
"""
from __future__ import annotations

from dataclasses import dataclass

#  Réserve laissée LIBRE sur la carte. Elle ne couvre pas le modèle mais tout
#  ce qui l'entoure : contexte CUDA, buffers d'attention, latents, VAE au
#  décodage, et le bureau Windows qui prend déjà sa part. Mesurée du côté
#  sd.cpp, où `--max-vram -1` réserve 1 Gio ; ici la marge est plus large parce
#  que PyTorch alloue par blocs et fragmente, là où un moteur natif place ses
#  tenseurs une fois pour toutes.
COMPUTE_RESERVE_GB = 2.5

FULL = "full"
MODEL_OFFLOAD = "model_offload"
SEQUENTIAL_OFFLOAD = "sequential_offload"

#  Facteurs appliqués au poids bf16, du plus précis au plus économe.
#  · int8 : exactement la moitié, 1 octet contre 2.
#  · NF4  : 4,127 bits par paramètre avec la double quantification, soit 0,516
#    octet — 0,258 fois le bf16. Les couches laissées intactes (normalisations,
#    embeddings, entrée et sortie) remontent le chiffre réel ; 0,28 est une
#    ESTIMATION haute, et elle est là pour ne pas promettre ce qui ne tiendra
#    pas. Le poids réel se mesure au premier chargement.
QUANT_LADDER: tuple[tuple[str, float], ...] = (
    ("none", 1.00),
    ("int8", 0.50),
    ("nf4", 0.28),
)


@dataclass(frozen=True)
class Plan:
    """Ce qu'on va demander à diffusers, et pourquoi."""
    mode: str
    dtype: str
    quant: str
    vae_tiling: bool
    attention_slicing: bool
    reason: str

    @property
    def is_offloaded(self) -> bool:
        return self.mode in (MODEL_OFFLOAD, SEQUENTIAL_OFFLOAD)

    @property
    def is_quantized(self) -> bool:
        return self.quant != "none"


def dtype_for(arch: str) -> str:
    """bf16 partout où la carte le sait faire, fp16 sinon.

    Ce n'est pas une préférence de goût. Les modèles de flow matching récents
    sont entraînés et publiés en bf16 : son exposant est celui du fp32, donc
    les activations très petites ou très grandes ne saturent pas. Le fp16 a un
    exposant plus court et produit des NaN sur ces mêmes réseaux — le symptôme
    classique étant une image entièrement noire, sans le moindre message.

    Le bf16 matériel arrive avec Ampere (RTX 30xx). Turing (RTX 20xx) et Pascal
    l'émulent, ce qui est lent ; sur ces cartes le fp16 reste le choix
    raisonnable, avec le risque de NaN assumé et signalé.
    """
    return "bfloat16" if arch in ("ampere", "ada", "hopper", "blackwell") \
        else "float16"


def plan_from_files(vram_gb: float | None, sizes_gb: dict[str, float],
                    arch: str = "unknown") -> Plan:
    """Le placement quand les poids arrivent DÉJÀ quantifiés, depuis le disque.

    `sizes_gb` est le poids réel de chaque composant du pipeline — pas une
    estimation, la taille des fichiers. C'est le chemin normal de cette
    branche, et le plus simple : il n'y a plus de précision à choisir, la
    quantification a été décidée au téléchargement par l'échelle du catalogue
    principal, exactement comme pour stable-diffusion.cpp.

    Une pièce manquante (un composant pas encore téléchargé) rend un plan
    prudent plutôt qu'un plan calculé sur des trous : on ne connaît pas la
    taille, donc on ne parie pas sur la solution rapide.
    """
    if not vram_gb or vram_gb <= 0:
        return Plan(SEQUENTIAL_OFFLOAD, "float32", "gguf", True, True,
                    "No GPU detected — everything runs on the CPU.")
    dtype = dtype_for(arch)
    budget = vram_gb - COMPUTE_RESERVE_GB
    known = [v for v in sizes_gb.values() if v and v > 0]
    if not known or len(known) < len(sizes_gb):
        return Plan(MODEL_OFFLOAD, dtype, "gguf", True, False,
                    "Some weights are not on disk yet: using per-module "
                    "offload, which works wherever keeping everything on the "
                    "card would have.")
    resident, largest = sum(known), max(known)
    detail = " + ".join(f"{name} {size:.1f}"
                        for name, size in sorted(sizes_gb.items(),
                                                 key=lambda kv: -kv[1]))
    if resident <= budget:
        return Plan(FULL, dtype, "gguf", False, False,
                    f"Already quantized on disk ({detail} = "
                    f"{resident:.1f} GB): the whole pipeline fits in "
                    f"{vram_gb:.0f} GB with room to compute.")
    if largest <= budget:
        return Plan(MODEL_OFFLOAD, dtype, "gguf", True, False,
                    f"Already quantized on disk ({detail} = "
                    f"{resident:.1f} GB): too much at once for "
                    f"{vram_gb:.0f} GB, but the largest part "
                    f"({largest:.1f} GB) fits, so modules take turns.")
    return Plan(SEQUENTIAL_OFFLOAD, dtype, "gguf", True, True,
                f"Even the largest part ({largest:.1f} GB) exceeds the "
                f"{budget:.1f} GB budget: weights stream layer by layer — "
                "slow, but it finishes. A lower quantization rung in Settings "
                "would be the better answer.")


def plan(vram_gb: float | None, resident_gb: float | None,
         largest_module_gb: float | None, arch: str = "unknown",
         allow_quant: bool = True) -> Plan:
    """Précision et placement pour un modèle chargé en PLEINE PRÉCISION.

    Ce chemin ne sert plus qu'aux modèles sans chargement fichier-unique en
    amont — aujourd'hui Krea 2, et lui seul. Partout ailleurs c'est
    `plan_from_files` qui décide, sur des tailles lues et non estimées.

    `resident_gb` est le poids de TOUT le pipeline en bf16 ;
    `largest_module_gb` celui du plus gros sous-modèle, qui décide à lui seul
    si la décharge par module peut fonctionner.

    Une taille inconnue (dépôt fermé, donc non mesurable) n'autorise PAS à
    parier sur la solution rapide : on prend la décharge par module, qui marche
    partout où la solution rapide aurait marché, et dans beaucoup d'autres. Le
    coût est du temps, pas un échec.
    """
    if not vram_gb or vram_gb <= 0:
        return Plan(SEQUENTIAL_OFFLOAD, "float32", "none", True, True,
                    "No GPU detected — everything runs on the CPU.")
    dtype = dtype_for(arch)
    budget = vram_gb - COMPUTE_RESERVE_GB

    if resident_gb is None or largest_module_gb is None:
        return Plan(MODEL_OFFLOAD, dtype, "none", True, False,
                    "Model size unknown (closed repository): using per-module "
                    "offload, which works wherever keeping everything on the "
                    "card would have.")

    ladder = QUANT_LADDER if allow_quant else QUANT_LADDER[:1]
    for quant, factor in ladder:
        resident = resident_gb * factor
        largest = largest_module_gb * factor
        if resident <= budget:
            return Plan(FULL, dtype, quant, False, False,
                        _why(quant, f"the whole pipeline ({resident:.1f} GB) "
                                    f"fits in {vram_gb:.0f} GB with room to "
                                    f"compute"))
        if largest <= budget:
            return Plan(MODEL_OFFLOAD, dtype, quant, True, False,
                        _why(quant, f"the pipeline ({resident:.1f} GB) does "
                                    f"not fit but its largest part "
                                    f"({largest:.1f} GB) does, so modules "
                                    f"take turns on the card"))

    quant, factor = ladder[-1]
    return Plan(SEQUENTIAL_OFFLOAD, dtype, quant, True, True,
                _why(quant, f"even the largest part "
                            f"({largest_module_gb * factor:.1f} GB) exceeds "
                            f"the {budget:.1f} GB budget, so weights stream "
                            f"layer by layer — slow, but it finishes"))


def _why(quant: str, tail: str) -> str:
    head = {"none": "In bf16/fp16",
            "int8": "Quantized to int8 (half the weight)",
            "nf4": "Quantized to NF4 (a quarter of the weight)"}[quant]
    return f"{head}, {tail}."


def forced(chosen: Plan, mode: str | None) -> Plan:
    """Le même plan, avec le placement imposé — pour le banc d'essai.

    La réserve de calcul est une ESTIMATION : 2,5 Gio pour le contexte CUDA,
    les tampons d'attention, les latents et le bureau. Elle décide à elle seule
    entre « tout sur la carte » et « les modules à tour de rôle », et elle n'a
    aucune raison d'être juste sur toutes les machines. Ce point d'entrée
    permet de la contourner LE TEMPS D'UNE MESURE, sans toucher au plan que
    l'application prend d'elle-même.
    """
    if not mode or mode == chosen.mode:
        return chosen
    if mode not in (FULL, MODEL_OFFLOAD, SEQUENTIAL_OFFLOAD):
        return chosen
    return Plan(mode, chosen.dtype, chosen.quant,
                mode != FULL, mode == SEQUENTIAL_OFFLOAD,
                f"Placement forced to “{mode}” for a measurement (the "
                f"planner had chosen “{chosen.mode}”).")


def describe(p: Plan) -> str:
    """Une ligne pour le journal, dans la langue de l'interface."""
    label = {FULL: "everything on the card",
             MODEL_OFFLOAD: "modules take turns on the card",
             SEQUENTIAL_OFFLOAD: "weights streamed layer by layer"}[p.mode]
    precision = {"none": p.dtype,
                 "gguf": f"GGUF ({p.dtype} compute)"}.get(
                     p.quant, f"{p.quant} ({p.dtype} compute)")
    extras = [name for name, on in (("VAE tiling", p.vae_tiling),
                                    ("attention slicing", p.attention_slicing))
              if on]
    tail = f" · {' · '.join(extras)}" if extras else ""
    return f"[torch] {label} · {precision}{tail} — {p.reason}"
