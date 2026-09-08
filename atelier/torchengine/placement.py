"""Faire tenir un modèle diffusers sur 11 ou 12 Go — la vraie difficulté.

Côté sd.cpp le problème est résolu à l'achat : on télécharge le fichier GGUF
quantifié qui tient dans la carte, et il tient. Rien de tel ici. Un dépôt
diffusers est publié dans SA précision (souvent fp32), le pipeline se charge en
entier, et il reste deux questions à trancher au chargement — dans quelle
précision, et où mettre les poids.

**La précision.** C'est l'équivalent exact de l'échelle GGUF, sauf que la
conversion se fait chez nous au lieu d'être téléchargée toute faite. Trois
crans, du meilleur au plus économe : bf16 (2 octets par paramètre), int8
(1 octet), NF4 (~0,5). Le back-end est bitsandbytes — vérifié : c'est le seul
des deux candidats à publier une roue `win_amd64`, torchao n'en publie pas.

**Le placement.** PyTorch offre trois réponses, et une seule est la bonne selon
la carte :

1. **Tout sur le GPU** — la plus rapide, et la seule sans allers-retours. Elle
   demande de la place pour les poids ET pour le calcul.
2. **Décharge par MODULE** (`enable_model_cpu_offload`) — chaque sous-modèle
   monte sur la carte quand c'est son tour, puis redescend. Un seul est présent
   à la fois : le pic est celui du plus gros, pas de la somme.
3. **Décharge par COUCHE** (`enable_sequential_cpu_offload`) — les poids
   restent en RAM et traversent le PCIe couche par couche, à chaque pas. Ça
   tient dans presque rien et c'est lent d'un ordre de grandeur.

La règle qui relie les deux : **on garde la meilleure précision qui évite la
décharge par couche.** Le troisième mode est un filet, pas un mode de travail —
descendre d'un cran de précision coûte un peu de qualité, y rester coûte un
facteur dix sur le temps.

Tout est décidé ICI, en Python pur, à partir de chiffres mesurables. Aucun
import de torch, donc le raisonnement se teste sur une machine sans GPU — ce
qui est précisément le cas où une erreur de placement passerait inaperçue
jusqu'à la première panne.
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


def plan(vram_gb: float | None, resident_gb: float | None,
         largest_module_gb: float | None, arch: str = "unknown",
         allow_quant: bool = True) -> Plan:
    """Précision et placement à demander pour ce modèle sur cette carte.

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


def describe(p: Plan) -> str:
    """Une ligne pour le journal, dans la langue de l'interface."""
    label = {FULL: "everything on the card",
             MODEL_OFFLOAD: "modules take turns on the card",
             SEQUENTIAL_OFFLOAD: "weights streamed layer by layer"}[p.mode]
    precision = p.dtype if p.quant == "none" else f"{p.quant} ({p.dtype} compute)"
    extras = [name for name, on in (("VAE tiling", p.vae_tiling),
                                    ("attention slicing", p.attention_slicing))
              if on]
    tail = f" · {' · '.join(extras)}" if extras else ""
    return f"[torch] {label} · {precision}{tail} — {p.reason}"
