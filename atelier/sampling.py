"""Échantillonneurs et schedulers : la liste ET ce qu'il faut en penser.

Pourquoi une documentation par MODÈLE plutôt qu'un glossaire neutre : sur nos
deux modèles, la moitié du menu est inutile, et pas un peu — structurellement.
Trois propriétés vérifiées dans le moteur l'expliquent, et elles suffisent à
classer tout le reste :

1. **Ce sont des modèles de FLOW MATCHING.** sd.cpp les fait tourner en
   `FLUX_FLOW_PRED` / `FluxFlowDenoiser` (Krea 2 avec un flow shift de 1,15).
   Les schedulers Karras et Exponential ont été conçus pour la diffusion EDM à
   prédiction d'epsilon : leur répartition de sigmas n'a pas le même sens ici.

2. **Ils sont DISTILLÉS à CFG 1.0.** Il n'y a donc aucun guidage à corriger, ce
   qui vide de leur objet toute la famille « CFG++ ». Et le prompt négatif est
   ignoré, quel que soit l'échantillonneur.

3. **Ils tournent en TRÈS PEU DE PAS** — 4 pour Flux.2 Klein, 8 pour Krea 2
   Turbo. Deux conséquences : les méthodes ANCESTRALES réinjectent du bruit à
   chaque pas et n'ont pas le temps de reconverger ; les méthodes MULTI-PAS
   doivent d'abord accumuler un historique d'évaluations, qui n'existe presque
   pas sur un budget aussi court.

Ces verdicts sont donc RAISONNÉS à partir des propriétés des modèles, pas tirés
d'un banc d'essai. Ils disent où porter ses essais, pas ce que vous allez
préférer : sur du rendu, l'œil tranche mieux qu'un principe.
"""
from __future__ import annotations

from . import i18n

# Niveaux de recommandation, du meilleur au pire.
BEST, OK, MEH, BAD = "best", "ok", "meh", "bad"

_MARK = {BEST: "⭐", OK: "", MEH: "△", BAD: "⚠️"}
_VERDICT = {
    BEST: "**Recommandé** pour ce modèle.",
    OK: "Utilisable, sans avantage net ici.",
    MEH: "Peu adapté à ce modèle.",
    BAD: "**Déconseillé** avec ce modèle.",
}

# clé -> (libellé, résumé, avantage, inconvénient, {famille: niveau})
#
# `flux2` = Flux.2 Klein (4 pas) · `krea2` = Krea 2 Turbo (8 pas).
SAMPLERS: dict[str, tuple] = {
    "euler": (
        "Euler",
        "La méthode de base : un pas, une évaluation, aucun ajout de bruit.",
        "Prévisible, reproductible, et la seule qui n'a rien à « rattraper » "
        "sur un budget de 4 à 8 pas. C'est le défaut des deux modèles.",
        "Aucun raffinement : sur BEAUCOUP de pas, d'autres méthodes la "
        "dépassent — mais on n'est pas dans ce régime.",
        {"flux2": BEST, "krea2": BEST}),
    "euler_a": (
        "Euler Ancestral",
        "Euler + réinjection de bruit frais à chaque pas.",
        "Sur des modèles non distillés et beaucoup de pas, apporte de la "
        "variété et du micro-détail.",
        "Le bruit réinjecté doit être reconvergé — impossible en 4 à 8 pas. "
        "Résultat mou ou bruité, et deux rendus jamais identiques.",
        {"flux2": BAD, "krea2": BAD}),
    "heun": (
        "Heun",
        "Euler avec une correction : deux évaluations par pas.",
        "Trajectoire plus juste par pas.",
        "**Deux fois plus lent** à nombre de pas égal. À 4 pas, mieux vaut "
        "dépenser ce budget en pas supplémentaires d'Euler.",
        {"flux2": MEH, "krea2": OK}),
    "dpm2": (
        "DPM2",
        "Méthode d'ordre 2, deux évaluations par pas.",
        "Bonne précision par pas sur les modèles classiques.",
        "Même coût double que Heun, pour un gain que le peu de pas ne laisse "
        "pas s'exprimer.",
        {"flux2": MEH, "krea2": MEH}),
    "dpm++2s_a": (
        "DPM++ 2S Ancestral",
        "Ordre 2, à un seul pas de mémoire, avec bruit ancestral.",
        "Réputée sur SD1.5/SDXL en 20-30 pas.",
        "Cumule les deux défauts qui comptent ici : coût double ET bruit "
        "ancestral non reconvergé.",
        {"flux2": BAD, "krea2": BAD}),
    "dpm++2m": (
        "DPM++ 2M",
        "Multi-pas : réutilise l'évaluation précédente au lieu d'en refaire une.",
        "Le meilleur rapport qualité/temps du lot… à partir d'une quinzaine "
        "de pas.",
        "Son historique n'existe qu'après le 2ᵉ pas : sur 4 pas, la moitié du "
        "parcours se fait sans lui.",
        {"flux2": MEH, "krea2": OK}),
    "dpm++2mv2": (
        "DPM++ 2M v2",
        "Variante de DPM++ 2M au calcul de pas révisé.",
        "Corrige des artefacts de la v1 sur les premiers pas.",
        "Même limite : le multi-pas a besoin de pas.",
        {"flux2": MEH, "krea2": OK}),
    "dpm++2m_sde": (
        "DPM++ 2M SDE",
        "DPM++ 2M en formulation stochastique (bruit à chaque pas).",
        "Texture plus riche sur les longs échantillonnages.",
        "Stochastique = même problème que l'ancestral sur un budget court, "
        "et rendu non reproductible.",
        {"flux2": BAD, "krea2": MEH}),
    "dpm++2m_sde_bt": (
        "DPM++ 2M SDE (Brownian)",
        "Variante à arbre brownien : le bruit devient reproductible.",
        "Retrouve la reproductibilité que la version SDE perd.",
        "Reste stochastique dans son principe, donc mal servi par 4 à 8 pas.",
        {"flux2": BAD, "krea2": MEH}),
    "ipndm": (
        "iPNDM",
        "Pseudo-multi-pas amélioré, sans bruit ajouté.",
        "Sobre et déterministe ; monte en qualité dès une dizaine de pas.",
        "Historique à construire, comme toute méthode multi-pas.",
        {"flux2": MEH, "krea2": OK}),
    "ipndm_v": (
        "iPNDM v",
        "iPNDM à coefficients variables.",
        "Un peu plus stable qu'iPNDM sur les schedules irréguliers.",
        "Même réserve sur le nombre de pas.",
        {"flux2": MEH, "krea2": OK}),
    "lcm": (
        "LCM",
        "Échantillonneur des modèles distillés **par Latent Consistency**.",
        "Excellent — sur un modèle LCM.",
        "Ni Flux.2 Klein ni Krea 2 Turbo ne sont distillés en LCM. Leur "
        "appliquer son parcours donne un rendu délavé.",
        {"flux2": BAD, "krea2": BAD}),
    "ddim_trailing": (
        "DDIM Trailing",
        "DDIM avec alignement des timesteps « trailing ».",
        "Utile sur les modèles où la fin du parcours est mal échantillonnée.",
        "Pensé pour la diffusion classique ; sans objet sur du flow matching.",
        {"flux2": MEH, "krea2": MEH}),
    "tcd": (
        "TCD",
        "Comme LCM : réservé aux modèles distillés **en TCD**.",
        "Très peu de pas — sur un modèle TCD.",
        "Nos modèles ne le sont pas.",
        {"flux2": BAD, "krea2": BAD}),
    "res_multistep": (
        "Res Multistep",
        "Intégrateur exponentiel multi-pas.",
        "Très bonne précision sur les modèles de flow, à pas moyens.",
        "Multi-pas, donc bridé à 4 pas. Le candidat le plus crédible pour "
        "essayer autre chose sur Krea 2.",
        {"flux2": MEH, "krea2": OK}),
    "res_2s": (
        "Res 2S",
        "Intégrateur exponentiel à un pas, ordre 2.",
        "Précis dès les premiers pas, sans historique à constituer — ce qui "
        "le rend, lui, compatible avec un budget court.",
        "Deux évaluations par pas : à durée égale, Euler en fait deux fois plus.",
        {"flux2": OK, "krea2": OK}),
    "er_sde": (
        "ER SDE",
        "Solveur SDE à réversibilité exacte.",
        "Le plus rigoureux des stochastiques.",
        "Stochastique : mauvais usage d'un budget de 4 à 8 pas.",
        {"flux2": BAD, "krea2": MEH}),
    "euler_cfg_pp": (
        "Euler CFG++",
        "Euler avec la correction de guidage « CFG++ ».",
        "Enlève les sur-saturations dues à un CFG élevé.",
        "**Nos deux modèles tournent à CFG 1.0** : il n'y a aucun guidage à "
        "corriger. Cette variante n'a rien à faire ici.",
        {"flux2": BAD, "krea2": BAD}),
    "euler_a_cfg_pp": (
        "Euler Ancestral CFG++",
        "La version ancestrale de la précédente : correction CFG++ plus "
        "réinjection de bruit à chaque pas.",
        "Aucun ici : la correction CFG++ est neutre à CFG 1.0, il ne reste "
        "que le bruit ancestral, qu'Euler Ancestral fournit déjà.",
        "Cumule l'inutilité du CFG++ à CFG 1.0 et le bruit ancestral, qui "
        "n'a pas le temps de se résorber en 4 à 8 pas.",
        {"flux2": BAD, "krea2": BAD}),
    "euler_ge": (
        "Euler GE",
        "Euler à extrapolation de gradient (paramètre `gamma`).",
        "Peut resserrer le trait à très peu de pas — le seul du lot à viser "
        "explicitement ce régime.",
        "Non exposé ici : `gamma` se règle via `--extra-sample-args`, et sans "
        "lui l'effet est marginal.",
        {"flux2": OK, "krea2": OK}),
    "lms": (
        "LMS (linear multi-step)",
        "Multi-pas linéaire classique (`lms_divisions`, défaut 1000).",
        "Ajout récent de sd.cpp ; méthode éprouvée sur de longs parcours.",
        "Multi-pas : c'est exactement ce que 4 à 8 pas ne permettent pas.",
        {"flux2": MEH, "krea2": MEH}),
}

# clé -> (libellé, résumé, avantage, inconvénient, {famille: niveau})
SCHEDULES: dict[str, tuple] = {
    "auto": (
        "Auto (modèle)",
        "Laisse le moteur choisir d'après le modèle chargé.",
        "Toujours cohérent avec le modèle : `flux2` pour Flux.2 Klein, "
        "`discrete` pour Krea 2. C'est le réglage documenté par sd.cpp.",
        "Aucun — sauf si vous voulez expérimenter en connaissance de cause.",
        {"flux2": BEST, "krea2": BEST}),
    "discrete": (
        "Discrete",
        "Répartition uniforme sur les sigmas du modèle.",
        "Neutre et sans surprise. C'est ce que « Auto » choisit sur Krea 2.",
        "Rien de particulier ; simplement pas optimisé pour un modèle donné.",
        {"flux2": OK, "krea2": BEST}),
    "karras": (
        "Karras",
        "Répartition concentrant les pas vers les bas sigmas.",
        "La référence sur SD1.5 / SDXL, où elle gagne beaucoup.",
        "Conçue pour la diffusion **EDM à prédiction d'epsilon**. Nos modèles "
        "sont en flow matching : la courbe ne correspond pas au parcours.",
        {"flux2": BAD, "krea2": MEH}),
    "exponential": (
        "Exponential",
        "Décroissance exponentielle des sigmas.",
        "Simple, parfois utile sur les modèles à v-prediction.",
        "Même inadéquation que Karras vis-à-vis du flow matching.",
        {"flux2": BAD, "krea2": MEH}),
    "ays": (
        "AYS (Align Your Steps)",
        "Répartition optimisée par NVIDIA pour les **petits budgets de pas**.",
        "Pensée exactement pour le régime 8-12 pas — l'idée est bonne ici.",
        "Ses tables sont calibrées sur SD1.5/SDXL, pas sur nos modèles : le "
        "transfert est plausible mais non garanti. À essayer sur Krea 2.",
        {"flux2": MEH, "krea2": OK}),
    "gits": (
        "GITS",
        "Répartition issue d'une recherche sur graphe.",
        "Bons résultats publiés à faible nombre de pas.",
        "Même réserve qu'AYS : calibrée ailleurs.",
        {"flux2": MEH, "krea2": OK}),
    "smoothstep": (
        "Smoothstep",
        "Courbe lissée aux deux extrémités.",
        "Transitions douces, peu d'à-coups en début de parcours.",
        "Effet discret ; rien qui compense un scheduler adapté au modèle.",
        {"flux2": OK, "krea2": OK}),
    "sgm_uniform": (
        "SGM Uniform",
        "Uniforme, à la façon des implémentations SGM.",
        "Proche de Discrete, comportement prévisible.",
        "Aucun avantage identifié sur nos modèles.",
        {"flux2": OK, "krea2": OK}),
    "simple": (
        "Simple",
        "Répartition linéaire élémentaire.",
        "Robuste, sans paramètre. Défaut de DDIM Trailing.",
        "Grossière quand les pas sont peu nombreux.",
        {"flux2": OK, "krea2": OK}),
    "kl_optimal": (
        "KL Optimal",
        "Répartition minimisant une divergence KL le long du parcours.",
        "Bien fondée théoriquement, correcte à pas moyens.",
        "Gain non démontré sur un budget de 4 à 8 pas.",
        {"flux2": MEH, "krea2": OK}),
    "lcm": (
        "LCM",
        "Répartition des modèles Latent Consistency.",
        "Indispensable — avec l'échantillonneur LCM.",
        "Hors de ce couple, elle écrase le parcours et délave le rendu.",
        {"flux2": BAD, "krea2": BAD}),
    "bong_tangent": (
        "Bong Tangent",
        "Courbe en tangente, très marquée.",
        "Effet stylistique parfois intéressant.",
        "Empirique, sans fondement pour nos modèles.",
        {"flux2": MEH, "krea2": MEH}),
    "flux2": (
        "Flux.2",
        "Répartition **taillée pour Flux.2**.",
        "Ce que « Auto » sélectionne sur Flux.2 Klein : le bon choix, "
        "explicitement.",
        "Sur Krea 2, rien ne dit qu'elle transfère.",
        {"flux2": BEST, "krea2": MEH}),
    "flux": (
        "Flux",
        "Répartition des sigmas taillée pour les modèles **Flux.1**, avec le "
        "décalage (shift) propre à cette génération.",
        "Reste une courbe de flow matching cohérente : elle ne casse rien, "
        "et donne un rendu légèrement plus contrasté sur les gros plans.",
        "Flux.2 a la sienne ; utiliser celle de Flux.1 revient à prendre "
        "l'ancienne version d'un réglage taillé sur mesure.",
        {"flux2": MEH, "krea2": MEH}),
    "beta": (
        "Beta",
        "Répartition suivant une loi Beta (paramètres `alpha`, `beta`).",
        "Très modulable — via `--extra-sample-args`.",
        "Sans réglage de ses paramètres, aucun intérêt par rapport à Discrete.",
        {"flux2": MEH, "krea2": MEH}),
    "logit_normal": (
        "Logit Normal",
        "Répartition logit-normale, celle utilisée à l'entraînement de "
        "beaucoup de modèles de flow.",
        "Cohérente avec la façon dont ces modèles ont été entraînés — la "
        "piste la plus défendable après « Auto ».",
        "Ses paramètres (`mu`, `std`) ne sont pas exposés ici.",
        {"flux2": OK, "krea2": OK}),
}


def _family(model_family: str) -> str:
    return "krea2" if model_family == "krea2" else "flux2"


def level(kind: str, key: str, model_family: str) -> str:
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    entry = table.get(key)
    if not entry:
        return OK
    return entry[4].get(_family(model_family), OK)


def choices(kind: str, model_family: str) -> list[tuple[str, str]]:
    """Menu ANNOTÉ : le libellé porte déjà le verdict.

    Marquer les options dans la liste évite d'avoir à ouvrir une aide pour
    savoir laquelle prendre — l'information est là où se fait le choix.
    """
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    out = []
    for key, entry in table.items():
        mark = _MARK[level(kind, key, model_family)]
        out.append((f"{i18n.t(entry[0])} {mark}".strip(), key))
    return out


_RATIONALE = """\
Sur **{model}**, trois propriétés du modèle décident presque tout — et elles
écartent des familles entières d'options, pas une ou deux au cas par cas.

**1. C'est un modèle de *flow matching*.** sd.cpp le fait tourner en mode
« Flux FLOW ». Les schedulers **Karras** et **Exponential**, qui font gagner
beaucoup sur SD 1.5 et SDXL, ont été conçus pour une autre mécanique (diffusion
EDM à prédiction d'epsilon) : leur répartition de sigmas ne correspond pas au
parcours suivi ici.

**2. Il est distillé à CFG 1.0.** Il n'y a donc **aucun guidage à corriger** :
toute la famille **CFG++** (`Euler CFG++`, `Euler Ancestral CFG++`) n'a
littéralement rien à faire. C'est aussi pourquoi le **prompt négatif est
ignoré**, quel que soit l'échantillonneur choisi.

**3. Il tourne en {steps} pas.** C'est très peu, et ça disqualifie deux familles :

- les méthodes **ancestrales** et **stochastiques** (`Euler Ancestral`,
  `DPM++ 2S Ancestral`, les `SDE`, `ER SDE`) réinjectent du bruit à chaque pas.
  Ce bruit doit ensuite être reconvergé — il n'y a pas le budget pour ça, et le
  rendu ressort mou ou bruité ;
- les méthodes **multi-pas** (`DPM++ 2M`, `iPNDM`, `Res Multistep`, `LMS`)
  doivent d'abord accumuler un historique d'évaluations. Sur {steps} pas, une
  bonne partie du parcours se fait avant que cet historique existe.

Enfin, **LCM** et **TCD** ne sont pas des options générales : ce sont les
échantillonneurs de modèles distillés *par ces méthodes-là*. Ce modèle ne l'est
pas ; les appliquer délave le rendu.

---

**Ce qu'il reste, en pratique :** `Euler` + `Auto`. {advice}

*Ces verdicts sont raisonnés à partir des propriétés du modèle, pas tirés d'un
banc d'essai — ils disent où porter vos essais, pas ce que votre œil va
préférer.*"""

_ADVICE = {
    "flux2": "Sur 4 pas, il n'y a pratiquement rien à gagner ailleurs ; si vous "
             "voulez expérimenter, `Res 2S` est le seul autre à être précis "
             "sans historique à constituer.",
    "krea2": "Sur 8 pas, la marge est un peu plus large : `Res Multistep`, "
             "`DPM++ 2M` et le scheduler `AYS` (pensé pour les petits budgets "
             "de pas) valent un essai comparatif à seed fixe.",
}


def rationale(model_family: str) -> str:
    fam = _family(model_family)
    # Traduire AVANT de formater : les placeholders survivent (garanti par
    # tests/test_i18n.py), et le texte inséré est traduit séparément.
    return i18n.t(_RATIONALE).format(
        model="Flux.2 Klein" if fam == "flux2" else "Krea 2 Turbo",
        steps="4" if fam == "flux2" else "8",
        advice=i18n.t(_ADVICE[fam]))


def describe(kind: str, key: str, model_family: str) -> str:
    """Fiche Markdown de l'option choisie : résumé, pour, contre, verdict."""
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    entry = table.get(key)
    if not entry:
        return ""
    name, summary, pro, con, _lv = entry
    lv = level(kind, key, model_family)
    return (f"**{i18n.t(name)}** — {i18n.t(summary)}\n\n"
            f"✅ {i18n.t(pro)}\n\n"
            f"❌ {i18n.t(con)}\n\n"
            f"{_MARK[lv]} {i18n.t(_VERDICT[lv])}".replace("  ", " "))
