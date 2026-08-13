"""Onglets « Xanax » : le style est CODÉ EN DUR, il n'y a rien à régler.

Transcription du system prompt Gemini de l'utilisateur, réduite à l'essentiel :
une phrase en entrée, une image en sortie. Contrairement aux onglets de
génération classiques, il n'y a ici ni menu de préréglages, ni banques de
styles, ni prompt système modifiable — le style est « fixe et non négociable »,
donc il n'est pas exposé.

Un point mérite d'être explicité, parce qu'il surprend : **la traduction FR→EN
a besoin de l'améliorateur**. Le style est un préfixe anglais collé devant le
texte ; à lui seul il ne traduit rien. Si l'améliorateur est installé on s'en
sert (avec le style transmis en contrainte, pour qu'il n'écrive rien qui le
contredise) ; sinon on le dit clairement au lieu de laisser croire que ça marche.
"""
from __future__ import annotations

import queue
import random
import re
import threading

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import tools
from ..i18n import t

# --------------------------------------------------------------------------- #
#  LE STYLE — fixe, non négociable, non exposé dans l'interface.
# --------------------------------------------------------------------------- #
XANAX_STYLE = (
    "amateur snapshot photograph, everyday life in provincial France between "
    "1995 and 2005, shot on a cheap low-end consumer point-and-shoot camera, "
    "naturalistic mundane unadorned photography, plain uncomposed careless "
    "framing, ordinary unremarkable people going about their day, raw "
    "unstaged daily life, flat grey overcast daylight, dull mundane "
    "surroundings, clean untouched image straight out of the camera, no grain, "
    "no filter, no post-processing"
)

# Format 4:3 imposé par le brief, sur la grille NATIVE de chaque modèle (32 px
# pour Flux.2, 64 px pour Krea 2 — sortir de la grille dégrade le rendu).
XANAX_SIZE = {"flux2": (1184, 880), "krea2": (1152, 896)}

# --------------------------------------------------------------------------- #
#  DE QUOI ON PART : une phrase de la vie courante, pas une description
# --------------------------------------------------------------------------- #
# Décrire une image (« un homme qui attend le bus devant un supermarché ») et
# raconter sa journée (« j'ai attendu le bus une plombe ») ne donnent pas le
# même résultat, et ce n'est pas une question de formulation. Une description
# est déjà cadrée : elle dit quoi montrer, donc le modèle met le sujet au
# milieu et compose. Une phrase de journal intime ne dit PAS ce qu'il faut
# montrer — il faut aller chercher le lieu, l'heure, les gens autour ; ce qui
# en sort ressemble à une photo prise en passant, ce que vise cet onglet.
#
# C'est pourquoi l'améliorateur reçoit ici son propre system prompt (style
# « xanax ») : les deux autres réclament un éclairage travaillé, un objectif
# nommé et une composition — la photo RÉUSSIE d'un photographe, alors qu'on
# veut la photo RATÉE d'un oncle.
ANECDOTES = [
    "j'ai mangé chez Flunch avec Mamie",
    "journée pas terrible mais j'ai pu aller acheter des clopes",
    "on a fait les courses au Leclerc, y'avait la queue à la caisse",
    "anniversaire de Papy, on était tous dans la véranda",
    "j'ai attendu le bus vingt minutes sous la pluie",
    "réveillon chez ma tante, on a mangé de la bûche",
    "j'ai lavé la voiture dans l'allée",
    "on est allés à la kermesse de l'école de mon fils",
    "j'ai poireauté à la CAF toute la matinée",
    "barbecue chez les voisins, il a commencé à pleuvoir",
    "j'ai repeint la chambre, c'est pas fini",
    "on a mangé au routier sur la nationale",
    "j'ai emmené le chien chez le véto",
    "communion de ma cousine, photo devant l'église",
    "on a bu un café au bar-tabac après le marché",
    "j'ai déménagé le canapé de ma sœur",
    "vide-grenier dimanche matin, j'ai rien vendu",
    "on a regardé le match chez Kévin",
    "j'ai passé l'après-midi à la laverie",
    "on est allés voir la mer, il faisait gris",
    "j'ai monté le meuble Ikea de la cuisine",
    "pot de départ au boulot dans la salle de pause",
    "j'ai fait la queue à la poste pour un colis",
    "on a pris l'apéro dans le jardin, rien de spécial",
    "j'ai gagné trois euros au PMU",
    "on a mangé une pizza devant la télé",
    "j'ai attendu ma fille à la sortie du collège",
    "on a fait une pause sur l'aire d'autoroute",
    "j'ai réparé le vélo dans le garage",
    "mariage de mon collègue, salle des fêtes",
    "j'ai tondu la pelouse avant qu'il pleuve",
    "on a fêté ça au kebab en bas de chez moi",
]

# Barres de progression de sd.cpp : converties en ligne de statut, jamais
# écrites dans le journal (elles arrivent par centaines et le noient).
_PROGRESS_BAR = re.compile(r"\|[#=>\-\s]*\|")


def _size_for(family: str) -> tuple[int, int]:
    return XANAX_SIZE.get(family, (1184, 880))


def build_prompt(subject: str) -> str:
    """Le prompt réellement envoyé : le style figé, puis le sujet."""
    return f"{XANAX_STYLE}, {(subject or '').strip().strip(',')}".strip(", ")


# Modèles proposés dans l'onglet, du plus rapide au plus lourd. Un SEUL onglet
# pour les deux : le style est identique, seul le moteur change — deux onglets
# jumeaux, c'était deux fois le même écran à maintenir et une case de plus à
# lire dans la barre.
XANAX_MODELS = [("⚡ Krea 2 Turbo", "krea2-turbo"),
                ("🟣 Flux.2 Klein 9B", "flux2-klein-9b")]


def _model_info(model_id: str):
    """(modèle, famille, défauts, largeur, hauteur, pas) pour un id donné."""
    model = registry.get_base_model(model_id, settings.load_prefs())
    family = model.family if model else "flux2"
    d = dict(model.defaults) if model else {}
    w, h = _size_for(family)
    return model, family, d, w, h, int(d.get("steps", 8) or 8)


def _recap_for(model_id: str) -> str:
    """État du modèle choisi + format qu'il produira, avant le clic."""
    model, _fam, _d, w, h, _st = _model_info(model_id)
    ready = model is not None and registry.model_is_ready(model)
    state = (t("● modèle prêt") if ready
             else t("○ à télécharger (onglet Catalogue de modèles)"))
    return f"{state} · {w}×{h}"


def build_xanax_tab(title: str = "💊 Xanax"):
    with gr.Tab(title):
        # Texte SÉPARÉ du titre : une f-string composée ne peut pas servir de
        # clé de traduction (elle ne correspondrait jamais au dictionnaire).
        gr.Markdown(
            "### Racontez votre journée, on en fait une photo\n"
            "N'écrivez **pas une description d'image** mais une phrase de la "
            "vie courante, comme dans un carnet : *« j'ai mangé chez Flunch "
            "avec Mamie »*, *« journée pas terrible mais j'ai pu aller acheter "
            "des clopes »*. C'est ce décalage qui donne la photo prise en "
            "passant plutôt que la photo posée.\n\n"
            "**Le style est figé et non modifiable** : photo amateur, France "
            "provinciale, 1995-2005, temps couvert, aucun post-traitement, "
            "format 4:3 sur la grille native du modèle. C'est le principe de "
            "cet onglet — pour régler quoi que ce soit, utilisez un onglet de "
            "génération normal.")

        with gr.Row():
            with gr.Column(scale=3):
                model_pick = gr.Radio(
                    choices=[(t(lbl), mid) for lbl, mid in XANAX_MODELS],
                    value=XANAX_MODELS[0][1], label="Modèle")
                model_state = gr.Markdown(_recap_for(XANAX_MODELS[0][1]),
                                          elem_classes="hint")
                prompt = gr.Textbox(
                    label="Ce que vous avez fait", lines=3,
                    placeholder="j'ai mangé chez Flunch avec Mamie…",
                    info="Une phrase de votre journée, à la première personne. "
                         "Pas « un homme attend le bus » mais « j'ai attendu "
                         "le bus une plombe ».")
                dice = gr.Button("🎲 Une journée au hasard", size="sm")
                with gr.Row(elem_classes="go-row"):
                    run = gr.Button("📷 Générer", variant="primary",
                                    size="lg", scale=4)
                    stop = gr.Button("⏹️ Stop", variant="stop", scale=1,
                                     min_width=90)
                status = gr.Markdown("")
                enhance = gr.Checkbox(
                    value=tools.enhance_is_installed(),
                    interactive=tools.enhance_is_installed(),
                    label="✨ Transformer ma phrase en photo (améliorateur IA)",
                    info=("Cherche ce qu'on VERRAIT sur la photo : le lieu, "
                          "les gens, l'heure. Traduit au passage, et sait ce "
                          "qu'est un Flunch — le modèle d'image, non."
                          if tools.enhance_is_installed() else
                          "Améliorateur non installé — installez-le depuis un "
                          "onglet de génération. Sans lui votre phrase part "
                          "TELLE QUELLE : écrivez alors en anglais et dites ce "
                          "qu'on voit, pas ce que vous avez fait."))
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = aléatoire)",
                                 info="Une graine fixe rejoue exactement la "
                                      "même photo.")
            with gr.Column(scale=4):
                result = gr.Image(label="Photo", type="filepath", height=460,
                                  format="png", show_download_button=True)
                used_md = gr.Markdown("", elem_classes="hint")
                log = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        model_pick.change(_recap_for, inputs=[model_pick],
                          outputs=[model_state])
        # Le dé ne sert pas qu'à dépanner l'inspiration : il montre le REGISTRE
        # attendu. Un exemple qu'on peut lire, modifier et relancer explique
        # mieux qu'un paragraphe ce que veut dire « pas une description ».
        dice.click(lambda: random.choice([t(a) for a in ANECDOTES]),
                   outputs=[prompt])

        def do_xanax(model_id, subject, use_enhance, seed_v):
            if not (subject or "").strip():
                raise gr.Error(t("Racontez d'abord quelque chose — "
                                 "une phrase suffit."))
            settings.ensure_dirs()
            model, _family, d, width, height, steps = _model_info(model_id)
            if model is None:
                raise gr.Error(t("Modèle indisponible."))
            try:
                base_seed = int(seed_v)
            except (TypeError, ValueError):
                base_seed = -1
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            logs: list[str] = []
            text = subject.strip()

            if use_enhance and tools.enhance_is_installed():
                yield (t("⏳ On cherche à quoi ressemblait ce moment…"),
                       gr.update(), gr.update(), "\n".join(logs))
                try:
                    # style « xanax » : le seul des trois qui parte d'une
                    # phrase de journal au lieu d'une description, et le seul
                    # qui n'ajoute PAS d'objectif, d'éclairage ni de
                    # composition — ici, une belle photo serait ratée.
                    out = tools.enhance_prompt_variants(
                        text, style="xanax", level="medium", variants=1,
                        style_constraint=XANAX_STYLE, log=logs.append)
                    if out and out[0].strip():
                        text = out[0].strip()
                except Exception as exc:  # noqa: BLE001
                    # Un échec de l'améliorateur ne doit pas empêcher de générer.
                    logs.append(f"[améliorateur indisponible] {exc}")

            full_prompt = build_prompt(text)
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    outs = gen_engine.generate(
                        model_id=model_id, prompt=full_prompt, negative="",
                        steps=steps,
                        cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
                        width=width, height=height, seed=base_seed,
                        batch_count=1, sampler=d.get("sampler") or "euler",
                        schedule=("" if d.get("scheduler") in (None, "", "auto")
                                  else d["scheduler"]),
                        log=q.put)
                    state["outs"] = [str(p) for p in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            step_re = re.compile(rf"(\d+)\s*/\s*{steps}\b")
            logs.append(f"Seed : {base_seed}")
            logs.append(f"Prompt : {full_prompt}")
            yield (t("⏳ Chargement du modèle…"), gr.update(), gr.update(),
                   "\n".join(logs))
            while True:
                line = q.get()
                if line is None:
                    break
                mt = step_re.search(line)
                if mt:
                    yield (t("🎨 Étape {cur}/{total}").format(
                        cur=min(int(mt.group(1)), steps), total=steps),
                        gr.update(), gr.update(), "\n".join(logs[-400:]))
                    continue
                if _PROGRESS_BAR.search(line) or "\x1b" in line:
                    continue
                logs.append(line)
                yield (gr.update(), gr.update(), gr.update(),
                       "\n".join(logs[-400:]))

            if "err" in state or not state.get("outs"):
                logs.append(f"\n[ERREUR] {state.get('err', 'aucune image')}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       gr.update(), "\n".join(logs[-400:]))
                return

            # Le prompt accompagne l'image, comme demandé par la consigne
            # « écris le prompt après chaque image ».
            yield (t("✅ Photo générée (seed {s})").format(s=base_seed),
                   gr.update(value=state["outs"][0]),
                   gr.update(value=f"**Prompt utilisé :** {full_prompt}"),
                   "\n".join(logs[-400:]))

        evt = run.click(do_xanax, inputs=[model_pick, prompt, enhance, seed],
                        outputs=[status, result, used_md, log])
        # QOL : Ctrl+Entrée depuis le champ de saisie lance la génération.
        prompt.submit(do_xanax, inputs=[model_pick, prompt, enhance, seed],
                      outputs=[status, result, used_md, log])
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])
