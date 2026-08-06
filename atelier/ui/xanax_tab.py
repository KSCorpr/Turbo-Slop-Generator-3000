"""Onglets « Xanax » : le style est CODÉ EN DUR, il n'y a rien à régler.

Transcription du system prompt Gemini de l'utilisateur. Contrairement aux
onglets de génération classiques, il n'y a ici ni menu de préréglages, ni
banques de styles, ni prompt système modifiable : le style est « fixe et non
négociable », donc il n'est pas exposé. On écrit une phrase, on obtient trois
photos.

Trois points de traduction méritent d'être expliqués, parce qu'ils ne sont pas
évidents en lisant le system prompt d'origine :

1. **L'anti-collage vit dans le prompt POSITIF.** Le system prompt interdit les
   collages/diptyques. Le réflexe serait de le mettre en négatif — sauf que les
   deux modèles sont distillés à CFG 1.0 et *ignorent le prompt négatif*. La
   consigne serait donc silencieusement perdue. Elle est intégrée au style.

2. **« 3 images » = 3 GÉNÉRATIONS séparées**, jamais un batch. Un batch produit
   trois variations du même cadrage ; le system prompt demande explicitement des
   angles/distances/moments différents. Chaque image reçoit donc sa propre
   directive de cadrage, et une graine différente.

3. **La traduction FR→EN a besoin de l'améliorateur.** Le style est un préfixe
   anglais collé devant le texte : à lui seul il ne traduit rien. Si
   l'améliorateur est installé on s'en sert (avec le style en contrainte), sinon
   on le dit clairement au lieu de laisser croire que ça marche.
"""
from __future__ import annotations

import queue
import random
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
    "no filter, no post-processing, "
    # Anti-collage : ici et pas en négatif — voir l'en-tête du module.
    "a single standalone photograph, one single frame, "
    "not a collage, not a diptych, not a triptych, not a split image"
)

# Les trois prises de vue. Même sujet, même scène : ce qui change est l'angle,
# la distance, le cadrage et l'instant — c'est la demande du system prompt
# (« cohérentes comme série, sans être identiques »).
XANAX_SHOTS = [
    ("Plan large",
     "wide shot, subject small within its surroundings, plenty of ordinary "
     "context around it, taken a few steps back"),
    ("Plan moyen",
     "medium shot at eye level, subject framed from the waist up or at "
     "mid-distance, slightly off-centre and carelessly framed"),
    ("Plan rapproché",
     "closer tighter view on the same scene a moment later, focusing on one "
     "detail of it, shot from a slightly different angle"),
]

# Format 4:3 imposé par le system prompt, sur la grille NATIVE de chaque modèle
# (32 px pour Flux.2, 64 px pour Krea 2 — sortir de la grille dégrade le rendu).
XANAX_SIZE = {"flux2": (1184, 880), "krea2": (1152, 896)}


def _size_for(family: str) -> tuple[int, int]:
    return XANAX_SIZE.get(family, (1184, 880))


def _build_prompts(subject: str) -> list[tuple[str, str]]:
    """(libellé du plan, prompt complet) pour les trois photos."""
    base = (subject or "").strip().strip(",")
    return [(label, f"{XANAX_STYLE}, {base}, {shot}".strip(", "))
            for label, shot in XANAX_SHOTS]


def build_xanax_tab(model_id: str, title: str):
    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    family = model.family if model else "flux2"
    d = dict(model.defaults) if model else {}
    width, height = _size_for(family)

    with gr.Tab(title):
        ready = model is not None and registry.model_is_ready(model)
        status_line = (t("● modèle prêt") if ready
                       else t("○ à télécharger (onglet Catalogue de modèles)"))
        gr.Markdown(
            f"### {title} — {status_line}\n"
            "Écrivez une phrase ou un thème. L'onglet produit **3 photos** de "
            "la même scène, sous des angles et des cadrages différents.\n\n"
            "**Le style est figé et non modifiable** : photo amateur, France "
            "provinciale, 1995-2005, temps couvert, aucun post-traitement, "
            f"format 4:3 ({width}×{height}). C'est le principe de cet onglet — "
            "pour régler quoi que ce soit, utilisez l'onglet de génération "
            "normal.")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Votre phrase ou thème", lines=3,
                    placeholder="un homme qui attend le bus devant un "
                                "supermarché…")
                with gr.Row(elem_classes="go-row"):
                    run = gr.Button("📷 Générer les 3 photos",
                                    variant="primary", size="lg", scale=4)
                    stop = gr.Button("⏹️ Stop", variant="stop", scale=1,
                                     min_width=90)
                status = gr.Markdown("")
                enhance = gr.Checkbox(
                    value=tools.enhance_is_installed(),
                    interactive=tools.enhance_is_installed(),
                    label="✨ Traduire et étoffer ma phrase (améliorateur IA)",
                    info=("Traduit le français en anglais et enrichit la "
                          "description, en respectant le style imposé."
                          if tools.enhance_is_installed() else
                          "Améliorateur non installé — installez-le depuis un "
                          "onglet de génération. Sans lui, écrivez en ANGLAIS : "
                          "le style est un préfixe anglais, il ne traduit rien."))
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = aléatoire)",
                                 info="Une graine fixe rejoue exactement la "
                                      "même série de 3 photos.")
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Les 3 photos", columns=3,
                                     height=460, object_fit="contain",
                                     show_download_button=True)
                log = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        def do_xanax(subject, use_enhance, seed_v):
            if not (subject or "").strip():
                raise gr.Error(t("Écrivez une phrase ou un thème."))
            settings.ensure_dirs()
            try:
                base_seed = int(seed_v)
            except (TypeError, ValueError):
                base_seed = -1
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            logs: list[str] = []
            text = subject.strip()

            # Traduction/étoffement AVANT de fabriquer les trois prompts : les
            # trois photos doivent décrire le MÊME sujet, sinon la série perd
            # sa cohérence.
            if use_enhance and tools.enhance_is_installed():
                yield (t("⏳ Traduction et mise en forme de la phrase…"),
                       gr.update(), "\n".join(logs))
                try:
                    out = tools.enhance_prompt_variants(
                        text, style=("krea2" if family == "krea2" else "generic"),
                        level="medium", variants=1,
                        style_constraint=XANAX_STYLE, log=logs.append)
                    if out and out[0].strip():
                        text = out[0].strip()
                        logs.append(f"Sujet retenu : {text}")
                except Exception as exc:  # noqa: BLE001
                    # Un échec de l'améliorateur ne doit pas empêcher de générer.
                    logs.append(f"[améliorateur indisponible] {exc}")

            shots = _build_prompts(text)
            results: list = []

            for i, (label, full_prompt) in enumerate(shots):
                q: "queue.Queue[str | None]" = queue.Queue()
                state: dict = {}
                shot_seed = base_seed + i

                def worker(p=full_prompt, s=shot_seed, qq=q, st=state):
                    try:
                        outs = gen_engine.generate(
                            model_id=model_id, prompt=p, negative="",
                            steps=int(d.get("steps", 8) or 8),
                            cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
                            width=width, height=height, seed=s, batch_count=1,
                            sampler=d.get("sampler") or "euler",
                            schedule=("" if d.get("scheduler") in
                                      (None, "", "auto") else d["scheduler"]),
                            log=qq.put)
                        st["outs"] = [str(x) for x in outs]
                    except Exception as exc:  # noqa: BLE001
                        st["err"] = str(exc)
                    finally:
                        qq.put(None)

                threading.Thread(target=worker, daemon=True).start()
                logs.append(f"\n── Photo {i + 1}/3 — {label} (seed {shot_seed})")
                yield (t("📷 Photo {n}/3 — {label}…").format(n=i + 1,
                                                            label=label),
                       gr.update(), "\n".join(logs[-400:]))
                while True:
                    line = q.get()
                    if line is None:
                        break
                    logs.append(line)
                    yield gr.update(), gr.update(), "\n".join(logs[-400:])

                if "err" in state or not state.get("outs"):
                    logs.append(f"[ERREUR] {state.get('err', 'aucune image')}")
                    yield (t("❌ Échec sur la photo {n}/3 — voir le "
                             "journal.").format(n=i + 1),
                           gr.update(value=results), "\n".join(logs[-400:]))
                    return
                # Le prompt accompagne l'image, comme demandé par la consigne
                # « écris le prompt après chaque image ».
                results.append((state["outs"][0], f"{label} — {full_prompt}"))
                yield (gr.update(), gr.update(value=results),
                       "\n".join(logs[-400:]))

            yield (t("✅ 3 photos générées (seeds {a} à {b})").format(
                a=base_seed, b=base_seed + 2),
                gr.update(value=results), "\n".join(logs[-400:]))

        evt = run.click(do_xanax, inputs=[prompt, enhance, seed],
                        outputs=[status, gallery, log])
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])
