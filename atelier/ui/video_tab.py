"""Onglet « 🎬 Vidéo » : LTX-2.3 (Lightricks) via stable-diffusion.cpp
(« -M vid_gen »). Texte → vidéo, image → vidéo, et début → fin.

Le clip sort en .webm avec sa **bande-son** : LTX génère l'image ET l'audio, et
sd.cpp les muxe dans le même fichier.

⚠️ C'est de très loin le plus lourd de l'application : diffusion 22 B + encodeur
Gemma-3-12B. Sur 11-12 Go de VRAM ça tourne uniquement par décharge en RAM, à
plusieurs minutes le clip. L'onglet dit franchement à quoi s'attendre plutôt que
de laisser croire à un bouton magique.
"""
from __future__ import annotations

import queue
import random
import threading

import gradio as gr

from .. import registry, settings
from ..engine import sdcpp
from ..engine import video as vid
from ..i18n import t

# Formats : tous alignés sur 32 px, donc rendus tels quels sans surprise.
# (720p « vrai » n'existe pas ici : 720 n'est pas un multiple de 32 et sd.cpp
# le ramènerait silencieusement à 704.)
FORMATS = [
    ("Paysage 16:9 — 1280×704 (lourd)", (1280, 704)),
    ("Paysage 16:9 — 960×544", (960, 544)),
    ("Paysage 16:9 — 704×384 (léger, à essayer en premier)", (704, 384)),
    ("Carré 1:1 — 768×768", (768, 768)),
    ("Portrait 9:16 — 544×960", (544, 960)),
    ("Portrait 9:16 — 384×704 (léger)", (384, 704)),
]
_DEFAULT_FORMAT = "704x384"
_DEFAULT_DURATION = 1.5
_DEFAULT_FPS = 24

MODES = [
    ("📝 Texte → vidéo", "t2v"),
    ("🖼️ Image → vidéo", "i2v"),
    ("🎞️ Début → fin", "flf2v"),
]


# Les libellés de menu sont traduits, PAS les valeurs : la traduction d'après
# construction ne touche pas aux `choices`, et une valeur traduite deviendrait
# une clé introuvable dès qu'on passe l'interface en anglais.
def _format_choices() -> list[tuple[str, str]]:
    return [(t(lbl), f"{w}x{h}") for lbl, (w, h) in FORMATS]


def _mode_choices() -> list[tuple[str, str]]:
    return [(t(lbl), val) for lbl, val in MODES]


def _size(fmt_value: str) -> tuple[int, int]:
    """« 1280x704 » -> (1280, 704). Repli sur le format léger si illisible."""
    try:
        w, h = str(fmt_value).lower().split("x")
        return int(w), int(h)
    except (AttributeError, TypeError, ValueError):
        return FORMATS[2][1]


def _recap_text(fmt_value, dur, fps_v, hires_on) -> str:
    """Ce qui sera RÉELLEMENT produit, une fois aligné sur les grilles du modèle.

    La demande (« 2 secondes en 720p ») ne correspond presque jamais à ce que
    LTX sait faire : 32 px de grille et des paquets de 8 images + 1. On l'affiche
    AVANT de lancer plusieurs minutes de calcul plutôt que de le laisser
    découvrir après coup."""
    w, h = _size(fmt_value)
    f = int(fps_v or _DEFAULT_FPS)
    p = vid.plan(w, h, round(float(dur or _DEFAULT_DURATION) * f), f)
    txt = f"→ {vid.describe(p)}"
    if hires_on:
        txt += t(" · reprise ×2 → {w}×{h}").format(w=p["width"] * 2,
                                                   h=p["height"] * 2)
    return txt


def _model_choices() -> tuple[list[tuple[str, str]], str | None]:
    """(choix du menu, modèle présélectionné : un installé, sinon le distillé)."""
    prefs = settings.load_prefs()
    choices, first_ready = [], None
    for m in vid.available_models(prefs):
        ready = registry.model_is_ready(m)
        if ready and first_ready is None:
            first_ready = m.id
        choices.append((m.name if ready else f"{m.name} — non installé", m.id))
    default = first_ready or (choices[0][1] if choices else None)
    return choices, default


def _defaults(model_id: str | None) -> dict:
    m = registry.get_base_model(model_id, settings.load_prefs()) \
        if model_id else None
    d = dict(m.defaults) if m else {}
    return {"steps": int(d.get("steps", 8) or 8),
            "cfg_scale": float(d.get("cfg_scale", 1.0) or 1.0),
            "negative": d.get("negative", "") or "",
            "supports_negative": bool(d.get("supports_negative", False)),
            "model": m}


def _state_note(model_id: str | None) -> str:
    """Ce qui manque pour pouvoir générer : moteur, puis poids."""
    ok, why = vid.engine_ready()
    if not ok:
        return t("❌ **Moteur** — {why}").format(why=why)
    m = registry.get_base_model(model_id, settings.load_prefs()) \
        if model_id else None
    if m is None:
        return t("❌ Aucun modèle vidéo au catalogue.")
    absent = vid.missing_parts(m)
    if absent:
        return t("⬇️ **« {name} » n'est pas installé** — il manque : "
                 "{parts}.  \nTéléchargez-le depuis l'onglet "
                 "**📚 Catalogue de modèles** (~25 Go, c'est long)."
                 ).format(name=m.name, parts=", ".join(absent))
    return t("✅ **« {name} » est prêt** ({up} upscaler latent ×2). "
             "Comptez plusieurs minutes par clip.").format(
        name=m.name, up=t("avec") if vid.spatial_upscaler(m) else t("sans"))


def build_video_tab(tab_id="video"):
    with gr.Tab("🎬 Vidéo (LTX-2.3)", id=tab_id):
        gr.Markdown(
            "### Génération vidéo — LTX-2.3, natif sd.cpp\n"
            "**Texte → vidéo**, **image → vidéo** (anime une image fixe) ou "
            "**début → fin** (deux images, le modèle fabrique l'entre-deux). "
            "Le clip sort en `.webm` **avec sa bande-son** : LTX génère aussi "
            "l'audio.\n\n"
            "> ⚠️ **C'est lourd et c'est lent.** Diffusion 22 B + encodeur "
            "Gemma-3-12B : ~25 Go à télécharger, et sur une carte 11-12 Go ça "
            "ne tient que par décharge en RAM (32 Go minimum, 64 Go "
            "confortable). Comptez **plusieurs minutes par clip** — ce n'est "
            "pas un plantage. Commencez en **704×384 sur 2 secondes** avant de "
            "monter quoi que ce soit.")

        _choices, _first = _model_choices()
        state_md = gr.Markdown(_state_note(_first), elem_classes="feedback")

        with gr.Row():
            with gr.Column(scale=3):
                model = gr.Dropdown(_choices, value=_first, label="Modèle",
                                    info="La version distillée (8 pas) est la "
                                         "seule raisonnable en 11-12 Go.")
                mode = gr.Radio(_mode_choices(), value="t2v", label="Mode")
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="Décrivez la scène ET le mouvement : « a red "
                                "fox walking through tall grass, camera slowly "
                                "pushing in »…",
                    info="En anglais de préférence. Décrire le MOUVEMENT "
                         "(caméra, sujet) compte autant que le décor.")
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      visible=False)
                with gr.Row():
                    init_image = gr.Image(label="Image de départ", type="pil",
                                          visible=False)
                    end_image = gr.Image(label="Image de fin", type="pil",
                                         visible=False)
                with gr.Row():
                    run = gr.Button("🎬 Générer la vidéo", variant="primary",
                                    scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
                status = gr.Markdown("")

                fmt = gr.Dropdown(_format_choices(), value=_DEFAULT_FORMAT,
                                  label="Format")
                with gr.Row():
                    duration = gr.Slider(0.5, 5.0, value=_DEFAULT_DURATION,
                                         step=0.5, label="Durée (secondes)")
                    fps = gr.Dropdown([16, 24, 30], value=_DEFAULT_FPS,
                                      label="Images/s")
                plan_md = gr.Markdown(
                    _recap_text(_DEFAULT_FORMAT, _DEFAULT_DURATION,
                                _DEFAULT_FPS, False),
                    elem_classes="hint")

                with gr.Accordion("Réglages avancés", open=False):
                    _d = _defaults(_first)
                    steps = gr.Slider(4, 40, value=_d["steps"], step=1,
                                      label="Étapes",
                                      info="Ajusté automatiquement au modèle "
                                           "choisi.")
                    cfg_s = gr.Slider(1.0, 12.0, value=_d["cfg_scale"],
                                      step=0.5, label="CFG",
                                      info="1.0 sur la version distillée : "
                                           "elle est entraînée pour ça.")
                    hires = gr.Checkbox(
                        value=False,
                        label="🔍 Détail ×2 (upscaler latent LTX)",
                        info="Passe de reprise à résolution doublée. Nettement "
                             "plus net, mais nettement plus long et plus "
                             "gourmand : à garder pour la fin.")
                    hires_steps = gr.Slider(2, 12, value=4, step=1,
                                            label="Étapes de la reprise ×2")
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = aléatoire)")
            with gr.Column(scale=4):
                result = gr.Video(label="Vidéo", height=420)
                logbox = gr.Textbox(label="Journal", lines=16, autoscroll=True,
                                    elem_classes="log-box")

        # Récapitulatif VIVANT (cf. _recap_text) : mis à jour à chaque réglage.
        for comp in (fmt, duration, fps, hires):
            comp.change(_recap_text, inputs=[fmt, duration, fps, hires],
                        outputs=[plan_md])

        # Le modèle choisi décide des étapes, du CFG et de l'existence même du
        # prompt négatif (ignoré à CFG 1.0 sur les modèles distillés).
        def _on_model(mid):
            d = _defaults(mid)
            return (gr.update(value=d["steps"]),
                    gr.update(value=d["cfg_scale"]),
                    gr.update(visible=d["supports_negative"],
                              value=d["negative"]),
                    gr.update(value=_state_note(mid)))

        model.change(_on_model, inputs=[model],
                     outputs=[steps, cfg_s, negative, state_md])

        def _on_mode(m):
            return (gr.update(visible=(m in ("i2v", "flf2v"))),
                    gr.update(visible=(m == "flf2v")))

        mode.change(_on_mode, inputs=[mode], outputs=[init_image, end_image])

        def do_video(model_id, mode_v, prompt_v, negative_v, img_a, img_b,
                     fmt_value, dur, fps_v, steps_v, cfg_v, hires_v,
                     hires_steps_v, seed_v):
            if not model_id:
                raise gr.Error(t("Choisissez un modèle."))
            if not (prompt_v or "").strip():
                raise gr.Error(t("Saisissez un prompt."))
            if mode_v in ("i2v", "flf2v") and img_a is None:
                raise gr.Error(t("Fournissez l'image de départ."))
            if mode_v == "flf2v" and img_b is None:
                raise gr.Error(t("Fournissez l'image de fin."))

            settings.ensure_dirs()
            w, h = _size(fmt_value)
            fps_i = int(fps_v or _DEFAULT_FPS)
            p = vid.plan(w, h,
                         round(float(dur or _DEFAULT_DURATION) * fps_i), fps_i)

            ip = ep = None
            if mode_v in ("i2v", "flf2v") and img_a is not None:
                ip = settings.TMP_DIR / "ltx_start.png"
                img_a.convert("RGB").save(ip)
            if mode_v == "flf2v" and img_b is not None:
                ep = settings.TMP_DIR / "ltx_end.png"
                img_b.convert("RGB").save(ep)

            try:
                s = int(seed_v)
            except (TypeError, ValueError):
                s = -1
            if s < 0:
                s = random.randint(0, 2**31 - 1)

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    state["out"] = vid.generate_video(
                        model_id=model_id, prompt=prompt_v,
                        negative=negative_v or "", mode=mode_v,
                        init_image=ip, end_image=ep,
                        width=p["width"], height=p["height"],
                        frames=p["frames"], fps=p["fps"],
                        steps=int(steps_v), cfg_scale=float(cfg_v), seed=s,
                        hires=bool(hires_v), hires_steps=int(hires_steps_v),
                        log=q.put)
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            mode_label = dict((v, k) for k, v in MODES).get(mode_v, mode_v)
            logs = [f"Mode : {mode_label}",
                    f"Modèle : {model_id}",
                    f"Vidéo : {vid.describe(p)}",
                    f"Seed : {s}",
                    f"{int(steps_v)} pas · CFG {float(cfg_v)}"
                    + (" · reprise ×2" if hires_v else "")]
            yield (t("⏳ Génération vidéo en cours — plusieurs minutes, "
                     "c'est normal…"), gr.update(), "\n".join(logs))
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-500:])

            if "err" in state or not state.get("out"):
                logs.append(f"\n[ERREUR] {state.get('err', 'aucune sortie')}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       "\n".join(logs))
                return

            out = state["out"]
            logs.append(f"\n✅ Vidéo : {out.name}")
            yield (t("✅ Vidéo générée : {n}").format(n=out.name),
                   gr.update(value=str(out)), "\n".join(logs))

        evt = run.click(
            do_video,
            inputs=[model, mode, prompt, negative, init_image, end_image,
                    fmt, duration, fps, steps, cfg_s, hires, hires_steps, seed],
            outputs=[status, result, logbox])
        stop.click(lambda: vid.cancel(), outputs=None, cancels=[evt])
