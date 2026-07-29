"""Onglet « 🖼️ Outpaint » : étendre une image à gauche / à droite / en haut /
en bas — ou tout autour — façon Midjourney.

Fonctionne avec **n'importe quel modèle du catalogue** et **sans prompt** :
l'extension repose sur un remplissage miroir + img2img, puis l'original est
recollé par-dessus (voir engine/outpaint.py). Aucun modèle d'inpainting requis.
"""
from __future__ import annotations

import queue
import random
import threading
import time

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import outpaint as op
from ..i18n import t

PRESETS = [
    ("← Gauche", ["left"]),
    ("→ Droite", ["right"]),
    ("↑ Haut", ["top"]),
    ("↓ Bas", ["bottom"]),
    ("↔ Horizontal", ["left", "right"]),
    ("↕ Vertical", ["top", "bottom"]),
    ("⤢ Tout autour", ["left", "right", "top", "bottom"]),
]


def _models() -> list:
    return registry.load_base_models(settings.load_prefs())


def _model_choices() -> tuple[list[tuple[str, str]], str | None]:
    """(choix du menu, modèle installé à présélectionner)."""
    choices, first_ready = [], None
    for m in _models():
        ready = registry.model_is_ready(m)
        if ready and first_ready is None:
            first_ready = m.id
        choices.append((m.name if ready else f"{m.name} — non installé", m.id))
    return choices, first_ready


def _defaults(model_id: str | None) -> dict:
    """Réglages d'échantillonnage propres au modèle choisi (catalogue)."""
    m = registry.get_base_model(model_id, settings.load_prefs()) if model_id \
        else None
    d = dict(m.defaults) if m else {}
    return {"steps": int(d.get("steps", 8) or 8),
            "cfg_scale": float(d.get("cfg_scale", 1.0) or 1.0),
            "sampler": d.get("sampler") or "euler",
            "schedule": d.get("scheduler") or "auto",
            "flow_shift": float(d.get("flow_shift", 0.0) or 0.0)}


def build_outpaint_tab(tab_id="outpaint", pending_outpaint=None, tabs=None):
    with gr.Tab("🖼️ Outpaint", id=tab_id):
        gr.Markdown(
            "### Étendre une image (outpaint)\n"
            "Agrandit la toile dans les directions choisies et laisse le modèle "
            "**inventer la suite**. Fonctionne avec **tous les modèles** du "
            "catalogue et **sans prompt** : les bords sont pré-remplis en "
            "**miroir** (continuité naturelle), le modèle harmonise, puis "
            "**l'image d'origine est recollée intacte** par-dessus.\n\n"
            "💡 Le prompt est **facultatif** — utile seulement pour orienter ce "
            "qui apparaît dans la nouvelle zone.")

        _choices, _first = _model_choices()
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à étendre", type="pil")
                direction = gr.Radio(
                    [(lbl, "|".join(d)) for lbl, d in PRESETS],
                    value="|".join(PRESETS[-1][1]), label="Direction")
                amount = gr.Slider(
                    0.05, 1.0, value=0.25, step=0.05,
                    label="Extension par côté (proportion de l'image)",
                    info="0.25 = +25 % de chaque côté choisi. La toile est "
                         "alignée sur 16 px et plafonnée à 2048 px.")
                plan_md = gr.Markdown("")
                model = gr.Dropdown(_choices, value=_first,
                                    label="Modèle utilisé")
                prompt = gr.Textbox(
                    label="Prompt (facultatif)", lines=2,
                    placeholder="Laisse vide pour une extension neutre…")
                with gr.Accordion("Réglages avancés", open=False):
                    strength = gr.Slider(
                        0.3, 1.0, value=0.85, step=0.05,
                        label="Force de génération sur la nouvelle zone",
                        info="Haut = invente librement. Bas = reste proche du "
                             "remplissage miroir.")
                    feather = gr.Slider(
                        0, 96, value=24, step=4,
                        label="Fondu de raccord (px)",
                        info="Adoucit la jonction avec l'image d'origine. "
                             "0 = collage net.")
                    steps = gr.Slider(
                        1, 40, value=_defaults(_first)["steps"], step=1,
                        label="Étapes",
                        info="Ajusté automatiquement au modèle choisi.")
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = aléatoire)")
                with gr.Row():
                    run = gr.Button("🖼️ Étendre l'image", variant="primary",
                                    scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
                status = gr.Markdown("")
            with gr.Column(scale=4):
                result = gr.Image(label="Résultat", type="filepath",
                                  show_download_button=True)
                again = gr.Button("♻️ Ré-étendre le résultat", size="sm")
                log = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        # --- Aperçu du plan (dimensions) ---
        def _preview(img, dirs, amt):
            if img is None:
                return ""
            p = op.plan(img.size, (dirs or "").split("|"), float(amt))
            return f"**Nouvelle taille :** {op.describe(p)}"

        for comp in (image, direction, amount):
            comp.change(_preview, inputs=[image, direction, amount],
                        outputs=[plan_md])

        # Le nombre d'étapes suit le modèle (4 pour Flux.2, 8 pour Krea 2…).
        model.change(lambda mid: gr.update(value=_defaults(mid)["steps"]),
                     inputs=[model], outputs=[steps])

        def do_outpaint(img, dirs, amt, model_id, prompt_txt, strength_v,
                        feather_v, steps_v, seed_v):
            if img is None:
                raise gr.Error(t("Chargez une image à étendre."))
            if not model_id:
                raise gr.Error(t("Choisissez un modèle."))
            p = op.plan(img.size, (dirs or "").split("|"), float(amt))
            if not any(p[d] for d in op.DIRECTIONS):
                raise gr.Error(t("Aucune direction sélectionnée."))

            settings.ensure_dirs()
            canvas = op.build_canvas(img, p)
            init_path = settings.TMP_DIR / "outpaint_init.png"
            canvas.save(init_path)

            try:
                s = int(seed_v)
            except (TypeError, ValueError):
                s = -1
            if s < 0:
                s = random.randint(0, 2**31 - 1)
            d = _defaults(model_id)

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    outs = gen_engine.generate(
                        model_id=model_id, prompt=prompt_txt or "",
                        negative="", steps=int(steps_v),
                        cfg_scale=d["cfg_scale"],
                        width=p["width"], height=p["height"], seed=s,
                        batch_count=1, sampler=d["sampler"],
                        schedule=d["schedule"], flow_shift=d["flow_shift"],
                        init_image=init_path, strength=float(strength_v),
                        log=q.put, save_prompt=False)
                    state["outs"] = [str(x) for x in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs = [f"Plan : {op.describe(p)}", f"Seed : {s}",
                    f"Modèle : {model_id} · {d['sampler']} · "
                    f"cfg {d['cfg_scale']} · {int(steps_v)} pas"]
            yield t("⏳ Extension en cours…"), gr.update(), "\n".join(logs)
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-400:])

            if "err" in state or not state.get("outs"):
                logs.append(f"\n[ERREUR] {state.get('err', 'aucune sortie')}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       "\n".join(logs))
                return

            # Recollage de l'original : la zone d'origine reste intacte.
            from PIL import Image as _PI
            gen = _PI.open(state["outs"][0])
            final = op.composite_back(gen, img, p, feather=int(feather_v))
            out_path = settings.OUTPUT_DIR / \
                f"outpaint-{time.strftime('%Y%m%d-%H%M%S')}.png"
            final.save(out_path)
            _sidecar(out_path, p, model_id, prompt_txt, s, strength_v,
                     feather_v, steps_v, d)
            logs.append(f"\n✅ Original recollé (fondu {int(feather_v)} px) → "
                        f"{out_path.name}")
            yield (t("✅ Étendu : {n}").format(n=out_path.name),
                   gr.update(value=str(out_path)), "\n".join(logs))

        evt = run.click(
            do_outpaint,
            inputs=[image, direction, amount, model, prompt, strength, feather,
                    steps, seed],
            outputs=[status, result, log])
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])

        # Enchaîner : le résultat redevient l'image d'entrée (extensions
        # successives, comme dans Midjourney).
        def _again(path):
            if not path:
                return gr.update(), t("Rien à ré-étendre.")
            from PIL import Image as _PI
            return (gr.update(value=_PI.open(path)),
                    t("Résultat rechargé — choisis une direction."))

        again.click(_again, inputs=[result], outputs=[image, status])

        # Réception d'une image envoyée depuis un onglet de génération.
        if pending_outpaint is not None and tabs is not None:
            def _consume(pend):
                if not pend:
                    return gr.update(), None, gr.update()
                from PIL import Image as _PI
                im = _PI.open(pend)
                return gr.update(value=im), None, gr.update(value="")

            tabs.select(_consume, inputs=[pending_outpaint],
                        outputs=[image, pending_outpaint, status])


def _sidecar(out_path, p, model_id, prompt_txt, seed, strength_v, feather_v,
             steps_v, d) -> None:
    """Journal .txt à côté du PNG, comme pour les images et les GLB."""
    lines = [
        f"Fichier: {out_path.name}",
        f"Modèle: {model_id}",
        f"Prompt: {prompt_txt or '(aucun)'}",
        f"Extension: {op.describe(p)}",
        f"Seed: {seed}",
        f"Force: {strength_v}",
        f"Fondu: {int(feather_v)} px",
        f"Étapes: {int(steps_v)}  ·  Sampler: {d['sampler']}  ·  "
        f"CFG: {d['cfg_scale']}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    try:
        out_path.with_suffix(".txt").write_text("\n".join(lines),
                                                encoding="utf-8")
    except OSError:
        pass
