"""Onglet « 🎬 Video » — Wan 2.2 TI2V 5B, nativement dans sd.cpp.

Texte → vidéo et image → vidéo dans le MÊME modèle : la seule différence entre
les deux est qu'on a chargé une image de départ ou non. L'interface ne pose
donc pas la question — elle regarde le champ image.

Ce que cet écran refuse de faire, et pourquoi :

  · **Pas de curseur « durée en secondes ».** Le VAE temporel de Wan travaille
    par groupes de quatre images plus une. Un curseur en secondes afficherait
    « 2,0 s » et produirait 1,9 s. On propose donc les valeurs exactes, avec
    leur durée calculée à côté.
  · **Pas de preview pas à pas.** sd.cpp sait prévisualiser une vidéo, mais
    sur un plan de cinq secondes chaque aperçu coûte un décodage VAE complet —
    c'est-à-dire le poste le plus cher de toute la génération.
"""
from __future__ import annotations

import queue
import threading

import gradio as gr

from .. import registry, settings
from ..engine import sdcpp, video
from ..i18n import t
from . import widgets

#  Formats d'image proposés. Wan 2.2 TI2V 5B est entraîné en 704×1280 (et son
#  transposé) : c'est sa résolution native, celle où il est le meilleur. Le
#  480×832 est là pour essayer une idée vite — quatre fois moins de pixels par
#  image, donc à peu près quatre fois moins de temps.
SHAPES = (
    ("Portrait 704×1280 (native)", "704x1280"),
    ("Landscape 1280×704 (native)", "1280x704"),
    ("Portrait 480×832 (light, for trying things out)", "480x832"),
    ("Landscape 832×480 (light, for trying things out)", "832x480"),
)

#  Longueurs proposées, toutes exactement 4n+1. La durée affichée est donc la
#  vraie durée, pas un arrondi.
LENGTHS = (33, 65, 81, 121)


def _length_choices(fps: int = 24) -> list[tuple[str, int]]:
    return [(f"{n} frames — {n / fps:.1f} s", n) for n in LENGTHS]


def _model_note() -> str:
    """Ce qui manque pour générer, ou "" si tout est là."""
    engine = video.engine_ready()
    if engine:
        return engine
    if not video.is_ready():
        return t("⚠️ **The model is not downloaded yet.** Open **📚 Model "
                 "catalog** and download **Wan 2.2 TI2V 5B** — about 8.5 GB "
                 "in three pieces (diffusion, VAE, text encoder).")
    return ""


def build_video_tab(tab_id: str = "video", parent_tabs=None) -> None:
    with gr.Tab("🎬 Video", id=tab_id):
        prefs = settings.load_prefs()
        model = registry.get_base_model(video.default_model_id() or "", prefs)
        d = dict(model.defaults) if model else {}

        gr.Markdown(t(
            "**Wan 2.2 TI2V 5B** — text → video, or image → video if you load "
            "a starting frame.\nSame engine as the images, same GGUF weights, "
            "no extra install. Apache-2.0, so\nnothing stops you selling the "
            "result."))
        note = _model_note()
        gr.Markdown(note, visible=bool(note))

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="What happens in the shot", lines=3,
                    placeholder=t("a lovely cat stretching in the morning "
                                  "sun, slow camera push in"))
                start = gr.Image(
                    label="Starting frame (optional — leave empty for "
                          "text → video)",
                    type="filepath", height=200,
                    buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    shape = gr.Radio(
                        list(SHAPES), value="704x1280", label="Frame size")
                with gr.Row():
                    length = gr.Radio(
                        _length_choices(int(d.get("fps", 24) or 24)),
                        value=int(d.get("video_frames", 33) or 33),
                        label="Length")
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, scale=1,
                                     label="Seed (-1 = random)")
                    fmt = gr.Dropdown(
                        list(sdcpp.VIDEO_FORMATS), value="webm", scale=2,
                        label="Output file")

                with gr.Accordion(t("🎛️ Details (optional)"), open=False):
                    gr.Markdown(t(
                        "The defaults are the ones Wan's own documentation "
                        "uses. Change them to\nmeasure something, not on "
                        "principle."))
                    with gr.Row():
                        steps = gr.Slider(
                            4, 50, value=int(d.get("steps", 20) or 20), step=1,
                            label="Steps")
                        cfg = gr.Slider(
                            1.0, 10.0, value=float(d.get("cfg_scale", 6.0)),
                            step=0.5, label="Guidance (CFG)")
                    with gr.Row():
                        shift = gr.Slider(
                            0.0, 12.0,
                            value=float(d.get("flow_shift", 3.0) or 3.0),
                            step=0.5, label="Flow shift",
                            info=t("Wan's documented value is 3.0. Higher "
                                   "moves more, and drifts more."))
                    negative = gr.Textbox(
                        label="What to keep out of the shot",
                        value=video.WAN_NEGATIVE, lines=3,
                        info=t("Wan's own negative prompt, in Chinese "
                               "because the model was trained on it — "
                               "translating it changes what it points at."))

                with gr.Row():
                    run = gr.Button("🎬 Generate the video",
                                    variant="primary", scale=3)
                    stop = gr.Button("⏹️ Cancel", variant="stop", scale=1)
                status = gr.Markdown("")

            with gr.Column(scale=4):
                out_video = gr.Video(label="Result", height=420)
                out_file = gr.File(label="File produced", interactive=False)
                log = gr.Textbox(label="Log", lines=14, autoscroll=True,
                                 elem_classes="log-box")

        def do_video(prompt_val, start_val, shape_val, length_val, seed_val,
                     fmt_val, steps_val, cfg_val, shift_val, negative_val):
            if not (prompt_val or "").strip():
                raise gr.Error(t("Describe what should happen in the shot."))
            blocked = _model_note()
            if blocked:
                raise gr.Error(blocked.replace("**", ""))
            width, _, height = str(shape_val or "704x1280").partition("x")

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    state["out"] = video.generate_video(
                        prompt=prompt_val, negative=negative_val or "",
                        steps=int(steps_val), cfg_scale=float(cfg_val),
                        width=int(width), height=int(height),
                        seed=int(seed_val if seed_val is not None else -1),
                        video_frames=int(length_val),
                        flow_shift=float(shift_val),
                        start_image=(start_val or None),
                        output_format=fmt_val or "webm",
                        log=q.put)
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield (t("⏳ Generating — a video is many images, this takes "
                     "minutes, not seconds."),
                   gr.update(), gr.update(), gr.update())
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield (gr.update(), gr.update(), gr.update(),
                       "\n".join(logs[-500:]))

            if "err" in state:
                logs.append(f"\n[ERROR] {state['err']}")
                yield (t("❌ Failed — see the log."), gr.update(),
                       gr.update(), "\n".join(logs))
                return
            out = state["out"]
            #  Une suite d'images n'est pas lisible par le lecteur vidéo, et
            #  un dossier n'est pas téléchargeable : on le dit plutôt que de
            #  laisser deux cadres vides à interpréter.
            if out.is_dir():
                yield (t("✅ {n} frames written to `{path}`").format(
                           n=len(list(out.glob('frame_*.png'))), path=out),
                       gr.update(value=None), gr.update(value=None),
                       "\n".join(logs))
                return
            yield (t("✅ Video ready: {name}").format(name=out.name),
                   gr.update(value=str(out)), gr.update(value=str(out)),
                   "\n".join(logs))

        gen_evt = run.click(
            do_video,
            inputs=[prompt, start, shape, length, seed, fmt,
                    steps, cfg, shift, negative],
            outputs=[status, out_video, out_file, log])
        widgets.stop_into_status(stop, sdcpp.cancel_active, status, [gen_evt])
