"""Image description, depth, background removal and click segmentation."""
from __future__ import annotations

import queue
import threading

import gradio as gr

from .. import model_catalog, registry, settings
from ..engine import tools
from ..i18n import t
from . import widgets
from . import preview
from .media_tabs import stream_task

SAM_CHOICES = [("SAM 2.1 small", "sam2"), ("SAM 1 base (legacy)", "sam")]


def _sam_installer(name, title):
    button = gr.Button("Install / repair "+title, size="sm")
    log = gr.Textbox(label=title+" install log", lines=3, autoscroll=True)

    def install_sam2():
        for status, _, lines in stream_task(lambda logger: model_catalog.install("addon:" + name, log=logger)):
            yield status+"\n"+lines
    button.click(install_sam2, outputs=log, **widgets.GPU_QUEUE)

# Titre de l'aperçu de secours, partagé par les outils où l'on CLIQUE sur
# l'image : c'est là que ne pas la voir est bloquant, pas seulement gênant.
_FALLBACK_TITLE = "🖼️ Fallback preview (if the image will not display)"


def _installer_block(title: str, note: str, stream_fn, installed: bool):
    """Installation 1 clic, commune aux outils.

    Un bloc d'installation ne sert QU'UNE FOIS. Le laisser en permanence dans
    l'onglet, c'est faire payer à vie un accordéon de plus à quelqu'un qui a
    déjà tout installé — et ils sont sept. Quand l'outil est là, le bloc
    disparaît ; quand il ne l'est pas, il est déplié d'emblée, parce qu'à ce
    moment-là c'est la seule chose à faire dans cet onglet.

    On garde `visible=` plutôt qu'un `if` : le bloc doit exister dans l'arbre
    pour pouvoir réapparaître si l'installation échoue plus tard.
    """
    # Déjà installé : une seule petite ligne, qui redonne accès au bloc si
    # l'installation est à refaire. Sans elle, « réparer » deviendrait
    # impossible depuis l'interface.
    # `title` est un nom d'outil français fourni par l'appelant : il se traduit
    # séparément, sinon seule l'enveloppe passerait en anglais.
    repair = gr.Button(
        t("⚙️ Reinstall / repair {title}").format(title=t(title)),
        size="sm", variant="secondary", visible=installed)
    with gr.Accordion(
            t("⚙️ Install {title} (1 click)").format(title=t(title)),
            open=not installed, visible=not installed) as box:
        gr.Markdown(note)
        log = gr.Textbox(label="Install log", lines=10,
                         autoscroll=True, elem_classes="log-box")
        btn = gr.Button(t("⬇️ Install {title}").format(title=t(title)))

        def _install():
            for msg in stream_fn():
                yield msg

        btn.click(_install, outputs=[log])

    repair.click(lambda: (gr.update(visible=True, open=True),
                          gr.update(visible=False)),
                 outputs=[box, repair])
    return box


def build_toolkit_tab(tab_id="toolkit", pending_toolkit=None, tabs=None,
                      parent_tabs=None, prompt_boxes=None, generation_tabs=None):
    """`parent_tabs` : le groupe « 🧰 Outils » qui contient cet onglet.

    Depuis le regroupement des onglets, atteindre un outil demande DEUX
    sélections — le groupe à la racine, puis l'outil dedans. L'appelant fournit
    donc le conteneur, sinon l'image arriverait dans un onglet resté masqué."""
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Utility tools\n**Depth** map, **background removal** "
            "(transparent PNG) and **click-to-cutout** (Segment Anything).")

        with gr.Tabs() as sub_tabs:
            # ---------- Image -> prompt ----------
            # Placé EN TÊTE : c'est le seul outil du Toolkit qui ne transforme
            # pas l'image mais qui alimente la génération. Il vient donc avant
            # ceux qui la retouchent, dans l'ordre où on s'en sert.
            with gr.Tab("📝 Image → prompt", id="describe"):
                gr.Markdown(
                    "Hand it an image, get back the **prompt** that would "
                    "recreate it. This is not a caption: a vision model would "
                    "say “a photo of a cat on a sofa”, which pasted into the "
                    "Prompt field gives a flat image. Here we name the "
                    "**medium**, the **light**, the **lens**, the **palette** "
                    "and the **framing** — the words that actually steer "
                    "diffusion. Always in **English**: that is the models' "
                    "language.")
                _installer_block(
                    "Image → prompt",
                    "**Qwen2.5-VL-3B** vision-language model (~7.5 GB). "
                    "Loaded then unloaded on each call: **no VRAM conflict** "
                    "with generation. *Qwen Research* licence — non-commercial.",
                    tools.install_describe_stream, tools.describe_is_installed())

                with gr.Row():
                    with gr.Column(scale=2):
                        p_image = gr.Image(label="Image to read", type="pil",
                                           height=340,
                                           buttons=widgets.IMAGE_BUTTONS)
                        p_mode = gr.Radio(
                            [(t("📸 Recreate this image — subject AND style"),
                              "full"),
                             (t("🎨 Style only — to apply to something else"),
                              "style"),
                             (t("🔍 Plain description — what is in it"),
                              "plain")],
                            value="full", label="What you want out of it")
                        p_hint = gr.Markdown("", elem_classes="hint")
                        with gr.Row():
                            p_run = gr.Button("📝 Read the image",
                                              variant="primary")
                            p_stop = gr.Button("⏹️ Cancel", variant="stop")
                    with gr.Column(scale=3):
                        p_out = gr.Textbox(
                            label="Resulting prompt", lines=9,
                            buttons=widgets.TEXT_COPY,
                            placeholder="The text will appear here — read it "
                                        "before sending, it is a starting "
                                        "point, not a verdict.")
                        gr.Markdown(t(
                            "**Send it straight to a generation tab** — the "
                            "prompt replaces the field there, and you "
                            "generate whenever you like."))
                        with gr.Row():
                            p_to_krea = gr.Button("→ ⚡ Krea 2 Turbo")
                            p_to_flux = gr.Button("→ 🟣 Flux.2 Klein")
                        p_log = gr.Textbox(label="Log", lines=6,
                                           autoscroll=True,
                                           elem_classes="log-box")

                # Ce que le mode va faire, écrit AVANT le clic : « juste le
                # style » et « refaire cette image » ne se devinent pas d'après
                # leur libellé seul, et se tromper coûte une minute de calcul.
                _MODE_HINT = {
                    "full": "Subject, setting, light, colours, medium — "
                            "enough to recreate a close image on another "
                            "model.",
                    "style": "**Not a word about the subject**: not the cat, "
                             "not the car, not the place. Only the look, to "
                             "paste in front of your own subject.",
                    "plain": "Two or three sentences, no prompt vocabulary. "
                             "To know what is in the image, not to regenerate "
                             "it.",
                }

                def _mode_hint(mode):
                    return t(_MODE_HINT.get(mode, ""))

                p_mode.change(_mode_hint, inputs=[p_mode], outputs=[p_hint])

                def do_describe(image, mode):
                    if image is None:
                        raise gr.Error(t("Load an image first."))
                    if not tools.describe_is_installed():
                        raise gr.Error(t("Install “Image → prompt” first."))
                    q: "queue.Queue[str | None]" = queue.Queue()
                    state: dict = {}

                    def worker():
                        try:
                            state["out"] = tools.image_to_prompt(
                                image, mode=mode, log=q.put)
                        except Exception as exc:  # noqa: BLE001
                            state["err"] = str(exc)
                        finally:
                            q.put(None)

                    threading.Thread(target=worker, daemon=True).start()
                    logs: list[str] = []
                    while True:
                        line = q.get()
                        if line is None:
                            break
                        logs.append(line)
                        yield gr.update(), "\n".join(logs[-500:])
                    if "err" in state:
                        logs.append(f"\n[ERROR] {state['err']}")
                        yield gr.update(), "\n".join(logs[-500:])
                        return
                    yield state["out"][0], "\n".join(logs[-500:])

                p_evt = p_run.click(do_describe, inputs=[p_image, p_mode],
                                    outputs=[p_out, p_log], **widgets.GPU_QUEUE)
                widgets.stop_into_log(p_stop, tools.cancel, p_log, [p_evt])

                # Envoi vers un onglet de génération. On écrit DIRECTEMENT
                # dans son champ Prompt (l'appelant nous l'a passé) puis on
                # bascule. Le détour par un State consommé au changement
                # d'onglet ne marche pas : une sélection programmatique ne
                # déclenche pas `Tabs.select`, le texte n'arrivait jamais.
                def _send_to(model_id, button):
                    box = (prompt_boxes or {}).get(model_id)
                    if box is None or tabs is None:
                        button.visible = False
                        return

                    def _go(text):
                        if not (text or "").strip():
                            raise gr.Error(t("Read an image first."))
                        if generation_tabs is not None:
                            return (text.strip(), gr.Tabs(selected="create"),
                                    gr.Tabs(selected=model_id))
                        return text.strip(), gr.Tabs(selected=model_id)

                    targets = [box, tabs]
                    if generation_tabs is not None:
                        targets.append(generation_tabs)
                    button.click(_go, inputs=[p_out], outputs=targets,
                                 queue=False, show_progress="hidden")

                _send_to("krea2-turbo", p_to_krea)
                _send_to("flux2-klein-9b", p_to_flux)

            # ---------- Profondeur ----------
            with gr.Tab("🌐 Depth", id="depth"):
                gr.Markdown(
                    "*Depth Anything V2* — depth map (light = near, dark = "
                    "far). Download the result to reuse it.")
                _installer_block(
                    "Depth Anything V2",
                    "Uses PyTorch + transformers (~100 MB model). No command "
                    "to type.",
                    tools.install_depth_stream, tools.depth_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        d_image = gr.Image(label="Source image", type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        d_run = gr.Button("🌐 Generate depth",
                                          variant="primary", size="lg")
                    with gr.Column(scale=4):
                        d_result = gr.Image(label="Depth map", height=520,
                                            format="png",
                                            buttons=widgets.IMAGE_BUTTONS)
                        d_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_depth(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    logs: list[str] = []
                    progress(0.1, desc="Depth…")
                    try:
                        out = tools.depth_map(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    return str(out), "\n".join(logs)

                d_run.click(do_depth, inputs=[d_image], outputs=[d_result, d_log], **widgets.GPU_QUEUE)

            # ---------- Suppression d'arrière-plan ----------
            with gr.Tab("✂️ Background removal", id="bg"):
                gr.Markdown(
                    "*RMBG-1.4* — cuts out the subject and returns a "
                    "**transparent PNG**.  \n⚠️ **Non-commercial** license "
                    "(BRIA RMBG-1.4).")
                _installer_block(
                    "RMBG-1.4",
                    "Uses PyTorch + transformers (~176 MB model). No command "
                    "to type.",
                    tools.install_bg_stream, tools.bg_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        b_image = gr.Image(label="Source image", type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        b_run = gr.Button("✂️ Cut out", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        b_result = gr.Image(label="Cutout subject (transparent PNG)",
                                            height=520, format="png",
                                            buttons=widgets.IMAGE_BUTTONS,
                                            image_mode="RGBA")
                        b_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_bg(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    logs: list[str] = []
                    progress(0.1, desc="Cutting out…")
                    try:
                        out = tools.bg_remove(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    return str(out), "\n".join(logs)

                b_run.click(do_bg, inputs=[b_image], outputs=[b_result, b_log], **widgets.GPU_QUEUE)

            # ---------- Segment Anything (clic) ----------
            with gr.Tab("🪄 Cut out (SAM)", id="sam"):
                gr.Markdown(
                    "*Segment Anything* — **click an object**: SAM instantly "
                    "shows the **selected area highlighted**. Re-click to "
                    "adjust, then “Extract” for the **transparent PNG**.")
                s_backend = gr.Dropdown(SAM_CHOICES, value="sam2", label="Segmentation model")
                with gr.Group(visible=False) as s_legacy:
                    _installer_block("Segment Anything", "Legacy SAM 1 base (~375 MB weights).",
                                     tools.install_sam_stream, tools.sam_is_installed())
                with gr.Group() as s_modern:
                    _sam_installer("sam2", "SAM 2.1 small (184 MB weights)")
                s_backend.change(lambda backend: (gr.update(visible=backend == "sam"),
                                                  gr.update(visible=backend == "sam2")),
                                 inputs=s_backend, outputs=[s_legacy, s_modern], queue=False)

                s_cut = gr.State(None)     # chemin du découpage déjà calculé
                with gr.Row():
                    with gr.Column(scale=3):
                        s_image = gr.Image(label="Image — click the object",
                                           type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        s_info = gr.Markdown("Click a point on the image.")
                        s_overlay = gr.Image(label="Selected area (preview)",
                                             height=300, interactive=False,
                                             buttons=widgets.IMAGE_VIEW_ONLY)
                        s_run = gr.Button("🪄 Extract object", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        s_result = gr.Image(label="Extracted object (transparent PNG)",
                                            height=520, format="png",
                                            buttons=widgets.IMAGE_BUTTONS,
                                            image_mode="RGBA")
                        s_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def _on_click(img, backend, evt: gr.SelectData, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not tools.sam_is_installed(backend):
                        raise gr.Error("Install the selected segmentation model first.")
                    x, y = int(evt.index[0]), int(evt.index[1])
                    progress(0.2, desc="Segmentation…")
                    try:
                        cut, overlay = tools.sam_segment(img, x, y, backend=backend)
                    except Exception as exc:  # noqa: BLE001
                        raise gr.Error(str(exc))
                    progress(1.0, desc="Done")
                    return (overlay or gr.update(), str(cut),
                            t("Area selected at ({x}, {y}). Click “Extract” "
                              "or re-click elsewhere.").format(x=x, y=y))

                s_image.select(_on_click, inputs=[s_image, s_backend],
                               outputs=[s_overlay, s_cut, s_info], **widgets.GPU_QUEUE)

                # Aperçu de secours : si la vignette du composant reste une
                # icône cassée, l'image est quand même visible ici — les
                # pixels sont dans la page, aucune requête n'est faite (voir
                # atelier/ui/preview.py). Replié : doublon inutile quand
                # l'aperçu normal fonctionne.
                with gr.Accordion(_FALLBACK_TITLE, open=False):
                    s_fallback = gr.HTML(preview.html(None))
                s_image.change(preview.html, inputs=[s_image],
                               outputs=[s_fallback])

                def do_sam(cut):
                    if not cut:
                        raise gr.Error(t("Click an object in the image first."))
                    return str(cut), f"✅ {cut}"

                s_run.click(do_sam, inputs=[s_cut], outputs=[s_result, s_log])
                s_image.input(lambda: (None, None, None, "Click a point on the image."),
                              outputs=[s_cut, s_overlay, s_result, s_info], queue=False)
                s_backend.change(lambda: (None, None, None, "Click a point on the image."),
                                 outputs=[s_cut, s_overlay, s_result, s_info], queue=False)

        # --- Réception d'une image envoyée depuis un onglet de génération ---
        if pending_toolkit is not None and tabs is not None:
            _keys = ["depth", "bg", "sam"]

            def _consume(pend):
                # +2 sorties fixes : le groupe parent et le sélecteur d'outil.
                path, dest = pend
                sub = gr.Tabs(selected=dest) if dest in _keys else gr.update()
                # Remonter le groupe « Outils » : sans ça l'outil est bien
                # sélectionné, mais dans un onglet que personne n'affiche.
                top = (gr.Tabs(selected=tab_id) if parent_tabs is not None
                       else gr.update())
                img_upd = [path if k == dest else gr.update()
                           for k in _keys]
                return tuple([top, sub] + img_upd)

            _outs = [parent_tabs if parent_tabs is not None else sub_tabs,
                     sub_tabs, d_image, b_image, s_image]
            pending_toolkit.receive(_consume, outputs=_outs)
