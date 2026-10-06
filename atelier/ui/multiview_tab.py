"""Image to ten object views or a fixed-camera 360 environment."""
from __future__ import annotations

import queue
import threading
from collections import deque

import gradio as gr

from .. import registry, settings
from ..engine import multiview, tools
from . import widgets


def model_choices():
    models = registry.load_base_models(settings.load_prefs())
    choices, default = [], None
    for model in models:
        ready = registry.model_is_ready(model)
        label = f"{model.name} · {multiview.conditioning(model)}"
        choices.append((label if ready else label + " (not installed)", model.id))
        if default is None and ready and multiview.conditioning(model) == "reference":
            default = model.id
    return choices, default or (models[0].id if models else None)


def stream_views(action):
    events = queue.Queue()

    def update(result):
        events.put(("result", ([ (str(p), name) for p, name in result.views ],
                              str(result.panorama) if result.panorama else None,
                              str(result.archive) if result.archive else None)))

    def worker():
        try:
            result = action(lambda line: events.put(("log", line)), update)
            update(result)
            events.put(("done", "Completed."))
        except Exception as exc:
            events.put(("done", "Failed: " + str(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    lines = deque(maxlen=100)
    gallery, pano, archive = [], None, None
    yield "Working…", gallery, pano, pano, archive, ""
    done = False
    while not done:
        try:
            event = events.get(timeout=.3)
        except queue.Empty:
            continue
        batch = [event]
        for _ in range(200):
            try:
                batch.append(events.get_nowait())
            except queue.Empty:
                break
        status = "Working…"
        changed = False
        for kind, value in batch:
            if kind == "log":
                lines.append(value)
            elif kind == "result":
                gallery, pano, archive = value
                changed = True
            elif kind == "done":
                status, done = value, True
        if not done:
            status = f"Working… {len(gallery)}/10 views saved."
        yield (status, gallery if changed else gr.skip(),
               pano if changed else gr.skip(), pano if changed else gr.skip(),
               archive if changed else gr.skip(), "\n".join(lines))
    thread.join()


def build_multiview_tab():
    choices, default = model_choices()
    with gr.Tab("10 views / 360°", id="multiview"):
        gr.Markdown("### Ten angles from one image\n"
                    "Object orbit: eight sides plus high and low angles. "
                    "360 scene: eight directions plus sky/ceiling and ground from a fixed camera. "
                    "Hidden details are invented; geometry and identity can vary between views. "
                    "Flux / Qwen references are the most suitable choices.")
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source image", type="pil", height=290,
                                  buttons=widgets.IMAGE_BUTTONS, elem_id="views-source")
                model = gr.Dropdown(choices, value=default, label="Image engine")
                note = gr.Markdown(multiview.model_note(default))
                mode = gr.Radio([("Object — camera orbits the subject", "object"),
                                 ("360 scene — camera rotates in place", "scene")],
                                value="object", label="Camera mode")
                description = gr.Textbox(label="Subject / scene description (optional)", lines=2,
                                         placeholder="Describe what must stay consistent.")
                size = gr.Dropdown([512, 768, 1024], value=768, label="Pixels per square view")
                export = gr.Checkbox(label="Also create an equirectangular 360×180 panorama",
                                     value=False, visible=False)
                width = gr.Dropdown([2048, 4096, 8192], value=4096,
                                    label="Panorama width (2:1 output)", visible=False)
                advanced = gr.Checkbox(value=False, label="Advanced and conversion tools")
                with gr.Group(visible=False) as advanced_group:
                    strength = gr.Slider(.1, 1, value=.8, step=.05,
                                         label="Image-to-image change strength",
                                         info="Krea / Z-Image only. High values allow bigger angle changes but reduce fidelity.")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    describe = gr.Checkbox(value=False, label="Read the source with the local vision model",
                                           info="Always used for Ming. Adds a description; it does not turn a text-only engine into an image editor.")
                    install = gr.Button("Install Image → prompt (needed for Ming)", size="sm")
                with gr.Row():
                    run = gr.Button("Generate 10 views", variant="primary")
                    stop = gr.Button("Stop views", variant="stop")
                status = gr.Markdown()
            with gr.Column():
                gallery = gr.Gallery(label="Generated angles", columns=5, height=390,
                                     type="filepath", buttons=widgets.GALLERY_BUTTONS)
                pano = gr.Image(label="Equirectangular panorama (360 scene only)",
                                type="filepath", height=220,
                                buttons=widgets.IMAGE_BUTTONS)
                with gr.Row():
                    download_pano = gr.DownloadButton("Download 360 PNG")
                    archive = gr.DownloadButton("Download views + settings ZIP")
                gr.Markdown("The 360 export reprojects and blends perspective views. "
                            "It can retain joins or conflicting geometry. Object orbit images "
                            "cannot be converted into a surrounding panorama.")
                with gr.Group(visible=False) as conversion_group:
                    files = gr.File(label="10 numbered scene PNGs from the ZIP", file_count="multiple",
                                    type="filepath", file_types=[".png"], elem_id="views-import")
                    convert_width = gr.Dropdown([2048, 4096, 8192], value=4096,
                                                label="Conversion width")
                    convert = gr.Button("Convert views to 360 PNG", size="sm")
                logs = gr.Textbox(label="Views log", lines=6, autoscroll=True, elem_id="views-log")

        def change_mode(value):
            return gr.update(visible=value == "scene", value=False), gr.update(visible=value == "scene")

        mode.change(change_mode, inputs=mode, outputs=[export, width], queue=False)
        model.change(multiview.model_note, inputs=model, outputs=note, queue=False)
        advanced.change(lambda value: (gr.update(visible=value), gr.update(visible=value)),
                        inputs=advanced, outputs=[advanced_group, conversion_group], queue=False)

        def run_views(src, engine, camera, text, pixels, denoise, s, equirect, w, caption):
            yield from stream_views(lambda log, update: multiview.generate_views(
                src, engine, mode=camera, description=text or "", size=pixels,
                strength=denoise, seed=int(s) if s is not None else -1,
                make_panorama=equirect, panorama_width=w, auto_describe=caption,
                log=log, update=update))

        def convert_panorama(paths, w):
            yield from stream_views(lambda log, update: multiview.convert_views(paths, width=w, log=log))

        def install_view_caption():
            lines = deque(maxlen=100)
            for line in tools.install_describe_stream():
                lines.append(line)
                yield "\n".join(lines)

        outputs = [status, gallery, pano, download_pano, archive, logs]
        run.click(run_views, inputs=[source, model, mode, description, size, strength,
                                    seed, export, width, describe], outputs=outputs,
                  **widgets.GPU_QUEUE)
        convert.click(convert_panorama, inputs=[files, convert_width], outputs=outputs,
                      **widgets.GPU_QUEUE)
        install.click(install_view_caption, outputs=logs, **widgets.GPU_QUEUE)
        stop.click(multiview.JOB.cancel, outputs=status, queue=False)
