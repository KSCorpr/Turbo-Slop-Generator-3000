"""Local, paginated gallery with original downloads and prompt hand-off."""
from __future__ import annotations

import gradio as gr

from .. import history
from . import widgets


def build_history_tab(tabs, generation_tabs, prompt_boxes, pending_toolkit):
    with gr.Tab("Gallery", id="history") as tab:
        gr.Markdown("### Your creations\nImages stay in **outputs/**. Browse, download the original, or reuse a prompt.")
        with gr.Row():
            query = gr.Textbox(label="Search filenames", placeholder="Model name, date…", scale=5)
            refresh = gr.Button("Refresh", scale=1)
        names = gr.State([])
        page = gr.State(1)
        selected = gr.State("")
        with gr.Row():
            with gr.Column(scale=5):
                gallery = gr.Gallery(label="Saved images", columns=4, height=600,
                                     object_fit="contain", allow_preview=False,
                                     elem_id="history-grid",
                                     buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    previous = gr.Button("Previous", size="sm")
                    count = gr.Markdown("Open this tab to load your images.")
                    following = gr.Button("Next", size="sm")
            with gr.Column(scale=3):
                image = gr.Image(label="Selected original", interactive=False,
                                 height=330, buttons=widgets.IMAGE_BUTTONS)
                original = gr.File(label="Download original", interactive=False)
                prompt = gr.Textbox(label="Saved prompt", lines=4, buttons=widgets.TEXT_COPY,
                                   elem_id="history-prompt")
                model = gr.Dropdown(choices=list(prompt_boxes), value=next(iter(prompt_boxes)),
                                    label="Reuse prompt with")
                reuse = gr.Button("Use prompt in Images", variant="primary")
                upscale = gr.Button("Send original to upscale")
                with gr.Accordion("Saved generation details", open=False):
                    metadata = gr.Textbox(label="Parameters", lines=8, interactive=False,
                                          buttons=widgets.TEXT_COPY)

        def load(search, current):
            entries, total, current = history.page(search, current)
            thumbs, keys = [], []
            for entry in entries:
                thumb = history.thumbnail(entry)
                if thumb:
                    thumbs.append((thumb, entry.name))
                    keys.append(entry.name)
            history.prune_thumbnails()
            pages = max(1, (total + history.PAGE_SIZE - 1) // history.PAGE_SIZE)
            message = (f"{total} images · page {current} / {pages}" if total else
                       "No matching images." if search else "Your first image will appear here after generation.")
            return (thumbs, keys, current, message,
                    gr.update(interactive=current > 1),
                    gr.update(interactive=current < pages), "", None, None, "", "")

        targets = [gallery, names, page, count, previous, following,
                   selected, image, original, prompt, metadata]
        tab.select(lambda search: load(search, 1), inputs=[query], outputs=targets)
        refresh.click(load, inputs=[query, page], outputs=targets)
        query.submit(lambda search: load(search, 1), inputs=[query], outputs=targets)
        previous.click(lambda search, current: load(search, current - 1), inputs=[query, page], outputs=targets)
        following.click(lambda search, current: load(search, current + 1), inputs=[query, page], outputs=targets)

        def select(keys, evt: gr.SelectData):
            index = evt.index
            if not isinstance(index, int) or not 0 <= index < len(keys):
                raise gr.Error("Select an image first.")
            try:
                data = history.details(keys[index])
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            chosen = data.get("model_id")
            return (data["name"], data["path"], data["path"], data.get("prompt", ""),
                    data.get("text", "No generation metadata found."),
                    gr.update(value=chosen) if chosen in prompt_boxes else gr.update())

        gallery.select(select, inputs=[names], outputs=[selected, image, original, prompt, metadata, model])

        def send_prompt(text, target):
            if not (text or "").strip() or target not in prompt_boxes:
                raise gr.Error("Select an image with a saved prompt, then choose a model.")
            return [gr.Tabs(selected="create"), gr.Tabs(selected=target)] + [
                text if key == target else gr.update() for key in prompt_boxes]

        reuse.click(send_prompt, inputs=[prompt, model],
                    outputs=[tabs, generation_tabs, *prompt_boxes.values()], queue=False)

        def send_upscale(name):
            try:
                path = history.resolve(name)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return (str(path), "esrgan"), gr.Tabs(selected="tools")

        pending_toolkit.send(upscale.click, send_upscale,
                             inputs=[selected], outputs=[tabs])
