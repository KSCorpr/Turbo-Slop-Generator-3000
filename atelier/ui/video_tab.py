"""MiniMax H3 video generation: Turbo frames or Ref2VA images."""
from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import gradio as gr

from .. import settings, video
from ..engine import sdcpp
from ..engine import video as video_engine
from . import widgets


def build_video_tab() -> None:
    with gr.Tab("MiniMax H3", id="minimax-h3"):
        gr.Markdown("### MiniMax H3 · video only\n"
                    "Turbo uses four steps and optional first/last frames. "
                    "Reference mode uses 1–3 images and 50 steps. Choose and "
                    "download each mode's diffusion weight in **Model Catalog**. "
                    "Both modes reuse the same encoder and video VAE; "
                    "no audio weights or soundtrack are used.")
        mode = gr.Dropdown(
            choices=[("Turbo · text or first/last frames (4 steps)", "turbo"),
                     ("References · 1–3 images (50 steps)", "refs")],
            value="turbo", label="Video mode")
        diffusion, encoder = video.selected()
        ready = video.ready(diffusion, encoder)
        availability = gr.Markdown(
            "**Weights ready.**" if ready else
            "**Weights missing.** Open Model Catalog → MiniMax H3, "
            "then download the selected weights.")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Video prompt", lines=3,
                                    placeholder="Describe the scene and camera movement…")
                prompt_hint = gr.Markdown(
                    "Describe the motion. Turbo can also use a first and last frame.")
                with gr.Row():
                    generate = gr.Button("🎬 Generate video", variant="primary")
                    stop = gr.Button("⏹️ Stop", variant="stop")
                status = gr.Markdown("")
                with gr.Row(visible=True) as frame_inputs:
                    first = gr.Image(label="First frame (optional)", type="filepath",
                                     buttons=widgets.IMAGE_VIEW_ONLY)
                    last = gr.Image(label="Last frame (optional; requires first)",
                                    type="filepath", buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row(visible=False) as reference_inputs:
                    ref1 = gr.Image(label="Picture 1 · required", type="filepath",
                                    buttons=widgets.IMAGE_VIEW_ONLY)
                    ref2 = gr.Image(label="Picture 2 · optional", type="filepath",
                                    buttons=widgets.IMAGE_VIEW_ONLY)
                    ref3 = gr.Image(label="Picture 3 · optional", type="filepath",
                                    buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[("640×352 · lower memory", "640x352"),
                                 ("864×480 · recommended", "864x480"),
                                 ("960×544 · more detail", "960x544")],
                        value="864x480", label="Resolution")
                    frames = gr.Dropdown(
                        choices=[("22 frames · 0.9 s", 22),
                                 ("39 frames · 1.6 s", 39),
                                 ("56 frames · 2.3 s", 56)],
                        value=22, label="Duration · 24 fps")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                log = gr.Textbox(label="Generation log", lines=7,
                                 autoscroll=True, elem_classes="log-box")
            with gr.Column(scale=3):
                result = gr.Video(label="Generated video (MP4)", format="mp4",
                                  interactive=False)
                gr.Markdown("**Convert a completed AVI** · uses no model or GPU. "
                            "Select the file left by an earlier H3 run.")
                saved = video_engine.saved_avis()
                existing_avi = gr.Dropdown(
                    choices=saved, value=saved[0] if saved else None,
                    label="Saved H3 AVI in outputs/")
                with gr.Row():
                    refresh_avi = gr.Button("↻ Refresh AVI list", size="sm")
                    convert_avi = gr.Button("Convert AVI to MP4", size="sm")

        def refresh(chosen_mode):
            d, e = video.selected(mode=chosen_mode)
            return ("**Weights ready.**" if video.ready(d, e, mode=chosen_mode) else
                    "**Weights missing.** Download them in Model Catalog → "
                    "MiniMax H3.")

        def switch_mode(chosen_mode):
            is_refs = chosen_mode == "refs"
            hint = ("Images 1–3 guide the video's appearance. Mention "
                    "`<Picture 1>`, `<Picture 2>` and `<Picture 3>` in the "
                    "prompt to describe their roles; these are references, "
                    "not timed keyframes." if is_refs else
                    "Describe the motion. Turbo can also use a first and last frame.")
            return (gr.update(visible=not is_refs), gr.update(visible=is_refs),
                    hint, refresh(chosen_mode))

        mode.change(switch_mode, inputs=[mode],
                    outputs=[frame_inputs, reference_inputs, prompt_hint, availability])

        gr.Button("↻ Refresh model status", size="sm").click(
            refresh, inputs=[mode], outputs=[availability])

        def avi_choices():
            names = video_engine.saved_avis()
            return gr.update(choices=names, value=names[0] if names else None)

        refresh_avi.click(avi_choices, outputs=[existing_avi])

        def recover(name):
            yield "Converting the saved AVI to MP4…", gr.update(), gr.update()
            try:
                mp4 = video_engine.convert_saved_avi(name)
                yield f"Video recovered: {mp4.name}", str(mp4), avi_choices()
            except sdcpp.EngineError as exc:
                yield f"Conversion failed: {exc}", gr.update(), gr.update()

        convert_avi.click(recover, inputs=[existing_avi],
                          outputs=[status, result, existing_avi])

        def run(chosen_mode, prompt_text, first_file, last_file,
                ref1_file, ref2_file, ref3_file, size, frame_count, chosen_seed):
            q: queue.Queue[str | None] = queue.Queue()
            state: dict = {}
            d, e = video.selected(mode=chosen_mode)

            def worker():
                try:
                    weights = video.weights(d, e, mode=chosen_mode)
                    width, height = map(int, size.split("x"))
                    refs = ()
                    if chosen_mode == "refs":
                        if not ref1_file or (ref3_file and not ref2_file):
                            raise sdcpp.EngineError(
                                "Add Picture 1 before Picture 2 or 3, "
                                "and Picture 2 before Picture 3.")
                        refs = tuple(Path(p) for p in
                                     (ref1_file, ref2_file, ref3_file) if p)
                    path = video_engine.generate(
                        *weights, prompt_text, width=width, height=height,
                        frames=int(frame_count),
                        seed=int(chosen_seed if chosen_seed is not None else -1),
                        first=Path(first_file) if chosen_mode == "turbo" and first_file else None,
                        last=Path(last_file) if chosen_mode == "turbo" and last_file else None,
                        mode=chosen_mode, refs=refs,
                        log=q.put)
                    state["path"] = str(path)
                except Exception as exc:  # noqa: BLE001
                    state["error"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            lines: list[str] = []
            current = f"Loading MiniMax H3 {'Ref2VA' if chosen_mode == 'refs' else 'Turbo'}…"
            yield current, None, ""
            steps = (video_engine.REF_STEPS if chosen_mode == "refs"
                     else video_engine.STEPS)
            step_re = re.compile(rf"\b(\d+)\s*/\s*{steps}\b")
            while True:
                try:
                    line = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line:
                    # sd-cli's animated progress bar floods stdout; keep the
                    # step visible without repainting hundreds of log lines.
                    match = step_re.search(line)
                    if match:
                        current = f"Step {match.group(1)}/{steps} · generating video…"
                    if "|" not in line or not match:
                        lines.append(line)
                        lines = lines[-100:]
                    if match or len(lines) < 15 or len(lines) % 10 == 0:
                        yield current, None, "\n".join(lines)
            if state.get("error"):
                yield f"❌ {state['error']}", None, "\n".join(lines)
            else:
                yield "✅ Video ready · MP4 without audio.", state["path"], \
                    "\n".join(lines)

        event = generate.click(run, inputs=[mode, prompt, first, last,
                                            ref1, ref2, ref3, resolution,
                                            frames, seed],
                               outputs=[status, result, log], **widgets.GPU_QUEUE)
        widgets.stop_into_status(stop, sdcpp.cancel_active, status, [event])
