"""MiniMax H3 Turbo: a short, video-only generation workflow."""
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
    with gr.Tab("🎬 Video · H3 Turbo", id="h3-video"):
        gr.Markdown("### MiniMax H3 Turbo · video only\n"
                    "Four sampling steps, 24 fps. Download the three video "
                    "weights in **Model Catalog** first. No audio weights or "
                    "soundtrack are used.")
        diffusion, encoder = video.selected()
        ready = video.ready(diffusion, encoder)
        availability = gr.Markdown(
            "**Weights ready.**" if ready else
            "**Weights missing.** Open Model Catalog → MiniMax H3 Turbo, "
            "then download the selected weights.")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Video prompt", lines=3,
                                    placeholder="Describe the scene and camera movement…")
                with gr.Row():
                    generate = gr.Button("🎬 Generate video", variant="primary")
                    stop = gr.Button("⏹️ Stop", variant="stop")
                status = gr.Markdown("")
                with gr.Row():
                    first = gr.Image(label="First frame (optional)", type="filepath",
                                     buttons=widgets.IMAGE_VIEW_ONLY)
                    last = gr.Image(label="Last frame (optional; requires first)",
                                    type="filepath", buttons=widgets.IMAGE_VIEW_ONLY)
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

        def refresh():
            d, e = video.selected()
            return ("**Weights ready.**" if video.ready(d, e) else
                    "**Weights missing.** Download them in Model Catalog → "
                    "MiniMax H3 Turbo.")

        gr.Button("↻ Refresh model status", size="sm").click(
            refresh, outputs=[availability])

        def run(prompt_text, first_file, last_file, size, frame_count, chosen_seed):
            q: queue.Queue[str | None] = queue.Queue()
            state: dict = {}
            d, e = video.selected()

            def worker():
                try:
                    weights = video.weights(d, e)
                    width, height = map(int, size.split("x"))
                    path = video_engine.generate(
                        *weights, prompt_text, width=width, height=height,
                        frames=int(frame_count),
                        seed=int(chosen_seed if chosen_seed is not None else -1),
                        first=Path(first_file) if first_file else None,
                        last=Path(last_file) if last_file else None,
                        log=q.put)
                    state["path"] = str(path)
                except Exception as exc:  # noqa: BLE001
                    state["error"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            lines: list[str] = []
            current = "Loading MiniMax H3 Turbo…"
            yield current, None, ""
            step_re = re.compile(r"\b([1-4])\s*/\s*4\b")
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
                        current = f"Step {match.group(1)}/4 · generating video…"
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

        event = generate.click(run, inputs=[prompt, first, last, resolution,
                                            frames, seed],
                               outputs=[status, result, log])
        widgets.stop_into_status(stop, sdcpp.cancel_active, status, [event])
