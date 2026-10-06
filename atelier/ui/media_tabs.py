"""Local restoration, LTX video and photo/video reconstruction workspaces."""
from __future__ import annotations

import queue
import sys
import threading
from collections import deque

import gradio as gr

from .. import addons, settings
from ..engine import seedvr2, ltx25, splat
from . import widgets


def stream_task(action):
    messages = queue.Queue()
    result = {}

    def worker():
        try:
            result["path"] = action(messages.put)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            messages.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    lines = deque(maxlen=120)
    yield "Working…", None, ""
    finished = False
    while not finished:
        try:
            line = messages.get(timeout=0.3)
        except queue.Empty:
            continue
        if line is None:
            finished = True
        else:
            lines.append(line)
        # Drain bursts before repainting Gradio, especially pip/COLMAP progress.
        for _ in range(300):
            try:
                line = messages.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
                break
            lines.append(line)
        if not finished:
            yield "Working…", None, "\n".join(lines)
    thread.join()
    if "error" in result:
        yield "Failed: " + result["error"], None, "\n".join(lines)
    else:
        path = str(result["path"]) if result.get("path") else None
        yield "Completed.", path, "\n".join(lines)


def install_action(name, job, log, quant=None):
    command = [sys.executable, "-u", str(settings.ROOT / "scripts" / "setup_media.py"), name]
    if quant:
        command += ["--quant", quant]
    with job.session():
        job.run(command, log)


def build_seedvr2_tab():
    with gr.Tab("SeedVR2 upscale", id="seedvr2"):
        gr.Markdown("### Restore images and videos\n"
                    "AI restoration with a memory-saving 11–12 GB preset. "
                    "Details can be reconstructed or changed. Compare fine structures at 100%. "
                    "The first run downloads the selected model and VAE.")
        with gr.Row():
            install = gr.Button("Install / repair SeedVR2", size="sm")
            gr.Markdown("Installed." if addons.ready("seedvr2") else "Installation required once.")
        with gr.Row():
            with gr.Column():
                source = gr.File(label="Image or video", type="filepath",
                                 file_types=sorted(seedvr2.IMAGES | seedvr2.VIDEOS))
                model = gr.Dropdown([("3B · balanced · Q8", "3b"),
                                     ("7B · quality · Q4 · slower", "7b")],
                                    value="3b", label="Restoration model")
                resolution = gr.Number(value=1080, minimum=256, maximum=8192,
                                       precision=0, label="Output short edge (px)")
                gr.Markdown("For ×4, enter four times the source's shortest edge. "
                            "Example: 1024×768 → short edge 3072 → 4096×3072. "
                            "Higher resolutions increase memory use and processing time.")
                seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    start = gr.Button("Restore", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                image = gr.Image(label="Restored image", interactive=False,
                                 buttons=widgets.IMAGE_BUTTONS)
                clip = gr.Video(label="Restored video", interactive=False)
                download = gr.File(label="Download result", interactive=False)
        status = gr.Markdown()
        log = gr.Textbox(label="Progress", lines=8, autoscroll=True)

        def install_seed():
            for state, _, text in stream_task(lambda report: install_action("seedvr2", seedvr2.JOB, report)):
                yield state, text

        install.click(install_seed, outputs=[status, log], **widgets.GPU_QUEUE)

        def run_seed(source, model, resolution, seed):
            for state, path, text in stream_task(
                    lambda report: seedvr2.restore(source, model, resolution, seed, report)):
                is_image = path and path.endswith(".png")
                yield state, path if is_image else None, path if path and not is_image else None, path, text

        start.click(run_seed, inputs=[source, model, resolution, seed],
                    outputs=[status, image, clip, download, log], **widgets.GPU_QUEUE)
        widgets.stop_into_status(stop, seedvr2.JOB.cancel, status, [])


def build_ltx25_tab():
    with gr.Tab("LTX 2.5", id="ltx25"):
        gr.Markdown("### LTX 2.5 · text or image to video\n"
                    "Local sd.cpp generation, video only. Start with 512×288 / 33 frames "
                    "on 11–12 GB GPUs. RAM offload is required; this is not a real-time mode.")
        quant = gr.Dropdown([("Q3_K_M · ~12.9 GB diffusion", "Q3_K_M"),
                             ("Q4_K_M · ~15.7 GB diffusion", "Q4_K_M")],
                            value="Q3_K_M", label="Model precision")
        with gr.Accordion("Install weights", open=not ltx25.ready()):
            gr.Markdown("Downloads the selected model, convolutional VAE and LTX-specific "
                        "Gemma encoder, then converts the encoder to Q4. Keep at least 60 GB "
                        "free for downloads and conversion. The original encoder is kept. "
                        "LTX weights use the LTX Community License; gated downloads may "
                        "require accepting the model terms in your Hugging Face account.")
            install = gr.Button("Download / prepare LTX 2.5", size="sm")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Scene and camera movement", lines=4)
                with gr.Row():
                    first = gr.Image(label="First frame (optional)", type="filepath",
                                     buttons=widgets.IMAGE_VIEW_ONLY)
                    last = gr.Image(label="Last frame (optional)", type="filepath",
                                    buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    size = gr.Dropdown([f"{w}x{h}" for w, h in ltx25.SIZES],
                                       value="512x288", label="Resolution")
                    frames = gr.Dropdown([(f"{n} frames · {n/24:.1f} s", n)
                                          for n in ltx25.FRAMES], value=33, label="Duration")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                with gr.Row():
                    start = gr.Button("Generate LTX video", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                result = gr.Video(label="LTX video", interactive=False)
                download = gr.File(label="Download MP4", interactive=False)
        status = gr.Markdown()
        log = gr.Textbox(label="Progress", lines=8, autoscroll=True)

        def install_ltx(quant):
            for state, _, text in stream_task(lambda report: install_action("ltx25", ltx25.JOB, report, quant)):
                yield state, text

        install.click(install_ltx, inputs=[quant], outputs=[status, log], **widgets.GPU_QUEUE)

        def run_ltx(prompt, quant, size, frames, seed, first, last):
            for state, path, text in stream_task(lambda report: ltx25.generate(
                    prompt, quant, size, frames, seed, first, last, report)):
                yield state, path, path, text

        start.click(run_ltx, inputs=[prompt, quant, size, frames, seed, first, last],
                    outputs=[status, result, download, log], **widgets.GPU_QUEUE)
        widgets.stop_into_status(stop, ltx25.JOB.cancel, status, [])


def build_splat_tab():
    with gr.Tab("Capture → splats", id="splat-capture"):
        gr.Markdown("### Reconstruct an object or a scene\n"
                    "Import overlapping photos **or** a video. Keep the subject still and "
                    "move the camera around it; use 30–150 sharp views with stable lighting. "
                    "The result is a Gaussian splat, not an editable polygon mesh.")
        install = gr.Button("Install / repair COLMAP + Brush", size="sm")
        with gr.Row():
            with gr.Column():
                photos = gr.File(label="Photos (8–300)", file_count="multiple", type="filepath",
                                 file_types=sorted(seedvr2.IMAGES))
                clip = gr.File(label="Or one video", type="filepath", file_types=sorted(seedvr2.VIDEOS))
                with gr.Row():
                    mode = gr.Dropdown([("Object · up to 500k splats", "object"),
                                        ("Scene · up to 1M splats", "scene")],
                                       value="object", label="Capture type")
                    quality = gr.Dropdown([("Preview · 5k steps", "preview"),
                                           ("Standard · 15k steps", "standard"),
                                           ("Fine · 30k steps", "fine")],
                                          value="standard", label="Quality")
                fps = gr.Dropdown([0.5, 1, 2, 4], value=1,
                                  label="Video samples / second (first 300 frames maximum)")
                with gr.Accordion("GPU selection and capture notes", open=False):
                    device = gr.Textbox(label="Brush adapter index (empty = automatic)", value="")
                    gr.Markdown("Brush uses WebGPU adapter numbering, which may differ from "
                                "CUDA. Its selected adapter appears in the log. Camera calibration "
                                "runs on CPU. Object mode limits memory; it does not remove the "
                                "background automatically. Avoid a rotating turntable with a "
                                "stationary background, reflections and moving people.")
                with gr.Row():
                    start = gr.Button("Reconstruct", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                viewer = gr.Model3D(label="Gaussian splat preview", interactive=False,
                                    height=480, clear_color=(0.08, 0.08, 0.09, 1))
                result = gr.File(label="Download full PLY", interactive=False)
                saved = gr.Dropdown(choices=[], label="Previous captures")
                refresh = gr.Button("Refresh saved captures", size="sm")
        status = gr.Markdown()
        log = gr.Textbox(label="Calibration and training", lines=8, autoscroll=True)

        def install_splat():
            for state, _, text in stream_task(lambda report: install_action("splat", splat.JOB, report)):
                yield state, text

        install.click(install_splat, outputs=[status, log], **widgets.GPU_QUEUE)

        def run_splat(photos, clip, mode, quality, fps, device):
            for state, path, text in stream_task(lambda report: splat.reconstruct(
                    photos, clip, mode, quality, fps, device, report)):
                yield state, path, path, text

        start.click(run_splat, inputs=[photos, clip, mode, quality, fps, device],
                    outputs=[status, viewer, result, log], **widgets.GPU_QUEUE)
        widgets.stop_into_status(stop, splat.JOB.cancel, status, [])
        refresh.click(lambda: gr.update(choices=splat.saved(), value=None), outputs=[saved])
        saved.change(lambda name: (splat.resolve_saved(name), splat.resolve_saved(name))
                     if name else (None, None), inputs=[saved], outputs=[viewer, result])
