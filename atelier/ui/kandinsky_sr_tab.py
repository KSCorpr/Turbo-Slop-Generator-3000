"""Kandinsky ImageSR 2.0 one-step diffusion upscaler UI tab."""
from __future__ import annotations

import gradio as gr
from PIL import Image

from .. import kandinsky_sr as artifacts, model_catalog
from ..engine import kandinsky_sr
from . import widgets
from .media_tabs import seed_source_preview, stream_task


def size_info(source, scale) -> str:
    if not source:
        return "Import an image to see the output dimensions."
    try:
        with Image.open(source) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        target = kandinsky_sr.target_size((width, height), int(scale))
        return f"{width} × {height} → **{target[0]} × {target[1]} px** (×{int(scale)})."
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return str(exc)


def creativity_info(scale, creativity) -> str:
    factor = 2 if int(scale or 4) == 2 else 4
    level = max(0, min(8, int(creativity if creativity is not None else artifacts.DEFAULT_CREATIVITY)))
    timestep = artifacts.CREATIVITY_TIMESTEPS[factor][level]
    hint = (
        "faithful / minimal invention (best for text & line art)" if level <= 2
        else "calibrated default balance" if level == 4
        else "more invented micro-detail" if level >= 6
        else "moderate detail synthesis"
    )
    return f"Creativity {level}/8 → calibrated `one_step_t = {timestep:g}` at ×{factor} ({hint})."


def build_kandinsky_sr_tab() -> None:
    with gr.Tab("Kandinsky SR 2.0 upscale", id="kandinsky-sr"):
        gr.Markdown(
            "### Kandinsky ImageSR 2.0 · one-step diffusion upscaler\n"
            "Prompt-free 5B one-step super-resolution (×2 / ×4) with 1024 px Hanning-blended tiles and "
            "calibrated creativity levels 0–8 (MIT + Apache-2.0). Deterministic (no seed). "
            "PNG output; transparency is retained."
        )
        with gr.Row():
            install = gr.Button(
                "Install / repair Kandinsky ImageSR 2.0",
                size="sm",
                elem_id="kandinsky-sr-install",
            )
            installed = gr.Markdown(
                "Installed." if artifacts.ready() else
                "Install once: isolated Diffusers 0.41+ runtime + 10.42 GB of weights. Also available in Models."
            )
        with gr.Row():
            with gr.Column():
                source = gr.File(
                    label="Source image",
                    file_types=sorted(kandinsky_sr.IMAGES),
                    type="filepath",
                    elem_id="kandinsky-sr-source",
                )
                preview = gr.Image(
                    label="Source preview",
                    interactive=False,
                    visible=False,
                    height=260,
                    buttons=widgets.IMAGE_VIEW_ONLY,
                    elem_id="kandinsky-sr-preview",
                )
                source_info = gr.Markdown()
                scale = gr.Radio(
                    [("×2", 2), ("×4 (default)", 4)],
                    value=4,
                    label="Upscale factor",
                    elem_id="kandinsky-sr-scale",
                )
                dimensions = gr.Markdown(
                    "Import an image to see the output dimensions.",
                    elem_id="kandinsky-sr-size",
                )
                creativity = gr.Slider(
                    0, 8, step=1, value=artifacts.DEFAULT_CREATIVITY,
                    label="Creativity (calibrated 0–8)",
                    info="0 = faithful (line art / text), 4 = calibrated default, 8 = maximum detail invention.",
                    elem_id="kandinsky-sr-creativity",
                )
                creativity_note = gr.Markdown(
                    creativity_info(4, artifacts.DEFAULT_CREATIVITY),
                    elem_id="kandinsky-sr-creativity-note",
                )
                with gr.Accordion("Memory, precision & tile overlap", open=False):
                    gr.Markdown(
                        "The 5B transformer is ~10.25 GB. **Auto** uses sequential layer offload (< 3 GB VRAM) "
                        "on 11–12 GB GPUs (RTX 2080 Ti / RTX 3060), model offload on 16 GB, and full GPU on 24 GB+."
                    )
                    memory_mode = gr.Dropdown(
                        [
                            ("Auto (recommended)", "auto"),
                            ("Sequential layer offload (lowest VRAM · <=12 GB GPUs)", "sequential"),
                            ("Model CPU offload (16 GB GPUs)", "model"),
                            ("Full GPU resident (24 GB+ GPUs)", "gpu"),
                        ],
                        value="auto",
                        label="Memory mode",
                    )
                    dtype = gr.Dropdown(
                        [
                            ("Auto (BF16 on Ampere+, FP16 on Turing)", "auto"),
                            ("BF16 (RTX 30xx / 40xx / 50xx)", "bf16"),
                            ("FP16 (RTX 20xx / Turing)", "fp16"),
                        ],
                        value="auto",
                        label="Precision",
                    )
                    overlap = gr.Slider(
                        0.0, 0.5, step=0.05, value=0.25,
                        label="Tile overlap fraction",
                        info="Default 0.25 matches the official benchmark; higher reduces seams at extra cost.",
                    )
                with gr.Row():
                    start = gr.Button(
                        "Upscale with Kandinsky SR 2.0",
                        variant="primary",
                        elem_id="kandinsky-sr-run",
                    )
                    stop = gr.Button("Stop")
            with gr.Column():
                result = gr.Image(
                    label="Upscaled image",
                    interactive=False,
                    format="png",
                    height=480,
                    buttons=widgets.IMAGE_BUTTONS,
                    elem_id="kandinsky-sr-result",
                )
                download = gr.File(label="Download PNG", interactive=False)
        status = gr.Markdown(elem_id="kandinsky-sr-status")
        log = gr.Textbox(label="Progress", lines=8, autoscroll=True)

        source.change(seed_source_preview, inputs=source, outputs=[preview, source_info], queue=False)
        source.change(size_info, inputs=[source, scale], outputs=dimensions, queue=False)
        scale.change(size_info, inputs=[source, scale], outputs=dimensions, queue=False)
        scale.change(creativity_info, inputs=[scale, creativity], outputs=creativity_note, queue=False)
        creativity.change(creativity_info, inputs=[scale, creativity], outputs=creativity_note, queue=False)

        def install_ksr():
            for state, _, text in stream_task(
                lambda report: model_catalog.install("addon:kandinsky-sr", log=report)
            ):
                yield state, text, (
                    "Installed." if artifacts.ready() else "Installation / download needed."
                )

        def run_upscale(src, sc, creat, ov, dt, mem):
            for state, path, text in stream_task(
                lambda report: kandinsky_sr.upscale(
                    src, int(sc), int(creat), overlap=float(ov), dtype=dt, memory_mode=mem, log=report
                )
            ):
                yield state, path, path, text

        install.click(install_ksr, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        start.click(
            run_upscale,
            inputs=[source, scale, creativity, overlap, dtype, memory_mode],
            outputs=[status, result, download, log],
            **widgets.GPU_QUEUE,
        )

        def cancel():
            return model_catalog.JOB.cancel() if model_catalog.JOB._busy else kandinsky_sr.JOB.cancel()

        widgets.stop_into_status(stop, cancel, status, [])
