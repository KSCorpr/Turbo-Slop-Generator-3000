"""VOSR 2.0 still-image restoration controls."""
import gradio as gr
from PIL import Image

from .. import model_catalog, vosr as artifacts
from ..engine import vosr
from . import widgets
from .media_tabs import stream_task, seed_source_preview


def size_info(source, scale):
    if not source:
        return "Import an image to see the output dimensions."
    try:
        with Image.open(source) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        target = vosr.target_size((width, height), scale)
        return f"{width} × {height} → **{target[0]} × {target[1]} px**. ×1 restores at the original size."
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return str(exc)


def build_vosr_tab():
    with gr.Tab("VOSR 2.0 upscale", id="vosr"):
        gr.Markdown("### VOSR 2.0 · image restoration\n"
                    "One-step AI restoration with overlapping tiles. Fine details and text may change; "
                    "compare the result at 100%. Images only. Output is PNG; transparency is retained.")
        with gr.Row():
            install = gr.Button("Install / repair VOSR 2.0", size="sm", elem_id="vosr-install")
            installed = gr.Markdown("Installed." if artifacts.ready() else
                                    "Install once: isolated runtime + 6.97 GB of models. Also available in Models.")
        with gr.Row():
            with gr.Column():
                source = gr.File(label="Source image", file_types=sorted(vosr.IMAGES), type="filepath",
                                 elem_id="vosr-source")
                preview = gr.Image(label="Source preview", interactive=False, visible=False, height=260,
                                   buttons=widgets.IMAGE_VIEW_ONLY, elem_id="vosr-preview")
                source_info = gr.Markdown()
                scale = gr.Radio([("×1 · restore", 1), ("×2", 2), ("×4", 4)], value=4,
                                 label="Output factor", elem_id="vosr-scale")
                dimensions = gr.Markdown("Import an image to see the output dimensions.", elem_id="vosr-size")
                seed = gr.Number(value=42, minimum=0, maximum=4294967295, precision=0,
                                 label="Seed", elem_id="vosr-seed")
                with gr.Accordion("Memory and color", open=False):
                    gr.Markdown("For 11–12 GB RTX cards, start with FP16, 512 px tiles and stage offload. "
                                "If VRAM runs out, reduce VAE tiles to 256. Larger tiles need more memory. "
                                "Tile sizes refer to the upscaled image.")
                    dtype = gr.Dropdown([("FP16 · recommended", "fp16"), ("BF16 · Ampere or newer", "bf16")],
                                        value="fp16", label="Precision")
                    color = gr.Dropdown([("Wavelet · preserve source colors", "wavelet"),
                                         ("AdaIN · match color statistics", "adain"), ("None", "none")],
                                        value="wavelet", label="Color alignment")
                    tile = gr.Dropdown([128, 256, 512], value=512, label="DiT tile size (px)")
                    overlap = gr.Slider(0, 112, step=16, value=64, label="DiT tile overlap (px)")
                    vae_tile = gr.Dropdown([128, 256, 512, 1024], value=512, label="VAE tile size (px)")
                    vae_overlap = gr.Slider(0, 256, step=16, value=64, label="VAE tile overlap (px)")
                    offload = gr.Checkbox(value=True, label="Offload models to RAM between stages",
                                          info="Only the active VAE, DINOv2 or DiT stays on the GPU.")
                with gr.Row():
                    start = gr.Button("Restore with VOSR", variant="primary", elem_id="vosr-run")
                    stop = gr.Button("Stop")
            with gr.Column():
                result = gr.Image(label="Restored image", interactive=False, format="png", height=480,
                                  buttons=widgets.IMAGE_BUTTONS, elem_id="vosr-result")
                download = gr.File(label="Download PNG", interactive=False)
        status = gr.Markdown(elem_id="vosr-status")
        log = gr.Textbox(label="Progress", lines=8, autoscroll=True)
        source.change(seed_source_preview, inputs=source, outputs=[preview, source_info], queue=False)
        source.change(size_info, inputs=[source, scale], outputs=dimensions, queue=False)
        scale.change(size_info, inputs=[source, scale], outputs=dimensions, queue=False)

        def install_vosr():
            for state, _, text in stream_task(lambda report: model_catalog.install("addon:vosr", log=report)):
                yield state, text, "Installed." if artifacts.ready() else "Installation / download needed."

        def restore(source, scale, seed, dtype, color, tile, overlap, vae_tile, vae_overlap, offload):
            for state, path, text in stream_task(lambda report: vosr.restore(
                    source, scale, seed, dtype=dtype, color=color, tile=tile, overlap=overlap,
                    vae_tile=vae_tile, vae_overlap=vae_overlap, offload=offload, log=report)):
                yield state, path, path, text

        install.click(install_vosr, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        start.click(restore, inputs=[source, scale, seed, dtype, color, tile, overlap, vae_tile,
                                    vae_overlap, offload], outputs=[status, result, download, log],
                    **widgets.GPU_QUEUE)
        def cancel():
            return model_catalog.JOB.cancel() if model_catalog.JOB._busy else vosr.JOB.cancel()
        widgets.stop_into_status(stop, cancel, status, [])
