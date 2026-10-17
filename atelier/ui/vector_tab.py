"""Local image/text to SVG and prepared LayerTracer sequence reconstruction."""
from pathlib import Path

import gradio as gr

from .. import addons
from ..engine import vectorize
from . import widgets
from .media_tabs import stream_task

CHOICES = [("VTracer · image → SVG · CPU", "trace"),
           ("HiVG · image → SVG · AI", "hivg-image"),
           ("HiVG · text → SVG · AI", "hivg-text"),
           ("LayerTracer · 4-panel drawing sequence → layers", "layer4"),
           ("LayerTracer · 9-panel drawing sequence → layers", "layer9")]


def mode_help(mode):
    descriptions = {
        "trace": "CPU tracing follows the source colors and contours. Best for logos and flat illustrations.",
        "hivg-image": "HiVG reconstructs an SVG from the image. Shapes, lettering and small details may change.",
        "hivg-text": "Describe a simple illustration or icon for HiVG to generate as vector shapes.",
        "layer4": "Import a prepared 2 × 2 sequence of the same drawing. Order: top left → top right → bottom right → bottom left. Equal cells, no gutters.",
        "layer9": "Import a prepared 3 × 3 sequence of the same drawing, read left to right, row by row. First cell: outlines; second: base colors. Equal cells, no gutters.",
    }
    ai = mode.startswith("hivg")
    return (descriptions[mode], gr.update(visible=mode != "hivg-text"),
            gr.update(visible=mode == "hivg-text"), gr.update(visible=ai),
            gr.update(visible=ai), gr.update(visible=mode.startswith("layer")))


def build_svg_tab():
    with gr.Tab("SVG & layers", id="svg"):
        gr.Markdown("### Vector graphics\nExport real SVG shapes, a PNG preview and a ZIP. "
            "LayerTracer here reconstructs layers from an existing drawing sequence; "
            "its FLUX image-to-sequence stage is not included. A single image is not a drawing sequence.")
        with gr.Row():
            install_cpu = gr.Button("Install / repair CPU tracing", size="sm")
            install_ai = gr.Button("Install / repair HiVG (~8.14 GB weights)", size="sm")
        installed = gr.Markdown(f"CPU tracing: {'installed' if addons.ready('svg') else 'not installed'}. "
                                f"HiVG: {'installed' if addons.ready('hivg') else 'not installed'}.")
        with gr.Row():
            with gr.Column():
                mode = gr.Dropdown(CHOICES, value="trace", label="SVG engine")
                help_text = gr.Markdown(mode_help("trace")[0])
                source = gr.Image(type="filepath", label="Source image or prepared drawing grid",
                                  format="png", height=260, buttons=widgets.IMAGE_BUTTONS)
                prompt = gr.Textbox(label="SVG description", lines=3, visible=False)
                tokens = gr.Slider(256, 4096, value=2048, step=256,
                    label="HiVG token budget (higher uses more memory)", visible=False)
                seed = gr.Number(value=0, precision=0, label="HiVG seed", visible=False)
                threshold = gr.Slider(1, 50, value=5, step=1,
                    label="LayerTracer difference threshold", visible=False)
                with gr.Row():
                    start = gr.Button("Create SVG", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                preview = gr.Image(label="Vector render preview", interactive=False, height=400,
                                   buttons=widgets.IMAGE_VIEW_ONLY)
                svg = gr.File(label="SVG document", interactive=False)
                bundle = gr.File(label="ZIP · vectors, layers and preview", interactive=False)
        status = gr.Markdown()
        log = gr.Textbox(lines=5, max_lines=8, interactive=False, label="SVG log")
        mode.change(mode_help, mode, [help_text, source, prompt, tokens, seed, threshold], queue=False)

        def install_svg(name):
            for message, _, lines in stream_task(lambda logger: vectorize.install(name, logger)):
                yield message, lines, (f"CPU tracing: {'installed' if addons.ready('svg') else 'not installed'}. "
                                      f"HiVG: {'installed' if addons.ready('hivg') else 'not installed'}.")

        def install_svg_cpu():
            yield from install_svg("svg")

        def install_svg_ai():
            yield from install_svg("hivg")

        def run_svg(mode, source, prompt, tokens, seed, threshold):
            for message, directory, lines in stream_task(lambda logger:
                    vectorize.run(mode, source, prompt, tokens, seed, threshold, logger)):
                folder = Path(directory) if directory else None
                yield (message, str(folder / "preview.png") if folder else None,
                       str(folder / "result.svg") if folder else None,
                       str(folder / "vectors.zip") if folder else None, lines)

        install_cpu.click(install_svg_cpu, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        install_ai.click(install_svg_ai, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        start.click(run_svg, [mode, source, prompt, tokens, seed, threshold],
                    [status, preview, svg, bundle, log], **widgets.GPU_QUEUE)
        stop.click(vectorize.JOB.cancel, outputs=status, queue=False)
