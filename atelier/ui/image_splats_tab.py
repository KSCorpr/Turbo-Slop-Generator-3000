"""Single-image Gaussian object generation, distinct from captured-scene training."""
import gradio as gr

from .. import image_splats as models
from ..engine import image_splats as engine
from . import widgets
from .media_tabs import stream_task


def build_image_splats_tab():
    with gr.Tab("Image → splats", id="image-splats"):
        gr.Markdown("**TripoSplat** turns one object image into Gaussian splats. "
                    "Exports PLY and SPLAT. NVIDIA CUDA required.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Object image", type="filepath", image_mode="RGBA", format="png",
                                 buttons=widgets.IMAGE_VIEW_ONLY, elem_id="image-splats-source")
                with gr.Row():
                    count = gr.Dropdown([(f"{n:,}", n) for n in engine.COUNTS], value=131072,
                                        label="Gaussians", elem_id="image-splats-count")
                    seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                with gr.Accordion("Installation, sampling & log", open=not models.ready("triposplat")):
                    info = gr.Markdown("TripoSplat installed." if models.ready("triposplat")
                                       else "Install the 3.78 GB public weights once.")
                    install = gr.Button("Install / repair TripoSplat", size="sm")
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=20, step=1, label="Sampling steps")
                        cfg = gr.Slider(1, 10, value=3, step=0.1, label="Image guidance")
                        shift = gr.Slider(0.1, 10, value=3, step=0.1, label="Time shift")
                    log = gr.Textbox(label="Progress", lines=3, max_lines=8, autoscroll=True, interactive=False)
                with gr.Row():
                    start = gr.Button("Generate splats", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                viewer = gr.Model3D(label="Generated Gaussian object", interactive=False, height=450,
                                    clear_color=(0.08, 0.08, 0.09, 1), elem_id="image-splats-viewer")
                last_ply = gr.State(None)
                with gr.Row():
                    show_preview = gr.Button("Show 3D preview", size="sm")
                    unload_preview = gr.Button("Unload 3D preview", size="sm")
                prepared = gr.Image(label="Input seen by the model", interactive=False, height=200,
                                    buttons=widgets.IMAGE_VIEW_ONLY)
                files = gr.File(label="Download splat and metadata", file_count="multiple", interactive=False,
                                elem_id="image-splats-files")
        status = gr.Markdown(elem_id="image-splats-status")

        def install_image_splats():
            for state, _, text in stream_task(lambda report: engine.install("triposplat", report)):
                yield state, text, ("TripoSplat installed." if models.ready("triposplat")
                                    else "TripoSplat needs installation.")
        install.click(install_image_splats, outputs=[status, log, info], **widgets.GPU_QUEUE)

        def generate_image_splats(source, count, steps, cfg, shift, seed):
            from pathlib import Path
            for state, path, text in stream_task(lambda report: engine.generate(
                    source, "triposplat", count, steps, cfg, shift, "synthetic", seed, report)):
                folder = Path(path).parent if path else None
                downloads = ([str(folder / name) for name in ("object.ply", "object.splat", "request.json")
                              if (folder / name).is_file()] if folder else None)
                yield state, path, str(folder / "input.png") if folder else None, downloads, text, path
        start.click(generate_image_splats, inputs=[image, count, steps, cfg, shift, seed],
                    outputs=[status, viewer, prepared, files, log, last_ply], **widgets.GPU_QUEUE)
        unload_preview.click(lambda: (None, "3D preview unloaded. Exported files are retained."),
                             outputs=[viewer, status], queue=False)
        show_preview.click(lambda path: (path, "3D preview restored." if path else "Generate an object first."),
                           inputs=last_ply, outputs=[viewer, status], queue=False)
        widgets.stop_into_status(stop, engine.JOB.cancel, status, [])
