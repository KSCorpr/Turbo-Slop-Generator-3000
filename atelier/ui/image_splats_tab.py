"""Single-image Gaussian object generation, distinct from captured-scene training."""
import gradio as gr

from .. import image_splats as models
from ..engine import image_splats as engine
from . import widgets
from .media_tabs import stream_task


def backend_controls(backend):
    models.artifacts(backend)
    counts = [n for n in engine.COUNTS if n <= (262144 if backend == "triposplat" else 1048576)]
    status = "Installed." if models.ready(backend) else "Install the selected model once, here or in Models."
    info = ("TripoSplat: MIT code and weights; PLY + SPLAT export. Source-image background removal is automatic."
            if backend == "triposplat" else
            "Idol: Apache-2.0 core. First accept [DINOv3](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) "
            "and [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) terms, then set HF_TOKEN or use hf auth login. "
            "Their own licenses apply. Exports PLY and the predicted camera; typical reported GPU memory is 7.2 GB, not a guarantee on every GPU.")
    return (gr.update(choices=[(f"{n:,}", n) for n in counts], value=131072),
            gr.update(value=20 if backend == "triposplat" else 50), gr.update(value="synthetic", visible=backend == "idol"),
            status + " " + info)


def build_image_splats_tab():
    with gr.Tab("Image → splats", id="image-splats"):
        gr.Markdown("Generate a 3D Gaussian object from one image. Hidden sides are invented; "
                    "this produces splats, not an editable polygon mesh. For measured scenes from "
                    "overlapping photos, use Capture → splats. NVIDIA CUDA required.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Object image", type="filepath", image_mode="RGBA", format="png",
                                 buttons=widgets.IMAGE_VIEW_ONLY, elem_id="image-splats-source")
                backend = gr.Dropdown([(name, key) for key, name in models.BACKENDS.items()],
                                      value="triposplat", label="3D model", elem_id="image-splats-model")
                install = gr.Button("Install / repair selected splat model", size="sm")
                info = gr.Markdown(backend_controls("triposplat")[-1], elem_id="image-splats-info")
                mode = gr.Radio([("Render / generated image", "synthetic"), ("Photograph", "real")],
                                value="synthetic", label="Idol crop mode", visible=False)
                count = gr.Dropdown([(f"{n:,}", n) for n in engine.COUNTS[:4]], value=131072,
                                    label="Gaussians", elem_id="image-splats-count")
                with gr.Row():
                    steps = gr.Slider(1, 100, value=20, step=1, label="Sampling steps")
                    seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                with gr.Row():
                    cfg = gr.Slider(1, 10, value=3, step=0.1, label="Image guidance")
                    shift = gr.Slider(0.1, 10, value=3, step=0.1, label="Time shift")
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
        log = gr.Textbox(label="Splat generation progress", lines=7, autoscroll=True)
        backend.change(backend_controls, inputs=backend, outputs=[count, steps, mode, info], queue=False)

        def install_image_splats(selected):
            for state, _, text in stream_task(lambda report: engine.install(selected, report)):
                yield state, text, backend_controls(selected)[-1]
        install.click(install_image_splats, inputs=backend, outputs=[status, log, info], **widgets.GPU_QUEUE)

        def generate_image_splats(source, selected, count, steps, cfg, shift, mode, seed):
            from pathlib import Path
            for state, path, text in stream_task(lambda report: engine.generate(
                    source, selected, count, steps, cfg, shift, mode, seed, report)):
                folder = Path(path).parent if path else None
                downloads = ([str(folder / name) for name in ("object.ply", "object.splat", "camera.json", "request.json")
                              if (folder / name).is_file()] if folder else None)
                yield state, path, str(folder / "input.png") if folder else None, downloads, text, path
        start.click(generate_image_splats, inputs=[image, backend, count, steps, cfg, shift, mode, seed],
                    outputs=[status, viewer, prepared, files, log, last_ply], **widgets.GPU_QUEUE)
        unload_preview.click(lambda: (None, "3D preview unloaded. Exported files are retained."),
                             outputs=[viewer, status], queue=False)
        show_preview.click(lambda path: (path, "3D preview restored." if path else "Generate an object first."),
                           inputs=last_ply, outputs=[viewer, status], queue=False)
        widgets.stop_into_status(stop, engine.JOB.cancel, status, [])
