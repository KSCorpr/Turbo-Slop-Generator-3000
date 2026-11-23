"""TensorForger/Steppe-Wan2.1 generative latent upscaler controls (images & videos)."""
from __future__ import annotations

from pathlib import Path
import gradio as gr
from PIL import Image

from .. import model_catalog, steppe as artifacts
from ..engine import steppe
from . import widgets
from .media_tabs import seed_source_preview, stream_task


def _size_info(source, scale, video_spatial, video_temporal):
    if not source:
        return "Import an image or video to see the output dimensions."
    path = Path(source)
    if path.suffix.lower() in steppe.VIDEOS:
        modes = []
        if video_spatial:
            modes.append("2× resolution")
        if video_temporal:
            modes.append("2× FPS interpolation")
        mode_str = " + ".join(modes) if modes else "Select 2× resolution and/or 2× FPS"
        return f"Video selected — **{mode_str}** in latent space (Wan 2.1 VAE)."
    try:
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        target = steppe.target_size((width, height), int(scale))
        passes = "2 chained 2× passes" if int(scale) == 4 else "1 latent pass"
        return f"{width} × {height} → **{target[0]} × {target[1]} px** ({passes})."
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return str(exc)


def _install_status(variant: str) -> str:
    if artifacts.ready(variant):
        return f"✅ Installed (`{variant}` + Wan 2.1 VAE ready)."
    return (
        f"Install once: isolated CUDA 12.6 runtime + Wan 2.1 VAE (508 MB) + "
        f"`{variant}` checkpoint. Also available in Models."
    )


def build_steppe_tab(tab_id: str = "steppe") -> None:
    """Build the Steppe-Wan2.1 latent upscaler tab under Tools & 3D (0 accordions)."""
    default_var = artifacts.DEFAULT_VARIANT
    with gr.Tab("Steppe-Wan2.1 upscale", id=tab_id):
        gr.Markdown(
            "### Steppe-Wan2.1 · Generative Latent Upscaler (Images & Videos)\n"
            "Runs directly in **Wan 2.1 VAE latent space** (MIT, by TensorForger) with no prompt and "
            "no second diffusion pass: **×2 / ×4 still images** (with alpha preserved) and **2× resolution "
            "and/or 2× FPS video interpolation**. Tiled VAE and stage offloading are enabled by default for "
            "11–12 GB GPUs (RTX 2080 Ti / RTX 3060)."
        )

        with gr.Row():
            variant = gr.Dropdown(
                [(lbl, key) for key, (_, lbl) in artifacts.MODELS.items()],
                value=default_var,
                label="Steppe model",
                elem_id="steppe-variant",
                scale=3,
            )
            install_btn = gr.Button(
                "Install / repair Steppe-Wan2.1",
                size="sm",
                elem_id="steppe-install",
                scale=1,
            )
            installed_md = gr.Markdown(_install_status(default_var))

        with gr.Row():
            with gr.Column(scale=3):
                source = gr.File(
                    label="Source image or video",
                    file_types=sorted(steppe.MEDIA),
                    type="filepath",
                    elem_id="steppe-source",
                )
                preview = gr.Image(
                    label="Source preview",
                    interactive=False,
                    visible=False,
                    height=220,
                    buttons=widgets.IMAGE_VIEW_ONLY,
                    elem_id="steppe-preview",
                )
                source_info = gr.Markdown()

                with gr.Group():
                    with gr.Row():
                        scale = gr.Radio(
                            [("×2 (1 pass)", 2), ("×4 (chained 2× → 2×)", 4)],
                            value=2,
                            label="Image scale",
                            elem_id="steppe-scale",
                            scale=2,
                        )
                        video_spatial = gr.Checkbox(
                            value=True,
                            label="Video: 2× resolution",
                            scale=1,
                        )
                        video_temporal = gr.Checkbox(
                            value=False,
                            label="Video: 2× FPS interpolation",
                            scale=1,
                        )
                    dimensions = gr.Markdown(
                        "Import an image or video to see the output dimensions.",
                        elem_id="steppe-size",
                    )

                with gr.Row():
                    hff = gr.Slider(
                        -1.0,
                        1.0,
                        value=0.9,
                        step=0.05,
                        label="Sharpness / HFF (-1 = auto, 0.9 = crisp default)",
                        scale=2,
                    )
                    guidance = gr.Slider(
                        1.0,
                        8.0,
                        value=1.0,
                        step=0.5,
                        label="Guidance (1.0-2.0 for 1-step; 4-8 for base)",
                        scale=2,
                    )
                with gr.Row():
                    steps = gr.Slider(
                        1,
                        50,
                        value=20,
                        step=1,
                        label="Steps (200M base model only)",
                        interactive=False,
                        scale=2,
                    )
                    dtype = gr.Dropdown(
                        [
                            ("Auto (BF16 on RTX 30/40/50 · FP16 on 2080 Ti / 1080)", "auto"),
                            ("FP16 (Turing / Pascal safe)", "fp16"),
                            ("BF16 (Ampere or newer)", "bf16"),
                            ("FP32 (highest precision)", "fp32"),
                        ],
                        value="auto",
                        label="Precision",
                        scale=2,
                    )
                with gr.Row():
                    seed = gr.Number(
                        value=42,
                        minimum=0,
                        maximum=4294967295,
                        precision=0,
                        label="Seed",
                        scale=1,
                    )
                    temporal_window = gr.Slider(
                        0,
                        32,
                        value=8,
                        step=2,
                        label="Video temporal window (latent frames · 0 = full clip)",
                        scale=2,
                    )
                    offload = gr.Checkbox(
                        value=True,
                        label="Stage offload (RAM)",
                        scale=1,
                    )

                with gr.Row():
                    start_btn = gr.Button(
                        "Upscale with Steppe-Wan2.1",
                        variant="primary",
                        elem_id="steppe-run",
                        scale=3,
                    )
                    stop_btn = gr.Button("Stop", scale=1)

            with gr.Column(scale=4):
                result_img = gr.Image(
                    label="Upscaled image (PNG)",
                    interactive=False,
                    format="png",
                    height=380,
                    buttons=widgets.IMAGE_BUTTONS,
                    elem_id="steppe-result-img",
                )
                result_vid = gr.Video(
                    label="Upscaled video (MP4)",
                    visible=False,
                    elem_id="steppe-result-vid",
                )
                download = gr.File(label="Download output", interactive=False)

        status = gr.Markdown(elem_id="steppe-status")
        log = gr.Textbox(label="Progress", lines=9, autoscroll=True, elem_classes="log-box")

        def _on_variant(var):
            return (
                _install_status(var),
                gr.update(interactive=var == "200m-base"),
            )

        variant.change(_on_variant, inputs=[variant], outputs=[installed_md, steps], queue=False)
        source.change(seed_source_preview, inputs=source, outputs=[preview, source_info], queue=False)
        for comp in (source, scale, video_spatial, video_temporal):
            comp.change(
                _size_info,
                inputs=[source, scale, video_spatial, video_temporal],
                outputs=dimensions,
                queue=False,
            )

        def install_steppe(var):
            for state, _, text in stream_task(
                lambda report: model_catalog.install("addon:steppe", var, log=report)
            ):
                yield state, text, _install_status(var)

        def run_steppe(src, var, sc, v_sp, v_tp, hff_v, guid_v, st_v, dt_v, s_v, tw_v, off_v):
            for state, path, text in stream_task(
                lambda report: steppe.upscale(
                    src,
                    variant=var,
                    scale=int(sc),
                    video_spatial=bool(v_sp),
                    video_temporal=bool(v_tp),
                    hff=float(hff_v),
                    guidance=float(guid_v),
                    steps=int(st_v),
                    dtype=dt_v,
                    seed=int(s_v),
                    temporal_window=int(tw_v),
                    offload=bool(off_v),
                    log=report,
                )
            ):
                if path and Path(path).suffix.lower() in steppe.VIDEOS:
                    yield state, gr.update(visible=False), gr.update(value=path, visible=True), path, text
                elif path:
                    yield state, gr.update(value=path, visible=True), gr.update(visible=False), path, text
                else:
                    yield state, gr.update(), gr.update(), None, text

        install_btn.click(
            install_steppe,
            inputs=[variant],
            outputs=[status, log, installed_md],
            **widgets.GPU_QUEUE,
        )
        start_btn.click(
            run_steppe,
            inputs=[
                source,
                variant,
                scale,
                video_spatial,
                video_temporal,
                hff,
                guidance,
                steps,
                dtype,
                seed,
                temporal_window,
                offload,
            ],
            outputs=[status, result_img, result_vid, download, log],
            **widgets.GPU_QUEUE,
        )

        def cancel():
            return model_catalog.JOB.cancel() if model_catalog.JOB._busy else steppe.JOB.cancel()

        widgets.stop_into_status(stop_btn, cancel, status, [])
