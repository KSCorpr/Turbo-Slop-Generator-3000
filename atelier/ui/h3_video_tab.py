"""MiniMax H3 Video & HyperFlow H3 Flash workspace (video + stereo audio).

Designed to run cleanly on 11-12 GB NVIDIA cards with 64 GB system RAM
(e.g. RTX 2080 Ti 11 GB or RTX 3060 12 GB + GTX 1080) using HyperFlow H3 Flash
GGUF Q4_K_M (4-8 steps), P-H3 4B or shared Qwen3-VL-32B text encoder, Sol-H3
distilled video VAE, auto-fit RAM offloading, and attention chunk budgeting.
"""
from __future__ import annotations

import queue
import tempfile
import threading
from collections import deque
from pathlib import Path

import gradio as gr
from PIL import Image

from .. import h3_video as catalog
from .. import settings
from ..engine import h3_video as engine
from ..engine import sdcpp
from . import widgets


def _status_text(diff_key: str, enc_key: str, vae_key: str) -> str:
    if catalog.ready(diff_key, enc_key, vae_key):
        return "✅ **Selected H3 Video weights are installed and ready.**"
    files = catalog.resolve_files(diff_key, enc_key, vae_key)
    missing = [k for k, v in files.items() if v is None]
    return (
        f"⚠️ **Missing components ({', '.join(missing)})** — click "
        "**Install / download selected H3 Video pack** to fetch them (shared H3 Image encoders are reused automatically)."
    )


def _stream_runner(fn):
    messages: "queue.Queue[str | None]" = queue.Queue()
    result: dict = {}

    def worker():
        try:
            result["value"] = fn(messages.put)
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
        finally:
            messages.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    lines: deque[str] = deque(maxlen=200)
    yield "Working…", None, ""
    done = False
    while not done:
        try:
            item = messages.get(timeout=0.3)
        except queue.Empty:
            continue
        if item is None:
            done = True
        else:
            lines.append(item)
        for _ in range(200):
            try:
                item = messages.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            lines.append(item)
        if not done:
            yield "Working…", None, "\n".join(lines)
    thread.join()
    if "error" in result:
        yield "Failed: " + result["error"], None, "\n".join(lines)
    else:
        yield "Completed.", result.get("value"), "\n".join(lines)


def build_h3_video_tab(tab_id: str = "minimax-h3-video") -> None:
    """Build the MiniMax H3 Video tab under Tools & 3D (flat layout, 0 accordions)."""
    diff_default = catalog.DEFAULT_DIFFUSION
    enc_default = catalog.first_installed_encoder()
    vae_default = catalog.first_installed_vae()

    with gr.Tab("🎬 MiniMax H3 Video", id=tab_id):
        gr.Markdown(
            "### MiniMax H3 Video & HyperFlow H3 Flash (Video + Stereo Audio)\n"
            "Runs locally via `sd-cli --mode vid_gen` with **auto-fit RAM offload**, "
            "**spatial + temporal VAE tiling**, and **attention chunk budgeting** tuned for "
            "**11–12 GB GPUs (RTX 2080 Ti / RTX 3060) + 64 GB RAM**. "
            "HyperFlow H3 Flash generates synchronized 24 fps video + stereo audio in **4–8 steps**; "
            "shared MiniMax H3 Image text encoders are reused without duplicating disk space."
        )

        with gr.Group():
            with gr.Row():
                diffusion = gr.Dropdown(
                    [(e.label, e.key) for e in catalog.DIFFUSION_VARIANTS.values()],
                    value=diff_default,
                    label="Video diffusion checkpoint",
                    elem_id="h3v-diffusion",
                    scale=2,
                )
                encoder = gr.Dropdown(
                    [(e.label, e.key) for e in catalog.ENCODER_VARIANTS.values()],
                    value=enc_default,
                    label="Text encoder (shared with H3 Image)",
                    elem_id="h3v-encoder",
                    scale=2,
                )
                vae = gr.Dropdown(
                    [(e.label, e.key) for e in catalog.VIDEO_VAE_VARIANTS.values()],
                    value=vae_default,
                    label="Video VAE",
                    elem_id="h3v-vae",
                    scale=2,
                )
            with gr.Row():
                install_btn = gr.Button(
                    "⬇️ Install / download selected H3 Video pack",
                    size="sm",
                    elem_id="h3v-install",
                    scale=2,
                )
                ready_md = gr.Markdown(
                    _status_text(diff_default, enc_default, vae_default),
                    elem_id="h3v-ready",
                )

        last_seed_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Video & audio prompt",
                    lines=4,
                    placeholder=(
                        "Describe the scene, camera motion, lighting, and soundtrack "
                        "(e.g. 'A rally car drifting down an alpine snowy mountain pass at sunset, "
                        "engine roar echoing off granite cliffs, snow spray, cinematic drone tracking shot')."
                    ),
                    elem_id="h3v-prompt",
                )

                with gr.Row():
                    res_preset = gr.Dropdown(
                        [(lbl, f"{w}x{h}") for lbl, (w, h) in engine.RESOLUTION_PRESETS],
                        value="864x480",
                        label="Resolution (multiples of 32)",
                        scale=2,
                    )
                    frames = gr.Dropdown(
                        [(lbl, count) for lbl, count in engine.FRAME_PRESETS],
                        value=56,
                        label="Duration (24 fps · 17k + 5 grid)",
                        scale=2,
                    )

                with gr.Row():
                    steps = gr.Dropdown(
                        [
                            ("4 steps (HyperFlow Flash · fastest)", 4),
                            ("6 steps (HyperFlow Flash · recommended)", 6),
                            ("8 steps (HyperFlow Flash · detailed)", 8),
                            ("20 steps (Base H3)", 20),
                            ("30 steps (Base H3 · high quality)", 30),
                        ],
                        value=6,
                        label="Steps",
                        scale=2,
                    )
                    attn_budget = gr.Dropdown(
                        [
                            ("0.25 GiB (safest for 11-12 GB / long clips)", 0.25),
                            ("0.5 GiB (recommended for 2080 Ti / 3060)", 0.5),
                            ("1.0 GiB (faster · more VRAM)", 1.0),
                            ("2.0 GiB (sd.cpp default · 16+ GB)", 2.0),
                        ],
                        value=0.5,
                        label="Attention chunk budget",
                        scale=2,
                    )

                with gr.Row():
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 = random)",
                        scale=2,
                        elem_id="h3v-seed",
                    )
                    reuse_seed = gr.Button("🔒 Reuse last seed", size="sm", scale=1)
                    temporal_tiling = gr.Checkbox(
                        value=True,
                        label="Temporal VAE tiling",
                        scale=1,
                    )

                with gr.Group():
                    gr.Markdown(
                        "**Optional conditioning:** Use **First / Last frame** with `FL2VA` checkpoints, "
                        "or **Reference image** with `Ref2VA` checkpoints."
                    )
                    with gr.Row():
                        init_img = gr.Image(
                            label="First frame (I2VA / FL2VA)",
                            type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY,
                            height=160,
                        )
                        end_img = gr.Image(
                            label="Last frame (FL2VA, optional)",
                            type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY,
                            height=160,
                        )
                        ref_img = gr.Image(
                            label="Reference image (Ref2VA only)",
                            type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY,
                            height=160,
                        )

                with gr.Row():
                    hires_enable = gr.Checkbox(
                        value=False,
                        label="2-stage latent hires refine (T2VA / Ref2VA only)",
                        scale=2,
                    )
                    hires_target = gr.Dropdown(
                        [
                            ("1216×672 (720p 16:9)", "1216x672"),
                            ("672×1216 (720p 9:16)", "672x1216"),
                            ("960×960 (Square HD)", "960x960"),
                        ],
                        value="1216x672",
                        label="Hires target",
                        scale=2,
                    )
                    hires_steps = gr.Slider(
                        2, 8, value=4, step=1, label="Hires steps", scale=1
                    )

                with gr.Row():
                    run_btn = gr.Button(
                        "🎬 Generate H3 Video + Audio",
                        variant="primary",
                        scale=3,
                        elem_id="h3v-run",
                    )
                    stop_btn = gr.Button("⏹️ Cancel", variant="stop", scale=1)
                status = gr.Markdown("", elem_id="h3v-status")

            with gr.Column(scale=4):
                video_out = gr.Video(
                    label="Generated video (24 fps + stereo audio)",
                    elem_id="h3v-result",
                )
                download = gr.File(label="Download MP4", interactive=False)
                log = gr.Textbox(
                    label="Log",
                    lines=14,
                    autoscroll=True,
                    elem_classes="log-box",
                )

        def _refresh_ready(diff_k, enc_k, vae_k):
            return _status_text(diff_k, enc_k, vae_k)

        for dd in (diffusion, encoder, vae):
            dd.change(
                _refresh_ready,
                inputs=[diffusion, encoder, vae],
                outputs=[ready_md],
                queue=False,
            )

        def _reuse(last_s):
            if last_s is None:
                return gr.update(), "No previous seed available yet."
            return gr.update(value=int(last_s)), f"🔒 Locked seed: `{int(last_s)}`"

        reuse_seed.click(
            _reuse, inputs=[last_seed_state], outputs=[seed, status], queue=False
        )

        def install_h3_video(diff_k, enc_k, vae_k):
            for state, _, text in _stream_runner(
                lambda report: catalog.download_pack(diff_k, enc_k, vae_k, log=report)
            ):
                yield state, text, _status_text(diff_k, enc_k, vae_k)

        install_btn.click(
            install_h3_video,
            inputs=[diffusion, encoder, vae],
            outputs=[status, log, ready_md],
            **widgets.GPU_QUEUE,
        )

        def do_h3_video(
            prompt_txt,
            diff_k,
            enc_k,
            vae_k,
            res_str,
            frame_count,
            steps_v,
            chunk_v,
            seed_v,
            temp_tile,
            first_im,
            last_im,
            ref_im,
            hires_on,
            hires_str,
            hires_st,
        ):
            w_s, h_s = (res_str or "864x480").split("x")
            hw_s, hh_s = (hires_str or "1216x672").split("x")

            def _job(report):
                with tempfile.TemporaryDirectory(prefix="h3v-", dir=settings.TMP_DIR) as tmp:
                    tmp_dir = Path(tmp)
                    init_p = end_p = None
                    refs_p: list[Path] = []
                    if isinstance(first_im, Image.Image):
                        init_p = tmp_dir / "first.png"
                        first_im.convert("RGB").save(init_p)
                    if isinstance(last_im, Image.Image):
                        end_p = tmp_dir / "last.png"
                        last_im.convert("RGB").save(end_p)
                    if isinstance(ref_im, Image.Image):
                        rp = tmp_dir / "ref1.png"
                        ref_im.convert("RGB").save(rp)
                        refs_p.append(rp)
                    return engine.generate_video(
                        prompt=prompt_txt,
                        diffusion_key=diff_k,
                        encoder_key=enc_k,
                        vae_key=vae_k,
                        width=int(w_s),
                        height=int(h_s),
                        video_frames=int(frame_count),
                        steps=int(steps_v),
                        seed=int(seed_v) if seed_v is not None else -1,
                        init_image=init_p,
                        end_image=end_p,
                        ref_images=refs_p or None,
                        attn_chunk_budget=float(chunk_v),
                        temporal_tiling=bool(temp_tile),
                        hires_enable=bool(hires_on),
                        hires_width=int(hw_s),
                        hires_height=int(hh_s),
                        hires_steps=int(hires_st),
                        log=report,
                    )

            for state, payload, text in _stream_runner(_job):
                if payload is not None:
                    out_path, used_seed = payload
                    yield (
                        f"✅ Video ready: `{out_path.name}` · **Seed:** `{used_seed}`",
                        str(out_path),
                        str(out_path),
                        text,
                        used_seed,
                    )
                else:
                    yield state, gr.update(), gr.update(), text, gr.update()

        evt = run_btn.click(
            do_h3_video,
            inputs=[
                prompt,
                diffusion,
                encoder,
                vae,
                res_preset,
                frames,
                steps,
                attn_budget,
                seed,
                temporal_tiling,
                init_img,
                end_img,
                ref_img,
                hires_enable,
                hires_target,
                hires_steps,
            ],
            outputs=[status, video_out, download, log, last_seed_state],
            **widgets.GPU_QUEUE,
        )
        widgets.stop_into_status(stop_btn, sdcpp.cancel_active, status, [evt])
