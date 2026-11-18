"""3D Render & PNG Texture Studio tab.

Lets the user load a raw 3D render (clay/grey viewport render) or a transparent
PNG silhouette, apply a material or artistic style prompt using a local
reference-capable image model, and automatically composite the exact source
alpha channel and proportions onto the generated result.
"""
from __future__ import annotations

import queue
import random
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import png_style as ps
from ..i18n import t
from . import widgets
from .outpaint_tab import _cleanup_preview, _stream_outpaint


_BACKDROPS: dict[str, tuple[int, int, int]] = {
    "grey": (128, 128, 128),
    "white": (245, 245, 245),
    "dark": (36, 36, 36),
}


def _is_edit(m) -> bool:
    return (m.defaults.get("edit") if m else None) in (True, "full")


def _models() -> list:
    return [
        m
        for m in registry.load_base_models(settings.load_prefs())
        if _is_edit(m) or m.defaults.get("supports_img2img", True)
    ]


def _model_choices() -> tuple[list[tuple[str, str]], str | None]:
    choices, first_edit, first_any = [], None, None
    for m in _models():
        ready = registry.model_is_ready(m)
        edit = _is_edit(m)
        if ready and edit and first_edit is None:
            first_edit = m.id
        if ready and first_any is None:
            first_any = m.id
        label = (
            m.name
            if edit
            else f"{m.name} — img2img fallback (reference editing recommended)"
        )
        choices.append((label if ready else f"{label} (not installed)", m.id))
    return choices, (first_edit or first_any)


def _defaults(model_id: str | None) -> dict:
    m = (
        registry.get_base_model(model_id, settings.load_prefs())
        if model_id
        else None
    )
    d = dict(m.defaults) if m else {}
    family = m.family if m else ""
    base_w = int(d.get("width", 1024) or 1024)
    base_h = int(d.get("height", 1024) or 1024)
    base_size = max(base_w, base_h)
    multiple = int(d.get("size_multiple", 16) or 16)
    default_max = 6272 if family == "minimax_h3_image" else 4096
    max_side = int(d.get("size_max", default_max) or default_max)
    return {
        "steps": int(d.get("steps", 8) or 8),
        "cfg_scale": float(d.get("cfg_scale", 1.0) or 1.0),
        "sampler": d.get("sampler") or "euler",
        "schedule": d.get("scheduler") or "auto",
        "flow_shift": float(d.get("flow_shift", 0.0) or 0.0),
        "edit": _is_edit(m),
        "family": family,
        "base_size": base_size,
        "multiple": multiple,
        "max_side": max_side,
    }


def _describe_source(img: Image.Image | None, model_id: str | None, preserve_prop: bool) -> str:
    if img is None:
        return "Upload a raw 3D render or transparent PNG to inspect its geometry and alpha channel."
    info = ps.analyze_image(img)
    d = _defaults(model_id)
    gen_w, gen_h = ps.generation_size_for_model(
        info["width"],
        info["height"],
        base_size=d["base_size"],
        multiple=d["multiple"],
        max_side=d["max_side"],
    )
    final_w, final_h = ps.compute_target_dimensions(
        (info["width"], info["height"]),
        (gen_w, gen_h),
        preserve_proportions=bool(preserve_prop),
    )
    alpha_badge = (
        "✅ **Transparent PNG** (alpha channel detected)"
        if info["has_alpha"]
        else "ℹ️ **Opaque PNG** (no transparency detected)"
    )
    return (
        f"**Source:** `{info['width']} × {info['height']} px` · "
        f"**Aspect ratio:** `{info['aspect_ratio_str']}` ({info['exact_ratio']:.3f}) · "
        f"{alpha_badge}  \n"
        f"**AI working canvas:** `{gen_w} × {gen_h} px` → "
        f"**Final output:** `{final_w} × {final_h} px`"
    )


def _write_sidecar(
    out_path: Path,
    *,
    model_id: str,
    prompt_used: str,
    seed: int,
    steps: int,
    d: dict,
    source_size: tuple[int, int],
    gen_size: tuple[int, int],
    final_size: tuple[int, int],
    preserve_alpha: bool,
    preserve_proportions: bool,
) -> None:
    lines = [
        f"File: {out_path.name}",
        f"Workflow: 3D / PNG Texture Studio",
        f"Model: {model_id}",
        f"Prompt: {prompt_used}",
        f"Source size: {source_size[0]}x{source_size[1]}",
        f"Generation size: {gen_size[0]}x{gen_size[1]}",
        f"Final size: {final_size[0]}x{final_size[1]}",
        f"Preserve alpha: {preserve_alpha}",
        f"Native resolution mode: {preserve_proportions}",
        f"Seed: {seed}",
        f"Steps: {int(steps)}  ·  Sampler: {d['sampler']}  ·  CFG: {d['cfg_scale']}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    try:
        out_path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


def build_png_style_tab(tab_id: str = "png-style") -> None:
    """Build the 3D Render & PNG Texture Studio tab under Tools & 3D."""
    with gr.Tab("🎨 3D / PNG Texture", id=tab_id):
        gr.Markdown(
            "### 3D Render & PNG Texture Studio\n"
            "Use an **untextured 3D render** (clay / viewport pass) or a **transparent PNG silhouette** "
            "as an exact geometric reference to generate a richly textured version.\n\n"
            "1. **Reference conditioning:** Sends your 3D render to a local editing model "
            "(**Flux.2 Klein**, **Qwen Image 2.1 Turbo**, **MiniMax H3 Image**, or **Fibo 2.0**) "
            "at the matching aspect ratio so geometry and proportions never stretch.\n"
            "2. **Exact silhouette & alpha transfer:** Automatically transfers the original PNG's "
            "sub-pixel alpha mask (`0..255`) and exact aspect ratio onto the textured output."
        )

        _choices, _first = _model_choices()
        preset_choices = [(p.label, p.id) for p in ps.STYLE_PRESETS]
        raw_gen_state = gr.State(None)
        last_out_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(
                    label="Raw 3D render / source PNG",
                    type="pil",
                    image_mode="RGBA",
                    buttons=widgets.IMAGE_VIEW_ONLY,
                    elem_id="png-style-source",
                )
                info_md = gr.Markdown(_describe_source(None, _first, False))

                model = gr.Dropdown(
                    _choices,
                    value=_first,
                    label="Model used",
                    elem_id="png-style-model",
                )
                mode_md = gr.Markdown("")

                preset = gr.Radio(
                    preset_choices,
                    value=ps.STYLE_PRESETS[0].id,
                    label="Quick 3D Material & Style Presets",
                    elem_id="png-style-preset",
                )
                prompt = gr.Textbox(
                    label="Material / style prompt",
                    value=ps.STYLE_PRESETS[0].prompt,
                    lines=3,
                    placeholder="Describe the target 3D surface materials, lighting, and texture style...",
                    elem_id="png-style-prompt",
                )

                with gr.Group():
                    lock_silhouette = gr.Checkbox(
                        value=True,
                        label="Lock 3D silhouette & full edge-to-edge coverage",
                        info="Appends strict 3D geometry and edge-to-edge silhouette coverage instructions.",
                    )
                    preserve_alpha = gr.Checkbox(
                        value=True,
                        label="Preserve PNG transparency (exact alpha mask)",
                        info="Transfers the original PNG alpha channel pixel-for-pixel onto the generated texture.",
                    )
                    preserve_proportions = gr.Checkbox(
                        value=False,
                        label="Keep model native resolution (exact aspect ratio)",
                        info="Unchecked: resizes output to exact source pixel dimensions (W × H). "
                        "Checked: keeps the AI model's native resolution adjusted to the exact source ratio.",
                    )
                    mask_png = gr.Image(
                        label="Optional external alpha mask PNG (to slice layers from a full render)",
                        type="pil",
                        image_mode="RGBA",
                        buttons=widgets.IMAGE_VIEW_ONLY,
                        elem_id="png-style-mask",
                    )

                with gr.Row():
                    backdrop = gr.Dropdown(
                        [
                            ("Neutral studio grey (recommended)", "grey"),
                            ("Pure white backdrop", "white"),
                            ("Dark studio backdrop", "dark"),
                            ("Raw RGBA (Qwen 2.1)", "rgba"),
                        ],
                        value="grey",
                        label="Reference backdrop for transparent areas",
                        scale=2,
                    )
                    steps = gr.Slider(
                        1,
                        50,
                        value=_defaults(_first)["steps"],
                        step=1,
                        label="Steps",
                        interactive=_first != "qwen-image-2.1-turbo",
                        scale=2,
                    )
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 = random)",
                        scale=1,
                    )

                with gr.Row():
                    run = gr.Button(
                        "✨ Generate Textured PNG",
                        variant="primary",
                        scale=3,
                        elem_id="png-style-run",
                    )
                    stop = gr.Button("⏹️ Cancel", variant="stop", scale=1)
                status = gr.Markdown("", elem_id="png-style-status")

            with gr.Column(scale=4):
                result = gr.Image(
                    label="Textured PNG result (RGBA)",
                    type="filepath",
                    image_mode="RGBA",
                    format="png",
                    buttons=widgets.IMAGE_BUTTONS,
                    elem_id="png-style-result",
                )
                comparator = gr.ImageSlider(
                    label="Before / After comparison (Raw 3D vs Textured)",
                    type="pil",
                    elem_id="png-style-compare",
                )
                with gr.Row():
                    download = gr.File(
                        label="Download PNG",
                        interactive=False,
                        scale=2,
                    )
                    again = gr.Button(
                        "♻️ Use textured result as new source PNG",
                        size="sm",
                        scale=2,
                    )
                log = gr.Textbox(
                    label="Log",
                    lines=10,
                    autoscroll=True,
                    elem_classes="log-box",
                )

        def _on_preset(preset_id: str) -> str:
            p = ps.PRESET_BY_ID.get(preset_id)
            return p.prompt if p else ""

        preset.change(_on_preset, inputs=[preset], outputs=[prompt], queue=False)

        def _on_source_or_plan(img, mid, preserve_prop):
            if img is None:
                return _describe_source(None, mid, preserve_prop), gr.update()
            info = ps.analyze_image(img)
            # Automatically enable alpha preservation when the source PNG is transparent
            return (
                _describe_source(img, mid, preserve_prop),
                gr.update(value=info["has_alpha"]),
            )

        image.change(
            _on_source_or_plan,
            inputs=[image, model, preserve_proportions],
            outputs=[info_md, preserve_alpha],
            queue=False,
        )

        def _on_model(mid):
            dd = _defaults(mid)
            if dd["edit"]:
                note = (
                    "✅ **Reference-editing model** — receives your 3D render via "
                    "its vision encoder (`-r`) and textures the exact 3D form."
                )
            else:
                note = (
                    "⚠️ **img2img fallback** — this model has no reference vision "
                    "encoder; it denoises the source image (`-i` at 0.70 strength)."
                )
            return (
                gr.update(
                    value=dd["steps"],
                    interactive=mid != "qwen-image-2.1-turbo",
                ),
                gr.update(value=note),
            )

        model.change(
            _on_model,
            inputs=[model],
            outputs=[steps, mode_md],
            queue=False,
        )
        for comp in (model, preserve_proportions):
            comp.change(
                _describe_source,
                inputs=[image, model, preserve_proportions],
                outputs=[info_md],
                queue=False,
            )

        def re_apply_mask(
            img: Image.Image | None,
            mask_override: Image.Image | None,
            raw_path: str | None,
            out_path_str: str | None,
            preserve_a: bool,
            preserve_prop: bool,
            mid: str | None,
        ):
            """Instant client-like re-compositing without re-running sd-cli."""
            desc = _describe_source(img, mid, preserve_prop)
            if img is None or not raw_path or not Path(raw_path).is_file():
                return desc, gr.update(), gr.update(), gr.update(), gr.update()
            alpha_src = mask_override if mask_override is not None else img
            with Image.open(raw_path) as raw_gen:
                final_img = ps.composite_and_transfer_alpha(
                    raw_gen,
                    alpha_src,
                    preserve_alpha=bool(preserve_a),
                    preserve_proportions=bool(preserve_prop),
                )
            comp_before = ps.prepare_comparison_source(img, final_img.size)
            target_path = (
                Path(out_path_str)
                if out_path_str and mask_override is None
                else settings.OUTPUT_DIR / f"png-style-{time.time_ns()}.png"
            )
            final_img.save(target_path, format="PNG")
            return (
                desc,
                gr.update(value=str(target_path)),
                (comp_before, final_img),
                gr.update(value=str(target_path)),
                t("✅ Updated mask & proportions ({w}×{h} px) → {n}").format(
                    w=final_img.width,
                    h=final_img.height,
                    n=target_path.name,
                ),
            )

        for toggle in (preserve_alpha, preserve_proportions, mask_png):
            toggle.change(
                re_apply_mask,
                inputs=[
                    image,
                    mask_png,
                    raw_gen_state,
                    last_out_state,
                    preserve_alpha,
                    preserve_proportions,
                    model,
                ],
                outputs=[info_md, result, comparator, download, status],
                queue=False,
            )

        def do_png_style(
            img: Image.Image | None,
            mask_override: Image.Image | None,
            model_id: str | None,
            prompt_txt: str,
            lock_sil: bool,
            preserve_a: bool,
            preserve_prop: bool,
            backdrop_mode: str,
            steps_v: int,
            seed_v: int,
        ):
            if img is None:
                raise gr.Error(t("Load a raw 3D render or PNG image first."))
            if not model_id:
                raise gr.Error(t("Pick a model."))
            if not (prompt_txt or "").strip():
                raise gr.Error(t("Enter a material or style prompt."))

            settings.ensure_dirs()
            d = _defaults(model_id)
            info = ps.analyze_image(img)
            gen_w, gen_h = ps.generation_size_for_model(
                info["width"],
                info["height"],
                base_size=d["base_size"],
                multiple=d["multiple"],
                max_side=d["max_side"],
            )

            keep_rgba = backdrop_mode == "rgba" and d["family"] == "qwen21"
            bg_rgb = _BACKDROPS.get(backdrop_mode, (128, 128, 128))
            ref_img = ps.prepare_reference_image(
                img, bg_color=bg_rgb, keep_rgba=keep_rgba
            )
            if ref_img.size != (gen_w, gen_h):
                ref_img = ref_img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)

            job_id = time.time_ns()
            ref_path = settings.TMP_DIR / f"png_style_{job_id}_ref.png"
            ref_img.save(ref_path, format="PNG")
            preview_path = settings.TMP_DIR / f"png_style_{job_id}_preview_%03d.png"

            try:
                s = int(seed_v)
            except (TypeError, ValueError):
                s = -1
            if s < 0:
                s = random.randint(0, 2**31 - 1)

            full_prompt = ps.build_texture_prompt(
                prompt_txt,
                model_family=d["family"],
                lock_silhouette=bool(lock_sil),
            )

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}
            consumer_closed = threading.Event()

            def worker():
                try:
                    kw = dict(
                        model_id=model_id,
                        prompt=full_prompt,
                        negative="",
                        steps=int(steps_v),
                        cfg_scale=d["cfg_scale"],
                        width=gen_w,
                        height=gen_h,
                        seed=s,
                        batch_count=1,
                        sampler=d["sampler"],
                        schedule=d["schedule"],
                        flow_shift=d["flow_shift"],
                        log=q.put,
                        save_prompt=False,
                        preview_path=preview_path,
                    )
                    if d["edit"]:
                        kw["ref_image"] = [ref_path]
                    else:
                        kw["init_image"] = ref_path
                        kw["strength"] = 0.70
                    outs = gen_engine.generate(**kw)
                    state["outs"] = [str(x) for x in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    try:
                        ref_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    state["done"] = True
                    if consumer_closed.is_set():
                        _cleanup_preview(preview_path)
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs = [
                f"Source PNG: {info['width']}×{info['height']} px · "
                f"ratio {info['aspect_ratio_str']} · "
                f"alpha={'yes' if info['has_alpha'] else 'no'}",
                f"Working canvas: {gen_w}×{gen_h} px · Seed: {s}",
                f"Model: {model_id} · {d['sampler']} · cfg {d['cfg_scale']} · {int(steps_v)} steps",
                f"Prompt: {full_prompt[:180]}...",
            ]
            try:
                for status_txt, frame_upd, log_txt in _stream_outpaint(
                    q, preview_path, int(steps_v), logs
                ):
                    yield (
                        status_txt,
                        frame_upd,
                        gr.update(),
                        gr.update(),
                        log_txt,
                        gr.update(),
                        gr.update(),
                    )
            finally:
                consumer_closed.set()
                if state.get("done"):
                    _cleanup_preview(preview_path)

            if "err" in state or not state.get("outs"):
                logs.append(f"\n[ERROR] {state.get('err', 'no output')}")
                yield (
                    t("❌ Failed — see the log."),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "\n".join(logs),
                    gr.update(),
                    gr.update(),
                )
                return

            raw_out_path = Path(state["outs"][0])
            # Keep a copy of the raw generated image in TMP_DIR so toggling
            # preserve_alpha / preserve_proportions re-masks without re-generating.
            cached_raw = settings.TMP_DIR / f"png_style_{job_id}_raw.png"
            alpha_src = mask_override if mask_override is not None else img
            with Image.open(raw_out_path) as raw_gen:
                raw_gen.save(cached_raw, format="PNG")
                final_img = ps.composite_and_transfer_alpha(
                    raw_gen,
                    alpha_src,
                    preserve_alpha=bool(preserve_a),
                    preserve_proportions=bool(preserve_prop),
                )

            out_path = settings.OUTPUT_DIR / f"png-style-{job_id}.png"
            final_img.save(out_path, format="PNG")
            # Remove the intermediate unmasked file created by generate() in OUTPUT_DIR
            if raw_out_path != out_path:
                try:
                    raw_out_path.unlink(missing_ok=True)
                except OSError:
                    pass

            _write_sidecar(
                out_path,
                model_id=model_id,
                prompt_used=full_prompt,
                seed=s,
                steps=int(steps_v),
                d=d,
                source_size=(info["width"], info["height"]),
                gen_size=(gen_w, gen_h),
                final_size=final_img.size,
                preserve_alpha=bool(preserve_a),
                preserve_proportions=bool(preserve_prop),
            )
            comp_before = ps.prepare_comparison_source(img, final_img.size)
            logs.append(
                f"\n✅ Alpha & proportions applied ({final_img.width}×{final_img.height} px) → {out_path.name}"
            )
            yield (
                t("✅ Textured PNG ready: {n} ({w}×{h} px)").format(
                    n=out_path.name,
                    w=final_img.width,
                    h=final_img.height,
                ),
                gr.update(value=str(out_path)),
                (comp_before, final_img),
                gr.update(value=str(out_path)),
                "\n".join(logs),
                str(cached_raw),
                str(out_path),
            )

        evt = run.click(
            do_png_style,
            inputs=[
                image,
                mask_png,
                model,
                prompt,
                lock_silhouette,
                preserve_alpha,
                preserve_proportions,
                backdrop,
                steps,
                seed,
            ],
            outputs=[
                status,
                result,
                comparator,
                download,
                log,
                raw_gen_state,
                last_out_state,
            ],
            stream_every=0.15,
            **widgets.GPU_QUEUE,
        )
        widgets.stop_into_status(stop, gen_engine.cancel, status, [evt])

        def _again(path):
            if not path:
                return gr.update(), t("No textured result to reload.")
            with Image.open(path) as loaded:
                rgba = loaded.convert("RGBA").copy()
            return (
                gr.update(value=rgba),
                t("Textured PNG reloaded as the new source reference."),
            )

        again.click(_again, inputs=[result], outputs=[image, status], queue=False)
