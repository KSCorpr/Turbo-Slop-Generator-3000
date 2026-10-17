"""Local image/text to SVG and prepared LayerTracer sequence reconstruction."""
from pathlib import Path
import json
import re

import gradio as gr

from .. import addons, settings
from ..engine import vectorize
from . import widgets
from .media_tabs import stream_task

CHOICES = [("VTracer · image → SVG · CPU", "trace"),
           ("HiVG (experimental) · image → SVG · AI", "hivg-image"),
           ("HiVG (experimental) · text → SVG · AI", "hivg-text"),
           ("OmniSVG 1.1 4B · image → SVG · AI", "omnisvg-image"),
           ("OmniSVG 1.1 4B · text → SVG · AI", "omnisvg-text"),
           ("LayerTracer · 4-panel drawing sequence → layers", "layer4"),
           ("LayerTracer · 9-panel drawing sequence → layers", "layer9")]


def mode_help(mode):
    descriptions = {
        "trace": "CPU tracing follows the source colors and contours. Best for logos and flat illustrations.",
        "hivg-image": "Experimental HiVG reconstruction. Shapes and lettering may change. Some compressed outputs cannot be decoded by the public release; failures provide a diagnostic ZIP. Try OmniSVG for those cases.",
        "hivg-text": "Experimental HiVG illustration generation. Some compressed outputs cannot be decoded by the public release. Start with a simple icon; failures provide a diagnostic ZIP.",
        "omnisvg-image": "OmniSVG reconstructs SVG shapes from your image. Compare lettering and small details. Variants run one at a time; memory is checked again before loading.",
        "omnisvg-text": "Describe an illustration or icon. OmniSVG generates vector paths with the controls below. Variants run one at a time.",
        "layer4": "Import a prepared 2 × 2 sequence of the same drawing. Order: top left → top right → bottom right → bottom left. Equal cells, no gutters.",
        "layer9": "Import a prepared 3 × 3 sequence of the same drawing, read left to right, row by row. First cell: outlines; second: base colors. Equal cells, no gutters.",
    }
    ai = mode.startswith(("hivg", "omnisvg"))
    text = mode.endswith("-text")
    return (descriptions[mode], gr.update(visible=not text),
            gr.update(visible=text), gr.update(visible=ai, maximum=16384 if ai else 4096,
                                             value=4096 if ai else 2048),
            gr.update(visible=ai), gr.update(visible=mode.startswith("layer")),
            gr.update(visible=mode.startswith("omnisvg")))


def installed_status():
    return " · ".join(f"{label}: {'installed' if addons.ready(name) else 'not installed'}"
                       for name, label in (("svg", "CPU tracing"), ("hivg", "HiVG"), ("omnisvg", "OmniSVG")))


def variant_result(number, directory):
    if not directory or number is None:
        return None, None, ""
    folder = Path(directory).resolve()
    if not folder.is_relative_to(settings.OUTPUT_DIR.resolve()) or not folder.name.startswith("svg-"):
        raise ValueError("The SVG export is unavailable.")
    number = int(number)
    metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    selected = next((item for item in metadata.get("variants", []) if item["number"] == number), None)
    if selected is None:
        raise ValueError("Choose a generated SVG variant.")
    seed_label = f"seed {selected['seed']}" if selected.get("seed") is not None else "saved tokens"
    info = f"Variant {number} · {seed_label} · {selected['paths']} paths."
    if not selected.get("complete", True):
        info = "PARTIAL SVG · generation did not finish. " + info + " Missing shapes may remain; only finished color-delimited paths were kept."
    if metadata.get("recovered_from_diagnostics"):
        info += " Recovered on CPU from saved tokens; no model inference was run."
    if selected.get("pixel_difference_percent") is not None:
        info += (f" Pixel difference: {selected['pixel_difference_percent']}% "
                 "(lower is closer in pixel colors; this does not measure semantic or text accuracy).")
    stem = selected.get("file_stem", f"{number:02d}")
    if not isinstance(stem, str) or not re.fullmatch(r"\d{2}(?:\.partial)?", stem):
        raise ValueError("Invalid SVG variant filename.")
    base = folder / "candidates" / stem
    return str(base)+".png", str(base)+".svg", info


def build_svg_tab():
    with gr.Tab("SVG & layers", id="svg"):
        gr.Markdown("### Vector graphics\nExport real SVG shapes, a PNG preview and a ZIP. "
            "LayerTracer here reconstructs layers from an existing drawing sequence; "
            "its FLUX image-to-sequence stage is not included. A single image is not a drawing sequence.")
        with gr.Row():
            install_cpu = gr.Button("Install / repair CPU tracing", size="sm")
            install_ai = gr.Button("Install / repair HiVG (~8.14 GB weights)", size="sm")
            install_omni = gr.Button("Install / repair OmniSVG 4B (~7.71 GB)", size="sm")
        installed = gr.Markdown(installed_status())
        with gr.Row():
            with gr.Column():
                mode = gr.Dropdown(CHOICES, value="trace", label="SVG engine")
                help_text = gr.Markdown(mode_help("trace")[0])
                source = gr.Image(type="filepath", label="Source image or prepared drawing grid",
                                  format="png", image_mode="RGBA", height=260, buttons=widgets.IMAGE_BUTTONS)
                prompt = gr.Textbox(label="SVG description", lines=3, visible=False)
                tokens = gr.Slider(256, 4096, value=2048, step=256,
                    label="SVG token budget (higher uses more memory)", visible=False)
                seed = gr.Number(value=0, precision=0, label="SVG seed", visible=False)
                threshold = gr.Slider(1, 50, value=5, step=1,
                    label="LayerTracer difference threshold", visible=False)
                with gr.Group(visible=False) as omni_controls:
                    gr.Markdown("**OmniSVG controls** · RAM is checked before loading. "
                        "The VRAM limit applies to Torch allocations; CUDA overhead is separate. "
                        "A RAM budget is an estimate for admission and placement, not a system-wide hard limit.")
                    memory = gr.Dropdown([("Auto · GPU + RAM", "auto"),
                        ("CPU offload · lower VRAM, slower", "offload"), ("CPU only · slow", "cpu")],
                        value="auto", label="OmniSVG memory mode")
                    with gr.Row():
                        vram = gr.Number(value=0, minimum=0, maximum=128, step=.5,
                            label="VRAM budget (GiB, 0 = auto)")
                        ram = gr.Number(value=0, minimum=0, maximum=1024, step=1,
                            label="RAM budget (GiB, 0 = auto)")
                    check_memory = gr.Button("Check OmniSVG memory", size="sm")
                    with gr.Row():
                        temperature = gr.Slider(.05, 1.5, value=.3, step=.05, label="Temperature")
                        top_p = gr.Slider(.05, 1, value=.9, step=.05, label="Top-p")
                    with gr.Row():
                        top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                        repetition = gr.Slider(1, 1.5, value=1.05, step=.01, label="Repetition penalty")
                    with gr.Row():
                        variants = gr.Slider(1, 4, value=1, step=1, label="Variants (sequential)")
                        timeout = gr.Slider(0, 7200, value=0, step=30,
                            label="Generation time budget (seconds, 0 = no time limit)")
                    cache = gr.Checkbox(value=True, label="KV cache (faster, uses more VRAM)")
                    keep_aspect = gr.Checkbox(value=True, label="Keep source proportions (white padding)")
                    gr.Markdown("If generation reaches a limit, finished paths can be exported as an explicitly **partial SVG**. "
                                "A partial drawing may be missing shapes. Recover an older failed run below without loading GPU weights.")
                    recovery_file = gr.File(label="Saved OmniSVG diagnostic ZIP (optional)", file_types=[".zip"], type="filepath")
                    recover_button = gr.Button("Recover SVG from saved tokens · CPU", size="sm")
                with gr.Row():
                    start = gr.Button("Create SVG", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                preview = gr.Image(label="Vector render preview", interactive=False, height=400,
                                   buttons=widgets.IMAGE_VIEW_ONLY)
                svg = gr.File(label="SVG document", interactive=False)
                bundle = gr.File(label="ZIP · vectors or failure diagnostics", interactive=False)
                variant = gr.Dropdown([], label="Generated SVG variant", value=None, visible=False)
                result_info = gr.Markdown()
                result_directory = gr.State(None)
        status = gr.Markdown()
        log = gr.Textbox(lines=5, max_lines=8, interactive=False, label="SVG log")
        mode.change(mode_help, mode, [help_text, source, prompt, tokens, seed, threshold, omni_controls], queue=False)

        def install_svg(name):
            for message, _, lines in stream_task(lambda logger: vectorize.install(name, logger)):
                yield message, lines, installed_status()

        def install_svg_cpu():
            yield from install_svg("svg")

        def install_svg_ai():
            yield from install_svg("hivg")

        def install_svg_omni():
            yield from install_svg("omnisvg")

        controls = [memory, vram, ram, temperature, top_p, top_k, repetition, variants, timeout, cache, keep_aspect]
        keys = ["memory", "vram", "ram", "temperature", "top_p", "top_k", "repetition", "variants", "timeout", "cache", "keep_aspect"]

        def check_svg_memory(*values):
            for message, _, lines in stream_task(lambda logger:
                    vectorize.diagnose(dict(zip(keys, values)), logger)):
                yield message, lines

        def present_svg(action):
            for message, directory, lines in stream_task(action):
                folder = Path(directory) if directory else None
                if folder and (folder / "diagnostic.json").is_file():
                    report = json.loads((folder / "diagnostic.json").read_text(encoding="utf-8"))
                    yield ("Failed: " + report["error"], None, None, str(folder / "diagnostics.zip"), lines,
                           gr.update(choices=[], value=None, visible=False), None,
                           "No SVG was exported. The diagnostic ZIP contains the raw model output and stop reason.")
                    continue
                metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8")) if folder else {}
                choices = [(f"Variant {item['number']}" + (" · PARTIAL" if not item.get("complete", True) else ""), item["number"])
                           for item in metadata.get("variants", [])]
                selected = metadata.get("default_variant", 1)
                preview_path, svg_path, info = variant_result(selected, directory) if choices else (
                    str(folder / "preview.png") if folder else None, str(folder / "result.svg") if folder else None, "")
                if folder and not metadata.get("complete", True):
                    message = "Partial SVG exported. Generation did not finish; the drawing may be incomplete."
                yield (message, preview_path, svg_path,
                       str(folder / "vectors.zip") if folder else None, lines,
                       gr.update(choices=choices, value=selected if choices else None, visible=bool(choices)),
                       directory, info)

        def run_svg(mode, source, prompt, tokens, seed, threshold, *values):
            def execute(logger):
                try:
                    return vectorize.run(mode, source, prompt, tokens, seed, threshold, logger,
                                         omni=dict(zip(keys, values)))
                except vectorize.SVGDiagnosticError as exc:
                    return exc.directory
            yield from present_svg(execute)

        def recover_svg(path):
            yield from present_svg(lambda logger: vectorize.recover(path, logger))

        install_cpu.click(install_svg_cpu, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        install_ai.click(install_svg_ai, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        install_omni.click(install_svg_omni, outputs=[status, log, installed], **widgets.GPU_QUEUE)
        check_memory.click(check_svg_memory, controls, [status, log], **widgets.GPU_QUEUE)
        start.click(run_svg, [mode, source, prompt, tokens, seed, threshold, *controls],
                    [status, preview, svg, bundle, log, variant, result_directory, result_info], **widgets.GPU_QUEUE)
        recover_button.click(recover_svg, recovery_file,
                    [status, preview, svg, bundle, log, variant, result_directory, result_info], **widgets.GPU_QUEUE)
        variant.change(variant_result, [variant, result_directory], [preview, svg, result_info], queue=False)
        stop.click(vectorize.JOB.cancel, outputs=status, queue=False)
