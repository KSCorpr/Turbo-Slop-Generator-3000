#!/usr/bin/env python3
"""Standalone CPU tracing, official LayerTracer stages and validated SVG exports."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
DRAWABLE = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "text"}
ALLOWED = DRAWABLE | {"svg", "g", "defs", "linearGradient", "radialGradient", "stop",
                      "clipPath", "mask", "title", "desc", "tspan"}


def parse_svg(text, *, require_paths=True):
    if len(text) > 20_000_000 or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
        raise ValueError("Invalid or oversized SVG.")
    root = ET.fromstring(text)
    if root.tag != f"{{{NS}}}svg":
        raise ValueError("Expected an SVG document.")
    drawn = False
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag not in ALLOWED:
            raise ValueError("Unsupported SVG element: " + tag)
        drawn |= tag in DRAWABLE
        for key, value in element.attrib.items():
            key = key.split("}")[-1].lower()
            if (key.startswith("on") or key == "style" or
                    (key == "href" and not value.startswith("#")) or
                    any(not ref.strip(" ' \"").startswith("#")
                        for ref in re.findall(r"url\s*\((.*?)\)", value, re.I))):
                raise ValueError("External or active SVG content is not supported.")
    if require_paths and not drawn:
        raise ValueError("No vector shapes were produced.")
    return root


def write_svg(root, path):
    text = ET.tostring(root, encoding="unicode")
    parse_svg(text)
    Path(path).write_text(text, encoding="utf-8")


def export(svg, output, metadata, layers=()):
    import resvg_py
    from PIL import Image
    import io
    output = Path(output)
    root = parse_svg(svg)
    write_svg(root, output / "result.svg")
    # Render only validated vectors, with a bounded preview and no resource directory.
    png = resvg_py.svg_to_bytes(svg_string=ET.tostring(root, encoding="unicode"),
                                width=1024, height=1024, skip_system_fonts=True)
    with Image.open(io.BytesIO(png)) as preview:
        preview.save(output / "preview.png")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "vectors.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [output / name for name in ("result.svg", "preview.png", "manifest.json")] + list(layers):
            archive.write(path, path.relative_to(output))


def trace(source, output):
    import vtracer
    target = Path(output) / "traced.svg"
    vtracer.convert_image_to_svg_py(str(source), str(target), colormode="color",
                                   mode="spline", filter_speckle=4, color_precision=6)
    export(target.read_text(encoding="utf-8"), output, {"engine": "VTracer", "version": "0.6.11"})
    target.unlink()


def layertrace(source, output, addon_root, grid, threshold=5):
    from PIL import Image
    side = 2 if grid == 4 else 3
    with Image.open(source) as image:
        width, height = image.size
    if width % side or height % side or min(width, height) < side * 32:
        raise ValueError(f"Use a {side} × {side} grid with equal cells, at least 32 px per cell.")
    script = Path(addon_root) / "source" / (
        "layertracer_emoji_4grid.py" if grid == 4 else "layertracer_icon_9grid.py")
    spec = importlib.util.spec_from_file_location("official_layertracer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Caller uses an ASCII working directory and relative input for OpenCV on Windows.
    split = module.split_image_to_4 if grid == 4 else module.split_image_to_9
    _, folder = split(str(source), "traced")
    folder = Path(folder)
    options = SimpleNamespace(colormode="color", mode="spline", filter_speckle=8,
        color_precision=8, corner_threshold=60, length_threshold=4,
        splice_threshold=45, path_precision=3)
    base = 1 if grid == 4 else 2
    base_svg = folder / f"cell_{base}.svg"
    module.convert_to_svg(str(folder / f"cell_{base}.png"), str(base_svg), options)
    sources = [("base", base_svg)]
    # Include 8 → 9, missing from the upstream nine-panel main() loop.
    for index in range(base, grid):
        png, svg = module.process_consecutive_images(str(folder / f"cell_{index}.png"),
            str(folder / f"cell_{index+1}.png"), index, str(folder), thresh=threshold)
        if not png or not svg:
            raise RuntimeError(f"LayerTracer failed on panel {index+1}.")
        module.convert_to_svg(png, svg, options)
        sources.append((f"step-{index+1}", Path(svg)))
    if grid == 9:
        module.extract_black_lines(str(folder / "cell_1.png"), str(folder))
        sources.append(("outlines", folder / "difference_1.svg"))
    # Empty difference layers are legitimate; missing files are upstream failures.
    root = parse_svg(base_svg.read_text(encoding="utf-8"))
    combined = ET.Element(root.tag, dict(root.attrib))
    layers = []
    layer_dir = Path(output) / "layers"
    layer_dir.mkdir()
    nonempty = 0
    for index, (label, path) in enumerate(sources):
        layer = parse_svg(path.read_text(encoding="utf-8"), require_paths=False)
        group = ET.SubElement(combined, f"{{{NS}}}g", {"id": f"layer-{index+1}-{label}"})
        for child in layer:
            group.append(copy.deepcopy(child))
        nonempty += any(node.tag.split("}")[-1] in DRAWABLE for node in layer.iter())
        dest = layer_dir / f"{index+1:02d}-{label}.svg"
        dest.write_text(ET.tostring(layer, encoding="unicode"), encoding="utf-8")
        layers.append(dest)
    export(ET.tostring(combined, encoding="unicode"), output,
        {"engine": "LayerTracer", "grid": grid, "layers": len(layers),
         "nonempty_layers": nonempty, "stage": "prepared-sequence-to-svg",
         "commit": "8e089f92795b4f32dcae8f7dbbd5ced20395dad7"}, layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trace", "layer4", "layer9"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=5, choices=range(1, 51))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "trace":
        trace(args.source, args.output)
    else:
        layertrace(args.source, args.output, args.addon_root, 4 if args.mode == "layer4" else 9, args.threshold)
    print("SVG export completed.", flush=True)
