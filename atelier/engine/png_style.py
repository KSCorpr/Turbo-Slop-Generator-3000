"""3D Render & PNG Texture Studio engine.

Transforms an untextured 3D render (clay/grey viewport render) or a transparent
PNG silhouette into a richly textured artwork while strictly preserving:
1. The exact 3D silhouette and sub-pixel alpha channel (`origA == 0` stays
   100% transparent, `0 < origA < 255` preserves anti-aliased edge transitions).
2. Either the exact source pixel dimensions (`sourceW x sourceH`) or the native
   generator resolution fitted to the exact mathematical aspect ratio
   (`sourceW / sourceH`) without any geometric stretching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from PIL import Image, ImageChops, ImageFilter


@dataclass(frozen=True)
class StylePreset:
    id: str
    label: str
    category: str
    prompt: str


STYLE_PRESETS: tuple[StylePreset, ...] = (
    # 3D Render / Asset Texturing presets
    StylePreset(
        id="pbr_realistic",
        label="PBR Realistic Materials",
        category="3d",
        prompt=(
            "photorealistic PBR 3D surface materials, physically based rendering, "
            "fine micro-surface roughness and specular detail, realistic ambient "
            "occlusion and subsurface scattering, crisp studio rim lighting"
        ),
    ),
    StylePreset(
        id="game_handpainted",
        label="Hand-Painted 3D Asset",
        category="3d",
        prompt=(
            "stylized hand-painted 3D game asset texture, expressive painterly "
            "brush strokes mapped onto the 3D form, rich color gradients, "
            "highlighted bevels and sculpted edges, stylized fantasy prop"
        ),
    ),
    StylePreset(
        id="weathered_metal",
        label="Weathered Metal & Patina",
        category="3d",
        prompt=(
            "heavy weathered industrial metal surface, scratched painted steel, "
            "worn bevel edges with exposed brushed iron, subtle oil patina and "
            "micro-pitting, realistic metallic roughness"
        ),
    ),
    StylePreset(
        id="scifi_hardsurface",
        label="Sci-Fi Hard Surface",
        category="3d",
        prompt=(
            "sleek sci-fi hard-surface 3D armor and carbon fiber panels, "
            "anodized matte titanium, recessed emissive neon seams, crisp "
            "machined panel lines and micro-decals"
        ),
    ),
    # PNG Style Studio presets (from the reference workflow)
    StylePreset(
        id="cyberpunk",
        label="Cyberpunk Neon",
        category="style",
        prompt=(
            "cyberpunk style, glowing holographic neon lines, high-tech "
            "intricate mechanical details, vibrant volumetric rim lighting"
        ),
    ),
    StylePreset(
        id="claymation",
        label="3D Claymation",
        category="style",
        prompt=(
            "3D claymation plasticine texture, handcrafted tactile clay "
            "stop-motion, soft studio shadows, charming tactile figurine"
        ),
    ),
    StylePreset(
        id="watercolor",
        label="Artistic Watercolor",
        category="style",
        prompt=(
            "ethereal delicate watercolor paint style, fluid pigment bleeding, "
            "wet-on-wet artistic paper texture, expressive splashes"
        ),
    ),
    StylePreset(
        id="anime_cel",
        label="Anime Cel Shading",
        category="style",
        prompt=(
            "clean modern Japanese anime aesthetic, vibrant crisp cel shading, "
            "bold dynamic line art, studio anime key visual"
        ),
    ),
    StylePreset(
        id="pixel_art",
        label="HD Pixel Art",
        category="style",
        prompt=(
            "crisp 16-bit retro pixel art sprite, sharp pixel grid, vibrant "
            "retro arcade color palette, clean edges"
        ),
    ),
    StylePreset(
        id="gold_metallic",
        label="Gold & Precious Metal",
        category="style",
        prompt=(
            "luxurious polished 24k molten gold and chrome metal finish, "
            "intricate filigree reflections, hyper-detailed specular highlights"
        ),
    ),
    StylePreset(
        id="glass_crystal",
        label="Crystal Glass & Prism",
        category="style",
        prompt=(
            "translucent tinted frosted glass, crystalline prism caustics, "
            "subtle iridescent chromatic dispersion"
        ),
    ),
    StylePreset(
        id="steampunk",
        label="Steampunk Brass",
        category="style",
        prompt=(
            "vintage Victorian steampunk aesthetic, polished brass gears, "
            "copper pipes, atmospheric vintage patina"
        ),
    ),
)

PRESET_BY_ID: dict[str, StylePreset] = {p.id: p for p in STYLE_PRESETS}


def get_closest_aspect_ratio(width: int, height: int) -> str:
    """Return the closest standard aspect ratio label for display/reference."""
    if width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    supported = (
        ("1:1", 1.0),
        ("16:9", 16.0 / 9.0),
        ("9:16", 9.0 / 16.0),
        ("4:3", 4.0 / 3.0),
        ("3:4", 3.0 / 4.0),
    )
    closest_name, closest_val = supported[0]
    min_diff = abs(ratio - closest_val)
    for name, val in supported[1:]:
        diff = abs(ratio - val)
        if diff < min_diff:
            min_diff = diff
            closest_name = name
    return closest_name


def analyze_alpha(img: Image.Image) -> bool:
    """Check whether the image contains any transparent or semi-transparent pixels."""
    if img.mode not in ("RGBA", "LA") and "transparency" not in img.info:
        return False
    rgba = img.convert("RGBA")
    sample = rgba
    if rgba.width > 256 or rgba.height > 256:
        sample = rgba.resize(
            (min(rgba.width, 256), min(rgba.height, 256)),
            Image.Resampling.NEAREST,
        )
    extrema = sample.getchannel("A").getextrema()
    if extrema is None:
        return False
    min_alpha, _ = extrema
    return min_alpha < 250


def analyze_image(img: Image.Image) -> dict:
    """Inspect a source 3D render / PNG image and return its geometry and alpha metadata."""
    w, h = img.size
    has_alpha = analyze_alpha(img)
    aspect_str = get_closest_aspect_ratio(w, h)
    return {
        "width": w,
        "height": h,
        "aspect_ratio_str": aspect_str,
        "exact_ratio": (w / h) if h else 1.0,
        "has_alpha": has_alpha,
    }


def generation_size_for_model(
    source_w: int,
    source_h: int,
    *,
    base_size: int = 1024,
    multiple: int = 16,
    max_side: int = 4096,
) -> tuple[int, int]:
    """Compute optimal generation dimensions preserving the source aspect ratio.

    Keeps the total pixel area close to `base_size * base_size` (e.g. 1024x1024
    for Flux.2/Qwen/Fibo/Krea, or 2048x2048 for MiniMax H3 Image) while snapping
    both dimensions to `multiple` so the generator never squishes the 3D render.
    """
    if source_w <= 0 or source_h <= 0:
        return base_size, base_size
    ratio = source_w / source_h
    target_area = float(base_size * base_size)
    raw_h = math.sqrt(target_area / ratio)
    raw_w = raw_h * ratio

    scale_cap = max_side / max(raw_w, raw_h)
    if scale_cap < 1.0:
        raw_w *= scale_cap
        raw_h *= scale_cap

    gen_w = max(multiple, min(max_side, int(round(raw_w / multiple)) * multiple))
    gen_h = max(multiple, min(max_side, int(round(raw_h / multiple)) * multiple))
    return gen_w, gen_h


def prepare_reference_image(
    img: Image.Image,
    *,
    bg_color: tuple[int, int, int] = (128, 128, 128),
    keep_rgba: bool = False,
) -> Image.Image:
    """Prepare the raw 3D render PNG as a clean conditioning reference.

    Transparent PNGs often store (0, 0, 0) in RGB where A == 0. When passed to
    an RGB vision encoder, raw stripping turns the background pitch black and
    bleeds dark fringes into the 3D edges. Compositing onto a clean neutral
    studio backdrop preserves exact shading and bevel cues for the encoder;
    the original alpha mask is re-applied after generation.
    """
    if keep_rgba:
        return img.convert("RGBA")
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, (*bg_color, 255))
        return Image.alpha_composite(backdrop, rgba).convert("RGB")
    return img.convert("RGB")


def build_texture_prompt(
    user_prompt: str,
    *,
    model_family: str = "",
    lock_silhouette: bool = True,
) -> str:
    """Build the full reference-guided 3D texturing & style transfer prompt."""
    clean = (user_prompt or "").strip()
    if not clean:
        clean = STYLE_PRESETS[0].prompt

    if lock_silhouette:
        body = (
            "The reference image is an untextured 3D grey clay render with "
            "ambient occlusion and directional lighting. Apply realistic "
            f"surface textures to this 3D mesh as: {clean}. "
            "Maintain a strictly uniform real-world material scale and "
            "consistent texel density across the entire object from top to "
            "bottom. Respect the 3D surface normals and relief: keep flat "
            "carved surfaces smooth and unobstructed, and texture steep "
            "embankments as continuous surfaces without adding out-of-scale "
            "3D objects, miniature trees, giant macro debris, or extra "
            "boulders. Strictly preserve the exact 3D geometry, curves, "
            "camera angle, ambient occlusion shadows, lighting direction, "
            "proportions, and outer silhouette of the reference object, "
            "complete edge-to-edge surface coverage inside the silhouette, "
            "isolated background"
        )
    else:
        body = clean

    if model_family == "minimax_h3_image" and "<Picture" not in body:
        body = f"In <Picture 1>, {body}"
    elif model_family.startswith("fibo2") and "<image_" not in body:
        body = f"Using <image_1> as the exact 3D structure reference, {body}"
    return body


def compute_target_dimensions(
    source_size: tuple[int, int],
    generated_size: tuple[int, int],
    *,
    preserve_proportions: bool = False,
) -> tuple[int, int]:
    """Compute final output dimensions `(final_w, final_h)`.

    - `preserve_proportions=False` (default): matches the exact source PNG pixel
      dimensions `(source_w, source_h)` for 1:1 3D compositing.
    - `preserve_proportions=True`: keeps the generator's native resolution scale
      adjusted to the exact mathematical aspect ratio (`source_w / source_h`) so
      nothing is stretched.
    """
    source_w, source_h = source_size
    gen_w, gen_h = generated_size
    if source_w <= 0 or source_h <= 0:
        return max(1, gen_w), max(1, gen_h)

    if not preserve_proportions:
        return source_w, source_h

    target_ratio = source_w / source_h
    gen_ratio = gen_w / gen_h if gen_h > 0 else target_ratio
    if gen_ratio >= target_ratio:
        final_h = max(1, gen_h)
        final_w = max(1, int(round(final_h * target_ratio)))
    else:
        final_w = max(1, gen_w)
        final_h = max(1, int(round(final_w / target_ratio)))
    return final_w, final_h


def composite_and_transfer_alpha(
    generated: Image.Image,
    source: Image.Image,
    *,
    preserve_alpha: bool = True,
    preserve_proportions: bool = False,
    alpha_feather: int = 0,
) -> Image.Image:
    """Transfer the exact alpha mask and proportions from `source` onto `generated`.

    Direct Python/Pillow equivalent of `compositeAndTransferTransparency`:
    - Resizes `generated` to `(final_w, final_h)` using Lanczos resampling.
    - When `preserve_alpha=True` and `source` has an alpha channel, resizes the
      source alpha mask to `(final_w, final_h)` and transfers `origA` pixel-for-pixel:
      * `origA == 0` -> `(0, 0, 0, 0)` (100% transparent, zero color fringe)
      * `0 < origA < 255` -> preserves exact anti-aliased edge transparency
      * `origA == 255` -> 100% opaque textured interior
    """
    final_w, final_h = compute_target_dimensions(
        source.size,
        generated.size,
        preserve_proportions=preserve_proportions,
    )
    gen_rgba = generated.convert("RGBA")
    if gen_rgba.size != (final_w, final_h):
        gen_rgba = gen_rgba.resize((final_w, final_h), Image.Resampling.LANCZOS)

    has_source_alpha = (
        source.mode in ("RGBA", "LA") or "transparency" in source.info
    )
    if not (preserve_alpha and has_source_alpha):
        return gen_rgba

    src_alpha = source.convert("RGBA").getchannel("A")
    if src_alpha.size != (final_w, final_h):
        src_alpha = src_alpha.resize((final_w, final_h), Image.Resampling.LANCZOS)

    if alpha_feather > 0:
        src_alpha = src_alpha.filter(
            ImageFilter.GaussianBlur(radius=float(alpha_feather) / 2.0)
        )

    # Zero out RGB channels where alpha == 0 so transparent background pixels
    # are strictly (0, 0, 0, 0) just like the reference canvas implementation.
    nonzero_mask = src_alpha.point(lambda a: 255 if a > 0 else 0)
    r, g, b, _ = gen_rgba.split()
    r = ImageChops.multiply(r, nonzero_mask)
    g = ImageChops.multiply(g, nonzero_mask)
    b = ImageChops.multiply(b, nonzero_mask)
    return Image.merge("RGBA", (r, g, b, src_alpha))


def prepare_comparison_source(
    source: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Resize the source PNG to match `target_size` for the Before/After comparator."""
    src_rgba = source.convert("RGBA")
    if src_rgba.size != target_size:
        src_rgba = src_rgba.resize(target_size, Image.Resampling.LANCZOS)
    return src_rgba
