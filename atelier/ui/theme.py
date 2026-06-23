"""Thème et habillage CSS (clair, moderne) — accent #add645."""
from __future__ import annotations

import gradio as gr

ACCENT = "#add645"
ACCENT_HOVER = "#9bc23a"
ACCENT_DARK = "#4a5d21"

# Rampe de teintes centrée sur #add645 (c500), pour les accents Gradio.
_ACCENT_RAMP = gr.themes.Color(
    name="federall",
    c50="#f7fbe8", c100="#eef7c9", c200="#e0ef9c", c300="#cde666",
    c400="#bcde52", c500="#add645", c600="#8fb52f", c700="#6f8c26",
    c800="#586e23", c900="#4a5d21", c950="#28340c",
)


def theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=_ACCENT_RAMP,
        secondary_hue=_ACCENT_RAMP,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f6f7fb",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_title_text_weight="600",
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_text_color="#1f2a10",   # texte sombre = lisible sur le lime
        slider_color=ACCENT,
    )


CSS = f"""
.gradio-container {{ max-width: 1750px !important; margin: auto; }}
#atelier-header {{ text-align: left; padding: 4px 0 2px 0; }}
#atelier-header h1 {{ font-size: 1.7rem; margin: 0; letter-spacing: .5px;
                      color: {ACCENT}; }}
#atelier-header .sub {{ color: #6b7280; font-size: .85rem; margin-top: 2px; }}
.model-card {{ border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
               background:#ffffff; margin-bottom:10px;
               box-shadow:0 1px 2px rgba(16,24,40,.04); }}
.model-card h3 {{ margin:0 0 4px 0; }}
.tag {{ display:inline-block; background:{ACCENT}22; color:{ACCENT_DARK};
        border-radius:999px; padding:2px 10px; font-size:.72rem; margin-right:6px; }}
.status-ok {{ color:#16a34a; font-weight:600; }}
.status-missing {{ color:#d97706; font-weight:600; }}
.log-box textarea {{ font-family: ui-monospace, monospace; font-size:.8rem; }}
/* Images (upload/preview) : éviter le collapse de largeur sur les ratios non
   carrés -> l'image s'inscrit en entier dans son cadre. */
[data-testid="image"] img, .image-frame img, .image-container img {{
    object-fit: contain !important; width: 100% !important;
    max-height: 65vh; }}
footer {{ display:none !important; }}
"""
