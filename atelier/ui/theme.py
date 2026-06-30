"""Thème et habillage CSS (clair, moderne) — accent cyan #00ccff."""
from __future__ import annotations

import gradio as gr

ACCENT = "#00ccff"
ACCENT_HOVER = "#00b3e6"
ACCENT_DARK = "#006688"

# Rampe de teintes centrée sur #00ccff (c500), pour les accents Gradio.
_ACCENT_RAMP = gr.themes.Color(
    name="cyan-accent",
    c50="#e6faff", c100="#c2f2ff", c200="#8ae8ff", c300="#4dddff",
    c400="#1ad4ff", c500="#00ccff", c600="#00a3cc", c700="#007a99",
    c800="#005c73", c900="#08384a", c950="#04222e",
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
        button_primary_text_color="#03303f",   # navy sombre = lisible sur le cyan
        slider_color=ACCENT,
    )


CSS = f"""
.gradio-container {{ max-width: 1750px !important; margin: auto; }}
#atelier-header {{ text-align: left; padding: 4px 0 2px 0; }}
#atelier-header h1 {{ font-size: 1.7rem; margin: 0; letter-spacing: .5px;
                      color: {ACCENT_DARK}; }}
#atelier-header .sub {{ color: #6b7280; font-size: .85rem; margin-top: 2px; }}

/* ---- Onglets : plus lisibles, état sélectionné net (accent cyan) ---- */
.tab-nav {{ border-bottom: 2px solid #e5e7eb !important; gap: 2px; }}
.tab-nav button {{ font-size: .98rem !important; font-weight: 600 !important;
                   padding: 9px 16px !important; color: #475569 !important;
                   border: none !important; border-radius: 8px 8px 0 0 !important; }}
.tab-nav button:hover {{ color: {ACCENT_DARK} !important;
                         background: {ACCENT}14 !important; }}
.tab-nav button.selected {{ color: {ACCENT_DARK} !important;
    background: {ACCENT}1f !important;
    border-bottom: 3px solid {ACCENT} !important; }}

.model-card {{ border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
               background:#ffffff; margin-bottom:10px;
               box-shadow:0 1px 2px rgba(16,24,40,.04); }}
.model-card h3 {{ margin:0 0 4px 0; }}
.tag {{ display:inline-block; background:{ACCENT}22; color:{ACCENT_DARK};
        border-radius:999px; padding:2px 10px; font-size:.72rem; margin-right:6px; }}
.status-ok {{ color:#16a34a; font-weight:600; }}
.status-missing {{ color:#d97706; font-weight:600; }}
.log-box textarea {{ font-family: ui-monospace, monospace; font-size:.8rem;
                     resize: vertical; }}

/* ---- Stabilité des dimensions (évite les sauts/collapse au resize) ---- */
/* Images (upload/preview) : l'image s'inscrit en entier, sans collapse de
   largeur sur les ratios non carrés, et sans déborder verticalement. */
[data-testid="image"] img, .image-frame img, .image-container img {{
    object-fit: contain !important; width: 100% !important;
    max-height: 70vh; }}
[data-testid="image"], .image-container {{ overflow: hidden; }}
/* Zones de texte : redimensionnables verticalement seulement (pas de
   débordement horizontal qui casse la mise en page). */
textarea {{ resize: vertical !important; max-width: 100% !important; }}
.gr-image, .gr-gallery {{ min-height: 0; }}
footer {{ display:none !important; }}
"""
