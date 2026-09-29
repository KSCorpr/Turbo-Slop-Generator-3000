"""Offline studio theme: legible type, quiet chrome, a single warm accent."""
from __future__ import annotations

import gradio as gr

ACCENT = "#b74726"
ACCENT_HOVER = "#963a20"
ACCENT_DARK = "#a33f21"
_FONTS = ["Segoe UI Variable Text", "Segoe UI", "-apple-system", "BlinkMacSystemFont",
          "Inter", "Helvetica Neue", "system-ui", "sans-serif"]
_MONO = ["ui-monospace", "Cascadia Mono", "SFMono-Regular", "Consolas", "monospace"]
_ACCENT_RAMP = gr.themes.Color(
    name="studio-copper", c50="#fff5ef", c100="#ffe7d7", c200="#ffcaab",
    c300="#ffa577", c400="#f87b46", c500="#df5e30", c600="#b74726",
    c700="#963a20", c800="#79311f", c900="#632b1d", c950="#35140b")


def theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=_ACCENT_RAMP, secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray, radius_size=gr.themes.sizes.radius_md,
        font=_FONTS, font_mono=_MONO,
    ).set(
        body_background_fill="#f4f3ef", body_background_fill_dark="#141617",
        block_background_fill="#ffffff", block_background_fill_dark="#1d2022",
        block_border_width="1px", block_border_color="#deded8",
        block_border_color_dark="#35393c", block_shadow="none",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color="#565b5e", block_label_text_color_dark="#b6bdc2",
        block_label_text_weight="500", block_label_text_size="0.8rem",
        block_title_text_color="#343a3d", block_title_text_color_dark="#d8dddf",
        block_title_text_weight="600", input_background_fill="#fafaf8",
        input_background_fill_dark="#181b1d", input_border_color="#d9dcd8",
        input_border_color_dark="#3c4245",
        button_primary_background_fill=ACCENT, button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_background_fill_dark=ACCENT, button_primary_background_fill_hover_dark=ACCENT_HOVER,
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_large_radius="8px", button_small_radius="6px", slider_color=ACCENT,
    )


# The shortcut runs locally and only targets the visible generation panel.
# No polling loop, MutationObserver or server request on each keystroke.
SHORTCUTS = """
<style>
/* Gradio scopes its CSS argument inside .contain. The outer application
   spacing must therefore live in head to avoid two stacked padding layers. */
gradio-app .gradio-container { padding: clamp(12px, 2vw, 28px) !important; }
gradio-app .gradio-container > .main { padding: 0 !important; }
</style>
<script>
(() => {
  if (window.__turboStudioKeys) return;
  window.__turboStudioKeys = true;
  document.addEventListener('keydown', event => {
    if (!(event.ctrlKey || event.metaKey) || event.key !== 'Enter' || event.repeat) return;
    const target = event.target;
    if (!(target instanceof Element) || !target.closest('.studio-prompt')) return;
    const workspace = target.closest('.generation-workspace');
    const button = workspace && workspace.querySelector('.go-row button.primary');
    if (button && !button.disabled && button.getClientRects().length) {
      event.preventDefault();
      event.stopImmediatePropagation();
      button.click();
    }
  }, true);
})();
</script>
"""

THEME_JS = """(mode) => {
    document.documentElement.classList.toggle('dark', mode === 'dark');
    document.body.classList.toggle('dark', mode === 'dark');
    const url = new URL(window.location.href);
    url.searchParams.set('__theme', mode);
    window.history.replaceState({}, '', url);
    return mode;
}"""

CSS = """
:root {
  --tsg-accent: #b74726; --tsg-accent-dark: #a33f21;
  --tsg-ink: #23292c; --tsg-muted: #596367; --tsg-line: #deded8;
  --tsg-surface: #fff; --tsg-ground: #efefea; --tsg-raise: none;
}
.dark {
  --tsg-accent: #ef926d; --tsg-accent-dark: #ef926d;
  --tsg-ink: #edf0ee; --tsg-muted: #b0b9bd; --tsg-line: #35393c;
  --tsg-surface: #1d2022; --tsg-ground: #181b1d;
}
.gradio-container { width:100% !important; padding:clamp(12px,2vw,28px) !important; max-width: 1680px !important; margin: 0 auto !important; }
#atelier-header { display:flex; justify-content:space-between; align-items:center;
  gap:20px; padding:12px 0 22px; border-bottom:1px solid var(--tsg-line); }
.studio-brand { display:flex; align-items:center; gap:13px; }
.studio-mark { display:grid; place-items:center; width:42px; height:42px;
  background:#b74726; color:white; border-radius:8px; font-size:16px;
  font-weight:800; letter-spacing:-.08em; }
#atelier-header h1 { margin:0; font-size:19px; font-weight:800;
  letter-spacing:-.035em; line-height:1.25; color:var(--tsg-ink); }
#atelier-header h1 span { font-weight:400; color:var(--tsg-muted); }
#atelier-header .sub { margin:3px 0 0; font-size:11px; color:var(--tsg-muted); }
.studio-hardware { display:flex; flex-direction:column; align-items:flex-end; gap:5px; }
.local-label { font-size:9px; letter-spacing:.16em; color:var(--tsg-muted); font-weight:600; }
#atelier-header .chip { font-size:11px; padding:3px 8px; border:1px solid var(--tsg-line);
  border-radius:5px; color:var(--tsg-muted); }
#atelier-header .chip.ok::before { content:''; display:inline-block; background:#33a078;
  width:6px; height:6px; border-radius:50%; margin-right:7px; }
.tab-container { border-bottom:1px solid var(--tsg-line) !important; gap:3px !important; }
button[role=tab] { font-size:13px !important; font-weight:500 !important;
  padding:11px 15px !important; color:var(--tsg-muted) !important;
  border:0 !important; border-bottom:2px solid transparent !important;
  background:transparent !important; border-radius:0 !important;
  transition:color 120ms ease,background 120ms ease; }
button[role=tab]:hover { color:var(--tsg-ink) !important; background:var(--tsg-ground) !important; }
button[role=tab][aria-selected=true] { color:var(--tsg-accent-dark) !important;
  font-weight:650 !important; border-bottom-color:var(--tsg-accent) !important; }
#studio-nav > .tab-wrapper > [role=tablist] button[role=tab] { font-size:14px !important; padding:14px 23px !important; }
#model-nav > .tab-wrapper > [role=tablist] { background:var(--tsg-ground); border:0 !important;
  border-radius:8px; padding:4px !important; margin:6px 0 14px !important; }
#model-nav > .tab-wrapper > [role=tablist] button[role=tab] { padding:7px 14px !important;
  border:0 !important; border-radius:5px !important; font-size:12px !important; }
#model-nav > .tab-wrapper > [role=tablist] button[aria-selected=true] { color:var(--tsg-ink) !important;
  background:var(--tsg-surface) !important; box-shadow:0 1px 3px #00000014; }
.model-summary { padding:2px 0 12px !important; color:var(--tsg-muted); font-size:12px !important; }
.model-summary p { margin:0 !important; }
.studio-controls { gap:12px !important; }
.studio-prompt textarea { font-size:15px !important; line-height:1.65 !important; min-height:116px; }
.studio-canvas { gap:12px !important; }
.studio-status { min-height:26px; color:var(--tsg-muted); font-size:12px !important; }
.studio-status p { margin:0 !important; }
.go-row { gap:8px !important; }
.go-row button { min-height:44px; font-weight:600 !important; }
.go-row button.primary { font-size:14px !important; box-shadow:none !important; }
.block-label, .block-title, .gradio-container .block > .label-wrap > span {
  background:transparent !important; border:0 !important; box-shadow:none !important;
  font-weight:500 !important; font-size:12px !important; color:var(--tsg-muted) !important; }
.block-label { backdrop-filter:none !important; }
.model-card { border:1px solid var(--tsg-line); border-radius:8px;
  padding:14px 16px; background:var(--tsg-surface); margin-bottom:10px; }
.model-card h3 { margin:0 0 5px; font-size:16px; }
.tag { display:inline-block; background:var(--tsg-ground); color:var(--tsg-muted);
  border:1px solid var(--tsg-line); border-radius:4px; padding:2px 7px;
  font-size:10px; font-weight:500; margin-right:4px; }
.status-ok { color:#247c60; font-weight:500; }
.status-missing { color:#a15721; font-weight:500; }
.dark .status-ok { color:#78c8a7; }
.dark .status-missing { color:#e6ae7b; }
.log-box textarea { font-family:ui-monospace,Consolas,monospace !important;
  font-size:11px !important; line-height:1.6; }
.hint p { margin:.2rem 0 !important; font-size:12px; color:var(--tsg-muted); line-height:1.5; }
.feedback:has(p) { border-left:2px solid var(--tsg-accent); background:var(--tsg-ground);
  padding:8px 12px; border-radius:0 6px 6px 0; }
.feedback:not(:has(p)) { display:none; }
.feedback p { margin:.15rem 0 !important; font-size:12px; }
#atelier-alerts { border:1px solid var(--tsg-line); border-radius:6px;
  padding:8px 12px; color:var(--tsg-muted); font-size:12px; background:var(--tsg-ground); }
#atelier-alerts p { margin:0 !important; }
[data-testid="image"] img, .image-frame img, .image-container img { object-fit:contain !important; max-height:70vh; }
textarea { resize:vertical !important; max-width:100% !important; }
footer { display:none !important; }
:where(button,input,textarea,select,[tabindex]):focus-visible {
  outline:2px solid var(--tsg-accent) !important; outline-offset:3px; }
@media (min-width:1100px) {
  .studio-canvas { position:sticky !important; top:16px; align-self:flex-start !important; }
}
@media (max-width:760px) {
  .gradio-container { padding:12px !important; }
  #atelier-header { padding:4px 0 16px; gap:10px; }
  #atelier-header h1 { font-size:16px; }
  .studio-hardware .local-label { display:none; }
  #atelier-header .chip { font-size:10px; max-width:140px; white-space:normal; }
  #studio-nav > .tab-wrapper > [role=tablist] button[role=tab] { padding:10px 11px !important; font-size:12px !important; }
  .studio-controls,.studio-canvas { min-width:min(100%,320px) !important; }
}
@media (prefers-reduced-motion:reduce) {
  *,*::before,*::after { transition:none !important; animation:none !important; scroll-behavior:auto !important; }
}
"""
