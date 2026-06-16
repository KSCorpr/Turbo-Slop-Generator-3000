#!/usr/bin/env python3
"""Atelier — studio d'inférence d'images en local (Gradio).

Onglets : Génération (Ideogram 4 / Z-Image Turbo, GGUF) · Bibliothèque ·
Upscale (SeedVR2 / NVIDIA PiD) · Réglages.
"""
from __future__ import annotations

import argparse
import os
import sys

# Le Python portable n'ajoute pas le dossier projet au chemin d'import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr


def _patch_gradio_client() -> None:
    """Contourne un bug de gradio_client sur les schémas booléens (au démarrage)."""
    try:
        import gradio_client.utils as gcu
        _orig = gcu._json_schema_to_python_type

        def _safe(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig(schema, defs)

        gcu._json_schema_to_python_type = _safe
    except Exception:  # noqa: BLE001
        pass


_patch_gradio_client()

from atelier import APP_NAME, __version__, hardware, settings
from atelier.ui.generate_tab import build_generate_tab
from atelier.ui.library_tab import build_library_tab
from atelier.ui.settings_tab import build_settings_tab
from atelier.ui.theme import CSS, theme
from atelier.ui.upscale_tab import build_upscale_tab


def build_app() -> gr.Blocks:
    settings.ensure_dirs()
    gpus = hardware.detect_gpus()
    sd_cli = settings.find_sd_cli()

    with gr.Blocks(title=f"{APP_NAME} {__version__}", theme=theme(), css=CSS) as demo:
        gr.HTML(
            f"<div id='atelier-header'><h1>🎨 {APP_NAME}</h1>"
            f"<div class='sub'>Génération d'images locale · Ideogram 4 · "
            f"Z-Image Turbo · GGUF · v{__version__}</div></div>")

        if sd_cli is None:
            gr.Markdown("> ⚠️ **Binaire `sd-cli` introuvable.** Lancez "
                        "`install.bat` / `install.sh`, ou "
                        "`python scripts/get_sdcpp.py`.")
        if not gpus:
            gr.Markdown("> ⚠️ **Aucun GPU NVIDIA détecté** (mode CPU très lent). "
                        "Vérifiez vos pilotes / `nvidia-smi`.")

        with gr.Tabs():
            build_generate_tab()
            build_library_tab()
            build_upscale_tab()
            build_settings_tab()

    return demo


def main():
    ap = argparse.ArgumentParser(description=f"{APP_NAME} {__version__}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port,
                        share=args.share, inbrowser=True, show_api=False)


if __name__ == "__main__":
    main()
