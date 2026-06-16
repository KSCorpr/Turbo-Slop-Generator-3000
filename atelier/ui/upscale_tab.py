"""Onglet Upscale : SeedVR2 ou NVIDIA PiD."""
from __future__ import annotations

import gradio as gr
from PIL import Image

from .. import registry
from ..engine import upscalers


def build_upscale_tab():
    ups = registry.load_upscalers()
    choices = []
    for u in ups:
        installed = upscalers.is_installed(u.id)
        mark = "●" if installed else "○ (à installer)"
        choices.append((f"{u.name} {mark}", u.id))

    with gr.Tab("🔍 Upscale"):
        gr.Markdown("### Agrandissement d'image\n"
                    "**SeedVR2** : restauration par diffusion, qualité maximale.  \n"
                    "**NVIDIA PiD** : décodeur pixel-diffusion, très rapide, 4×/8×.")
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=320)
                engine = gr.Radio(choices=choices,
                                  value=(choices[0][1] if choices else None),
                                  label="Moteur d'upscale")
                scale = gr.Radio([2, 4, 8], value=4, label="Facteur")
                run = gr.Button("🚀 Agrandir", variant="primary", size="lg")
            with gr.Column(scale=4):
                result = gr.Image(label="Résultat", height=520)
                logbox = gr.Textbox(label="Journal", lines=10, autoscroll=True,
                                    elem_classes="log-box")

        def do_upscale(image, engine_id, scale, progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            if not engine_id:
                raise gr.Error("Choisissez un moteur d'upscale.")
            logs: list[str] = []
            progress(0.05, desc=f"Upscale ({engine_id})…")
            try:
                out = upscalers.upscale(engine_id, image, scale=int(scale),
                                        log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            return Image.open(out), "\n".join(logs)

        run.click(do_upscale, inputs=[image, engine, scale],
                  outputs=[result, logbox])
