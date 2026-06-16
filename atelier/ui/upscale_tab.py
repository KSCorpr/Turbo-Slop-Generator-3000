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

        with gr.Accordion("⚙️ Installer les moteurs d'upscale (en 1 clic)", open=False):
            gr.Markdown(
                "Ces moteurs reposent sur PyTorch (volumineux). Cliquez pour "
                "installer — **aucune commande à taper**. Le téléchargement "
                "(code + PyTorch + poids) peut prendre un long moment.")
            install_log = gr.Textbox(label="Journal d'installation", lines=10,
                                     autoscroll=True, elem_classes="log-box")
            with gr.Row():
                for u in ups:
                    btn = gr.Button(f"⬇️ Installer {u.name}")

                    def make_installer(uid):
                        def _install():
                            for msg in upscalers.install_stream(uid):
                                yield msg
                        return _install

                    btn.click(make_installer(u.id), outputs=[install_log])
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=320)
                engine = gr.Radio(choices=choices,
                                  value=(choices[0][1] if choices else None),
                                  label="Moteur d'upscale")
                refresh = gr.Button("↻ Rafraîchir l'état des moteurs", size="sm")
                scale = gr.Radio([2, 4, 8], value=4, label="Facteur")
                run = gr.Button("🚀 Agrandir", variant="primary", size="lg")
            with gr.Column(scale=4):
                result = gr.Image(label="Résultat", height=520)
                logbox = gr.Textbox(label="Journal", lines=10, autoscroll=True,
                                    elem_classes="log-box")

        def _refresh_engines():
            ch = []
            for u in registry.load_upscalers():
                mark = "●" if upscalers.is_installed(u.id) else "○ (à installer)"
                ch.append((f"{u.name} {mark}", u.id))
            return gr.update(choices=ch, value=(ch[0][1] if ch else None))

        refresh.click(_refresh_engines, outputs=[engine])

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
