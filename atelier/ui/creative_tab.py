"""Onglet Upscale créatif (façon Magnific) : pré-agrandissement + raffinage
img2img via un modèle de diffusion. Réutilise le moteur de génération.
"""
from __future__ import annotations

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine


def _model_choices() -> list[tuple[str, str]]:
    prefs = settings.load_prefs()
    out = []
    for m in registry.load_base_models(prefs):
        ready = registry.model_is_ready(m)
        out.append((f"{m.name} {'●' if ready else '○ (à télécharger)'}", m.id))
    return out


def build_creative_tab():
    choices = _model_choices()
    # Z-Image Turbo par défaut (rapide et efficace pour le raffinage).
    default = next((v for _, v in choices if v == "zimage-turbo"),
                   choices[0][1] if choices else None)

    with gr.Tab("✨ Upscale créatif"):
        gr.Markdown(
            "### Upscale créatif (façon Magnific / Topaz Wonder)\n"
            "Pré-agrandit l'image, puis **ré-invente le détail** via un modèle de "
            "diffusion (img2img **par tuiles**, recollées avec fondu). Le curseur "
            "*Créativité* dose le détail ajouté.\n\n"
            "> ⚠️ Le modèle se recharge à chaque tuile : **×2 = 4 tuiles** (rapide), "
            "**×4 = 16 tuiles** (long). **Z-Image Turbo** est le plus rapide pour ça.")
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                model = gr.Dropdown(choices=choices, value=default,
                                    label="Modèle de raffinage")
                scale = gr.Radio([2, 4], value=2, label="Facteur")
                creativity = gr.Slider(0.1, 0.6, value=0.3, step=0.05,
                                       label="Créativité (détail ajouté / dérive)")
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                run = gr.Button("✨ Upscaler", variant="primary", size="lg")
            with gr.Column(scale=4):
                result = gr.Image(label="Résultat", height=620, format="png",
                                  show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                    elem_classes="log-box")

        def do_creative(image, model_id, scale, creativity, prompt,
                        progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            if not model_id:
                raise gr.Error("Choisissez un modèle de raffinage.")
            m = registry.get_base_model(model_id, settings.load_prefs())
            if m is None or not registry.model_is_ready(m):
                raise gr.Error("Ce modèle n'est pas téléchargé (onglet Bibliothèque).")
            logs: list[str] = []
            progress(0.05, desc="Upscale créatif…")
            try:
                out = gen_engine.creative_upscale(
                    model_id=model_id, image=image, scale=int(scale),
                    prompt=prompt or "", creativity=float(creativity),
                    log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            return str(out), "\n".join(logs)

        run.click(do_creative, inputs=[image, model, scale, creativity, prompt],
                  outputs=[result, logbox])
