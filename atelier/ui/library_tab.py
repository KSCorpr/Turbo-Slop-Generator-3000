"""Onglet Bibliothèque : catalogue des modèles, recommandations selon le
matériel, téléchargement à la demande."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings


def _card_md(model: registry.BaseModel, recos: dict[str, list[str]]) -> str:
    ready = registry.model_is_ready(model)
    status = ("<span class='status-ok'>● installé</span>" if ready
              else "<span class='status-missing'>○ non installé</span>")
    tags = " ".join(f"<span class='tag'>{t}</span>" for t in model.tags)
    reco = " · ".join(recos.get(model.id, []))
    return (f"<div class='model-card'><h3>{model.name} &nbsp; {status}</h3>"
            f"{tags}<p>{model.description}</p>"
            f"<small>{reco}</small></div>")


def build_library_tab():
    with gr.Tab("📚 Bibliothèque"):
        gr.Markdown("### Modèles de base\n"
                    "Téléchargement à la demande. La quantification est choisie "
                    "automatiquement selon votre VRAM/RAM (modifiable dans Réglages).")

        prefs = settings.load_prefs()
        models = registry.load_base_models(prefs)
        recos = registry.recommend(prefs)

        cards: list[gr.Markdown] = []
        log = gr.Textbox(label="Journal des téléchargements", lines=8,
                         autoscroll=True, elem_classes="log-box")

        for m in models:
            with gr.Row():
                with gr.Column(scale=5):
                    card = gr.Markdown(_card_md(m, recos))
                with gr.Column(scale=1, min_width=160):
                    btn = gr.Button(f"⬇️ Télécharger", variant="primary")
            cards.append(card)

            def make_handler(model_id):
                def handler(progress=gr.Progress()):
                    p = settings.load_prefs()
                    model = registry.get_base_model(model_id, p)
                    lines: list[str] = []
                    for msg in downloader.download_model(model, log=lines.append):
                        lines.append(msg)
                        yield "\n".join(lines)
                return handler

            btn.click(make_handler(m.id), outputs=[log])

        refresh = gr.Button("↻ Rafraîchir l'état")

        def refresh_cards():
            p = settings.load_prefs()
            r = registry.recommend(p)
            ups = [gr.update(value=_card_md(m, r))
                   for m in registry.load_base_models(p)]
            # Avec une seule carte, Gradio attend une valeur unique (pas une
            # liste), sinon la liste est passée telle quelle au Markdown.
            return ups[0] if len(ups) == 1 else ups

        refresh.click(refresh_cards, outputs=cards)

        gr.Markdown(
            "---\n*Upscale créatif (par tuiles) et outils (profondeur, "
            "détourage) sont dans leurs onglets dédiés.*")
