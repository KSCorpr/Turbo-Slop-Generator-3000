"""Onglet Upscale créatif (façon Magnific) : SDXL + ControlNet Tile (diffusers),
raffinage par tuiles cohérentes. Outil PyTorch installable en 1 clic.
"""
from __future__ import annotations

import gradio as gr

from ..engine import tools


def build_creative_tab(tab_id="creative"):
    """Construit l'onglet Upscale créatif. Renvoie le composant Image d'entrée
    (pour que l'onglet de génération puisse y envoyer une image)."""
    with gr.Tab("✨ Upscale créatif", id=tab_id):
        gr.Markdown(
            "### Upscale créatif (façon Magnific / Topaz Wonder)\n"
            "**SDXL + ControlNet Tile** : ré-invente le détail par tuiles, "
            "ancrées sur l'image agrandie → tuiles cohérentes, **fondu sans "
            "couture**. Le curseur *Créativité* dose le détail ajouté.\n\n"
            "> ⚠️ Chaque tuile = une passe SDXL → c'est **lent** (le modèle reste "
            "sur le GPU si la VRAM le permet, sinon offload CPU). **×2 conseillé** ; "
            "×4/×8 = beaucoup de tuiles. Sortie plafonnée à 4096 px.")

        with gr.Accordion("⚙️ Installer l'upscale créatif (en 1 clic)",
                          open=not tools.upscale_is_installed()):
            gr.Markdown(
                "Repose sur PyTorch + diffusers et télécharge **~9 Go** de "
                "modèles (SDXL base + ControlNet Tile + VAE). Aucune commande à "
                "taper. Le premier téléchargement est long.")
            install_log = gr.Textbox(label="Journal d'installation", lines=10,
                                     autoscroll=True, elem_classes="log-box")
            install_btn = gr.Button("⬇️ Installer (SDXL + ControlNet Tile)")

            def _install():
                for msg in tools.install_upscale_stream():
                    yield msg

            install_btn.click(_install, outputs=[install_log])

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                scale = gr.Radio([2, 4, 8], value=2,
                                 label="Facteur (×8 = très long, beaucoup de tuiles)")
                creativity = gr.Slider(0.15, 0.7, value=0.4, step=0.05,
                                       label="Créativité (détail inventé — ↑ = plus)")
                cn_scale = gr.Slider(0.2, 1.0, value=0.5, step=0.05,
                                     label="Fidélité à l'original (↓ = plus de détail inventé)")
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                gr.Markdown(
                    "<small>Pas assez de détail / trop flou ? **Baissez la "
                    "fidélité** (≈0,4) et **montez la créativité** (≈0,5). Trop de "
                    "dérive ? L'inverse.</small>")
                with gr.Row():
                    run = gr.Button("✨ Upscaler", variant="primary",
                                    size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
            with gr.Column(scale=4):
                result = gr.Image(label="Aperçu (résolution complète dans outputs/)",
                                  height=620, format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                    elem_classes="log-box")

        def do_creative(image, scale, creativity, cn_scale, prompt,
                        progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            logs: list[str] = []
            progress(0.05, desc="Upscale créatif…")
            try:
                out = tools.creative_upscale(
                    image, scale=int(scale), prompt=prompt or "",
                    creativity=float(creativity), cn_scale=float(cn_scale),
                    log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            # Aperçu RÉDUIT pour l'affichage : servir une image énorme via Gradio
            # déclenche un bug HTTP (Content-Length). La pleine résolution est
            # enregistrée dans outputs/.
            from PIL import Image as _PILImage
            try:
                im = _PILImage.open(out)
                logs.append(f"\n✅ Image pleine résolution ({im.width}x{im.height}) "
                            f"enregistrée : {out}")
                disp = im
                if max(im.size) > 1600:
                    r = 1600 / max(im.size)
                    disp = im.resize((max(1, int(im.width * r)),
                                      max(1, int(im.height * r))))
                return disp, "\n".join(logs)
            except Exception:  # noqa: BLE001
                return str(out), "\n".join(logs)

        evt = run.click(do_creative,
                        inputs=[image, scale, creativity, cn_scale, prompt],
                        outputs=[result, logbox])
        stop.click(lambda: tools.cancel(), outputs=None, cancels=[evt])

    return image
