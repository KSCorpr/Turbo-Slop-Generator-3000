"""Onglet Upscale : PiD (rapide, GPU natif via sd.cpp) ou SDXL créatif (tuilé)."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry
from ..engine import generate as gen_engine
from ..engine import tools


def build_creative_tab(tab_id="creative"):
    """Construit l'onglet Upscale. Renvoie le composant Image d'entrée (pour que
    l'onglet de génération puisse y envoyer une image)."""
    with gr.Tab("✨ Upscale", id=tab_id):
        gr.Markdown(
            "### Upscale\n"
            "**⚡ PiD** : décodeur NVIDIA en espace pixel, **natif sd.cpp** → "
            "rapide, **100% GPU**, agrandit vers ~2K en 4 pas (pas de PyTorch).  \n"
            "**🎨 Créatif (SDXL)** : ré-invente le détail par tuiles (façon "
            "Magnific) — plus lent, plus « créatif ».")

        method = gr.Radio(
            [("⚡ Rapide — PiD (GPU natif, ~2K)", "pid"),
             ("🎨 Créatif — SDXL ControlNet Tile (lent)", "sdxl")],
            value="pid", label="Méthode")

        with gr.Accordion("⚙️ Installer PiD (en 1 clic)",
                          open=not registry.pid_ready()):
            gr.Markdown(
                "Via sd.cpp (GPU natif, **pas de PyTorch**). Télécharge le "
                "décodeur PiD + l'encodeur Gemma-2-2B + la VAE FLUX.1 (plusieurs "
                "Go). Aucune commande à taper.")
            pid_log = gr.Textbox(label="Journal d'installation", lines=8,
                                 autoscroll=True, elem_classes="log-box")
            pid_btn = gr.Button("⬇️ Installer PiD")

            def _install_pid():
                lines: list[str] = []
                for msg in downloader.download_pid(log=lines.append):
                    lines.append(msg)
                    yield "\n".join(lines)

            pid_btn.click(_install_pid, outputs=[pid_log])

        with gr.Accordion("⚙️ Installer l'upscale créatif SDXL (en 1 clic)",
                          open=False):
            gr.Markdown(
                "Repose sur PyTorch + diffusers (~9 Go : SDXL + ControlNet Tile + "
                "VAE). Aucune commande à taper. Le premier téléchargement est long.")
            sdxl_log = gr.Textbox(label="Journal d'installation", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            sdxl_btn = gr.Button("⬇️ Installer (SDXL + ControlNet Tile)")

            def _install_sdxl():
                for msg in tools.install_upscale_stream():
                    yield msg

            sdxl_btn.click(_install_sdxl, outputs=[sdxl_log])

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                with gr.Group(visible=False) as sdxl_opts:
                    scale = gr.Radio([2, 4, 8], value=2,
                                     label="Facteur (×8 = très long)")
                    creativity = gr.Slider(0.15, 0.7, value=0.4, step=0.05,
                                           label="Créativité (détail inventé — ↑ = plus)")
                    cn_scale = gr.Slider(0.2, 1.0, value=0.5, step=0.05,
                                         label="Fidélité (↓ = plus de détail inventé)")
                with gr.Row():
                    run = gr.Button("✨ Upscaler", variant="primary",
                                    size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
            with gr.Column(scale=4):
                result = gr.Image(label="Aperçu (résolution complète dans outputs/)",
                                  height=620, format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                    elem_classes="log-box")

        def _toggle(m):
            return gr.update(visible=(m == "sdxl"))

        method.change(_toggle, inputs=[method], outputs=[sdxl_opts])

        def do_upscale(method, image, prompt, scale, creativity, cn_scale,
                       progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            logs: list[str] = []
            try:
                if method == "pid":
                    progress(0.1, desc="PiD (GPU)…")
                    out = gen_engine.pid_upscale(image, prompt=prompt or "",
                                                 log=logs.append)
                else:
                    progress(0.05, desc="Upscale créatif SDXL…")
                    out = tools.creative_upscale(
                        image, scale=int(scale), prompt=prompt or "",
                        creativity=float(creativity), cn_scale=float(cn_scale),
                        log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            # Aperçu réduit (servir une image énorme via Gradio plante) ; la pleine
            # résolution est dans outputs/.
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

        evt = run.click(
            do_upscale,
            inputs=[method, image, prompt, scale, creativity, cn_scale],
            outputs=[result, logbox])

        def _cancel():
            tools.cancel()        # sous-process SDXL
            gen_engine.cancel()   # process sd.cpp (PiD)

        stop.click(_cancel, outputs=None, cancels=[evt])

    return image
