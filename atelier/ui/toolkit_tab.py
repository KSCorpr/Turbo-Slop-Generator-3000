"""Onglet Toolkit : profondeur et suppression d'arrière-plan."""
from __future__ import annotations

import gradio as gr

from ..engine import tools


def _installer_block(title: str, note: str, stream_fn, installed: bool):
    """Accordéon d'installation 1 clic commun aux outils."""
    with gr.Accordion(f"⚙️ Installer {title} (en 1 clic)", open=not installed):
        gr.Markdown(note)
        log = gr.Textbox(label="Journal d'installation", lines=10,
                         autoscroll=True, elem_classes="log-box")
        btn = gr.Button(f"⬇️ Installer {title}")

        def _install():
            for msg in stream_fn():
                yield msg

        btn.click(_install, outputs=[log])


def build_toolkit_tab(tab_id="toolkit"):
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Outils utilitaires\n"
            "Carte de **profondeur**, **suppression d'arrière-plan** (PNG "
            "transparent) et **détourage d'objet au clic** (Segment Anything).")

        with gr.Tabs():
            # ---------- Profondeur ----------
            with gr.Tab("🌐 Profondeur"):
                gr.Markdown(
                    "*Depth Anything V2* — carte de profondeur (clair = proche, "
                    "sombre = loin). Téléchargez le résultat pour le réutiliser.")
                _installer_block(
                    "Depth Anything V2",
                    "Repose sur PyTorch + transformers (~100 Mo de modèle). "
                    "Aucune commande à taper.",
                    tools.install_depth_stream, tools.depth_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        d_image = gr.Image(label="Image source", type="pil")
                        d_run = gr.Button("🌐 Générer la profondeur",
                                          variant="primary", size="lg")
                    with gr.Column(scale=4):
                        d_result = gr.Image(label="Carte de profondeur", height=520,
                                            format="png", show_download_button=True)
                        d_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_depth(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    logs: list[str] = []
                    progress(0.1, desc="Profondeur…")
                    try:
                        out = tools.depth_map(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                d_run.click(do_depth, inputs=[d_image], outputs=[d_result, d_log])

            # ---------- Suppression d'arrière-plan ----------
            with gr.Tab("✂️ Sans arrière-plan"):
                gr.Markdown(
                    "*RMBG-1.4* — détoure le sujet et renvoie un **PNG "
                    "transparent**.  \n"
                    "⚠️ Modèle sous licence **non commerciale** (BRIA RMBG-1.4).")
                _installer_block(
                    "RMBG-1.4",
                    "Repose sur PyTorch + transformers (~176 Mo de modèle). "
                    "Aucune commande à taper.",
                    tools.install_bg_stream, tools.bg_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        b_image = gr.Image(label="Image source", type="pil")
                        b_run = gr.Button("✂️ Détourer", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        b_result = gr.Image(label="Sujet détouré (PNG transparent)",
                                            height=520, format="png",
                                            show_download_button=True,
                                            image_mode="RGBA")
                        b_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_bg(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    logs: list[str] = []
                    progress(0.1, desc="Détourage…")
                    try:
                        out = tools.bg_remove(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                b_run.click(do_bg, inputs=[b_image], outputs=[b_result, b_log])

            # ---------- Segment Anything (clic) ----------
            with gr.Tab("🪄 Détourer un objet (SAM)"):
                gr.Markdown(
                    "*Segment Anything* — **cliquez sur un objet** dans l'image "
                    "puis « Extraire » : SAM le détoure en **PNG transparent**.")
                _installer_block(
                    "Segment Anything",
                    "PyTorch + transformers (~375 Mo, facebook/sam-vit-base). "
                    "Aucune commande à taper.",
                    tools.install_sam_stream, tools.sam_is_installed())

                s_point = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        s_image = gr.Image(label="Image — cliquez sur l'objet",
                                           type="pil")
                        s_info = gr.Markdown("Cliquez un point sur l'image.")
                        s_run = gr.Button("🪄 Extraire l'objet", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        s_result = gr.Image(label="Objet extrait (PNG transparent)",
                                            height=520, format="png",
                                            show_download_button=True,
                                            image_mode="RGBA")
                        s_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def _on_click(evt: gr.SelectData):
                    x, y = int(evt.index[0]), int(evt.index[1])
                    return (x, y), f"Point : ({x}, {y}). Cliquez « Extraire l'objet »."

                s_image.select(_on_click, outputs=[s_point, s_info])

                def do_sam(img, point, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    if not point:
                        raise gr.Error("Cliquez d'abord sur un objet dans l'image.")
                    logs: list[str] = []
                    progress(0.1, desc="Segmentation…")
                    try:
                        out = tools.sam_segment(img, point[0], point[1],
                                                log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                s_run.click(do_sam, inputs=[s_image, s_point],
                            outputs=[s_result, s_log])
