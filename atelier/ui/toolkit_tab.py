"""Onglet Toolkit : profondeur, détourage, SAM et upscale ESRGAN."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry
from ..engine import generate as gen_engine
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
            "transparent), **détourage d'objet au clic** (Segment Anything) et "
            "**agrandissement ESRGAN** (simple, 100% GPU).")

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

            # ---------- Agrandir (ESRGAN, sd.cpp) ----------
            with gr.Tab("🔼 Agrandir (ESRGAN)"):
                gr.Markdown(
                    "Agrandissement **simple** par réseau ESRGAN GGUF, natif "
                    "**sd.cpp** : déterministe, **100% GPU**, aucun PyTorch ni "
                    "prompt. Le facteur (×2 ou ×4) dépend du modèle choisi ; "
                    "« Répéter » ré-applique le modèle (×2 deux fois = ×4).")

                with gr.Accordion("⬇️ Télécharger les upscalers (en 1 clic)",
                                  open=not registry.upscalers_ready()):
                    gr.Markdown(
                        "Récupère **tous** les modèles ESRGAN GGUF (~1 Go au "
                        "total) depuis `wbruna/upscalers-sdcpp-gguf`. "
                        "Réutilisables ensuite hors-ligne.")
                    u_inst_log = gr.Textbox(label="Journal de téléchargement",
                                            lines=8, autoscroll=True,
                                            elem_classes="log-box")
                    u_inst = gr.Button("⬇️ Télécharger les upscalers")

                with gr.Row():
                    with gr.Column(scale=3):
                        u_image = gr.Image(label="Image à agrandir", type="pil")
                        u_model = gr.Dropdown(
                            registry.list_upscalers(),
                            value=(registry.list_upscalers() or [None])[0],
                            label="Modèle d'upscale (×2 / ×4 selon le nom)")
                        u_repeats = gr.Radio([("×1 (natif)", 1), ("Répéter ×2", 2)],
                                             value=1, label="Répétition")
                        with gr.Row():
                            u_refresh = gr.Button("↻ Rafraîchir la liste", size="sm")
                            u_run = gr.Button("🔼 Agrandir", variant="primary",
                                              size="lg", scale=2)
                        u_stop = gr.Button("⏹️ Annuler", variant="stop", size="sm")
                    with gr.Column(scale=4):
                        u_result = gr.Image(
                            label="Résultat (pleine résolution dans outputs/)",
                            height=520, format="png", show_download_button=True)
                        u_log = gr.Textbox(label="Journal", lines=10,
                                           autoscroll=True, elem_classes="log-box")

                def _install_upscalers():
                    lines: list[str] = []
                    for msg in downloader.download_upscalers(log=lines.append):
                        lines.append(msg)
                        yield "\n".join(lines), gr.update()
                    yield ("\n".join(lines),
                           gr.update(choices=registry.list_upscalers(),
                                     value=(registry.list_upscalers() or [None])[0]))

                u_inst.click(_install_upscalers, outputs=[u_inst_log, u_model])

                def _refresh_upscalers():
                    ups = registry.list_upscalers()
                    return gr.update(choices=ups, value=(ups or [None])[0])

                u_refresh.click(_refresh_upscalers, outputs=[u_model])

                def do_upscale(img, model, repeats, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    if not model:
                        raise gr.Error("Choisissez un modèle d'upscale "
                                       "(téléchargez-les d'abord).")
                    logs: list[str] = []
                    progress(0.1, desc="Agrandissement…")
                    try:
                        out = gen_engine.upscale_image(
                            img, model, repeats=int(repeats), log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    logs.append(f"\n✅ Image agrandie : {out}")
                    return str(out), "\n".join(logs)

                u_evt = u_run.click(do_upscale,
                                    inputs=[u_image, u_model, u_repeats],
                                    outputs=[u_result, u_log])
                u_stop.click(lambda: gen_engine.cancel(), outputs=None,
                             cancels=[u_evt])
