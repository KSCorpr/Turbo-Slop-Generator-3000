"""Onglet Vidéo : LTX-2.3 (Lightricks) via stable-diffusion.cpp (-M vid_gen).
Texte→vidéo, image→vidéo, et image début→fin. ⚠️ 22B : très lourd.
"""
from __future__ import annotations

import gradio as gr

from .. import downloader, i18n, registry, settings
from ..engine import generate as gen_engine
from ..i18n import t

RATIOS = {
    "Paysage 16:9 — 1280×720": (1280, 720),
    "Paysage 16:9 — 960×544": (960, 544),
    "Léger 16:9 — 640×360": (640, 360),
    "Carré — 768×768": (768, 768),
    "Portrait 9:16 — 720×1280": (720, 1280),
}


def build_video_tab(tab_id="video"):
    cfg = registry.ltx_config()
    with gr.Tab("🎬 Vidéo (LTX-2.3)", id=tab_id):
        gr.Markdown(
            "### Génération vidéo — LTX-2.3 (natif sd.cpp)\n"
            "Texte→vidéo, image→vidéo, ou image **début→fin**.  \n"
            "> ⚠️ Modèle **22B** + encodeur Gemma-3-12B : **TRÈS lourd**. Idéal "
            "≥16 Go. Sur 11–12 Go : quant basse (Réglages → `Q3_K`/`Q2_K`) + "
            "offload, et compte **plusieurs minutes** par clip. Commence petit "
            "(640×360, 25 images).")

        with gr.Accordion("⚙️ Installer LTX-2.3 (en 1 clic)",
                          open=not registry.ltx_ready()):
            gr.Markdown(
                "Télécharge la diffusion 22B GGUF + l'encodeur Gemma-3-12B + les "
                "VAE vidéo/audio + les connecteurs (**plusieurs Go**, c'est long).")
            inst_log = gr.Textbox(label="Journal d'installation", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            inst_btn = gr.Button("⬇️ Installer LTX-2.3")

            def _install():
                lines: list[str] = []
                for msg in downloader.download_ltx(log=lines.append):
                    lines.append(msg)
                    yield "\n".join(lines)

            inst_btn.click(_install, outputs=[inst_log])

        with gr.Row():
            with gr.Column(scale=3):
                mode = gr.Radio(
                    [(t("Texte → vidéo"), "t2v"), (t("Image → vidéo"), "i2v"),
                     (t("Début → fin"), "flf2v")], value="t2v", label="Mode")
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Décrivez la scène / le mouvement…")
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      value=cfg.get("negative", ""))
                with gr.Row():
                    init_image = gr.Image(label="Image (début)", type="pil",
                                          visible=False)
                    end_image = gr.Image(label="Image de fin", type="pil",
                                         visible=False)
                ratio = gr.Dropdown([t(k) for k in RATIOS],
                                    value=t("Léger 16:9 — 640×360"), label="Format")
                with gr.Row():
                    frames = gr.Slider(9, 121, value=33, step=8,
                                       label="Images (≈ durée × fps)")
                    fps = gr.Slider(8, 30, value=int(cfg.get("fps", 24)), step=1,
                                    label="FPS")
                with gr.Row():
                    steps = gr.Slider(8, 50, value=int(cfg.get("steps", 30)),
                                      step=1, label="Étapes")
                    cfg_s = gr.Slider(1.0, 12.0, value=float(cfg.get("cfg_scale", 6.0)),
                                      step=0.5, label="CFG")
                with gr.Row():
                    run = gr.Button("🎬 Générer la vidéo", variant="primary",
                                    size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
            with gr.Column(scale=4):
                result = gr.Video(label="Vidéo", height=420)
                logbox = gr.Textbox(label="Journal", lines=14, autoscroll=True,
                                    elem_classes="log-box")

        def _on_mode(m):
            return (gr.update(visible=(m in ("i2v", "flf2v"))),
                    gr.update(visible=(m == "flf2v")))

        mode.change(_on_mode, inputs=[mode], outputs=[init_image, end_image])

        def do_video(mode, prompt, negative, init_image, end_image, ratio,
                     frames, fps, steps, cfg_s, progress=gr.Progress()):
            if not (prompt or "").strip():
                raise gr.Error(t("Saisissez un prompt."))
            if mode in ("i2v", "flf2v") and init_image is None:
                raise gr.Error(t("Fournissez l'image de départ."))
            if mode == "flf2v" and end_image is None:
                raise gr.Error(t("Fournissez l'image de fin."))
            settings.ensure_dirs()
            w, h = RATIOS.get(i18n.to_source(ratio), (640, 360))
            ip = ep = None
            if init_image is not None:
                ip = settings.TMP_DIR / "ltx_start.png"; init_image.save(ip)
            if end_image is not None:
                ep = settings.TMP_DIR / "ltx_end.png"; end_image.save(ep)
            logs: list[str] = []
            progress(0.05, desc="Génération vidéo (long)…")
            try:
                out = gen_engine.generate_video(
                    prompt=prompt, negative=negative or "", mode=mode,
                    init_image=ip, end_image=ep, width=w, height=h,
                    frames=int(frames), fps=int(fps), cfg_scale=float(cfg_s),
                    steps=int(steps), log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            logs.append(f"\n✅ Vidéo : {out}")
            return str(out), "\n".join(logs)

        evt = run.click(
            do_video,
            inputs=[mode, prompt, negative, init_image, end_image, ratio,
                    frames, fps, steps, cfg_s],
            outputs=[result, logbox])
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])
