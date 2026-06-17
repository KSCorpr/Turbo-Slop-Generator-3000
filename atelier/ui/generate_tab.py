"""Onglets de génération, un par modèle (sections séparées Z-Image / Ideogram).

`build_generative_tab(model_id, title, is_ideogram)` construit un onglet complet
text-to-image / image-to-image pour UN modèle donné. Pour Ideogram 4, un canvas
de composition permet de dessiner les boîtes (objets/texte) et de fabriquer le
prompt JSON structuré.
"""
from __future__ import annotations

import re

import gradio as gr

from .. import ideogram_prompt, registry, settings
from ..engine import generate as gen_engine
from .canvas import CANVAS_MARKUP, READ_BOXES_JS

SAMPLERS = ["euler", "euler_a", "heun", "dpm++2m", "dpm++2mv2", "dpm2",
            "ipndm", "lcm", "ddim_trailing"]
SCHEDULES = ["discrete", "karras", "exponential", "ays", "gits",
             "smoothstep", "sgm_uniform"]
RATIOS: dict[str, tuple[int, int]] = {
    "Carré 1:1 — 1024×1024": (1024, 1024),
    "Paysage 3:2 — 1216×832": (1216, 832),
    "Portrait 2:3 — 832×1216": (832, 1216),
    "Paysage 4:3 — 1152×896": (1152, 896),
    "Portrait 3:4 — 896×1152": (896, 1152),
    "Large 16:9 — 1344×768": (1344, 768),
    "Vertical 9:16 — 768×1344": (768, 1344),
    "Cinéma 21:9 — 1536×640": (1536, 640),
    "Personnalisé (sliders)": (0, 0),
}


def _defaults(model_id: str) -> dict:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return m.defaults if m else {}


def build_generative_tab(model_id: str, title: str, is_ideogram: bool = False):
    d = _defaults(model_id)

    with gr.Tab(title):
        m = registry.get_base_model(model_id, settings.load_prefs())
        ready = m is not None and registry.model_is_ready(m)
        status = ("<span class='status-ok'>● modèle prêt</span>" if ready
                  else "<span class='status-missing'>○ à télécharger "
                       "(onglet Bibliothèque)</span>")
        gr.Markdown(f"### {title} — text-to-image & image-to-image  ·  {status}")

        # ----- Composition visuelle (Ideogram uniquement) -----------------
        ie_widgets = {}
        if is_ideogram:
            with gr.Accordion("🖼️ Composition visuelle — dessinez les zones "
                              "(canvas)", open=True):
                gr.HTML(CANVAS_MARKUP)
                with gr.Row():
                    ie_mode = gr.Radio(["art_style", "photo"], value="art_style",
                                       label="Mode")
                    ie_hl = gr.Textbox(label="Description générale", scale=3)
                with gr.Row():
                    ie_aes = gr.Textbox(label="Esthétique")
                    ie_light = gr.Textbox(label="Lumière")
                with gr.Row():
                    ie_med = gr.Textbox(label="Medium")
                    ie_style = gr.Textbox(label="Style artistique / Photo")
                ie_bg = gr.Textbox(label="Arrière-plan", lines=2)
                ie_colors = gr.Textbox(label="Palette globale (#hex)",
                                       placeholder="#E7C84B, #1B3A5B")
                boxes_holder = gr.Textbox(visible=False)
                ie_build = gr.Button("🧱 Construire le prompt depuis le canvas "
                                     "→ Prompt", variant="secondary")
                ie_widgets = dict(mode=ie_mode, hl=ie_hl, aes=ie_aes, light=ie_light,
                                  med=ie_med, style=ie_style, bg=ie_bg,
                                  colors=ie_colors, boxes=boxes_holder, build=ie_build)

        with gr.Row():
            # ----- Entrées -----
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Décrivez l'image…")
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      visible=d.get("supports_negative", False))

                with gr.Accordion("🖼️ Image de départ (image-to-image)", open=False):
                    init_image = gr.Image(label="Source", type="pil", height=300)
                    strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05,
                                         label="Force de transformation")

                with gr.Accordion("🧩 LoRA", open=False):
                    with gr.Row():
                        lora1 = gr.Dropdown(label="LoRA 1",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora1_w = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Poids")
                    with gr.Row():
                        lora2 = gr.Dropdown(label="LoRA 2",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora2_w = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Poids")
                    with gr.Row():
                        refresh_lora = gr.Button("↻ Rafraîchir la liste", size="sm")
                        clear_lora = gr.Button("✖ Vider les LoRA", size="sm")
                    gr.Markdown(f"Déposez vos fichiers LoRA dans `{settings.LORA_DIR}`")

                ratio = gr.Dropdown(list(RATIOS.keys()),
                                    value="Carré 1:1 — 1024×1024",
                                    label="Format (ratio)")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=d.get("width", 1024), step=16,
                                      label="Largeur")
                    height = gr.Slider(256, 2048, value=d.get("height", 1024), step=16,
                                       label="Hauteur")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=d.get("steps", 8), step=1,
                                      label="Étapes")
                    cfg = gr.Slider(1.0, 12.0, value=d.get("cfg_scale", 1.0),
                                    step=0.1, label="CFG")
                with gr.Row():
                    sampler = gr.Dropdown(SAMPLERS, value=d.get("sampler", "euler"),
                                          label="Sampler")
                    schedule = gr.Dropdown(SCHEDULES, value="discrete",
                                           label="Schedule (bruit)")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)",
                                     precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                run = gr.Button(f"✨ Générer ({title})", variant="primary", size="lg")

            # ----- Sorties -----
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Résultats", columns=2, height=680,
                                     object_fit="contain", show_label=True,
                                     format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=14, max_lines=24,
                                    autoscroll=True, elem_classes="log-box")

        # ----- Comportements -----
        def refresh_loras():
            choices = gen_engine.list_loras()
            return gr.update(choices=choices), gr.update(choices=choices)

        refresh_lora.click(refresh_loras, outputs=[lora1, lora2])

        def clear_loras():
            return (gr.update(value=None), gr.update(value=0.8),
                    gr.update(value=None), gr.update(value=0.8))

        clear_lora.click(clear_loras, outputs=[lora1, lora1_w, lora2, lora2_w])

        def on_ratio(label):
            w, h = RATIOS.get(label, (0, 0))
            if not w:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        ratio.change(on_ratio, inputs=[ratio], outputs=[width, height])

        if is_ideogram:
            def build_ideogram(boxes_json, mode, hl, aes, light, med, style,
                               bg, colors):
                rows = ideogram_prompt.boxes_json_to_rows(boxes_json)
                js = ideogram_prompt.build_prompt(mode, hl, aes, light, med, style,
                                                  bg, colors, rows)
                return gr.update(value=js)

            ie_widgets["build"].click(
                build_ideogram,
                inputs=[ie_widgets["boxes"], ie_widgets["mode"], ie_widgets["hl"],
                        ie_widgets["aes"], ie_widgets["light"], ie_widgets["med"],
                        ie_widgets["style"], ie_widgets["bg"], ie_widgets["colors"]],
                outputs=[prompt],
                js=READ_BOXES_JS,
            )

        def do_generate(prompt, negative, init_image, strength, width, height,
                        steps, cfg, sampler, schedule, seed, batch,
                        lora1, lora1_w, lora2, lora2_w, progress=gr.Progress()):
            if not (prompt or "").strip() and init_image is None:
                raise gr.Error("Saisissez un prompt (ou une image de départ).")

            logs: list[str] = []
            init_path = None
            if init_image is not None:
                settings.ensure_dirs()
                init_path = settings.TMP_DIR / "i2i_init.png"
                init_image.save(init_path)

            loras = [(lora1, float(lora1_w)), (lora2, float(lora2_w))]
            loras = [(n, w) for n, w in loras if n]

            total = max(1, int(steps))
            step_re = re.compile(rf"(\d+)\s*/\s*{total}\b")

            def log(line: str):
                logs.append(line)
                mt = step_re.search(line)
                if mt:
                    cur = min(int(mt.group(1)), total)
                    progress(0.05 + 0.9 * cur / total,
                             desc=f"Génération… étape {cur}/{total}")

            progress(0.02, desc="Chargement du modèle…")
            try:
                outs = gen_engine.generate(
                    model_id=model_id, prompt=prompt or "", negative=negative or "",
                    steps=int(steps), cfg_scale=float(cfg), width=int(width),
                    height=int(height), seed=int(seed), batch_count=int(batch),
                    sampler=sampler, schedule=schedule, init_image=init_path,
                    strength=float(strength), loras=loras, log=log)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return [], "\n".join(logs)
            progress(1.0, desc="Terminé")
            return [str(p) for p in outs], "\n".join(logs)

        run.click(
            do_generate,
            inputs=[prompt, negative, init_image, strength, width, height,
                    steps, cfg, sampler, schedule, seed, batch,
                    lora1, lora1_w, lora2, lora2_w],
            outputs=[gallery, logbox],
        )
