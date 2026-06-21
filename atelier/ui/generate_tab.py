"""Onglets de génération, un par modèle (sections séparées Z-Image / Ideogram).

`build_generative_tab(model_id, title, is_ideogram)` construit un onglet complet
text-to-image / image-to-image pour UN modèle donné. Pour Ideogram 4, un canvas
de composition permet de dessiner les boîtes (objets/texte) et de fabriquer le
prompt JSON structuré.
"""
from __future__ import annotations

import random
import re

import gradio as gr

from .. import downloader, ideogram_prompt, registry, settings
from ..engine import generate as gen_engine
from .canvas import CANVAS_MARKUP, READ_BOXES_JS

TURBO_LORA_REPO = "ostris/ideogram_4_turbotime_lora"
UNCOND_LORA_REPO = "ostris/ideogram_4_unconditional_lora"

# (libellé affiché, valeur réelle passée à sd-cli). Liste complète des
# sampling methods réellement supportées par stable-diffusion.cpp.
SAMPLERS = [
    ("Euler", "euler"),
    ("Euler Ancestral", "euler_a"),
    ("Heun", "heun"),
    ("DPM2", "dpm2"),
    ("DPM++ 2S Ancestral", "dpm++2s_a"),
    ("DPM++ 2M", "dpm++2m"),
    ("DPM++ 2M v2", "dpm++2mv2"),
    ("iPNDM", "ipndm"),
    ("iPNDM v", "ipndm_v"),
    ("LCM", "lcm"),
    ("DDIM Trailing", "ddim_trailing"),
    ("TCD", "tcd"),
    ("Res Multistep", "res_multistep"),
    ("Res 2S", "res_2s"),
    ("ER SDE", "er_sde"),
    ("Euler CFG++", "euler_cfg_pp"),
    ("Euler Ancestral CFG++", "euler_a_cfg_pp"),
]
# Schedulers (sigmas) réellement supportés par stable-diffusion.cpp.
SCHEDULES = [
    ("Auto (modèle)", "auto"),
    ("Discrete", "discrete"),
    ("Karras", "karras"),
    ("Exponential", "exponential"),
    ("AYS", "ays"),
    ("GITS", "gits"),
    ("Smoothstep", "smoothstep"),
    ("SGM Uniform", "sgm_uniform"),
    ("Simple", "simple"),
    ("KL Optimal", "kl_optimal"),
    ("LCM", "lcm"),
    ("Bong Tangent", "bong_tangent"),
]
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


def _presets(model_id: str) -> list[dict]:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return list(m.presets) if (m and m.presets) else []


def _ratio_label(w: int, h: int) -> str:
    """Libellé de format correspondant à une résolution, sinon « Personnalisé »."""
    for label, (rw, rh) in RATIOS.items():
        if rw == w and rh == h:
            return label
    return "Personnalisé (sliders)"


def build_generative_tab(model_id: str, title: str, is_ideogram: bool = False,
                         tabs=None, pending_upscale=None, upscale_tab_id="upscale"):
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
                with gr.Row():
                    ie_build = gr.Button("🧱 Construire depuis le canvas → Prompt",
                                         variant="secondary")
                    ie_text2json = gr.Button("✍️ Convertir le Prompt en JSON "
                                             "Ideogram", variant="secondary")
                ie_widgets = dict(mode=ie_mode, hl=ie_hl, aes=ie_aes, light=ie_light,
                                  med=ie_med, style=ie_style, bg=ie_bg,
                                  colors=ie_colors, boxes=boxes_holder, build=ie_build,
                                  text2json=ie_text2json)

        with gr.Row():
            # ----- Entrées -----
            with gr.Column(scale=3):
                with gr.Accordion("🎭 Prompt système / style (préfixe, optionnel)",
                                  open=False):
                    system_prompt = gr.Textbox(
                        label="Appliqué en tête de chaque génération", lines=2,
                        placeholder="ex. : style aquarelle, palette pastel, "
                                    "éclairage doux")
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
                    if is_ideogram:
                        with gr.Row():
                            turbo_lora = gr.Button("⚡ Turbo LoRA (accélère)",
                                                   size="sm")
                            uncond_lora = gr.Button("✨ Unconditional LoRA "
                                                    "(qualité)", size="sm")
                    gr.Markdown(f"Déposez vos fichiers LoRA dans `{settings.LORA_DIR}`")

                ratio = gr.Dropdown(list(RATIOS.keys()),
                                    value=_ratio_label(d.get("width", 1024),
                                                       d.get("height", 1024)),
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
                preset_list = _presets(model_id)
                preset = gr.Dropdown(
                    [p["name"] for p in preset_list],
                    value=(preset_list[0]["name"] if preset_list else None),
                    label="Préréglage (sampler/scheduler/pas)",
                    visible=bool(preset_list))
                with gr.Row():
                    sampler = gr.Dropdown(SAMPLERS, value=d.get("sampler", "euler"),
                                          label="Sampler")
                    schedule = gr.Dropdown(SCHEDULES,
                                           value=d.get("scheduler", "auto"),
                                           label="Scheduler (sigmas)")
                flow_shift = gr.Slider(
                    0.0, 12.0, value=float(d.get("flow_shift", 0.0)), step=0.1,
                    label="Flow shift (0 = auto · ~3 = + de structure)")
                if is_ideogram:
                    quality_preset = gr.Button("🎯 Préréglage Qualité "
                                               "(DPM++ 2M · 28 pas)", size="sm")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)",
                                     precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                run = gr.Button(f"✨ Générer ({title})", variant="primary", size="lg")

            # ----- Sorties -----
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Résultats (légende = seed)", columns=2,
                                     height=680, object_fit="contain",
                                     show_label=True, format="png",
                                     show_download_button=True)
                with gr.Row():
                    send_upscale = gr.Button("📤 Envoyer l'image sélectionnée "
                                             "vers l'Upscale")
                logbox = gr.Textbox(label="Journal", lines=12, max_lines=24,
                                    autoscroll=True, elem_classes="log-box")

        last_paths = gr.State([])
        sel_index = gr.State(0)

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

        if preset_list:
            def apply_preset(name):
                for p in preset_list:
                    if p["name"] == name:
                        return (gr.update(value=p.get("sampler", "euler")),
                                gr.update(value=p.get("scheduler", "auto")),
                                gr.update(value=int(p.get("steps", 8))),
                                gr.update(value=float(p.get("cfg_scale", 1.0))))
                return gr.update(), gr.update(), gr.update(), gr.update()

            preset.change(apply_preset, inputs=[preset],
                          outputs=[sampler, schedule, steps, cfg])

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

            def text_to_json(plain, mode, aes, light, med, style, colors):
                # Convertit un prompt classique en JSON Ideogram (format
                # d'entraînement) : le texte devient la description générale.
                if not (plain or "").strip():
                    raise gr.Error("Écrivez d'abord un prompt classique.")
                js = ideogram_prompt.build_prompt(
                    mode, high_level=plain, aesthetics=aes, lighting=light,
                    medium=med, style_or_photo=style, background="",
                    global_colors=colors, element_rows=[])
                return gr.update(value=js)

            ie_widgets["text2json"].click(
                text_to_json,
                inputs=[prompt, ie_widgets["mode"], ie_widgets["aes"],
                        ie_widgets["light"], ie_widgets["med"],
                        ie_widgets["style"], ie_widgets["colors"]],
                outputs=[prompt])

        def do_generate(system_prompt, prompt, negative, init_image, strength,
                        width, height, steps, cfg, sampler, schedule, flow_shift,
                        seed, batch, lora1, lora1_w, lora2, lora2_w,
                        progress=gr.Progress()):
            if not (prompt or "").strip() and init_image is None:
                raise gr.Error("Saisissez un prompt (ou une image de départ).")

            # Prompt système appliqué en préfixe (style réutilisable).
            full_prompt = prompt or ""
            if (system_prompt or "").strip():
                full_prompt = f"{system_prompt.strip()}, {full_prompt}".strip(", ")

            # Seed concrète : on la fixe nous-mêmes pour pouvoir l'afficher.
            base_seed = int(seed)
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

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
                    model_id=model_id, prompt=full_prompt, negative=negative or "",
                    steps=int(steps), cfg_scale=float(cfg), width=int(width),
                    height=int(height), seed=base_seed, batch_count=int(batch),
                    sampler=sampler, schedule=schedule,
                    flow_shift=float(flow_shift or 0.0), init_image=init_path,
                    strength=float(strength), loras=loras, log=log)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return [], "\n".join(logs), []
            progress(1.0, desc="Terminé")
            paths = [str(p) for p in outs]
            # sd-cli incrémente la seed par image (base, base+1, …) -> légendes.
            items = [(p, f"seed {base_seed + i}") for i, p in enumerate(paths)]
            return items, "\n".join(logs), paths

        run.click(
            do_generate,
            inputs=[system_prompt, prompt, negative, init_image, strength, width,
                    height, steps, cfg, sampler, schedule, flow_shift, seed, batch,
                    lora1, lora1_w, lora2, lora2_w],
            outputs=[gallery, logbox, last_paths],
        )

        if is_ideogram:
            quality_preset.click(
                lambda: (gr.update(value="dpm++2m"), gr.update(value=28)),
                outputs=[sampler, steps])

        # Suivi de l'image sélectionnée dans la galerie.
        def _on_select(evt: gr.SelectData):
            return evt.index

        gallery.select(_on_select, outputs=[sel_index])

        # Envoi vers l'onglet Upscale (si câblé par app.py).
        if pending_upscale is not None and tabs is not None:
            def _send(paths, idx):
                if not paths:
                    raise gr.Error("Générez d'abord une image.")
                i = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
                return paths[i], gr.Tabs(selected=upscale_tab_id)

            send_upscale.click(_send, inputs=[last_paths, sel_index],
                               outputs=[pending_upscale, tabs])
        else:
            send_upscale.visible = False

        # Turbo LoRA Ideogram : télécharge ostris/ideogram_4_turbotime_lora et
        # l'active (réduit fortement le nombre d'étapes).
        if is_ideogram:
            def _enable_lora(repo, new_steps):
                try:
                    name = downloader.download_lora(repo)
                except Exception as exc:  # noqa: BLE001
                    raise gr.Error(f"Échec du téléchargement du LoRA : {exc}")
                return (gr.update(choices=gen_engine.list_loras(), value=name),
                        gr.update(value=1.0), gr.update(value=new_steps))

            turbo_lora.click(lambda: _enable_lora(TURBO_LORA_REPO, 8),
                             outputs=[lora1, lora1_w, steps])
            uncond_lora.click(lambda: _enable_lora(UNCOND_LORA_REPO, 20),
                              outputs=[lora1, lora1_w, steps])
