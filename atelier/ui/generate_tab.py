"""Onglet Génération : text-to-image / image-to-image, réglages auto par modèle,
LoRA. Le moteur est stable-diffusion.cpp (GGUF).
"""
from __future__ import annotations

import re

import gradio as gr

from .. import ideogram_prompt, registry, settings
from ..engine import generate as gen_engine

SAMPLERS = ["euler", "euler_a", "heun", "dpm++2m", "dpm++2mv2", "dpm2",
            "ipndm", "lcm", "ddim_trailing"]

# Formats prédéfinis (multiples de 64, ~1 Mpx) compatibles Ideogram 4 et Z-Image.
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


def _model_choices() -> list[tuple[str, str]]:
    prefs = settings.load_prefs()
    choices: list[tuple[str, str]] = []
    for m in registry.load_base_models(prefs):
        ready = registry.model_is_ready(m)
        mark = "●" if ready else "○ (à télécharger)"
        choices.append((f"{m.name} {mark}", m.id))
    return choices


def _defaults_for(model_id: str):
    prefs = settings.load_prefs()
    m = registry.get_base_model(model_id, prefs)
    if not m:
        return {}
    return m.defaults


def build_generate_tab():
    with gr.Tab("🎨 Génération"):
        with gr.Row():
            # ---------------- Colonne entrées ----------------
            with gr.Column(scale=3):
                model = gr.Dropdown(
                    label="Modèle de base", choices=_model_choices(),
                    value=(_model_choices()[0][1] if _model_choices() else None),
                )
                model_info = gr.Markdown("")

                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Décrivez l'image…")
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      visible=True)

                with gr.Accordion("🧠 Éditeur de prompt Ideogram 4 (JSON, optionnel)",
                                  open=False):
                    gr.Markdown(
                        "Construit le **prompt structuré JSON** qu'Ideogram 4 "
                        "comprend le mieux (style + composition + texte). "
                        "Remplissez, puis cliquez sur *Construire* : le résultat "
                        "remplace le Prompt ci-dessus. *(Conçu pour Ideogram 4.)*")
                    ie_mode = gr.Radio(["art_style", "photo"], value="art_style",
                                       label="Mode")
                    ie_hl = gr.Textbox(label="Description générale (high level)",
                                       lines=2)
                    with gr.Row():
                        ie_aes = gr.Textbox(label="Esthétique")
                        ie_light = gr.Textbox(label="Lumière")
                    with gr.Row():
                        ie_med = gr.Textbox(label="Medium")
                        ie_style = gr.Textbox(label="Style artistique / Photo")
                    ie_bg = gr.Textbox(label="Arrière-plan", lines=2)
                    ie_colors = gr.Textbox(
                        label="Palette globale (#hex, séparés par des virgules)",
                        placeholder="#E7C84B, #1B3A5B")
                    ie_elements = gr.Dataframe(
                        headers=ideogram_prompt.ELEMENT_HEADERS,
                        value=ideogram_prompt.EXAMPLE_ELEMENTS,
                        datatype=["str", "str", "str", "number", "number",
                                  "number", "number", "str"],
                        row_count=(1, "dynamic"), col_count=(8, "fixed"),
                        label="Éléments / boîtes — coordonnées bbox en 0–1000")
                    ie_build = gr.Button("🧱 Construire le prompt JSON → Prompt")

                with gr.Accordion("🖼️ Image de départ (image-to-image)", open=False):
                    init_image = gr.Image(label="Source", type="pil", height=220)
                    strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05,
                                         label="Force de transformation")

                with gr.Accordion("🧩 LoRA", open=False):
                    with gr.Row():
                        lora1 = gr.Dropdown(label="LoRA 1", choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora1_w = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Poids")
                    with gr.Row():
                        lora2 = gr.Dropdown(label="LoRA 2", choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora2_w = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Poids")
                    refresh_lora = gr.Button("↻ Rafraîchir la liste", size="sm")
                    gr.Markdown(f"Déposez vos fichiers LoRA dans `{settings.LORA_DIR}`")

                ratio = gr.Dropdown(list(RATIOS.keys()),
                                    value="Carré 1:1 — 1024×1024",
                                    label="Format (ratio)")
                with gr.Row():
                    width = gr.Slider(256, 1536, value=1024, step=64, label="Largeur")
                    height = gr.Slider(256, 1536, value=1024, step=64, label="Hauteur")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=8, step=1, label="Étapes")
                    cfg = gr.Slider(1.0, 12.0, value=1.0, step=0.1, label="CFG")
                with gr.Row():
                    sampler = gr.Dropdown(SAMPLERS, value="euler", label="Sampler")
                    seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)", precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                run = gr.Button("✨ Générer", variant="primary", size="lg")

            # ---------------- Colonne sorties ----------------
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Résultats", columns=2, height=520,
                                     object_fit="contain", show_label=True,
                                     format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=10, max_lines=20,
                                    autoscroll=True, elem_classes="log-box")

        # ---------------- Comportements ----------------
        def on_model_change(model_id):
            prefs = settings.load_prefs()
            m = registry.get_base_model(model_id, prefs) if model_id else None
            if not m:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update())
            d = m.defaults
            ready = registry.model_is_ready(m)
            tags = " ".join(f"`{t}`" for t in m.tags)
            status = ("<span class='status-ok'>● prêt</span>" if ready
                      else "<span class='status-missing'>○ à télécharger "
                           "(onglet Bibliothèque)</span>")
            info = f"{status} · {tags}\n\n{m.description}"
            supports_neg = d.get("supports_negative", True)
            return (
                gr.update(value=info),
                gr.update(value=d.get("steps", 8)),
                gr.update(value=d.get("cfg_scale", 1.0)),
                gr.update(value=d.get("sampler", "euler")),
                gr.update(value=d.get("width", 1024)),
                gr.update(value=d.get("height", 1024)),
                gr.update(visible=supports_neg),
            )

        model.change(on_model_change, inputs=[model],
                     outputs=[model_info, steps, cfg, sampler, width, height, negative])

        def refresh_loras():
            choices = gen_engine.list_loras()
            return gr.update(choices=choices), gr.update(choices=choices)

        refresh_lora.click(refresh_loras, outputs=[lora1, lora2])

        def on_ratio(label):
            w, h = RATIOS.get(label, (0, 0))
            if not w:  # « Personnalisé » : on laisse les sliders tels quels
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        ratio.change(on_ratio, inputs=[ratio], outputs=[width, height])

        def build_ideogram(mode, hl, aes, light, med, style, bg, colors, df):
            rows = df.values.tolist() if hasattr(df, "values") else (df or [])
            js = ideogram_prompt.build_prompt(mode, hl, aes, light, med, style,
                                              bg, colors, rows)
            return gr.update(value=js)

        ie_build.click(
            build_ideogram,
            inputs=[ie_mode, ie_hl, ie_aes, ie_light, ie_med, ie_style, ie_bg,
                    ie_colors, ie_elements],
            outputs=[prompt])

        def do_generate(model_id, prompt, negative, init_image, strength,
                        width, height, steps, cfg, sampler, seed, batch,
                        lora1, lora1_w, lora2, lora2_w,
                        progress=gr.Progress()):
            if not model_id:
                raise gr.Error("Sélectionnez un modèle.")
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

            # Suit la progression réelle : sd-cli imprime « étape/total ».
            total = max(1, int(steps))
            step_re = re.compile(rf"(\d+)\s*/\s*{total}\b")

            def log(line: str):
                logs.append(line)
                m = step_re.search(line)
                if m:
                    cur = min(int(m.group(1)), total)
                    progress(0.05 + 0.9 * cur / total,
                             desc=f"Génération… étape {cur}/{total}")

            progress(0.02, desc="Chargement du modèle…")
            try:
                outs = gen_engine.generate(
                    model_id=model_id, prompt=prompt or "", negative=negative or "",
                    steps=int(steps), cfg_scale=float(cfg), width=int(width),
                    height=int(height), seed=int(seed), batch_count=int(batch),
                    sampler=sampler, init_image=init_path, strength=float(strength),
                    loras=loras, log=log)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return [], "\n".join(logs)
            progress(1.0, desc="Terminé")
            # On renvoie les CHEMINS .png (et non des objets image) : le
            # téléchargement conserve ainsi le nom de fichier et l'extension.
            return [str(p) for p in outs], "\n".join(logs)

        run.click(
            do_generate,
            inputs=[model, prompt, negative, init_image, strength, width, height,
                    steps, cfg, sampler, seed, batch, lora1, lora1_w, lora2, lora2_w],
            outputs=[gallery, logbox],
        )

        return {"model": model}
