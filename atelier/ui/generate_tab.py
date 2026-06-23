"""Onglet de génération (un par modèle) : text-to-image / image-to-image,
presets sampler/scheduler, LoRA, fichiers locaux, envoi vers l'upscale.
"""
from __future__ import annotations

import random
import re

import gradio as gr

from .. import registry, settings, styles
from ..engine import generate as gen_engine

# (libellé affiché, valeur réelle sd-cli). Samplers supportés par sd.cpp.
SAMPLERS = [
    ("Euler", "euler"), ("Euler Ancestral", "euler_a"), ("Heun", "heun"),
    ("DPM2", "dpm2"), ("DPM++ 2S Ancestral", "dpm++2s_a"), ("DPM++ 2M", "dpm++2m"),
    ("DPM++ 2M v2", "dpm++2mv2"), ("iPNDM", "ipndm"), ("iPNDM v", "ipndm_v"),
    ("LCM", "lcm"), ("DDIM Trailing", "ddim_trailing"), ("TCD", "tcd"),
    ("Res Multistep", "res_multistep"), ("Res 2S", "res_2s"), ("ER SDE", "er_sde"),
    ("Euler CFG++", "euler_cfg_pp"), ("Euler Ancestral CFG++", "euler_a_cfg_pp"),
]
# Schedulers (sigmas) supportés par sd.cpp.
SCHEDULES = [
    ("Auto (modèle)", "auto"), ("Discrete", "discrete"), ("Karras", "karras"),
    ("Exponential", "exponential"), ("AYS", "ays"), ("GITS", "gits"),
    ("Smoothstep", "smoothstep"), ("SGM Uniform", "sgm_uniform"),
    ("Simple", "simple"), ("KL Optimal", "kl_optimal"), ("LCM", "lcm"),
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
    for label, (rw, rh) in RATIOS.items():
        if rw == w and rh == h:
            return label
    return "Personnalisé (sliders)"


def build_generative_tab(model_id: str, title: str,
                         tabs=None, pending_upscale=None, upscale_tab_id="creative"):
    d = _defaults(model_id)

    with gr.Tab(title):
        m = registry.get_base_model(model_id, settings.load_prefs())
        ready = m is not None and registry.model_is_ready(m)
        status = ("<span class='status-ok'>● modèle prêt</span>" if ready
                  else "<span class='status-missing'>○ à télécharger "
                       "(onglet Bibliothèque)</span>")
        gr.Markdown(f"### {title} — text-to-image & image-to-image  ·  {status}")

        with gr.Row():
            # ----- Entrées -----
            with gr.Column(scale=3):
                with gr.Accordion("🎭 Prompt système / style (préfixe, optionnel)",
                                  open=False):
                    system_prompt = gr.Textbox(
                        label="Appliqué en tête de chaque génération", lines=2,
                        placeholder="ex. : style aquarelle, palette pastel, "
                                    "éclairage doux")
                    with gr.Row():
                        style_pick = gr.Dropdown(
                            styles.list_styles(), value=None, scale=3,
                            label="Styles enregistrés", allow_custom_value=False)
                        style_name = gr.Textbox(
                            label="Nom du style à enregistrer", scale=2,
                            placeholder="ex. : Aquarelle pastel")
                    with gr.Row():
                        style_save = gr.Button("💾 Enregistrer", size="sm")
                        style_del = gr.Button("🗑️ Supprimer", size="sm")
                        style_refresh = gr.Button("↻ Rafraîchir", size="sm")
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Décrivez l'image…")
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      visible=d.get("supports_negative", False))

                with gr.Accordion("🖼️ Image de référence (édition d'image)",
                                  open=False):
                    gr.Markdown(
                        "**Éditer une image existante.** Chargez une image, puis "
                        "décrivez **la modification** dans le prompt — inutile de "
                        "tout redécrire :  \n"
                        "• *« remplace le ciel par un coucher de soleil »*  \n"
                        "• *« transforme la voiture en version cyberpunk »*  \n"
                        "• *« ajoute de la neige au sol »*  \n"
                        "Le **format de sortie s'adapte automatiquement** à votre "
                        "image. L'édition est pilotée par le prompt (pas de curseur "
                        "de force).  \n"
                        "⏱️ Plus lourde que la génération → restez sur **⚡ Rapide "
                        "(4 pas)** et une résolution **≤ 1024**.")
                    init_image = gr.Image(label="Image à éditer", type="pil")

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

                with gr.Accordion("📂 Fichiers locaux (modèle perso)", open=False):
                    gr.Markdown(
                        "Pour utiliser un modèle **téléchargé ailleurs** : déposez "
                        f"le(s) fichier(s) dans `{settings.CUSTOM_DIR}` puis "
                        "sélectionnez-le ci-dessous. Vide = modèle du catalogue.")
                    custom_diff = gr.Dropdown(gen_engine.list_custom_models(),
                                              value=None, label="Diffusion (local)")
                    with gr.Row():
                        custom_vae = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="VAE (local)")
                        custom_enc = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="Encodeur (local)")
                    refresh_custom = gr.Button("↻ Rafraîchir les fichiers locaux",
                                               size="sm")

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
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)",
                                     precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                with gr.Row():
                    run = gr.Button(f"✨ Générer ({title})", variant="primary",
                                    size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)

            # ----- Sorties -----
            with gr.Column(scale=4):
                preview = gr.Image(label="Aperçu (temps réel)", height=320,
                                   visible=True)
                gallery = gr.Gallery(label="Résultats (légende = seed)", columns=2,
                                     height=420, object_fit="contain",
                                     show_label=True, format="png",
                                     show_download_button=True)
                send_upscale = gr.Button("📤 Envoyer l'image sélectionnée "
                                         "vers l'Upscale créatif")
                logbox = gr.Textbox(label="Journal", lines=10, max_lines=24,
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

        def _refresh_custom():
            c = gen_engine.list_custom_models()
            return gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        refresh_custom.click(_refresh_custom,
                             outputs=[custom_diff, custom_vae, custom_enc])

        # --- Styles enregistrés (prompt système) ---
        def _load_style(name):
            return gr.update(value=styles.get_style(name))

        style_pick.change(_load_style, inputs=[style_pick],
                          outputs=[system_prompt])

        def _save_style(name, text):
            try:
                saved = styles.save_style(name, text)
            except ValueError as exc:
                raise gr.Error(str(exc))
            return (gr.update(choices=styles.list_styles(), value=saved),
                    gr.update(value=""))

        style_save.click(_save_style, inputs=[style_name, system_prompt],
                         outputs=[style_pick, style_name])

        def _delete_style(name):
            styles.delete_style(name)
            return gr.update(choices=styles.list_styles(), value=None)

        style_del.click(_delete_style, inputs=[style_pick], outputs=[style_pick])

        def _refresh_styles():
            return gr.update(choices=styles.list_styles())

        style_refresh.click(_refresh_styles, outputs=[style_pick])

        def on_ratio(label):
            w, h = RATIOS.get(label, (0, 0))
            if not w:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        ratio.change(on_ratio, inputs=[ratio], outputs=[width, height])

        def _fit_to_ref(img):
            """Édition : cale la sortie sur le format de l'image de référence
            (côté long plafonné à 1024 px pour rester rapide, multiples de 16)."""
            if img is None:
                return gr.update(), gr.update(), gr.update()
            w0, h0 = img.size
            longest = max(w0, h0) or 1
            sc = 1024 / longest if longest > 1024 else 1.0
            w = max(256, min(2048, int(round(w0 * sc / 16)) * 16))
            h = max(256, min(2048, int(round(h0 * sc / 16)) * 16))
            return (gr.update(value=w), gr.update(value=h),
                    gr.update(value="Personnalisé (sliders)"))

        init_image.upload(_fit_to_ref, inputs=[init_image],
                          outputs=[width, height, ratio])

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

        def do_generate(system_prompt, prompt, negative, init_image,
                        width, height, steps, cfg, sampler, schedule, flow_shift,
                        seed, batch, lora1, lora1_w, lora2, lora2_w,
                        custom_diff, custom_vae, custom_enc,
                        progress=gr.Progress()):
            import queue
            import threading
            import time

            if not (prompt or "").strip():
                raise gr.Error("Saisissez un prompt "
                               "(décrivez l'image, ou la modification à appliquer).")

            full_prompt = prompt or ""
            if (system_prompt or "").strip():
                full_prompt = f"{system_prompt.strip()}, {full_prompt}".strip(", ")

            base_seed = int(seed)
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            settings.ensure_dirs()
            ref_path = None
            if init_image is not None:
                ref_path = settings.TMP_DIR / "edit_ref.png"
                init_image.save(ref_path)

            loras = [(lora1, float(lora1_w)), (lora2, float(lora2_w))]
            loras = [(n, w) for n, w in loras if n]

            preview_path = settings.TMP_DIR / f"preview_{int(time.time()*1000)}.png"
            try:
                preview_path.unlink()
            except OSError:
                pass

            total = max(1, int(steps))
            step_re = re.compile(rf"(\d+)\s*/\s*{total}\b")
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    outs = gen_engine.generate(
                        model_id=model_id, prompt=full_prompt,
                        negative=negative or "", steps=int(steps),
                        cfg_scale=float(cfg), width=int(width), height=int(height),
                        seed=base_seed, batch_count=int(batch), sampler=sampler,
                        schedule=schedule, flow_shift=float(flow_shift or 0.0),
                        ref_image=ref_path, loras=loras,
                        diffusion_override=gen_engine.custom_path(custom_diff),
                        vae_override=gen_engine.custom_path(custom_vae),
                        encoder_override=gen_engine.custom_path(custom_enc),
                        preview_path=preview_path, log=q.put)
                    state["outs"] = [str(p) for p in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            last_mtime = None
            progress(0.02, desc="Chargement du modèle…")
            while True:
                try:
                    line = q.get(timeout=0.4)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                changed = False
                if line:
                    logs.append(line)
                    changed = True
                    mt = step_re.search(line)
                    if mt:
                        cur = min(int(mt.group(1)), total)
                        progress(0.05 + 0.9 * cur / total,
                                 desc=f"étape {cur}/{total}")
                # Aperçu : on lit le fichier en mémoire (copie PIL) plutôt que de
                # passer le chemin à Gradio — sous Windows sd-cli écrit ce fichier
                # en continu et Gradio ne peut pas l'ouvrir pendant l'écriture.
                # On ne pousse une nouvelle image QUE si le fichier a changé
                # (mtime), sinon Gradio re-rend la même image -> clignotement.
                prev = gr.update()
                if preview_path.exists():
                    try:
                        m = preview_path.stat().st_mtime
                        if m != last_mtime:
                            from PIL import Image
                            with Image.open(preview_path) as _pim:
                                prev = _pim.copy()
                            last_mtime = m
                            changed = True
                    except (OSError, ValueError):
                        pass
                if not changed:
                    continue
                yield gr.update(), prev, "\n".join(logs[-400:]), gr.update()

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                yield [], gr.update(value=None), "\n".join(logs), []
                return
            progress(1.0, desc="Terminé")
            paths = state.get("outs", [])
            items = [(p, f"seed {base_seed + i}") for i, p in enumerate(paths)]
            yield items, gr.update(value=None), "\n".join(logs), paths

        gen_evt = run.click(
            do_generate,
            inputs=[system_prompt, prompt, negative, init_image, width,
                    height, steps, cfg, sampler, schedule, flow_shift, seed, batch,
                    lora1, lora1_w, lora2, lora2_w,
                    custom_diff, custom_vae, custom_enc],
            outputs=[gallery, preview, logbox, last_paths],
        )
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[gen_evt])

        def _on_select(evt: gr.SelectData):
            return evt.index

        gallery.select(_on_select, outputs=[sel_index])

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
