"""Onglet de génération (un par modèle) : text-to-image / image-to-image,
presets sampler/scheduler, LoRA, fichiers locaux, envoi vers l'upscale.
"""
from __future__ import annotations

import random
import re

import gradio as gr

from .. import downloader, i18n, registry, settings, styles
from ..engine import generate as gen_engine
from ..engine import tools
from ..i18n import t

# (libellé affiché, valeur réelle sd-cli). Samplers supportés par sd.cpp.
SAMPLERS = [
    ("Euler", "euler"), ("Euler Ancestral", "euler_a"), ("Heun", "heun"),
    ("DPM2", "dpm2"), ("DPM++ 2S Ancestral", "dpm++2s_a"), ("DPM++ 2M", "dpm++2m"),
    ("DPM++ 2M v2", "dpm++2mv2"),
    ("DPM++ 2M SDE", "dpm++2m_sde"),
    ("DPM++ 2M SDE (Brownian)", "dpm++2m_sde_bt"),
    ("iPNDM", "ipndm"), ("iPNDM v", "ipndm_v"),
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
    # Schedulers récents sd.cpp. « Flux.2 » cible les modèles Flux.2 (recommandé
    # pour Flux.2 Klein) ; « Flux » pour Flux.1 ; « Beta » / « Logit Normal » génériques.
    ("Flux.2", "flux2"), ("Flux", "flux"), ("Beta", "beta"),
    ("Logit Normal", "logit_normal"),
]
# Préréglages de résolution PAR FAMILLE de modèle, alignés sur les résolutions
# natives d'entraînement (le modèle rend mieux sur ces formats).
#
#  • Flux.2 Klein : entraîné en ~1 MP, grille de 32 px, gère jusqu'à ~4 MP.
#    Formats natifs documentés : 1024², 1248×832, 1184×880, 1392×752, 1568×672…
#  • Krea 2 : famille SDXL/1024, multiples de 64 px : 1024², 1152×896, 1216×832,
#    1344×768, etc. (+ option 2K via VAE WAN).
RATIOS_FLUX2: dict[str, tuple[int, int]] = {
    "Carré 1:1 — 1024×1024": (1024, 1024),
    "Carré 1:1 — 1440×1440 (2K)": (1440, 1440),
    "Paysage 3:2 — 1248×832": (1248, 832),
    "Portrait 2:3 — 832×1248": (832, 1248),
    "Paysage 4:3 — 1184×880": (1184, 880),
    "Portrait 3:4 — 880×1184": (880, 1184),
    "Large 16:9 — 1392×752": (1392, 752),
    "Vertical 9:16 — 752×1392": (752, 1392),
    "Cinéma 21:9 — 1568×672": (1568, 672),
    "Personnalisé (sliders)": (0, 0),
}
RATIOS_KREA2: dict[str, tuple[int, int]] = {
    "Carré 1:1 — 1024×1024": (1024, 1024),
    "Carré 1:1 — 1536×1536 (2K)": (1536, 1536),
    "Paysage 3:2 — 1216×832": (1216, 832),
    "Portrait 2:3 — 832×1216": (832, 1216),
    "Paysage 4:3 — 1152×896": (1152, 896),
    "Portrait 3:4 — 896×1152": (896, 1152),
    "Large 16:9 — 1344×768": (1344, 768),
    "Vertical 9:16 — 768×1344": (768, 1344),
    "Personnalisé (sliders)": (0, 0),
}
# Chroma1-Radiance est pixel-space (pas de compression latente) → très gourmand
# en VRAM à haute résolution. Ratios centrés sur 768 px (sûrs sur 12 Go), le
# 1024 est marqué « VRAM++ » (carte large requise).
RATIOS_CHROMA: dict[str, tuple[int, int]] = {
    "Carré 1:1 — 768×768 (sûr)": (768, 768),
    "Carré 1:1 — 1024×1024 (VRAM++)": (1024, 1024),
    "Paysage 3:2 — 912×608": (912, 608),
    "Portrait 2:3 — 608×912": (608, 912),
    "Paysage 4:3 — 880×656": (880, 656),
    "Portrait 3:4 — 656×880": (656, 880),
    "Large 16:9 — 1024×576": (1024, 576),
    "Personnalisé (sliders)": (0, 0),
}
_CUSTOM_LABEL = "Personnalisé (sliders)"


def _ratios_for(family: str) -> dict[str, tuple[int, int]]:
    if family == "krea2":
        return RATIOS_KREA2
    if family == "chroma":
        return RATIOS_CHROMA
    return RATIOS_FLUX2


def _defaults(model_id: str) -> dict:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return m.defaults if m else {}


def _presets(model_id: str) -> list[dict]:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return list(m.presets) if (m and m.presets) else []


def _ratio_label(ratios: dict[str, tuple[int, int]], w: int, h: int) -> str:
    for label, (rw, rh) in ratios.items():
        if rw == w and rh == h:
            return label
    return _CUSTOM_LABEL


def build_generative_tab(model_id: str, title: str,
                         pending_toolkit=None, tabs=None, toolkit_tab_id="toolkit"):
    d = _defaults(model_id)

    with gr.Tab(title):
        m = registry.get_base_model(model_id, settings.load_prefs())
        ready = m is not None and registry.model_is_ready(m)
        family = m.family if m else "flux2"
        ratios = _ratios_for(family)
        # Famille « édition » (Flux.2/Kontext) : modification d'image via -r ;
        # sinon img2img classique (-i + force de transformation).
        is_edit = bool(m) and m.family == "flux2"
        status = (f"<span class='status-ok'>{t('● modèle prêt')}</span>" if ready
                  else "<span class='status-missing'>"
                       f"{t('○ à télécharger (onglet Catalogue de modèles)')}</span>")
        _mode = t("édition d'image") if is_edit else t("image-to-image")
        gr.Markdown(t("### {title} — text-to-image & {mode}  ·  {status}").format(
            title=title, mode=_mode, status=status))

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
                with gr.Row():
                    enhance_btn = gr.Button("✨ Améliorer le prompt (IA)",
                                            size="sm", scale=3)
                    enh_level = gr.Dropdown(
                        [(t("Léger"), "light"), (t("Moyen"), "medium"),
                         (t("Fort"), "strong")],
                        value="medium", label="Intensité", scale=2)
                negative = gr.Textbox(label="Prompt négatif", lines=1,
                                      visible=d.get("supports_negative", False))

                with gr.Accordion("✨ Améliorateur de prompt — installer (1 clic)",
                                  open=not tools.enhance_is_installed()):
                    gr.Markdown(
                        "Petit LLM (**Qwen2.5-3B-Instruct**, PyTorch ~6 Go) qui "
                        "réécrit votre idée en un prompt **anglais** détaillé "
                        "(sujet, lumière, cadrage, style). Chargé puis déchargé à "
                        "chaque appel : **aucun conflit de VRAM** avec la "
                        "génération. Aucune commande à taper.")
                    enh_log = gr.Textbox(label="Journal d'installation", lines=6,
                                         autoscroll=True, elem_classes="log-box")
                    enh_inst = gr.Button("⬇️ Installer l'améliorateur de prompt")

                    def _install_enh():
                        for msg in tools.install_enhance_stream():
                            yield msg

                    enh_inst.click(_install_enh, outputs=[enh_log])

                _acc_title = ("🖼️ Images de référence (édition d'image)" if is_edit
                              else "🖼️ Image de référence / départ (image-to-image)")
                with gr.Accordion(_acc_title, open=False):
                    if is_edit:
                        gr.Markdown(
                            "**Éditer une image** : chargez-la et décrivez **la "
                            "modification** dans le prompt (ex. *« change la "
                            "couleur de la voiture en rouge »*, *« ajoute de la "
                            "neige »*). Édition pilotée par le prompt (pas de "
                            "curseur de force). Le format de sortie s'adapte à "
                            "votre image. Vous pouvez ajouter **2 images de "
                            "référence** supplémentaires pour combiner des éléments "
                            "(ex. *« mets le personnage de l'image 1 dans le décor "
                            "de l'image 2 »*).")
                    else:
                        gr.Markdown(
                            "**Image de référence** (image-to-image) : chargez une "
                            "photo, décrivez le rendu voulu, et réglez la **force "
                            "de transformation** — **bas (0.2–0.4)** = garde la "
                            "structure de la référence ; **haut (0.7–1.0)** = "
                            "réinventé. Le format de sortie s'adapte à votre image.")
                    init_image = gr.Image(
                        label="Image à éditer" if is_edit else "Image de départ",
                        type="pil")
                    if is_edit:
                        with gr.Row():
                            ref_image2 = gr.Image(label="Référence 2 (option)",
                                                  type="pil")
                            ref_image3 = gr.Image(label="Référence 3 (option)",
                                                  type="pil")
                    else:
                        ref_image2 = gr.State(None)
                        ref_image3 = gr.State(None)
                    strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05,
                                         label="Force de transformation",
                                         visible=not is_edit)
                    if is_edit:
                        outpaint = gr.Slider(
                            1.0, 2.0, value=1.0, step=0.1,
                            label="🧩 Outpaint — étendre la toile (1.0 = off ; "
                                  "⚠️ expérimental)",
                            info="Agrandit la toile et laisse le modèle remplir "
                                 "les bords. Décrivez l'extension dans le prompt.")
                    else:
                        outpaint = gr.State(1.0)

                with gr.Accordion("🧩 LoRA", open=False):
                    with gr.Row():
                        lora1 = gr.Dropdown(label="LoRA 1",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora1_w = gr.Slider(0.0, 5000.0, value=0.8, step=0.05,
                                            label="Poids")
                    with gr.Row():
                        lora2 = gr.Dropdown(label="LoRA 2",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora2_w = gr.Slider(0.0, 5000.0, value=0.8, step=0.05,
                                            label="Poids")
                    with gr.Row():
                        refresh_lora = gr.Button("↻ Rafraîchir la liste", size="sm")
                        clear_lora = gr.Button("✖ Vider les LoRA", size="sm")
                    gr.Markdown(t("Déposez vos fichiers LoRA dans `{dir}`")
                                .format(dir=settings.LORA_DIR))
                    with gr.Row():
                        civitai_ref = gr.Textbox(
                            label="Importer un LoRA Civitai (URL ou ID de version)",
                            scale=3, placeholder="https://civitai.com/…"
                                                 "?modelVersionId=3067151")
                        civitai_btn = gr.Button("⬇️ Importer", scale=1)
                    civitai_msg = gr.Markdown("")

                with gr.Accordion("🚀 Sortie haute résolution (PiD ×4)",
                                  open=False):
                    gr.Markdown(
                        "Décode la sortie via le **décodeur pixel-diffusion "
                        "NVIDIA (PiD, variante Flux.2)** : agrandit **×4 par "
                        "diffusion pixel** (→ 2048 px, 4 pas). Aligné sur le "
                        "backbone **Flux.2** + son VAE (cohérent avec ta "
                        "génération Flux.2 Klein → moins d'artefacts). Coché, "
                        "l'image sort directement en haute résolution.  \n"
                        "⚠️ Poids officiels **NSCLv1 (non commercial)**. "
                        "Ratio **×4 fixe** (l'image est ramenée à ~512 px avant "
                        "décodage).")
                    pid_hires = gr.Checkbox(
                        value=False,
                        label="Décoder la sortie en haute résolution (PiD ×4)")
                    with gr.Accordion("⚙️ Installer PiD (1 clic)",
                                      open=not registry.pid_ready()):
                        gr.Markdown("Via sd.cpp (pas de PyTorch). Décodeur PiD + "
                                    "encodeur Gemma-2-2B + VAE FLUX.1 (~5 Go).")
                        pid_install_log = gr.Textbox(
                            label="Journal d'installation", lines=6,
                            autoscroll=True, elem_classes="log-box")
                        pid_install_btn = gr.Button("⬇️ Installer PiD")

                        def _install_pid():
                            lines: list[str] = []
                            for msg in downloader.download_pid(log=lines.append):
                                lines.append(msg)
                                yield "\n".join(lines)

                        pid_install_btn.click(_install_pid,
                                              outputs=[pid_install_log])

                with gr.Accordion("🔍 Hires fix (upscale intégré)", open=False):
                    gr.Markdown(
                        "Agrandit **pendant** la génération : rend à la base, "
                        "agrandit le latent, puis **raffine en 2e passe**. Plus "
                        "cohérent qu'un upscale post-hoc (moins de « collage »), "
                        "mais plus lent et plus gourmand en VRAM.  \n"
                        "⚠️ La **force de débruitage** contrôle l'invention de "
                        "détail : ~0.3 = fidèle, ~0.6 = créatif.")
                    hires_on = gr.Checkbox(
                        value=False, label="Activer le hires fix")
                    with gr.Row():
                        hires_scale = gr.Slider(
                            1.0, 4.0, value=2.0, step=0.5,
                            label="Facteur d'agrandissement (×)")
                        hires_denoise = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="Force de débruitage (raffinage)")

                with gr.Accordion("📂 Fichiers locaux (modèle perso)", open=False):
                    gr.Markdown(t(
                        "Pour utiliser un modèle **téléchargé ailleurs** : déposez "
                        "le(s) fichier(s) dans `{dir}` puis "
                        "sélectionnez-le ci-dessous. Vide = modèle du catalogue.")
                        .format(dir=settings.CUSTOM_DIR))
                    custom_diff = gr.Dropdown(gen_engine.list_custom_models(),
                                              value=None, label="Diffusion (local)")
                    with gr.Row():
                        custom_vae = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="VAE (local)")
                        custom_enc = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="Encodeur (local)")
                    with gr.Row():
                        refresh_custom = gr.Button("↻ Rafraîchir les fichiers locaux",
                                                   size="sm")
                        clear_custom = gr.Button("✖ Vider les champs perso",
                                                 size="sm")

                ratio = gr.Dropdown(
                    [t(k) for k in ratios],
                    value=t(_ratio_label(ratios, d.get("width", 1024),
                                         d.get("height", 1024))),
                    label="Format (ratio)")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=d.get("width", 1024), step=16,
                                      label="Largeur")
                    height = gr.Slider(256, 2048, value=d.get("height", 1024), step=16,
                                       label="Hauteur")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=d.get("steps", 8), step=1,
                                      label="Étapes")
                    cfg = gr.Slider(0.0, 12.0, value=d.get("cfg_scale", 1.0),
                                    step=0.1, label="CFG",
                                    info="Sur sd.cpp, CFG désactivé = 1.0 (normal "
                                         "pour les modèles distillés). 0.0 = pur "
                                         "inconditionnel : peut IGNORER le prompt "
                                         "(la « cfg 0 » de Krea = sa convention "
                                         "maison, ≠ sd.cpp). >1 = guidage.")
                preset_list = _presets(model_id)
                preset = gr.Dropdown(
                    [t(p["name"]) for p in preset_list],
                    value=(t(preset_list[0]["name"]) if preset_list else None),
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
                    label="Flow shift",
                    info="Laissez 0 (auto) : le modèle choisit la bonne valeur "
                         "selon la résolution. Une valeur trop basse (1–2) laisse "
                         "du GRAIN/bruit en haute résolution ; ~3–4 renforce la "
                         "structure.")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)",
                                     precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                with gr.Row():
                    run = gr.Button(t("✨ Générer ({title})").format(title=title),
                                    variant="primary", size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)

            # ----- Sorties (aperçu temps réel ET résultats fusionnés) -----
            with gr.Column(scale=4):
                gallery = gr.Gallery(
                    label="Aperçu temps réel → résultats (légende = seed)",
                    columns=2, height=560, object_fit="contain", show_label=True,
                    format="png", show_download_button=True)
                with gr.Row():
                    seed_box = gr.Textbox(label="Seed de l'image sélectionnée",
                                          interactive=False, show_copy_button=True,
                                          scale=2)
                    seed_reuse = gr.Button("♻️ Réutiliser ce seed", size="sm",
                                           scale=1)
                send_tool = gr.Dropdown(
                    [(t("🌐 Profondeur"), "depth"),
                     (t("✂️ Sans arrière-plan"), "bg"),
                     (t("🪄 Détourer un objet (SAM)"), "sam"),
                     (t("🔼 Agrandir (ESRGAN)"), "esrgan"),
                     (t("✨ Upscale créatif (SDXL)"), "creative")],
                    value=None, label="📤 Envoyer la sélection vers le Toolkit",
                    visible=pending_toolkit is not None)
                logbox = gr.Textbox(label="Journal", lines=10, max_lines=24,
                                    autoscroll=True, elem_classes="log-box")

        last_paths = gr.State([])
        last_seeds = gr.State([])
        sel_index = gr.State(0)

        # ----- Comportements -----
        def refresh_loras():
            choices = gen_engine.list_loras()
            # En mode serveur, on prévient aussi le serveur résident du nouveau
            # fichier (sinon il ne le voit qu'au prochain redémarrage).
            msg = gen_engine.refresh_server_loras()
            return (gr.update(choices=choices), gr.update(choices=choices),
                    msg or "")

        refresh_lora.click(refresh_loras, outputs=[lora1, lora2, civitai_msg])

        def clear_loras():
            return (gr.update(value=None), gr.update(value=0.8),
                    gr.update(value=None), gr.update(value=0.8))

        clear_lora.click(clear_loras, outputs=[lora1, lora1_w, lora2, lora2_w])

        def _civitai_import(ref):
            if not (ref or "").strip():
                raise gr.Error(t("Collez une URL ou un ID de version Civitai."))
            try:
                name = downloader.download_lora_civitai(ref)
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(str(exc))
            choices = gen_engine.list_loras()
            return (gr.update(choices=choices), gr.update(choices=choices),
                    t("✓ LoRA importé : **{name}** — sélectionnez-le ci-dessus."
                      ).format(name=name))

        civitai_btn.click(_civitai_import, inputs=[civitai_ref],
                          outputs=[lora1, lora2, civitai_msg])

        def _refresh_custom():
            c = gen_engine.list_custom_models()
            return gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        refresh_custom.click(_refresh_custom,
                             outputs=[custom_diff, custom_vae, custom_enc])

        def _clear_custom():
            return gr.update(value=None), gr.update(value=None), gr.update(value=None)

        clear_custom.click(_clear_custom,
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
            w, h = ratios.get(i18n.to_source(label), (0, 0))
            if not w:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        ratio.change(on_ratio, inputs=[ratio], outputs=[width, height])

        # --- Améliorateur de prompt (LLM) ---
        # System prompt adapté au modèle : Krea 2 -> guide Krea ; sinon générique.
        _enh_style = "krea2" if family == "krea2" else "generic"

        def _enhance(text, level):
            try:
                better = tools.enhance_prompt(text or "", style=_enh_style,
                                              level=level or "medium")
            except tools.ToolError as exc:
                raise gr.Error(str(exc))
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(f"Échec de l'amélioration : {exc}")
            return gr.update(value=better)

        enhance_btn.click(_enhance, inputs=[prompt, enh_level], outputs=[prompt])

        def _fit_to_ref(img):
            """img2img : cale la sortie sur le format de l'image de départ
            (côté long plafonné à 1024 px, multiples de 16)."""
            if img is None:
                return gr.update(), gr.update(), gr.update()
            w0, h0 = img.size
            longest = max(w0, h0) or 1
            sc = 1024 / longest if longest > 1024 else 1.0
            w = max(256, min(2048, int(round(w0 * sc / 16)) * 16))
            h = max(256, min(2048, int(round(h0 * sc / 16)) * 16))
            return (gr.update(value=w), gr.update(value=h),
                    gr.update(value=t(_CUSTOM_LABEL)))

        init_image.upload(_fit_to_ref, inputs=[init_image],
                          outputs=[width, height, ratio])

        if preset_list:
            def apply_preset(name):
                name = i18n.to_source(name)
                for p in preset_list:
                    if p["name"] == name:
                        return (gr.update(value=p.get("sampler", "euler")),
                                gr.update(value=p.get("scheduler", "auto")),
                                gr.update(value=int(p.get("steps", 8))),
                                gr.update(value=float(p.get("cfg_scale", 1.0))))
                return gr.update(), gr.update(), gr.update(), gr.update()

            preset.change(apply_preset, inputs=[preset],
                          outputs=[sampler, schedule, steps, cfg])

        def do_generate(system_prompt, prompt, negative, init_image, ref_image2,
                        ref_image3, strength, outpaint,
                        width, height, steps, cfg, sampler, schedule, flow_shift,
                        seed, batch, lora1, lora1_w, lora2, lora2_w,
                        custom_diff, custom_vae, custom_enc, pid_hires,
                        hires_on, hires_scale, hires_denoise,
                        progress=gr.Progress()):
            import queue
            import threading
            import time

            if not (prompt or "").strip():
                raise gr.Error(t("Saisissez un prompt (décrivez l'image, ou la "
                                 "modification à appliquer)."))

            full_prompt = prompt or ""
            if (system_prompt or "").strip():
                full_prompt = f"{system_prompt.strip()}, {full_prompt}".strip(", ")

            try:
                base_seed = int(seed)
            except (TypeError, ValueError):   # champ vidé -> aléatoire
                base_seed = -1
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            settings.ensure_dirs()
            # Modèle d'édition (Flux.2) -> image(s) via -r (ref_image) ; sinon
            # img2img classique via -i (init_image) + force.
            init_path = None
            ref_paths: list = []
            of = float(outpaint) if outpaint else 1.0
            if is_edit and init_image is not None and of > 1.01:
                # Outpaint : on agrandit la toile et on demande au modèle de
                # remplir les bords. Fond = image étirée+floutée (continuation
                # plausible), image d'origine recollée au centre.
                from PIL import Image as _PI, ImageFilter as _IF
                ow, oh = init_image.size
                nw = max(256, min(2048, int(round(ow * of / 16)) * 16))
                nh = max(256, min(2048, int(round(oh * of / 16)) * 16))
                bg = init_image.convert("RGB").resize((nw, nh), _PI.LANCZOS)
                bg = bg.filter(_IF.GaussianBlur(28))
                bg.paste(init_image.convert("RGB"), ((nw - ow) // 2, (nh - oh) // 2))
                rp = settings.TMP_DIR / "outpaint_ref.png"
                bg.save(rp)
                ref_paths.append(rp)
                width, height = nw, nh
                full_prompt = (full_prompt + ", extend and naturally fill the "
                               "outer borders to complete the scene seamlessly, "
                               "matching perspective, lighting and content").strip(", ")
            elif is_edit:
                # Édition multi-référence : jusqu'à 3 images (-r répété).
                for i, im in enumerate((init_image, ref_image2, ref_image3)):
                    if im is not None:
                        rp = settings.TMP_DIR / f"edit_ref{i + 1}.png"
                        im.save(rp)
                        ref_paths.append(rp)
            elif init_image is not None:
                init_path = settings.TMP_DIR / "i2i_init.png"
                init_image.save(init_path)

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
                        init_image=init_path, strength=float(strength),
                        ref_image=(ref_paths or None), loras=loras,
                        diffusion_override=gen_engine.custom_path(custom_diff),
                        vae_override=gen_engine.custom_path(custom_vae),
                        encoder_override=gen_engine.custom_path(custom_enc),
                        hires=bool(hires_on), hires_scale=float(hires_scale),
                        hires_denoise=float(hires_denoise),
                        preview_path=preview_path, log=q.put)
                    state["outs"] = [str(p) for p in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            last_mtime = None
            last_emit = 0.0
            progress(0.02, desc="Chargement du modèle…")
            while True:
                try:
                    line = q.get(timeout=0.3)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    logs.append(line)
                    mt = step_re.search(line)
                    if mt:
                        cur = min(int(mt.group(1)), total)
                        progress(0.05 + 0.9 * cur / total,
                                 desc=f"étape {cur}/{total}")
                # Aperçu fusionné dans la galerie : nouvelle frame SEULEMENT si le
                # fichier a changé (mtime). On lit en mémoire (copie PIL) car sous
                # Windows sd-cli écrit ce fichier en continu (verrou en écriture).
                gal = gr.update()
                new_prev = False
                if preview_path.exists():
                    try:
                        m = preview_path.stat().st_mtime
                        if m != last_mtime:
                            from PIL import Image
                            with Image.open(preview_path) as _pim:
                                gal = [(_pim.copy(), t("aperçu en cours…"))]
                            last_mtime = m
                            new_prev = True
                    except (OSError, ValueError):
                        pass
                # On émet tout de suite pour une nouvelle frame d'aperçu, sinon on
                # throttle le journal à ~2x/s : au démarrage sd-cli crache beaucoup
                # de lignes -> évite un re-rendu permanent qui fait « clignoter ».
                now = time.time()
                if new_prev or (now - last_emit) >= 0.5:
                    last_emit = now
                    yield gal, "\n".join(logs[-400:]), gr.update(), gr.update()

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                # On garde la dernière frame d'aperçu (pas de flash vers le vide).
                yield gr.update(), "\n".join(logs), gr.update(), gr.update()
                return
            paths = state.get("outs", [])

            # Décodeur de sortie PiD (optionnel) : chaque image générée est
            # décodée/agrandie ×4 par diffusion pixel → sortie haute résolution
            # directe. Repli sur l'image de base en cas d'absence/erreur.
            if pid_hires and paths:
                if not registry.pid_ready():
                    logs.append("\n⚠️ PiD non installé — sortie normale conservée "
                                "(dépliez « Installer PiD » pour l'obtenir).")
                else:
                    hi: list[str] = []
                    for i, p in enumerate(paths):
                        logs.append(f"\n🚀 Décodage PiD ×4 — image "
                                    f"{i + 1}/{len(paths)}…")
                        progress(0.95, desc=f"PiD ×4 {i + 1}/{len(paths)}")
                        yield (gr.update(), "\n".join(logs[-400:]),
                               gr.update(), gr.update())
                        try:
                            out = gen_engine.pid_decode(p, prompt=full_prompt,
                                                        log=logs.append)
                            hi.append(str(out))
                        except Exception as exc:  # noqa: BLE001
                            logs.append(f"[ERREUR PiD] {exc} — image de base "
                                        "conservée.")
                            hi.append(p)
                    paths = hi

            progress(1.0, desc="Terminé")
            seeds = [base_seed + i for i in range(len(paths))]
            items = [(p, f"seed {s}") for p, s in zip(paths, seeds)]
            yield items, "\n".join(logs), paths, seeds

        gen_evt = run.click(
            do_generate,
            inputs=[system_prompt, prompt, negative, init_image, ref_image2,
                    ref_image3, strength, outpaint, width,
                    height, steps, cfg, sampler, schedule, flow_shift, seed, batch,
                    lora1, lora1_w, lora2, lora2_w,
                    custom_diff, custom_vae, custom_enc, pid_hires,
                    hires_on, hires_scale, hires_denoise],
            outputs=[gallery, logbox, last_paths, last_seeds],
        )
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[gen_evt])

        # --- Seed : vidé -> -1 ; sélection -> affichage copiable ; réutiliser ---
        def _seed_default(v):
            return -1 if v in (None, "") else gr.update()

        seed.change(_seed_default, inputs=[seed], outputs=[seed])

        def _on_select(seeds, evt: gr.SelectData):
            i = evt.index if isinstance(evt.index, int) else 0
            s = seeds[i] if (seeds and 0 <= i < len(seeds)) else ""
            return i, str(s)

        gallery.select(_on_select, inputs=[last_seeds],
                       outputs=[sel_index, seed_box])

        def _reuse_seed(seeds, idx):
            if seeds and isinstance(idx, int) and 0 <= idx < len(seeds):
                return gr.update(value=int(seeds[idx]))
            return gr.update()

        seed_reuse.click(_reuse_seed, inputs=[last_seeds, sel_index],
                         outputs=[seed])

        if pending_toolkit is not None and tabs is not None:
            def _send_toolkit(paths, idx, dest):
                if not paths or not dest:
                    raise gr.Error(t("Générez puis sélectionnez une image."))
                i = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
                return ((paths[i], dest), gr.Tabs(selected=toolkit_tab_id),
                        gr.update(value=None))

            send_tool.change(
                _send_toolkit, inputs=[last_paths, sel_index, send_tool],
                outputs=[pending_toolkit, tabs, send_tool])
