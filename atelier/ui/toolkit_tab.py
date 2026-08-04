"""Onglet Toolkit : profondeur, détourage, SAM et upscale ESRGAN."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings
from ..engine import generate as gen_engine
from ..engine import tools
from ..i18n import t

# Préréglages de l'upscale créatif.
#
# Un préréglage ne se réduit PAS à « un prompt + une créativité » : sur du
# dessin, ce qui décide de la propreté du résultat est ailleurs — le
# pré-agrandissement (Lanczos interpole, un ESRGAN dessin non), le négatif (le
# défaut est orienté photo et fait poser du grain sur les aplats), le CFG et le
# verrouillage de structure. D'où un dictionnaire d'options par préréglage, dont
# tous les champs sont facultatifs.
#
# Champs : prompt · negative · denoise · cfg · steps · controlnet · cn_scale ·
#          esrgan ("drawing" = choisir automatiquement un modèle dessin installé)
UPSCALE_PRESETS = [
    {"name": "🔍 Net & fidèle (aucun ajout)",
     "prompt": "sharp focus, clean precise detail, faithful to the original, "
               "no added elements, high fidelity",
     "denoise": 0.20},
    {"name": "✨ Ajouter du détail",
     "prompt": "highly detailed, intricate fine textures, crisp micro-detail, "
               "enhanced clarity, sharp focus",
     "denoise": 0.40},
    {"name": "🧴 Peau réaliste (portrait)",
     "prompt": "highly detailed realistic skin with fine pores, natural "
               "complexion, sharp eyes and individual hair strands, "
               "true-to-life photographic detail",
     "denoise": 0.35},
    {"name": "🌿 Nature / paysage",
     "prompt": "crisp natural textures, detailed foliage and rock, fine "
               "vegetation, clear sharp landscape detail",
     "denoise": 0.40},
    {"name": "🏙️ Architecture / produit",
     "prompt": "clean sharp edges, precise material textures, accurate "
               "reflections, crisp surface detail",
     "denoise": 0.30},

    # ---- Dessin : le préréglage qui traite VRAIMENT le problème -------------
    # Sur une illustration, un upscale « photo » fait trois dégâts :
    #  1. la base Lanczos interpole -> traits mous, aplats baveux ;
    #  2. le négatif par défaut ne défend pas les aplats -> SDXL y pose du grain
    #     et de la matière « photo » ;
    #  3. un débruitage à 0,40 redessine le trait, qui se met à onduler.
    # On corrige les trois : ESRGAN dessin en base (agrandissement réel, pas une
    # interpolation), négatif anti-photo/anti-grain, débruitage bas et structure
    # verrouillée par ControlNet — SDXL ne fait plus que nettoyer.
    {"name": "🖍️ Illustration / BD — trait net, sans interpolation",
     "prompt": "clean crisp linework, flat solid color areas, smooth even "
               "fills, sharp precise edges, clean vector-like illustration, "
               "no texture on flat colors",
     "negative": "photorealistic, photo texture, film grain, noise, gradient "
                 "banding, jpeg artifacts, blurry soft edges, halo, ringing, "
                 "oversharpened, painterly brush texture on flat areas, "
                 "3d render, deformed lines",
     "denoise": 0.18, "cfg": 4.0, "steps": 20,
     "controlnet": True, "cn_scale": 0.85, "esrgan": "drawing"},
    {"name": "🎨 Illustration peinte / concept art",
     "prompt": "crisp clean brushwork, refined shapes, vivid consistent "
               "colors, sharp stylized detail",
     "negative": "photorealistic, film grain, noise, jpeg artifacts, "
                 "blurry, oversharpened, halo",
     "denoise": 0.35, "cfg": 5.0,
     "controlnet": True, "cn_scale": 0.7, "esrgan": "drawing"},

    {"name": "🚀 Détail maximum (créatif)",
     "prompt": "ultra detailed, hyper-detailed intricate surfaces, rich fine "
               "texture everywhere, razor sharp",
     "denoise": 0.55},
    {"name": "🪶 Doux & propre (anti-grain)",
     "prompt": "clean smooth surfaces, gently denoised, soft natural detail, "
               "no artifacts, no grain",
     "denoise": 0.25},
]


def _installer_block(title: str, note: str, stream_fn, installed: bool):
    """Accordéon d'installation 1 clic commun aux outils."""
    with gr.Accordion(t("⚙️ Installer {title} (en 1 clic)").format(title=title),
                      open=not installed):
        gr.Markdown(note)
        log = gr.Textbox(label="Journal d'installation", lines=10,
                         autoscroll=True, elem_classes="log-box")
        btn = gr.Button(t("⬇️ Installer {title}").format(title=title))

        def _install():
            for msg in stream_fn():
                yield msg

        btn.click(_install, outputs=[log])


def build_toolkit_tab(tab_id="toolkit", pending_toolkit=None, tabs=None):
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Outils utilitaires\n"
            "Carte de **profondeur**, **suppression d'arrière-plan** (PNG "
            "transparent), **détourage d'objet au clic** (Segment Anything) et "
            "**agrandissement ESRGAN** (simple, 100% GPU).")

        with gr.Tabs() as sub_tabs:
            # ---------- Profondeur ----------
            with gr.Tab("🌐 Profondeur", id="depth"):
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
                        raise gr.Error(t("Fournissez une image."))
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
            with gr.Tab("✂️ Sans arrière-plan", id="bg"):
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
                        raise gr.Error(t("Fournissez une image."))
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
            with gr.Tab("🪄 Détourer un objet (SAM)", id="sam"):
                gr.Markdown(
                    "*Segment Anything* — **cliquez sur un objet** : SAM affiche "
                    "aussitôt la **zone sélectionnée en surbrillance**. Ajustez en "
                    "recliquant, puis « Extraire » pour le **PNG transparent**.")
                _installer_block(
                    "Segment Anything",
                    "PyTorch + transformers (~375 Mo, facebook/sam-vit-base). "
                    "Aucune commande à taper.",
                    tools.install_sam_stream, tools.sam_is_installed())

                s_cut = gr.State(None)     # chemin du découpage déjà calculé
                with gr.Row():
                    with gr.Column(scale=3):
                        s_image = gr.Image(label="Image — cliquez sur l'objet",
                                           type="pil")
                        s_info = gr.Markdown("Cliquez un point sur l'image.")
                        s_overlay = gr.Image(label="Zone sélectionnée (aperçu)",
                                             height=300, interactive=False)
                        s_run = gr.Button("🪄 Extraire l'objet", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        s_result = gr.Image(label="Objet extrait (PNG transparent)",
                                            height=520, format="png",
                                            show_download_button=True,
                                            image_mode="RGBA")
                        s_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def _on_click(img, evt: gr.SelectData, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Fournissez une image."))
                    if not tools.sam_is_installed():
                        raise gr.Error(t("Segment Anything n'est pas installé "
                                         "(bouton « Installer » ci-dessus)."))
                    x, y = int(evt.index[0]), int(evt.index[1])
                    progress(0.2, desc="Segmentation…")
                    try:
                        cut, overlay = tools.sam_segment(img, x, y)
                    except Exception as exc:  # noqa: BLE001
                        raise gr.Error(str(exc))
                    progress(1.0, desc="Terminé")
                    return (overlay or gr.update(), str(cut),
                            t("Zone sélectionnée en ({x}, {y}). Cliquez "
                              "« Extraire » ou recliquez ailleurs.").format(x=x, y=y))

                s_image.select(_on_click, inputs=[s_image],
                               outputs=[s_overlay, s_cut, s_info])

                def do_sam(cut):
                    if not cut:
                        raise gr.Error(t("Cliquez d'abord sur un objet dans l'image."))
                    return str(cut), f"✅ {cut}"

                s_run.click(do_sam, inputs=[s_cut], outputs=[s_result, s_log])

            # ---------- Agrandir (ESRGAN, sd.cpp) ----------
            with gr.Tab("🔼 Agrandir (ESRGAN)", id="esrgan"):
                gr.Markdown(
                    "Agrandissement **simple** par réseau ESRGAN, natif "
                    "**sd.cpp** : déterministe, **100% GPU**, aucun PyTorch ni "
                    "prompt. Le facteur (×2 ou ×4) dépend du modèle choisi ; "
                    "« Répéter » ré-applique le modèle (×2 deux fois = ×4).\n\n"
                    "🎨 **BD, illustration, dessin au trait** : prends un modèle "
                    "marqué **dessin / anime**. Les modèles photo (Remacri, "
                    "Nomos, UltraSharp…) sont entraînés sur des textures "
                    "naturelles : sur un aplat ils inventent du grain, et sur un "
                    "trait net ils posent un halo. C'est ça, l'« interpolation "
                    "dégueulasse ».\n\n"
                    "📥 **Ajouter tes propres modèles** : dépose un fichier "
                    "`.pth`, `.safetensors` ou `.gguf` dans le dossier des "
                    "upscalers, puis « ↻ Rafraîchir ». sd.cpp lit la plupart des "
                    "`.pth` directement — tout le catalogue "
                    "[OpenModelDB](https://openmodeldb.info) est donc "
                    "utilisable ; filtre-le sur *anime* / *manga* / *cartoon*. "
                    "Le GGUF charge plus vite et évite d'exécuter un pickle, "
                    "mais il n'est pas obligatoire.")

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
                            registry.upscaler_choices(),
                            value=registry.default_upscaler(),
                            label="Modèle d'upscale (×2 / ×4 selon le nom)",
                            info="🎨 = entraîné pour le DESSIN (trait net, "
                                 "aplats propres) · 📷 = photo. Sur une planche "
                                 "de BD, un modèle photo bave et pose des halos.")
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
                           gr.update(choices=registry.upscaler_choices(),
                                     value=registry.default_upscaler()))

                u_inst.click(_install_upscalers, outputs=[u_inst_log, u_model])

                def _refresh_upscalers():
                    return gr.update(choices=registry.upscaler_choices(),
                                     value=registry.default_upscaler())

                u_refresh.click(_refresh_upscalers, outputs=[u_model])

                def do_upscale(img, model, repeats, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Fournissez une image."))
                    if not model:
                        raise gr.Error(t("Choisissez un modèle d'upscale "
                                         "(téléchargez-les d'abord)."))
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

            # ---------- Restauration SeedVR2 (1 pas, sans prompt) ----------
            with gr.Tab("🎯 Restaurer (SeedVR2)", id="seedvr2"):
                gr.Markdown(
                    "Agrandissement par **restauration** : SeedVR2 reconstruit "
                    "le détail réellement plausible (peau, tissu, feuillage, "
                    "texte) au lieu de lisser comme ESRGAN ou de réinventer "
                    "comme l'upscale créatif. **Un seul pas de diffusion, sans "
                    "prompt** — c'est presque instantané et il n'y a rien à "
                    "régler.\n\n"
                    "Zone de confort : **×2 à ×4**. Au-delà, la qualité "
                    "décroche.")

                with gr.Accordion("⬇️ Installer SeedVR2 (1 clic)",
                                  open=not tools.seedvr2_is_installed()):
                    gr.Markdown(
                        "Récupère le **code d'inférence** (PyTorch pur : ni "
                        "apex, ni flash-attn, ni compilation) et les **poids "
                        "1.4B** (~2,9 Go). Environ **4,6 Go de VRAM** au pic "
                        "pour un 512→2048.\n\n"
                        "ℹ️ Le code vient du dépôt de référence SeedVR2, qui est "
                        "distribué comme nœud ComfyUI mais fournit un "
                        "`inference_cli.py` officiellement prévu pour tourner "
                        "**sans ComfyUI** : on n'installe ni ne lance ComfyUI, "
                        "on appelle ce script en sous-process. Le **VAE "
                        "(~0,5 Go)** est téléchargé automatiquement au tout "
                        "premier agrandissement.")
                    v_inst_log = gr.Textbox(label="Journal d'installation",
                                            lines=8, autoscroll=True,
                                            elem_classes="log-box")
                    v_inst = gr.Button("⬇️ Installer SeedVR2")

                with gr.Row():
                    with gr.Column(scale=3):
                        v_image = gr.Image(label="Image à restaurer", type="pil")
                        v_res = gr.Slider(
                            512, 2560, value=1080, step=64,
                            label="Côté court visé (px)",
                            info="Le rapport d'aspect est conservé. Visez 2 à "
                                 "4 fois le côté court de votre image. Plus haut "
                                 "= plus de VRAM : au-delà de ~1440 px sur "
                                 "11–12 Go, gardez le VAE par tuiles activé.")
                        v_model = gr.Dropdown(
                            tools.seedvr2_models(),
                            value=tools.seedvr2_default_model(),
                            label="Poids utilisés")
                        with gr.Accordion("Réglages avancés", open=False):
                            v_color = gr.Dropdown(
                                ["lab", "wavelet", "wavelet_adaptive", "hsv",
                                 "adain", "none"], value="lab",
                                label="Correction colorimétrique",
                                info="Recale les couleurs sur l'original. "
                                     "« lab » convient presque toujours.")
                            v_tiled = gr.Checkbox(
                                value=True, label="VAE par tuiles (recommandé)",
                                info="Découpe l'encodage et le décodage. Sur "
                                     "11–12 Go, le décocher fait déborder la "
                                     "VRAM dès ~1440 px. Ne le décochez que si "
                                     "vous visez petit.")
                            v_tile = gr.Slider(
                                256, 512, value=512, step=128,
                                label="Taille de tuile",
                                info="512 par défaut. Le VAE s'auto-attentionne "
                                     "sur la tuile entière : le coût explose en "
                                     "O(n²), c'est pourquoi on ne monte pas "
                                     "au-dessus. Descendez à 256 si ça déborde "
                                     "encore.")
                            v_seed = gr.Number(value=42, precision=0,
                                               label="Seed")
                        with gr.Row():
                            v_refresh = gr.Button("↻ Rafraîchir", size="sm")
                            v_run = gr.Button("🎯 Restaurer", variant="primary",
                                              size="lg", scale=2)
                        v_stop = gr.Button("⏹️ Annuler", variant="stop",
                                           size="sm")
                    with gr.Column(scale=4):
                        v_result = gr.Image(
                            label="Résultat (pleine résolution dans outputs/)",
                            height=520, format="png", show_download_button=True)
                        v_log = gr.Textbox(label="Journal", lines=10,
                                           autoscroll=True,
                                           elem_classes="log-box")

                def _install_seedvr2():
                    for msg in tools.install_seedvr2_stream():
                        yield msg, gr.update()
                    yield gr.update(), gr.update(
                        choices=tools.seedvr2_models(),
                        value=tools.seedvr2_default_model())

                v_inst.click(_install_seedvr2, outputs=[v_inst_log, v_model])

                def _refresh_seedvr2():
                    return gr.update(choices=tools.seedvr2_models(),
                                     value=tools.seedvr2_default_model())

                v_refresh.click(_refresh_seedvr2, outputs=[v_model])

                def do_seedvr2(img, res, model, color, tiled, tile, seed,
                               progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Fournissez une image."))
                    logs: list[str] = []
                    progress(0.1, desc="Restauration…")
                    try:
                        out = tools.seedvr2_upscale(
                            img, resolution=int(res), model=model or None,
                            seed=int(seed), color_correction=color,
                            tiled=bool(tiled), tile_size=int(tile),
                            log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    logs.append(f"\n✅ Image restaurée : {out}")
                    return str(out), "\n".join(logs)

                v_evt = v_run.click(
                    do_seedvr2,
                    inputs=[v_image, v_res, v_model, v_color, v_tiled, v_tile,
                            v_seed],
                    outputs=[v_result, v_log])
                v_stop.click(lambda: tools.cancel(), outputs=None,
                             cancels=[v_evt])

            # ---------- Upscale créatif tuilé (SDXL, façon Magnific) ----------
            with gr.Tab("✨ Upscale créatif (SDXL)", id="creative"):
                gr.Markdown(
                    "Upscale **créatif** « Ultimate SD Upscale » : pré-agrandit "
                    "puis **raffine tuile par tuile** en SDXL img2img à faible "
                    "débruitage (modèle **résident** → tuiles rapides, fondu par "
                    "recouvrement). Invente du détail fin façon Magnific. "
                    "**100% GPU** (PyTorch).")
                _installer_block(
                    "Upscale créatif SDXL",
                    "PyTorch + diffusers (~9,5 Go : SDXL base + VAE fp16-fix + "
                    "ControlNet Tile). Modèle résident sur le GPU. Aucune "
                    "commande à taper.",
                    tools.install_upscale_stream, tools.upscale_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        c_image = gr.Image(label="Image à agrandir", type="pil")
                        _ckpts = tools.list_upscale_checkpoints()
                        c_model = gr.Dropdown(
                            choices=_ckpts,
                            value=(_ckpts[0][1] if _ckpts else None),
                            label="Modèle SDXL (déposez vos .safetensors dans "
                                  "tools_repo/upscale/checkpoints/)")
                        c_vae = gr.Radio(
                            [(t("VAE fp16-fix (externe, recommandé)"), False),
                             (t("VAE intégrée au modèle"), True)],
                            value=False, label="VAE")
                        _ups = registry.list_upscalers()
                        c_esrgan = gr.Dropdown(
                            choices=[(t("Lanczos (par défaut)"), "")]
                                    + [(u, u) for u in _ups],
                            value="", label="Pré-agrandissement (base avant SDXL)")
                        c_refresh = gr.Button("↻ Rafraîchir les modèles", size="sm")
                        c_preset = gr.Dropdown(
                            choices=[(t(p["name"]), p["name"])
                                     for p in UPSCALE_PRESETS],
                            value=None,
                            label="Préréglage (règle prompt, négatif, "
                                  "créativité, CFG et structure)")
                        c_preset_msg = gr.Markdown("", elem_classes="feedback")
                        c_prompt = gr.Textbox(
                            label="Prompt (optionnel — guide le détail, COURT : "
                                  "~77 tokens max SDXL ; inutile de recopier le "
                                  "prompt de génération)", lines=2,
                            placeholder="highly detailed skin texture, sharp "
                                        "focus, photorealistic")
                        c_negative = gr.Textbox(
                            label="Prompt négatif (vide = défaut orienté photo)",
                            lines=2,
                            placeholder="photorealistic, film grain, noise…",
                            info="Ce qu'on interdit à SDXL d'ajouter. Sur du "
                                 "dessin, c'est ce qui empêche le grain et la "
                                 "matière photo de se poser sur les aplats.")
                        c_scale = gr.Slider(
                            1.5, 8.0, value=2.0, step=0.5,
                            label="Facteur d'agrandissement",
                            info="Jusqu'à ~8K (plafonné à 8192 px). ×6–×8 = "
                                 "beaucoup de tuiles : très long + ~1–2 Go de RAM.")
                        c_denoise = gr.Slider(
                            0.15, 0.75, value=0.35, step=0.05,
                            label="Créativité (débruitage — ↑ = détail inventé)")
                        _cn_ok = tools.upscale_cn_is_installed()
                        c_controlnet = gr.Checkbox(
                            value=_cn_ok,
                            label="🔒 ControlNet Tile (verrouille la structure — "
                                  "permet de monter la créativité sans dériver)")
                        if not _cn_ok:
                            gr.Markdown(
                                "> ℹ️ ControlNet **pas encore téléchargé** : "
                                "relancez « Installer l'upscale créatif SDXL » "
                                "ci-dessus (ajoute ~2,5 Go) puis **redémarrez** "
                                "pour activer le verrouillage de structure.")
                        c_cnscale = gr.Slider(
                            0.2, 1.0, value=0.6, step=0.05,
                            label="Fidélité ControlNet (↑ = plus fidèle)")
                        with gr.Row():
                            c_steps = gr.Slider(10, 40, value=24, step=1,
                                                label="Pas / tuile")
                            c_cfg = gr.Slider(1.0, 12.0, value=6.0, step=0.5,
                                              label="CFG")
                        c_tile = gr.Slider(640, 1280, value=1024, step=64,
                                           label="Taille de tuile")
                        with gr.Row():
                            c_run = gr.Button("✨ Upscaler", variant="primary",
                                              size="lg", scale=2)
                            c_stop = gr.Button("⏹️ Annuler", variant="stop",
                                               size="sm")
                    with gr.Column(scale=4):
                        c_result = gr.Image(
                            label="Aperçu temps réel (pleine résolution dans "
                                  "outputs/)", height=520, format="png",
                            show_download_button=True)
                        c_log = gr.Textbox(label="Journal", lines=12,
                                           autoscroll=True, elem_classes="log-box")

                def _apply_preset(name):
                    """Applique TOUT le préréglage et dit ce qu'il a changé.

                    Un préréglage qui règle six choses en silence est
                    indéfendable : on affiche donc, en clair, ce qui vient
                    d'être posé — et notamment quel modèle de pré-agrandissement
                    a été choisi, puisque c'est lui qui fait le gros du travail
                    sur du dessin."""
                    pre = next((p for p in UPSCALE_PRESETS
                                if p["name"] == name), None)
                    if pre is None:
                        return ((gr.update(),) * 8) + (gr.update(value=""),)

                    done: list[str] = []
                    esrgan_up = gr.update()
                    want = pre.get("esrgan")
                    if want == "drawing":
                        model = registry.drawing_upscaler()
                        if model:
                            esrgan_up = gr.update(value=model)
                            done.append(t("pré-agrandissement **{m}** (dessin) "
                                          "au lieu de Lanczos").format(m=model))
                        else:
                            done.append(t("⚠️ aucun upscaler **dessin** "
                                          "installé — la base restera en "
                                          "Lanczos (traits plus mous). "
                                          "Téléchargez les upscalers dans "
                                          "l'onglet « 🔼 Agrandir »."))
                    elif want:
                        esrgan_up = gr.update(value=want)

                    cn = pre.get("controlnet")
                    cn_up = gr.update()
                    if cn is not None:
                        # ControlNet ne peut être coché que s'il est téléchargé.
                        cn = bool(cn) and tools.upscale_cn_is_installed()
                        cn_up = gr.update(value=cn)
                        if pre.get("controlnet") and not cn:
                            done.append(t("⚠️ ControlNet Tile pas installé : la "
                                          "structure ne sera pas verrouillée."))
                        elif cn:
                            done.append(t("structure verrouillée par ControlNet "
                                          "Tile ({v})").format(
                                v=pre.get("cn_scale", 0.6)))

                    done.append(t("créativité {d} · CFG {c} · {s} pas").format(
                        d=pre.get("denoise", 0.35), c=pre.get("cfg", 6.0),
                        s=pre.get("steps", 24)))
                    if pre.get("negative"):
                        done.append(t("négatif adapté"))

                    return (gr.update(value=pre.get("prompt", "")),
                            gr.update(value=pre.get("negative", "")),
                            gr.update(value=pre.get("denoise", 0.35)),
                            gr.update(value=pre.get("cfg", 6.0)),
                            gr.update(value=pre.get("steps", 24)),
                            cn_up,
                            gr.update(value=pre.get("cn_scale", 0.6)),
                            esrgan_up,
                            gr.update(value="✅ " + " · ".join(done)))

                c_preset.change(
                    _apply_preset, inputs=[c_preset],
                    outputs=[c_prompt, c_negative, c_denoise, c_cfg, c_steps,
                             c_controlnet, c_cnscale, c_esrgan, c_preset_msg])

                def _refresh_models():
                    ck = tools.list_upscale_checkpoints()
                    ups = registry.list_upscalers()
                    return (gr.update(choices=ck,
                                      value=(ck[0][1] if ck else None)),
                            gr.update(choices=[(t("Lanczos (par défaut)"), "")]
                                              + [(u, u) for u in ups]))

                c_refresh.click(_refresh_models, outputs=[c_model, c_esrgan])

                def do_creative(img, prompt, negative, scale, denoise, steps,
                                cfg, tile, controlnet, cn_scale, model,
                                vae_integrated, esrgan,
                                progress=gr.Progress()):
                    import queue
                    import threading
                    import time
                    from PIL import Image as _PILImage

                    if img is None:
                        raise gr.Error(t("Fournissez une image."))
                    if not tools.upscale_is_installed():
                        raise gr.Error(t("Installez d'abord l'upscale créatif SDXL "
                                         "(accordéon ci-dessus)."))
                    if controlnet and not tools.upscale_cn_is_installed():
                        gr.Warning(t("ControlNet pas installé : upscale sans "
                                     "ControlNet. Relancez l'installateur pour "
                                     "l'activer."))
                        controlnet = False
                    settings.ensure_dirs()
                    preview_path = (settings.TMP_DIR /
                                    f"usdu_preview_{int(time.time()*1000)}.png")
                    try:
                        preview_path.unlink()
                    except OSError:
                        pass
                    q: "queue.Queue[str | None]" = queue.Queue()
                    state: dict = {}

                    def worker():
                        try:
                            out = tools.ultimate_upscale(
                                img, scale=float(scale), prompt=prompt or "",
                                negative=negative or "",
                                denoise=float(denoise), steps=int(steps),
                                cfg=float(cfg), tile=int(tile),
                                use_controlnet=bool(controlnet),
                                cn_scale=float(cn_scale),
                                base_model=(model or None),
                                integrated_vae=bool(vae_integrated),
                                esrgan_model=(esrgan or None),
                                preview_path=preview_path, log=q.put)
                            state["out"] = str(out)
                        except Exception as exc:  # noqa: BLE001
                            state["err"] = str(exc)
                        finally:
                            q.put(None)

                    threading.Thread(target=worker, daemon=True).start()
                    logs: list[str] = []
                    last_mtime = None
                    last_emit = 0.0
                    import re as _re
                    tile_re = _re.compile(r"tuile (\d+)/(\d+)")
                    progress(0.03, desc="Préparation…")
                    while True:
                        try:
                            line = q.get(timeout=0.3)
                        except queue.Empty:
                            line = ""
                        if line is None:
                            break
                        if line:
                            logs.append(line)
                            mt = tile_re.search(line)
                            if mt:
                                cur, tot = int(mt.group(1)), max(1, int(mt.group(2)))
                                progress(0.05 + 0.9 * cur / tot,
                                         desc=f"Tuile {cur}/{tot}")
                        prev = gr.update()
                        new_prev = False
                        if preview_path.exists():
                            try:
                                mt = preview_path.stat().st_mtime
                                if mt != last_mtime:
                                    with _PILImage.open(preview_path) as _p:
                                        prev = _p.copy()
                                    last_mtime = mt
                                    new_prev = True
                            except (OSError, ValueError):
                                pass
                        now = time.time()
                        if new_prev or (line and now - last_emit >= 0.5):
                            last_emit = now
                            yield prev, "\n".join(logs[-400:])

                    if "err" in state:
                        logs.append(f"\n[ERREUR] {state['err']}")
                        yield gr.update(), "\n".join(logs)
                        return
                    progress(1.0, desc="Terminé")
                    out = state.get("out")
                    if out:
                        try:
                            im = _PILImage.open(out)
                            logs.append(f"\n✅ Image pleine résolution "
                                        f"({im.width}x{im.height}) : {out}")
                        except Exception:  # noqa: BLE001
                            pass
                    # On affiche le FICHIER pleine résolution (téléchargement =
                    # image réelle, pas une preview réduite).
                    yield (str(out) if out else gr.update()), "\n".join(logs)

                c_evt = c_run.click(
                    do_creative,
                    inputs=[c_image, c_prompt, c_negative, c_scale, c_denoise,
                            c_steps, c_cfg, c_tile, c_controlnet, c_cnscale,
                            c_model, c_vae, c_esrgan],
                    outputs=[c_result, c_log])
                c_stop.click(lambda: tools.cancel(), outputs=None, cancels=[c_evt])

        # --- Réception d'une image envoyée depuis un onglet de génération ---
        if pending_toolkit is not None and tabs is not None:
            _keys = ["depth", "bg", "sam", "esrgan", "seedvr2", "creative"]

            def _consume(pend):
                if not pend:
                    return tuple([gr.update()] * (len(_keys) + 1) + [None])
                path, dest = pend
                sub = gr.Tabs(selected=dest) if dest in _keys else gr.update()
                img_upd = [gr.update(value=path) if k == dest else gr.update()
                           for k in _keys]
                return tuple([sub] + img_upd + [None])

            tabs.select(_consume, inputs=[pending_toolkit],
                        outputs=[sub_tabs, d_image, b_image, s_image, u_image,
                                 v_image, c_image, pending_toolkit])
