"""Onglet Réglages : matériel détecté, optimisations auto/manuelles, quant."""
from __future__ import annotations

import gradio as gr

from .. import hardware, settings
from ..i18n import t

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]
LANGS = [("Français", "fr"), ("English", "en")]


def _gpu_choices() -> list[tuple[str, int]]:
    return [(f"#{g.index} — {g.label()}", g.index)
            for g in hardware.detect_gpus()]


def _profile_md() -> str:
    prefs = settings.load_prefs()
    prof = hardware.auto_profile(prefs.get("gpu_index"))
    flags = [k for k, v in prof.flags().items() if v]
    lines = [hardware.summary_text(), "",
             t("**Profil automatique :**"),
             t("- Diffusion : `{quant}` · Encodeur : `{enc}`").format(
                 quant=prof.quant, enc=prof.enc_quant),
             t("- Optimisations : `{flags}`").format(
                 flags=", ".join(flags) or t("aucune"))]
    lines += [f"- {n}" for n in prof.notes]
    return "\n".join(lines)


def _gpu_strategy(prefs) -> str:
    """Déduit la stratégie multi-GPU courante depuis les préférences."""
    if prefs.get("auto_fit"):
        return "autofit"
    eg = prefs.get("encoder_gpu_index")
    if eg is not None and eg != prefs.get("gpu_index"):
        return "encoder"
    return "single"


def build_settings_tab():
    with gr.Tab("⚙️ Réglages"):
        gr.Markdown("### Matériel détecté")
        profile_md = gr.Markdown(_profile_md())

        prefs = settings.load_prefs()
        multi_gpu = len(_gpu_choices()) > 1

        # ------------------------------------------------------------------ #
        #  Interface
        # ------------------------------------------------------------------ #
        with gr.Accordion("🌍 Interface (langue, thème)", open=False):
            with gr.Row():
                lang_dd = gr.Dropdown(
                    LANGS, value=prefs.get("lang", "fr"),
                    label="🌐 Langue / Language (redémarrage requis)")
                theme_dd = gr.Dropdown(
                    [(t("Clair"), "light"), (t("Sombre"), "dark")],
                    value=prefs.get("theme", "light"),
                    label="🎨 Thème (redémarrage requis)")
            lang_msg = gr.Markdown("")

        def _save_lang(lang):
            p = settings.load_prefs()
            p["lang"] = lang
            settings.save_prefs(p)
            disp = {v: k for k, v in LANGS}.get(lang, lang)
            return t("✅ Langue enregistrée. **Redémarrez l'application** "
                     "(`run.bat` / `run.sh`) pour appliquer « {lang} »."
                     ).format(lang=disp)

        lang_dd.change(_save_lang, inputs=[lang_dd], outputs=[lang_msg])

        def _save_theme(th):
            p = settings.load_prefs()
            p["theme"] = "dark" if th == "dark" else "light"
            settings.save_prefs(p)
            return t("✅ Thème enregistré. **Redémarrez l'application** pour "
                     "l'appliquer.")

        theme_dd.change(_save_theme, inputs=[theme_dd], outputs=[lang_msg])

        # ------------------------------------------------------------------ #
        #  Optimisation (l'essentiel, visible)
        # ------------------------------------------------------------------ #
        gr.Markdown("### 🎛️ Optimisation")
        with gr.Row():
            auto = gr.Checkbox(value=prefs.get("auto_optimize", True),
                               label="Automatique (recommandé — selon GPU + RAM)")
            gpu = gr.Dropdown(label="GPU de génération",
                              choices=_gpu_choices(),
                              value=prefs.get("gpu_index"))
        gr.Markdown(
            "**Ou : optimiser pour ma carte en 1 clic** — applique quant + offload "
            "+ tiling selon la VRAM réelle (désactive l'automatique).")
        with gr.Row():
            gen_btns = {key: gr.Button(spec["label"], size="sm")
                        for key, spec in hardware.GENERATIONS.items()}

        with gr.Accordion("Réglages manuels avancés (quant + flags)", open=False):
            gr.Markdown("Utilisés uniquement si **l'automatique est décoché**.")
            with gr.Row():
                quant = gr.Dropdown(label="Quant. diffusion (vide = auto)",
                                    choices=["auto"] + QUANTS,
                                    value=prefs.get("quant") or "auto")
                enc_quant = gr.Dropdown(label="Quant. encodeur (vide = auto)",
                                        choices=["auto"] + QUANTS,
                                        value=prefs.get("enc_quant") or "auto")
            f = prefs.get("flags", {})
            with gr.Row():
                fa = gr.Checkbox(value=f.get("diffusion_fa", True), label="Flash attention")
                offload = gr.Checkbox(value=f.get("offload_to_cpu", True), label="Offload CPU")
                tiling = gr.Checkbox(value=f.get("vae_tiling", True), label="VAE tiling")
            with gr.Row():
                clip_cpu = gr.Checkbox(value=f.get("clip_on_cpu", False), label="CLIP sur CPU")
                vae_cpu = gr.Checkbox(value=f.get("vae_on_cpu", False), label="VAE sur CPU")

        # ------------------------------------------------------------------ #
        #  Multi-GPU : UN seul choix (mutuellement exclusif) + carte texte
        # ------------------------------------------------------------------ #
        combo_btn = None
        combo_msg = None
        if multi_gpu:
            with gr.Accordion("🧮 Multi-GPU (2 cartes détectées)", open=False):
                gr.Markdown(
                    "**Une seule stratégie à la fois** (elles s'excluent) :\n"
                    "- **Une seule carte** : tout sur le GPU de génération (offload "
                    "RAM par défaut). Le plus fiable.\n"
                    "- **Encodeur sur la 2e carte** : l'encodeur de texte va sur "
                    "l'autre GPU, la diffusion reste sur le principal.\n"
                    "- **Auto-fit** : sd.cpp répartit diffusion/encodeur/VAE sur "
                    "toutes les cartes. ⚠️ force tout en VRAM (désactive l'offload) "
                    "→ risque d'OOM sur les modèles à gros encodeur (Flux.2 Klein). "
                    "À réserver aux modèles qui tiennent dans la VRAM cumulée.")
                gpu_strategy = gr.Radio(
                    [(t("Une seule carte (recommandé)"), "single"),
                     (t("Encodeur de texte sur la 2e carte"), "encoder"),
                     (t("Auto-fit : répartir sur toutes les cartes"), "autofit")],
                    value=_gpu_strategy(prefs), label="Stratégie multi-GPU")
                tools_gpu = gr.Dropdown(
                    label="GPU pour l'améliorateur de prompt (texte, séparé)",
                    choices=[(t("Auto (même que génération)"), None)]
                            + _gpu_choices(),
                    value=prefs.get("text_gpu_index"))
                combo = hardware.rtx3060_1080ti_combo()
                if combo:
                    combo_btn = gr.Button(
                        "⚡ Appliquer le profil RTX 3060 12 Go + GTX 1080 Ti",
                        variant="primary")
                    combo_msg = gr.Markdown(
                        "La RTX 3060 calcule diffusion/VAE ; la GTX 1080 Ti "
                        "prend l'encodeur et l'améliorateur de prompt. "
                        "Auto-fit et row split restent désactivés.")
        else:
            gpu_strategy = gr.State(_gpu_strategy(prefs))
            tools_gpu = gr.State(prefs.get("text_gpu_index"))

        # ------------------------------------------------------------------ #
        #  Accélération par cache (avancé)
        # ------------------------------------------------------------------ #
        with gr.Accordion("⚡ Accélération (avancé)", open=False):
            gr.Markdown(
                "**Cache entre les pas** (`caching.md`) — réutilise des calculs "
                "d'un pas à l'autre. Utile surtout > ~10 pas ; sur les modèles "
                "distillés (4–8 pas) le gain est faible et des artefacts sont "
                "possibles. Laisser désactivé en général.")
            with gr.Row():
                cache_mode = gr.Dropdown(
                    [(t("Désactivé (recommandé)"), ""),
                     ("easycache", "easycache"), ("dbcache", "dbcache"),
                     ("taylorseer", "taylorseer"), ("cache-dit", "cache-dit"),
                     ("spectrum", "spectrum")],
                    value=prefs.get("cache_mode", ""),
                    label="Mode de cache (Flux/Krea = DiT)")
                cache_opt = gr.Textbox(
                    value=prefs.get("cache_option", ""),
                    label="Option (vide = défauts)", placeholder="ex. threshold=0.2")

            gr.Markdown(
                "---\n"
                "**Convolution directe** — remplace l'algorithme de convolution "
                "(im2col) par un calcul direct. im2col déplie l'image en une "
                "grande matrice avant de multiplier : c'est rapide, mais ce "
                "tampon intermédiaire pèse lourd. En direct, il disparaît.\n\n"
                "👉 Ce qu'on peut promettre : **moins de mémoire**. La vitesse, "
                "elle, dépend de la forme des tenseurs — parfois mieux, parfois "
                "moins bien. **À essayer et à chronométrer**, pas à cocher les "
                "yeux fermés. Utile surtout si vous frôlez la saturation "
                "mémoire.\n\n"
                "*Options récentes de sd.cpp : si votre moteur ne les connaît "
                "pas, elles sont simplement ignorées (aucun risque de plantage). "
                "`update-engine.bat` pour l'avoir.*")
            with gr.Row():
                conv_diff = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_diffusion")),
                    label="Convolution directe — modèle de diffusion")
                conv_vae = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_vae")),
                    label="Convolution directe — VAE")

            gr.Markdown(
                "---\n"
                "**Exécution segmentée** (`--max-vram`) — par défaut, le moteur "
                "réserve son graphe de calcul **d'un seul bloc** : si le bloc ne "
                "tient pas, la génération s'arrête sur une erreur mémoire. Avec "
                "un budget, il a le droit de **découper le graphe** pour tenir "
                "dedans.\n\n"
                "👉 « Auto » ne fixe rien en dur : le moteur mesure la VRAM "
                "**libre** au lancement et s'en réserve une marge — il s'adapte "
                "donc à votre carte *et* à ce qui l'occupe déjà.\n\n"
                "⚠️ Découper coûte des allers-retours mémoire : **c'est plus lent**. "
                "À activer pour les résolutions qui ne passent pas autrement, pas "
                "par défaut. L'onglet **🚀 HD** s'en sert de toute façon — c'est "
                "là que le tout-ou-rien casse.")
            with gr.Row():
                max_vram = gr.Dropdown(
                    [(t("Désactivé (recommandé pour la génération)"), ""),
                     (t("Auto — VRAM libre moins 1 Go"), "auto"),
                     (t("Plafond ferme : 6 Go"), "6"),
                     (t("Plafond ferme : 8 Go"), "8"),
                     (t("Plafond ferme : 10 Go"), "10")],
                    value=prefs.get("max_vram", ""), allow_custom_value=True,
                    label="Budget VRAM du graphe",
                    info="Valeur libre acceptée : « 6 », ou « cuda0=6,cuda1=4 » "
                         "sur une machine multi-cartes.")
                stream_layers = gr.Checkbox(
                    value=bool(prefs.get("stream_layers")),
                    label="Streaming des couches (sans effet sans budget)",
                    info="Précharge les couches à la demande. Encore plus "
                         "dépendant du PCIe : à n'essayer que si le budget seul "
                         "ne suffit pas.")

        # ------------------------------------------------------------------ #
        #  Réseau & comptes
        # ------------------------------------------------------------------ #
        with gr.Accordion("🔗 Réseau & comptes", open=False):
            hf_ep = gr.Textbox(
                value=prefs.get("hf_endpoint", "https://huggingface.co"),
                label="Endpoint Hugging Face (miroir éventuel)")
            civitai_tok = gr.Textbox(
                value=prefs.get("civitai_token", ""),
                label="Jeton Civitai (optionnel — LoRA protégés)", type="password")

        save = gr.Button("💾 Enregistrer", variant="primary")
        saved = gr.Markdown("")

        def do_save(auto, gpu, tools_gpu, gpu_strategy, quant, enc_quant, fa,
                    offload, tiling, clip_cpu, vae_cpu, cache_mode, cache_opt,
                    conv_diff, conv_vae, max_vram, stream_layers,
                    hf_ep, civitai_tok):
            p = settings.load_prefs()
            # Nettoyage des anciens réglages moteur (serveur/ComfyUI, retirés).
            for stale in ("engine", "use_sd_server", "sd_server_port",
                          "comfyui_port"):
                p.pop(stale, None)
            p["auto_optimize"] = bool(auto)
            p["gpu_index"] = gpu if gpu is not None else None
            p["text_gpu_index"] = tools_gpu
            # Stratégie multi-GPU → auto_fit + encoder_gpu_index (mutuellement excl.)
            gpus = hardware.detect_gpus()
            sel = gpu if gpu is not None else (
                max(gpus, key=lambda x: x.vram_gb).index if gpus else None)
            other = next((g.index for g in gpus if g.index != sel), None)
            p["auto_fit"] = (gpu_strategy == "autofit")
            p["encoder_gpu_index"] = other if gpu_strategy == "encoder" else None
            p["cache_mode"] = cache_mode or ""
            p["cache_option"] = (cache_opt or "").strip()
            # Hors de « flags » : voir settings.DEFAULT_PREFS.
            p["conv_direct_diffusion"] = bool(conv_diff)
            p["conv_direct_vae"] = bool(conv_vae)
            # Exécution segmentée : la valeur est reprise telle quelle (elle
            # peut être « auto », un nombre, ou une affectation par carte).
            p["max_vram"] = (max_vram or "").strip()
            p["stream_layers"] = bool(stream_layers)
            p["quant"] = None if quant == "auto" else quant
            p["enc_quant"] = None if enc_quant == "auto" else enc_quant
            p["flags"] = {
                "diffusion_fa": bool(fa), "offload_to_cpu": bool(offload),
                "vae_tiling": bool(tiling), "clip_on_cpu": bool(clip_cpu),
                "vae_on_cpu": bool(vae_cpu),
            }
            p["hf_endpoint"] = hf_ep or "https://huggingface.co"
            p["civitai_token"] = (civitai_tok or "").strip()
            settings.save_prefs(p)
            return gr.update(value=_profile_md()), t("✅ Réglages enregistrés.")

        save.click(do_save,
                   inputs=[auto, gpu, tools_gpu, gpu_strategy, quant,
                           enc_quant, fa, offload, tiling, clip_cpu, vae_cpu,
                           cache_mode, cache_opt, conv_diff, conv_vae,
                           max_vram, stream_layers, hf_ep, civitai_tok],
                   outputs=[profile_md, saved])

        # --- Optimisation curatée par génération de carte (1 clic) ---
        def _apply_generation(gen_key):
            def handler(gpu_idx):
                p = settings.load_prefs()
                gpus = hardware.detect_gpus()
                g = next((x for x in gpus if x.index == gpu_idx), None) \
                    if gpu_idx is not None else None
                if g is None and gpus:
                    g = max(gpus, key=lambda x: x.vram_gb)
                vram = g.vram_gb if g else None
                prof = hardware.generation_profile(
                    gen_key, vram, hardware.detect_ram_gb())
                fl = prof.flags()
                p["auto_optimize"] = False
                p["quant"] = prof.quant
                p["enc_quant"] = prof.enc_quant
                p["flags"] = fl
                if g is not None:
                    p["gpu_index"] = g.index
                settings.save_prefs(p)
                label = hardware.GENERATIONS[gen_key]["label"]
                return (
                    gr.update(value=False), gr.update(value=prof.quant),
                    gr.update(value=prof.enc_quant),
                    gr.update(value=fl["diffusion_fa"]),
                    gr.update(value=fl["offload_to_cpu"]),
                    gr.update(value=fl["vae_tiling"]),
                    gr.update(value=fl["clip_on_cpu"]),
                    gr.update(value=fl["vae_on_cpu"]),
                    gr.update(value=_profile_md()),
                    t("✅ Optimisé pour **{label}** : diffusion `{quant}`, "
                      "encodeur `{enc}` (optimisation auto désactivée).").format(
                        label=label, quant=prof.quant, enc=prof.enc_quant))
            return handler

        gen_outputs = [auto, quant, enc_quant, fa, offload, tiling, clip_cpu,
                       vae_cpu, profile_md, saved]
        for key, btn in gen_btns.items():
            btn.click(_apply_generation(key), inputs=[gpu], outputs=gen_outputs)

        if combo_btn is not None:
            def _apply_combo():
                p = settings.load_prefs()
                preset = hardware.rtx3060_1080ti_prefs()
                p.update({k: v for k, v in preset.items() if k != "flags"})
                p["flags"] = preset["flags"]
                settings.save_prefs(p)
                fl = preset["flags"]
                return (
                    gr.update(value=False),
                    gr.update(value=preset["gpu_index"]),
                    gr.update(value=preset["text_gpu_index"]),
                    gr.update(value="encoder"),
                    gr.update(value=preset["quant"]),
                    gr.update(value=preset["enc_quant"]),
                    gr.update(value=fl["diffusion_fa"]),
                    gr.update(value=fl["offload_to_cpu"]),
                    gr.update(value=fl["vae_tiling"]),
                    gr.update(value=fl["clip_on_cpu"]),
                    gr.update(value=fl["vae_on_cpu"]),
                    gr.update(value=""), gr.update(value=""),
                    gr.update(value=_profile_md()),
                    "✅ Profil double GPU appliqué et enregistré."
                )

            combo_btn.click(
                _apply_combo,
                outputs=[auto, gpu, tools_gpu, gpu_strategy, quant, enc_quant,
                         fa, offload, tiling, clip_cpu, vae_cpu, cache_mode,
                         cache_opt, profile_md, combo_msg])
