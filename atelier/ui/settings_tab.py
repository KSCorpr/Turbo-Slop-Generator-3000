"""Onglet Réglages : matériel détecté, optimisations auto/manuelles, quant."""
from __future__ import annotations

import gradio as gr

from .. import hardware, settings
from ..i18n import t

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]
LANGS = [("Français", "fr"), ("English", "en")]


def _gpu_choices() -> list[tuple[str, int]]:
    return [(f"#{g.index} — {g.name} ({g.vram_gb:.0f} Go, {g.arch})", g.index)
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
        else:
            gpu_strategy = gr.State(_gpu_strategy(prefs))
            tools_gpu = gr.State(prefs.get("text_gpu_index"))

        # ------------------------------------------------------------------ #
        #  Moteur & accélération (avancé)
        # ------------------------------------------------------------------ #
        with gr.Accordion("🚀 Moteur & accélération (avancé)", open=False):
            gr.Markdown(
                "**Moteur serveur résident** — garde le modèle en mémoire entre "
                "les images (itération quasi instantanée). Un modèle à la fois ; "
                "repli auto sur le mode classique en cas de souci. *(Pas d'aperçu "
                "temps réel.)*")
            server_mode = gr.Checkbox(
                value=prefs.get("use_sd_server", False),
                label="Activer le moteur serveur")
            gr.Markdown(
                "**Accélération par cache** — réutilise des calculs entre les pas "
                "(`caching.md`). Utile surtout > ~10 pas ; sur les modèles distillés "
                "(4–8 pas) gain faible + artefacts possibles. Laisser désactivé "
                "en général.")
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
                    server_mode, hf_ep, civitai_tok):
            p = settings.load_prefs()
            p["use_sd_server"] = bool(server_mode)
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
                           cache_mode, cache_opt, server_mode, hf_ep, civitai_tok],
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
