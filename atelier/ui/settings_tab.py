"""Onglet Réglages — mono-GPU : le profil automatique s'adapte à la meilleure
carte détectée (quant selon VRAM, encodeur selon RAM) ; le manuel dépanne.
"""
from __future__ import annotations

import gradio as gr

from .. import hardware, settings
from ..i18n import t

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]
LANGS = [("Français", "fr"), ("English", "en")]


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


def build_settings_tab():
    with gr.Tab("⚙️ Réglages"):
        gr.Markdown("### Matériel détecté")
        profile_md = gr.Markdown(_profile_md())

        prefs = settings.load_prefs()

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
        #  Optimisation : l'automatique s'adapte à la carte ; manuel = dépannage
        # ------------------------------------------------------------------ #
        gr.Markdown("### 🎛️ Optimisation")
        auto = gr.Checkbox(
            value=prefs.get("auto_optimize", True),
            label="Automatique (recommandé — quant selon la VRAM de la "
                  "carte, encodeur déchargé en RAM)")

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
        #  Accélération par cache (avancé)
        # ------------------------------------------------------------------ #
        with gr.Accordion("⚡ Accélération par cache (avancé)", open=False):
            gr.Markdown(
                "Réutilise des calculs entre les pas (`caching.md`). Utile "
                "surtout > ~10 pas ; sur les modèles distillés (4–8 pas) gain "
                "faible + artefacts possibles. Laisser désactivé en général.")
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

        def do_save(auto, quant, enc_quant, fa, offload, tiling, clip_cpu,
                    vae_cpu, cache_mode, cache_opt, hf_ep, civitai_tok):
            p = settings.load_prefs()
            # Purge des réglages des anciennes versions (multi-GPU, moteurs).
            for stale in ("engine", "use_sd_server", "sd_server_port",
                          "comfyui_port", "auto_fit", "split_mode",
                          "encoder_gpu_index"):
                p.pop(stale, None)
            p["auto_optimize"] = bool(auto)
            p["gpu_index"] = None          # mono-GPU : auto = meilleure carte détectée
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
                   inputs=[auto, quant, enc_quant, fa, offload, tiling,
                           clip_cpu, vae_cpu, cache_mode, cache_opt, hf_ep,
                           civitai_tok],
                   outputs=[profile_md, saved])
