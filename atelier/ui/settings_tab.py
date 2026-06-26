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


def build_settings_tab():
    with gr.Tab("⚙️ Réglages"):
        gr.Markdown("### Matériel & optimisations")
        profile_md = gr.Markdown(_profile_md())

        prefs = settings.load_prefs()

        with gr.Row():
            lang_dd = gr.Dropdown(
                LANGS, value=prefs.get("lang", "fr"),
                label="🌐 Langue / Language (redémarrage requis)")
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

        with gr.Row():
            auto = gr.Checkbox(value=prefs.get("auto_optimize", True),
                               label="Optimisation automatique (selon GPU + RAM)")
            gpu = gr.Dropdown(label="GPU à utiliser",
                              choices=_gpu_choices(),
                              value=prefs.get("gpu_index"))

        if len(_gpu_choices()) > 1:
            gr.Markdown(
                "#### 🧮 Multi-GPU (2e carte, ex. 1080 Ti)\n"
                "Déchargez les **outils PyTorch** (améliorateur de prompt, "
                "profondeur, détourage, upscale SDXL) et/ou l'**encodeur de "
                "texte** sur une 2e carte pour libérer le GPU de génération.")
            with gr.Row():
                tools_gpu = gr.Dropdown(
                    label="GPU des outils PyTorch (enhancer, depth, SDXL…)",
                    choices=[(t("Auto (même que génération)"), None)]
                            + _gpu_choices(),
                    value=prefs.get("tools_gpu_index"))
                enc_gpu = gr.Dropdown(
                    label="Encodeur de texte sur ce GPU (⚠️ expérimental)",
                    choices=[(t("Désactivé (normal)"), None)] + _gpu_choices(),
                    value=prefs.get("encoder_gpu_index"))
        else:
            tools_gpu = gr.State(prefs.get("tools_gpu_index"))
            enc_gpu = gr.State(prefs.get("encoder_gpu_index"))

        gr.Markdown(
            "#### ⚡ Optimiser pour ma génération de carte (1 clic)\n"
            "Applique un préréglage adapté (quantification + offload + tiling) "
            "calé sur la VRAM réelle de la carte sélectionnée. Désactive "
            "l'optimisation automatique.")
        with gr.Row():
            gen_btns = {key: gr.Button(spec["label"], size="sm")
                        for key, spec in hardware.GENERATIONS.items()}

        with gr.Row():
            quant = gr.Dropdown(label="Quant. diffusion (vide = auto)",
                                choices=["auto"] + QUANTS,
                                value=prefs.get("quant") or "auto")
            enc_quant = gr.Dropdown(label="Quant. encodeur (vide = auto)",
                                    choices=["auto"] + QUANTS,
                                    value=prefs.get("enc_quant") or "auto")

        gr.Markdown("**Réglages manuels** (utilisés si l'auto est décochée)")
        f = prefs.get("flags", {})
        with gr.Row():
            fa = gr.Checkbox(value=f.get("diffusion_fa", True), label="Flash attention")
            offload = gr.Checkbox(value=f.get("offload_to_cpu", True), label="Offload CPU")
            tiling = gr.Checkbox(value=f.get("vae_tiling", True), label="VAE tiling")
        with gr.Row():
            clip_cpu = gr.Checkbox(value=f.get("clip_on_cpu", False), label="CLIP sur CPU")
            vae_cpu = gr.Checkbox(value=f.get("vae_on_cpu", False), label="VAE sur CPU")

        hf_ep = gr.Textbox(value=prefs.get("hf_endpoint", "https://huggingface.co"),
                           label="Endpoint Hugging Face (miroir éventuel)")

        save = gr.Button("💾 Enregistrer", variant="primary")
        saved = gr.Markdown("")

        def do_save(auto, gpu, tools_gpu, enc_gpu, quant, enc_quant, fa, offload,
                    tiling, clip_cpu, vae_cpu, hf_ep):
            p = settings.load_prefs()
            p["auto_optimize"] = bool(auto)
            p["gpu_index"] = gpu if gpu is not None else None
            p["tools_gpu_index"] = tools_gpu
            p["encoder_gpu_index"] = enc_gpu
            p["quant"] = None if quant == "auto" else quant
            p["enc_quant"] = None if enc_quant == "auto" else enc_quant
            p["flags"] = {
                "diffusion_fa": bool(fa), "offload_to_cpu": bool(offload),
                "vae_tiling": bool(tiling), "clip_on_cpu": bool(clip_cpu),
                "vae_on_cpu": bool(vae_cpu),
            }
            p["hf_endpoint"] = hf_ep or "https://huggingface.co"
            settings.save_prefs(p)
            return gr.update(value=_profile_md()), t("✅ Réglages enregistrés.")

        save.click(do_save,
                   inputs=[auto, gpu, tools_gpu, enc_gpu, quant, enc_quant, fa,
                           offload, tiling, clip_cpu, vae_cpu, hf_ep],
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
