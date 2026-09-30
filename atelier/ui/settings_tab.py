"""Réglages système : automatique par modèle, manuel explicite et diagnostic.

Le placement automatique part du poids réellement installé et de la VRAM
disponible au moment de générer. Un tableau explique le plan de chaque modèle.
La priorité mémoire/qualité est facultative ; les paramètres sd.cpp restent
accessibles sous Expert et basculent en manuel dès qu'on y touche. Aucun bouton
Enregistrer : les changements sont appliqués immédiatement.
"""
from __future__ import annotations

import json
import queue
import threading

import gradio as gr

from .. import benchmark, diagnostics, hardware, registry, settings, system_profile
from ..i18n import t
from .theme import THEME_JS
from . import widgets

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]

# Préfixe des confirmations. Elles sont au passé et concrètes — « appliqué »,
# pas « enregistré » : ce qu'on veut savoir, c'est que c'est FAIT.
_OK = "✅ "


def _said(msg: str):
    """Affiche une confirmation — et fait EXISTER la ligne à ce moment-là.

    Un Markdown vide n'est pas invisible dans Gradio : son conteneur garde son
    cadre. Une bande vide en permanence au-dessus des réglages, c'est du bruit
    qui ressemble à un message qu'on n'arrive pas à lire. La ligne apparaît donc
    au premier changement, et pas avant.
    """
    return gr.update(value=msg, visible=True)

_STRATEGY_SAID = {
    "single": "Everything will run on a single card.",
    "encoder": "The 2nd card will read your text; the image stays on the first.",
    "autofit": "sd.cpp will place the weights itself — measure it before "
               "believing it.",
}


def _gpu_choices() -> list[tuple[str, int]]:
    return [(f"#{g.index} — {g.label()}", g.index)
            for g in hardware.detect_gpus()]


def _save(**changes) -> dict:
    """Applique des changements aux préférences et renvoie le tout."""
    p = settings.load_prefs()
    # Nettoyage des anciens réglages moteur (serveur/ComfyUI, retirés).
    for stale in ("engine", "use_sd_server", "sd_server_port",
                  "comfyui_port", "resident_engine"):
        p.pop(stale, None)
    p.update(changes)
    settings.save_prefs(p)
    return p


def _strategy_of(prefs: dict) -> str:
    if prefs.get("auto_fit"):
        return "autofit"
    eg = prefs.get("encoder_gpu_index")
    if eg is not None and eg != prefs.get("gpu_index"):
        return "encoder"
    return "single"


def _apply_strategy_prefs(choice: str, gpu_idx) -> dict:
    """Traduit le choix en préférences (les trois stratégies s'excluent)."""
    gpus = hardware.detect_gpus()
    sel = gpu_idx if gpu_idx is not None else (
        max(gpus, key=lambda x: x.vram_gb).index if gpus else None)
    other = next((g.index for g in gpus if g.index != sel), None)
    split = (choice == "encoder") and sel is not None and other is not None
    return _save(
        system_mode="manual", auto_optimize=False,
        auto_fit=(choice == "autofit"),
        encoder_gpu_index=other if split else None,
        # Résidence explicite : le calcul ET les poids de l'encodeur vont sur
        # la 2e carte. Sans cela, un encodeur « sur la 2e carte » relirait ses
        # poids depuis la RAM à chaque image, à travers le PCIe.
        params_backend=(f"diffusion=cuda{sel},vae=cuda{sel},te=cuda{other}"
                        if split else ""))


# --------------------------------------------------------------------------- #
#  Ce que la machine a décidé — en français, pas en noms d'options
# --------------------------------------------------------------------------- #
def _headline(prefs: dict | None = None) -> str:
    """Carte lisible : le matériel, puis ce qui en découle.

    On ne recopie surtout pas la liste des flags : `vae_tiling`, `clip_on_cpu`
    ne veulent rien dire pour qui ne connaît pas sd.cpp. Chacun est traduit en
    conséquence observable.
    """
    prefs = prefs if prefs is not None else settings.load_prefs()
    bias = hardware.bias_from_prefs(prefs)
    prof = hardware.biased_profile(bias, settings.generation_gpu_index(prefs))
    gpus = hardware.detect_gpus()

    if prof.gpu is None:
        return t("### ⚠️ No compatible GPU detected\nThe app will run on the "
                 "processor: that is **very slow** (minutes per image). Check "
                 "your drivers, or type `nvidia-smi` in a terminal.")

    lines = [t("### Your hardware"),
             t("**{name}** — {vram} GB of video memory · {ram} GB of RAM"
               ).format(name=prof.gpu.name, vram=f"{prof.gpu.vram_gb:.0f}",
                        ram=f"{prof.ram_gb:.0f}")]
    if len(gpus) > 1:
        others = ", ".join(g.name for g in gpus if g.index != prof.gpu.index)
        lines.append(t("Second card available: {other}.").format(
            other=others))

    if prefs.get("system_mode") == "auto":
        lines += ["", t("**Automatic system setup, recalculated for each model:**")]
        lines.append(t("- choose the image and text weights for this model "
                       "from the hardware profile or your Model Catalog choice;"))
        lines.append(t("- inspect the installed image weights and free VRAM "
                       "at generation time, then place image and text weights "
                       "separately when sd.cpp supports it;"))
    else:
        lines += ["", t("**Manual system setup active:** the saved expert "
                         "flags, weight choices and GPU placement are used.")]
        return "\n".join(lines)
    lines.append(
        t("- “flash attention” acceleration is on (your card supports it);") if prof.diffusion_fa else
        t("- “flash attention” stays off: your card lacks the units for it, "
          "turning it on would slow things down;"))
    lines.append(t("- image assembly uses memory-saving tiles when the "
                   "model or output size needs them."))
    return "\n".join(lines)


def _system_table(prefs: dict | None = None) -> str:
    """Preview by model; the engine recalculates from free VRAM on every run."""
    prefs = prefs if prefs is not None else settings.load_prefs()
    if prefs.get("system_mode") != "auto":
        return ("**Manual mode:** your saved system settings are active. "
                "Switch to Automatic to let each model choose its placement.")
    from ..engine import sdcpp
    sd_cli = settings.find_sd_cli()
    gpu = hardware.auto_profile(settings.generation_gpu_index(prefs)).gpu
    free = hardware.free_vram_gb(gpu.index) if gpu and not gpu.is_apple else None
    lines = ["**System choices by model** — preview from installed weights. "
             "The available memory is checked again for each generation.",
             "", "| Model | Image weights | Text weights | Placement |",
             "|---|---:|---:|---|"]
    for model in registry.load_base_models(prefs):
        by_role = {c.role: registry.resolve_component_path(c)
                   for c in model.components if not c.optional}
        image = by_role.get("diffusion") or by_role.get("model")
        vae = by_role.get("vae")
        if not registry.model_is_ready(model) or not image:
            lines.append(f"| {model.name} | Download required | — | "
                         "Computed after download |")
            continue
        files = {"diffusion": image, "vae": vae,
                 "enc": by_role.get("text_encoder")}
        plan = system_profile.plan_for_model(
            model, files, prefs, sd_cli,
            width=int(model.defaults.get("width", 1024)),
            height=int(model.defaults.get("height", 1024)),
            available_gib=free)
        image_size = f"{plan.image_gib:.1f} GiB" if plan.image_gib else "—"
        enc_size = f"{plan.encoder_gib:.1f} GiB" if plan.encoder_gib else "—"
        lines.append(f"| {model.name} | {image_size} | {enc_size} | "
                     f"{plan.placement} |")
    return "\n".join(lines)


def _bias_note(bias: str, prefs: dict | None = None) -> str:
    """Ce que le cran choisi change — une raison, puis un chiffre."""
    prefs = prefs if prefs is not None else settings.load_prefs()
    spec = hardware.BIASES.get(bias) or hardware.BIASES["balanced"]
    idx = settings.generation_gpu_index(prefs)
    prof = hardware.biased_profile(bias, idx)
    ref = hardware.biased_profile("balanced", idx)
    detail = t(spec["why"])
    if prof.quant != ref.quant:
        detail += t("  \n→ base GGUF target `{quant}` instead of `{ref}`."
                    ).format(quant=prof.quant, ref=ref.quant)
    else:
        detail += t("  \n→ base GGUF target `{quant}`.").format(quant=prof.quant)
    detail += t(" Individual models can adjust this target; an explicit "
                "Catalog weight always takes priority.")
    return detail


def build_settings_tab():
    with gr.Tab("⚙️ Settings"):
        prefs = settings.load_prefs()
        gpus = hardware.detect_gpus()
        multi_gpu = len(gpus) > 1

        headline = gr.Markdown(_headline(prefs))
        status = gr.Markdown("", elem_classes="feedback", visible=False)
        mode = gr.Radio(
            [("Automatic for each model (recommended)", "auto"),
             ("Manual / measured configuration", "manual")],
            value=prefs.get("system_mode", "auto"), label="System settings",
            info="Automatic uses the installed model's actual weight sizes and "
                 "free VRAM every time you generate. Manual keeps your saved "
                 "expert settings.")
        model_plans = gr.Markdown(_system_table(prefs))
        refresh_plans = gr.Button("Refresh model system choices", size="sm")

        # ------------------------------------------------------------------ #
        #  LE réglage : marge mémoire ou détail
        # ------------------------------------------------------------------ #
        gr.Markdown(t("Balanced automatically gives each model room to run. "
                      "Adjust the priority only if you need more headroom or "
                      "detail."))
        show_bias = gr.Button("Change memory / detail priority (optional)",
                              size="sm")
        _bias = hardware.bias_from_prefs(prefs)
        bias = gr.Radio(
            [(t(spec["label"]), key) for key, spec in hardware.BIASES.items()],
            value=_bias, label="Priority", show_label=False, visible=False)
        bias_note = gr.Markdown(_bias_note(_bias, prefs), visible=False)
        show_bias.click(lambda: (gr.update(visible=True),
                                 gr.update(visible=True)),
                        outputs=[bias, bias_note])
        # Le vrai mode d'emploi : partir du SYMPTÔME. C'est ainsi qu'on arrive
        # sur cette page — pas en se demandant « quelle quantification ? ».
        # Deux des trois réponses renvoient ailleurs, et c'est volontaire :
        # laisser croire que tout se règle ici serait la même impasse.
        gr.Markdown(t(
            "> **Something specific going wrong?**  \n> *“Out of memory / "
            "generation stops”* → pick **🪶 More memory headroom** above.  \n> "
            "*“It is too slow”* → not settled here but in the generation tab: "
            "lower the **step count** and the **image size**, which weigh far "
            "more.  \n> *“My images look dull”* → not here either: that is "
            "the **prompt** and the **styles**, not a hardware setting."))

        # ------------------------------------------------------------------ #
        #  Profils de MACHINE — un bouton par tour connue
        # ------------------------------------------------------------------ #
        _presets = hardware.available_presets()
        preset_buttons = []
        gpu = gr.Dropdown(
            label="Card used for generating",
            choices=[("Automatic (best card)", None)] + _gpu_choices(),
            value=settings.generation_gpu_index(prefs), visible=multi_gpu,
            info=t("Automatic picks the largest card (preferring tensor cores "
                   "when VRAM ties). Set a card here only to override it."))
        with gr.Column(visible=prefs.get("system_mode", "auto") == "manual"
                       and bool(_presets or multi_gpu)) as manual_controls:
            gr.Markdown("**Manual placement overrides** — optional. Applying "
                        "one turns on Manual mode.")
            if _presets:
                with gr.Row():
                    for _preset, _cards in _presets:
                        preset_buttons.append(
                            (_preset, gr.Button(t(_preset.label), size="sm")))
                preset_status = gr.Markdown("", elem_classes="feedback",
                                            visible=False)
            if multi_gpu:
                strategy = gr.Radio(
                    [(t("Everything on one card — the most reliable"), "single"),
                     (t("The 2nd card handles the text — frees memory for the "
                        "image"), "encoder"),
                     (t("Let sd.cpp place the weights — measure before "
                        "believing it"), "autofit")],
                    value=_strategy_of(prefs), label="Manual split",
                    show_label=False)
        if multi_gpu:
            tools_gpu = gr.Dropdown(
                label="Card for the prompt enhancer",
                choices=[(t("The same one as for the image"), None)] + _gpu_choices(),
                value=prefs.get("text_gpu_index"))
        else:
            strategy = gr.State(_strategy_of(prefs))
            tools_gpu = gr.State(prefs.get("text_gpu_index"))

        # ------------------------------------------------------------------ #
        #  Mesurer plutôt que deviner
        # ------------------------------------------------------------------ #
        gr.Markdown(t(
            "---\n### 🧪 When in doubt, measure\nThe app generates the **same "
            "image** {n} times per configuration (plus a first one thrown "
            "away, the time for everything to load) and keeps the median. It "
            "changes **no setting**: it tells you which is fastest, then you "
            "decide."
        ).format(n=benchmark.MEASURED_RUNS))
        with gr.Row():
            bench_btn = gr.Button(t("⏱️ Measure on my machine"),
                                  variant="primary")
            apply_bench = gr.Button(t("Apply the fastest"))
            bench_stop = gr.Button(t("⏹️ Stop"), variant="stop")
        bench_status = gr.Markdown("")
        with gr.Accordion(t("Measurement details (log and report)"),
                          open=False):
            bench_file = gr.File(label="JSON report", interactive=False)
            bench_log = gr.Textbox(label="Test log", lines=10,
                                   autoscroll=True, elem_classes="log-box")
            system_md = gr.Markdown(diagnostics.summary_markdown())
            report_btn = gr.Button(t("📋 Export the system report"),
                                   size="sm")

        # ------------------------------------------------------------------ #
        #  Expert : les options brutes de sd.cpp, sous UN seul repli
        # ------------------------------------------------------------------ #
        with gr.Accordion(
                t("🔧 Expert — raw sd.cpp options (optional)"),
                open=False):
            gr.Markdown(t(
                "⚠️ **Nothing here is required.** These options exist because "
                "sd.cpp exposes them, not because you should touch them. They "
                "are set by measuring, not by guessing. Changing a control "
                "switches to **Manual** and keeps your chosen values."))

            gr.Markdown(t("**Forced quantization** — “auto” = let the app "
                          "decide from the card."))
            with gr.Row():
                quant = gr.Dropdown(label="Image model",
                                    choices=["auto"] + QUANTS,
                                    value=prefs.get("quant") or "auto")
                enc_quant = gr.Dropdown(label="Text analysis",
                                        choices=["auto"] + QUANTS,
                                        value=prefs.get("enc_quant") or "auto")
            f = prefs.get("flags", {})
            gr.Markdown(t("**Engine memory options.**"))
            with gr.Row():
                fa = gr.Checkbox(value=f.get("diffusion_fa", True),
                                 label="Flash attention")
                offload = gr.Checkbox(value=f.get("offload_to_cpu", True),
                                      label="Model kept in RAM")
                tiling = gr.Checkbox(value=f.get("vae_tiling", True),
                                     label="Image assembled in pieces")
            with gr.Row():
                clip_cpu = gr.Checkbox(value=f.get("clip_on_cpu", False),
                                       label="Text on the processor")
                vae_cpu = gr.Checkbox(value=f.get("vae_on_cpu", False),
                                      label="Assembly on the processor")

            gr.Markdown(t(
                "---\n**Cache between steps** — reuses computations from one "
                "diffusion step to the next. Off in Automatic mode: it can "
                "change the output, including on models that run many steps. "
                "Measure before enabling it manually."))
            with gr.Row():
                cache_mode = gr.Dropdown(
                    [(t("Disabled (recommended)"), ""),
                     ("easycache", "easycache"), ("dbcache", "dbcache"),
                     ("taylorseer", "taylorseer"), ("cache-dit", "cache-dit"),
                     ("spectrum", "spectrum")],
                    value=prefs.get("cache_mode", ""), label="Cache mode")
                cache_opt = gr.Textbox(
                    value=prefs.get("cache_option", ""),
                    label="Option (blank = defaults)",
                    placeholder="ex. threshold=0.2")

            gr.Markdown(t(
                "---\n**Direct convolution** — removes a large intermediate "
                "buffer. The memory gain is certain; the speed effect is "
                "**unpredictable** (sometimes better, sometimes worse). To be "
                "timed, not ticked blindly."))
            with gr.Row():
                conv_diff = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_diffusion")),
                    label="Direct convolution — image model")
                conv_vae = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_vae")),
                    label="Direct convolution — assembly")

            gr.Markdown(t(
                "---\n**Splitting the computation** — lets the engine cut its "
                "graph to fit a budget instead of failing. **It is slower**: "
                "keep it for resolutions that will not fit otherwise. The 🚀 "
                "HD tab already uses it on its own.\n\nWithout a budget the "
                "engine has no target: it cuts blind and finds out part-way "
                "through that a segment will not fit — the *“cannot make "
                "enough memory available … workspace capacity check”* "
                "failure. Generation retries once with **auto** by itself "
                "when that happens; setting it here avoids the failed first "
                "attempt."))
            with gr.Row():
                max_vram = gr.Dropdown(
                    [(t("Disabled (recommended)"), ""),
                     (t("Auto — free memory minus 1 GB"), "auto"),
                     (t("Hard cap: 6 GB"), "6"),
                     (t("Hard cap: 8 GB"), "8"),
                     (t("Hard cap: 10 GB"), "10")],
                    value=prefs.get("max_vram", ""), allow_custom_value=True,
                    label="Memory budget for the computation")
                stream_layers = gr.Checkbox(
                    value=bool(prefs.get("stream_layers")),
                    label="Layer streaming from RAM",
                    info=t("Requires the model to be kept in RAM. Without "
                           "that, the engine ignores the option."))

            # Confirmation LOCALE : la ligne d'état du haut est hors de l'écran
            # quand on coche quelque chose ici. Un réglage qui s'applique sans
            # rien dire de visible, c'est un réglage dont on doute.
            expert_status = gr.Markdown("", elem_classes="feedback", visible=False)


        # ------------------------------------------------------------------ #
        #  Ce qui n'a rien à voir avec la génération
        # ------------------------------------------------------------------ #
        with gr.Accordion(t("🌍 Theme and accounts"), open=False):
            with gr.Row():
                theme_dd = gr.Dropdown(
                    [(t("Light"), "light"), (t("Dark"), "dark")],
                    value=prefs.get("theme", "light"),
                    label="🎨 Theme (restart required)")
            hf_ep = gr.Textbox(
                value=prefs.get("hf_endpoint", "https://huggingface.co"),
                label="Hugging Face endpoint (optional mirror)")
            civitai_tok = gr.Textbox(
                value=prefs.get("civitai_token", ""),
                label="Civitai token (optional — gated LoRAs)",
                type="password")
            account_status = gr.Markdown("", elem_classes="feedback", visible=False)

        # ================================================================== #
        #  Câblage — tout s'applique à la volée
        # ================================================================== #
        def _apply_mode(choice):
            p = _save(system_mode=choice,
                      auto_optimize=(choice == "auto"))
            priority = hardware.bias_from_prefs(p)
            return (_headline(p), _system_table(p),
                    gr.update(visible=choice == "manual" and
                              bool(_presets or multi_gpu)),
                    _said(_OK + ("Automatic system setup is active for every "
                                 "model." if choice == "auto" else
                                 "Saved manual system settings are active.")),
                    gr.update(value=priority), _bias_note(priority, p),
                    gr.update(value=settings.generation_gpu_index(p)))

        mode.change(_apply_mode, inputs=[mode],
                    outputs=[headline, model_plans, manual_controls, status,
                             bias, bias_note, gpu])
        refresh_plans.click(lambda: _system_table(), outputs=[model_plans])

        def _apply_bias(choice):
            p = _save(system_mode="auto", auto_optimize=True,
                      hardware_bias=choice)
            return (_headline(p), _bias_note(choice, p),
                    _said(_OK + t("Priority applied: **{label}**.").format(
                        label=t(hardware.BIASES[choice]["label"]))),
                    gr.update(value="auto"), _system_table(p))

        bias.input(_apply_bias, inputs=[bias],
                   outputs=[headline, bias_note, status, mode, model_plans])

        def _apply_gpu(idx, choice, selected_mode):
            if selected_mode == "manual":
                p = _save(gpu_index=idx)
                p = _apply_strategy_prefs(choice, idx)
            else:
                p = _save(auto_gpu_index=idx)
            label = f"#{idx}" if idx is not None else "automatic"
            return (_headline(p), _said(_OK + t(
                "Generation card: {idx}.").format(idx=label)),
                    _system_table(p))

        gpu.input(_apply_gpu, inputs=[gpu, strategy, mode],
                  outputs=[headline, status, model_plans])

        if multi_gpu:
            def _apply_strategy(choice, idx):
                p = _apply_strategy_prefs(choice, idx)
                return (_headline(p),
                        _said(_OK + t(_STRATEGY_SAID[choice])),
                        _system_table(p))

            strategy.change(_apply_strategy, inputs=[strategy, gpu],
                            outputs=[headline, status, model_plans])

            def _apply_tools_gpu(v):
                _save(text_gpu_index=v)
                return _said(_OK + t("Enhancer card saved."))

            tools_gpu.change(_apply_tools_gpu, inputs=[tools_gpu],
                             outputs=[status])

        # ---- Expert : chaque contrôle s'applique seul --------------------- #
        def _apply_expert(q, eq, fa_, off, til, clip, vae, cm, co,
                          cd, cv, mv, sl):
            p = _save(system_mode="manual", auto_optimize=False,
                      quant=None if q == "auto" else q,
                      enc_quant=None if eq == "auto" else eq,
                      flags={"diffusion_fa": bool(fa_),
                             "offload_to_cpu": bool(off),
                             "vae_tiling": bool(til),
                             "clip_on_cpu": bool(clip),
                             "vae_on_cpu": bool(vae)},
                      cache_mode=cm or "", cache_option=(co or "").strip(),
                      conv_direct_diffusion=bool(cd),
                      conv_direct_vae=bool(cv),
                      max_vram=(mv or "").strip(),
                      stream_layers=bool(sl))
            return (_headline(p),
                    _bias_note(hardware.bias_from_prefs(p), p),
                    _said(_OK + t("Expert setting applied (automatic tuning "
                                  "off).")),
                    gr.update(value="manual"), _system_table(p))

        _expert = [quant, enc_quant, fa, offload, tiling, clip_cpu, vae_cpu,
                   cache_mode, cache_opt, conv_diff, conv_vae, max_vram,
                   stream_layers]
        for comp in _expert:
            comp.change(_apply_expert, inputs=_expert,
                        outputs=[headline, bias_note, expert_status, mode,
                                 model_plans])

        # ---- Theme, accounts ---------------------------------------------- #
        def _apply_theme(th):
            _save(theme="dark" if th == "dark" else "light")
            return _said(_OK + t("Theme applied and saved."))

        def _apply_endpoint(v):
            _save(hf_endpoint=v or "https://huggingface.co")
            return _said(_OK + t("Endpoint saved."))

        def _apply_token(v):
            _save(civitai_token=(v or "").strip())
            return _said(_OK + t("Civitai token saved."))

        theme_dd.change(_apply_theme, inputs=[theme_dd], outputs=[account_status],
                        js=THEME_JS, queue=False)
        hf_ep.change(_apply_endpoint, inputs=[hf_ep], outputs=[account_status])
        civitai_tok.change(_apply_token, inputs=[civitai_tok],
                           outputs=[account_status])

        # ---- Mesure -------------------------------------------------------- #
        _bench_stop = threading.Event()

        def _verdict(data: dict) -> str:
            """Dire ce qui a été mesuré, pas seulement qui gagne."""
            rows = [r for r in data.get("results", []) if r.get("ok")]
            if not rows:
                return t("❌ No configuration could be measured (see the log).")
            lines = [f"- **{r['label']}** — {r['seconds']:.2f} s "
                     f"(± {r.get('spread_seconds', 0):.2f} s)" for r in rows]
            best = data.get("recommended_mode") or data.get("fastest_model")
            head = t("✅ Median over {n} runs · fastest: **{best}**"
                     ).format(n=data.get("measured_runs", "?"), best=best)
            return head + "\n" + "\n".join(lines)

        def _do_bench():
            """Diffuse le journal au fil de l'eau.

            Rendre le résultat d'un bloc à la fin laisserait l'interface muette
            plusieurs minutes, sans dire si ça avance ni comment l'arrêter.
            """
            _bench_stop.clear()
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    state["path"] = benchmark.run_hardware_benchmark(
                        log=q.put, cancel=_bench_stop.is_set)
                except Exception as exc:  # noqa: BLE001
                    state["err"] = exc
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield t("⏳ Measuring…"), gr.update(), ""
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-500:])
            tail = "\n".join(logs[-500:])
            if isinstance(state.get("err"), benchmark.Cancelled):
                yield (t("⏹️ Measurement interrupted — no setting changed."),
                       gr.update(), tail)
                return
            if "err" in state:
                yield f"❌ {state['err']}", gr.update(), tail
                return
            path = state["path"]
            yield (_verdict(json.loads(path.read_text(encoding="utf-8"))),
                   str(path), tail)

        _bench_evt = bench_btn.click(
            _do_bench, outputs=[bench_status, bench_file, bench_log], **widgets.GPU_QUEUE)

        def _cancel_bench() -> str:
            # On POSE le drapeau au lieu de tuer le fil : la génération en
            # cours va au bout et la mesure s'arrête proprement ensuite.
            # Couper au milieu laisserait un tir à moitié chronométré.
            _bench_stop.set()
            return t("⏹️ Stop requested — the run in progress will finish first.")

        widgets.stop_into_log(bench_stop, _cancel_bench, bench_log,
                              [_bench_evt])

        def _apply_report(raw):
            if not raw:
                return (gr.update(), _said(t("❌ Run the measurement first.")),
                        gr.update(), gr.update())
            try:
                mode = benchmark.apply_recommendation(getattr(raw, "name", raw))
            except Exception as exc:  # noqa: BLE001
                return gr.update(), _said(f"❌ {exc}"), gr.update(), gr.update()
            return (_headline(), _said(_OK + t(
                "Measured configuration applied: **{mode}**.").format(
                    mode=mode)), gr.update(value="manual"), _system_table())

        apply_bench.click(_apply_report, inputs=[bench_file],
                          outputs=[headline, status, mode, model_plans])

        def _system_report():
            return (diagnostics.summary_markdown(),
                    str(diagnostics.write_system_report()))

        report_btn.click(_system_report, outputs=[system_md, bench_file])

        def _apply_machine_preset(key: str, label: str, summary: str):
            def _run():
                values = hardware.preset_prefs(key)
                p = _save(system_mode="manual", **values)
                #  La stratégie affichée doit suivre ce que le profil a
                #  RÉELLEMENT posé : un profil mono-carte qui laisserait le
                #  sélecteur sur « encodeur sur la 2e carte » se contredirait
                #  à l'écran.
                return (_headline(p),
                        gr.update(value=values["gpu_index"]),
                        gr.update(value=_strategy_of(p)),
                        gr.update(value=_OK + t(label) + " — " + t(summary),
                                  visible=True),
                        gr.update(value="manual"), _system_table(p))
            return _run

        for _preset, _btn in preset_buttons:
            _btn.click(_apply_machine_preset(_preset.key, _preset.label,
                                             _preset.summary),
                       outputs=[headline, gpu, strategy, preset_status, mode,
                                model_plans])
