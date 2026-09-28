"""Onglet Catalogue de modèles : catalogue des modèles, recommandations selon le
matériel, téléchargement à la demande."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings, video
from ..i18n import t

_AUTO = "__auto__"


def _weight_menu(model: registry.BaseModel, role: str,
                 variants: list[tuple[str, str]] | None = None):
    comp = next((c for c in model.components if c.role == role), None)
    if comp is None:
        return [], _AUTO
    auto_label = (f"Automatic (target: {comp.requested()})" if comp.token else
                  f"Default ({comp.requested()})")
    menu = [(auto_label, _AUTO)]
    menu.extend(variants or [])
    if not variants and comp.token is None:
        # Un choix enregistré doit déjà figurer dans le menu avant même la
        # requête de métadonnées, sinon Gradio le réinitialise au lancement.
        menu.append((f"Selected: {comp.requested()} (load sizes)",
                     comp.requested()))
    selected = comp.requested() if comp.token is None else _AUTO
    if variants and selected not in {value for _, value in menu}:
        selected = _AUTO
    return menu, selected


def _card_md(model: registry.BaseModel, recos: dict[str, list[str]]) -> str:
    ready = registry.model_is_ready(model)
    status = (f"<span class='status-ok'>{t('● installed')}</span>" if ready
              else f"<span class='status-missing'>{t('○ not installed')}</span>")
    tags = " ".join(f"<span class='tag'>{t}</span>" for t in model.tags)
    reco = " · ".join(recos.get(model.id, []))
    enc = next((c for c in model.components if c.role == "text_encoder"), None)
    existing = registry.resolve_component_path(enc) if enc else None
    shared = (f"<p><small>Text encoder: {existing.name} "
              f"({existing.stat().st_size / 1e9:.2f} GB on disk). "
              "Compatible installed weights are reused automatically.</small></p>"
              if existing else "")
    return (f"<div class='model-card'><h3>{model.name} &nbsp; {status}</h3>"
            f"{tags}<p>{model.description}</p>"
            f"<small>{reco}</small>{shared}</div>")


def _build_h3_catalog():
    """Keep video weights separate from image models and their encoder presets."""
    gr.Markdown("---\n### 🎬 MiniMax H3 · video only\n"
                "Turbo uses a merged 4-step GGUF for text and first/last "
                "frames. Ref2VA uses a separate GGUF for 1–3 reference "
                "images (50 steps). Both modes share the same H3 text "
                "encoder and video VAE; no audio model is downloaded. "
                "Weight placement and compute segmentation are automatic.")
    d, e = video.selected()
    rd, _ = video.selected(mode="refs")

    def _status():
        td, te = video.selected()
        refd, refe = video.selected(mode="refs")
        turbo = "ready" if video.ready(td, te) else "missing weights"
        refs = ("ready" if video.ready(refd, refe, mode="refs")
                else "missing weights")
        return f"**Turbo:** {turbo} · **References:** {refs}."

    status = gr.Markdown(_status())
    with gr.Row():
        diffusion = gr.Dropdown(choices=video.choices("diffusion"), value=d,
                                label="Turbo diffusion · GGUF")
        ref_diffusion = gr.Dropdown(choices=video.choices("ref_diffusion"),
                                    value=rd, label="Ref2VA diffusion · GGUF")
        encoder = gr.Dropdown(choices=video.choices("text_encoder"), value=e,
                              label="Shared H3 text encoder · GGUF")
    gr.Markdown(f"Video VAE · {video.VAE} — {video.VAE_SIZE / 1e9:.2f} GB "
                "(shared across Turbo and Ref2VA).")
    with gr.Row():
        scan = gr.Button("Load exact sizes", size="sm")
        download = gr.Button("⬇️ Download Turbo", variant="primary")
        download_refs = gr.Button("⬇️ Download Ref2VA", variant="primary")
    with gr.Row():
        delete = gr.Button("🗑️ Delete selected Turbo weights", size="sm")
        delete_refs = gr.Button("🗑️ Delete selected Ref2VA weights", size="sm")
    log = gr.Textbox(label="Video weight download log", lines=6,
                     autoscroll=True, elem_classes="log-box")

    def _scan(current_d, current_rd, current_e):
        try:
            ds = video.choices("diffusion", exact=True)
            rds = video.choices("ref_diffusion", exact=True)
            es = video.choices("text_encoder", exact=True)
            return (gr.update(choices=ds, value=current_d),
                    gr.update(choices=rds, value=current_rd),
                    gr.update(choices=es, value=current_e),
                    "Exact file sizes loaded from the repositories.")
        except Exception as exc:  # noqa: BLE001
            return gr.update(), gr.update(), gr.update(), f"Cannot load sizes: {exc}"

    scan.click(_scan, inputs=[diffusion, ref_diffusion, encoder],
               outputs=[diffusion, ref_diffusion, encoder, log])

    def _download(current_d, current_e, mode):
        lines: list[str] = []
        try:
            for message in video.download(current_d, current_e, log=lines.append,
                                          mode=mode):
                lines.append(message)
                yield "\n".join(lines), gr.update()
            prefs = settings.load_prefs()
            chosen = prefs.get("video_model_files") or {}
            if not isinstance(chosen, dict):
                chosen = {}
            chosen["diffusion" if mode == "turbo" else "ref_diffusion"] = current_d
            chosen["text_encoder"] = current_e
            prefs["video_model_files"] = chosen
            settings.save_prefs(prefs)
            yield "\n".join(lines), _status()
        except Exception as exc:  # noqa: BLE001
            yield "\n".join(lines + [f"Download failed: {exc}"]), gr.update()

    def _download_turbo(d, e):
        yield from _download(d, e, "turbo")

    def _download_refs(d, e):
        yield from _download(d, e, "refs")

    download.click(_download_turbo,
                   inputs=[diffusion, encoder], outputs=[log, status])
    download_refs.click(_download_refs,
                        inputs=[ref_diffusion, encoder], outputs=[log, status])

    def _delete(current_d, current_e, mode):
        try:
            count = video.delete(current_d, current_e, mode=mode)
            return (f"Deleted {count} selected file(s); shared weights "
                    "are kept while another H3 diffusion weight is installed.",
                    _status())
        except (ValueError, OSError) as exc:
            return f"Delete failed: {exc}", gr.update()

    delete.click(lambda d, e: _delete(d, e, "turbo"),
                 inputs=[diffusion, encoder], outputs=[log, status])
    delete_refs.click(lambda d, e: _delete(d, e, "refs"),
                      inputs=[ref_diffusion, encoder], outputs=[log, status])


def build_library_tab():
    with gr.Tab("📚 Model Catalog"):
        gr.Markdown("### Base models\nChoose the exact weights for each "
                    "model below, or keep the automatic VRAM/RAM choice. "
                    "Click **Load weights and sizes** to see the real "
                    "file sizes before downloading.")
        gr.Markdown(
            "> ℹ️ The quantization shown (Settings) is a **target**. If the "
            "repo doesn't offer it, the closest available quant **below** it "
            "is downloaded (to fit your VRAM) — shown in the log and flagged "
            "after the download. Text encoders are reused when a compatible "
            "weight is already installed. Flux, Qwen Image, Ming Image, Krea and Z-Image "
            "require different encoder weights, which cannot be combined.")

        prefs = settings.load_prefs()
        models = registry.load_base_models(prefs)
        recos = registry.recommend(prefs)

        cards: list[gr.Markdown] = []
        log = gr.Textbox(label="Download log", lines=8,
                         autoscroll=True, elem_classes="log-box")

        for m in models:
            with gr.Row():
                with gr.Column(scale=5):
                    card = gr.Markdown(_card_md(m, recos))
                    with gr.Group():
                        gr.Markdown("**Weights and file sizes**")
                        scan_btn = gr.Button("Load weights and sizes",
                                             size="sm")
                        d_menu, d_selected = _weight_menu(m, "diffusion")
                        e_menu, e_selected = _weight_menu(m, "text_encoder")
                        diffusion_file = gr.Dropdown(
                            choices=d_menu, value=d_selected,
                            label="Image model · weights",
                            interactive=True)
                        encoder_file = gr.Dropdown(
                            choices=e_menu, value=e_selected,
                            label="Text encoder · weights",
                            interactive=True)
                with gr.Column(scale=1, min_width=170):
                    btn = gr.Button("⬇️ Download", variant="primary")
                    del_btn = gr.Button("🗑️ Delete", size="sm")
            cards.append(card)

            def make_scanner(model_id):
                def scan():
                    model = registry.get_base_model(model_id, settings.load_prefs())
                    try:
                        variants = downloader.weight_choices(model)
                    except Exception as exc:  # noqa: BLE001
                        return gr.update(), gr.update(), \
                            f"Cannot read weight file sizes: {exc}"
                    if not variants.get("diffusion"):
                        return gr.update(), gr.update(), \
                            "No matching image-model weight was found in the repository."
                    d, ds = _weight_menu(model, "diffusion",
                                         variants.get("diffusion"))
                    e, es = _weight_menu(model, "text_encoder",
                                         variants.get("text_encoder"))
                    return (gr.update(choices=d, value=ds),
                            gr.update(choices=e, value=es),
                            "Actual file sizes loaded. Pick the weights "
                            "then click Download.")
                return scan

            def make_handler(model_id):
                def handler(diff_choice, enc_choice):
                    p = settings.load_prefs()
                    files = dict(p.get("model_files") or {})
                    selected = dict(files.get(model_id) or {})
                    for role, choice in (("diffusion", diff_choice),
                                         ("text_encoder", enc_choice)):
                        if choice and choice != _AUTO:
                            selected[role] = choice
                        else:
                            selected.pop(role, None)
                    if selected:
                        files[model_id] = selected
                    else:
                        files.pop(model_id, None)
                    p["model_files"] = files
                    settings.save_prefs(p)
                    model = registry.get_base_model(model_id, p)
                    lines: list[str] = []
                    for msg in downloader.download_model(model, log=lines.append):
                        lines.append(msg)
                        yield "\n".join(lines), gr.update()
                    yield "\n".join(lines), gr.update(
                        value=_card_md(model, registry.recommend(p)))
                    # Avertit visiblement si un quant a été remplacé par un repli.
                    if any("unavailable" in line for line in lines):
                        gr.Warning(t("Quantization adjusted: the repo doesn't "
                                     "offer the target quant, fell back to "
                                     "the closest available (see the log)."))
                return handler

            def make_deleter(model_id):
                def deleter():
                    p = settings.load_prefs()
                    model = registry.get_base_model(model_id, p)
                    deleted = registry.delete_model(model, p)
                    msg = (t("🗑️ “{name}” deleted: {n} file(s) removed.").format(name=model.name, n=len(deleted))
                           if deleted else
                           t("Nothing to delete for “{name}” (not installed "
                             "or shared files).").format(name=model.name))
                    return _card_md(model, registry.recommend(p)), msg
                return deleter

            scan_btn.click(make_scanner(m.id),
                           outputs=[diffusion_file, encoder_file, log])
            btn.click(make_handler(m.id),
                      inputs=[diffusion_file, encoder_file],
                      outputs=[log, card])
            del_btn.click(make_deleter(m.id), outputs=[card, log])

        refresh = gr.Button("↻ Refresh status")

        def refresh_cards():
            p = settings.load_prefs()
            r = registry.recommend(p)
            ups = [gr.update(value=_card_md(m, r))
                   for m in registry.load_base_models(p)]
            # Avec une seule carte, Gradio attend une valeur unique (pas une
            # liste), sinon la liste est passée telle quelle au Markdown.
            return ups[0] if len(ups) == 1 else ups

        refresh.click(refresh_cards, outputs=cards)

        _build_h3_catalog()

        gr.Markdown(
            "---\n*The tools (depth, background removal, SAM, prompt "
            "improver, enlargement) live in the Toolkit tab.*")
