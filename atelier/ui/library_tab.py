"""Onglet Catalogue de modèles : catalogue des modèles, recommandations selon le
matériel, téléchargement à la demande."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings, model_catalog, transfers, weight_precision
from . import widgets
from .media_tabs import stream_task
import math
from ..i18n import t

_AUTO = "__auto__"


def _selection_prefs(prefs, model, diff_choice, enc_choice, vae_choice=None):
    prefs = dict(prefs)
    files = dict(prefs.get("model_files") or {})
    selected = dict(files.get(model.id) or {})
    choices = [("diffusion", diff_choice), ("text_encoder", enc_choice)]
    if any(c.role == "vae" and c.alternatives for c in model.components):
        choices.append(("vae", vae_choice))
    for role, choice in choices:
        if choice and choice != _AUTO:
            selected[role] = choice
        else:
            selected.pop(role, None)
    if selected:
        files[model.id] = selected
    else:
        files.pop(model.id, None)
    prefs["model_files"] = files
    return prefs


def _weight_menu(model: registry.BaseModel, role: str,
                 variants: list[tuple[str, str]] | None = None):
    comp = next((c for c in model.components if c.role == role), None)
    if comp is None:
        return [], _AUTO
    prof = weight_precision.profile()
    auto_label = (f"Automatic (target: {comp.requested()})" if comp.token else
                  f"Default ({comp.alternatives[0].label})" if comp.alternatives else
                  f"Default ({comp.requested()})")
    menu = [(auto_label, _AUTO)]
    menu.extend(variants or [])
    if not variants:
        known = [(v.template, v.label) for v in comp.alternatives
                 if not any(token in v.template for token in ("{quant}", "{enc_quant}"))]
        known += [(name, "") for name in comp.selectable]
        seen = set()
        for name, label in known:
            if name in seen:
                continue
            seen.add(name)
            detail = weight_precision.describe(name, prof, label=label,
                recommended=weight_precision.is_target(comp, name))
            menu.append((f"{label or name} — {detail} (load sizes)", name))
    if comp.token is None and comp.requested() not in {v for _, v in menu}:
        # Un choix enregistré doit déjà figurer dans le menu avant même la
        # requête de métadonnées, sinon Gradio le réinitialise au lancement.
        menu.append((f"Selected: {comp.requested()} (size unavailable)",
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




def build_library_tab():
    def image_model(key):
        if not key or not key.startswith("image:"):
            return None
        return registry.get_base_model(key[6:], settings.load_prefs())

    def weight_updates(model, variants=None):
        updates=[]
        for role in ("diffusion", "text_encoder", "vae"):
            component=next((c for c in model.components if c.role == role),None) if model else None
            visible=bool(component and (role != "vae" or component.alternatives))
            choices,selected=_weight_menu(model,role,(variants or {}).get(role)) if visible else ([],None)
            updates.append(gr.update(choices=choices,value=selected,visible=visible))
        return updates

    def shared_panel(key, diff_choice=None, enc_choice=None, vae_choice=None):
        text=("**Shared files:** identical downloaded weights are reused by checksum. "
              "Consolidate existing installations once below. Unique weights and runtime versions are retained.")
        model=image_model(key)
        if model:
            if diff_choice is not None:
                prefs = _selection_prefs(settings.load_prefs(), model, diff_choice, enc_choice, vae_choice)
                model = registry.get_base_model(model.id, prefs)
            return (text+"\n\n"+weight_precision.model_advice(model)+"\n\n"
                    +model_catalog.component_details(model))
        return text

    with gr.Tab("Models", id="models"):
        gr.Markdown("### Models & toolkits\nChoose a package for image generation, Toolkit, layers, video or 3D. "
                    "Choose GGUF or approved INT8 / BF16 / FP16 / FP32 weights where published. "
                    "GPU advice distinguishes hardware support from memory needs. Load sizes for exact files.")
        prefs=settings.load_prefs()
        models=registry.load_base_models(prefs)
        initial="image:"+models[0].id if models else None
        category=gr.Radio(model_catalog.CATEGORIES,value="All",label="Usage")
        search=gr.Textbox(label="Find a model or tool",placeholder="SAM, upscale, Qwen, video…")
        package=gr.Dropdown(model_catalog.choices(),value=initial,label="Model or toolkit package",filterable=True)
        limit=gr.Number(value=prefs.get("download_limit_mbps",5),minimum=0,maximum=10000,
                        label="Download limit (MB/s; 0 = unlimited)",info="Shared by image and toolkit downloads.")
        model=image_model(initial)
        with gr.Group(visible=bool(model)) as image_group:
            card=gr.Markdown(_card_md(model,registry.recommend(prefs)) if model else "")
            scan=gr.Button("Load weights and sizes",size="sm")
            d_choices,d_value=_weight_menu(model,"diffusion") if model else ([],None)
            e_choices,e_value=_weight_menu(model,"text_encoder") if model else ([],None)
            diffusion=gr.Dropdown(d_choices,value=d_value,label="Image model · weights")
            encoder=gr.Dropdown(e_choices,value=e_value,label="Text encoder · weights")
            v_visible=bool(model and any(c.role == "vae" and c.alternatives for c in model.components))
            v_choices,v_value=_weight_menu(model,"vae") if v_visible else ([],None)
            vae=gr.Dropdown(v_choices,value=v_value,label="VAE · textures",visible=v_visible)
            with gr.Row():
                download=gr.Button("Download selected image weights",variant="primary")
                delete=gr.Button("Delete selected image model",size="sm")
        with gr.Group(visible=False) as tool_group:
            tool_note=gr.Markdown("")
            variant=gr.Dropdown([],value=None,label="Weight variant",visible=False)
            install_tool=gr.Button("Install / repair selected package",variant="primary")
        shared_note=gr.Markdown(shared_panel(initial))
        for control in (diffusion, encoder, vae):
            control.change(shared_panel,[package,diffusion,encoder,vae],shared_note,queue=False)
        with gr.Row():
            refresh=gr.Button("Refresh status",size="sm")
            consolidate=gr.Button("Consolidate existing identical weights",size="sm")
            stop=gr.Button("Stop download / installation",variant="stop")
        gr.Markdown("Consolidation reads existing weights and can take several minutes. "
                    "Only byte-identical files share storage; different formats and quantizations remain separate.")
        log=gr.Textbox(label="Download log",lines=8,autoscroll=True,elem_classes="log-box")

        def filter_packages(usage,query,current):
            choices=model_catalog.choices(usage,query)
            value=current if current in {v for _,v in choices} else (choices[0][1] if choices else None)
            return gr.update(choices=choices,value=value)
        category.change(filter_packages,[category,search,package],package,queue=False)
        search.input(filter_packages,[category,search,package],package,queue=False)

        def select_package(key):
            selected=image_model(key)
            entry=model_catalog.get(key)
            choices=list(entry.variants) if entry else []
            value=choices[0][1] if choices else None
            return (gr.update(visible=bool(selected)),gr.update(visible=bool(entry and not selected)),
                    _card_md(selected,registry.recommend(settings.load_prefs())) if selected else "",
                    *weight_updates(selected),model_catalog.details(key,value),
                    gr.update(choices=choices,value=value,visible=bool(choices)),shared_panel(key))
        package.change(select_package,package,
            [image_group,tool_group,card,diffusion,encoder,vae,tool_note,variant,shared_note],queue=False)
        variant.change(model_catalog.details,[package,variant],tool_note,queue=False)

        def scan_weights(key):
            selected=image_model(key)
            if not selected:
                return gr.update(),gr.update(),gr.update(),"Choose an image model first."
            try:
                variants=downloader.weight_choices(selected)
                return (*weight_updates(selected,variants),"Actual file sizes loaded. Pick weights, then download.")
            except Exception as exc:
                return gr.update(),gr.update(),gr.update(),f"Cannot read weight file sizes: {exc}"
        scan.click(scan_weights,package,[diffusion,encoder,vae,log])

        def handler(key,diff_choice,enc_choice,vae_choice=None):
            selected_model=image_model(key)
            if selected_model is None:
                raise gr.Error("Choose an image model first.")
            model_id=selected_model.id
            prefs=_selection_prefs(settings.load_prefs(),selected_model,diff_choice,enc_choice,vae_choice)
            settings.save_prefs(prefs)
            selected_model=registry.get_base_model(model_id,prefs)
            lines=[]
            with model_catalog.JOB.session(),transfers.cancel_scope(model_catalog.JOB.check):
                for message in downloader.download_model(selected_model,log=lines.append):
                    model_catalog.JOB.check()
                    lines.append(message)
                    yield "\n".join(lines),gr.update()
            yield "\n".join(lines),gr.update(value=_card_md(selected_model,registry.recommend(prefs)))
        download.click(handler,[package,diffusion,encoder,vae],[log,card],**widgets.GPU_QUEUE)

        def deleter(key):
            selected=image_model(key)
            if not selected:
                raise gr.Error("Choose an image model first.")
            prefs=settings.load_prefs()
            removed=registry.delete_model(selected,prefs)
            return _card_md(selected,registry.recommend(prefs)),f"Removed {len(removed)} file(s); required shared files are retained."
        delete.click(deleter,package,[card,log],**widgets.GPU_QUEUE)

        def refresh_status(key,choice):
            selected=image_model(key)
            return (_card_md(selected,registry.recommend(settings.load_prefs())) if selected else "",
                    model_catalog.details(key,choice),shared_panel(key))
        refresh.click(refresh_status,[package,variant],[card,tool_note,shared_note],queue=False)

        def install_catalog_package(key,choice):
            for status,_,lines in stream_task(lambda logger:model_catalog.install(key,choice,logger)):
                yield status+"\n"+lines,model_catalog.details(key,choice)
        install_tool.click(install_catalog_package,[package,variant],[log,tool_note],**widgets.GPU_QUEUE)

        def consolidate_catalog_weights():
            def action(logger):
                model_catalog.consolidate(logger)
            for status,_,lines in stream_task(action):
                yield status+"\n"+lines
        consolidate.click(consolidate_catalog_weights,outputs=log,**widgets.GPU_QUEUE)
        stop.click(model_catalog.JOB.cancel,outputs=log,queue=False)

        def set_download_limit(value):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 10000:
                raise gr.Error("Enter a download limit between 0 and 10000 MB/s.")
            prefs=settings.load_prefs()
            prefs["download_limit_mbps"]=value
            settings.save_prefs(prefs)
        limit.change(set_download_limit,limit,queue=False)
