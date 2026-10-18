"""One download catalog for image models, Toolkit models and media dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from . import addons, downloader, model_store, registry, settings, transfers, moss_dmd
from .engine import tools
from .engine.local_jobs import LocalJob

JOB = LocalJob()
CATEGORIES = ["All", "Image generation", "Toolkit", "Layers & SVG", "Video & 3D"]


@dataclass(frozen=True)
class Package:
    id: str
    name: str
    category: str
    note: str
    variants: tuple = ()


def packages(prefs=None):
    images = [Package("image:"+m.id, m.name, "Image generation", m.description)
              for m in registry.load_base_models(prefs or settings.load_prefs())]
    toolkit = [
        ("depth", "Depth Anything V2 Small", "Depth and normals; one shared installation for all depth tools."),
        ("bg", "RMBG 1.4", "Background removal. Includes the upstream Python model code."),
        ("describe", "Image description · Qwen2.5-VL 3B", "One model for Image → prompt and multi-view captions (~7.5 GB)."),
        ("enhance", "Prompt enhancer · Qwen2.5 3B", "Prompt enhancement and Xanax use the same local language model (~6 GB)."),
        ("clip", "CLIP · region naming", "Optional semantic naming/merging for SAM layers (~600 MB)."),
        ("sam", "SAM 1 base · legacy", "Legacy click segmentation and masks (~375 MB). SAM 2 is listed under Layers & SVG."),
        ("face", "Face restoration pack", "GFPGAN, RestoreFormer++, CodeFormer and their shared detector/parser. CodeFormer is non-commercial."),
        ("upscale", "Creative upscale · SDXL", "SDXL base, fp16-fix VAE and optional Tile ControlNet; existing checkpoints are retained."),
        ("adetailer", "ADetailer detectors", "Face/person detectors, converted locally for the image engine."),
        ("upscalers", "ESRGAN upscalers · GGUF", "Shared ESRGAN weights for the image engine; no extra image-generation model."),
        ("modern", "Modern upscalers · Spandrel", "Shared modern upscale weights and their runtime (SwinIR / RealPLKSR / DAT)."),
    ]
    return images + [Package("moss:dmd", "MOSS SFT · DMD 8-step adapter", "Image generation",
                            "Optional 1.19 GB LoRA for MOSS-Image SFT. Select DMD fast (8 steps) in Images after downloading. "
                            "Reuses the installed image model, encoder and VAE; no second base checkpoint.")] + [Package("tool:"+key,name,"Toolkit",note) for key,name,note in toolkit] + [
        Package("addon:sam2","SAM 2.1 small","Layers & SVG","Click cut-out and automatic/manual layers share the same 184 MB checkpoint."),
        Package("addon:qwen-layered","Qwen-Image-Layered Q4","Layers & SVG","RGBA decomposition: selected Q4 transformer + text encoder + VAE (~30 GB). Separate from Qwen Image Turbo."),
        Package("addon:svg","VTracer / LayerTracer · CPU","Layers & SVG","CPU runtime and official source only; no neural weights."),
        Package("addon:hivg","HiVG · experimental","Layers & SVG","Pinned HiVG weights (~8.14 GB) and isolated inference runtime. Public compressed outputs can be undecodable."),
        Package("addon:omnisvg","OmniSVG 1.1 4B","Layers & SVG","SVG model + Qwen processor (~7.71 GB); base Qwen weights are not downloaded."),
        Package("seed:convrot","SeedVR2 · ConvRot","Video & 3D","Installs the native ConvRot runtime and only the selected model. All SeedVR2 variants reuse one VAE.",
                tuple((key,key) for key in addons.SEED_CONVROT_MODELS)),
        Package("seed:classic","SeedVR2 · classic GGUF","Video & 3D","Legacy GGUF runtime, selected model and the same shared SeedVR2 VAE.",tuple((key,key) for key in addons.SEED_MODELS)),
        Package("video:ltx","LTX 2.5 dev","Video & 3D","Selected diffusion + convolutional VAE + Gemma encoder converted to Q4. Needs sd.cpp and substantial free disk space.",
                (("Q3_K_M (smaller)","Q3_K_M"),("Q4_K_M","Q4_K_M"))),
        Package("3d:trellis","TRELLIS.2","Video & 3D","3D engine and selected weight set; decoders/DINO are reused by Pixal3D.",(("Q4","q4"),("Q8","q8"),("F16","f16"))),
        Package("3d:pixal","Pixal3D","Video & 3D","3D engine, selected Pixal3D flows and required TRELLIS shared components.",
                tuple((f"{v.upper()} · {r}",f"{v}:{r}") for v in ("q4","q8","f16") for r in (512,1024))),
        Package("addon:splat","Gaussian splats · Brush","Video & 3D","Brush runtime only; no pretrained model weights."),
    ]


def get(key):
    return next((p for p in packages() if p.id == key), None)


def choices(category="All", query=""):
    query = (query or "").strip().casefold()
    return [(f"{p.name} · {p.category}",p.id) for p in packages()
            if (category == "All" or p.category == category) and query in (p.name+" "+p.note).casefold()]


def _variant(package, variant):
    if package.variants:
        if variant not in {v for _,v in package.variants}:
            raise ValueError("Choose a listed model variant.")
    elif variant not in (None, ""):
        raise ValueError("This package has no weight variant selector.")


def ready(key, variant=None):
    p = get(key)
    if not p:
        return False
    variant = variant or (p.variants[0][1] if p.variants else None)
    if p.variants and variant not in {v for _,v in p.variants}:
        return False
    if key.startswith("image:"):
        return registry.model_is_ready(registry.get_base_model(key[6:],settings.load_prefs()))
    if key == "moss:dmd":
        return moss_dmd.ready()
    if key.startswith("tool:"):
        name=key[5:]
        if name in {"modern","upscalers"}:
            entries = registry.modern_upscaler_files() if name == "modern" else registry.upscaler_config().get("files",[])
            return bool(entries) and all((registry.upscalers_dir() / (e['file'] if isinstance(e,dict) else e)).is_file() for e in entries)
        return bool(getattr(tools,name+"_is_installed")())
    if key.startswith("addon:"):
        return addons.ready(key[6:])
    if key.startswith("seed:"):
        addon = "seedvr2-convrot" if key.endswith("convrot") else "seedvr2"
        filename = (addons.SEED_CONVROT_MODELS if key.endswith("convrot") else addons.SEED_MODELS).get(variant)
        folder=addons.root("seedvr2") / "models"
        return addons.ready(addon) and bool(filename) and all((folder / f).is_file() for f in (filename,"ema_vae_fp16.safetensors"))
    if key == "video:ltx":
        from .engine import ltx25
        return ltx25.ready(variant)
    from .engine import trellis
    if key == "3d:trellis":
        return trellis.is_ready() and trellis.models_ready(variant)
    if key == "3d:pixal":
        v,res=variant.split(":")
        return trellis.is_ready() and trellis.pixal_ready(v,int(res))
    return False


def details(key, variant=None):
    p=get(key)
    if not p:
        return "Choose a model or toolkit package."
    state="Installed" if ready(key,variant) else "Installation / download needed"
    return f"### {p.name}\n**{state}** · {p.category}\n\n{p.note}\n\n" \
           "Existing compatible weights are retained. Runtime packages may need additional disk space."


def component_details(model):
    all_models=registry.load_base_models(settings.load_prefs())
    lines=["| Component | Selected weight | Used by |", "| --- | --- | --- |"]
    for c in model.components:
        users=[m.name for m in all_models if any(x.repo == c.repo and x.base_glob() == c.base_glob() for x in m.components)]
        file=registry.resolve_component_path(c)
        label=(file.name+" · installed") if file else c.requested()
        lines.append(f"| {c.role} | `{label}` | {', '.join(users)} |")
    return "\n".join(lines)


def install(key, variant=None, log=print):
    p=get(key)
    if p is None or key.startswith("image:"):
        raise ValueError("Choose a toolkit/media package; image weights use their dedicated controls.")
    _variant(p,variant)
    command=None
    with JOB.session(), transfers.cancel_scope(JOB.check):
        if key == "moss:dmd":
            moss_dmd.install(log)
        elif key.startswith("tool:"):
            name=key[5:]
            if name in {"modern","upscalers"}:
                if name == "modern":
                    JOB.run([sys.executable,"-u",settings.ROOT/"scripts/setup_tools.py","spandrel"],log,
                            env=settings.child_env(settings.generation_gpu_index(settings.load_prefs())))
                for message in (downloader.download_modern_upscalers(log) if name == "modern" else downloader.download_upscalers(log)):
                    JOB.check();log(message)
            else:
                command=["setup_adetailer.py"] if name == "adetailer" else ["setup_tools.py",name]
        elif key.startswith("addon:"):
            name=key[6:]
            command=["setup_layers.py",name] if name in {"sam2","qwen-layered"} else (
                     ["setup_media.py",name] if name == "splat" else ["setup_svg.py",name])
        elif key.startswith("seed:"):
            addon="seedvr2-convrot" if key.endswith("convrot") else "seedvr2"
            command=["setup_media.py",addon]
            if key.endswith("convrot"):
                command += ["--precision", "int8" if variant.endswith("int8") else "w4a8"]
        elif key == "video:ltx":
            command=["setup_media.py","ltx25","--quant",variant]
        elif key.startswith("3d:"):
            from .engine import trellis
            if trellis.resident_is_running():
                trellis.resident_stop()
            command=["get_trellis.py","--variant",variant] if key.endswith("trellis") else [
                "get_trellis.py","--pixal3d","--variant",variant.split(":")[0],"--resolution",variant.split(":")[1]]
        if command:
            env=settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
            if command[0] == "setup_media.py" and command[1].startswith("seedvr2"):
                from . import seedvr2_acceleration
                env=seedvr2_acceleration.triton_env(env,addons.root(command[1])/".venv/Lib/site-packages")
                env["PYTHONPATH"]=str(settings.ROOT)
            JOB.run([sys.executable,"-u",settings.ROOT/"scripts"/command[0],*command[1:]],log,
                    env=env)
        if key.startswith("seed:"):
            name=(addons.SEED_CONVROT_MODELS if key.endswith("convrot") else addons.SEED_MODELS)[variant]
            repo="ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8" if key.endswith("convrot") else "AInVFX/SeedVR2_comfyUI"
            for source,filename in ((repo,name),("numz/SeedVR2_comfyUI","ema_vae_fp16.safetensors")):
                JOB.check()
                transfers.hf_hub_download(repo_id=source,filename=filename,local_dir=str(addons.root("seedvr2")/"models"))
        JOB.check()
        if not ready(key,variant):
            raise RuntimeError("The package is incomplete. See the install log and retry to resume missing files.")
    log("Selected package is installed. Refresh the tool page or restart the interface if its controls were already open.")


def consolidate(log=print):
    with JOB.session():
        result=model_store.consolidate(log, JOB.check)
    log(f"Shared {result['linked_files']} identical file(s); released {result['saved_bytes']/1e9:.2f} GB. "
        f"Kept separate: {result['separate_files']} (different volumes or hard links unavailable).")
    return result
