"""Onglet « 🧊 Texte / Image → 3D » : un maillage 3D texturé (GLB), via le
binaire natif trellis-cli (trellis.cpp — C++/GGML/CUDA, no PyTorch).

One-shot : le process se termine et libère la VRAM (stratégie low-VRAM). Le mode
512 « light » vise les cartes ≤ 12 Go ; 1024/1536 demandent ~16 Go+.

**TEXTE → 3D est un ENCHAÎNEMENT, pas un modèle de plus**, et c'est aussi ce que
la doc de trellis.cpp entend par là : « optionally driven end-to-end from a text
prompt with stable-diffusion.cpp producing the input image ». TRELLIS ne lit que
des images. On fabrique donc l'image avec le modèle du catalogue, puis on la lui
donne.

Deux conséquences de conception, et les deux comptent :

  · **l'image intermédiaire est MONTRÉE**, pas cachée. C'est là que ça rate — un
    sujet mal cadré, une ombre portée, un décor — et un bouton unique qui
    enchaîne tout ferait payer un maillage complet pour le découvrir après. Elle
    atterrit dans le champ image : on peut la garder, la relancer, ou en charger
    une autre ;
  · **le prompt est réécrit** pour ce que TRELLIS sait traiter : un objet unique,
    centré, entier dans le cadre, sur fond neutre. Un prompt de belle photo
    donne un mauvais maillage, et ce n'est pas un réglage que l'utilisateur
    peut deviner. Voir `STUDIO_STYLE`, modifiable et affiché.
"""
from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
import time

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import sdcpp
from ..engine import trellis
from ..i18n import t
from . import widgets

#  Ce qu'on ajoute au prompt pour obtenir une image que TRELLIS sait traiter.
#
#  Ce n'est pas de la décoration. TRELLIS reconstruit UN objet : il détoure
#  l'image, la traite en carré, et suppose que ce qu'il voit est le sujet
#  entier. Un cadrage serré lui fait inventer ce qui dépasse ; une ombre portée
#  devient de la géométrie ; un décor devient du bruit sur le maillage. Les
#  termes ci-dessous adressent chacun un de ces échecs, et ils sont affichés
#  plutôt que codés en dur quelque part : c'est le genre de réglage qu'on veut
#  pouvoir corriger sans lire le source.
STUDIO_STYLE = ("{subject}, one object only, centred, entire object visible "
                "in frame, plain flat neutral grey background, even soft "
                "studio lighting, no cast shadow, no floor, no props, sharp "
                "focus, product photograph")


def studio_prompt(subject: str, template: str = STUDIO_STYLE) -> str:
    """Le prompt d'objet isolé, ou le sujet tel quel si le gabarit est vidé."""
    subject = (subject or "").strip()
    template = (template or "").strip()
    if not template:
        return subject
    if "{subject}" not in template:
        return f"{subject}, {template}"
    return template.format(subject=subject)


def _ready_models(prefs: dict) -> list[tuple[str, str]]:
    """Modèles d'image TÉLÉCHARGÉS. Proposer les autres serait un piège :
    on clique, et la génération échoue sur un fichier absent."""
    return [(m.name, m.id) for m in registry.load_base_models(prefs)
            if registry.model_is_ready(m)]


def _res_choices():
    return [(lbl, val) for lbl, val in trellis.RESOLUTIONS]


def _pad_to_square(img, mode: str = "white"):
    """Complète l'image en CARRÉ sans la déformer (sujet centré).

    TRELLIS.2 pré-traite l'entrée en carré : lui donner une image 16:9 revient
    à l'écraser horizontalement → modèle 3D déformé. On ajoute donc des bandes
    neutres (que le détourage retire) au lieu de laisser l'étirement se faire.
    """
    from PIL import Image as _PI
    w, h = img.size
    side = max(w, h)
    if mode == "transparent":
        src = img.convert("RGBA")
        canvas = _PI.new("RGBA", (side, side), (0, 0, 0, 0))
    else:
        src = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else \
            img.convert("RGB")
        fill = (255, 255, 255) if mode == "white" else (0, 0, 0)
        canvas = _PI.new("RGB", (side, side), fill)
    off = ((side - w) // 2, (side - h) // 2)
    # Le masque alpha évite un liseré noir quand la source est transparente.
    canvas.paste(src, off, src if src.mode == "RGBA" else None)
    return canvas


def _variant_choices() -> list[tuple[str, str]]:
    """Variantes de poids, en signalant celles qui ne sont pas installées."""
    installed = trellis.installed_variants()
    out = []
    for lbl, v in trellis.VARIANTS:
        out.append((lbl if v in installed else f"{lbl} — not installed", v))
    return out


def _default_variant() -> str:
    """Priorité à f16 : c'est le PLUS RAPIDE quand il tient en mémoire.

    En ggml les poids quantifiés sont déquantifiés à la volée pendant le calcul.
    Sur une charge compute-bound comme la 3D, ce surcoût n'est jamais amorti :
    q8/q4 sont plus LENTS que f16. Ils ne servent qu'à faire tenir un mode
    (1024/1536) qui déborderait autrement.
    """
    installed = trellis.installed_variants()
    for v in ("f16", "q8", "q4"):
        if v in installed:
            return v
    return "f16"


def _gpu_choices() -> list[tuple[str, int]]:
    from .. import hardware
    return [(f"#{g.index} — {g.label()}", g.index)
            for g in hardware.detect_gpus()]


def build_threed_tab(tab_id="threed", pending_3d=None, tabs=None,
                     parent_tabs=None):
    """`parent_tabs` : le groupe « 🧰 Outils » qui contient cet onglet (voir
    build_toolkit_tab — l'image doit atterrir dans un onglet VISIBLE)."""
    with gr.Tab("🧊 Text / Image → 3D", id=tab_id):
        ready = trellis.is_ready()
        # trellis.cpp n'est publié qu'en binaire Windows CUDA. Sur Mac (et sur
        # une machine sans NVIDIA), mieux vaut le dire tout de suite que laisser
        # cliquer sur un installeur qui ne trouvera rien.
        if platform.system() == "Darwin":
            gr.Markdown(
                "### Image → 3D model (GLB)\n> ⛔ **Unavailable on macOS.** "
                "trellis.cpp ships as a **Windows CUDA binary**\nonly: there "
                "is no Apple Silicon build and no Metal path. This is not a "
                "setting\nto find, it is an engine that does not exist for "
                "this platform.\n\nEverything else in the application works: "
                "generation, outpaint, the Toolkit and\nupscaling all go "
                "through stable-diffusion.cpp, which does have a Metal build.")
            return
        gr.Markdown(
            "### Image → 3D model (GLB)\nTurns an image into a **textured 3D "
            "mesh** (GLB) through **trellis.cpp**\n(TRELLIS.2, a native CUDA "
            "binary — no PyTorch). Load a sharp image of a\n**single object** "
            "on a simple background; background removal is automatic. "
            "The\ntrellis server **starts and then stops** for each "
            "generation, so all the VRAM\nis released afterwards (the "
            "low-VRAM strategy).\n\n💡 **Stay at 512 below 16 GB of VRAM.** "
            "The **1024/1536** cascade is documented\nfor a **16 GB** card: "
            "below that it does not fail cleanly, it **degrades "
            "the\ncomputation** and returns a mesh made of **“blobs”**. No "
            "setting works around\nit (trellis has neither offload nor "
            "tiling).\n👉 To gain quality **without touching the resolution**, "
            "raise the **UV atlas**\n(2048/4096) and the **decimation**: a "
            "well-textured 512 mesh beats a botched\n1024 one, at almost no "
            "VRAM cost.")

        # ---- Installation (binaire + modèles) ----
        with gr.Accordion("⚙️ Install trellis.cpp (binary + models, one click)",
                          open=not ready):
            gr.Markdown(
                "Downloads the **Windows CUDA binary** "
                "(`pwilkin/trellis.cpp`, ~700 MB) into\n`bin/trellis/` and a "
                "**set of GGUF models** (`ilintar/trellis2-gguf`) "
                "into\n`models/trellis/`.\n\n**Weight variant** — **f16 is "
                "the FASTEST** when it fits in memory: keep it for\n512. The "
                "quantized versions (q8/q4) use far less memory — which can "
                "make\n**1024/1536 reachable** — but they are **SLOWER** "
                "(weights are dequantized on\nthe fly at every computation, "
                "an overhead that a 3D workload does not amortize).\nYou can "
                "install several and switch at generation time.")
            diag_md = gr.Markdown(trellis.diagnose())
            diag_btn = gr.Button("↻ Check the installation", size="sm")
            inst_variant = gr.Radio(
                [(lbl, v) for lbl, v in trellis.VARIANTS], value="f16",
                label="Variant to install",
                info="Start with f16 (the fastest). Only add q8/q4 if you "
                     "want to attempt 1024/1536.")
            inst_log = gr.Textbox(label="Install log", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            inst_btn = gr.Button("⬇️ Install trellis.cpp (binary + models)")
            gr.Markdown(
                "⬆️ **Update the binary** — the installation above **does not "
                "replace** a binary\nthat is already there: once trellis is "
                "installed it stays as-is indefinitely.\nThis does **not** "
                "download the ~10 GB of models again.\n\nThe **archive is "
                "chosen from your card**. Since **v0.6.0** (August "
                "2026)\ntrellis.cpp publishes two: `cuda` for **Turing and "
                "newer** (RTX 20xx → 50xx)\nand `cuda12` for **Pascal and "
                "Volta** (GTX 10xx). Before that a single build\nexisted and "
                "covered only the RTX 30xx/50xx — hence the systematic "
                "fallback to\nVulkan. **Vulkan remains the fallback** for any "
                "card neither list covers: it\ncompiles nothing per "
                "architecture and works everywhere.",
                elem_classes="hint")
            upd_btn = gr.Button("⬆️ Update the binary (models untouched)",
                                size="sm")

            def _run_installer(args: list[str], first_msg: str):
                """Lance get_trellis.py en streamant son journal."""
                cmd = [sys.executable,
                       str(settings.ROOT / "scripts" / "get_trellis.py")] + args
                logs: list[str] = []
                yield first_msg
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(settings.ROOT),
                    env=settings.child_env(),
                    encoding="utf-8", errors="replace")
                assert proc.stdout is not None
                for line in proc.stdout:
                    logs.append(line.rstrip("\n"))
                    yield "\n".join(logs[-400:])
                proc.wait()
                logs.append("\n✅ Done." if trellis.is_ready()
                            else "\n⚠️ Incomplete installation — see above.")
                yield "\n".join(logs[-400:])

            def _install(inst_var):
                yield from _run_installer(
                    ["--variant", str(inst_var or "f16")],
                    t("⏳ Installing (binary + ~10 GB of models)…"))

            def _update_binary():
                # Le serveur résident VERROUILLE l'exécutable sous Windows :
                # sans cet arrêt, l'extraction échouerait sur un « accès refusé »
                # difficile à relier à sa cause.
                if trellis.resident_is_running():
                    yield t("⏹️ Stopping the resident server (it locks the "
                            "binary)…")
                    trellis.resident_stop()
                yield from _run_installer(
                    ["--binary", "--force"],
                    t("⏳ Downloading the latest trellis binary…"))

            inst_evt = inst_btn.click(_install, inputs=[inst_variant],
                                      outputs=[inst_log])
            upd_btn.click(_update_binary, outputs=[inst_log])

        # ---- Génération ----
        _prefs = settings.load_prefs()
        _models = _ready_models(_prefs)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(t(
                    "**Describe an object, or load an image.** TRELLIS only "
                    "reads images, so a prompt\nis turned into one first — "
                    "with the model you already have — and you get to **see "
                    "that\nimage** before it becomes a mesh. That is where "
                    "it goes wrong, and it is cheap to redo."))
                prompt = gr.Textbox(
                    label="Object to model (leave empty to use an image)",
                    lines=2, placeholder=t("a weathered bronze dragon statue"))
                with gr.Row():
                    gen_model = gr.Dropdown(
                        _models or [(t("No model downloaded"), "")],
                        value=(_models[0][1] if _models else ""),
                        scale=2, label="Model used for the image")
                    img_seed = gr.Number(
                        value=-1, precision=0, scale=1,
                        label="Image seed (-1 = random)")
                image = gr.Image(label="Input image (a single object)",
                                 type="filepath",
                                 buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    square_pad = gr.Checkbox(
                        value=True, scale=2,
                        label="Pad to square (keeps the proportions)",
                        info="TRELLIS processes its input as a square: "
                             "without this, a non-square image comes out "
                             "DISTORTED.")
                    pad_color = gr.Dropdown(
                        [("Blanc", "white"), ("Noir", "black"),
                         ("Transparent", "transparent")],
                        value="white", scale=1, label="Bars added")
                with gr.Row():
                    res = gr.Radio(_res_choices(), value=512, scale=2,
                                   label="Geometry resolution")
                    variant = gr.Dropdown(
                        _variant_choices(), value=_default_variant(), scale=1,
                        label="Weights used",
                        info="f16 = the FASTEST when it fits. q8/q4 = less "
                             "memory but SLOWER (dequantized on the fly) — "
                             "keep them for 1024/1536.")
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = random)")
                    bg = gr.Dropdown(
                        [("BiRefNet (quality, recommended)", "birefnet"),
                         ("Seuil (rapide)", "threshold")],
                        value="birefnet", label="Background removal")
                with gr.Accordion("⚡ Resident server (for a run of 3D objects)", open=False):
                    gr.Markdown(
                        "By default the server **starts and stops** for each "
                        "generation (VRAM is\nreleased). Ticked, it **stays "
                        "alive**: the next 3D objects skip the model\nreload "
                        "(~30 s saved), but **the VRAM stays occupied** — "
                        "stop it before\ngenerating images.\nℹ️ If you change "
                        "a **launch** setting (resolution, decimation, atlas, "
                        "GPU,\ntexture…), the server **restarts "
                        "automatically** to apply it — only seed "
                        "and\nbackground removal can change without a reload.")
                    resident = gr.Checkbox(
                        value=False,
                        label="Keep the server resident between generations")
                    with gr.Row():
                        res_status = gr.Markdown(trellis.resident_status())
                        res_stop_btn = gr.Button("⏹️ Stop the resident server",
                                                 size="sm")
                with gr.Accordion("🎛️ Quality / mesh", open=False):
                    with gr.Row():
                        decim = gr.Number(
                            value=0, precision=0,
                            label="Decimation — target faces (0 = default)",
                            info="Lower = lighter mesh.")
                        atlas = gr.Dropdown(
                            [("Default", 0), ("1024 px", 1024),
                             ("2048 px", 2048), ("4096 px", 4096)],
                            value=0, label="UV atlas size (texture)")
                    with gr.Row():
                        no_texture = gr.Checkbox(
                            value=False,
                            label="Geometry only (no texture, faster)")
                        box_uv = gr.Checkbox(value=False,
                                             label="“Box” UV unwrap")
                with gr.Accordion("🩺 Engine (troubleshooting)", open=False):
                    gr.Markdown(
                        "**Card used** — passed to the engine through its own "
                        "`--gpu N` flag. Pick the card with the most VRAM "
                        "(1024 mode wants ~16 GB).")
                    gpu_pick = gr.Dropdown(
                        [(t("Engine default (card 0)"), -1)] + _gpu_choices(),
                        value=(_gpu_choices()[0][1] if _gpu_choices() else -1),
                        label="Card used for 3D")
                    band = gr.Number(
                        value=0, precision=4,
                        label="Offset de remaillage « band » (0 = auto)",
                        info="v0.5.4 adapts it to the resolution (fixes the "
                             "1024 speckles). Change it only when "
                             "troubleshooting.")
                    with gr.Row():
                        require_gpu = gr.Checkbox(
                            value=True,
                            label="Require the GPU (prevents a very slow CPU fallback)")
                        f32 = gr.Checkbox(value=False,
                                          label="f32 precision (instead of f16)")
                        no_fa = gr.Checkbox(value=False,
                                            label="Disable FlashAttention")
                with gr.Accordion("Advanced options", open=False):
                    style_tpl = gr.Textbox(
                        value=STUDIO_STYLE, lines=3,
                        label="How the prompt is rewritten for TRELLIS",
                        info=t("`{subject}` is replaced by what you typed. "
                               "Each term here fixes one way a mesh comes "
                               "out wrong: a tight crop makes TRELLIS invent "
                               "what is cut off, a cast shadow becomes "
                               "geometry, a background becomes noise. Empty "
                               "it to send your prompt untouched."))
                    extra = gr.Textbox(
                        label="Extra trellis-server arguments (optional)",
                        placeholder="e.g. extra server flags")
                with gr.Row():
                    run = gr.Button("🧊 Generate the 3D", variant="primary",
                                    scale=3)
                    stop = gr.Button("⏹️ Cancel", variant="stop", scale=1)
                status = gr.Markdown("")
            with gr.Column(scale=4):
                model3d = gr.Model3D(label="3D preview (GLB)", clear_color=[
                    0.1, 0.1, 0.12, 1.0])
                glb_file = gr.File(label="GLB file", interactive=False)
                with gr.Row():
                    seed_used = gr.Textbox(
                        label="Seed used (to replay this object)",
                        interactive=False, buttons=widgets.TEXT_COPY, scale=2)
                    seed_reuse = gr.Button("♻️ Reuse this seed", size="sm",
                                           scale=1)
                log = gr.Textbox(label="Log", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        def _image_from_prompt(text, model_id, tpl, seed_val, say):
            """Fabrique l'image de départ à partir du prompt. Renvoie son chemin.

            Le serveur 3D résident est arrêté AVANT : il garde son modèle en
            VRAM, et la génération d'image tomberait sur une carte déjà pleine
            — avec un manque de mémoire dont la cause serait l'onglet 3D
            lui-même, ce qui est exactement le genre de panne qu'on ne relie
            jamais à sa cause.
            """
            if not model_id:
                raise gr.Error(t("No image model is downloaded. Open "
                                 "📚 Model catalog, or load an image "
                                 "instead."))
            if trellis.resident_is_running():
                say(t("⏹️ Stopping the 3D server first — it is holding the "
                      "card."))
                say(trellis.resident_stop())
            prefs = settings.load_prefs()
            model = registry.get_base_model(model_id, prefs)
            if model is None:
                raise gr.Error(t("Unknown model: {m}").format(m=model_id))
            d = dict(model.defaults)
            full = studio_prompt(text, tpl)
            say(f"🖼️ Image prompt: {full}")
            try:
                seed = int(seed_val)
            except (TypeError, ValueError):
                seed = -1
            #  CARRÉ, et pas la résolution par défaut du modèle. TRELLIS
            #  traite son entrée en carré de toute façon : lui donner du 16:9
            #  revient à choisir nous-mêmes ce qui sera rogné ou déformé.
            outs = gen_engine.generate(
                model_id=model_id, prompt=full, negative="",
                steps=int(d.get("steps", 8) or 8),
                cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
                width=1024, height=1024, seed=seed, batch_count=1,
                sampler=d.get("sampler") or "euler",
                schedule=("" if d.get("scheduler") in (None, "", "auto")
                          else d["scheduler"]),
                log=say)
            if not outs:
                raise gr.Error(t("The image could not be generated."))
            return str(outs[0])

        def do_generate3d(image_path, prompt_val, model_val, img_seed_val,
                          style_val,
                          square_val, pad_val, res_val,
                          variant_val, band_val, seed_val,
                          bg_val, resident_val,
                          decim_val, atlas_val, no_tex_val, box_uv_val,
                          gpu_val, req_gpu_val, f32_val, no_fa_val,
                          extra_args):
            if not image_path and not (prompt_val or "").strip():
                raise gr.Error(t("Describe an object, or load an image."))
            if not trellis.is_ready():
                raise gr.Error(t("trellis.cpp is not installed — open "
                                 "“Install trellis.cpp” above."))
            settings.ensure_dirs()
            logs: list[str] = []

            def _out(status=None, img=None):
                """Un tuple de sortie complet, pour ne pas compter à la main.

                Sept composants, et une seule position qui change à chaque
                fois : les recopier en toutes lettres à chaque `yield` est
                exactement comment on en décale un.
                """
                return (gr.update() if status is None else status,
                        gr.update(), gr.update(),
                        "\n".join(logs[-500:]),
                        gr.update(), gr.update(),
                        gr.update() if img is None else gr.update(value=img))

            #  TEXTE → 3D. Le fil séparé n'est pas du zèle : `generate` bloque
            #  jusqu'à l'image, et sans lui le journal n'apparaîtrait qu'à la
            #  fin — sur une minute de calcul, ça ressemble à un plantage.
            if not image_path:
                pq: "queue.Queue[str | None]" = queue.Queue()
                pstate: dict = {}

                def prompt_worker():
                    try:
                        pstate["path"] = _image_from_prompt(
                            prompt_val, model_val, style_val, img_seed_val,
                            pq.put)
                    except Exception as exc:  # noqa: BLE001
                        pstate["err"] = str(exc)
                    finally:
                        pq.put(None)

                threading.Thread(target=prompt_worker, daemon=True).start()
                yield _out(t("⏳ Making the starting image…"))
                while True:
                    line = pq.get()
                    if line is None:
                        break
                    logs.append(line)
                    yield _out()
                if "err" in pstate:
                    logs.append(f"\n[ERROR] {pstate['err']}")
                    yield _out(t("❌ The image failed — see the log."))
                    return
                image_path = pstate["path"]
                #  Montrée tout de suite : c'est le moment où l'on décide de
                #  relancer plutôt que de payer un maillage pour rien.
                yield _out(t("🖼️ Image ready — now the mesh."), image_path)

            # Normalise l'entrée en PNG (trellis-cli attend un fichier image).
            from PIL import Image as _PI
            in_png = settings.TMP_DIR / "trellis_in.png"
            try:
                src = _PI.open(image_path)
                w0, h0 = src.size
                if square_val and w0 != h0:
                    src = _pad_to_square(src, pad_val or "white")
                elif src.mode not in ("RGB", "RGBA"):
                    src = src.convert("RGB")
                src.save(in_png)
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(t("Image illisible : {e}").format(e=exc))
            out_glb = settings.OUTPUT_DIR / \
                f"trellis-{time.strftime('%Y%m%d-%H%M%S')}.glb"

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            try:
                _seed = int(seed_val)
            except (TypeError, ValueError):
                _seed = -1

            meta: dict = {}

            def worker():
                try:
                    _gpu = None if (gpu_val is None or int(gpu_val) < 0) \
                        else int(gpu_val)
                    trellis.generate(in_png, out_glb, res=int(res_val),
                                     seed=(None if _seed < 0 else _seed),
                                     bg_removal=bg_val or "birefnet",
                                     resident=bool(resident_val),
                                     gpu_index=_gpu,
                                     decim=int(decim_val or 0),
                                     atlas=int(atlas_val or 0),
                                     no_texture=bool(no_tex_val),
                                     box_uv=bool(box_uv_val),
                                     require_gpu=bool(req_gpu_val),
                                     f32=bool(f32_val), no_fa=bool(no_fa_val),
                                     variant=variant_val or "f16",
                                     band=float(band_val or 0),
                                     extra=extra_args or "", log=q.put,
                                     meta=meta)
                    state["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            # (status, model3d, glb_file, log, res_status, seed_used, image)
            yield _out(t("⏳ Generating 3D ({r} mode)…").format(r=res_val))
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield _out()

            if "err" in state:
                logs.append(f"\n[ERROR] {state['err']}")
                yield (t("❌ Failed — see the log."), gr.update(),
                       gr.update(), "\n".join(logs),
                       gr.update(value=trellis.resident_status()), gr.update(),
                       gr.update())
                return
            yield (t("✅ 3D generated: {name}").format(name=out_glb.name),
                   gr.update(value=str(out_glb)), gr.update(value=str(out_glb)),
                   "\n".join(logs), gr.update(value=trellis.resident_status()),
                   gr.update(value=str(meta.get("seed", ""))), gr.update())

        gen_evt = run.click(
            do_generate3d,
            inputs=[image, prompt, gen_model, img_seed, style_tpl,
                    square_pad, pad_color, res, variant, band,
                    seed, bg, resident,
                    decim, atlas, no_texture,
                    box_uv, gpu_pick, require_gpu, f32, no_fa, extra],
            outputs=[status, model3d, glb_file, log, res_status, seed_used,
                     image])
        widgets.stop_into_status(stop, sdcpp.cancel_active, status, [gen_evt])

        def _reuse_seed(v):
            try:
                return gr.update(value=int(str(v).strip()))
            except (TypeError, ValueError):
                return gr.update()

        seed_reuse.click(_reuse_seed, inputs=[seed_used], outputs=[seed])

        # Diagnostic d'installation : câblé ICI car il rafraîchit aussi le
        # sélecteur « variant », créé plus bas dans la mise en page.
        def _refresh_diag():
            return (gr.update(value=trellis.diagnose()),
                    gr.update(choices=_variant_choices(),
                              value=_default_variant()))

        diag_btn.click(_refresh_diag, outputs=[diag_md, variant])
        inst_evt.then(_refresh_diag, outputs=[diag_md, variant])

        def _stop_resident():
            msg = trellis.resident_stop()
            return gr.update(value=trellis.resident_status()), msg

        res_stop_btn.click(_stop_resident, outputs=[res_status, status])

        # --- Réception d'une image envoyée depuis un onglet de génération ---
        if pending_3d is not None and tabs is not None:
            def _consume3d(pend):
                if not pend:
                    return gr.update(), None, gr.update()
                top = (gr.Tabs(selected=tab_id) if parent_tabs is not None
                       else gr.update())
                return gr.update(value=pend), None, top

            tabs.select(_consume3d, inputs=[pending_3d],
                        outputs=[image, pending_3d,
                                 parent_tabs if parent_tabs is not None
                                 else image])
