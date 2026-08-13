"""Onglet « 🧊 Image → 3D » : génère un maillage 3D texturé (GLB) à partir d'une
image, via le binaire natif trellis-cli (trellis.cpp — C++/GGML/CUDA, no PyTorch).

One-shot : le process se termine et libère la VRAM (stratégie low-VRAM). Le mode
512 « light » vise les cartes ≤ 12 Go ; 1024/1536 demandent ~16 Go+.
"""
from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
import time

import gradio as gr

from .. import settings
from ..engine import sdcpp
from ..engine import trellis
from ..i18n import t
from . import widgets


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
        out.append((lbl if v in installed else f"{lbl} — non installé", v))
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
    with gr.Tab("🧊 Image → 3D", id=tab_id):
        ready = trellis.is_ready()
        # trellis.cpp n'est publié qu'en binaire Windows CUDA. Sur Mac (et sur
        # une machine sans NVIDIA), mieux vaut le dire tout de suite que laisser
        # cliquer sur un installeur qui ne trouvera rien.
        if platform.system() == "Darwin":
            gr.Markdown(
                "### Image → modèle 3D (GLB)\n"
                "> ⛔ **Indisponible sur macOS.** trellis.cpp n'est publié qu'en "
                "**binaire Windows CUDA** : il n'existe ni build Apple Silicon "
                "ni chemin Metal. Ce n'est pas un réglage à trouver, c'est le "
                "moteur qui n'existe pas pour cette plateforme.\n\n"
                "Tout le reste de l'application fonctionne : génération, "
                "outpaint, Toolkit et agrandissement passent par "
                "stable-diffusion.cpp, qui a bien une build Metal.")
            return
        gr.Markdown(
            "### Image → modèle 3D (GLB)\n"
            "Transforme une image en **maillage 3D texturé** (GLB) via "
            "**trellis.cpp** (TRELLIS.2, binaire natif CUDA — aucun PyTorch). "
            "Chargez une image nette d'un **objet unique** sur fond simple ; le "
            "détourage est automatique. Le serveur trellis **démarre puis "
            "s'arrête** à chaque génération → toute la VRAM est libérée ensuite "
            "(stratégie low-VRAM).\n\n"
            "💡 **Reste en 512 sous 16 Go de VRAM.** Le cascade **1024/1536** "
            "est documenté pour une carte **16 Go** : en dessous il ne plante "
            "pas proprement, il **dégrade le calcul** et sort un maillage "
            "**en « blobs »**. Aucun réglage ne contourne ça (trellis n'a ni "
            "offload ni tiling).  \n"
            "👉 Pour gagner en qualité **sans toucher à la résolution**, monte "
            "l'**atlas UV** (2048/4096) et la **décimation** : une géométrie "
            "512 bien texturée bat un 1024 raté, pour un coût VRAM quasi nul.")

        # ---- Installation (binaire + modèles) ----
        with gr.Accordion("⚙️ Installer trellis.cpp (binaire + modèles, 1 clic)",
                          open=not ready):
            gr.Markdown(
                "Télécharge le **binaire Windows CUDA** "
                "(`pwilkin/trellis.cpp`, ~700 Mo) dans `bin/trellis/` et un "
                "**jeu de modèles GGUF** (`ilintar/trellis2-gguf`) dans "
                "`models/trellis/`.\n\n"
                "**Variante de poids** — **f16 est le plus RAPIDE** quand il "
                "tient en mémoire : garde-le pour le 512. Les versions "
                "quantifiées (q8/q4) occupent beaucoup moins de mémoire — ce "
                "qui peut rendre le **1024/1536 atteignable** — mais elles "
                "sont **plus LENTES** (les poids sont déquantifiés à la volée "
                "à chaque calcul, surcoût non amorti sur une charge 3D). "
                "Tu peux en installer plusieurs et basculer à la génération.")
            diag_md = gr.Markdown(trellis.diagnose())
            diag_btn = gr.Button("↻ Vérifier l'installation", size="sm")
            inst_variant = gr.Radio(
                [(lbl, v) for lbl, v in trellis.VARIANTS], value="f16",
                label="Variante à installer",
                info="Commence par f16 (le plus rapide). N'ajoute q8/q4 que si "
                     "tu veux tenter le 1024/1536.")
            inst_log = gr.Textbox(label="Journal d'installation", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            inst_btn = gr.Button("⬇️ Installer trellis.cpp (binaire + modèles)")
            gr.Markdown(
                "⬆️ **Mettre à jour le binaire** — l'installation ci-dessus "
                "**ne remplace pas** un binaire déjà présent : une fois "
                "trellis installé, il reste tel quel indéfiniment. Ne "
                "retélécharge **pas** les ~10 Go de modèles.\n\n"
                "Le **backend est choisi d'après votre carte**. La build CUDA "
                "amont n'embarque de code machine que pour les **RTX 30xx** et "
                "**RTX 50xx** : son `CMakeLists.txt` épingle les noyaux de "
                "trellis à `86;120`, écrasant la liste complète de sa propre "
                "CI. Sur RTX 20xx, RTX 40xx, A100 ou H100 elle échoue par "
                "« no kernel image » — on installe donc la build **Vulkan**, "
                "qui ne compile rien par architecture et marche partout.",
                elem_classes="hint")
            upd_btn = gr.Button("⬆️ Mettre à jour le binaire (sans les modèles)",
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
                logs.append("\n✅ Terminé." if trellis.is_ready()
                            else "\n⚠️ Installation incomplète — voir ci-dessus.")
                yield "\n".join(logs[-400:])

            def _install(inst_var):
                yield from _run_installer(
                    ["--variant", str(inst_var or "f16")],
                    t("⏳ Installation en cours (binaire + ~10 Go de modèles)…"))

            def _update_binary():
                # Le serveur résident VERROUILLE l'exécutable sous Windows :
                # sans cet arrêt, l'extraction échouerait sur un « accès refusé »
                # difficile à relier à sa cause.
                if trellis.resident_is_running():
                    yield t("⏹️ Arrêt du serveur résident (il verrouille le "
                            "binaire)…")
                    trellis.resident_stop()
                yield from _run_installer(
                    ["--binary", "--force"],
                    t("⏳ Téléchargement du binaire trellis le plus récent…"))

            inst_evt = inst_btn.click(_install, inputs=[inst_variant],
                                      outputs=[inst_log])
            upd_btn.click(_update_binary, outputs=[inst_log])

        # ---- Génération ----
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image d'entrée (objet unique)",
                                 type="filepath",
                                 buttons=widgets.IMAGE_VIEW_ONLY)
                with gr.Row():
                    square_pad = gr.Checkbox(
                        value=True, scale=2,
                        label="Compléter en carré (garde les proportions)",
                        info="TRELLIS traite l'entrée en carré : sans ça, une "
                             "image non carrée sort DÉFORMÉE.")
                    pad_color = gr.Dropdown(
                        [("Blanc", "white"), ("Noir", "black"),
                         ("Transparent", "transparent")],
                        value="white", scale=1, label="Bandes ajoutées")
                with gr.Row():
                    res = gr.Radio(_res_choices(), value=512, scale=2,
                                   label="Résolution géométrie")
                    variant = gr.Dropdown(
                        _variant_choices(), value=_default_variant(), scale=1,
                        label="Poids utilisés",
                        info="f16 = le PLUS RAPIDE s'il tient. q8/q4 = moins "
                             "de mémoire mais plus LENTS (déquantification à "
                             "la volée) — à réserver au 1024/1536.")
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = aléatoire)")
                    bg = gr.Dropdown(
                        [("BiRefNet (qualité, recommandé)", "birefnet"),
                         ("Seuil (rapide)", "threshold")],
                        value="birefnet", label="Détourage du fond")
                with gr.Accordion("⚡ Serveur résident (séries de 3D)", open=False):
                    gr.Markdown(
                        "Par défaut le serveur **démarre puis s'arrête** à "
                        "chaque génération (VRAM libérée). Coché, il **reste en "
                        "vie** : les 3D suivantes évitent le rechargement des "
                        "modèles (~30 s gagnées), mais **la VRAM reste "
                        "occupée** — arrête-le avant de générer des images.  \n"
                        "ℹ️ Si tu changes un réglage de **lancement** "
                        "(résolution, décimation, atlas, GPU, texture…), le "
                        "serveur **redémarre automatiquement** pour "
                        "l'appliquer — seuls seed et détourage sont "
                        "modifiables sans rechargement.")
                    resident = gr.Checkbox(
                        value=False,
                        label="Garder le serveur résident entre les générations")
                    with gr.Row():
                        res_status = gr.Markdown(trellis.resident_status())
                        res_stop_btn = gr.Button("⏹️ Arrêter le serveur résident",
                                                 size="sm")
                with gr.Accordion("🎛️ Qualité / maillage", open=False):
                    with gr.Row():
                        decim = gr.Number(
                            value=0, precision=0,
                            label="Décimation — faces cibles (0 = défaut)",
                            info="Plus bas = maillage plus léger.")
                        atlas = gr.Dropdown(
                            [("Défaut", 0), ("1024 px", 1024),
                             ("2048 px", 2048), ("4096 px", 4096)],
                            value=0, label="Taille de l'atlas UV (texture)")
                    with gr.Row():
                        no_texture = gr.Checkbox(
                            value=False,
                            label="Géométrie seule (sans texture, + rapide)")
                        box_uv = gr.Checkbox(value=False,
                                             label="Dépliage UV « box »")
                with gr.Accordion("🩺 Moteur (dépannage)", open=False):
                    gr.Markdown(
                        "**Carte utilisée** — passé au moteur via son flag "
                        "officiel `--gpu N`. Choisis la carte avec le plus de "
                        "VRAM (le mode 1024 en réclame ~16 Go).")
                    gpu_pick = gr.Dropdown(
                        [(t("Défaut du moteur (carte 0)"), -1)] + _gpu_choices(),
                        value=(_gpu_choices()[0][1] if _gpu_choices() else -1),
                        label="Carte utilisée pour la 3D")
                    band = gr.Number(
                        value=0, precision=4,
                        label="Offset de remaillage « band » (0 = auto)",
                        info="v0.5.4 l'adapte à la résolution (corrige les "
                             "speckles en 1024). À ne changer qu'en dépannage.")
                    with gr.Row():
                        require_gpu = gr.Checkbox(
                            value=True,
                            label="Exiger le GPU (évite un repli CPU très lent)")
                        f32 = gr.Checkbox(value=False,
                                          label="Précision f32 (au lieu de f16)")
                        no_fa = gr.Checkbox(value=False,
                                            label="Désactiver FlashAttention")
                with gr.Accordion("Options avancées", open=False):
                    extra = gr.Textbox(
                        label="Arguments trellis-server supplémentaires (optionnel)",
                        placeholder="ex. flags additionnels du serveur")
                with gr.Row():
                    run = gr.Button("🧊 Générer le 3D", variant="primary", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
                status = gr.Markdown("")
            with gr.Column(scale=4):
                model3d = gr.Model3D(label="Aperçu 3D (GLB)", clear_color=[
                    0.1, 0.1, 0.12, 1.0])
                glb_file = gr.File(label="Fichier GLB", interactive=False)
                with gr.Row():
                    seed_used = gr.Textbox(
                        label="Seed utilisé (pour rejouer cet objet)",
                        interactive=False, buttons=widgets.TEXT_COPY, scale=2)
                    seed_reuse = gr.Button("♻️ Réutiliser ce seed", size="sm",
                                           scale=1)
                log = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        def do_generate3d(image_path, square_val, pad_val, res_val,
                          variant_val, band_val, seed_val,
                          bg_val, resident_val,
                          decim_val, atlas_val, no_tex_val, box_uv_val,
                          gpu_val, req_gpu_val, f32_val, no_fa_val,
                          extra_args):
            if not image_path:
                raise gr.Error(t("Chargez une image d'entrée."))
            if not trellis.is_ready():
                raise gr.Error(t("trellis.cpp n'est pas installé — dépliez "
                                 "« Installer trellis.cpp » ci-dessus."))
            settings.ensure_dirs()
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
            logs: list[str] = []
            # (status, model3d, glb_file, log, res_status, seed_used)
            yield (t("⏳ Génération 3D en cours (mode {r})…").format(r=res_val),
                   gr.update(), gr.update(), gr.update(), gr.update(),
                   gr.update())
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield (gr.update(), gr.update(), gr.update(),
                       "\n".join(logs[-500:]), gr.update(), gr.update())

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       gr.update(), "\n".join(logs),
                       gr.update(value=trellis.resident_status()), gr.update())
                return
            yield (t("✅ 3D généré : {name}").format(name=out_glb.name),
                   gr.update(value=str(out_glb)), gr.update(value=str(out_glb)),
                   "\n".join(logs), gr.update(value=trellis.resident_status()),
                   gr.update(value=str(meta.get("seed", ""))))

        gen_evt = run.click(
            do_generate3d,
            inputs=[image, square_pad, pad_color, res, variant, band,
                    seed, bg, resident,
                    decim, atlas, no_texture,
                    box_uv, gpu_pick, require_gpu, f32, no_fa, extra],
            outputs=[status, model3d, glb_file, log, res_status, seed_used])
        stop.click(lambda: sdcpp.cancel_active(), outputs=None,
                   cancels=[gen_evt])

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
