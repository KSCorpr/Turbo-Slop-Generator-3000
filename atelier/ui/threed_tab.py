"""Onglet « 🧊 Image → 3D » : génère un maillage 3D texturé (GLB) à partir d'une
image, via le binaire natif trellis-cli (trellis.cpp — C++/GGML/CUDA, no PyTorch).

One-shot : le process se termine et libère la VRAM (stratégie low-VRAM). Le mode
512 « light » vise les cartes ≤ 12 Go ; 1024/1536 demandent ~16 Go+.
"""
from __future__ import annotations

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


def _res_choices():
    return [(lbl, val) for lbl, val in trellis.RESOLUTIONS]


def _gpu_choices() -> list[tuple[str, int]]:
    from .. import hardware
    return [(f"#{g.index} — {g.name} ({g.vram_gb:.0f} Go, {g.arch})", g.index)
            for g in hardware.detect_gpus()]


def build_threed_tab(tab_id="threed", pending_3d=None, tabs=None):
    with gr.Tab("🧊 Image → 3D", id=tab_id):
        ready = trellis.is_ready()
        gr.Markdown(
            "### Image → modèle 3D (GLB)\n"
            "Transforme une image en **maillage 3D texturé** (GLB) via "
            "**trellis.cpp** (TRELLIS.2, binaire natif CUDA — aucun PyTorch). "
            "Chargez une image nette d'un **objet unique** sur fond simple ; le "
            "détourage est automatique. Le serveur trellis **démarre puis "
            "s'arrête** à chaque génération → toute la VRAM est libérée ensuite "
            "(stratégie low-VRAM).\n\n"
            "💡 Sur une carte ≤ 12 Go, reste en **512** : les modes **1024/1536** "
            "demandent **~16 Go+**. En dessous, la géométrie peut sortir "
            "**corrompue (maillage en « blobs »)** plutôt que d'échouer "
            "franchement — et c'est pire si le calcul déborde sur une carte "
            "**Pascal (GTX 10xx)**. Voir « 🩺 Moteur » pour épingler une carte.")

        # ---- Installation (binaire + modèles) ----
        with gr.Accordion("⚙️ Installer trellis.cpp (binaire + modèles, 1 clic)",
                          open=not ready):
            gr.Markdown(
                "Télécharge le **binaire Windows CUDA** "
                "(`pwilkin/trellis.cpp`, ~700 Mo) dans `bin/trellis/` et le "
                "**jeu de modèles GGUF** (`ilintar/trellis2-gguf`, ~10 Go) dans "
                "`models/trellis/`. À faire une seule fois.")
            inst_log = gr.Textbox(label="Journal d'installation", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            inst_btn = gr.Button("⬇️ Installer trellis.cpp (binaire + modèles)")

            def _install():
                cmd = [sys.executable, str(settings.ROOT / "scripts"
                                           / "get_trellis.py")]
                logs: list[str] = []
                yield t("⏳ Installation en cours (binaire + ~10 Go de modèles)…")
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(settings.ROOT),
                    encoding="utf-8", errors="replace")
                assert proc.stdout is not None
                for line in proc.stdout:
                    logs.append(line.rstrip("\n"))
                    yield "\n".join(logs[-400:])
                proc.wait()
                logs.append("\n✅ Terminé." if trellis.is_ready()
                            else "\n⚠️ Installation incomplète — voir ci-dessus.")
                yield "\n".join(logs[-400:])

            inst_btn.click(_install, outputs=[inst_log])

        # ---- Génération ----
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image d'entrée (objet unique)",
                                 type="filepath")
                res = gr.Radio(_res_choices(), value=512,
                               label="Résolution géométrie")
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
                        "occupée** — arrête-le avant de générer des images.")
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
                        "⚠️ **Multi-GPU** : ggml peut répartir le calcul sur "
                        "toutes les cartes. Une carte **Pascal (GTX 10xx)** gère "
                        "très mal le BF16 → géométrie corrompue (**maillage en "
                        "« blobs »**), surtout en 1024. **Épingle la carte la "
                        "plus récente** ci-dessous.")
                    gpu_pick = gr.Dropdown(
                        [(t("Auto (toutes les cartes)"), -1)] + _gpu_choices(),
                        value=(_gpu_choices()[0][1] if _gpu_choices() else -1),
                        label="Carte utilisée pour la 3D")
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
                log = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        def do_generate3d(image_path, res_val, seed_val, bg_val, resident_val,
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
                _PI.open(image_path).convert("RGB").save(in_png)
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
                                     extra=extra_args or "", log=q.put)
                    state["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            # (status, model3d, glb_file, log, res_status)
            yield (t("⏳ Génération 3D en cours (mode {r})…").format(r=res_val),
                   gr.update(), gr.update(), gr.update(), gr.update())
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield (gr.update(), gr.update(), gr.update(),
                       "\n".join(logs[-500:]), gr.update())

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       gr.update(), "\n".join(logs),
                       gr.update(value=trellis.resident_status()))
                return
            yield (t("✅ 3D généré : {name}").format(name=out_glb.name),
                   gr.update(value=str(out_glb)), gr.update(value=str(out_glb)),
                   "\n".join(logs), gr.update(value=trellis.resident_status()))

        gen_evt = run.click(
            do_generate3d,
            inputs=[image, res, seed, bg, resident, decim, atlas, no_texture,
                    box_uv, gpu_pick, require_gpu, f32, no_fa, extra],
            outputs=[status, model3d, glb_file, log, res_status])
        stop.click(lambda: sdcpp.cancel_active(), outputs=None,
                   cancels=[gen_evt])

        def _stop_resident():
            msg = trellis.resident_stop()
            return gr.update(value=trellis.resident_status()), msg

        res_stop_btn.click(_stop_resident, outputs=[res_status, status])

        # --- Réception d'une image envoyée depuis un onglet de génération ---
        if pending_3d is not None and tabs is not None:
            def _consume3d(pend):
                if not pend:
                    return gr.update(), None
                return gr.update(value=pend), None

            tabs.select(_consume3d, inputs=[pending_3d],
                        outputs=[image, pending_3d])
