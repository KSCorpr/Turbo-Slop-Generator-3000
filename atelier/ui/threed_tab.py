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


def build_threed_tab():
    with gr.Tab("🧊 Image → 3D"):
        ready = trellis.is_ready()
        gr.Markdown(
            "### Image → modèle 3D (GLB)\n"
            "Transforme une image en **maillage 3D texturé** (GLB) via "
            "**trellis.cpp** (TRELLIS.2, binaire natif CUDA — aucun PyTorch). "
            "Chargez une image nette d'un **objet unique** sur fond simple ; le "
            "détourage est automatique. Génération **one-shot** : le moteur "
            "libère toute la VRAM en fin de course.\n\n"
            "💡 Sur tes cartes (≤ 12 Go), reste en **512** (les modes 1024/1536 "
            "demandent ~16 Go+).")

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
                with gr.Accordion("Options avancées", open=False):
                    extra = gr.Textbox(
                        label="Arguments trellis-cli supplémentaires (optionnel)",
                        placeholder="ex. flags additionnels du CLI")
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

        def do_generate3d(image_path, res_val, extra_args):
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

            def worker():
                try:
                    trellis.generate(in_png, out_glb, res=int(res_val),
                                     extra=extra_args or "", log=q.put)
                    state["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            # (status, model3d, glb_file, log)
            yield (t("⏳ Génération 3D en cours (mode {r})… le binaire libère la "
                     "VRAM à la fin.").format(r=res_val),
                   gr.update(), gr.update(), gr.update())
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), gr.update(), "\n".join(logs[-500:])

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                yield (t("❌ Échec — voir le journal."), gr.update(),
                       gr.update(), "\n".join(logs))
                return
            yield (t("✅ 3D généré : {name}").format(name=out_glb.name),
                   gr.update(value=str(out_glb)), gr.update(value=str(out_glb)),
                   "\n".join(logs))

        gen_evt = run.click(
            do_generate3d, inputs=[image, res, extra],
            outputs=[status, model3d, glb_file, log])
        stop.click(lambda: sdcpp.cancel_active(), outputs=None,
                   cancels=[gen_evt])
