"""Onglet « Convertir en GGUF » : quantifie un modèle (checkpoint / safetensors
/ diffusion) vers un GGUF plus léger, via sd.cpp (--mode convert).

100% CPU, aucune diffusion : c'est une transformation des poids. But : faire
tenir un modèle communautaire sur une carte modeste (quant selon la VRAM).
"""
from __future__ import annotations

import queue
import threading

import gradio as gr

from .. import settings
from ..engine import generate as gen_engine
from ..engine import sdcpp
from ..i18n import t

# (valeur sd.cpp, libellé). Du plus lourd/fidèle au plus léger/agressif.
# Les k-quants (q*_k) demandent un moteur récent ; en cas de refus, le journal
# le dira et on retombe sur q5_1 / q8_0. Défaut : q4_k (bon compromis VRAM).
# (libellé affiché, valeur sd.cpp). Du plus lourd/fidèle au plus léger/agressif.
QTYPES = [
    ("q4_k — 4 bits (k-quant) · recommandé (≈ modèles de l'app)", "q4_k"),
    ("q5_k — 5 bits (k-quant) · plus fidèle", "q5_k"),
    ("q6_k — 6 bits (k-quant) · très fidèle", "q6_k"),
    ("q8_0 — 8 bits · quasi sans perte (gros)", "q8_0"),
    ("q5_1 — 5 bits", "q5_1"),
    ("q5_0 — 5 bits", "q5_0"),
    ("q4_1 — 4 bits", "q4_1"),
    ("q4_0 — 4 bits · le plus léger utile", "q4_0"),
    ("q3_k — 3 bits (agressif, perte visible)", "q3_k"),
    ("q2_k — 2 bits (très agressif)", "q2_k"),
    ("f16 — 16 bits (aucune perte, aucun gain de place)", "f16"),
]
_DEFAULT_QTYPE = "q4_k"


def _suggest_name(model_name: str | None, qtype: str) -> str:
    if not model_name:
        return ""
    stem = model_name.rsplit(".", 1)[0]
    return f"{stem}-{qtype or _DEFAULT_QTYPE}.gguf"


def build_convert_tab():
    with gr.Tab("🔧 Convertir en GGUF"):
        gr.Markdown(
            "### Quantifier un modèle en GGUF\n"
            "Transforme un modèle **checkpoint / safetensors / diffusion** en "
            "**GGUF** plus léger, pour le faire tenir sur ta carte. C'est du "
            "**100% CPU** (pas de diffusion) : quelques minutes selon la taille "
            "et le disque. Une seule fois — ensuite tu réutilises le GGUF.\n\n"
            f"1. Dépose ton modèle dans **`{settings.CUSTOM_DIR}`** puis "
            "**↻ Rafraîchir**.  \n"
            "2. Choisis la quant (voir l'échelle : `q8_0` ≈ sans perte → `q4_k` "
            "bon compromis → `q3_k` agressif).  \n"
            "3. **Convertir** : le GGUF est écrit dans le même dossier "
            "`models/custom/` et devient utilisable comme **modèle local** dans "
            "les onglets de génération (bouton « Rafraîchir les fichiers "
            "locaux »).")

        with gr.Row():
            src = gr.Dropdown(
                gen_engine.list_custom_models(), value=None, scale=3,
                label="Modèle à convertir (dans models/custom/)",
                allow_custom_value=False)
            refresh = gr.Button("↻ Rafraîchir", size="sm", scale=1)
        with gr.Row():
            qtype = gr.Dropdown(QTYPES, value=_DEFAULT_QTYPE, scale=2,
                                label="Quantification cible")
            out_name = gr.Textbox(label="Nom du fichier GGUF de sortie", scale=3,
                                  placeholder="ex. mon-modele-q4_k.gguf")

        with gr.Row():
            run = gr.Button("🔧 Convertir", variant="primary", scale=3)
            stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
        status = gr.Markdown("")
        log = gr.Textbox(label="Journal", lines=14, autoscroll=True,
                         elem_classes="log-box")

        # --- Comportements ---
        def _refresh():
            return gr.update(choices=gen_engine.list_custom_models())

        refresh.click(_refresh, outputs=[src])

        def _on_pick(name, qt):
            return gr.update(value=_suggest_name(name, qt))

        src.change(_on_pick, inputs=[src, qtype], outputs=[out_name])
        qtype.change(_on_pick, inputs=[src, qtype], outputs=[out_name])

        def do_convert(src_name, qt, out):
            sd_cli = settings.find_sd_cli()
            if sd_cli is None:
                raise gr.Error(t("Binaire sd-cli introuvable (install.bat / "
                                 "get_sdcpp.py)."))
            in_path = gen_engine.custom_path(src_name)
            if in_path is None:
                raise gr.Error(t("Choisissez un modèle à convertir (déposé dans "
                                 "models/custom/)."))
            out = (out or "").strip() or _suggest_name(src_name, qt)
            if not out.lower().endswith(".gguf"):
                out += ".gguf"
            out_path = settings.CUSTOM_DIR / out
            if out_path.resolve() == in_path.resolve():
                raise gr.Error(t("Le fichier de sortie doit différer de l'entrée."))

            cmd = sdcpp.build_convert_cmd(sd_cli, in_path, out_path, qt)
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    sdcpp.run(cmd, log=q.put)   # convert = CPU, pas de GPU épinglé
                    state["ok"] = out_path.is_file()
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield t("⏳ Conversion en cours… (CPU, quelques minutes)"), ""
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), "\n".join(logs[-500:])

            if "err" in state:
                logs.append(f"\n[ERREUR] {state['err']}")
                yield (t("❌ Échec de la conversion — voir le journal."),
                       "\n".join(logs))
                return
            if not state.get("ok"):
                yield (t("⚠️ Terminé mais fichier de sortie introuvable — voir le "
                         "journal (quant refusée par le moteur ?)."),
                       "\n".join(logs))
                return
            size_mb = out_path.stat().st_size / 1e6
            yield (t("✅ Converti : **{name}** ({size:.0f} Mo) dans "
                     "`models/custom/`. Utilisable via « Rafraîchir les fichiers "
                     "locaux » dans un onglet de génération.").format(
                        name=out, size=size_mb),
                   "\n".join(logs))

        evt = run.click(do_convert, inputs=[src, qtype, out_name],
                        outputs=[status, log])
        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])
