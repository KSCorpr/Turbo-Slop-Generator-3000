#!/usr/bin/env python3
"""GEN.Ai Image Workshop — studio d'inférence d'images en local (Gradio).

Onglets : Génération (Flux.2 Klein 9B / Krea 2 Turbo, GGUF) · Catalogue de
modèles · Toolkit (profondeur, détourage, SAM, upscale) · Réglages.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# Le Python portable n'ajoute pas le dossier projet au chemin d'import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Avertissements bénins de Gradio (paramètres déplacés en v6.0) : on les masque
# pour ne pas inquiéter inutilement au démarrage. L'usage actuel (5.x) est correct.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

import gradio as gr


def _patch_gradio_client() -> None:
    """Contourne un bug de gradio_client sur les schémas booléens (au démarrage)."""
    try:
        import gradio_client.utils as gcu
        _orig = gcu._json_schema_to_python_type

        def _safe(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig(schema, defs)

        gcu._json_schema_to_python_type = _safe
    except Exception:  # noqa: BLE001
        pass


def _disable_brotli() -> None:
    """Désactive la compression Brotli de Gradio : son middleware calcule mal le
    Content-Length et casse le service des images (erreurs « Too much/little data
    for declared Content-Length »), ce qui faisait planter l'upscale ET l'aperçu
    temps réel. On le rend transparent (compression inutile en local)."""
    try:
        import gradio.brotli_middleware as bm

        async def _passthrough(self, scope, receive, send):
            await self.app(scope, receive, send)

        bm.BrotliMiddleware.__call__ = _passthrough
    except Exception:  # noqa: BLE001
        pass


_patch_gradio_client()
_disable_brotli()

from atelier import APP_NAME, __version__, hardware, i18n, net, settings
from atelier.ui.generate_tab import build_generative_tab
from atelier.ui.library_tab import build_library_tab
from atelier.ui.settings_tab import build_settings_tab
from atelier.ui.theme import CSS, theme
from atelier.ui.toolkit_tab import build_toolkit_tab

# Force le thème clair quel que soit le réglage clair/sombre du navigateur/OS.
_HEAD = (
    "<script>"
    "if(!new URLSearchParams(window.location.search).has('__theme')){"
    "const u=new URL(window.location);u.searchParams.set('__theme','light');"
    "window.location.replace(u);}"
    "</script>"
)


def build_app() -> gr.Blocks:
    settings.ensure_dirs()
    # Langue de l'interface : lue dans les préférences. Les chaînes dynamiques
    # sont traduites à la construction (via t()), le reste après coup.
    i18n.init_from_prefs()
    gpus = hardware.detect_gpus()
    sd_cli = settings.find_sd_cli()

    with gr.Blocks(title=f"{APP_NAME} {__version__}", theme=theme(), css=CSS,
                   head=_HEAD) as demo:
        _subtitle = i18n.t("Génération d'images locale")
        gr.HTML(
            f"<div id='atelier-header'><h1>🎨 {APP_NAME}</h1>"
            f"<div class='sub'>{_subtitle} · "
            f"Flux.2 Klein 9B · Krea 2 Turbo · v{__version__}</div></div>")

        if sd_cli is None:
            gr.Markdown("> ⚠️ **Binaire `sd-cli` introuvable.** Lancez "
                        "`install.bat` / `install.sh`, ou "
                        "`python scripts/get_sdcpp.py`.")
        if not gpus:
            gr.Markdown("> ⚠️ **Aucun GPU NVIDIA détecté** (mode CPU très lent). "
                        "Vérifiez vos pilotes / `nvidia-smi`.")

        # Image en attente d'envoi vers le Toolkit : (chemin, destination).
        pending_toolkit = gr.State(None)
        with gr.Tabs() as tabs:
            build_generative_tab("flux2-klein-9b", "🟣 Flux.2 Klein 9B",
                                 pending_toolkit=pending_toolkit, tabs=tabs)
            build_generative_tab("krea2-turbo", "⚡ Krea 2 Turbo",
                                 pending_toolkit=pending_toolkit, tabs=tabs)
            build_library_tab()
            build_toolkit_tab(pending_toolkit=pending_toolkit, tabs=tabs)
            build_settings_tab()

    i18n.translate_blocks(demo)   # traduit les libellés statiques (mode EN)
    return demo


def _print_lan_banner(port: int, auth: bool) -> None:
    urls = [f"http://{ip}:{port}" for ip in net.lan_ips()]
    line = "═" * 64
    print("\n" + line)
    print("  " + i18n.t("{app} est accessible sur le réseau local !").format(
        app=APP_NAME))
    print("  " + i18n.t("Partagez cette adresse à vos collègues "
                        "(Mac/PC, même Wi-Fi),"))
    print("  " + i18n.t("à ouvrir dans Safari ou Chrome :"))
    for u in urls or [f"http://<IP-de-ce-PC>:{port}"]:
        print(f"      →  {u}")
    if auth:
        print("  " + i18n.t("(un identifiant/mot de passe leur sera demandé)"))
    print("  " + i18n.t("Si l'accès échoue : autorisez le port dans le "
                        "pare-feu Windows."))
    print(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=f"{APP_NAME} {__version__}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="lien public temporaire gradio.live")
    ap.add_argument("--listen", action="store_true",
                    help="exposer sur le réseau local (0.0.0.0)")
    ap.add_argument("--auth", default=None,
                    help="protéger par mot de passe : utilisateur:motdepasse")
    args = ap.parse_args()

    host = "0.0.0.0" if args.listen else args.host
    auth = None
    if args.auth and ":" in args.auth:
        u, p = args.auth.split(":", 1)
        auth = (u, p)

    demo = build_app().queue()
    port = net.find_free_port(args.port, host=host)

    if args.listen:
        _print_lan_banner(port, auth is not None)

    demo.launch(server_name=host, server_port=port, share=args.share,
                auth=auth, inbrowser=not args.listen, show_api=False)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
