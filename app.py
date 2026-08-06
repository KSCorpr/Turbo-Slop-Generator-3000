#!/usr/bin/env python3
"""Turbo Slop Generator 3000 — studio d'inférence d'images en local (Gradio).

Onglets : Génération (Flux.2 Klein 9B / Krea 2 Turbo, GGUF) · Xanax (style figé) ·
Catalogue de modèles ·
Toolkit (profondeur, détourage, SAM, upscale) · Outpaint · Image → 3D · Réglages.
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


def _quiet_connection_reset() -> None:
    """Windows : « ConnectionResetError [WinError 10054] » dans la boucle asyncio.

    Quand le navigateur ferme brutalement une connexion (F5, onglet fermé,
    chargement d'image annulé), la boucle Proactor de Windows appelle
    `_call_connection_lost`, dont le bloc `finally` fait un
    `socket.shutdown()` sur une socket déjà morte. Ça lève, asyncio l'imprime
    en « Exception in callback », et ça inquiète pour rien : la requête est
    finie côté serveur. Bug Python connu (bpo-39010).

    On ne se contente PAS d'avaler l'exception : elle interrompt le `finally`
    en plein milieu, donc `close()`, le détachement du serveur et le drapeau de
    fin ne s'exécutent jamais — la socket fuirait. On termine donc le ménage
    nous-mêmes. Seules les erreurs de connexion sont interceptées ; tout le
    reste continue de remonter normalement.
    """
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
    except Exception:  # noqa: BLE001
        return
    orig = _T._call_connection_lost

    def _call_connection_lost(self, exc):
        try:
            orig(self, exc)
        except (ConnectionResetError, ConnectionAbortedError):
            try:
                if getattr(self, "_sock", None) is not None:
                    self._sock.close()
                self._sock = None
                server = getattr(self, "_server", None)
                if server is not None:
                    server._detach()
                    self._server = None
                self._called_connection_lost = True
            except Exception:  # noqa: BLE001
                pass

    _T._call_connection_lost = _call_connection_lost


_patch_gradio_client()
_disable_brotli()
_quiet_connection_reset()

from atelier import APP_NAME, __version__, hardware, i18n, net, settings
from atelier.ui.convert_tab import build_convert_tab
from atelier.ui.generate_tab import build_generative_tab
from atelier.ui.library_tab import build_library_tab
from atelier.ui.manage_tab import build_manage_tab
from atelier.ui.outpaint_tab import build_outpaint_tab
from atelier.ui.settings_tab import build_settings_tab
from atelier.ui.threed_tab import build_threed_tab
from atelier.ui.theme import CSS, theme
from atelier.ui.toolkit_tab import build_toolkit_tab
from atelier.ui.xanax_tab import build_xanax_tab

# Force le thème choisi (clair/sombre) quel que soit le réglage du navigateur/OS.
def _head_for(mode: str) -> str:
    mode = "dark" if mode == "dark" else "light"
    return (
        "<script>"
        "if(!new URLSearchParams(window.location.search).has('__theme')){"
        "const u=new URL(window.location);"
        f"u.searchParams.set('__theme','{mode}');"
        "window.location.replace(u);}"
        "</script>")


def build_app() -> gr.Blocks:
    settings.ensure_dirs()
    # Langue de l'interface : lue dans les préférences. Les chaînes dynamiques
    # sont traduites à la construction (via t()), le reste après coup.
    i18n.init_from_prefs()
    first_run = not settings.PREFS_FILE.exists()
    gpus = hardware.detect_gpus()
    sd_cli = settings.find_sd_cli()
    head = _head_for(settings.load_prefs().get("theme", "light"))

    with gr.Blocks(title=f"{APP_NAME} {__version__}", theme=theme(), css=CSS,
                   head=head) as demo:
        _subtitle = i18n.t("Génération d'images locale")
        gr.HTML(
            f"<div id='atelier-header'><h1>🎨 {APP_NAME}</h1>"
            f"<div class='sub'>{_subtitle} · "
            f"Flux.2 Klein 9B · Krea 2 Turbo · "
            f"v{__version__}</div></div>")

        # Premier démarrage : choix de la langue (bilingue, persisté).
        if first_run:
            gr.Markdown("### 🌐 Choisissez la langue · Choose your language")
            with gr.Row():
                _fr_btn = gr.Button("🇫🇷 Français", variant="primary")
                _en_btn = gr.Button("🇬🇧 English", variant="primary")
            _lang_msg = gr.Markdown("")

            def _pick_lang(code):
                p = settings.load_prefs()
                p["lang"] = code
                settings.save_prefs(p)
                return ("✅ Enregistré — **redémarrez** l'application (run.bat). · "
                        "Saved — **restart** the app.")

            _fr_btn.click(lambda: _pick_lang("fr"), outputs=[_lang_msg])
            _en_btn.click(lambda: _pick_lang("en"), outputs=[_lang_msg])

        if sd_cli is None:
            gr.Markdown("> ⚠️ **Binaire `sd-cli` introuvable.** Lancez "
                        "`install.bat` / `install.sh`, ou "
                        "`python scripts/get_sdcpp.py`.")
        # Sur Mac Apple Silicon, detect_gpus() renvoie le GPU intégré : pas
        # d'avertissement, il n'y a rien à installer. L'alerte ne vise que les
        # PC où un GPU NVIDIA est attendu mais absent (pilotes manquants).
        if not gpus:
            gr.Markdown("> ⚠️ **Aucun GPU détecté** (mode CPU très lent). "
                        "Sur PC, vérifiez vos pilotes NVIDIA / `nvidia-smi`.")

        # Image en attente d'envoi vers le Toolkit : (chemin, destination).
        pending_toolkit = gr.State(None)
        # Image en attente d'envoi vers l'onglet « Image → 3D » (chemin).
        pending_3d = gr.State(None)
        # Image en attente d'envoi vers l'onglet « Outpaint » (chemin).
        pending_outpaint = gr.State(None)
        with gr.Tabs() as tabs:
            build_generative_tab("flux2-klein-9b", "🟣 Flux.2 Klein 9B",
                                 pending_toolkit=pending_toolkit, tabs=tabs,
                                 pending_3d=pending_3d,
                                 pending_outpaint=pending_outpaint)
            build_generative_tab("krea2-turbo", "⚡ Krea 2 Turbo",
                                 pending_toolkit=pending_toolkit, tabs=tabs,
                                 pending_3d=pending_3d,
                                 pending_outpaint=pending_outpaint)
            # Onglets « Xanax » : style figé, aucun réglage de style exposé.
            build_xanax_tab("krea2-turbo", "💊 Krea 2 — Xanax")
            build_xanax_tab("flux2-klein-9b", "💊 Flux.2 Klein — Xanax")
            build_library_tab()
            build_toolkit_tab(pending_toolkit=pending_toolkit, tabs=tabs)
            build_outpaint_tab(pending_outpaint=pending_outpaint, tabs=tabs)
            build_threed_tab(pending_3d=pending_3d, tabs=tabs)
            build_convert_tab()
            build_manage_tab()
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
