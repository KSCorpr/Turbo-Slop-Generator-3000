"""Turbo Slop Generator 3000 — studio d'inférence d'images en local.

Génération Flux.2 Klein 9B et Krea 2 Turbo au format GGUF (via
stable-diffusion.cpp), catalogue de modèles à la demande, optimisations
automatiques selon le GPU et la RAM, support LoRA, résolutions natives par
modèle, améliorateur de prompt (IA), et outils (profondeur, détourage, SAM).
"""

__version__ = "1.2.0rc1"
APP_NAME = "Turbo Slop Generator 3000"
ACCENT = "#00ccff"


def _sanitize_no_proxy() -> None:
    """Retire les crochets autour des adresses IPv6 dans NO_PROXY/no_proxy.

    httpx enveloppe déjà tout hôte contenant ':' dans des crochets, ce qui
    transforme '[::1]' en '[[::1]]' et lève InvalidURL("Invalid port: ':1]'")
    dès l'import de Gradio.
    """
    import os
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key)
        if not raw or "[" not in raw:
            continue
        parts = [p.strip()[1:-1] if p.strip().startswith("[") and p.strip().endswith("]")
                 and ":" in p else p.strip() for p in raw.split(",")]
        os.environ[key] = ",".join(parts)


_sanitize_no_proxy()

