#!/usr/bin/env python3
"""Installe trellis.cpp (image → 3D) : binaire Windows CUDA + jeu de modèles GGUF.

- Binaire : release pré-compilée de pwilkin/trellis.cpp
    (trellis-cuda-windows-x64.zip) → bin/trellis/ (contient trellis-cli.exe).
- Modèles : dépôt Hugging Face ilintar/trellis2-gguf → models/trellis/
    (jeu complet : SS/SLAT flow, décodeurs, DINOv3, BiRefNet).

trellis-cli est un binaire NATIF (C++/GGML/CUDA, aucun PyTorch), invoqué en
one-shot (comme sd-cli) : le process se termine après chaque génération et
libère toute la VRAM — la stratégie « low-VRAM » (inspirée d'AISmith-3D) pour
faire tenir la 3D sur des cartes modestes (mode 512).

Usage :
    python scripts/get_trellis.py            # binaire + modèles
    python scripts/get_trellis.py --binary   # binaire seul
    python scripts/get_trellis.py --models   # modèles seuls
    python scripts/get_trellis.py --force    # re-télécharger le binaire
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# Réutilise les téléchargements ROBUSTES de get_sdcpp (reprise, miroirs, IPv4).
import get_sdcpp  # noqa: E402

BIN_DIR = ROOT / "bin"
TRELLIS_BIN_DIR = BIN_DIR / "trellis"
MODELS_DIR = ROOT / "models" / "trellis"

GH_RELEASE = "https://api.github.com/repos/pwilkin/trellis.cpp/releases/latest"
ASSET_MATCH = ("cuda", "win")          # archive Windows CUDA
HF_MODEL_REPO = "ilintar/trellis2-gguf"


def _is_cli(p: Path) -> bool:
    """Détection souple du binaire CLI trellis (nom variable selon la release)."""
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    if stem.startswith("trellis") and "cli" in stem:
        return True
    # Repli : un exécutable « trellis* » qui n'est ni server/test/studio/bench.
    return ("trellis" in stem and not any(
        x in stem for x in ("server", "test", "studio", "bench", "convert")))


def _find_cli() -> Path | None:
    if not BIN_DIR.exists():
        return None
    # 1re passe : correspondance stricte (…cli…) ; 2e passe : repli.
    strict = [p for p in BIN_DIR.rglob("*")
              if p.is_file() and _is_cli(p) and "cli" in p.name.lower()]
    if strict:
        return strict[0]
    loose = [p for p in BIN_DIR.rglob("*") if _is_cli(p)]
    return loose[0] if loose else None


def has_cli() -> bool:
    return _find_cli() is not None


def _list_exes(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(p.name for p in base.rglob("*")
                  if p.is_file() and p.name.lower().endswith(".exe"))


def has_models() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.rglob("*.gguf"))


def _pick_asset(assets: list[dict]) -> dict | None:
    """Choisit l'archive Windows CUDA (trellis-cuda-windows-x64.zip)."""
    best = None
    for a in assets:
        n = a.get("name", "").lower()
        if not n.endswith(".zip"):
            continue
        if all(tok in n for tok in ASSET_MATCH) and "rocm" not in n \
                and "vulkan" not in n:
            return a
        if "win" in n and best is None:
            best = a
    return best


def install_binary(force: bool = False, log=print) -> bool:
    if has_cli() and not force:
        log("Binaire trellis-cli déjà présent, on saute (--force pour MAJ).")
        return True
    log("Recherche de la dernière release pwilkin/trellis.cpp…")
    rel = get_sdcpp._fetch_json(GH_RELEASE)
    if isinstance(rel, dict) and rel.get("message") and not rel.get("assets"):
        log(f"API GitHub : {rel.get('message')}")
        return False
    asset = _pick_asset(rel.get("assets", []))
    if not asset:
        log("Aucune archive Windows CUDA trouvée dans la release. Disponibles :")
        for a in rel.get("assets", []):
            log("  " + a.get("name", "?"))
        return False
    log(f"Release : {rel.get('tag_name')} — {asset['name']} "
        f"({asset.get('size', 0) / 1e6:.0f} Mo)")
    # get_sdcpp._extract décompresse dans bin/ ; on cible bin/trellis/.
    TRELLIS_BIN_DIR.mkdir(parents=True, exist_ok=True)
    blob = get_sdcpp._download(asset["browser_download_url"])
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(TRELLIS_BIN_DIR)
    cli = _find_cli()
    if cli is not None:
        log(f"[OK] Binaire installé : {cli}")
        return True
    exes = _list_exes(TRELLIS_BIN_DIR)
    log("[!] trellis-cli introuvable après extraction. Exécutables trouvés :")
    for name in exes:
        log("    - " + name)
    if not exes:
        log("    (aucun .exe — l'archive n'a peut-être pas le binaire attendu)")
    return False


def install_models(log=print) -> bool:
    if has_models():
        log("Modèles trellis déjà présents, on saute.")
        return True
    try:
        import os
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        from huggingface_hub import snapshot_download
    except ImportError:
        log("huggingface_hub manquant (pip install huggingface_hub).")
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Téléchargement du jeu de modèles {HF_MODEL_REPO} → {MODELS_DIR} "
        "(~10 Go, reprise auto)…")
    try:
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(MODELS_DIR),
                          allow_patterns=["*.gguf", "*.json", "*.txt"])
    except Exception as exc:  # noqa: BLE001
        log(f"Échec du téléchargement des modèles : {exc}")
        return False
    if has_models():
        log("[OK] Modèles trellis en place.")
        return True
    log("[!] Aucun .gguf après téléchargement — vérifiez le dépôt HF.")
    return False


def install_all(force: bool = False, log=print) -> bool:
    ok_bin = install_binary(force=force, log=log)
    ok_mdl = install_models(log=log)
    return ok_bin and ok_mdl


def main():
    # Console Windows en cp1252 : force l'UTF-8 pour ne pas planter sur un
    # caractère non-encodable (accents, symboles) dans les logs.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="store_true", help="binaire seul")
    ap.add_argument("--models", action="store_true", help="modèles seuls")
    ap.add_argument("--force", action="store_true",
                    help="re-télécharger le binaire même s'il est présent")
    ap.add_argument("--allow-ipv6", action="store_true")
    args = ap.parse_args()
    if not args.allow_ipv6:
        get_sdcpp._force_ipv4()

    if args.binary:
        ok = install_binary(force=args.force)
    elif args.models:
        ok = install_models()
    else:
        ok = install_all(force=args.force)
    print("Terminé." if ok else "Terminé avec des erreurs (voir ci-dessus).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
