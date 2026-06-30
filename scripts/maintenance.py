#!/usr/bin/env python3
"""Maintenance après mise à jour par copier-coller.

Copier-coller le dépôt par-dessus l'ancien AJOUTE et REMPLACE les fichiers, mais
n'efface JAMAIS ceux supprimés en amont : ils restent en orphelins et peuvent
casser/embrouiller l'app. Ce script :
  • supprime les fichiers devenus OBSOLÈTES (liste ci-dessous) ;
  • purge tous les __pycache__ (.pyc périmés d'anciens modules) ;
  • vide le dossier tmp/ (fichiers de travail) ;
  • vérifie que tout compile, que le catalogue YAML est valide, que les
    dépendances et le binaire sd-cli sont présents.

Ne touche JAMAIS à : python/, bin/, models/, loras/, outputs/, userdata/.
Lancer :  maintenance.bat  (Windows)  ·  ./maintenance.sh  (Linux/Mac)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Fichiers retirés du projet au fil des versions : à effacer s'ils traînent.
# (À compléter ici quand un fichier source est supprimé en amont.)
OBSOLETE = [
    "atelier/ui/creative_tab.py",          # ancien onglet Upscale (retiré)
    "scripts/tools/run_creative_upscale.py",  # ancien runner SDXL+ControlNet
    "atelier/ui/video_tab.py",             # ancien onglet Vidéo LTX (retiré)
]

# Dossiers de données à NE JAMAIS toucher.
PROTECTED = {"python", "bin", "models", "loras", "outputs", "userdata", ".git"}

OK, WARN, ERR = "  [OK] ", "  [!] ", "  [X] "
_problems = 0


def _warn(msg: str) -> None:
    global _problems
    _problems += 1
    print(WARN + msg)


def remove_obsolete() -> None:
    print("• Fichiers obsolètes…")
    found = False
    for rel in OBSOLETE:
        p = ROOT / rel
        if p.exists():
            try:
                p.unlink()
                print(OK + f"supprimé : {rel}")
                found = True
            except OSError as exc:
                _warn(f"impossible de supprimer {rel} : {exc}")
    if not found:
        print(OK + "aucun fichier obsolète (propre).")


def clean_pycache() -> None:
    print("• Caches Python (__pycache__ / .pyc)…")
    n = 0
    for p in ROOT.rglob("__pycache__"):
        if p.is_dir() and not any(part in PROTECTED for part in p.parts):
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    print(OK + f"{n} dossier(s) __pycache__ purgé(s).")


def clean_tmp() -> None:
    print("• Dossier tmp/…")
    tmp = ROOT / "tmp"
    n = 0
    if tmp.is_dir():
        for p in tmp.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                n += 1
            except OSError:
                pass
    print(OK + f"{n} élément(s) temporaire(s) effacé(s).")


def compile_check() -> None:
    print("• Compilation (syntaxe)…")
    import compileall
    ok = True
    for target in ("atelier", "scripts"):
        ok &= compileall.compile_dir(str(ROOT / target), quiet=1, force=True)
    ok &= compileall.compile_file(str(ROOT / "app.py"), quiet=1, force=True)
    if ok:
        print(OK + "tout le code Python compile.")
    else:
        global _problems
        _problems += 1
        print(ERR + "erreur(s) de syntaxe ci-dessus — mise à jour incomplète ?")


def check_catalog() -> None:
    print("• Catalogue de modèles (config/models.yaml)…")
    try:
        import yaml
        cat = yaml.safe_load((ROOT / "config" / "models.yaml")
                             .read_text(encoding="utf-8")) or {}
        models = [m.get("id") for m in cat.get("base_models", [])]
        print(OK + f"YAML valide — modèles : {', '.join(models) or '(aucun)'}.")
    except Exception as exc:  # noqa: BLE001
        _warn(f"models.yaml illisible : {exc}")


def check_deps() -> None:
    print("• Dépendances Python…")
    missing = []
    for mod in ("gradio", "yaml", "PIL", "requests", "huggingface_hub"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if missing:
        _warn(f"manquantes : {', '.join(missing)} → relancez install.bat "
              "(ou install.sh).")
    else:
        print(OK + "présentes.")


def check_engine() -> None:
    print("• Moteur stable-diffusion.cpp (sd-cli)…")
    try:
        from atelier import settings
        sd = settings.find_sd_cli()
    except Exception as exc:  # noqa: BLE001
        _warn(f"vérification impossible : {exc}")
        return
    if sd:
        print(OK + f"trouvé : {sd}")
    else:
        _warn("binaire sd-cli introuvable → install.bat, ou "
              "python scripts/get_sdcpp.py --variant cuda")


def main() -> int:
    print("=" * 60)
    print("  Maintenance — Turbo Slop Generator 3000")
    print("=" * 60)
    remove_obsolete()
    clean_pycache()
    clean_tmp()
    compile_check()
    check_catalog()
    check_deps()
    check_engine()
    print("-" * 60)
    if _problems == 0:
        print("✅ Tout est propre et vérifié. Vous pouvez lancer run.bat.")
    else:
        print(f"⚠️  Terminé avec {_problems} point(s) d'attention "
              "ci-dessus (voir les lignes [!]/[X]).")
    print("=" * 60)
    return 0 if _problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
