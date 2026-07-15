#!/usr/bin/env python3
"""Maintenance après mise à jour par copier-coller.

Copier-coller le dépôt par-dessus l'ancien AJOUTE et REMPLACE les fichiers, mais
n'efface JAMAIS ceux supprimés en amont : ils restent en orphelins et peuvent
casser/embrouiller l'app. Ce script :
  • supprime les fichiers de code devenus OBSOLÈTES (liste ci-dessous) ;
  • purge tous les __pycache__ (.pyc périmés d'anciens modules) ;
  • vide le dossier tmp/ (fichiers de travail) ;
  • REPÈRE les dossiers de MODÈLES orphelins (plus référencés par le catalogue —
    ex. un encodeur remplacé, un modèle retiré) et l'espace récupérable ;
  • vérifie que tout compile, que le catalogue YAML est valide, que les
    dépendances et le binaire sd-cli sont présents.

Par défaut il NE SUPPRIME PAS de modèles (il les liste seulement). Pour libérer
l'espace :  python scripts/maintenance.py --prune-models  (ou maintenance.bat
--prune-models). Ne touche jamais à models/custom/, loras/, outputs/, userdata/,
python/, bin/.
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
    # Backend ComfyUI + mode serveur (retirés : un seul moteur, sd-cli + aperçu).
    "atelier/engine/comfyui.py",
    "atelier/engine/sdserver.py",
    "scripts/get_comfyui.py",
    "config/comfyui_workflows/flux2.json",
    "config/comfyui_workflows/krea2.json",
    "config/comfyui_workflows/krea2int8.json",
    "config/comfyui_workflows/krea2convrot.json",
]

# Dossiers devenus obsolètes : supprimés s'ils sont VIDES après le nettoyage
# ci-dessus ; signalés (avec leur taille) s'ils contiennent encore des données
# volumineuses à la charge de l'utilisateur (ex. l'installation ComfyUI).
OBSOLETE_DIRS = ["config/comfyui_workflows"]
LEFTOVER_HEAVY = ["comfyui"]   # installation ComfyUI (retirée) : ~4–6 Go

# Dossiers de données à NE JAMAIS toucher.
PROTECTED = {"python", "bin", "models", "loras", "outputs", "userdata", ".git",
             "comfyui"}

OK, WARN, ERR, INFO = "  [OK] ", "  [!] ", "  [X] ", "  [i] "
_problems = 0


def _warn(msg: str) -> None:
    global _problems
    _problems += 1
    print(WARN + msg)


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


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
    for rel in OBSOLETE_DIRS:
        p = ROOT / rel
        if p.is_dir():
            try:
                p.rmdir()   # seulement s'il est vide
                print(OK + f"dossier vide supprimé : {rel}")
                found = True
            except OSError:
                pass
    for rel in LEFTOVER_HEAVY:
        p = ROOT / rel
        if p.is_dir():
            print(INFO + f"le dossier {rel}/ ({_human(_dir_size(p))}) date d'une "
                  "ancienne version (backend ComfyUI, retiré) — vous pouvez le "
                  "supprimer pour libérer l'espace.")
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


def _expected_model_dirs() -> set[str]:
    """Noms de dossiers (owner__repo) attendus d'après le catalogue courant :
    tous les composants des modèles + PiD + upscalers."""
    from atelier import registry, settings
    prefs = settings.load_prefs()
    repos: set[str] = set()
    for m in registry.load_base_models(prefs):
        repos.update(c.repo for c in m.components)
    repos.update(c.repo for c in registry.pid_components())
    up = registry.upscaler_config().get("repo")
    if up:
        repos.add(up)
    return {settings.model_repo_dir(r).name for r in repos if r}


def report_orphan_models(prune: bool) -> None:
    print("• Modèles orphelins (dossiers plus référencés par le catalogue)…")
    try:
        from atelier import settings
        models_dir = settings.MODELS_DIR
        expected = _expected_model_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return
    if not models_dir.is_dir():
        print(OK + "aucun dossier models/.")
        return
    orphans = [d for d in sorted(models_dir.iterdir())
               if d.is_dir() and d.name != "custom" and d.name not in expected]
    if not orphans:
        print(OK + "aucun modèle orphelin (propre).")
        return
    total = 0
    for d in orphans:
        size = _dir_size(d)
        total += size
        print(f"    - {d.name}  ({_human(size)})")
    print(INFO + f"{len(orphans)} dossier(s) orphelin(s) = "
          f"{_human(total)} récupérables.")
    if prune:
        freed = 0
        for d in orphans:
            sz = _dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                freed += sz
                print(OK + f"supprimé : {d.name}")
            else:
                _warn(f"suppression partielle : {d.name}")
        print(OK + f"{_human(freed)} libérés.")
    else:
        print("    → pour libérer l'espace : "
              "python scripts/maintenance.py --prune-models")


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
    prune = "--prune-models" in sys.argv
    print("=" * 60)
    print("  Maintenance — Turbo Slop Generator 3000")
    if prune:
        print("  (--prune-models : suppression des modèles orphelins activée)")
    print("=" * 60)
    remove_obsolete()
    clean_pycache()
    clean_tmp()
    check_catalog()
    report_orphan_models(prune)
    compile_check()
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
