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
    check_diffusers()
    check_addons_sync()


def check_addons_sync() -> None:
    """Add-ons dont l'installation applique des correctifs au code téléchargé.

    « tools_repo/ » n'est PAS dans le dépôt : une mise à jour par copier-coller
    remplace notre code mais ne retouche à rien dans les add-ons déjà installés.
    Si on a modifié la façon dont un add-on s'installe, il reste donc figé dans
    son ancien état — sans que rien ne le signale, jusqu'à l'erreur au premier
    usage. On le détecte ici.
    """
    print("• Add-ons à ré-installer après mise à jour…")
    base = ROOT / "tools_repo" / "seedvr2"
    if not base.is_dir():
        print(OK + "aucun add-on concerné (SeedVR2 non installé).")
        return

    repo = base / "repo"
    todo: list[str] = []
    cfg = repo / "configs_1_4b" / "main.yaml"
    sel = repo / "src" / "core" / "model_configuration.py"

    if not (repo / "inference_cli.py").is_file():
        todo.append("code d'inférence absent")
    if not cfg.is_file():
        todo.append("config 1.4B absente")
    elif "DÉRIVÉ AUTOMATIQUEMENT" not in cfg.read_text(encoding="utf-8",
                                                       errors="replace"):
        todo.append("config 1.4B figée (ancienne version) au lieu d'être "
                    "dérivée du dépôt")
    if sel.is_file() and "configs_1_4b" not in sel.read_text(encoding="utf-8",
                                                             errors="replace"):
        todo.append("sélection d'architecture non étendue au 1.4B")
    def _dits(d: Path) -> list:
        """Modèles de diffusion : le VAE cohabite dans le même dossier et ne
        compte pas — sinon un dossier ne contenant que lui passerait pour OK."""
        if not d.is_dir():
            return []
        return [p for p in list(d.glob("*.safetensors")) + list(d.glob("*.gguf"))
                if "vae" not in p.name.lower()]

    if not _dits(base / "models" / "SEEDVR2"):
        if _dits(base / "models"):
            todo.append("poids restés dans l'ancien dossier (models/ au lieu "
                        "de models/SEEDVR2/)")
        else:
            todo.append("poids absents")

    if todo:
        _warn("SeedVR2 n'est pas à jour :")
        for t in todo:
            _warn(f"    - {t}")
        _warn("  Correctif : Toolkit → Restaurer (SeedVR2) → « Installer "
              "SeedVR2 ». Les poids déjà présents ne sont pas retéléchargés.")
    else:
        print(OK + "SeedVR2 conforme à l'installeur actuel.")


def check_diffusers() -> None:
    """Cohérence des paquets PARTAGÉS par les add-ons PyTorch.

    Tous vivent dans le même Python : un add-on installé avec une contrainte
    plus large écrase la version dont un autre a besoin, et la casse ne se voit
    qu'au premier usage de l'autre — sous forme d'une erreur illisible à
    l'import. On vérifie donc chaque paquet épinglé par l'installeur.
    """
    print("• Paquets partagés par les add-ons PyTorch…")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from setup_tools import _PINS
    except Exception:  # noqa: BLE001
        print(OK + "non vérifiable (installeur absent).")
        return

    import importlib.metadata as md
    checked = bad = 0
    for name, spec in sorted(_PINS.items()):
        try:
            got = md.version(name)
        except Exception:  # noqa: BLE001
            continue                      # paquet absent = add-on non installé
        checked += 1
        if not _spec_ok(got, spec):
            bad += 1
            _warn(f"{name} {got} installé — attendu « {spec} ».")
    if not checked:
        print(OK + "aucun add-on PyTorch installé.")
    elif bad:
        _warn("  Un add-on a changé une version sous les autres.")
        _warn("  Correctif : relancez l'installation de l'add-on concerné "
              "(Toolkit → Installer), qui repose les bonnes versions.")
    else:
        print(OK + f"{checked} paquet(s) conforme(s).")


def _spec_ok(version: str, spec: str) -> bool:
    """Version conforme à un spec pip simple (« ==x », « >=a,<b »).

    Comparaison numérique par composants : « 1.26.4 » < « 2 » proprement, sans
    dépendre de packaging (absent du Python embarqué minimal).
    """
    def key(v: str):
        out = []
        for part in v.split("."):
            num = "".join(c for c in part if c.isdigit())
            out.append(int(num) if num else 0)
        return tuple(out)

    if "==" in spec:
        return version == spec.split("==")[-1].strip()
    for clause in spec.split(","):
        clause = clause.strip()
        for op in (">=", "<=", "!=", "<", ">"):
            if op in clause:
                bound = clause.split(op, 1)[1].strip()
                # retire un eventuel prefixe de nom de paquet
                bound = bound.split()[0] if bound else bound
                a, b = key(version), key(bound)
                a, b = a + (0,) * (len(b) - len(a)), b + (0,) * (len(a) - len(b))
                ok = {">=": a >= b, "<=": a <= b, "<": a < b,
                      ">": a > b, "!=": a != b}[op]
                if not ok:
                    return False
                break
    return True


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
