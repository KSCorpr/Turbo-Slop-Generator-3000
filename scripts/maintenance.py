#!/usr/bin/env python3
"""Maintenance après mise à jour par copier-coller.

Copier-coller le dépôt par-dessus l'ancien AJOUTE et REMPLACE les fichiers, mais
n'efface JAMAIS ceux supprimés en amont. Ce script rattrape ça :

  • supprime le CODE des fonctions retirées (table REMOVED_FEATURES) ;
  • CHIFFRE les DONNÉES qu'elles ont laissées (poids, dépôts clonés) sans les
    supprimer — plusieurs gigaoctets ne s'effacent pas sans prévenir ;
  • repère les ADD-ONS orphelins de tools_repo/ (dossiers ne correspondant à
    aucun add-on du code actuel) et les MODÈLES orphelins de models/ (plus
    référencés par le catalogue) ;
  • purge les __pycache__ (.pyc d'anciens modules) et le dossier tmp/ ;
  • vérifie que tout compile, que le catalogue YAML est valide, que les
    dépendances et le binaire sd-cli sont présents.

Par défaut il ne supprime AUCUNE donnée : il affiche l'espace récupérable et la
commande pour le libérer.

    maintenance.bat                 # vérifie et nettoie le code seulement
    maintenance.bat --purge         # + supprime les données des fonctions
                                    #   retirées et les orphelins
    (./maintenance.sh sur Linux/Mac)

Ne touche jamais à models/custom/, loras/, outputs/, userdata/, python/, bin/.
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

# --------------------------------------------------------------------------- #
#  FONCTIONS RETIRÉES DU PROJET
#
#  Une mise à jour par copier-coller ajoute et écrase, mais n'efface JAMAIS. Une
#  fonction retirée laisse donc deux traces bien différentes :
#    · son CODE, qui nous appartient -> on le supprime sans rien demander ;
#    · ses DONNÉES (poids téléchargés, dépôts clonés), qui pèsent parfois des
#      gigaoctets -> on les CHIFFRE et on les signale, mais on ne supprime
#      qu'avec « --purge », parce que l'utilisateur peut vouloir les récupérer
#      ailleurs avant.
#
#  Ajouter une entrée ici est la SEULE chose à faire quand on retire une
#  fonction : le nettoyage, le calcul de taille et le message suivent.
# --------------------------------------------------------------------------- #
REMOVED_FEATURES = [
    {"name": "Onglet Upscale (ancienne version)",
     "files": ["atelier/ui/creative_tab.py",
               "scripts/tools/run_creative_upscale.py"],
     "dirs": []},
    {"name": "Module Midjourney",
     "files": ["atelier/mjparams.py"], "dirs": []},
    {"name": "Backend ComfyUI et mode serveur",
     "files": ["atelier/engine/comfyui.py", "atelier/engine/sdserver.py",
               "scripts/get_comfyui.py",
               "config/comfyui_workflows/flux2.json",
               "config/comfyui_workflows/krea2.json",
               "config/comfyui_workflows/krea2int8.json",
               "config/comfyui_workflows/krea2convrot.json"],
     "dirs": ["config/comfyui_workflows", "comfyui"]},
    {"name": "Génération vidéo (LTX-2.3, MiniMax-H3)",
     "files": ["atelier/ui/video_tab.py", "atelier/engine/video.py"],
     "dirs": []},
]

# Dossiers de données à NE JAMAIS toucher.
PROTECTED = {"python", "bin", "models", "loras", "outputs", "userdata", ".git"}

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


def clean_removed_features(purge: bool) -> int:
    """Nettoie ce que les fonctions retirées ont laissé derrière elles.

    Le CODE part sans discussion (c'est le nôtre, et le garder fait tourner de
    l'ancien code par accident). Les DONNÉES sont d'abord CHIFFRÉES et
    signalées : supprimer plusieurs gigaoctets de poids sans prévenir n'est pas
    à nous de le décider. Renvoie l'espace récupérable restant, en octets.
    """
    print("• Fonctions retirées (code + données laissées derrière)…")
    touched = False
    recoverable = 0
    for feat in REMOVED_FEATURES:
        gone: list[str] = []
        for rel in feat["files"]:
            f = ROOT / rel
            if f.exists():
                try:
                    f.unlink()
                    gone.append(rel)
                except OSError as exc:
                    _warn(f"impossible de supprimer {rel} : {exc}")
        if gone:
            touched = True
            print(OK + f"{feat['name']} : {len(gone)} fichier(s) de code "
                  "supprimé(s).")
        for rel in feat["dirs"]:
            d = ROOT / rel
            if not d.is_dir():
                continue
            size = _dir_size(d)
            if size == 0:
                # Dossier vide : aucune donnée en jeu, on peut l'enlever.
                try:
                    shutil.rmtree(d)
                    print(OK + f"{feat['name']} : dossier vide {rel}/ supprimé.")
                    touched = True
                except OSError:
                    pass
                continue
            if purge:
                shutil.rmtree(d, ignore_errors=True)
                if d.exists():
                    _warn(f"suppression partielle : {rel}/")
                else:
                    print(OK + f"{feat['name']} : {rel}/ supprimé "
                          f"({_human(size)} libérés).")
                    touched = True
            else:
                recoverable += size
                print(INFO + f"{feat['name']} : {rel}/ occupe encore "
                      f"{_human(size)}.")
    if not touched and recoverable == 0:
        print(OK + "rien à nettoyer (propre).")
    return recoverable


def _known_addon_dirs() -> set[str]:
    """Add-ons LÉGITIMES, déduits du code plutôt que recopiés à la main.

    Ainsi, retirer un add-on de tools.py suffit à ce que son dossier devienne
    automatiquement un orphelin signalé ici — il n'y a pas de seconde liste à
    penser à mettre à jour."""
    from atelier.engine import tools
    dirs = (tools.DEPTH_MODEL_DIR, tools.BG_MODEL_DIR, tools.SAM_MODEL_DIR,
            tools.ENHANCE_MODEL_DIR, tools.UPSCALE_DIR, tools.SEEDVR2_DIR)
    out = set()
    for d in dirs:
        try:
            out.add(d.relative_to(tools.TOOLS_DIR).parts[0])
        except ValueError:
            pass
    return out


def report_orphan_addons(purge: bool) -> int:
    """Dossiers de tools_repo/ ne correspondant à aucun add-on du code actuel."""
    print("• Add-ons orphelins (tools_repo/)…")
    try:
        from atelier.engine import tools
        base, known = tools.TOOLS_DIR, _known_addon_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not base.is_dir():
        print(OK + "aucun add-on installé.")
        return 0
    # Les dossiers déjà nommés dans REMOVED_FEATURES sont traités plus haut :
    # les recompter ici gonflerait le total d'espace récupérable.
    declared = {Path(rel).name for f in REMOVED_FEATURES for rel in f["dirs"]}
    orphans = [d for d in sorted(base.iterdir())
               if d.is_dir() and d.name not in known and d.name not in declared]
    if not orphans:
        print(OK + "aucun add-on orphelin (propre).")
        return 0
    total = 0
    for d in orphans:
        size = _dir_size(d)
        total += size
        print(f"    - {d.name}  ({_human(size)})")
    if purge:
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
        return 0
    print(INFO + f"{len(orphans)} add-on(s) d'une version précédente = "
          f"{_human(total)} récupérables.")
    return total


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
    tous les composants des modèles + upscalers."""
    from atelier import registry, settings
    prefs = settings.load_prefs()
    repos: set[str] = set()
    for m in registry.load_base_models(prefs):
        repos.update(c.repo for c in m.components)
    up = registry.upscaler_config().get("repo")
    if up:
        repos.add(up)
    return {settings.model_repo_dir(r).name for r in repos if r}


def report_orphan_models(prune: bool) -> int:
    print("• Modèles orphelins (dossiers plus référencés par le catalogue)…")
    try:
        from atelier import settings
        models_dir = settings.MODELS_DIR
        expected = _expected_model_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not models_dir.is_dir():
        print(OK + "aucun dossier models/.")
        return 0
    orphans = [d for d in sorted(models_dir.iterdir())
               if d.is_dir() and d.name != "custom" and d.name not in expected]
    if not orphans:
        print(OK + "aucun modèle orphelin (propre).")
        return 0
    total = 0
    for d in orphans:
        size = _dir_size(d)
        total += size
        print(f"    - {d.name}  ({_human(size)})")
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
        return 0
    print(INFO + f"{len(orphans)} dossier(s) orphelin(s) = "
          f"{_human(total)} récupérables.")
    return total


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
        # Pas de « --variant cuda » en dur : sur Mac ce serait un mauvais
        # conseil (il n'existe que des builds Metal). get_sdcpp déduit seul.
        _warn("binaire sd-cli introuvable → install.bat / ./install.sh, ou "
              "python scripts/get_sdcpp.py")


def main() -> int:
    # « --purge » supprime TOUT ce qui reste des fonctions retirées : dossiers
    # d'add-ons, modèles orphelins, données laissées derrière. « --prune-models »
    # est conservé comme alias historique (il ne visait que les modèles).
    purge = "--purge" in sys.argv
    prune_models = purge or "--prune-models" in sys.argv
    print("=" * 60)
    print("  Maintenance — Turbo Slop Generator 3000")
    if purge:
        print("  (--purge : suppression des restes des fonctions retirées)")
    elif prune_models:
        print("  (--prune-models : suppression des modèles orphelins)")
    print("=" * 60)
    recoverable = clean_removed_features(purge)
    clean_pycache()
    clean_tmp()
    check_catalog()
    recoverable += report_orphan_addons(purge)
    recoverable += report_orphan_models(prune_models)
    compile_check()
    check_deps()
    check_engine()
    print("-" * 60)
    if recoverable > 0:
        # Un chiffre global, puis la commande exacte : c'est tout ce qu'il faut
        # pour décider, sans avoir à additionner les lignes soi-même.
        print(f"💾 {_human(recoverable)} récupérables (restes de fonctions "
              "retirées).")
        print("   Pour libérer :  maintenance.bat --purge"
              "   (./maintenance.sh --purge sur Linux/Mac)")
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
