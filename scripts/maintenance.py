#!/usr/bin/env python3
"""Maintenance : vérifie l'installation et nettoie ce qui traîne.

`update.bat` fait désormais les mises à jour proprement (il sait supprimer ce
qui a disparu du projet). Ce script reste utile pour deux choses : rattraper
les copies mises à jour à la main — dézipper par-dessus AJOUTE et REMPLACE,
mais n'efface JAMAIS ce qui a été retiré en amont — et vérifier que
l'installation est saine.

  • supprime le CODE des fonctions retirées (table REMOVED_FEATURES) ;
  • CHIFFRE les DONNÉES qu'elles ont laissées (poids, dépôts clonés) sans les
    supprimer — plusieurs gigaoctets ne s'effacent pas sans prévenir ;
  • repère les ADD-ONS orphelins de tools_repo/ (dossiers ne correspondant à
    aucun add-on du code actuel) et les MODÈLES orphelins de models/ (plus
    référencés par le catalogue) ;
  • purge les __pycache__ (.pyc d'anciens modules) et le dossier tmp/ ;
  • vérifie que tout compile, que le catalogue YAML est valide, que les
    dépendances et le binaire sd-cli sont présents, et qu'aucun fichier de
    l'application n'a disparu depuis la dernière mise à jour.

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
#
#  ⚠️ UN NOM DE FICHIER PEUT ÊTRE REPRIS. C'est arrivé : `atelier/engine/
#  sdserver.py` était listé ici (ancien backend serveur, retiré) et un nouveau
#  module du même nom est arrivé des mois plus tard. La maintenance l'effaçait
#  à chaque passage, et l'application ne démarrait plus. Une table de noms ne
#  peut pas savoir ça — c'est pourquoi `clean_removed_features` demande
#  maintenant au CODE ACTUEL s'il utilise le fichier avant de le supprimer.
# --------------------------------------------------------------------------- #
REMOVED_FEATURES = [
    {"name": "Upscale tab (old version)",
     "files": ["atelier/ui/creative_tab.py",
               "scripts/tools/run_creative_upscale.py"],
     "dirs": []},
    {"name": "Module Midjourney",
     "files": ["atelier/mjparams.py"], "dirs": []},
    {"name": "Backend ComfyUI",
     "files": ["atelier/engine/comfyui.py",
               "scripts/get_comfyui.py",
               "config/comfyui_workflows/flux2.json",
               "config/comfyui_workflows/krea2.json",
               "config/comfyui_workflows/krea2int8.json",
               "config/comfyui_workflows/krea2convrot.json"],
     "dirs": ["config/comfyui_workflows", "comfyui"]},
    {"name": "In-house engine build (the project's own CI)",
     "files": ["update-engine-ci.bat",
               ".github/workflows/build-sdcpp.yml"],
     "dirs": []},
    {"name": "Video generation (LTX-2.3, MiniMax-H3)",
     "files": ["atelier/ui/video_tab.py", "atelier/engine/video.py"],
     "dirs": []},
    # La sonde MiniMax-H3 a répondu à sa question (l'encodeur ne tient pas sur
    # une carte de 12 Go) ; elle est retirée avec le reste de MiniMax. Déclarée
    # ici pour que les copies déjà installées soient nettoyées à la maintenance.
    {"name": "Sonde MiniMax-H3",
     "files": ["scripts/try_minimax.py", "try-minimax.bat", "try-minimax.sh",
               "tests/test_try_minimax.py"],
     "dirs": []},
    # SeedVR2 : retiré. Ses DONNÉES pèsent lourd — un Python isolé complet
    # plus les poids 3B/7B — d'où le dossier déclaré ici plutôt que supprimé
    # d'office : plusieurs gigaoctets ne s'effacent pas sans prévenir.
    {"name": "SeedVR2 restoration",
     "files": ["scripts/setup_seedvr2.py", "tests/test_seedvr2.py"],
     "dirs": ["tools_repo/seedvr2"]},
    # Krea 2 INT8 ConvRot : la variante expérimentale et son banc A/B. Le
    # checkpoint de 13 Go vivait dans models/, sous le nom du dépôt Comfy-Org
    # — que le catalogue ne référence plus, donc le contrôle des modèles
    # orphelins le signalera de lui-même. Rien à déclarer ici pour lui.
    {"name": "Krea 2 INT8 ConvRot (experimental variant)",
     "files": [], "dirs": []},
    # Le moteur PyTorch de la branche Test7000, retiré à son tour. Aucune
    # DONNÉE propre : il lisait les mêmes fichiers GGUF que le moteur natif.
    {"name": "PyTorch generation engine",
     "files": ["scripts/setup_torch_engine.py", "setup-torch-engine.bat",
               "README-TORCH.md", "config/models_torch.yaml",
               "atelier/hfaccess.py", "atelier/engine/backends.py",
               "tests/test_torch_engine.py",
               "tests/fixtures/krea2_tensor_names.json"],
     "dirs": ["atelier/torchengine", "tests/fixtures"]},
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
    """Une taille lisible — en anglais, comme le reste de l'interface.

    Les unités étaient restées françaises (« 4.7 Go »), et le détecteur de
    français ne pouvait pas les voir : il lit les littéraux du code, or celles-
    ci sont assemblées à l'exécution. Troisième copie de la même échelle après
    `inventory` et `storage`, et la troisième à avoir eu le même oubli.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _still_in_service(rel: str) -> bool:
    """Ce fichier est-il utilisé par le code ACTUEL ?

    Un nom de fichier peut être repris des mois après le retrait de ce qu'il
    désignait. La table REMOVED_FEATURES ne peut pas le deviner : elle ne
    connaît que des chaînes de caractères. On demande donc au code actuel —
    et un module que quelqu'un importe est vivant, quoi qu'en dise la table.

    C'est un garde-fou, pas une devinette : en cas de doute (fichier hors
    atelier/, analyse impossible), on répond « oui, en service ». Refuser une
    suppression coûte un fichier mort de plus ; l'accepter à tort a coûté une
    application qui ne démarrait plus.
    """
    if not rel.endswith(".py"):
        return False
    path = ROOT / rel
    if not path.is_file():
        return False
    try:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    except ValueError:
        return True
    if not module.startswith("atelier."):
        # Un script ou un runner : il n'est importé par personne par
        # construction, la table reste seule juge.
        return False
    try:
        return module in _reachable_modules()
    except Exception:  # noqa: BLE001
        return True


def clean_removed_features(purge: bool) -> int:
    """Nettoie ce que les fonctions retirées ont laissé derrière elles.

    Le CODE part sans discussion (c'est le nôtre, et le garder fait tourner de
    l'ancien code par accident) — SAUF s'il est encore utilisé, cf.
    `_still_in_service`. Les DONNÉES sont d'abord CHIFFRÉES et signalées :
    supprimer plusieurs gigaoctets de poids sans prévenir n'est pas à nous de
    le décider. Renvoie l'espace récupérable restant, en octets.
    """
    print("• Removed features (code + the data they left behind)…")
    touched = False
    recoverable = 0
    for feat in REMOVED_FEATURES:
        gone: list[str] = []
        for rel in feat["files"]:
            f = ROOT / rel
            if f.exists():
                if _still_in_service(rel):
                    _warn(f"{rel} is listed as removed but the current code "
                          "uses it: NOT deleted. The name was reused — "
                          "take it out of REMOVED_FEATURES.")
                    continue
                try:
                    f.unlink()
                    gone.append(rel)
                except OSError as exc:
                    _warn(f"impossible de supprimer {rel} : {exc}")
        if gone:
            touched = True
            print(OK + f"{feat['name']}: {len(gone)} code file(s) deleted.")
        for rel in feat["dirs"]:
            d = ROOT / rel
            if not d.is_dir():
                continue
            size = _dir_size(d)
            if size == 0:
                # Dossier vide : aucune donnée en jeu, on peut l'enlever.
                try:
                    shutil.rmtree(d)
                    print(OK + f"{feat['name']}: empty folder {rel}/ deleted.")
                    touched = True
                except OSError:
                    pass
                continue
            if purge:
                shutil.rmtree(d, ignore_errors=True)
                if d.exists():
                    _warn(f"partial deletion: {rel}/")
                else:
                    print(OK + f"{feat['name']}: {rel}/ deleted "
                          f"({_human(size)} reclaimed).")
                    touched = True
            else:
                recoverable += size
                print(INFO + f"{feat['name']} : {rel}/ still holds "
                      f"{_human(size)}.")
    if not touched and recoverable == 0:
        print(OK + "nothing to clean up (clean).")
    return recoverable


def _known_addon_dirs() -> set[str]:
    """Add-ons LÉGITIMES, déduits du code plutôt que recopiés à la main.

    La liste était recopiée à la main et avait déjà pris du retard : `clip` et
    `describe` — le modèle image → prompt, 7,5 Go — étaient signalés comme
    orphelins, donc proposés à la suppression par `--purge`. On énumère
    maintenant les chemins déclarés par tools.py lui-même : ajouter un add-on
    suffit, en retirer un le rend automatiquement orphelin, et il n'y a plus de
    seconde liste à tenir à jour."""
    from atelier.engine import tools
    dirs = [value for name, value in vars(tools).items()
            if name.endswith("_DIR") and isinstance(value, Path)
            and value != tools.TOOLS_DIR]
    out = set()
    for d in dirs:
        try:
            out.add(d.relative_to(tools.TOOLS_DIR).parts[0])
        except ValueError:
            pass
    return out


def report_orphan_addons(purge: bool) -> int:
    """Dossiers de tools_repo/ ne correspondant à aucun add-on du code actuel."""
    print("• Orphan add-ons (tools_repo/)…")
    try:
        from atelier.engine import tools
        base, known = tools.TOOLS_DIR, _known_addon_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not base.is_dir():
        print(OK + "no add-on installed.")
        return 0
    # Les dossiers déjà nommés dans REMOVED_FEATURES sont traités plus haut :
    # les recompter ici gonflerait le total d'espace récupérable.
    declared = {Path(rel).name for f in REMOVED_FEATURES for rel in f["dirs"]}
    orphans = [d for d in sorted(base.iterdir())
               if d.is_dir() and d.name not in known and d.name not in declared]
    if not orphans:
        print(OK + "no orphan add-on (clean).")
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
                print(OK + f"deleted: {d.name}")
            else:
                _warn(f"partial deletion: {d.name}")
        print(OK + f"{_human(freed)} reclaimed.")
        return 0
    print(INFO + f"{len(orphans)} add-on(s) from a previous version = "
          f"{_human(total)} reclaimable.")
    return total


def clean_pycache() -> None:
    print("• Python caches (__pycache__ / .pyc)…")
    n = 0
    for p in ROOT.rglob("__pycache__"):
        if p.is_dir() and not any(part in PROTECTED for part in p.parts):
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    print(OK + f"{n} __pycache__ folder(s) purged.")


def clean_tmp() -> None:
    print("• tmp/ folder…")
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
    print(OK + f"{n} temporary item(s) deleted.")


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
    print("• Orphan models (folders the catalog no longer references)…")
    try:
        from atelier import settings
        models_dir = settings.MODELS_DIR
        expected = _expected_model_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not models_dir.is_dir():
        print(OK + "no models/ folder.")
        return 0
    orphans = [d for d in sorted(models_dir.iterdir())
               if d.is_dir() and d.name != "custom" and d.name not in expected]
    if not orphans:
        print(OK + "no orphan model (clean).")
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
                print(OK + f"deleted: {d.name}")
            else:
                _warn(f"partial deletion: {d.name}")
        print(OK + f"{_human(freed)} reclaimed.")
        return 0
    print(INFO + f"{len(orphans)} orphan folder(s) = "
          f"{_human(total)} reclaimable.")
    return total


def compile_check() -> None:
    print("• Compilation (syntax)…")
    import compileall
    ok = True
    for target in ("atelier", "scripts"):
        ok &= compileall.compile_dir(str(ROOT / target), quiet=1, force=True)
    ok &= compileall.compile_file(str(ROOT / "app.py"), quiet=1, force=True)
    if ok:
        print(OK + "all the Python code compiles.")
    else:
        global _problems
        _problems += 1
        print(ERR + "syntax error(s) above — an incomplete update?")


def check_catalog() -> None:
    print("• Model catalog (config/models.yaml)…")
    try:
        import yaml
        cat = yaml.safe_load((ROOT / "config" / "models.yaml")
                             .read_text(encoding="utf-8")) or {}
        models = [m.get("id") for m in cat.get("base_models", [])]
        print(OK + f"valid YAML — models: {', '.join(models) or '(none)'}.")
    except Exception as exc:  # noqa: BLE001
        _warn(f"models.yaml is unreadable: {exc}")


def check_deps() -> None:
    print("• Python dependencies…")
    missing = []
    for mod in ("gradio", "yaml", "PIL", "requests", "huggingface_hub"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if missing:
        _warn(f"missing: {', '.join(missing)} → run install.bat again "
              "(or install.sh).")
    else:
        print(OK + "present.")
    check_gradio_major()
    check_diffusers()


# Version de Gradio sous laquelle l'application est écrite. Une 5.x installée
# ne « manque » pas — elle est là, elle s'importe, et elle plante à la
# construction de la première image sur un argument inconnu. C'est le genre de
# panne qu'on veut voir NOMMÉE ici plutôt qu'à travers un TypeError.
GRADIO_MAJOR = 6


def check_gradio_major() -> None:
    print("• Gradio version…")
    try:
        import gradio
    except Exception as exc:  # noqa: BLE001
        _warn(f"gradio introuvable : {exc}")
        return
    version = getattr(gradio, "__version__", "0")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        _warn(f"version illisible : {version}")
        return
    if major < GRADIO_MAJOR:
        _warn(f"gradio {version} installed — the application asks for "
              f"{GRADIO_MAJOR}.x.")
        _warn("  The image components will refuse `buttons=` and the "
              "interface will not build.")
        _warn("  Fix: run install.bat again (or `pip install -U -r "
              "requirements.txt`).")
    elif major > GRADIO_MAJOR:
        _warn(f"gradio {version} installed, the application is written for "
              f"{GRADIO_MAJOR}.x — worth checking.")
    else:
        print(OK + f"gradio {version}.")


def check_diffusers() -> None:
    """Cohérence des paquets PARTAGÉS par les add-ons PyTorch.

    Tous vivent dans le même Python : un add-on installé avec une contrainte
    plus large écrase la version dont un autre a besoin, et la casse ne se voit
    qu'au premier usage de l'autre — sous forme d'une erreur illisible à
    l'import. On vérifie donc chaque paquet épinglé par l'installeur.
    """
    print("• Packages shared by the PyTorch add-ons…")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from setup_tools import _PINS
    except Exception:  # noqa: BLE001
        print(OK + "cannot be checked (the installer is missing).")
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
            _warn(f"{name} {got} installed — expected “{spec}”.")
    if not checked:
        print(OK + "no PyTorch add-on installed.")
    elif bad:
        _warn("  One add-on changed a version out from under the others.")
        _warn("  Fix: run that add-on's installer again (Toolkit → Install); "
              "it puts the right versions back.")
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


# --------------------------------------------------------------------------- #
#  CAPACITÉS ATTENDUES DU MOTEUR
#
#  Le dépôt et le binaire sd-cli se mettent à jour SÉPARÉMENT : copier le code
#  par-dessus l'ancien ne touche pas à bin/. Une fonction de l'application peut
#  donc réclamer une option que le moteur installé ne connaît pas encore, et
#  l'utilisateur ne le découvre qu'au moment où ça casse.
#
#  On liste donc ici ce dont le code a réellement besoin. Chaque entrée dit
#  quelle FONCTION dépend de quelle option : un « --hires manquant » ne parle à
#  personne, « l'onglet HD ne marchera pas » si.
# --------------------------------------------------------------------------- #
ENGINE_FEATURES = [
    {"option": "--hires", "needed_by": "the “🚀 HD” tab",
     "blocking": True},
    {"option": "--upscale-tile-size",
     "needed_by": "seam-free ESRGAN upscaling", "blocking": False},
    {"option": "--max-vram",
     "needed_by": "segmented execution (HD on a tight card)",
     "blocking": False},
    {"option": "--diffusion-conv-direct",
     "needed_by": "direct convolution (Settings)", "blocking": False},
    {"option": "--preview", "needed_by": "the live preview",
     "blocking": False},
]


def check_engine(update: bool) -> bool:
    """Présence ET capacités du moteur. Renvoie True si une MAJ est conseillée."""
    print("• stable-diffusion.cpp engine (sd-cli)…")
    try:
        from atelier import settings
        from atelier.engine import sdcpp
        sd = settings.find_sd_cli()
    except Exception as exc:  # noqa: BLE001
        _warn(f"cannot check: {exc}")
        return False

    if sd is None:
        # Pas de « --variant cuda » en dur : sur Mac ce serait un mauvais
        # conseil (il n'existe que des builds Metal). get_sdcpp déduit seul.
        if update:
            print(INFO + "binaire absent → installation…")
            return not _run_get_sdcpp()
        _warn("sd-cli binary not found → maintenance.bat --update-engine (or "
              "install.bat)")
        return True
    print(OK + f"found: {sd}")

    opts = sdcpp.supported_options(sd)
    if not opts:
        _warn("the binary does not answer “-h”: its capabilities cannot be "
              "checked. If it does not start either, reinstall it "
              "(maintenance.bat --update-engine).")
        return False

    missing = [f for f in ENGINE_FEATURES if f["option"] not in opts]
    if not missing:
        print(OK + f"{len(ENGINE_FEATURES)} expected capability/capabilities present.")
        return False
    for f in missing:
        line = f"{f['option']} absent → {f['needed_by']} will not work."
        if f["blocking"]:
            _warn(line)
        else:
            print(INFO + line)
    if update:
        print(INFO + "updating the engine…")
        return not _run_get_sdcpp(force=True)
    print(INFO + "The engine is older than the code. To bring it in line:")
    print("        maintenance.bat --update-engine"
          "   (./maintenance.sh --update-engine)")
    return True


def _run_get_sdcpp(force: bool = False) -> bool:
    """Lance scripts/get_sdcpp.py dans CE Python. True si ça a réussi.

    Sous-process plutôt qu'import : le script est fait pour être un programme
    (il appelle sys.exit), et un échec de téléchargement ne doit pas emporter
    la maintenance avec lui.
    """
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "get_sdcpp.py")]
    if force:
        cmd.append("--force")
    print(INFO + "$ " + " ".join(cmd))
    try:
        code = subprocess.call(cmd, cwd=str(ROOT))
    except OSError as exc:
        _warn(f"lancement impossible : {exc}")
        return False
    if code == 0:
        # Le cache d'options est indexé sur (chemin, mtime, taille) : un
        # nouveau binaire produit une clé différente, la relecture est donc
        # automatique. On revérifie pour AFFICHER le résultat, pas pour purger.
        print(OK + "engine installed/updated.")
        return True
    _warn(f"the engine update failed (code {code}). Network? "
          "Try again, or run update-engine.bat.")
    return False


def _module_map() -> dict[str, "Path"]:
    """Nom de module -> fichier, pour tout atelier/."""
    def module_of(path: Path) -> str:
        rel = path.relative_to(ROOT).with_suffix("")
        parts = [x for x in rel.parts if x != "__init__"]
        return ".".join(parts)

    return {module_of(x): x for x in (ROOT / "atelier").rglob("*.py")}


def _imports_of(path: "Path") -> set[str]:
    """Modules cités par les imports de ce fichier (relatifs résolus)."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    rel = path.relative_to(ROOT).with_suffix("")
    parts = [x for x in rel.parts if x != "__init__"]
    module = ".".join(parts)
    # Le paquet CONTENANT le fichier — et un __init__.py est contenu par son
    # propre paquet, pas par celui du dessus. Sans cette distinction,
    # « from . import sdserver » écrit dans atelier/engine/__init__.py
    # résolvait vers « atelier.sdserver », qui n'existe pas : le module
    # importé passait pour orphelin, et le garde-fou de suppression pour
    # inutile. Exactement le module qu'on venait d'effacer par erreur.
    if path == ROOT / "app.py":
        pkg = ""
    elif path.name == "__init__.py":
        pkg = module
    else:
        pkg = module.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                       # import relatif
                up = pkg.split(".")
                base = ".".join(up[:len(up) - node.level + 1]
                                + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{a.name}" for a in node.names)
    return found


_REACHABLE: "set[str] | None" = None


def _reachable_modules() -> set[str]:
    """Modules de atelier/ réellement atteints depuis les points d'entrée.

    Le calcul sert deux fois — signaler les orphelins, et protéger un fichier
    dont le nom a été repris — donc il est fait une seule fois.

    RGLOB sur scripts/, pas glob : les runners d'outils vivent dans
    scripts/tools/ et sont eux aussi des points d'entrée. Les oublier faisait
    passer pour orphelin tout module importé uniquement par eux — un faux
    positif qui pousse à supprimer du code vivant, soit exactement l'inverse
    du but.
    """
    global _REACHABLE
    if _REACHABLE is not None:
        return _REACHABLE
    files = _module_map()
    seen: set[str] = set()
    queue = [ROOT / "app.py"] + sorted((ROOT / "scripts").rglob("*.py"))
    while queue:
        path = queue.pop()
        for name in _imports_of(path):
            if name in seen or name not in files:
                continue
            seen.add(name)
            # Importer « atelier.ui.generate_tab » importe forcément le paquet
            # « atelier.ui » : sans ça, chaque __init__.py serait signalé
            # orphelin alors qu'il est la condition de tous ses modules.
            #
            # Et le paquet est EMPILÉ, pas seulement marqué : un __init__.py
            # contient du code, donc des imports. Le marquer « vu » sans le
            # lire faisait dépendre le résultat de l'ordre de parcours — si le
            # paquet était rencontré comme parent avant d'être rencontré comme
            # import, ses propres imports n'étaient jamais suivis. C'est ce qui
            # rendait `atelier.engine.sdserver` invisible : il n'est importé
            # que depuis `atelier/engine/__init__.py`.
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in seen:
                    seen.add(parent)
                    if parent in files:
                        queue.append(files[parent])
            queue.append(files[name])
    _REACHABLE = seen
    return seen


def check_orphan_modules() -> None:
    """Modules Python de atelier/ que plus RIEN n'importe.

    Complément générique à REMOVED_FEATURES : celle-ci ne connaît que les
    fonctions qu'on a pensé à y déclarer. Ici on part de app.py et des scripts,
    on suit les imports, et tout module de atelier/ jamais atteint est un reste
    d'une version précédente — quelle qu'elle soit, déclarée ou non.
    """
    print("• Orphan Python modules (nothing imports them any more)…")
    files = _module_map()
    seen = _reachable_modules()
    orphans = sorted(m for m in files if m not in seen and m != "atelier")
    if not orphans:
        print(OK + "no orphan module (clean).")
        return
    for m in orphans:
        print(f"    - {files[m].relative_to(ROOT)}")
    _warn(f"{len(orphans)} module(s) nothing imports — probably leftovers "
          "from a previous version. Check before deleting: a dynamically "
          "loaded module would show up here wrongly.")


USAGE = """Maintenance — Turbo Slop Generator 3000

  maintenance.bat                   asks: 1 update, 2 clean, 3 both,
                                    0 just check (the safe default)
  maintenance.bat --update-engine   + bring the sd-cli engine in line with the code
  maintenance.bat --purge           + delete the data of removed features,
                                    and the orphans
  maintenance.bat --all             everything: purge + engine update
                                    ("after an update, all tidy")

To update the APPLICATION itself: update.bat (this script downloads nothing).

(./maintenance.sh … on Linux/Mac)
Never touches models/custom/, loras/, outputs/, userdata/, python/.
"""


def check_install_complete() -> None:
    """Des fichiers de l'application ont-ils disparu depuis la mise à jour ?

    C'est le diagnostic qui manquait le jour où un module s'est volatilisé :
    l'application ne démarrait plus, avec un ImportError qui nommait le module
    mais pas la cause. Le manifeste de `update.bat` sait exactement ce qui
    devrait être là.
    """
    print("• Installation integrity…")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from update_app import MANIFEST, missing_files
    except Exception as exc:  # noqa: BLE001
        print(INFO + f"cannot be checked ({exc}).")
        return
    if not MANIFEST.is_file():
        print(INFO + "never updated by update.bat — nothing to compare.")
        return
    missing = missing_files()
    if not missing:
        print(OK + "every file from the last update is there.")
        return
    for rel in missing[:10]:
        print(f"    - {rel}")
    if len(missing) > 10:
        print(f"    … and {len(missing) - 10} more")
    _warn(f"{len(missing)} application file(s) have gone missing. "
          "Run update.bat: it will put them back.")


MENU = """What do you want to do?

  1  Update   — bring the sd-cli engine in line with the code
  2  Clean    — delete what removed features and orphans left behind
  3  Both
  0  Just check — the default: measures, deletes nothing

Your choice [0]: """


def ask_choice(read=input) -> tuple[bool, bool]:
    """(nettoyer, mettre à jour le moteur), demandé plutôt que deviné.

    Le menu n'existe que sans argument : les options en ligne de commande
    restent la référence, et un script qui appelle celui-ci ne doit jamais se
    retrouver bloqué sur une question.

    Tout ce qui n'est pas reconnu vaut « juste vérifier ». C'est le choix sûr,
    et c'est celui qu'on veut par défaut quand quelqu'un tape au hasard ou
    ferme la fenêtre.
    """
    print(MENU, end="", flush=True)
    try:
        answer = (read() or "").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False, False
    return {"1": (False, True), "2": (True, False),
            "3": (True, True)}.get(answer, (False, False))


def confirm_purge(total: int, read=input) -> bool:
    """Le garde-fou du choix 2 : on MONTRE avant de supprimer.

    Le script mesure d'abord, annonce le total, et ne supprime qu'ensuite.
    C'est toute la différence entre un nettoyage et une perte : plusieurs
    gigaoctets peuvent partir ici, et certains ne se retéléchargent qu'à
    travers une acceptation de licence.
    """
    if total <= 0:
        print(OK + "nothing to reclaim — nothing was deleted.")
        return False
    print("-" * 60)
    print(f"💾 {_human(total)} can be freed. This DELETES the files listed "
          "above.")
    print("   models/custom/, loras/, outputs/, userdata/ are never touched.")
    try:
        answer = (read("   Delete them? [y/N]: ") or "").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes", "o", "oui")


def main() -> int:
    # « --purge » supprime TOUT ce qui reste des fonctions retirées : dossiers
    # d'add-ons, modèles orphelins, données laissées derrière. « --prune-models »
    # est conservé comme alias historique (il ne visait que les modèles).
    if "--help" in sys.argv or "-h" in sys.argv:
        print(USAGE)
        return 0
    everything = "--all" in sys.argv
    purge = everything or "--purge" in sys.argv
    update_engine = everything or "--update-engine" in sys.argv
    prune_models = purge or "--prune-models" in sys.argv
    #  Menu SEULEMENT sans argument et devant un vrai terminal. Un appel
    #  automatisé — ou une sortie redirigée — ne doit jamais rester bloqué
    #  sur une question que personne ne lira.
    asked = False
    if len(sys.argv) == 1 and sys.stdin is not None and sys.stdin.isatty():
        purge, update_engine = ask_choice()
        prune_models = purge
        asked = True
    #  Quand le choix vient du menu, la suppression se fait en DEUX temps :
    #  la première passe mesure et montre, la confirmation décide, la seconde
    #  supprime. En ligne de commande `--purge` reste direct : celui qui l'a
    #  tapée a déjà décidé.
    deferred_purge = purge and asked
    if deferred_purge:
        purge = prune_models = False
    print("=" * 60)
    print("  Maintenance — Turbo Slop Generator 3000")
    modes = []
    if purge:
        modes.append("deleting what removed features left behind")
    if update_engine:
        modes.append("engine update")
    if modes:
        print("  (" + " + ".join(modes) + ")")
    print("=" * 60)
    recoverable = clean_removed_features(purge)
    clean_pycache()
    clean_tmp()
    check_catalog()
    recoverable += report_orphan_addons(purge)
    recoverable += report_orphan_models(prune_models)
    check_orphan_modules()
    check_install_complete()

    if deferred_purge:
        if confirm_purge(recoverable):
            print("-" * 60)
            freed = clean_removed_features(True)
            freed += report_orphan_addons(True)
            freed += report_orphan_models(True)
            recoverable = 0
        else:
            print(INFO + "nothing deleted.")

    compile_check()
    check_deps()
    engine_stale = check_engine(update_engine)
    print("-" * 60)
    if recoverable > 0:
        # Un chiffre global, puis la commande exacte : c'est tout ce qu'il faut
        # pour décider, sans avoir à additionner les lignes soi-même.
        print(f"💾 {_human(recoverable)} reclaimable (leftovers from removed "
              "features).")
        print("   To reclaim it:  maintenance.bat --purge   (./maintenance.sh "
              "--purge on Linux/Mac)")
    if engine_stale and not update_engine:
        print("🔧 The engine is behind the code.")
        print("   To bring everything in line at once:  maintenance.bat --all")
    if recoverable > 0 or (engine_stale and not update_engine):
        print("-" * 60)
    if _problems == 0:
        print("✅ Everything is clean and checked. You can run run.bat.")
    else:
        print(f"⚠️  Finished with {_problems} point(s) needing attention "
              "above (see the [!]/[X] lines).")
    print("=" * 60)
    return 0 if _problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
