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
import io
import json
import os
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

# Réutilise les téléchargements ROBUSTES de get_sdcpp (reprise, miroirs, IPv4).
import get_sdcpp  # noqa: E402

# SOURCE DE VÉRITÉ UNIQUE pour les emplacements : atelier.settings (qui honore
# la préférence « models_dir », donc un dossier externe type NVMe). Ne JAMAIS
# redéfinir les chemins ici : l'installeur écrirait ailleurs que là où l'app
# cherche, et rien ne le signalerait.
from atelier import settings as _settings  # noqa: E402
from atelier.trellis_models import (TRELLIS_FILES, all_present,
                                   pixal_files)  # noqa: E402

BIN_DIR = _settings.BIN_DIR
TRELLIS_BIN_DIR = BIN_DIR / "trellis"
MANIFEST = "engine-manifest.json"
MODELS_DIR = _settings.MODELS_DIR / "trellis"

GH_RELEASE = "https://api.github.com/repos/pwilkin/trellis.cpp/releases/latest"
HF_MODEL_REPO = "ilintar/trellis2-gguf"
PIXAL_MODEL_REPO = "vegax87/Pixal3D"  # GGUF recommandés par trellis.cpp v0.8

# --------------------------------------------------------------------------- #
#  CHOIX DU PAQUET — et pourquoi ce n'est plus « Vulkan par défaut ».
#
#  Longtemps, la build CUDA de trellis.cpp ne tournait PAS sur toutes les cartes
#  NVIDIA : son CMakeLists ÉCRASAIT la liste d'architectures passée par la CI
#  (« set_target_properties(... CUDA_ARCHITECTURES "86;120") »), donc seules les
#  RTX 30xx et 50xx recevaient du code machine pour `deform_conv.cu` et
#  `decimate_qem.cu`. Ailleurs, « no kernel image is available » — et comme les
#  erreurs CUDA sont rémanentes, c'est l'opération ggml SUIVANTE qui la
#  rapportait, ce qui égarait le diagnostic. D'où Vulkan par défaut.
#
#  v0.6.0 (19 août 2026) corrige ça à la racine : l'écrasement est devenu un
#  simple défaut (« if(NOT CMAKE_CUDA_ARCHITECTURES) »), donc la liste de la CI
#  est enfin respectée, et une seconde archive vise les cartes anciennes.
#  Vérifié dans .github/workflows/release.yml au tag v0.6.0 :
#
#      cuda    (CUDA 13.1) -> archs 75;80;86;89;90;120   Turing et plus récent
#      cuda12  (CUDA 12.9) -> archs 60;61;70             Pascal et Volta
#
#  Concrètement : RTX 2080 Ti (7.5) et GTX 1080 Ti (6.1), jusqu'ici renvoyées
#  sur Vulkan, ont maintenant chacune leur paquet CUDA.
#
#  Vulkan reste le repli — rien n'y est compilé par architecture, la
#  convolution déformable passe par un shader de calcul — pour toute carte
#  qu'aucune des deux listes ne couvre (AMD, Intel, NVIDIA plus récente que la
#  CI amont).
# --------------------------------------------------------------------------- #
CUDA_ARCHS = frozenset({"7.5", "8.0", "8.6", "8.9", "9.0", "12.0"})
CUDA12_ARCHS = frozenset({"6.0", "6.1", "7.0"})


def cuda_build_supports(compute_cap: str) -> bool:
    """Une build CUDA amont couvre-t-elle cette carte ?"""
    cap = (compute_cap or "").strip()
    return cap in CUDA_ARCHS or cap in CUDA12_ARCHS


def preferred_backend(compute_cap: str | None = None) -> str:
    """« cuda », « cuda12 » ou « vulkan », selon la carte détectée.

    Sans capacité de calcul connue on choisit Vulkan : mieux vaut un paquet qui
    marche partout qu'un paquet plus rapide qui refuse de démarrer.
    """
    cap = (compute_cap or "").strip()
    if cap in CUDA_ARCHS:
        return "cuda"
    if cap in CUDA12_ARCHS:
        return "cuda12"
    return "vulkan"


def _is_binary(p: Path) -> bool:
    """Détecte un exécutable trellis utilisable (serveur OU cli, nom variable)."""
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    # La release Windows fournit trellis-server ; on accepte aussi trellis-cli.
    return ("trellis" in stem and not any(
        x in stem for x in ("test", "studio", "bench", "convert")))


def _find_binary(root: Path | None = None) -> Path | None:
    if root is None:
        installed = _find_binary(TRELLIS_BIN_DIR)
        return installed or _find_binary(BIN_DIR)
    if not root.exists():
        return None
    hits = [p for p in root.rglob("*") if _is_binary(p)]
    if not hits:
        return None
    # Priorité au serveur (ce que fournit la release), sinon le premier trellis.
    srv = [p for p in hits if "server" in p.name.lower()]
    return srv[0] if srv else hits[0]


def has_cli() -> bool:
    return _find_binary() is not None


def _list_exes(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(p.name for p in base.rglob("*")
                  if p.is_file() and p.name.lower().endswith(".exe"))


def has_models() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.rglob("*.gguf"))


def _backend_of(name: str) -> str:
    """Backend d'une archive, d'après son nom — sans confondre cuda et cuda12.

    « cuda » est un préfixe de « cuda12 » : un simple `in` ferait passer
    l'archive Pascal pour l'archive Turing+, et la carte récente hériterait
    d'un binaire compilé pour des architectures qu'elle n'a pas. On lit donc le
    segment ENTIER entre deux tirets (« trellis-<backend>-windows-x64.zip »).
    """
    parts = name.lower().replace(".zip", "").replace(".tar.gz", "").split("-")
    for token in parts:
        if token in ("cuda", "cuda12", "vulkan", "rocm"):
            return token
    return ""


def _pick_asset(assets: list[dict], backend: str = "cuda") -> dict | None:
    """Archive Windows du paquet demandé, avec repli sur Vulkan.

    Le repli compte : une release où un job de CI a échoué ne doit pas bloquer
    l'installation.
    """
    windows = {}
    for a in assets:
        n = a.get("name", "").lower()
        if not n.endswith(".zip") or "win" not in n or "studio" in n:
            continue
        kind = _backend_of(n)
        if kind and kind != "rocm" and kind not in windows:
            windows[kind] = a
    if backend in windows:
        return windows[backend]
    # Le SEUL repli valable est Vulkan. Passer de « cuda » à « cuda12 » (ou
    # l'inverse) donnerait un binaire compilé pour d'autres architectures que
    # celles de la carte : il se téléchargerait, s'installerait, et échouerait
    # au premier noyau par « no kernel image is available ».
    return windows.get("vulkan")


def install_binary(force: bool = False, log=print,
                   backend: str | None = None, update: bool = False) -> bool:
    installed = has_cli()
    if installed and not (force or update):
        log("The trellis binary is already there (use --update to check releases).")
        return True
    if backend is None:
        # Choix guidé par la carte : voir CUDA_ARCHS plus haut.
        cap = ""
        try:
            from atelier import hardware
            gpus = hardware.detect_gpus()
            if gpus:
                cap = max(gpus, key=lambda g: g.vram_gb).compute_cap
        except Exception:  # noqa: BLE001
            pass
        backend = preferred_backend(cap)
        sm = f"sm_{cap.replace('.', '')}" if cap else ""
        if backend == "vulkan":
            log("Paquet VULKAN retenu : "
                + (f"neither upstream CUDA archive compiles for {sm}."
                   if cap else "your card could not be identified.")
                + " Vulkan works on every card.")
        elif backend == "cuda12":
            log(f"CUDA12 package picked ({sm} — Pascal/Volta, the CUDA 12.9 "
                "archive meant for older cards).")
        else:
            log(f"CUDA package picked ({sm} — Turing or newer).")
    log("Looking for the latest pwilkin/trellis.cpp release…")
    try:
        rel = get_sdcpp._fetch_json(GH_RELEASE)
    except Exception as exc:  # noqa: BLE001
        log(f"Cannot check the latest release: {exc}; keeping the installed engine.")
        return False
    if isinstance(rel, dict) and rel.get("message") and not rel.get("assets"):
        log(f"API GitHub : {rel.get('message')}")
        return False
    tag = rel.get("tag_name", "")
    # Une release peut paraître avant ses archives CUDA. Pendant une mise à
    # jour, on garde le moteur installé au lieu de passer subrepticement à
    # Vulkan en attendant la fin des jobs CI. Une première installation peut
    # toujours se rabattre sur Vulkan.
    asset = _pick_asset(rel.get("assets", []), backend)
    if installed and asset and _backend_of(asset["name"]) != backend:
        log(f"The {backend} archive for {tag} is not published yet; "
            "keeping the installed engine. Try the update again later.")
        return True
    if not asset:
        if installed:
            log("The latest release has no compatible Windows archive yet; "
                "keeping the installed engine.")
            return True
        log("No Windows archive found in the release. Available:")
        for a in rel.get("assets", []):
            log("  " + a.get("name", "?"))
        return False
    manifest_path = TRELLIS_BIN_DIR / MANIFEST
    if installed and update and not force and manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (current.get("tag") == tag
                    and current.get("asset") == asset["name"]):
                log(f"trellis.cpp {tag} is already installed.")
                return True
        except (OSError, ValueError):
            pass
    log(f"Release : {tag} — {asset['name']} "
        f"({asset.get('size', 0) / 1e6:.0f} Mo)")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Même disque pour permettre un renommage atomique des dossiers. Aucun
        # fichier de l'ancienne version ne se retrouve dans la nouvelle.
        with tempfile.TemporaryDirectory(prefix=".trellis-update-",
                                         dir=BIN_DIR.parent) as temp:
            staged = Path(temp) / "new"
            staged.mkdir()
            blob = get_sdcpp._download(asset["browser_download_url"])
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(staged)
            if _find_binary(staged) is None:
                log("[!] The downloaded archive has no trellis executable; "
                    "keeping the installed engine. Contents: "
                    + ", ".join(_list_exes(staged)))
                return False
            (staged / MANIFEST).write_text(json.dumps({
                "tag": tag, "backend": _backend_of(asset["name"]),
                "asset": asset["name"],
            }, indent=2), encoding="utf-8")
            backup = Path(temp) / "previous"
            had_previous = TRELLIS_BIN_DIR.exists()
            if had_previous:
                os.replace(TRELLIS_BIN_DIR, backup)
            try:
                os.replace(staged, TRELLIS_BIN_DIR)
            except OSError:
                if had_previous:
                    os.replace(backup, TRELLIS_BIN_DIR)
                raise
        log(f"[OK] Binary installed: {_find_binary(TRELLIS_BIN_DIR)}")
        return True
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        log(f"[!] trellis.cpp update failed: {exc}; "
            "the previous engine was kept when possible.")
        return False


# Variantes de poids : f16 (défaut, dépôt racine) ou quantifiées (sous-dossiers
# q8/ et q4/ du dépôt HF). Tailles annoncées en amont.
VARIANTS = {
    "f16": ("repository root", "~16.5 GB — reference"),
    "q8": ("q8/", "~9.9 GB — near lossless"),
    "q4": ("q4/", "~6 GB — slight texture grain"),
}


def variant_dir(variant: str) -> Path:
    """Dossier local d'une variante (f16 = racine de models/trellis)."""
    return MODELS_DIR if variant == "f16" else MODELS_DIR / variant


def has_models(variant: str = "f16") -> bool:
    d = variant_dir(variant)
    # Des fichiers Pixal3D seuls, ou un téléchargement interrompu, ne rendent
    # pas TRELLIS.2 utilisable. Tester les dix fichiers réellement chargés.
    return all_present(d, TRELLIS_FILES)


def install_models(variant: str = "f16", log=print) -> bool:
    if variant not in VARIANTS:
        log(f"Variante inconnue : {variant} (attendu : "
            f"{', '.join(VARIANTS)}).")
        return False
    if has_models(variant):
        log(f"Trellis models “{variant}” already there, skipping.")
        return True
    try:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        _settings.configure_hf_env()
        from huggingface_hub import snapshot_download
    except ImportError:
        log("huggingface_hub manquant (pip install huggingface_hub).")
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "f16":
        # Racine seulement : on EXCLUT les sous-dossiers quantifiés.
        allow = ["*.gguf", "*.json", "*.txt"]
        ignore = ["q8/*", "q4/*"]
    else:
        allow = [f"{variant}/*"]
        ignore = None
    log(f"Downloading the trellis models “{variant}” "
        f"({VARIANTS[variant][1]}) → {variant_dir(variant)} (auto-resume)…")
    try:
        # local_dir = racine : les fichiers q8/ et q4/ atterrissent
        # naturellement dans leur sous-dossier.
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(MODELS_DIR),
                          allow_patterns=allow, ignore_patterns=ignore)
    except Exception as exc:  # noqa: BLE001
        log(f"Model download failed: {exc}")
        return False
    if has_models(variant):
        log(f"[OK] Trellis models “{variant}” in place.")
        return True
    log("[!] No .gguf after the download — check the HF repository.")
    return False


def install_pixal_models(variant: str = "f16", resolution: int = 512,
                         log=print) -> bool:
    """Ajoute 3 ou 5 GGUF Pixal sans dupliquer les décodeurs TRELLIS.

    Le téléchargement vit à la racine. Pour q4/q8, on crée des liens durs
    locaux vers ces mêmes octets : trellis.cpp n'accepte qu'un dossier --models
    et charge les décodeurs de la variante choisie dans ce dossier.
    """
    if variant not in VARIANTS or resolution not in (512, 1024):
        log("Pixal3D: choose a weight variant and 512 or 1024 resolution.")
        return False
    if not install_models(variant, log=log):
        return False
    required = pixal_files(resolution)
    if not all_present(MODELS_DIR, required):
        try:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            _settings.configure_hf_env()
            from huggingface_hub import snapshot_download
            missing = [name for name in required
                       if not (MODELS_DIR / name).is_file()]
            log(f"Downloading {len(missing)} Pixal3D GGUF from "
                f"{PIXAL_MODEL_REPO} into {MODELS_DIR} (auto-resume)…")
            snapshot_download(repo_id=PIXAL_MODEL_REPO,
                              local_dir=str(MODELS_DIR),
                              allow_patterns=missing)
        except Exception as exc:  # noqa: BLE001
            log(f"Pixal3D download failed: {exc}")
            return False
    if not all_present(MODELS_DIR, required):
        log("Pixal3D: one or more GGUF files are still missing.")
        return False
    if variant != "f16":
        target = variant_dir(variant)
        for name in required:
            src, dest = MODELS_DIR / name, target / name
            try:
                if dest.exists():
                    if os.path.samefile(src, dest):
                        continue
                    log(f"Pixal3D: {dest} already exists with different "
                        "content. Move it aside before installing.")
                    return False
                os.link(src, dest)
            except OSError as exc:
                log(f"Cannot share {name} with {variant} through a hard link: "
                    f"{exc}. Choose f16 or use an NTFS drive.")
                return False
        log(f"Pixal3D flows shared with {variant} without another disk copy.")
    log(f"[OK] Pixal3D {resolution} ready with {variant} weights "
        f"({'geometry only' if resolution == 512 else 'textured'}).")
    return True


def install_all(force: bool = False, variant: str = "f16", log=print,
                backend: str | None = None, update: bool = False) -> bool:
    ok_bin = install_binary(force=force, log=log, backend=backend,
                            update=update)
    ok_mdl = install_models(variant=variant, log=log)
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
    ap.add_argument("--models", action="store_true", help="models only")
    ap.add_argument("--pixal3d", action="store_true",
                    help="install Pixal3D 512 geometry or 1024 textured weights")
    ap.add_argument("--resolution", type=int, choices=[512, 1024], default=512,
                    help="Pixal3D weights to install: 512 geometry or 1024 textured")
    ap.add_argument("--variant", choices=list(VARIANTS), default="f16",
                    help="weight variant: f16 (default), q8 or q4")
    ap.add_argument("--force", action="store_true",
                    help="download the binary again even if it is present")
    ap.add_argument("--update", action="store_true",
                    help="check the latest release, download only if newer")
    ap.add_argument("--backend", choices=["auto", "cuda", "cuda12", "vulkan"],
                    default="auto",
                    help="package to install. auto (default) = cuda for "
                         "Turing and newer, cuda12 for Pascal/Volta, vulkan "
                         "when no CUDA archive covers the card.")
    ap.add_argument("--allow-ipv6", action="store_true")
    args = ap.parse_args()
    if not args.allow_ipv6:
        get_sdcpp._force_ipv4()

    # Trace explicite : on voit tout de suite OÙ ça s'installe (et donc si un
    # dossier de modèles externe est bien pris en compte).
    print(f"Models folder: {MODELS_DIR}")
    print(f"Engine folder: {TRELLIS_BIN_DIR}")

    backend = None if args.backend == "auto" else args.backend
    if args.pixal3d:
        ok = install_binary(update=True, backend=backend)
        if ok:
            from atelier.engine import trellis
            if not trellis.supports_pixal3d():
                print("This trellis.cpp binary does not support Pixal3D. "
                      "Update the binary to v0.8.0+ and retry.")
                ok = False
        if ok:
            ok = install_pixal_models(variant=args.variant,
                                      resolution=args.resolution)
    elif args.binary:
        ok = install_binary(force=args.force, update=args.update, backend=backend)
    elif args.models:
        ok = install_models(variant=args.variant)
    else:
        ok = install_all(force=args.force, update=args.update,
                         variant=args.variant,
                         backend=backend)
    print("Done." if ok else "Finished with errors (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
