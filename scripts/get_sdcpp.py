#!/usr/bin/env python3
"""Télécharge un binaire pré-compilé de stable-diffusion.cpp (sd-cli) dans ./bin.

Pour Windows + CUDA, récupère DEUX archives :
  - la build principale  : sd-master-*-bin-win-cuda12-x64.zip  (contient sd-cli)
  - le runtime CUDA      : cudart-sd-bin-win-cu12-x64.zip       (DLLs CUDA)
…et les décompresse côte à côte (indispensable sans CUDA Toolkit installé).

Usage :
    python scripts/get_sdcpp.py                 # auto (CUDA)
    python scripts/get_sdcpp.py --variant cpu   # build CPU/AVX2
    python scripts/get_sdcpp.py --list          # liste les archives dispo
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

try:
    import requests  # plus robuste qu'urllib sur les redirections GitHub (Windows)
except ImportError:
    requests = None

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=10"
# Build MAISON (CI GitHub Actions du projet) : release au tag mouvant, binaire
# compilé pour les archis des cartes du projet (2080 Ti=75, 3060=86).
OUR_REPO = "KSCorpr/Turbo-Slop-Generator-3000"
OUR_TAG = "engine-latest"
_UA = {"User-Agent": "atelier"}


def _has_sd_cli() -> bool:
    names = ("sd-cli.exe", "sd.exe") if platform.system() == "Windows" \
        else ("sd-cli", "sd")
    return any(any(BIN_DIR.rglob(n)) for n in names) if BIN_DIR.exists() else False


# DLL du runtime CUDA 12 nécessaires à la build CUDA de stable-diffusion.cpp.
_CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def _cuda_runtime_present() -> bool:
    return all((BIN_DIR / d).is_file() for d in _CUDA_DLLS) or \
        bool(list(BIN_DIR.rglob("cudart64_12.dll")))


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def _copy_dlls_from(root: Path) -> int:
    """Copie les DLL CUDA trouvées sous `root` (récursif) vers bin/."""
    n = 0
    wanted = {d.lower() for d in _CUDA_DLLS}
    for dll in root.rglob("*.dll"):
        if dll.name.lower() in wanted and not (BIN_DIR / dll.name).exists():
            shutil.copy(dll, BIN_DIR / dll.name)
            print(f"     + {dll.name}", flush=True)
            n += 1
    return n


def _torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.submodule_search_locations:
        return None
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return lib if lib.is_dir() else None


def _nvidia_pkg_dir() -> Path | None:
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return None


def ensure_cuda_runtime() -> bool:
    """Met les DLL du runtime CUDA dans bin/ via des sources qui marchent
    partout (PyPI), SANS dépendre du CDN des releases GitHub.

    Ordre : déjà présent -> torch/lib (si torch CUDA installé) -> wheels NVIDIA
    PyPI -> en dernier recours, torch CUDA puis copie.
    """
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if _cuda_runtime_present():
        print("Runtime CUDA déjà présent dans bin/.")
        return True

    # 1) Réutiliser les DLL embarquées par un torch CUDA déjà installé.
    lib = _torch_lib_dir()
    if lib and (lib / "cudart64_12.dll").is_file():
        print("Copie des DLL CUDA depuis torch/lib…")
        if _copy_dlls_from(lib) >= 2:
            return True

    # 2) Wheels NVIDIA depuis PyPI (léger, et PyPI fonctionne sur votre réseau).
    try:
        print("Récupération du runtime CUDA via PyPI (nvidia-*-cu12)…")
        _pip("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")
        nv = _nvidia_pkg_dir()
        if nv and _copy_dlls_from(nv) >= 2:
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"   (wheels NVIDIA indisponibles : {exc})", flush=True)

    # 3) Dernier recours : torch CUDA (volumineux) puis copie des DLL.
    try:
        print("Installation de PyTorch CUDA 12.1 (fournit le runtime CUDA)…")
        # On évite --force-reinstall (verrouille tbb/mkl). On désinstalle juste
        # torch/torchvision puis on réinstalle la build CUDA.
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                         "torch", "torchvision"])
        _pip("--no-cache-dir", "torch==2.3.0", "torchvision==0.18.0",
             "--index-url", "https://download.pytorch.org/whl/cu121")
        lib = _torch_lib_dir()
        if lib and _copy_dlls_from(lib) >= 2:
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"   (échec torch : {exc})", flush=True)

    return _cuda_runtime_present()


def _force_ipv4():
    """Force la résolution IPv4 uniquement.

    Sur beaucoup de réseaux Windows, IPv6 est annoncé mais non routable : Python
    tente l'IPv6 du CDN GitHub et reste bloqué (pas de Happy-Eyeballs comme les
    navigateurs). On filtre getaddrinfo pour ne garder que l'IPv4.
    """
    _orig = socket.getaddrinfo

    def _ipv4_only(host, *args, **kwargs):
        res = _orig(host, *args, **kwargs)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 or res

    socket.getaddrinfo = _ipv4_only


def _fetch_json(url: str):
    if requests is not None:
        r = requests.get(url, headers=_UA, timeout=60)
        r.raise_for_status()
        return r.json()
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _platform_tokens() -> list[str]:
    s = platform.system().lower()
    return {"windows": ["win"], "linux": ["linux", "ubuntu"],
            "darwin": ["darwin", "macos"]}.get(s, [])


def _is_archive(n: str) -> bool:
    return n.endswith((".zip", ".tar.gz", ".tgz"))


def _score_main(name: str, variant: str) -> int:
    """Note l'archive PRINCIPALE (qui contient sd-cli). Exclut le cudart."""
    n = name.lower()
    if not _is_archive(n) or "cudart" in n:
        return -1000
    toks = _platform_tokens()
    if toks and not any(t in n for t in toks):
        return -1000
    score = 10
    if variant == "cuda":
        if "cuda12" in n or "cu12" in n:
            score += 10
        elif "cuda" in n:
            score += 8
        else:
            score -= 4              # pas une build CUDA
        if "rocm" in n or "vulkan" in n:
            score -= 20
    else:  # cpu
        if any(x in n for x in ("cuda", "rocm", "vulkan")):
            score -= 20
        if "avx2" in n:
            score += 6
        elif "avx512" in n:
            score += 3
        elif "avx" in n:
            score += 4
        elif "noavx" in n:
            score += 1
    if any(t in n for t in ("x64", "amd64", "x86_64")):
        score += 1
    return score


def _find_cudart(assets: list[dict]) -> dict | None:
    for a in assets:
        n = a["name"].lower()
        if "cudart" in n and "win" in n and _is_archive(n):
            return a
    return None


def _latest_release_with_assets() -> dict:
    data = _fetch_json(RELEASES)
    if isinstance(data, dict):  # message d'erreur (rate limit, etc.)
        sys.exit(f"API GitHub : {data.get('message', data)}")
    for rel in data:
        if rel.get("assets"):
            return rel
    sys.exit("Aucune release avec archives trouvée.")


def _progress(got: int, total: int, last: int) -> int:
    step = (total / 20) if total else 5_000_000
    if got - last >= step:
        if total:
            print(f"     {got/1e6:6.1f} / {total/1e6:.1f} Mo ({got*100//total}%)",
                  flush=True)
        else:
            print(f"     {got/1e6:6.1f} Mo téléchargés…", flush=True)
        return got
    return last


# Miroirs GitHub (essayés seulement si le téléchargement direct échoue).
# Utiles sur les réseaux qui filtrent/ralentissent le CDN des releases GitHub.
_MIRRORS = ["https://ghfast.top/", "https://ghproxy.net/", "https://gh.llkk.cc/"]


def _download(url: str) -> bytes:
    """Télécharge un fichier de façon ROBUSTE puis renvoie son contenu.

    Le CDN des releases GitHub coupe souvent la connexion en cours de route.
    On télécharge donc dans un fichier .part avec REPRISE (HTTP Range) : si la
    connexion lâche, on relance là où on s'était arrêté au lieu de tout refaire.
    Direct d'abord (avec reprise, nombreux essais), puis miroirs en secours.
    """
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in url.split("/")[-1] if c.isalnum() or c in "._-")
    tmp = BIN_DIR / ("._part_" + (safe or "download"))

    try:
        return _resumable(url, tmp, retries=10, resume=True)
    except Exception as exc:  # noqa: BLE001
        print(f"   Direct indisponible ({exc}). Essai via miroirs…", flush=True)

    for m in _MIRRORS:
        try:
            print(f"   Miroir : {m.split('/')[2]}", flush=True)
            if tmp.exists():
                tmp.unlink()
            return _resumable(m + url, tmp, retries=3, resume=True)
        except Exception as exc:  # noqa: BLE001
            print(f"     (miroir échoué : {exc})", flush=True)
    raise RuntimeError("téléchargement impossible (direct + miroirs)")


def _resumable(url: str, tmp: Path, retries: int, resume: bool) -> bytes:
    """Télécharge `url` dans `tmp` avec reprise, puis renvoie les octets."""
    import time
    if requests is None:  # repli minimal sans reprise
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    total = None
    last_print = 0
    for attempt in range(1, retries + 1):
        existing = tmp.stat().st_size if (resume and tmp.exists()) else 0
        headers = dict(_UA)
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=(10, 45)) as r:
                if r.status_code == 416:  # déjà complet
                    break
                r.raise_for_status()
                if r.status_code == 206:  # reprise acceptée
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            total = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                    mode = "ab"
                else:  # 200 : pas de reprise -> on repart de zéro
                    existing = 0
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl else None
                    mode = "wb"
                got = existing
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(262144):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                        last_print = _progress(got, total or 0, last_print)
            if total is None or tmp.stat().st_size >= total:
                break  # terminé
            raise IOError(f"interrompu à {tmp.stat().st_size}/{total} octets")
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise
            print(f"     (coupure : {exc} — reprise {attempt+1}/{retries}…)",
                  flush=True)
            time.sleep(min(2 * attempt, 8))

    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    print(f"     terminé ({len(data)/1e6:.1f} Mo).", flush=True)
    return data


def _extract(blob: bytes, name: str) -> None:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(BIN_DIR)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            t.extractall(BIN_DIR)


def _purge_old_binaries() -> None:
    for n in ("sd-cli.exe", "sd.exe", "sd-cli", "sd"):
        for p in BIN_DIR.rglob(n):
            try:
                p.unlink()
                print(f"     - ancien binaire retiré : {p.name}", flush=True)
            except OSError:
                pass


def _install_from_ours(args) -> None:
    """Installe le binaire depuis NOTRE release CI (build maison, archis ciblées).

    L'archive publiée par le workflow « Build sd.cpp (Windows CUDA) » contient
    déjà sd.exe + les DLL runtime CUDA : on la décompresse telle quelle dans bin/.
    """
    tag = args.tag or OUR_TAG
    url = f"https://api.github.com/repos/{OUR_REPO}/releases/tags/{tag}"
    print(f"Moteur build maison : release « {tag} » de {OUR_REPO}…")
    try:
        rel = _fetch_json(url)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Release CI introuvable ({exc}). Lancez d'abord le workflow "
                 "« Build sd.cpp (Windows CUDA) » dans l'onglet Actions du dépôt "
                 "GitHub (bouton « Run workflow »).")
    if not isinstance(rel, dict) or not rel.get("assets"):
        msg = rel.get("message") if isinstance(rel, dict) else rel
        sys.exit(f"Release CI « {tag} » absente ou vide ({msg}). Lancez le "
                 "workflow « Build sd.cpp (Windows CUDA) » (onglet Actions) "
                 "puis relancez cette mise à jour.")
    print(f"Release : {rel.get('tag_name')}")
    asset = next((a for a in rel["assets"]
                  if a["name"].lower().endswith(".zip")
                  and "win" in a["name"].lower()), None)
    if asset is None:
        print("Aucune archive Windows dans la release CI. Disponibles :")
        for a in rel["assets"]:
            print(" ", a["name"])
        sys.exit("Rien à installer.")

    if args.force:
        _purge_old_binaries()
    if _has_sd_cli() and not args.force:
        print("Binaire sd-cli déjà présent, on saute (utilisez --force pour MAJ).")
        return
    print(f"Téléchargement (build maison) : {asset['name']}")
    _extract(_download(asset["browser_download_url"]), asset["name"])

    # L'archive CI embarque déjà les DLL CUDA ; repli PyPI si jamais absentes.
    if platform.system().lower() == "windows" and not _cuda_runtime_present():
        print("Runtime CUDA absent de l'archive — repli via PyPI…")
        ensure_cuda_runtime()
    print(f"Décompressé dans {BIN_DIR}. Moteur CI (archis ciblées) prêt.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--source", choices=["official", "ours"], default="official",
                    help="official = releases leejet/stable-diffusion.cpp ; "
                         "ours = build maison CI du projet (archis 75;86)")
    ap.add_argument("--tag", default=OUR_TAG,
                    help="tag de la release CI (--source ours). Défaut : "
                         "engine-latest")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-télécharger même si un binaire est déjà présent "
                         "(pour METTRE À JOUR le moteur)")
    ap.add_argument("--allow-ipv6", action="store_true",
                    help="ne pas forcer l'IPv4 (par défaut on force l'IPv4)")
    args = ap.parse_args()

    if not args.allow_ipv6:
        _force_ipv4()

    if args.source == "ours":
        _install_from_ours(args)
        return

    print("Recherche de la dernière release stable-diffusion.cpp…")
    rel = _latest_release_with_assets()
    assets = rel["assets"]
    print(f"Release : {rel.get('tag_name')}")
    if args.list:
        for a in assets:
            print(" ", a["name"])
        return

    best = max(assets, key=lambda a: _score_main(a["name"], args.variant))
    if _score_main(best["name"], args.variant) <= 0:
        print("Aucune archive principale ne correspond. Disponibles :")
        for a in assets:
            print(" ", a["name"])
        sys.exit("Téléchargez-en une manuellement dans ./bin.")

    # Binaire principal (skip si déjà présent, utile en cas de relance).
    if args.force:
        _purge_old_binaries()   # MAJ : retirer les anciens binaires pour re-DL
    if _has_sd_cli() and not args.force:
        print("Binaire sd-cli déjà présent, on saute le téléchargement.")
    else:
        print(f"Téléchargement (binaire) : {best['name']}")
        _extract(_download(best["browser_download_url"]), best["name"])

    # Runtime CUDA (Windows) : récupéré via PyPI (pas le CDN GitHub, qui est
    # peu fiable sur certains réseaux).
    if args.variant == "cuda" and platform.system().lower() == "windows":
        if ensure_cuda_runtime():
            print("Runtime CUDA en place.")
        else:
            print("⚠️ Runtime CUDA non installé. Si sd-cli ne démarre pas, "
                  "installez le CUDA Toolkit 12, ou un upscaler (qui installe "
                  "PyTorch CUDA) depuis l'onglet Upscale.")

    print(f"Décompressé dans {BIN_DIR}. Binaire sd-cli prêt.")


if __name__ == "__main__":
    main()
