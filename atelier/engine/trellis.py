"""Moteur trellis.cpp : image → maillage 3D (GLB) via le binaire natif.

La release Windows prête fournit **`trellis-server.exe`** (serveur HTTP,
C++/GGML/CUDA, aucun PyTorch), pas de CLI one-shot. On le pilote donc de façon
**TRANSITOIRE** : on le démarre, on attend `/health`, on poste l'image sur
`/generate`, on récupère le GLB, puis on **arrête le serveur** → toute la VRAM
est libérée (stratégie low-VRAM, comme AISmith-3D).

API serveur (POST /generate, multipart) :
  - image        : fichier image (requis)
  - resolution   : 512 | 1024 | 1536   (512 « light » tient sur ≤ 12 Go)
  - seed         : entier (optionnel)
  - bg_removal   : threshold | birefnet
  Réponse : octets GLB bruts (model/gltf-binary).
"""
from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from .. import settings
from . import sdcpp

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

TRELLIS_BIN_DIR = settings.BIN_DIR / "trellis"
MODELS_DIR = settings.MODELS_DIR / "trellis"
DEFAULT_PORT = 8000

# Résolutions proposées. Seul le 512 tient sur 11-12 Go (les autres ~16 Go+).
RESOLUTIONS = [
    ("512 — léger (recommandé, ≤ 12 Go)", 512),
    ("1024 — cascade (≥ 16 Go)", 1024),
    ("1536 — haute (≥ 16 Go+)", 1536),
]


def _is_trellis_exe(p: Path) -> bool:
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    return "trellis" in stem and not any(
        x in stem for x in ("test", "bench", "studio", "convert"))


def find_server() -> Path | None:
    """Localise le binaire serveur trellis (nom variable selon la release)."""
    if settings.BIN_DIR.exists():
        exes = [p for p in settings.BIN_DIR.rglob("*") if _is_trellis_exe(p)]
        # Priorité à un binaire « server » ; sinon le premier exécutable trellis.
        srv = [p for p in exes if "server" in p.name.lower()]
        if srv:
            return srv[0]
        if exes:
            return exes[0]
    for name in ("trellis-server", "trellis"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def models_ready() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.rglob("*.gguf"))


def is_ready() -> bool:
    return find_server() is not None and models_ready()


def _wait_health(port: int, proc: subprocess.Popen, deadline: float) -> bool:
    """Attend que GET /health réponde « ok », tant que le serveur vit."""
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False                       # serveur mort pendant le chargement
        try:
            r = requests.get(url, timeout=2)
            if r.ok and "ok" in r.text.lower():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------
#  Serveur RÉSIDENT (optionnel) : garde le serveur en vie entre les générations
#  → plus de rechargement des modèles (~30 s gagnées par objet), mais la VRAM
#  reste occupée. À réserver aux séries de 3D ; à arrêter avant de générer des
#  images. Par défaut le mode transitoire (start/stop) reste actif.
# --------------------------------------------------------------------------
_RESIDENT: dict = {"proc": None, "port": None, "res": None}
_RES_LOCK = threading.Lock()


def resident_is_running() -> bool:
    p = _RESIDENT.get("proc")
    return p is not None and p.poll() is None


def resident_status() -> str:
    if resident_is_running():
        return f"🟢 Serveur résident actif (port {_RESIDENT['port']}, "\
               f"res {_RESIDENT['res']}) — VRAM occupée."
    return "⚪ Serveur résident arrêté (mode transitoire : démarrage/arrêt à "\
           "chaque génération, VRAM libérée)."


def resident_stop() -> str:
    with _RES_LOCK:
        p = _RESIDENT.get("proc")
        if p is not None:
            try:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            except Exception:  # noqa: BLE001
                pass
        _RESIDENT.update({"proc": None, "port": None, "res": None})
    return "⏹️ Serveur résident arrêté — VRAM libérée."


def build_server_args(res: int, decim: int = 0, atlas: int = 0,
                      no_texture: bool = False, box_uv: bool = False,
                      require_gpu: bool = True, f32: bool = False,
                      no_fa: bool = False,
                      gpu: int | None = None) -> list[str]:
    """Flags de lancement du serveur trellis (voir README trellis.cpp)."""
    args = ["--models", str(MODELS_DIR), "--res", str(int(res))]
    if gpu is not None:
        # Flag OFFICIEL de trellis (« --gpu N », N<0 = CPU) : plus propre que
        # CUDA_VISIBLE_DEVICES, et l'index correspond à celui affiché au démarrage.
        args += ["--gpu", str(int(gpu))]
    if decim and int(decim) > 0:
        args += ["--decim", str(int(decim))]       # cible de décimation (faces)
    if atlas and int(atlas) > 0:
        args += ["--atlas", str(int(atlas))]       # taille de l'atlas UV (px)
    if no_texture:
        args.append("--no-texture")                # géométrie seule (plus rapide)
    if box_uv:
        args.append("--box-uv")
    if require_gpu:
        # Évite un repli CPU SILENCIEUX (des heures de calcul) si la VRAM manque.
        args.append("--require-gpu")
    if f32:
        args.append("--f32")                       # précision 32 bits (défaut f16)
    if no_fa:
        args.append("--no-fa")                     # désactive FlashAttention
    return args


def generate(image_path: Path, out_path: Path, res: int = 512,
             seed: int | None = None, bg_removal: str = "birefnet",
             port: int = DEFAULT_PORT, extra: str = "",
             log: Callable[[str], None] | None = None,
             gpu_index: int | None = None,
             resident: bool = False,
             decim: int = 0, atlas: int = 0, no_texture: bool = False,
             box_uv: bool = False, require_gpu: bool = True,
             f32: bool = False, no_fa: bool = False) -> Path:
    """Génère un GLB 3D à partir d'une image via le serveur trellis.

    `resident=False` (défaut) : serveur démarré puis ARRÊTÉ (VRAM libérée).
    `resident=True` : serveur gardé en vie pour les générations suivantes.
    `gpu_index` : carte de calcul, via le flag officiel « --gpu N » (N<0 = CPU).
      Choisir la carte avec le plus de VRAM : le cascade 1024 est documenté
      pour ~16 Go, en dessous la géométrie sort dégradée (« blobs ») au lieu
      d'échouer proprement.
    """
    if requests is None:
        raise sdcpp.EngineError("Module « requests » manquant (pip install requests).")
    server = find_server()
    if server is None:
        raise sdcpp.EngineError(
            "Serveur trellis introuvable — installez trellis.cpp (onglet 3D).")
    if not models_ready():
        raise sdcpp.EngineError(
            "Modèles trellis absents — installez-les (onglet 3D).")
    settings.ensure_dirs()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(m: str) -> None:
        if log:
            log(m)

    # Réutilise le serveur résident s'il tourne DÉJÀ avec la même résolution.
    reuse = (resident and resident_is_running()
             and _RESIDENT.get("res") == int(res))
    if reuse:
        proc = _RESIDENT["proc"]
        use_port = _RESIDENT["port"]
        _log(f"♻️ Réutilisation du serveur résident (port {use_port}) — "
             "pas de rechargement des modèles.")
        return _post_generate(image_path, out_path, res, seed, bg_removal,
                              use_port, _log)

    # Un résident d'une AUTRE résolution doit céder la place.
    if resident_is_running():
        _log(resident_stop())

    cmd = [str(server)] + build_server_args(
        res, decim=decim, atlas=atlas, no_texture=no_texture, box_uv=box_uv,
        require_gpu=require_gpu, f32=f32, no_fa=no_fa, gpu=gpu_index)
    if extra and extra.strip():
        cmd += shlex.split(extra)
    # NB : on n'utilise PAS CUDA_VISIBLE_DEVICES ici — le flag « --gpu N » de
    # trellis fait le travail, et masquer les cartes en plus décalerait les
    # index (la carte N deviendrait la 0 pour le process).
    env = None

    _log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, cwd=str(settings.ROOT), env=env,
        encoding="utf-8", errors="replace")

    state: dict = {"port": None}

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.rstrip("\n")
            _log(s)
            if state["port"] is None:
                m = re.search(r"https?://[^\s:]+:(\d{2,5})", s) \
                    or re.search(r"(?:listen|port)\D{0,12}(\d{4,5})", s, re.I)
                if m:
                    state["port"] = int(m.group(1))

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()

    try:
        # Attente du chargement des modèles (~10 Go) puis de /health.
        _log("⏳ Démarrage du serveur trellis (chargement des modèles)…")
        deadline = time.time() + 600
        use_port = port
        # Laisse au serveur le temps d'annoncer son port dans ses logs.
        for _ in range(15):
            if state["port"]:
                use_port = state["port"]
                break
            if proc.poll() is not None:
                break
            time.sleep(1.0)
        if state["port"]:
            use_port = state["port"]
        if not _wait_health(use_port, proc, deadline):
            raise sdcpp.EngineError(
                f"Le serveur trellis n'a pas répondu sur le port {use_port} "
                "(voir le journal : port différent, VRAM insuffisante, ou "
                "modèles incomplets ?).")

        _log(f"✅ Serveur prêt (port {use_port}) — envoi de l'image…")
        _post_generate(image_path, out_path, res, seed, bg_removal, use_port,
                       _log)
    finally:
        if resident:
            # Mode résident : on GARDE le serveur en vie pour la suite.
            with _RES_LOCK:
                _RESIDENT.update({"proc": proc, "port": use_port,
                                  "res": int(res)})
            _log("♻️ Serveur gardé résident (VRAM occupée — « Arrêter le "
                 "serveur résident » pour la libérer).")
        else:
            # Mode transitoire : arrêt → libération VRAM (stratégie low-VRAM).
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:  # noqa: BLE001
                pass
    return out_path


def _post_generate(image_path: Path, out_path: Path, res: int,
                   seed: int | None, bg_removal: str, port: int,
                   _log: Callable[[str], None]) -> Path:
    """POST /generate (multipart) → écrit les octets GLB reçus."""
    data = {"resolution": str(int(res)), "bg_removal": bg_removal or "birefnet"}
    if seed is not None:
        data["seed"] = str(int(seed))
    with open(image_path, "rb") as fh:
        r = requests.post(f"http://127.0.0.1:{port}/generate",
                          files={"image": fh}, data=data, timeout=3600)
    if not r.ok:
        raise sdcpp.EngineError(
            f"trellis /generate a échoué (HTTP {r.status_code}) : "
            f"{r.text[:300]}")
    out_path.write_bytes(r.content)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise sdcpp.EngineError(
            "Aucun GLB produit — voir le journal (VRAM insuffisante en 512 ?).")
    _log(f"✅ GLB reçu : {out_path.name} ({out_path.stat().st_size/1e6:.1f} Mo)")
    return out_path
