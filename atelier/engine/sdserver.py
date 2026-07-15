"""Moteur en mode « serveur résident » : pilote sd-server (stable-diffusion.cpp).

Au lieu de relancer sd-cli à chaque image (ce qui recharge le modèle du disque
vers la VRAM), on garde un process sd-server chargé en mémoire et on lui envoie
des requêtes HTTP. Résultat : après le 1er chargement, chaque génération démarre
« à chaud » → itération quasi instantanée.

Contraintes de l'API sd-server (examples/server/api.md) exploitées ici :
  • Le modèle se choisit AU DÉMARRAGE (flags CLI). Un seul modèle résident à la
    fois : changer de modèle = tuer puis relancer le serveur.
  • Génération via POST /sdcpp/v1/img_gen → 202 + job id → polling
    GET /sdcpp/v1/jobs/{id} jusqu'à « completed ». Images en base64 (b64_json).
  • Les LoRA passent par le champ structuré `lora` (les tags <lora:…> du prompt
    sont volontairement ignorés par le serveur).
  • Sonde de disponibilité : GET /sdcpp/v1/capabilities (200 = modèle chargé).
"""
from __future__ import annotations

import atexit
import base64
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import settings
from .sdcpp import EngineError, GenRequest, _flag_args, _ref_list, _require


class ServerUnavailable(EngineError):
    """Le serveur ne peut pas être démarré/joint → repli CLI légitime.

    (À distinguer d'un job qui a bien été lancé mais a échoué : celui-là remonte
    en EngineError « normale » pour ne pas masquer un vrai problème.)"""


try:
    import requests
except ImportError:  # requests est dans requirements.txt, mais restons robustes
    requests = None  # type: ignore


# Délais (secondes) : le chargement d'un modèle en VRAM peut être long.
_READY_TIMEOUT = 600.0
_JOB_TIMEOUT = 1800.0
_POLL_INTERVAL = 0.4


@dataclass
class _Handle:
    proc: subprocess.Popen
    port: int
    signature: tuple
    base_url: str
    current_job: str | None = None


_CURRENT: _Handle | None = None
_LOCK = threading.RLock()
# Puits de log courant : la sortie du serveur (drainée par un thread démon) y est
# renvoyée, pour que l'utilisateur voie le chargement / la progression.
_LOG_SINK: Callable[[str], None] | None = None


def _log(msg: str) -> None:
    if _LOG_SINK:
        _LOG_SINK(msg)


# --- signature du modèle : ce qui impose (ou non) un redémarrage --------------
def _signature(req: GenRequest) -> tuple:
    """Tuple des paramètres FIXÉS AU DÉMARRAGE du serveur. S'il change, on relance.

    Les réglages par requête (prompt, steps, cfg, seed, taille, LoRA, images) n'y
    figurent PAS : ils passent dans le corps HTTP, sans redémarrage.
    """
    def s(p: Path | None) -> str:
        return str(p) if p else ""
    flags = tuple(sorted((k, bool(v)) for k, v in (req.flags or {}).items()))
    return (
        s(req.model_path), s(req.diffusion_model), s(req.vae),
        s(req.text_encoder), s(req.llm_vision),
        s(req.t5xxl), s(req.clip_l), s(req.uncond_model),
        req.vae_format, req.cache_mode, req.cache_option,
        req.gpu_index if req.gpu_index is not None else -1,
        req.encoder_gpu_index if req.encoder_gpu_index is not None else -1,
        bool(req.auto_fit), req.split_mode,
        flags,
    )


def _build_server_cmd(sd_server: Path, req: GenRequest, port: int) -> list[str]:
    """Arguments de DÉMARRAGE du serveur (chargement du modèle + optimisations).

    Reprend la partie « chargement de modèle » de sdcpp.build_gen_cmd : mêmes
    flags, même logique de split multi-GPU. Les paramètres par image ne sont PAS
    ici — ils partent dans chaque requête HTTP.
    """
    _require(req.model_path, req.diffusion_model, req.vae, req.text_encoder,
             req.llm_vision, req.t5xxl, req.clip_l, req.uncond_model)
    # sd-server nomme l'écoute réseau --listen-ip/--listen-port (PAS --host/--port,
    # qui sont ceux d'autres serveurs). Vérifié via `sd-server -h`.
    cmd: list[str] = [str(sd_server),
                      "--listen-ip", "127.0.0.1", "--listen-port", str(port)]
    if req.model_path:
        cmd += ["-m", str(req.model_path)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
    else:
        cmd += ["--diffusion-model", str(req.diffusion_model)]
        if req.uncond_model:
            cmd += ["--uncond-diffusion-model", str(req.uncond_model)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
        if req.text_encoder:
            cmd += ["--llm", str(req.text_encoder)]
        if req.llm_vision:
            cmd += ["--llm_vision", str(req.llm_vision)]
        if req.t5xxl:
            cmd += ["--t5xxl", str(req.t5xxl)]
        if req.clip_l:
            cmd += ["--clip_l", str(req.clip_l)]
    if req.vae_format:
        cmd += ["--vae-format", req.vae_format]
    # Le serveur charge en cache, AU DÉMARRAGE, les LoRA présents dans ce dossier.
    # Les requêtes les référencent ensuite par nom de fichier (voir _payload). Sans
    # ce flag, le cache est vide et toute requête LoRA échoue en « invalid lora ».
    # NB : un LoRA ajouté APRÈS le démarrage n'est pas vu tant qu'on ne relance pas.
    cmd += ["--lora-model-dir", str(settings.LORA_DIR)]
    cmd += list(req.extra_flags)
    if req.cache_mode:
        cmd += ["--cache-mode", req.cache_mode]
        if req.cache_option:
            cmd += ["--cache-option", req.cache_option]
    cmd += _flag_args(req.flags)
    # Multi-GPU : auto-fit (tout le modèle réparti, prioritaire) ou split d'encodeur.
    if req.auto_fit:
        cmd += ["--auto-fit"]
        if req.split_mode:
            cmd += ["--split-mode", req.split_mode]
    elif (req.encoder_gpu_index is not None
            and req.encoder_gpu_index != req.gpu_index):
        g = req.gpu_index if req.gpu_index is not None else 0
        e = req.encoder_gpu_index
        cmd += ["--backend", f"diffusion=cuda{g},vae=cuda{g},te=cuda{e}"]
    return cmd


def _drain_stdout(proc: subprocess.Popen, tail: deque[str]) -> None:
    """Thread démon : recopie la sortie du serveur vers le log courant + tail."""
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip("\n")
        tail.append(s)
        _log(s)


def _wait_ready(base_url: str, proc: subprocess.Popen, tail: deque[str]) -> None:
    """Attend que /sdcpp/v1/capabilities réponde 200 (modèle chargé)."""
    deadline = time.time() + _READY_TIMEOUT
    url = f"{base_url}/sdcpp/v1/capabilities"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ServerUnavailable(
                "sd-server s'est arrêté pendant le chargement du modèle.\n"
                + "\n".join(list(tail)[-15:]))
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:  # noqa: BLE001 — serveur pas encore prêt
            pass
        time.sleep(0.5)
    raise ServerUnavailable(
        "sd-server n'a pas répondu à temps (timeout de chargement).")


def _stop_locked(handle: _Handle | None) -> None:
    if handle is None:
        return
    try:
        handle.proc.terminate()
        try:
            handle.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _ensure(req: GenRequest, gpu_index: int | None,
            all_gpus: bool, log: Callable[[str], None] | None) -> _Handle:
    """Garantit un serveur vivant chargé avec le bon modèle. Relance si besoin."""
    global _CURRENT
    if requests is None:
        raise ServerUnavailable(
            "Le paquet Python « requests » est requis pour le mode serveur.")
    sd_server = settings.find_sd_server()
    if sd_server is None:
        raise ServerUnavailable(
            "Binaire sd-server introuvable. Mettez le moteur à jour "
            "(update-engine.bat) — il est livré avec sd-cli.")

    sig = _signature(req)
    with _LOCK:
        if (_CURRENT is not None and _CURRENT.proc.poll() is None
                and _CURRENT.signature == sig):
            return _CURRENT
        # Modèle différent (ou serveur mort) : on relance proprement.
        _stop_locked(_CURRENT)
        _CURRENT = None

        import os
        env = None
        if all_gpus:
            env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            env.pop("CUDA_VISIBLE_DEVICES", None)
        elif gpu_index is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}

        port = int(settings.load_prefs().get("sd_server_port", 7861))
        base_url = f"http://127.0.0.1:{port}"
        cmd = _build_server_cmd(sd_server, req, port)
        if log:
            log("$ " + " ".join(cmd))
            log("⏳ Démarrage du serveur et chargement du modèle "
                "(uniquement au 1er lancement / changement de modèle)…")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, cwd=str(settings.ROOT), env=env,
            encoding="utf-8", errors="replace")
        tail: deque[str] = deque(maxlen=200)
        threading.Thread(target=_drain_stdout, args=(proc, tail),
                         daemon=True).start()
        _wait_ready(base_url, proc, tail)
        _CURRENT = _Handle(proc=proc, port=port, signature=sig, base_url=base_url)
        if log:
            log("✅ Serveur prêt (modèle résident en mémoire).")
        return _CURRENT


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _payload(req: GenRequest) -> dict:
    """Corps JSON pour POST /sdcpp/v1/img_gen (réglages PAR IMAGE)."""
    sample: dict = {
        "sample_method": req.sampler,
        "sample_steps": int(req.steps),
        "guidance": {"txt_cfg": float(req.cfg_scale)},
    }
    if req.schedule:
        sample["scheduler"] = req.schedule
    body: dict = {
        "prompt": req.prompt,
        "width": int(req.width),
        "height": int(req.height),
        "seed": int(req.seed),
        "batch_count": int(req.batch_count),
        "sample_params": sample,
        "output_format": "png",
    }
    if req.negative and req.cfg_scale > 1.0:
        body["negative_prompt"] = req.negative
    if req.init_image:
        body["init_image"] = _b64(req.init_image)
        body["strength"] = float(req.strength)
    refs = _ref_list(req.ref_image)
    if refs:
        body["ref_images"] = [_b64(r) for r in refs]
    if req.lora_specs:
        # Le serveur résout le LoRA par sa clé de cache = chemin RELATIF au
        # --lora-model-dir (nom de fichier avec extension), pas le chemin absolu.
        # Les LoRA de l'app sont à plat dans LORA_DIR → la clé est le basename.
        body["lora"] = [{"path": Path(p).name, "multiplier": float(w)}
                        for p, w in req.lora_specs if p]
    return body


def generate(req: GenRequest, output: Path, gpu_index: int | None = None,
             all_gpus: bool = False,
             log: Callable[[str], None] | None = None) -> list[Path]:
    """Génère via le serveur résident et écrit les images dans outputs/.

    Renvoie la liste des chemins écrits. Lève EngineError en cas d'échec.
    """
    global _LOG_SINK
    _LOG_SINK = log
    refs = _ref_list(req.ref_image)
    _require(req.init_image, *refs, *[p for p, _ in req.lora_specs])
    handle = _ensure(req, gpu_index, all_gpus, log)
    base = handle.base_url
    try:
        r = requests.post(f"{base}/sdcpp/v1/img_gen", json=_payload(req),
                          timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise ServerUnavailable(f"Impossible de contacter sd-server : {exc}")
    if r.status_code not in (200, 202):
        raise EngineError(f"sd-server a refusé la requête ({r.status_code}) : "
                          f"{r.text[:300]}")
    job = r.json()
    job_id = job.get("id")
    poll_url = job.get("poll_url") or f"/sdcpp/v1/jobs/{job_id}"
    with _LOCK:
        if _CURRENT is not None:
            _CURRENT.current_job = job_id

    result = _poll_job(base, poll_url)
    images = (result or {}).get("images", [])
    if not images:
        raise EngineError("Le serveur n'a renvoyé aucune image.")
    paths: list[Path] = []
    for i, img in enumerate(images):
        b64 = img.get("b64_json") or ""
        if not b64:
            continue
        dest = output if len(images) == 1 else \
            output.with_name(f"{output.stem}_{i}{output.suffix}")
        dest.write_bytes(base64.b64decode(b64))
        paths.append(dest)
    with _LOCK:
        if _CURRENT is not None:
            _CURRENT.current_job = None
    if not paths:
        raise EngineError("Le serveur a renvoyé des images vides.")
    return paths


def _poll_job(base: str, poll_url: str) -> dict:
    deadline = time.time() + _JOB_TIMEOUT
    url = f"{base}{poll_url}"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Perte de contact avec sd-server : {exc}")
        if r.status_code == 410:
            raise EngineError("Job expiré côté serveur.")
        if r.status_code == 404:
            raise EngineError("Job introuvable côté serveur.")
        data = r.json()
        status = data.get("status", "")
        if status == "completed":
            return data.get("result", {})
        if status in ("failed", "error"):
            raise EngineError(
                f"Génération échouée : {data.get('error', 'raison inconnue')}")
        if status in ("cancelled", "canceled"):
            raise EngineError("Interrompu par l'utilisateur.")
        time.sleep(_POLL_INTERVAL)
    raise EngineError("Timeout : le serveur n'a pas fini la génération.")


def cancel_active() -> str:
    """Annule le job en cours (le serveur reste chaud, modèle conservé)."""
    with _LOCK:
        handle = _CURRENT
        job = handle.current_job if handle else None
    if handle is None or job is None or requests is None:
        return ""
    try:
        requests.post(f"{handle.base_url}/sdcpp/v1/jobs/{job}/cancel", timeout=5)
        return "⏹️ Génération annulée."
    except Exception:  # noqa: BLE001
        return ""


def refresh_loras() -> str:
    """Rafraîchit la liste des LoRA connus du serveur (après ajout d'un fichier).

    Le cache LoRA du serveur est peuplé au démarrage depuis --lora-model-dir. Pour
    prendre en compte un nouveau fichier sans redémarrer :
      1. on tente l'endpoint A1111 /sdapi/v1/refresh-loras (rapide, modèle reste
         chaud) — présent sur les builds récents ;
      2. sinon, on force un redémarrage propre au prochain usage (repeuple le
         cache ; coûte un rechargement du modèle).
    """
    global _CURRENT
    with _LOCK:
        handle = _CURRENT
    if handle is None or handle.proc.poll() is not None:
        return ""  # aucun serveur en cours : le prochain démarrage relira le dossier
    if requests is not None:
        try:
            r = requests.post(f"{handle.base_url}/sdapi/v1/refresh-loras",
                             timeout=15)
            if r.status_code in (200, 204):
                return "🔄 LoRA rafraîchis (serveur toujours chaud)."
        except Exception:  # noqa: BLE001 — endpoint absent → repli redémarrage
            pass
    with _LOCK:
        _stop_locked(_CURRENT)
        _CURRENT = None
    return ("🔄 LoRA rafraîchis — le serveur se relancera à la prochaine "
            "génération (un rechargement du modèle).")


def shutdown() -> None:
    """Arrête le serveur (appelé à la sortie de l'application)."""
    global _CURRENT
    with _LOCK:
        _stop_locked(_CURRENT)
        _CURRENT = None


atexit.register(shutdown)
