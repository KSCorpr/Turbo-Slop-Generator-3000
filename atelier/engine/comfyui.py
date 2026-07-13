"""Moteur ComfyUI (backend PyTorch) piloté par API. EXPÉRIMENTAL.

Pourquoi : sd.cpp ne mappe qu'une partie des formats LoRA et prend du retard sur
les nouveaux modèles. ComfyUI (PyTorch) fait tourner les MÊMES modèles GGUF
(Flux.2/Krea, via le node ComfyUI-GGUF) avec un support LoRA natif complet.

Principe (comme sd-server) : un process ComfyUI résident, piloté en HTTP.
  • Démarrage : ComfyUI garde les modèles en cache entre les requêtes (pas de
    redémarrage au changement de modèle, contrairement à sd-server).
  • Génération : POST /prompt {graphe} → prompt_id → polling GET /history/{id} →
    images récupérées via GET /view.
  • Le graphe est un TEMPLATE (format API ComfyUI) dont on remplace des jetons
    (%POSITIVE%, %SEED%, %UNET%…). Tu peux remplacer un template par un que tu
    exportes depuis ton ComfyUI (« Save (API Format) ») s'il ne convient pas.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import settings
from .sdcpp import EngineError, GenRequest, _ref_list

try:
    import requests
except ImportError:
    requests = None  # type: ignore


class ComfyUnavailable(EngineError):
    """ComfyUI ne peut pas être démarré/joint → repli CLI légitime."""


_READY_TIMEOUT = 600.0
_JOB_TIMEOUT = 1800.0
_POLL_INTERVAL = 0.5

# sd.cpp -> ComfyUI : noms de samplers / schedulers.
_SAMPLER = {
    "euler": "euler", "euler_a": "euler_ancestral", "heun": "heun",
    "dpm2": "dpm_2", "dpm++2s_a": "dpmpp_2s_ancestral", "dpm++2m": "dpmpp_2m",
    "dpm++2mv2": "dpmpp_2m", "dpm++2m_sde": "dpmpp_2m_sde",
    "res_multistep": "res_multistep", "ipndm": "ipndm", "lcm": "lcm",
}
_SCHEDULER = {
    "simple": "simple", "karras": "karras", "exponential": "exponential",
    "discrete": "normal", "normal": "normal", "sgm_uniform": "sgm_uniform",
    "beta": "beta", "": "normal",
}


@dataclass
class _Handle:
    proc: subprocess.Popen
    port: int
    base_url: str
    client_id: str


_CURRENT: _Handle | None = None
_LOCK = threading.RLock()
_LOG_SINK: Callable[[str], None] | None = None
_CANCEL_ID: str | None = None


def _log(msg: str) -> None:
    if _LOG_SINK:
        _LOG_SINK(msg)


# --------------------------------------------------------------------------- #
#  Cycle de vie du process ComfyUI
# --------------------------------------------------------------------------- #
def _drain_stdout(proc: subprocess.Popen, tail: deque[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip("\n")
        tail.append(s)
        _log(s)


def _wait_ready(base_url: str, proc: subprocess.Popen, tail: deque[str]) -> None:
    deadline = time.time() + _READY_TIMEOUT
    url = f"{base_url}/system_stats"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ComfyUnavailable(
                "ComfyUI s'est arrêté au démarrage.\n"
                + "\n".join(list(tail)[-20:]))
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.7)
    raise ComfyUnavailable("ComfyUI n'a pas répondu à temps (timeout).")


def _ensure(gpu_index: int | None,
            log: Callable[[str], None] | None) -> _Handle:
    global _CURRENT
    if requests is None:
        raise ComfyUnavailable("Le paquet Python « requests » est requis.")
    comfy = settings.find_comfyui()
    if comfy is None:
        raise ComfyUnavailable(
            "ComfyUI n'est pas installé. Lancez « python scripts/get_comfyui.py » "
            "(ou le bouton d'installation dans Réglages).")
    with _LOCK:
        if _CURRENT is not None and _CURRENT.proc.poll() is None:
            return _CURRENT
        _CURRENT = None
        port = int(settings.load_prefs().get("comfyui_port", 8188))
        base_url = f"http://127.0.0.1:{port}"
        env = {**os.environ}
        cmd = [sys.executable, "main.py",
               "--listen", "127.0.0.1", "--port", str(port)]
        if gpu_index is not None:
            cmd += ["--cuda-device", str(gpu_index)]
        if log:
            log("$ " + " ".join(cmd))
            log("⏳ Démarrage de ComfyUI (long au 1er lancement)…")
        proc = subprocess.Popen(
            cmd, cwd=str(comfy), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
            encoding="utf-8", errors="replace")
        tail: deque[str] = deque(maxlen=300)
        threading.Thread(target=_drain_stdout, args=(proc, tail),
                         daemon=True).start()
        _wait_ready(base_url, proc, tail)
        _CURRENT = _Handle(proc=proc, port=port, base_url=base_url,
                           client_id=uuid.uuid4().hex)
        if log:
            log("✅ ComfyUI prêt.")
        return _CURRENT


# --------------------------------------------------------------------------- #
#  Template de workflow : chargement + patch par jetons
# --------------------------------------------------------------------------- #
def _template_path(family: str) -> Path:
    return settings.CONFIG_DIR / "comfyui_workflows" / f"{family}.json"


def _rel_to_models(p: Path | None) -> str:
    """Chemin d'un composant relatif à models/ (référence ComfyUI via
    extra_model_paths). Repli sur le nom de fichier si hors models/."""
    if not p:
        return ""
    p = Path(p)
    try:
        return p.relative_to(settings.MODELS_DIR).as_posix()
    except ValueError:
        return p.name


def _patch(node_tree: dict, tokens: dict) -> dict:
    """Remplace récursivement toute valeur EXACTEMENT égale à un jeton connu."""
    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str) and v in tokens:
            return tokens[v]
        return v
    return walk(node_tree)


def _real_nodes(graph: dict) -> dict:
    """Ne garde que les vrais nœuds (dict avec class_type) — vire __doc__ & co."""
    return {k: v for k, v in graph.items()
            if isinstance(v, dict) and "class_type" in v}


def _bypass_empty_loras(graph: dict) -> dict:
    """Contourne tout LoraLoader dont lora_name est vide : on rebranche ses
    consommateurs directement sur ses entrées model/clip, puis on le retire.
    Permet de garder UN template avec/sans LoRA sans le casser quand aucun LoRA."""
    remove = {nid: (n["inputs"].get("model"), n["inputs"].get("clip"))
              for nid, n in graph.items()
              if n.get("class_type") == "LoraLoader"
              and not n.get("inputs", {}).get("lora_name")}
    if not remove:
        return graph

    def redirect(link):
        if (isinstance(link, list) and len(link) == 2 and link[0] in remove):
            model_src, clip_src = remove[link[0]]
            return model_src if link[1] == 0 else clip_src
        return link

    for nid, node in graph.items():
        if nid in remove:
            continue
        for k, v in list(node.get("inputs", {}).items()):
            node["inputs"][k] = redirect(v)
    for nid in remove:
        graph.pop(nid, None)
    return graph


def build_workflow(req: GenRequest, family: str) -> dict:
    tmpl_path = _template_path(family)
    if not tmpl_path.is_file():
        raise ComfyUnavailable(
            f"Aucun template ComfyUI pour « {family} » "
            f"({tmpl_path.name}). Exportez un workflow (Save API Format) "
            "et déposez-le ici.")
    graph = json.loads(tmpl_path.read_text(encoding="utf-8"))
    tokens = {
        "%POSITIVE%": req.prompt or "",
        "%NEGATIVE%": req.negative or "",
        "%SEED%": int(req.seed) if req.seed is not None and req.seed >= 0
                  else int(time.time()) % (2**31),
        "%STEPS%": int(req.steps),
        "%CFG%": float(req.cfg_scale),
        "%WIDTH%": int(req.width),
        "%HEIGHT%": int(req.height),
        "%BATCH%": int(req.batch_count),
        "%SAMPLER%": _SAMPLER.get(req.sampler, "euler"),
        "%SCHEDULER%": _SCHEDULER.get(req.schedule, "normal"),
        "%UNET%": _rel_to_models(req.diffusion_model),
        "%CLIP%": _rel_to_models(req.text_encoder),
        "%VAE%": _rel_to_models(req.vae),
        "%DENOISE%": float(req.strength) if req.init_image else 1.0,
    }
    # LoRA (1er seulement pour l'itération 1 ; le template doit contenir un
    # LoraLoader avec %LORA_NAME%/%LORA_STRENGTH% pour que ce soit pris en compte).
    if req.lora_specs:
        p, w = req.lora_specs[0]
        tokens["%LORA_NAME%"] = Path(p).name
        tokens["%LORA_STRENGTH%"] = float(w)
    else:
        tokens["%LORA_NAME%"] = ""
        tokens["%LORA_STRENGTH%"] = 0.0
    graph = _patch(_real_nodes(graph), tokens)
    return _bypass_empty_loras(graph)


# --------------------------------------------------------------------------- #
#  Soumission + récupération
# --------------------------------------------------------------------------- #
def generate(req: GenRequest, output: Path, family: str,
             gpu_index: int | None = None,
             log: Callable[[str], None] | None = None) -> list[Path]:
    global _LOG_SINK, _CANCEL_ID
    _LOG_SINK = log
    graph = build_workflow(req, family)   # peut lever ComfyUnavailable
    handle = _ensure(gpu_index, log)
    base = handle.base_url
    try:
        r = requests.post(f"{base}/prompt",
                          json={"prompt": graph, "client_id": handle.client_id},
                          timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise ComfyUnavailable(f"Impossible de contacter ComfyUI : {exc}")
    if r.status_code != 200:
        # 400 = graphe invalide (souvent un nom de nœud/fichier qui ne colle pas).
        raise EngineError(f"ComfyUI a refusé le workflow ({r.status_code}) : "
                          f"{r.text[:400]}")
    prompt_id = r.json().get("prompt_id")
    _CANCEL_ID = prompt_id
    imgs = _await_images(base, prompt_id)
    if not imgs:
        raise EngineError("ComfyUI n'a produit aucune image.")
    paths: list[Path] = []
    for i, meta in enumerate(imgs):
        data = _fetch_view(base, meta)
        if not data:
            continue
        dest = output if len(imgs) == 1 else \
            output.with_name(f"{output.stem}_{i}{output.suffix}")
        dest.write_bytes(data)
        paths.append(dest)
    _CANCEL_ID = None
    if not paths:
        raise EngineError("ComfyUI a renvoyé des images vides.")
    return paths


def _await_images(base: str, prompt_id: str) -> list[dict]:
    deadline = time.time() + _JOB_TIMEOUT
    hurl = f"{base}/history/{prompt_id}"
    while time.time() < deadline:
        try:
            r = requests.get(hurl, timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise ComfyUnavailable(f"Perte de contact avec ComfyUI : {exc}")
        if r.status_code == 200:
            hist = r.json().get(prompt_id)
            if hist:
                status = (hist.get("status") or {}).get("status_str", "")
                if status == "error":
                    raise EngineError("ComfyUI : la génération a échoué "
                                      "(voir le journal).")
                out: list[dict] = []
                for node_out in (hist.get("outputs") or {}).values():
                    out += node_out.get("images", []) or []
                if out:
                    return [m for m in out if m.get("type") != "temp"]
        time.sleep(_POLL_INTERVAL)
    raise EngineError("Timeout : ComfyUI n'a pas fini la génération.")


def _fetch_view(base: str, meta: dict) -> bytes | None:
    params = {"filename": meta.get("filename", ""),
              "subfolder": meta.get("subfolder", ""),
              "type": meta.get("type", "output")}
    try:
        r = requests.get(f"{base}/view", params=params, timeout=30)
        return r.content if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def cancel_active() -> str:
    with _LOCK:
        handle = _CURRENT
    if handle is None or requests is None:
        return ""
    try:
        requests.post(f"{handle.base_url}/interrupt", timeout=5)
        return "⏹️ Génération ComfyUI annulée."
    except Exception:  # noqa: BLE001
        return ""


def shutdown() -> None:
    global _CURRENT
    with _LOCK:
        if _CURRENT is not None:
            try:
                _CURRENT.proc.terminate()
                try:
                    _CURRENT.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _CURRENT.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        _CURRENT = None


atexit.register(shutdown)
