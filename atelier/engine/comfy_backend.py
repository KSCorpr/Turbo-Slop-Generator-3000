"""Optional ComfyUI HTTP API backend for image generation.

Connects to a local ComfyUI server (default http://127.0.0.1:8188), reuses the
weights already downloaded in Turbo-Slop-Generator's ``models/`` directory via
an auto-generated ``extra_model_paths.yaml``, submits native ComfyUI API graphs
for every catalog model family, and saves rendered PNGs back into ``outputs/``.
"""
from __future__ import annotations

import atexit
import json
import os
import random
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

from .. import addons, hardware, registry, settings
from ..processes import close_worker
from . import sdcpp

DEFAULT_URL = "http://127.0.0.1:8188"
_PROC_LOCK = threading.Lock()
_MANAGED_PROC: subprocess.Popen | None = None
_MANAGED_URL: str | None = None

# Exact ComfyUI CLIPLoader ``type`` per Turbo-Slop-Generator model family
# (verified against Comfy-Org/ComfyUI nodes.py CLIPLoader.INPUT_TYPES).
_CLIP_TYPE_BY_FAMILY: dict[str, str] = {
    "minimax_h3_image": "minimax",
    "moss_image": "moss_image",
    "moss_image_turbo": "moss_image",
    "qwen21": "qwen_image",
    "fibo2": "fibo2",
    "fibo2_turbo": "fibo2",
    "flux2": "flux2",
    "krea2": "krea2",
    "krea2_realism": "krea2",
}

# Map sd.cpp sampler names to ComfyUI KSampler ``sampler_name`` values.
_SAMPLER_MAP: dict[str, str] = {
    "euler": "euler",
    "euler_a": "euler_ancestral",
    "heun": "heun",
    "lcm": "lcm",
    "dpm++2m": "dpmpp_2m",
    "dpm++2m_sde": "dpmpp_2m_sde",
    "dpm++2s_a": "dpmpp_2s_ancestral",
    "dpm2": "dpm_2",
    "ipndm": "ipndm",
    "ipndm_v": "ipndm_v",
    "res_2s": "res_2s",
    "res_multistep": "res_multistep",
    "er_sde": "er_sde",
    "ddim": "ddim",
}

# Map sd.cpp schedule names to ComfyUI KSampler ``scheduler`` values.
_SCHEDULER_MAP: dict[str, str] = {
    "": "simple",
    "auto": "simple",
    "simple": "simple",
    "discrete": "simple",
    "normal": "normal",
    "karras": "karras",
    "exponential": "exponential",
    "sgm_uniform": "sgm_uniform",
    "beta": "beta",
    "bong_tangent": "bong_tangent",
    "flux2": "simple",
}


def normalize_url(url: str | None) -> str:
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return DEFAULT_URL
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw


def extra_model_paths_file() -> Path:
    return settings.USERDATA_DIR / "comfyui_extra_model_paths.yaml"


def _collect_model_dirs() -> dict[str, list[str]]:
    """List local model directories so ComfyUI shares downloaded weights without copies."""
    diffusion_dirs: list[str] = []
    encoder_dirs: list[str] = []
    vae_dirs: list[str] = []

    def _add(target: list[str], path: Path) -> None:
        text = str(path.resolve())
        if text not in target:
            target.append(text)

    if settings.MODELS_DIR.is_dir():
        _add(diffusion_dirs, settings.MODELS_DIR)
        _add(encoder_dirs, settings.MODELS_DIR)
        _add(vae_dirs, settings.MODELS_DIR)
        for repo_dir in sorted(settings.MODELS_DIR.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                continue
            _add(diffusion_dirs, repo_dir)
            _add(encoder_dirs, repo_dir)
            _add(vae_dirs, repo_dir)
            for sub, dest in (
                ("diffusion_models", diffusion_dirs),
                ("text_encoders", encoder_dirs),
                ("vae", vae_dirs),
                ("split_files/vae", vae_dirs),
                ("split_files/diffusion_models", diffusion_dirs),
                ("split_files/text_encoders", encoder_dirs),
            ):
                candidate = repo_dir / sub
                if candidate.is_dir():
                    _add(dest, candidate)
    if settings.CUSTOM_DIR.is_dir():
        _add(diffusion_dirs, settings.CUSTOM_DIR)
        _add(encoder_dirs, settings.CUSTOM_DIR)
        _add(vae_dirs, settings.CUSTOM_DIR)
    lora_dirs = [str(settings.LORA_DIR.resolve())] if settings.LORA_DIR.is_dir() else [str(settings.LORA_DIR)]
    return {
        "diffusion_models": diffusion_dirs,
        "unet": diffusion_dirs,
        "text_encoders": encoder_dirs,
        "clip": encoder_dirs,
        "vae": vae_dirs,
        "loras": lora_dirs,
    }


def write_extra_model_paths() -> Path:
    """Write ``userdata/comfyui_extra_model_paths.yaml`` pointing ComfyUI to ``models/``."""
    settings.ensure_dirs()
    dirs = _collect_model_dirs()
    config = {
        "turbo_slop_generator": {
            "base_path": str(settings.MODELS_DIR.resolve()),
            "diffusion_models": "\n".join(dirs["diffusion_models"]),
            "unet": "\n".join(dirs["unet"]),
            "text_encoders": "\n".join(dirs["text_encoders"]),
            "clip": "\n".join(dirs["clip"]),
            "vae": "\n".join(dirs["vae"]),
            "loras": "\n".join(dirs["loras"]),
        }
    }
    target = extra_model_paths_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise sdcpp.EngineError(
            f"Cannot reach ComfyUI at {url}: {exc}. "
            "Start ComfyUI first (or switch the backend back to sd.cpp in Settings)."
        ) from exc


def _node_info(base_url: str, node_class: str, timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{base_url}/object_info/{node_class}", timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get(node_class) if isinstance(data, dict) else None
    except requests.RequestException:
        return None


def _input_choices(node_info: dict[str, Any] | None, field: str) -> list[str]:
    if not isinstance(node_info, dict):
        return []
    req = (node_info.get("input") or {}).get("required") or {}
    spec = req.get(field)
    if not isinstance(spec, list) or not spec:
        return []
    first = spec[0]
    if isinstance(first, list):
        return [str(x) for x in first]
    if isinstance(first, dict) and isinstance(first.get("options"), list):
        return [str(x) for x in first["options"]]
    return []


def is_managed_installed() -> bool:
    """Return True when the isolated managed ComfyUI + ComfyUI-GGUF runtime is ready."""
    root = addons.root("comfyui-backend")
    return (
        addons.ready("comfyui-backend")
        and (root / "source" / "main.py").is_file()
        and (root / "source" / "custom_nodes" / "ComfyUI-GGUF").is_dir()
    )


def is_running(url: str | None = None, timeout: float = 1.5) -> bool:
    base = normalize_url(url)
    try:
        resp = requests.get(f"{base}/system_stats", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def _host_port_from_url(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(normalize_url(url))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8188)
    return host, int(port)


def stop_managed() -> str:
    """Stop the managed ComfyUI server subprocess if we started it."""
    global _MANAGED_PROC, _MANAGED_URL
    with _PROC_LOCK:
        proc = _MANAGED_PROC
        _MANAGED_PROC = None
        _MANAGED_URL = None
    if proc is None or proc.poll() is not None:
        return "Managed ComfyUI server is not running."
    close_worker(proc)
    return "Managed ComfyUI server stopped (VRAM released)."


atexit.register(stop_managed)


def _server_log_path() -> Path:
    return settings.TMP_DIR / "comfyui_server.log"


def _read_server_log_tail(max_chars: int = 1600) -> str:
    try:
        text = _server_log_path().read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def _drain_server_log(offset: int, log: Callable[[str], None] | None) -> tuple[int, bool]:
    """Stream newly appended ComfyUI server log lines into ``log``."""
    path = _server_log_path()
    if not path.is_file():
        return offset, False
    try:
        size = path.stat().st_size
        if size <= offset:
            return offset, False
        with path.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read(size - offset)
    except OSError:
        return offset, False
    emitted = False
    if log and chunk:
        text = chunk.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            # Collapse tqdm carriage-return progress updates to the latest frame.
            line = raw_line.rsplit("\r", 1)[-1].strip()
            if line:
                log(f"[ComfyUI] {line}")
                emitted = True
    return size, emitted


def _managed_exit_code() -> int | None:
    with _PROC_LOCK:
        proc = _MANAGED_PROC
    if proc is None:
        return None
    return proc.poll()


def _format_exit_code(code: int) -> str:
    if code < 0 or code >= 0x80000000:
        return f"{code} (0x{code & 0xFFFFFFFF:08X})"
    return str(code)


def _managed_memory_flags(gpu_idx: int | None) -> list[str]:
    """Select ComfyUI memory-safety flags to prevent pinned-RAM/VRAM crashes."""
    flags: list[str] = []
    try:
        prof = hardware.auto_profile(gpu_idx)
        gpu = prof.gpu
        ram_gb = float(prof.ram_gb or 0)
    except Exception:
        gpu = None
        ram_gb = 0.0
    # On Windows (or <=16 GB VRAM hosts with <48 GB RAM), CUDA pinned host memory
    # locks huge non-pageable RAM slabs for 20+ GB models (e.g. MiniMax H3 Image),
    # triggering OS kills / HostBuffer crashes. Disabling pinned memory lets the
    # OS page offloaded tensors cleanly while Dynamic VRAM stays active.
    if os.name == "nt" or (gpu is not None and gpu.vram_gb <= 16.5 and ram_gb < 48.0):
        flags.append("--disable-pinned-memory")
    if gpu is not None and not gpu.is_apple and gpu.vram_gb <= 16.5:
        flags += ["--reserve-vram", "1.0"]
    return flags


def start_managed(
    url: str | None = None,
    *,
    timeout: float = 120.0,
    log: Callable[[str], None] | None = None,
) -> str:
    """Start the managed ComfyUI server in the background and wait until its API responds."""
    global _MANAGED_PROC, _MANAGED_URL
    base = normalize_url(url)
    if is_running(base, timeout=1.5):
        return f"ComfyUI is already running at {base}."
    if not is_managed_installed():
        raise sdcpp.EngineError(
            "Managed ComfyUI backend is not installed yet. "
            "Click “Install / update ComfyUI backend” in Settings."
        )
    host, port = _host_port_from_url(base)
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        raise sdcpp.EngineError(
            f"Cannot auto-start a remote ComfyUI server ({base}). "
            "Use http://127.0.0.1:8188 for the managed local backend."
        )
    cfg_path = write_extra_model_paths()
    root = addons.root("comfyui-backend")
    py = addons.python("comfyui-backend")
    main_py = root / "source" / "main.py"
    prefs = settings.load_prefs()
    gpu_idx = settings.generation_gpu_index(prefs)
    env = settings.child_env(gpu_idx)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    cmd = [
        str(py), "-u", str(main_py),
        "--listen", host,
        "--port", str(port),
        "--extra-model-paths-config", str(cfg_path),
        "--disable-auto-launch",
        *_managed_memory_flags(gpu_idx),
    ]
    if log:
        log(f"Starting managed ComfyUI backend on {base}…")
    with _PROC_LOCK:
        if _MANAGED_PROC is not None and _MANAGED_PROC.poll() is None:
            close_worker(_MANAGED_PROC)
        log_path = _server_log_path()
        settings.TMP_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd,
            cwd=str(root / "source"),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.close()
        _MANAGED_PROC = proc
        _MANAGED_URL = base

    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if sdcpp.was_cancelled():
            stop_managed()
            raise sdcpp.EngineError("ComfyUI startup cancelled.")
        if proc.poll() is not None:
            tail = _read_server_log_tail(1200)
            code_str = _format_exit_code(proc.returncode)
            stop_managed()
            raise sdcpp.EngineError(
                f"Managed ComfyUI server exited during startup (code {code_str}).\n{tail}"
            )
        if is_running(base, timeout=1.5):
            if log:
                log(f"Managed ComfyUI backend ready at {base}.")
            return f"Managed ComfyUI server started at {base}."
        time.sleep(0.5)

    stop_managed()
    raise sdcpp.EngineError(
        f"Managed ComfyUI server did not respond on {base} within {int(timeout)}s. "
        f"Check {_server_log_path()}."
    )


def ensure_running(url: str | None = None, log: Callable[[str], None] | None = None) -> str:
    """Return a reachable ComfyUI URL, auto-starting the managed server when installed."""
    base = normalize_url(url)
    if is_running(base, timeout=2.0):
        return base
    if is_managed_installed():
        start_managed(base, log=log)
        return base
    raise sdcpp.EngineError(
        f"ComfyUI is not running at {base} and the managed ComfyUI backend is not installed. "
        "Open Settings → Generation backend → ComfyUI and click “Install / update ComfyUI backend”."
    )


def check_connection(url: str | None = None) -> tuple[bool, str]:
    """Probe a running ComfyUI server and refresh ``comfyui_extra_model_paths.yaml``."""
    base = normalize_url(url)
    cfg_path = write_extra_model_paths()
    installed = is_managed_installed()
    try:
        stats = _get_json(f"{base}/system_stats", timeout=5.0)
    except sdcpp.EngineError:
        if installed:
            return (
                False,
                f"Managed ComfyUI backend is **installed** (`{addons.root('comfyui-backend')}`), "
                f"currently stopped. Click **Start ComfyUI** or just generate — it auto-starts on demand.",
            )
        return (
            False,
            f"ComfyUI is not installed and no external server responded at `{base}`.  \n"
            "Click **Install / update ComfyUI backend** above for 1-click setup "
            f"(or start your own server with `--extra-model-paths-config \"{cfg_path}\"`).",
        )
    sys_info = stats.get("system") or {}
    version = sys_info.get("comfyui_version") or "unknown"
    devices = stats.get("devices") or []
    gpu_desc = ""
    if devices:
        dev = devices[0]
        vram = float(dev.get("vram_total") or 0) / (1024 ** 3)
        gpu_desc = f" · {dev.get('name', 'GPU')} ({vram:.1f} GB VRAM)"
    has_gguf = _node_info(base, "UnetLoaderGGUF", timeout=4.0) is not None
    has_h3 = _node_info(base, "MiniMaxH3ImageConditioning", timeout=4.0) is not None
    kind = "managed" if installed else "external"
    notes = [
        f"Connected to **ComfyUI {version}** ({kind}) at `{base}`{gpu_desc}.",
        f"- GGUF loaders (`ComfyUI-GGUF`): **{'available' if has_gguf else 'not installed (safetensors models only)'}**",
        f"- Native MiniMax H3 Image nodes: **{'available' if has_h3 else 'not found (update ComfyUI)'}**",
        f"- Shared models config: `{cfg_path}`",
    ]
    return True, "\n".join(notes)


def interrupt(url: str | None = None) -> bool:
    """Request ComfyUI to interrupt the currently executing prompt."""
    base = normalize_url(url)
    try:
        resp = requests.post(f"{base}/interrupt", timeout=3.0)
        return resp.ok
    except requests.RequestException:
        return False


def resolve_comfy_weight(
    local_path: Path,
    available: list[str],
    *,
    kind: str,
    extra_paths: Path,
) -> str:
    """Match a local Turbo-Slop weight file to the exact relative name known by ComfyUI."""
    if not available:
        raise sdcpp.EngineError(
            f"ComfyUI lists no {kind} files. Launch ComfyUI with "
            f"`--extra-model-paths-config \"{extra_paths}\"` so it can read {settings.MODELS_DIR}."
        )
    norm_local = local_path.as_posix().lower()
    basename = local_path.name.lower()
    # Prefer longest suffix match (preserves repo/subfolder disambiguation).
    suffix_matches: list[tuple[int, str]] = []
    base_matches: list[str] = []
    for entry in available:
        norm_entry = entry.replace("\\", "/").lower()
        if norm_local.endswith("/" + norm_entry) or norm_local == norm_entry:
            suffix_matches.append((len(norm_entry), entry))
        elif Path(norm_entry).name == basename:
            base_matches.append(entry)
    if suffix_matches:
        suffix_matches.sort(reverse=True)
        return suffix_matches[0][1]
    if base_matches:
        return base_matches[0]
    raise sdcpp.EngineError(
        f"ComfyUI does not see {kind} file “{local_path.name}”. "
        f"Restart ComfyUI with `--extra-model-paths-config \"{extra_paths}\"` "
        f"(generated in Settings) or copy the weight into ComfyUI's models folder."
    )


def _upload_image(base_url: str, path: Path) -> str:
    with open(path, "rb") as fh:
        files = {"image": (path.name, fh, "application/octet-stream")}
        data = {"overwrite": "true", "type": "input"}
        try:
            resp = requests.post(f"{base_url}/upload/image", files=files, data=data, timeout=60.0)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise sdcpp.EngineError(f"Failed to upload {path.name} to ComfyUI: {exc}") from exc
    name = payload.get("name")
    subfolder = payload.get("subfolder") or ""
    if not name:
        raise sdcpp.EngineError(f"ComfyUI /upload/image returned no filename for {path.name}.")
    return f"{subfolder}/{name}" if subfolder else str(name)


def comfy_sampler_name(sampler: str | None) -> str:
    key = (sampler or "euler").strip().lower()
    return _SAMPLER_MAP.get(key, key)


def comfy_scheduler_name(schedule: str | None, family: str) -> str:
    key = (schedule or "").strip().lower()
    if key in ("", "auto"):
        if family in ("moss_image", "fibo2"):
            return "normal"
        return "simple"
    return _SCHEDULER_MAP.get(key, "simple")


def build_workflow(
    *,
    model: registry.BaseModel,
    diffusion_name: str,
    diffusion_is_gguf: bool,
    encoder_name: str | None,
    encoder_is_gguf: bool,
    vae_name: str,
    loras: list[tuple[str, float]],
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    schedule: str,
    seed: int,
    batch_count: int,
    uploaded_refs: list[str],
    uploaded_init: str | None,
    strength: float,
    vae_tiling: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build a native ComfyUI API prompt graph for any catalog image model family."""
    graph: dict[str, dict[str, Any]] = {}
    counter = 0

    def add(class_type: str, inputs: dict[str, Any]) -> str:
        nonlocal counter
        counter += 1
        nid = str(counter)
        graph[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    # 1) Diffusion model
    if diffusion_is_gguf:
        model_id = add("UnetLoaderGGUF", {"unet_name": diffusion_name})
    else:
        model_id = add("UNETLoader", {"unet_name": diffusion_name, "weight_dtype": "default"})
    model_ref: list[Any] = [model_id, 0]

    # 2) LoRA chain (model-only so it works across Flux2 / MOSS / Krea2 / H3)
    for lora_name, weight in loras:
        lora_id = add("LoraLoaderModelOnly", {
            "model": model_ref,
            "lora_name": lora_name,
            "strength_model": float(weight),
        })
        model_ref = [lora_id, 0]

    # 3) Text encoder (CLIP)
    clip_type = _CLIP_TYPE_BY_FAMILY.get(model.family, "stable_diffusion")
    if encoder_name:
        if encoder_is_gguf:
            clip_id = add("CLIPLoaderGGUF", {"clip_name": encoder_name, "type": clip_type})
        else:
            clip_id = add("CLIPLoader", {"clip_name": encoder_name, "type": clip_type, "device": "default"})
        clip_ref: list[Any] | None = [clip_id, 0]
    else:
        clip_ref = None

    # 4) VAE
    vae_id = add("VAELoader", {"vae_name": vae_name})
    vae_ref: list[Any] = [vae_id, 0]

    # 5) Reference / init image loaders
    ref_nodes = [add("LoadImage", {"image": name}) for name in uploaded_refs]
    init_node = add("LoadImage", {"image": uploaded_init}) if uploaded_init else None

    # 6) Family-specific conditioning + latent
    denoise = 1.0
    fam = model.family
    if fam == "minimax_h3_image":
        cond_inputs: dict[str, Any] = {
            "clip": clip_ref,
            "vae": vae_ref,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
        }
        for idx, rn in enumerate(ref_nodes, start=1):
            cond_inputs[f"images.image_{idx}"] = [rn, 0]
        cond_id = add("MiniMaxH3ImageConditioning", cond_inputs)
        pos_ref: list[Any] = [cond_id, 0]
        neg_id = add("ConditioningZeroOut", {"conditioning": pos_ref})
        neg_ref: list[Any] = [neg_id, 0]
        latent_ref: list[Any] = [cond_id, 1]
        if int(batch_count) > 1:
            rep_id = add("RepeatLatentBatch", {"samples": latent_ref, "amount": int(batch_count)})
            latent_ref = [rep_id, 0]
    elif fam in ("fibo2", "fibo2_turbo"):
        fibo_inputs: dict[str, Any] = {
            "clip": clip_ref,
            "prompt": prompt,
            "negative_prompt": negative or "",
            "vae": vae_ref,
            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_count),
        }
        for idx, rn in enumerate(ref_nodes, start=1):
            fibo_inputs[f"images.image_{idx}"] = [rn, 0]
        fibo_id = add("TextEncodeFibo2", fibo_inputs)
        pos_ref = [fibo_id, 0]
        neg_ref = [fibo_id, 1]
        latent_ref = [fibo_id, 2]
    elif fam == "qwen21":
        qwen_inputs: dict[str, Any] = {
            "clip": clip_ref,
            "prompt": prompt,
            "negative_prompt": negative or "",
            "vae": vae_ref,
            "resolution": int(max(width, height)),
        }
        for idx, rn in enumerate(ref_nodes, start=1):
            qwen_inputs[f"images.image_{idx}"] = [rn, 0]
        qwen_id = add("TextEncodeQwenImage21", qwen_inputs)
        pos_ref = [qwen_id, 0]
        neg_ref = [qwen_id, 1]
        if ref_nodes:
            latent_ref = [qwen_id, 2]
            if int(batch_count) > 1:
                rep_id = add("RepeatLatentBatch", {"samples": latent_ref, "amount": int(batch_count)})
                latent_ref = [rep_id, 0]
        else:
            lat_id = add("EmptySD3LatentImage", {
                "width": int(width), "height": int(height), "batch_size": int(batch_count),
            })
            latent_ref = [lat_id, 0]
    else:
        pos_id = add("CLIPTextEncode", {"clip": clip_ref, "text": prompt})
        pos_ref = [pos_id, 0]
        if negative and float(cfg_scale) > 1.0:
            neg_id = add("CLIPTextEncode", {"clip": clip_ref, "text": negative})
        else:
            neg_id = add("ConditioningZeroOut", {"conditioning": pos_ref})
        neg_ref = [neg_id, 0]

        if fam == "flux2" and ref_nodes:
            for rn in ref_nodes:
                enc_id = add("VAEEncode", {"pixels": [rn, 0], "vae": vae_ref})
                rl_id = add("ReferenceLatent", {"conditioning": pos_ref, "latent": [enc_id, 0]})
                pos_ref = [rl_id, 0]

        if init_node is not None:
            enc_id = add("VAEEncode", {"pixels": [init_node, 0], "vae": vae_ref})
            latent_ref = [enc_id, 0]
            if int(batch_count) > 1:
                rep_id = add("RepeatLatentBatch", {"samples": latent_ref, "amount": int(batch_count)})
                latent_ref = [rep_id, 0]
            denoise = max(0.01, min(1.0, float(strength)))
        elif fam in ("flux2", "moss_image", "moss_image_turbo"):
            lat_id = add("EmptyFlux2LatentImage", {
                "width": int(width), "height": int(height), "batch_size": int(batch_count),
            })
            latent_ref = [lat_id, 0]
        else:
            lat_id = add("EmptySD3LatentImage", {
                "width": int(width), "height": int(height), "batch_size": int(batch_count),
            })
            latent_ref = [lat_id, 0]

    # 7) Sampler
    sampler_id = add("KSampler", {
        "model": model_ref,
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg_scale),
        "sampler_name": comfy_sampler_name(sampler),
        "scheduler": comfy_scheduler_name(schedule, fam),
        "positive": pos_ref,
        "negative": neg_ref,
        "latent_image": latent_ref,
        "denoise": float(denoise),
    })

    # 8) Decode + Save
    if vae_tiling:
        dec_id = add("VAEDecodeTiled", {
            "samples": [sampler_id, 0],
            "vae": vae_ref,
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        })
    else:
        dec_id = add("VAEDecode", {"samples": [sampler_id, 0], "vae": vae_ref})

    add("SaveImage", {"images": [dec_id, 0], "filename_prefix": f"turboslop_{fam}"})
    return graph


def generate(
    *,
    model: registry.BaseModel,
    diffusion: Path,
    vae: Path | None,
    text_encoder: Path | None,
    loras: list[tuple[Path, float]],
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    schedule: str,
    seed: int,
    batch_count: int,
    ref_images: list[Path],
    init_image: Path | None,
    strength: float,
    vae_tiling: bool = True,
    url: str | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """Execute an image generation request on a running local ComfyUI server."""
    sdcpp._CANCELLED = False
    base = normalize_url(url)
    extra_paths = write_extra_model_paths()
    if vae is None:
        raise sdcpp.EngineError("ComfyUI backend requires a VAE file for this model.")

    # Ensure ComfyUI is reachable (auto-starting our managed server if installed).
    ensure_running(base, log=log)

    diff_gguf = diffusion.suffix.lower() == ".gguf"
    enc_gguf = bool(text_encoder and text_encoder.suffix.lower() == ".gguf")

    diff_loader = "UnetLoaderGGUF" if diff_gguf else "UNETLoader"
    diff_info = _node_info(base, diff_loader)
    if diff_info is None and diff_gguf:
        raise sdcpp.EngineError(
            f"{diffusion.name} is a GGUF model, which requires ComfyUI-GGUF (`UnetLoaderGGUF`) "
            "installed in ComfyUI — or select a .safetensors diffusion weight in Models."
        )
    diffusion_name = resolve_comfy_weight(
        diffusion, _input_choices(diff_info, "unet_name"),
        kind="diffusion model", extra_paths=extra_paths,
    )

    encoder_name: str | None = None
    if text_encoder is not None:
        enc_loader = "CLIPLoaderGGUF" if enc_gguf else "CLIPLoader"
        enc_info = _node_info(base, enc_loader)
        if enc_info is None and enc_gguf:
            raise sdcpp.EngineError(
                f"{text_encoder.name} is a GGUF text encoder, which requires ComfyUI-GGUF "
                "(`CLIPLoaderGGUF`) installed in ComfyUI — or pick a .safetensors encoder."
            )
        encoder_name = resolve_comfy_weight(
            text_encoder, _input_choices(enc_info, "clip_name"),
            kind="text encoder", extra_paths=extra_paths,
        )

    vae_info = _node_info(base, "VAELoader")
    vae_name = resolve_comfy_weight(
        vae, _input_choices(vae_info, "vae_name"),
        kind="VAE", extra_paths=extra_paths,
    )

    resolved_loras: list[tuple[str, float]] = []
    if loras:
        lora_info = _node_info(base, "LoraLoaderModelOnly")
        lora_avail = _input_choices(lora_info, "lora_name")
        for lora_path, weight in loras:
            resolved_loras.append((
                resolve_comfy_weight(lora_path, lora_avail, kind="LoRA", extra_paths=extra_paths),
                weight,
            ))

    if model.family == "minimax_h3_image" and _node_info(base, "MiniMaxH3ImageConditioning") is None:
        raise sdcpp.EngineError(
            "Your ComfyUI build does not expose `MiniMaxH3ImageConditioning`. "
            "Update ComfyUI to the latest version to run MiniMax H3 Image."
        )
    if model.family in ("fibo2", "fibo2_turbo") and _node_info(base, "TextEncodeFibo2") is None:
        raise sdcpp.EngineError(
            "Your ComfyUI build does not expose `TextEncodeFibo2`. Update ComfyUI first."
        )
    if model.family == "qwen21" and _node_info(base, "TextEncodeQwenImage21") is None:
        raise sdcpp.EngineError(
            "Your ComfyUI build does not expose `TextEncodeQwenImage21`. Update ComfyUI first."
        )

    uploaded_refs = [_upload_image(base, ref) for ref in ref_images]
    uploaded_init = _upload_image(base, init_image) if init_image else None

    eff_seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
    workflow = build_workflow(
        model=model,
        diffusion_name=diffusion_name,
        diffusion_is_gguf=diff_gguf,
        encoder_name=encoder_name,
        encoder_is_gguf=enc_gguf,
        vae_name=vae_name,
        loras=resolved_loras,
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        schedule=schedule,
        seed=eff_seed,
        batch_count=batch_count,
        uploaded_refs=uploaded_refs,
        uploaded_init=uploaded_init,
        strength=strength,
        vae_tiling=vae_tiling,
    )

    if log:
        log(f"ComfyUI ({base}): queuing {model.name} ({width}x{height}, {steps} steps, "
            f"{comfy_sampler_name(sampler)} / {comfy_scheduler_name(schedule, model.family)}, seed {eff_seed})…")

    log_offset = 0
    try:
        if _server_log_path().is_file():
            log_offset = _server_log_path().stat().st_size
    except OSError:
        log_offset = 0

    client_id = str(uuid.uuid4())
    try:
        resp = requests.post(
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30.0,
        )
        if not resp.ok:
            detail = ""
            try:
                err_json = resp.json()
                detail = json.dumps(err_json, ensure_ascii=False)[:600]
            except Exception:
                detail = resp.text[:600]
            raise sdcpp.EngineError(f"ComfyUI rejected the workflow ({resp.status_code}): {detail}")
        prompt_id = resp.json().get("prompt_id")
    except requests.RequestException as exc:
        raise sdcpp.EngineError(f"Failed to submit prompt to ComfyUI: {exc}") from exc

    if not prompt_id:
        raise sdcpp.EngineError("ComfyUI did not return a prompt_id.")

    # Poll /history/{prompt_id} until completed or cancelled.
    # Under heavy 20+ GB model loads on Windows, ComfyUI's aiohttp loop can
    # momentarily reset polling sockets (WinError 10054) while still healthy;
    # retry transient failures while the server is alive, and surface the real
    # server log + exit code immediately if the managed server crashes.
    t0 = time.monotonic()
    last_activity = t0
    error_since: float | None = None
    history_entry: dict[str, Any] | None = None
    while True:
        if sdcpp.was_cancelled():
            interrupt(base)
            raise sdcpp.EngineError("Generation cancelled.")

        log_offset, emitted = _drain_server_log(log_offset, log)
        now = time.monotonic()
        if emitted:
            last_activity = now

        try:
            resp = requests.get(
                f"{base}/history/{prompt_id}",
                timeout=15.0,
                headers={"Connection": "close"},
            )
            resp.raise_for_status()
            hist = resp.json()
            error_since = None
        except requests.RequestException as exc:
            exit_code = _managed_exit_code()
            if exit_code is not None:
                log_offset, _ = _drain_server_log(log_offset, log)
                tail = _read_server_log_tail(1600)
                code_str = _format_exit_code(exit_code)
                stop_managed()
                if "out of memory" in tail.lower():
                    raise sdcpp.VramError(
                        f"Managed ComfyUI server ran out of memory (exit code {code_str}).\n{tail}"
                    ) from exc
                raise sdcpp.EngineError(
                    f"Managed ComfyUI server crashed during generation (exit code {code_str}).\n"
                    f"Log tail ({_server_log_path()}):\n{tail}"
                ) from exc

            # Server is still running (or external): retry transient socket resets.
            managed_alive = _MANAGED_PROC is not None
            grace = 60.0 if managed_alive else 20.0
            if error_since is None:
                error_since = now
                if log:
                    log(f"ComfyUI: temporary connection hiccup ({type(exc).__name__}), retrying while server is busy…")
            elif now - error_since >= grace:
                tail = _read_server_log_tail(1200) if managed_alive else ""
                suffix = f"\nServer log tail:\n{tail}" if tail else ""
                raise sdcpp.EngineError(
                    f"Lost connection to ComfyUI at {base}/history/{prompt_id} for {int(grace)}s: {exc}.{suffix}"
                ) from exc
            time.sleep(1.0)
            continue

        if prompt_id in hist:
            log_offset, _ = _drain_server_log(log_offset, log)
            history_entry = hist[prompt_id]
            break

        if log and now - last_activity >= 5.0:
            log(f"ComfyUI: still running ({now - t0:.0f}s)…")
            last_activity = now
        time.sleep(0.5)

    status_block = history_entry.get("status") or {}
    if status_block.get("status_str") == "error":
        msgs = status_block.get("messages") or []
        err_text = next(
            (str(m[1].get("exception_message") or m[1]) for m in msgs
             if isinstance(m, list) and len(m) >= 2 and m[0] == "execution_error" and isinstance(m[1], dict)),
            str(msgs),
        )
        if "out of memory" in err_text.lower():
            raise sdcpp.VramError(f"ComfyUI CUDA out of memory: {err_text}")
        raise sdcpp.EngineError(f"ComfyUI execution failed: {err_text}")

    # Download generated images into OUTPUT_DIR with Turbo-Slop filenames.
    outputs = history_entry.get("outputs") or {}
    images_meta: list[dict[str, Any]] = []
    for node_out in outputs.values():
        if isinstance(node_out, dict) and isinstance(node_out.get("images"), list):
            images_meta.extend(node_out["images"])
    if not images_meta:
        raise sdcpp.EngineError("ComfyUI finished without producing any output image.")

    base_out = sdcpp.unique_output(model.family)
    saved: list[Path] = []
    for idx, meta in enumerate(images_meta):
        params = {
            "filename": meta["filename"],
            "subfolder": meta.get("subfolder", ""),
            "type": meta.get("type", "output"),
        }
        try:
            img_resp = requests.get(f"{base}/view", params=params, timeout=60.0)
            img_resp.raise_for_status()
        except requests.RequestException as exc:
            raise sdcpp.EngineError(f"Failed to download output image from ComfyUI: {exc}") from exc
        dest = base_out if len(images_meta) == 1 else base_out.with_name(f"{base_out.stem}_{idx}{base_out.suffix}")
        dest.write_bytes(img_resp.content)
        saved.append(dest)

    if log:
        log(f"ComfyUI: saved {len(saved)} image(s) in {time.monotonic() - t0:.1f}s.")
    return saved
