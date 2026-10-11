"""Owned loopback Comfy process, cancellable jobs and step previews.

The UI never imports Torch, attaches to somebody else's Comfy instance or
accepts a remote engine URL. The process is retained between compatible jobs.
"""
from __future__ import annotations

import atexit
from collections import deque
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import struct
import subprocess
import threading
import time
import uuid

from PIL import Image, ImageOps
import requests
import websocket

from .. import comfy_runtime as runtime, settings
from ..fileio import atomic_write_text
from . import comfy_workflows as workflows
from .sdcpp import EngineError, VramError


class Engine:
    def __init__(self):
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._busy = False
        self._proc = None
        self._prompt_id = None
        self._fingerprint = None
        self._token = ""
        self.port = None
        self._tail = deque(maxlen=60)
        self._uploads = set()
        self._http = requests.Session()
        self._http.trust_env = False  # Corporate/cloud proxies never see prompts.

    @property
    def url(self):
        if self.port is None:
            raise EngineError("The local engine is not running.")
        return f"http://127.0.0.1:{self.port}"

    def request(self, path, *, payload=None, files=None, timeout=10):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Use a local engine route.")
        method = "POST" if payload is not None or files is not None else "GET"
        try:
            response = self._http.request(method, self.url + path, json=payload,
                         files=files, timeout=(2, timeout), allow_redirects=False)
            if response.status_code >= 300:
                raise EngineError("Local Comfy request failed: " + response.text[:1500])
            return response
        except requests.RequestException as exc:
            raise EngineError(f"The local Comfy engine did not respond: {exc}") from exc

    def check(self):
        if self._cancelled.is_set():
            raise EngineError("Generation cancelled. Completed images have been kept.")

    @contextmanager
    def session(self):
        with self._lock:
            with self._state_lock:
                self._busy = True
                self._cancelled.clear()
            try:
                yield self
            finally:
                with self._state_lock:
                    self._busy = False
                    self._prompt_id = None

    def _read_output(self, proc):
        try:
            for line in proc.stdout:
                self._tail.append(line.rstrip())
        finally:
            proc.stdout.close()

    def _stop(self):
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._fingerprint = None
        self._uploads.clear()
        self.port = None
        for path in (runtime.root() / "input").glob("*.png"):
            if len(path.stem) == 64 and all(c in "0123456789abcdef" for c in path.stem):
                path.unlink(missing_ok=True)  # Owned copies, never originals.

    def close(self):
        # An active sampler must not hold application shutdown on its lock.
        self._cancelled.set()
        self._stop()

    def release(self):
        # All UI GPU work uses the common queue. Also guard direct API callers.
        with self._state_lock:
            if self._busy:
                raise EngineError("A local Comfy task is still running. Wait or cancel it first.")
        with self._lock:
            self._stop()  # Exiting guarantees CUDA memory is returned before another tool.

    def ensure(self, roots, *, prefs=None, log=None):
        self.check()
        if not runtime.ready():
            raise EngineError("Install the Comfy local engine in System → Settings first.")
        from . import trellis
        if trellis.resident_is_running():
            message = trellis.resident_stop()
            if log:
                log(message)
        gpu = runtime.selected_gpu(prefs)
        fingerprint = (roots, gpu.index if gpu else None,
                       runtime.manifest().get("flavor"), runtime.COMMIT)
        if (self._proc is not None and self._proc.poll() is None
                and self._fingerprint == fingerprint):
            return
        self._stop()
        folder = runtime.root()
        for name in ("input", "output", "user"):
            (folder / name).mkdir(parents=True, exist_ok=True)
        settings.TMP_DIR.mkdir(parents=True, exist_ok=True)
        config = runtime.write_paths(roots)
        nodes = folder / "source" / "custom_nodes" / "turboslop"
        shutil.copytree(settings.ROOT / "scripts" / "comfy_nodes" / "turboslop",
                        nodes, dirs_exist_ok=True)
        self._token = secrets.token_hex(24)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        env = runtime.gpu_env(gpu.index if gpu else None)
        env["TURBOSLOP_COMFY_TOKEN"] = self._token
        self._tail.clear()
        if log:
            log("Starting the local Comfy engine on " + (gpu.name if gpu else "the processor") + "…")
            if runtime.manifest().get("flavor") == "cu126":
                log("CUDA 12.6 compatibility mode: standard attention and CPU offload; "
                    "the latest optimized INT8 CUDA kernels are unavailable with this runtime.")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        try:
            self._proc = subprocess.Popen(runtime.command(self.port, config, gpu=gpu),
                cwd=folder / "source", env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, **options)
            proc = self._proc
            threading.Thread(target=self._read_output, args=(proc,), daemon=True).start()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                self.check()
                if proc.poll() is not None:
                    raise EngineError("Comfy startup failed.\n" + "\n".join(self._tail))
                try:
                    info = self.request("/turboslop/health", timeout=1).json()
                    if info.get("token") != self._token or info.get("schema") != runtime.SCHEMA:
                        raise EngineError("The local engine port belongs to another process.")
                    self._fingerprint = fingerprint
                    if log:
                        log("Local Comfy engine ready. Models stay cached between images when memory allows.")
                    return
                except EngineError as exc:
                    if "another process" in str(exc):
                        raise
                self._cancelled.wait(0.2)
            raise EngineError("Local Comfy startup timed out.\n" + "\n".join(self._tail))
        except BaseException:
            self._stop()
            raise

    def cancel(self):
        with self._state_lock:
            if not self._busy:
                return "No Comfy generation running."
            self._cancelled.set()
            prompt_id = self._prompt_id
        if prompt_id:
            self._interrupt(prompt_id)
        return "Stopping the local Comfy generation…"

    def _interrupt(self, prompt_id):
        try:
            self.request("/queue", payload={"delete": [prompt_id]}, timeout=2)
            self.request("/interrupt", payload={"prompt_id": prompt_id}, timeout=2)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                queue = self.request("/queue", timeout=2).json()
                jobs = queue.get("queue_running", []) + queue.get("queue_pending", [])
                if not any(len(job) > 1 and job[1] == prompt_id for job in jobs):
                    return
                threading.Event().wait(0.1)
            self._stop()  # An unresponsive sampler cannot keep occupying VRAM.
        except EngineError:
            self._stop()

    def upload(self, source, *, size=None, rgba=False, mask=False) -> str:
        self.check()
        with Image.open(source) as opened:
            im = ImageOps.exif_transpose(opened).convert("L" if mask else "RGBA" if rgba else "RGB")
            if size:
                im = im.resize(size, Image.Resampling.BILINEAR if mask else Image.Resampling.LANCZOS)
            data = io.BytesIO()
            im.save(data, format="PNG")
        contents = data.getvalue()
        name = hashlib.sha256(contents).hexdigest() + ".png"
        if name not in self._uploads:
            response = self.request("/upload/image", files={
                        "image": (name, contents, "image/png")}, timeout=30).json()
            if response.get("name") != name or response.get("subfolder", ""):
                raise EngineError("The engine returned an unexpected local upload path.")
            self._uploads.add(name)
        return name

    @staticmethod
    def preview(blob: bytes, path: Path | None, index=0):
        if not path or len(blob) < 8:
            return
        kind = struct.unpack(">I", blob[:4])[0]
        if kind != 1:  # BinaryEventTypes.PREVIEW_IMAGE: type + format + bytes
            return
        try:
            with Image.open(io.BytesIO(blob[8:])) as im:
                dest = Path(str(path).replace("%03d", f"{index:03d}"))
                temporary = dest.with_name(dest.stem + ".part" + dest.suffix)
                im.convert("RGB").save(temporary)
                temporary.replace(dest)
                return True
        except (OSError, ValueError):
            return

    def execute(self, graph: workflows.Workflow, *, preview_path=None, preview_offset=0, log=None) -> list[Path]:
        self.check()
        client = uuid.uuid4().hex
        ws = None
        prompt_id = None
        start = time.monotonic()
        last_poll = 0
        history = None
        preview_index = preview_offset
        try:
            # Subscribe before submission: an immediate cached execution can
            # finish before a subscription made after POST /prompt.
            ws = websocket.create_connection(f"ws://127.0.0.1:{self.port}/ws?clientId={client}",
                                             timeout=0.5, http_no_proxy=["127.0.0.1"])
            response = self.request("/prompt", payload={"prompt": graph.nodes,
                                                       "client_id": client}, timeout=30).json()
            if response.get("node_errors") or not response.get("prompt_id"):
                raise EngineError("The local engine rejected this workflow: " + json.dumps(response)[:1800])
            prompt_id = response["prompt_id"]
            with self._state_lock:
                self._prompt_id = prompt_id
            deadline = time.monotonic() + 7200
            while time.monotonic() < deadline:
                self.check()
                if self._proc is not None and self._proc.poll() is not None:
                    raise EngineError("The local engine stopped during generation.\n" + "\n".join(self._tail))
                message = None
                if ws is not None:
                    try:
                        message = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        pass
                    except (websocket.WebSocketException, OSError):
                        ws.close()
                        ws = None  # History polling recovers a dropped preview connection.
                else:
                    self._cancelled.wait(0.25)
                if isinstance(message, bytes):
                    if self.preview(message, preview_path, preview_index):
                        preview_index += 1
                elif message:
                    try:
                        event = json.loads(message)
                    except (ValueError, TypeError):
                        continue
                    data = event.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        kind = event.get("type")
                        if kind == "progress" and log:
                            log(f"{data.get('value', 0)}/{data.get('max', 0)} · Comfy sampling")
                        elif kind == "execution_error":
                            detail = data.get("exception_message", "Unknown execution error")
                            error = VramError if "out of memory" in detail.lower() else EngineError
                            raise error("Comfy generation failed: " + detail)
                        elif kind == "execution_interrupted":
                            raise EngineError("Comfy generation interrupted.")
                if time.monotonic() - last_poll >= 0.5:
                    last_poll = time.monotonic()
                    history = self.request("/history/" + prompt_id).json().get(prompt_id)
                    if history:
                        break
            else:
                raise EngineError("Local generation timed out after two hours.")
            self.check()
            status = history.get("status", {})
            if status.get("status_str") == "error":
                detail = json.dumps(status.get("messages", []), ensure_ascii=False)[-1800:]
                error = VramError if "out of memory" in detail.lower() else EngineError
                raise error("Comfy generation failed: " + detail)
            images = history.get("outputs", {}).get(graph.output, {}).get("images", [])
            if not images:
                raise EngineError("The local engine produced no final image.")
            outputs = []
            settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for record in images:
                self.check()
                response = self._http.get(self.url + "/view", params=record,
                            timeout=(2, 30), allow_redirects=False)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as im:
                    im.load()
                    dest = settings.OUTPUT_DIR / ("comfy-" + uuid.uuid4().hex + ".png")
                    im.save(dest)  # RGBA preserved, output path chosen by TurboSlop.
                    outputs.append(dest)
                owned = runtime.root() / "output"
                stored = (owned / record.get("subfolder", "") / record.get("filename", "")).resolve()
                if stored.is_relative_to(owned.resolve()) and record.get("type") == "output":
                    stored.unlink(missing_ok=True)
            if log:
                log(f"Comfy generation completed in {time.monotonic() - start:.2f}s.")
            return outputs
        finally:
            if ws is not None:
                ws.close()
            if prompt_id:
                if self._cancelled.is_set() or history is None:
                    self._interrupt(prompt_id)
                try:
                    self.request("/history", payload={"delete": [prompt_id]}, timeout=2)
                except EngineError:
                    pass
            with self._state_lock:
                self._prompt_id = None


ENGINE = Engine()
atexit.register(ENGINE.close)


def release():
    """Return the resident process's GPU memory before a different engine runs."""
    ENGINE.release()


def cancel() -> str:
    return ENGINE.cancel()


def _lora_paths(loras):
    from .generate import _lora_file
    found = []
    for name, strength in loras or []:
        if not name or not strength:
            continue
        path = _lora_file(name)
        if path is None:
            raise EngineError(f"LoRA missing: {name}")
        if path.suffix.lower() != ".safetensors":
            raise EngineError("The Comfy edition uses safetensors LoRAs. Select a compatible local LoRA.")
        found.append((path, strength))
    return found


def generate(*, model, files, prompt, negative, steps, cfg_scale, width, height,
             seed, batch_count, sampler=None, schedule="auto", flow_shift=0,
             init_image=None, strength=0.6, mask_image=None, ref_image=None,
             loras=None, preview_path=None, hires=None, circular=False,
             log=None, prefs=None, on_output=None, **unused):
    if hires or circular or flow_shift:
        raise EngineError("Native HD, seamless patterns and flow shift are not available "
                          "for Comfy workflows yet. Use SDXL upscale for enlargement.")
    if not 1 <= int(batch_count) <= 16:
        raise EngineError("Choose between 1 and 16 images.")
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise EngineError("Choose an image size between 32 and 8192 pixels.")
    multiple = 32 if model.family == "qwen21" else 8
    if width % multiple or height % multiple:
        raise EngineError(f"This model needs dimensions that are multiples of {multiple}.")
    if model.family == "qwen21" and (steps != 8 or cfg_scale != 1.0
            or sampler not in (None, "", "euler") or schedule not in (None, "", "auto")):
        raise EngineError("Use the official Qwen Turbo preset: 8 steps, CFG 1, Euler and Auto schedule.")
    if model.family == "qwen21" and init_image and strength != 1.0:
        raise EngineError("Qwen Turbo editing uses reference images and its full eight-step schedule.")
    paths = [p for p in (files.get("diffusion"), files.get("model_path"),
              files.get("vae"), files.get("enc")) if p]
    lora_files = _lora_paths(loras)
    roots = runtime.model_roots(tuple(paths + [p for p, _ in lora_files]))
    named_loras = [(runtime.model_name(p, roots), w) for p, w in lora_files]
    references = list(ref_image) if isinstance(ref_image, (list, tuple)) else [ref_image] if ref_image else []
    if len(references) > 3:
        raise EngineError("Use up to three reference images.")
    with ENGINE.session():
        ENGINE.ensure(roots, prefs=prefs, log=log)
        refs = [ENGINE.upload(p, rgba=True, size=(width, height) if i == 0 else None)
                for i, p in enumerate(references)]
        init = ENGINE.upload(init_image, size=(width, height), rgba=model.family == "qwen21") if init_image else None
        mask = ENGINE.upload(mask_image, size=(width, height), mask=True) if mask_image else None
        result = []
        base_seed = secrets.randbelow(2**32) if seed is None or int(seed) < 0 else int(seed)
        for index in range(int(batch_count)):
            ENGINE.check()
            prefix = "turboslop/" + uuid.uuid4().hex
            if model.family == "qwen21":
                graph = workflows.qwen(diffusion=runtime.model_name(files["diffusion"], roots),
                    encoder=runtime.model_name(files["enc"], roots),
                    vae=runtime.model_name(files["vae"], roots), prompt=prompt,
                    width=width, height=height, seed=base_seed + index, prefix=prefix,
                    references=refs, init=init, mask=mask, loras=named_loras)
            else:
                graph = workflows.sdxl(checkpoint=runtime.model_name(files["model_path"], roots),
                    prompt=prompt, negative=negative, width=width, height=height,
                    seed=base_seed + index, steps=steps, cfg=cfg_scale, prefix=prefix,
                    init=init, mask=mask, strength=strength, sampler=sampler or "euler",
                    scheduler=schedule, loras=named_loras,
                    vae=runtime.model_name(files["vae"], roots) if files.get("vae") else None)
            paths = ENGINE.execute(graph, preview_path=preview_path,
                                   preview_offset=index * steps, log=log)
            result.extend(paths)
            if on_output:
                on_output(paths, base_seed + index)
        return result


def _tile_origins(length, tile, overlap):
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, tile - overlap))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def upscale(source, *, checkpoint, vae=None, controlnet=None, cn_scale=0.6,
            width, height, denoise=0.35, steps=24, cfg=6.0, tile=1024,
            overlap=128, prompt="", negative="", preview_path=None, log=None):
    """SDXL tile refinement with shared cached loaders and cosine blending."""
    import numpy as np
    if not 0 < float(denoise) <= 1:
        raise EngineError("SDXL refinement strength must be above zero and at most one.")
    if not Path(checkpoint).is_file():
        raise EngineError("Install or select a local SDXL checkpoint first.")
    width, height = int(width), int(height)
    if max(width, height) > 8192:
        ratio = 8192 / max(width, height)
        width, height = [max(8, round(v * ratio / 8) * 8) for v in (width, height)]
    tile = max(512, round(int(tile) / 8) * 8)
    overlap = max(32, min(int(overlap), tile // 2))
    with Image.open(source) as im:
        base = ImageOps.exif_transpose(im).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    originals = np.asarray(base, dtype=np.float32)
    accumulated = np.zeros_like(originals)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    paths = tuple(Path(p) for p in (checkpoint, vae, controlnet) if p)
    roots = runtime.model_roots(paths)
    named = {"checkpoint": runtime.model_name(checkpoint, roots),
             "vae": runtime.model_name(vae, roots) if vae else None,
             "controlnet": runtime.model_name(controlnet, roots) if controlnet else None}
    xs, ys = _tile_origins(width, tile, overlap), _tile_origins(height, tile, overlap)
    total = len(xs) * len(ys)
    with ENGINE.session():
        ENGINE.ensure(roots, log=log)
        counter = 0
        for y in ys:
            for x in xs:
                ENGINE.check()
                counter += 1
                x2, y2 = min(x + tile, width), min(y + tile, height)
                crop = base.crop((x, y, x2, y2))
                stream = io.BytesIO()
                crop.save(stream, format="PNG")
                temporary = settings.TMP_DIR / ("comfy-tile-" + uuid.uuid4().hex + ".png")
                temporary.write_bytes(stream.getvalue())
                refined = []
                try:
                    uploaded = ENGINE.upload(temporary)
                    graph = workflows.sdxl(**named, prompt=prompt or
                        "highly detailed, sharp focus, high quality", negative=negative or
                        "blurry, jpeg artifacts, lowres, oversharpened, deformed",
                        width=crop.width, height=crop.height, seed=counter - 1,
                        steps=steps, cfg=cfg, strength=denoise, cn_scale=cn_scale,
                        init=uploaded, prefix="turboslop/" + uuid.uuid4().hex)
                    if log:
                        log(f"SDXL tile {counter}/{total} ({x},{y}) · Comfy")
                    refined = ENGINE.execute(graph, preview_path=preview_path, log=log)
                    with Image.open(refined[0]) as out:
                        pixels = np.asarray(out.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS), dtype=np.float32)
                finally:
                    temporary.unlink(missing_ok=True)
                    for path in refined:
                        path.unlink(missing_ok=True)  # Tiles do not become gallery images.
                fade = max(1, min(overlap, crop.width // 2, crop.height // 2))
                ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))
                fx, fy = np.ones(crop.width, np.float32), np.ones(crop.height, np.float32)
                if x > 0:
                    fx[:fade] = ramp
                if x2 < width:
                    fx[-fade:] = ramp[::-1]
                if y > 0:
                    fy[:fade] = ramp
                if y2 < height:
                    fy[-fade:] = ramp[::-1]
                mask = (fy[:, None] * fx[None, :])[:, :, None]
                accumulated[y:y2, x:x2] += pixels * mask
                weights[y:y2, x:x2] += mask
                if preview_path:
                    view = np.where(weights > 0, accumulated / np.maximum(weights, 1e-6), originals)
                    image = Image.fromarray(view.clip(0, 255).astype("uint8"))
                    image.thumbnail((1280, 1280))
                    dest = Path(preview_path)
                    partial = dest.with_name(dest.stem + ".part.png")
                    image.save(partial)
                    partial.replace(dest)
        ENGINE.check()
        final = Image.fromarray((accumulated / np.maximum(weights, 1e-6)).clip(0, 255).astype("uint8"))
        destination = settings.OUTPUT_DIR / ("sdxl-comfy-" + uuid.uuid4().hex + ".png")
        final.save(destination)
        atomic_write_text(destination.with_suffix(".txt"), (prompt or "SDXL upscale") +
                          f"\nModel: SDXL (sdxl-base)\nSteps: {steps}, CFG: {cfg}, Denoise: {denoise}, Seed: 0\nBackend: Comfy local")
        return destination
