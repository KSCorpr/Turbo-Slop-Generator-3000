"""Outils PyTorch installés à la demande (profondeur, détourage, image → prompt).

Installation et exécution en sous-process (Python embarqué), pour ne pas
verrouiller les DLL de torch dans le process Gradio.
"""
from __future__ import annotations

from ..processes import close_worker

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import hardware, settings

TOOLS_DIR = settings.ROOT / "tools_repo"
DEPTH_MODEL_DIR = TOOLS_DIR / "depth" / "model"
BG_MODEL_DIR = TOOLS_DIR / "bg" / "model"
SAM_MODEL_DIR = TOOLS_DIR / "sam" / "model"
DESCRIBE_MODEL_DIR = TOOLS_DIR / "describe" / "model"

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Registre des sous-process d'outils en cours (pour le bouton « Annuler »).
_ACTIVE: set[subprocess.Popen] = set()
_LOCK = threading.Lock()
_CANCELLED = False


def cancel() -> str:
    """Termine le(s) sous-process d'outil en cours (upscale, etc.)."""
    global _CANCELLED
    with _LOCK:
        procs = list(_ACTIVE)
    if not procs:
        return "No task running."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Task cancelled."


class ToolError(RuntimeError):
    pass


def _model_present(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))


def depth_is_installed() -> bool:
    return _model_present(DEPTH_MODEL_DIR)


def bg_is_installed() -> bool:
    return _model_present(BG_MODEL_DIR)


def sam_is_installed(backend="sam") -> bool:
    if backend == "sam2":
        from .. import addons
        return addons.ready("sam2")
    if backend != "sam":
        raise ValueError("Choose SAM or SAM 2.1.")
    return _model_present(SAM_MODEL_DIR)


def describe_is_installed() -> bool:
    return _model_present(DESCRIBE_MODEL_DIR)


def _install_stream(tool: str):
    """Installe un outil (depth|bg) en streamant le journal (pour l'UI)."""
    setup = settings.ROOT / "scripts" / "setup_tools.py"
    cmd = [sys.executable, str(setup), tool]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            env=settings.child_env(),
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf.append("")
    buf.append("✅ Installation complete." if code == 0
               else f"❌ Failed (code {code}). See the log above.")
    yield "\n".join(buf[-500:])


def install_depth_stream():
    yield from _install_stream("depth")


def install_bg_stream():
    yield from _install_stream("bg")


def install_sam_stream():
    yield from _install_stream("sam")


def install_describe_stream():
    yield from _install_stream("describe")


def _gen_gpu_index() -> int | None:
    """GPU de GÉNÉRATION d'images (Flux/Krea, depth/bg/SAM).
    Jamais le GPU secondaire dédié au texte."""
    prefs = settings.load_prefs()
    choice = settings.generation_gpu_index(prefs)
    if choice is not None:
        return choice
    prof = hardware.auto_profile()
    return prof.gpu.index if prof.gpu else None


def _text_gpu_index() -> int | None:
    """GPU pour Image → prompt. GPU secondaire si défini
    (ex. 1080 Ti), sinon le GPU de génération."""
    prefs = settings.load_prefs()
    if prefs.get("text_gpu_index") is not None:
        return prefs["text_gpu_index"]
    return _gen_gpu_index()


def _to_src(image: Image.Image | str | Path, prefix: str) -> Path:
    settings.ensure_dirs()
    if isinstance(image, (str, Path)):
        return Path(image)
    src = settings.TMP_DIR / f"{prefix}_src_{int(time.time()*1000)}.png"
    image.save(src)
    return src


def _run_tool(cmd: list[str], log: Callable[[str], None] | None,
              err_msg: str, gpu_index: int | None = None,
              cwd: Path | None = None, env: dict | None = None) -> None:
    global _CANCELLED
    run_env = env if env is not None else settings.child_env(gpu_index)
    if log:
        log("$ " + " ".join(cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            cwd=str(cwd or settings.ROOT), env=run_env,
                            encoding="utf-8", errors="replace")
    with _LOCK:
        _ACTIVE.add(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log:
                log(line.rstrip("\n"))
        code = proc.wait()
    finally:
        try:
            close_worker(proc)
        finally:
            with _LOCK:
                _ACTIVE.discard(proc)
    if _CANCELLED:
        raise ToolError("Cancelled by the user.")
    if code != 0:
        raise ToolError(err_msg)


def _collect(out_dir: Path, final_prefix: str, stamp: str) -> Path:
    produced = sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in _IMG_EXT)
    if not produced:
        raise ToolError("No image produced (see the log).")
    final = settings.OUTPUT_DIR / f"{final_prefix}-{stamp}.png"
    Image.open(produced[0]).save(final)  # conserve l'alpha (RGBA) si présent
    return final


def depth_map(image, log: Callable[[str], None] | None = None) -> Path:
    if not depth_is_installed():
        raise ToolError("The depth tool is not installed (“Install” button in "
                        "the Toolkit).")
    src = _to_src(image, "depth")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"depth_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_depth.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(DEPTH_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "Depth estimation failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "depth", stamp)


# --------------------------------------------------------------------------- #
def modern_upscale(image, model_name: str,
                   log: Callable[[str], None] | None = None) -> Path:
    """Agrandissement par un modèle MODERNE (DAT, SPAN, PLKSR…) via spandrel.

    Aucune diffusion, aucun prompt, aucune graine : la même image donne
    toujours exactement le même résultat. C'est ce qui le distingue des
    upscales génératifs, dont « l'effet peinture » n'est pas un défaut de
    réglage mais leur fonctionnement même.

    Le facteur vient du MODÈLE, pas d'une cible : on ne redimensionne jamais
    après coup pour tomber juste. Un ×4 réduit ensuite à ×3 est propre (c'est
    du suréchantillonnage) ; un ×2 étiré jusqu'à ×3 est exactement
    l'interpolation qu'on cherche à fuir.
    """
    from .. import registry
    weights = registry.upscaler_path(model_name)
    if weights is None:
        raise ToolError(f"Upscaler not found: “{model_name}”.")
    src = _to_src(image, "upscale")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"upscale_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"upscale-{stamp}.png"
    runner = settings.ROOT / "scripts" / "tools" / "run_spandrel.py"
    cmd = [sys.executable, str(runner), "--model", str(weights),
           "--input", str(src), "--output", str(out)]
    _run_tool(cmd, log, "The upscale failed (see the log).",
              gpu_index=_gen_gpu_index())
    if not out.is_file():
        raise ToolError("The upscale produced no image (see the log).")
    final = settings.OUTPUT_DIR / out.name
    out.replace(final)
    return final


def bg_remove(image, log: Callable[[str], None] | None = None) -> Path:
    if not bg_is_installed():
        raise ToolError("The background removal tool is not installed "
                        "(“Install” button in the Toolkit).")
    src = _to_src(image, "nobg")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"nobg_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_rembg.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(BG_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "Background removal failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "nobg", stamp)


def sam_segment(image, x: int, y: int,
                log: Callable[[str], None] | None = None, *, backend="sam") -> tuple[Path, Path | None]:
    """Segment Anything au point (x, y). Renvoie (découpage PNG transparent,
    aperçu overlay) — l'overlay montre la zone sélectionnée en surbrillance."""
    if not sam_is_installed(backend):
        raise ToolError("Segment Anything is not installed (“Install” button "
                        "in the Toolkit).")
    src = _to_src(image, "sam")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"sam_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = settings.TMP_DIR / f"sam_overlay_{stamp}.png"
    runner = settings.ROOT / "scripts" / "tools" / "run_sam.py"
    from .. import addons
    py = addons.python("sam2") if backend == "sam2" else sys.executable
    model_dir = addons.root("sam2") / "model" if backend == "sam2" else SAM_MODEL_DIR
    cmd = [str(py), str(runner), "--model-dir", str(model_dir), "--backend", backend,
           "--input", str(src), "--output-dir", str(out_dir),
           "--x", str(int(x)), "--y", str(int(y)),
           "--overlay-path", str(overlay)]
    _run_tool(cmd, log, "Segmentation failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "sam", stamp), (overlay if overlay.exists() else None)


DESCRIBE_MODES = ("full", "style", "plain")


def image_to_prompt(image, mode: str = "full", variants: int = 1,
                    log: Callable[[str], None] | None = None) -> list[str]:
    """Lit une image et en écrit un PROMPT (pas une légende).

    Tourne en sous-process, comme l'améliorateur : le modèle est chargé puis
    déchargé, donc rien ne reste en VRAM pendant la génération sd.cpp. Et comme
    l'améliorateur, c'est du TEXTE : il part sur le GPU dédié au texte quand il
    y en a un (la 1080 Ti, par exemple), ce qui laisse la carte de génération
    tranquille.
    """
    if not describe_is_installed():
        raise ToolError("The “Image → prompt” module is not installed "
                        "(install button in its own tab).")
    src = _to_src(image, "describe")
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = settings.TMP_DIR / f"describe_{stamp}.json"
    runner = settings.ROOT / "scripts" / "tools" / "run_describe.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(DESCRIBE_MODEL_DIR),
           "--image", str(src), "--output", str(out_file),
           "--mode", mode if mode in DESCRIBE_MODES else "full",
           "--variants", str(max(1, min(4, int(variants or 1))))]
    _run_tool(cmd, log, "Reading the image failed (see the log).",
              gpu_index=_text_gpu_index())
    try:
        raw = out_file.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    out: list[str] = []
    if raw:
        try:
            out = [str(x).strip() for x in json.loads(raw) if str(x).strip()]
        except (ValueError, TypeError):
            out = [raw]
    if not out:
        raise ToolError("The model returned no text (see the log).")
    return out
