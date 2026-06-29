"""Outils PyTorch installés à la demande (profondeur, détourage, upscale créatif).

Installation et exécution en sous-process (Python embarqué), pour ne pas
verrouiller les DLL de torch dans le process Gradio.
"""
from __future__ import annotations

import os
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
ENHANCE_MODEL_DIR = TOOLS_DIR / "enhance" / "model"
UPSCALE_DIR = TOOLS_DIR / "upscale"

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
        return "Aucune tâche en cours."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Tâche annulée."


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


def sam_is_installed() -> bool:
    return _model_present(SAM_MODEL_DIR)


def enhance_is_installed() -> bool:
    return _model_present(ENHANCE_MODEL_DIR)


def upscale_is_installed() -> bool:
    base = UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    vae = UPSCALE_DIR / "vae"
    return base.is_file() and vae.is_dir() and any(vae.glob("*.safetensors"))


def upscale_cn_is_installed() -> bool:
    """ControlNet Tile présent (optionnel — verrouille la structure)."""
    cn = UPSCALE_DIR / "controlnet"
    return cn.is_dir() and any(cn.glob("*.safetensors"))


def _install_stream(tool: str):
    """Installe un outil (depth|bg) en streamant le journal (pour l'UI)."""
    setup = settings.ROOT / "scripts" / "setup_tools.py"
    cmd = [sys.executable, str(setup), tool]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf.append("")
    buf.append("✅ Installation terminée." if code == 0
               else f"❌ Échec (code {code}). Voir le journal ci-dessus.")
    yield "\n".join(buf[-500:])


def install_depth_stream():
    yield from _install_stream("depth")


def install_bg_stream():
    yield from _install_stream("bg")


def install_sam_stream():
    yield from _install_stream("sam")


def install_enhance_stream():
    yield from _install_stream("enhance")


def install_upscale_stream():
    yield from _install_stream("upscale")


def _gen_gpu_index() -> int | None:
    """GPU de GÉNÉRATION d'images (Flux/Krea, upscale SDXL, depth/bg/SAM).
    Jamais le GPU secondaire dédié au texte."""
    prefs = settings.load_prefs()
    if prefs.get("gpu_index") is not None:
        return prefs["gpu_index"]
    prof = hardware.auto_profile()
    return prof.gpu.index if prof.gpu else None


def _text_gpu_index() -> int | None:
    """GPU pour le TEXTE (améliorateur de prompt). GPU secondaire si défini
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
              err_msg: str, gpu_index: int | None = None) -> None:
    global _CANCELLED
    env = dict(os.environ)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if log:
        log("$ " + " ".join(cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
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
        with _LOCK:
            _ACTIVE.discard(proc)
    if _CANCELLED:
        raise ToolError("Annulé par l'utilisateur.")
    if code != 0:
        raise ToolError(err_msg)


def _collect(out_dir: Path, final_prefix: str, stamp: str) -> Path:
    produced = sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in _IMG_EXT)
    if not produced:
        raise ToolError("Aucune image produite (voir le journal).")
    final = settings.OUTPUT_DIR / f"{final_prefix}-{stamp}.png"
    Image.open(produced[0]).save(final)  # conserve l'alpha (RGBA) si présent
    return final


def depth_map(image, log: Callable[[str], None] | None = None) -> Path:
    if not depth_is_installed():
        raise ToolError("L'outil de profondeur n'est pas installé "
                        "(bouton « Installer » du Toolkit).")
    src = _to_src(image, "depth")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"depth_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_depth.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(DEPTH_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "L'estimation de profondeur a échoué (voir le journal).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "depth", stamp)


def bg_remove(image, log: Callable[[str], None] | None = None) -> Path:
    if not bg_is_installed():
        raise ToolError("L'outil de suppression d'arrière-plan n'est pas installé "
                        "(bouton « Installer » du Toolkit).")
    src = _to_src(image, "nobg")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"nobg_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_rembg.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(BG_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "La suppression d'arrière-plan a échoué (voir le journal).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "nobg", stamp)


def sam_segment(image, x: int, y: int,
                log: Callable[[str], None] | None = None) -> tuple[Path, Path | None]:
    """Segment Anything au point (x, y). Renvoie (découpage PNG transparent,
    aperçu overlay) — l'overlay montre la zone sélectionnée en surbrillance."""
    if not sam_is_installed():
        raise ToolError("Segment Anything n'est pas installé "
                        "(bouton « Installer » du Toolkit).")
    src = _to_src(image, "sam")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"sam_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = settings.TMP_DIR / f"sam_overlay_{stamp}.png"
    runner = settings.ROOT / "scripts" / "tools" / "run_sam.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(SAM_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir),
           "--x", str(int(x)), "--y", str(int(y)),
           "--overlay-path", str(overlay)]
    _run_tool(cmd, log, "La segmentation a échoué (voir le journal).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "sam", stamp), (overlay if overlay.exists() else None)


def enhance_prompt(prompt: str, style: str = "generic",
                   log: Callable[[str], None] | None = None) -> str:
    """Améliore un prompt brut via un petit LLM instruct (transformers).

    `style` choisit le system prompt : "krea2" (guide Krea) ou "generic"
    (Flux/SD/MJ). Renvoie UNIQUEMENT le prompt enrichi en anglais. S'exécute en
    sous-process (chargé puis déchargé : aucun conflit VRAM avec sd.cpp)."""
    if not enhance_is_installed():
        raise ToolError("L'améliorateur de prompt n'est pas installé "
                        "(accordéon « ✨ Améliorer » de l'onglet de génération).")
    if not (prompt or "").strip():
        raise ToolError("Saisissez d'abord un prompt à améliorer.")
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = settings.TMP_DIR / f"enhance_{stamp}.txt"
    runner = settings.ROOT / "scripts" / "tools" / "run_enhance.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(ENHANCE_MODEL_DIR),
           "--prompt", prompt, "--output", str(out_file),
           "--style", style if style in ("generic", "krea2") else "generic"]
    # Améliorateur = TEXTE → GPU secondaire dédié au texte (ex. 1080 Ti).
    _run_tool(cmd, log, "L'amélioration du prompt a échoué (voir le journal).",
              gpu_index=_text_gpu_index())
    try:
        text = out_file.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if not text:
        raise ToolError("L'améliorateur n'a renvoyé aucun texte (voir le journal).")
    return text


def ultimate_upscale(image, scale: float = 2.0, prompt: str = "",
                     denoise: float = 0.35, steps: int = 24, cfg: float = 6.0,
                     tile: int = 1024, overlap: int = 128,
                     use_controlnet: bool = False, cn_scale: float = 0.6,
                     preview_path: Path | None = None,
                     log: Callable[[str], None] | None = None) -> Path:
    """Upscale créatif tuilé « Ultimate SD Upscale » (SDXL img2img résident).

    Pré-agrandit puis raffine tuile par tuile à faible débruitage (fondu par
    recouvrement). Modèle résident → tuiles rapides. 100% GPU (PyTorch).
    `use_controlnet` active ControlNet Tile (verrouille la structure)."""
    if not upscale_is_installed():
        raise ToolError("L'upscale créatif SDXL n'est pas installé "
                        "(bouton « Installer » de l'onglet Toolkit → Upscale).")
    src = _to_src(image, "usdu")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"usdu_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_ultimate_upscale.py"
    cmd = [sys.executable, str(runner),
           "--base-model", str(UPSCALE_DIR / "sd_xl_base_1.0.safetensors"),
           "--vae", str(UPSCALE_DIR / "vae"),
           "--input", str(src), "--output-dir", str(out_dir),
           "--scale", str(float(scale)), "--denoise", str(float(denoise)),
           "--steps", str(int(steps)), "--cfg", str(float(cfg)),
           "--tile", str(int(tile)), "--overlap", str(int(overlap)),
           "--prompt", prompt or ""]
    if use_controlnet and upscale_cn_is_installed():
        cmd += ["--controlnet", str(UPSCALE_DIR / "controlnet"),
                "--cn-scale", str(float(cn_scale))]
    if preview_path:
        cmd += ["--preview-path", str(preview_path)]
    # VRAM serrée (< 12 Go) → offload CPU du modèle pour éviter l'OOM.
    prof = hardware.auto_profile(settings.load_prefs().get("gpu_index"))
    if prof.gpu and prof.gpu.vram_gb < 12:
        cmd.append("--low-vram")
    # Upscale SDXL = génération d'IMAGES → GPU de génération (jamais le secondaire).
    _run_tool(cmd, log, "L'upscale créatif SDXL a échoué (voir le journal).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "usdu", stamp)
