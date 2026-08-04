"""Outils PyTorch installés à la demande (profondeur, détourage, upscale créatif).

Installation et exécution en sous-process (Python embarqué), pour ne pas
verrouiller les DLL de torch dans le process Gradio.
"""
from __future__ import annotations

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
ENHANCE_MODEL_DIR = TOOLS_DIR / "enhance" / "model"
UPSCALE_DIR = TOOLS_DIR / "upscale"
UPSCALE_CKPT_DIR = UPSCALE_DIR / "checkpoints"   # checkpoints SDXL perso (.safetensors)
# SeedVR2 : code d'inférence cloné + poids. On appelle son « inference_cli.py »
# en sous-process (chemin officiellement documenté « sans ComfyUI »).
SEEDVR2_DIR = TOOLS_DIR / "seedvr2"
SEEDVR2_REPO_DIR = SEEDVR2_DIR / "repo"
# Sans ComfyUI, le CLI amont résout son dossier de modèles en RELATIF —
# « ./models/SEEDVR2 » depuis le répertoire courant — et il bâtit la liste des
# valeurs acceptées par --dit_model À PARTIR DE CE DOSSIER, au moment où argparse
# se construit. Nos poids doivent donc y être, et le process doit tourner avec
# SEEDVR2_DIR comme répertoire courant : c'est ce couple qui rend le 1.4B
# sélectionnable.
SEEDVR2_MODEL_DIR = SEEDVR2_DIR / "models" / "SEEDVR2"
_SEEDVR2_LEGACY_MODEL_DIR = SEEDVR2_DIR / "models"   # emplacement des 1res installs

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


def list_upscale_checkpoints() -> list[tuple[str, str]]:
    """Checkpoints SDXL disponibles pour l'upscale créatif : (libellé, chemin).
    Le modèle de base + tout .safetensors déposé dans tools_repo/upscale/checkpoints/."""
    out: list[tuple[str, str]] = []
    base = UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    if base.is_file():
        out.append(("SDXL Base 1.0 (par défaut)", str(base)))
    if UPSCALE_CKPT_DIR.is_dir():
        for p in sorted(UPSCALE_CKPT_DIR.glob("*.safetensors")):
            out.append((p.stem, str(p)))
    return out


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


def install_seedvr2_stream():
    yield from _install_stream("seedvr2")


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
              err_msg: str, gpu_index: int | None = None,
              cwd: Path | None = None) -> None:
    global _CANCELLED
    env = settings.child_env(gpu_index)
    if log:
        log("$ " + " ".join(cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            cwd=str(cwd or settings.ROOT), env=env,
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


ENHANCE_STYLES = ("generic", "krea2")


# --------------------------------------------------------------------------- #
#  SeedVR2 — upscale de RESTAURATION en UN pas (image fixe, sans prompt)
# --------------------------------------------------------------------------- #
def seedvr2_is_installed() -> bool:
    """Code d'inférence + au moins un modèle de DIFFUSION (le VAE seul ne
    suffit pas : le CLI le télécharge tout seul dans le même dossier)."""
    return ((SEEDVR2_REPO_DIR / "inference_cli.py").is_file()
            and bool(seedvr2_models()))


def seedvr2_has_1_4b() -> bool:
    """Le 1.4B n'est sélectionnable que si la config d'architecture a bien été
    posée ET la sélection étendue en amont (cf. setup_tools._seedvr2_enable_1_4b)."""
    cfg = SEEDVR2_REPO_DIR / "configs_1_4b" / "main.yaml"
    sel = SEEDVR2_REPO_DIR / "src" / "core" / "model_configuration.py"
    if not (cfg.is_file() and sel.is_file()):
        return False
    try:
        return "configs_1_4b" in sel.read_text(encoding="utf-8")
    except OSError:
        return False


# Le VAE est téléchargé par le CLI dans LE MÊME dossier que les modèles de
# diffusion (« ema_vae_fp16.safetensors »). Sans filtre il apparaissait dans la
# liste des modèles — et en tête, par ordre alphabétique : c'est lui qui partait
# en --dit_model, que le CLI refusait à juste titre.
_SEEDVR2_NOT_DIT = ("vae",)


def seedvr2_models() -> list[str]:
    """Modèles de DIFFUSION présents sur le disque.

    Le CLI amont accepte, en plus de son catalogue 3B/7B, tout fichier trouvé
    dans SON dossier de modèles — c'est ce qui rend le 1.4B utilisable sans
    toucher à son registre. On ne liste donc QUE ce dossier-là (un fichier
    ailleurs serait proposé puis refusé), et on en écarte le VAE, qui y cohabite.
    """
    if not SEEDVR2_MODEL_DIR.is_dir():
        return []
    # Pas de repli « à défaut, tout lister » : ça réintroduirait le VAE dans le
    # menu. Une liste vide est la réponse honnête — il manque un modèle.
    return [p.name for p in sorted(SEEDVR2_MODEL_DIR.iterdir())
            if p.suffix.lower() in (".safetensors", ".gguf")
            and not any(k in p.name.lower() for k in _SEEDVR2_NOT_DIT)]


def seedvr2_default_model() -> str | None:
    """Modèle présélectionné : le 1.4B, celui que l'installeur met en place."""
    ms = seedvr2_models()
    if not ms:
        return None
    return next((m for m in ms
                 if "1.4b" in m.lower() or "6l" in m.lower()), ms[0])


def seedvr2_upscale(image, resolution: int = 1440, model: str | None = None,
                    seed: int = 42, color_correction: str = "lab",
                    tiled: bool = False, tile_size: int = 512,
                    log: Callable[[str], None] | None = None) -> Path:
    """Restaure/agrandit une image fixe (1 pas de diffusion, aucun prompt).

    `resolution` = côté COURT visé en pixels (sémantique du CLI amont) ; le
    rapport d'aspect est conservé. Recommandé : ×2 à ×4 de l'original."""
    if not seedvr2_is_installed():
        raise ToolError("SeedVR2 n'est pas installé "
                        "(bouton « Installer » de l'onglet Restauration).")
    models = seedvr2_models()
    if not models:
        raise ToolError("Aucun poids SeedVR2 trouvé. Relancez l'installation.")
    model = model or seedvr2_default_model()
    if any(k in (model or "").lower() for k in _SEEDVR2_NOT_DIT):
        raise ToolError(
            f"« {model} » est le VAE, pas un modèle de diffusion. Cliquez "
            "« ↻ Rafraîchir » puis choisissez un modèle « seedvr2_… ».")

    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    src = _to_src(image, "seedvr2_src")
    out_dir = settings.TMP_DIR / f"seedvr2_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SEEDVR2_REPO_DIR / "inference_cli.py"), str(src),
           "--output", str(out_dir / "seedvr2.png"),
           "--output_format", "png",
           "--dit_model", model,
           "--model_dir", str(SEEDVR2_MODEL_DIR),
           "--resolution", str(int(resolution)),
           "--seed", str(int(seed)),
           "--color_correction", color_correction or "lab",
           "--batch_size", "1"]
    if tiled:
        # Le VAE s'auto-attentionne sur toute la tuile : le coût est en O(n²)
        # sur (tuile/8)². 512 est le réglage sûr ; monter dessus n'apporte rien
        # et fait exploser la mémoire. Descendre à 256 si ça déborde encore.
        cmd += ["--vae_decode_tiled", "--vae_decode_tile_size", str(int(tile_size)),
                "--vae_encode_tiled", "--vae_encode_tile_size", str(int(tile_size))]
    # Le GPU est choisi via CUDA_VISIBLE_DEVICES (_run_tool) : on ne passe PAS
    # --cuda_device en plus, les deux se marcheraient dessus.
    # On écoute la sortie pour distinguer les deux familles d'échec : manque de
    # VRAM (réglages à baisser) et incompatibilité de paquets (réinstallation).
    # Sans ça, l'utilisateur reçoit le même message dans les deux cas.
    seen = {"oom": False, "imp": False}
    _OOM = ("outofmemoryerror", "allocation on device", "out of memory",
            "cuda error: out of memory")
    _IMP = ("importerror", "modulenotfounderror",
            "failed to import", "cannot import name")

    def _sniff(line: str) -> None:
        low = (line or "").lower()
        if any(k in low for k in _OOM):
            seen["oom"] = True
        if any(k in low for k in _IMP):
            seen["imp"] = True
        if log:
            log(line)

    # cwd = SEEDVR2_DIR : le CLI résout « ./models/SEEDVR2 » depuis là, et c'est
    # ce dossier qu'il scanne pour décider des valeurs acceptées par --dit_model.
    try:
        _run_tool(cmd, _sniff, "L'agrandissement SeedVR2 a échoué "
                               "(voir le journal).",
                  gpu_index=_gen_gpu_index(), cwd=SEEDVR2_DIR)
    except ToolError:
        if seen["oom"]:
            raise ToolError(
                "Mémoire GPU insuffisante pendant l'agrandissement.\n"
                "Dans l'ordre, et sans rien réinstaller :\n"
                f"  1. cochez « VAE par tuiles » (actuellement "
                f"{'coché' if tiled else 'DÉCOCHÉ'}) ;\n"
                f"  2. baissez « Côté court visé » — vous êtes à "
                f"{int(resolution)} px ;\n"
                "  3. descendez la taille de tuile à 256 ;\n"
                "  4. fermez ce qui occupe la carte (une autre génération, "
                "un jeu, un navigateur lourd).") from None
        if seen["imp"]:
            raise ToolError(
                "SeedVR2 n'a pas pu démarrer : erreur à l'IMPORT (voir le "
                "journal). Une version de paquet a changé sous lui. Lancez "
                "maintenance.bat — il nomme le paquet — puis relancez "
                "« Installer SeedVR2 ».") from None
        raise
    return _collect(out_dir, "seedvr2", stamp)


def enhance_prompt_variants(prompt: str, style: str = "generic",
                            level: str = "medium", variants: int = 1,
                            style_constraint: str = "",
                            log: Callable[[str], None] | None = None) -> list[str]:
    """Améliore un prompt brut via un petit LLM instruct (transformers).

    `style` choisit le system prompt : "krea2" (guide Krea) ou "generic".
    `variants` demande plusieurs PROPOSITIONS en un seul chargement du modèle.
    `style_constraint` transmet le préréglage de style actif, pour que le LLM
    écrive AVEC lui. S'exécute en sous-process (chargé puis déchargé : aucun
    conflit VRAM avec sd.cpp)."""
    if not enhance_is_installed():
        raise ToolError("L'améliorateur de prompt n'est pas installé "
                        "(accordéon « ✨ Améliorer » de l'onglet de génération).")
    if not (prompt or "").strip():
        raise ToolError("Saisissez d'abord un prompt à améliorer.")
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = settings.TMP_DIR / f"enhance_{stamp}.json"
    runner = settings.ROOT / "scripts" / "tools" / "run_enhance.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(ENHANCE_MODEL_DIR),
           "--prompt", prompt, "--output", str(out_file),
           "--style", style if style in ENHANCE_STYLES else "generic",
           "--level", level if level in ("light", "medium", "strong") else "medium",
           "--variants", str(max(1, min(8, int(variants or 1))))]
    # Préréglage de style actif : le LLM doit écrire AVEC lui, pas contre lui.
    if (style_constraint or "").strip():
        cmd += ["--style-constraint", style_constraint.strip()]
    # Améliorateur = TEXTE → GPU secondaire dédié au texte (ex. 1080 Ti).
    _run_tool(cmd, log, "L'amélioration du prompt a échoué (voir le journal).",
              gpu_index=_text_gpu_index())
    try:
        raw = out_file.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    out: list[str] = []
    if raw:
        try:
            data = json.loads(raw)
            out = [str(x).strip() for x in data if str(x).strip()]
        except (ValueError, TypeError):
            out = [raw]           # repli : ancien format texte brut
    if not out:
        raise ToolError("L'améliorateur n'a renvoyé aucun texte (voir le journal).")
    return out


def enhance_prompt(prompt: str, style: str = "generic", level: str = "medium",
                   log: Callable[[str], None] | None = None) -> str:
    """Une seule proposition (compatibilité)."""
    return enhance_prompt_variants(prompt, style=style, level=level,
                                   variants=1, log=log)[0]


def ultimate_upscale(image, scale: float = 2.0, prompt: str = "",
                     negative: str = "",
                     denoise: float = 0.35, steps: int = 24, cfg: float = 6.0,
                     tile: int = 1024, overlap: int = 128,
                     use_controlnet: bool = False, cn_scale: float = 0.6,
                     base_model: str | None = None, integrated_vae: bool = False,
                     esrgan_model: str | None = None,
                     preview_path: Path | None = None,
                     log: Callable[[str], None] | None = None) -> Path:
    """Upscale créatif tuilé « Ultimate SD Upscale » (SDXL img2img résident).

    Pré-agrandit puis raffine tuile par tuile à faible débruitage. Options :
    `base_model` (checkpoint SDXL ; défaut = base 1.0), `integrated_vae` (utiliser
    la VAE du checkpoint au lieu de la fp16-fix externe), `esrgan_model` (pré-
    agrandir avec un ESRGAN GGUF plutôt qu'en Lanczos), `use_controlnet`."""
    if not upscale_is_installed():
        raise ToolError("L'upscale créatif SDXL n'est pas installé "
                        "(bouton « Installer » de l'onglet Toolkit → Upscale).")
    base = Path(base_model) if base_model else UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    if not base.is_file():
        raise ToolError(f"Checkpoint SDXL introuvable : {base}")
    from PIL import Image as _PILImage
    src = _to_src(image, "usdu")
    with _PILImage.open(src) as _im:
        ow, oh = _im.size
    tw = max(8, int(round(ow * scale / 8)) * 8)
    th = max(8, int(round(oh * scale / 8)) * 8)

    # Pré-agrandissement ESRGAN optionnel (sd.cpp) : base plus nette que Lanczos.
    #
    # Le nombre de passes est calculé, pas laissé à 1 : un modèle ×2 utilisé une
    # seule fois pour une cible ×4 sort SOUS la cible, et le runner doit alors
    # ré-agrandir en Lanczos — soit exactement l'interpolation floue qu'on
    # voulait éviter en prenant un ESRGAN. Dépasser la cible, en revanche, est
    # sans danger : la réduction qui suit est nette (suréchantillonnage).
    inp = src
    if esrgan_model:
        from .. import registry
        from . import generate as gen_engine
        factor = registry.upscaler_factor(esrgan_model)
        # On ajoute une passe tant qu'on est sous la cible ET que le résultat
        # intermédiaire reste sous le plafond de 8192 px du runner. Sans ce
        # second garde-fou, un ×4 visé en ×8 partirait à ×16, soit une image
        # intermédiaire de plusieurs centaines de mégapixels.
        repeats = 1
        while (factor ** repeats < float(scale) and repeats < 3
               and max(ow, oh) * factor ** (repeats + 1) <= 8192):
            repeats += 1
        reached = factor ** repeats
        try:
            if log:
                log(f"Pré-agrandissement ESRGAN « {esrgan_model} » (×{factor}"
                    + (f", {repeats} passes → ×{reached}" if repeats > 1
                       else "") + ")…")
                if reached < float(scale):
                    # Honnêteté : au-delà, c'est le runner qui complète en
                    # Lanczos, donc le rendu sera plus mou que promis.
                    log(f"[usdu] ×{reached} < cible ×{scale:g} : le reste sera "
                        "complété en Lanczos (rendu plus doux). Visez un "
                        "facteur plus bas pour un trait parfaitement net.")
            inp = gen_engine.upscale_image(src, esrgan_model, repeats=repeats,
                                           log=log)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"[usdu] ESRGAN échoué ({exc}) → repli Lanczos.")
            inp = src

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"usdu_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_ultimate_upscale.py"
    cmd = [sys.executable, str(runner),
           "--base-model", str(base),
           "--input", str(inp), "--output-dir", str(out_dir),
           "--width", str(tw), "--height", str(th),
           "--scale", str(float(scale)), "--denoise", str(float(denoise)),
           "--steps", str(int(steps)), "--cfg", str(float(cfg)),
           "--tile", str(int(tile)), "--overlap", str(int(overlap)),
           "--prompt", prompt or ""]
    # Négatif : vide = le défaut PHOTO du runner. Le préréglage dessin en
    # fournit un autre (anti-grain, anti-photoréalisme sur les aplats).
    if negative:
        cmd += ["--negative", negative]
    # VAE : externe fp16-fix (défaut) sauf si on veut celle intégrée au checkpoint.
    if not integrated_vae and (UPSCALE_DIR / "vae").is_dir():
        cmd += ["--vae", str(UPSCALE_DIR / "vae")]
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
