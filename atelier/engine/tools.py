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
from . import release_resident_engine, sdcpp
from ..i18n import t

TOOLS_DIR = settings.ROOT / "tools_repo"
DEPTH_MODEL_DIR = TOOLS_DIR / "depth" / "model"
BG_MODEL_DIR = TOOLS_DIR / "bg" / "model"
SAM_MODEL_DIR = TOOLS_DIR / "sam" / "model"
CLIP_MODEL_DIR = TOOLS_DIR / "clip" / "model"
ENHANCE_MODEL_DIR = TOOLS_DIR / "enhance" / "model"
DESCRIBE_MODEL_DIR = TOOLS_DIR / "describe" / "model"
FACE_MODEL_DIR = TOOLS_DIR / "face" / "model"
UPSCALE_DIR = TOOLS_DIR / "upscale"
UPSCALE_CKPT_DIR = UPSCALE_DIR / "checkpoints"   # checkpoints SDXL perso (.safetensors)

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


def sam_is_installed() -> bool:
    return _model_present(SAM_MODEL_DIR)


def clip_is_installed() -> bool:
    return _model_present(CLIP_MODEL_DIR)


def describe_is_installed() -> bool:
    return _model_present(DESCRIBE_MODEL_DIR)


def enhance_is_installed() -> bool:
    return _model_present(ENHANCE_MODEL_DIR)


# Détection et segmentation : COMMUNES à tous les restaurateurs, et aussi
# indispensables qu'eux. Sans elles, le premier clic échouerait au milieu du
# traitement au lieu d'afficher le bouton « Installer ».
FACE_SHARED_FILES = ("detection_Resnet50_Final.pth", "parsing_parsenet.pth")

# Les restaurateurs proposés : (fichier, libellé, licence).
#
# La LICENCE est affichée avec le modèle, pas enterrée dans un avertissement de
# bas de page. Elle décide de ce qu'on a le droit de faire du résultat, et deux
# des trois sont libres de toute restriction commerciale — c'est une
# information de premier plan pour qui vend ses images, pas une note de bas de
# page juridique.
FACE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("GFPGANv1.4.pth",
     "GFPGAN v1.4 — preserves identity best",
     "Apache-2.0 · usage commercial libre"),
    ("RestoreFormer++.ckpt",
     "RestoreFormer++ — better on badly damaged photos",
     "Apache-2.0 · usage commercial libre"),
    ("codeformer.pth",
     "CodeFormer — adjustable fidelity dial",
     "S-Lab 1.0 · NON COMMERCIAL"),
)
FACE_MODEL_FILES = frozenset(f for f, _, _ in FACE_MODELS)
# Par défaut : le modèle libre qui garde le mieux le visage de la personne.
FACE_DEFAULT = "GFPGANv1.4.pth"
# Seul CodeFormer expose le « w » de fidélité ; pour les autres le curseur
# n'aurait aucun effet, et l'interface le dit plutôt que de le laisser croire.
FACE_FIDELITY_MODELS = frozenset({"codeformer.pth"})


def face_models_installed() -> list[tuple[str, str, str]]:
    """Restaurateurs réellement présents sur le disque."""
    return [m for m in FACE_MODELS if (FACE_MODEL_DIR / m[0]).is_file()]


def face_is_installed() -> bool:
    """Les briques communes ET au moins un restaurateur.

    « Au moins un » et pas « tous » : une installation interrompue après le
    premier modèle reste utilisable, et proposer de tout réinstaller pour un
    fichier manquant serait disproportionné.
    """
    shared = all((FACE_MODEL_DIR / n).is_file() for n in FACE_SHARED_FILES)
    return shared and bool(face_models_installed())


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
        out.append(("SDXL Base 1.0 (default)", str(base)))
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
    buf.append("✅ Installation complete." if code == 0
               else f"❌ Failed (code {code}). See the log above.")
    yield "\n".join(buf[-500:])


def install_depth_stream():
    yield from _install_stream("depth")


def install_bg_stream():
    yield from _install_stream("bg")


def install_sam_stream():
    yield from _install_stream("sam")


def install_clip_stream():
    yield from _install_stream("clip")


def install_enhance_stream():
    yield from _install_stream("enhance")


def install_describe_stream():
    yield from _install_stream("describe")


def install_face_stream():
    yield from _install_stream("face")


def install_upscale_stream():
    yield from _install_stream("upscale")


def _setup_script_stream(script: str, label: str):
    """Lance un script d'installation et rend son journal ligne par ligne.

    Ces installations durent des minutes : sans flux, le bouton reste muet et
    on ne sait pas distinguer « ça travaille » de « c'est planté ».
    """
    setup = settings.ROOT / "scripts" / script
    cmd = [sys.executable, str(setup)]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            env=settings.child_env(), encoding="utf-8",
                            errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf += ["", f"✅ {label} installation complete." if code == 0
            else f"❌ {label} failed (code {code}). See the log."]
    yield "\n".join(buf[-500:])


def install_adetailer_stream():
    """Convertit les détecteurs YOLOv8 dans un environnement jetable."""
    yield from _setup_script_stream("setup_adetailer.py", "ADetailer")


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
              cwd: Path | None = None, env: dict | None = None) -> None:
    global _CANCELLED
    # Un outil PyTorch qui démarre pendant que le moteur résident garde 8 Go de
    # modèle en VRAM, c'est un OOM. Le serveur rend la place ; il se rechargera
    # tout seul à la prochaine image.
    release_resident_engine("a Toolkit tool needs the GPU", log)
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
#  ADetailer : détecter une zone, la redessiner, la recoller
#
#  C'est le seul outil de la boîte qui répare les MAINS. Les restaurateurs de
#  visages (GFPGAN, RestoreFormer) sont entraînés sur des visages et ne savent
#  rien faire d'autre ; ici un détecteur YOLOv8 trouve la zone et c'est le
#  modèle de génération qui la redessine en inpainting.
#
#  Contrepartie assumée : ça REDESSINE. Sur un visage, « 🙂 Faces » reste plus
#  fidèle parce qu'il restaure au lieu d'inventer. Sur une main à six doigts,
#  il n'y a rien à restaurer.
# --------------------------------------------------------------------------- #
ADETAILER_DIR = settings.MODELS_DIR / "adetailer"

ADETAILER_MODELS: tuple[tuple[str, str], ...] = (
    ("face_yolov8n.safetensors", "Faces — fast"),
    ("face_yolov8s.safetensors", "Faces — more accurate"),
    ("hand_yolov8n.safetensors", "Hands — fast"),
    ("hand_yolov8s.safetensors", "Hands — more accurate"),
)


def adetailer_models_installed() -> list[str]:
    """Détecteurs convertis présents sur le disque."""
    if not ADETAILER_DIR.is_dir():
        return []
    return [name for name, _label in ADETAILER_MODELS
            if (ADETAILER_DIR / name).is_file()]


def adetailer_is_installed() -> bool:
    return bool(adetailer_models_installed())


def adetailer_reason() -> str:
    """Ce qui manque pour utiliser ADetailer, ou "" si tout est prêt.

    On NOMME la pièce absente au lieu de cacher l'onglet : « rien ne
    s'affiche » est le pire message d'erreur possible.
    """
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        return ("The sd-cli binary was not found. Run install.bat.")
    if not sdcpp.adetailer_supported(sd_cli):
        return ("Your sd.cpp engine does not know ADetailer yet (the "
                "`--ad-model` option). Run update-engine.bat, then come back.")
    if not adetailer_is_installed():
        return ("No detector installed — use the button below (~12 MB once "
                "converted).")
    return ""


def adetailer_repair(image, model_id: str, detector: str,
                     prompt: str = "", negative: str = "",
                     denoise: float = 0.4, steps: int = 0,
                     confidence: float = 0.3, padding: int = 32,
                     mask_blur: int = 4, only_largest: int = 0,
                     seed: int = -1,
                     log: Callable[[str], None] | None = None) -> Path:
    """Détecte puis redessine chaque zone trouvée, sur une image existante.

    Passe par `sd-cli -M adetailer`, donc par le MÊME moteur et le même modèle
    que la génération : pas de PyTorch, pas d'add-on, rien à installer sinon le
    détecteur. C'est aussi ce qui garantit que la zone redessinée est dans le
    style du modèle qui a fait l'image.
    """
    from .. import registry
    from . import generate as gen_engine

    reason = adetailer_reason()
    if reason:
        raise ToolError(reason)
    weights = ADETAILER_DIR / detector
    if not weights.is_file():
        raise ToolError(f"Detector not found: “{detector}”.")

    src = _to_src(image, "adetail")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = settings.OUTPUT_DIR / f"adetail-{stamp}.png"
    settings.ensure_dirs()
    # La commande de génération est construite par le pipeline habituel : c'est
    # lui qui sait résoudre les composants du modèle, la quantification et le
    # placement mémoire. On ne réimplémente rien de tout ça ici.
    cmd = gen_engine.adetailer_command(
        model_id, src, out, weights,
        prompt=prompt, negative=negative, denoise=denoise, steps=steps,
        seed=seed,
        extra={"confidence": confidence, "inpaint_padding": padding,
               "mask_blur": mask_blur, "mask_k_largest": only_largest})
    sdcpp.run(cmd, log=log, gpu_index=_gen_gpu_index())
    if not out.is_file():
        raise ToolError("ADetailer produced no image (see the log).")
    return out


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
    if not face_is_installed():
        # spandrel arrive avec la restauration de visages : c'est le même
        # paquet, donc rien de plus à installer une fois celle-ci en place.
        raise ToolError(
            "Modern upscalers need the “🙂 Faces” add-on installed — they "
            "share the same spandrel package. Install it once from the "
            "Toolkit, and no further download is needed.")
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


def face_restore(image, fidelity: float = 0.5, only_center: bool = False,
                 model: str = FACE_DEFAULT,
                 log: Callable[[str], None] | None = None) -> Path:
    """Restaure les visages d'une image (CodeFormer), sans toucher au reste.

    `fidelity` est le « w » de CodeFormer : 0 laisse le modèle reconstruire
    librement (visage très abîmé, mais le résultat peut ne plus être tout à
    fait la même personne), 1 colle au pixel d'origine. 0,5 est le réglage de
    référence. À passer APRÈS l'upscale : le visage est alors plus grand, donc
    mieux détecté et mieux recollé.
    """
    if not face_is_installed():
        raise ToolError("Face restoration is not installed (“Install” button "
                        "in the Toolkit).")
    if model not in FACE_MODEL_FILES:
        raise ToolError(f"Unknown restoration model: {model}")
    weights = FACE_MODEL_DIR / model
    if not weights.is_file():
        raise ToolError(f"The model “{model}” is not downloaded. "
                        "Run the installation again from the Toolkit.")
    src = _to_src(image, "face")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"face_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_face.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(FACE_MODEL_DIR),
           "--weights", weights.name,
           "--input", str(src), "--output-dir", str(out_dir),
           "--fidelity", f"{min(1.0, max(0.0, float(fidelity))):.2f}"]
    if only_center:
        cmd.append("--only-center")
    _run_tool(cmd, log, "Face restoration failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "face", stamp)


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
                log: Callable[[str], None] | None = None) -> tuple[Path, Path | None]:
    """Segment Anything au point (x, y). Renvoie (découpage PNG transparent,
    aperçu overlay) — l'overlay montre la zone sélectionnée en surbrillance."""
    if not sam_is_installed():
        raise ToolError("Segment Anything is not installed (“Install” button "
                        "in the Toolkit).")
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
    _run_tool(cmd, log, "Segmentation failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "sam", stamp), (overlay if overlay.exists() else None)


def _layers_to_files(src: Path, masks: list, names: list[str], stamp: str,
                     want_psd: bool, want_png: bool,
                     log: Callable[[str], None] | None = None) -> list[Path]:
    """Assemble des masques en PSD et/ou en PNG transparents séparés.

    Volontairement HORS du sous-process : l'écriture PSD n'a besoin ni de torch
    ni de transformers, et la faire ici évite d'imposer l'add-on à qui veut
    seulement rassembler des calques déjà découpés (mode manuel).
    """
    import numpy as np
    from . import masks as mask_utils
    from . import psd as psd_writer

    rgb = np.asarray(Image.open(src).convert("RGB"))
    h, w = rgb.shape[:2]
    out: list[Path] = []

    # Le fond, c'est l'image entière : même si une zone est détourée par-dessus,
    # on ne laisse jamais un trou dans le fichier final.
    layers = [(t("Background (full image)"),
               np.dstack([rgb, np.full((h, w), 255, "uint8")]))]
    for name, m in zip(names, masks):
        # Bord ADOUCI : un masque binaire collé tel quel donne ce contour en
        # marches d'escalier qui trahit le détourage automatique. Un pixel de
        # transition suffit à le faire disparaître, sans manger la zone.
        alpha = mask_utils.feather_alpha(m, radius=1)
        layers.append((name, np.dstack([rgb, alpha])))

    if want_psd:
        dest = settings.OUTPUT_DIR / f"layers-{stamp}.psd"
        psd_writer.write_psd(dest, rgb, layers)
        size = dest.stat().st_size / (1024 * 1024)
        if log:
            log(f"[layers] PSD written: {dest.name} ({size:.1f} MB, "
                f"{len(layers)} layers)")
        out.append(dest)
    if want_png:
        folder = settings.OUTPUT_DIR / f"layers-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, (name, rgba) in enumerate(layers):
            safe = "".join(c if (c.isalnum() or c in " -_") else "_"
                           for c in name).strip() or f"layer{i}"
            Image.fromarray(rgba, "RGBA").save(folder / f"{i:02d}_{safe}.png")
        if log:
            log(f"[layers] {len(layers)} transparent PNGs: {folder.name}/")
        out.append(folder)
    return out


def image_to_layers(image, points_per_side: int = 12, max_layers: int = 24,
                    min_area_pct: float = 0.4, want_psd: bool = True,
                    want_png: bool = False,
                    log: Callable[[str], None] | None = None) -> list[Path]:
    """AUTOMATIQUE : SAM balaie l'image, on en tire des calques triés en
    profondeur, puis on assemble le PSD.

    Ce que ça ne fait PAS, et qu'il faut savoir avant de cliquer : les calques
    sont des DÉCOUPES à plat. Déplacer un objet révèle un trou, parce que le
    fond derrière lui n'a jamais existé. C'est utile pour masquer, retoucher une
    zone ou exporter un élément — pas pour recomposer la scène.
    """
    if not sam_is_installed():
        raise ToolError("Segment Anything is not installed (“Install” button "
                        "in the Toolkit).")
    import numpy as np

    src = _to_src(image, "layers")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = settings.TMP_DIR / f"layers_{stamp}"
    runner = settings.ROOT / "scripts" / "tools" / "run_layers.py"
    cmd = [sys.executable, str(runner),
           "--sam-dir", str(SAM_MODEL_DIR),
           "--input", str(src), "--output-dir", str(work),
           "--points-per-side", str(int(points_per_side)),
           "--max-layers", str(int(max_layers)),
           "--min-area", f"{max(0.0005, float(min_area_pct) / 100.0):g}"]
    # La profondeur n'est PAS requise : sans elle on trie par surface, et le
    # runner le dit. L'exiger transformerait un add-on optionnel en dépendance.
    if depth_is_installed():
        cmd += ["--depth-dir", str(DEPTH_MODEL_DIR)]
    # CLIP : étiquetage sémantique, fusion des morceaux d'un même objet, et
    # rejet des zones qui ne ressemblent à rien. Facultatif comme la profondeur.
    if clip_is_installed():
        cmd += ["--clip-dir", str(CLIP_MODEL_DIR)]
    _run_tool(cmd, log, "Splitting into layers failed (see the log).",
              gpu_index=_gen_gpu_index())

    manifest = work / "layers.json"
    if not manifest.is_file():
        raise ToolError("No layer produced (see the log).")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    from . import masks as mask_utils
    rgb = np.asarray(Image.open(src).convert("RGB"))
    entries = data.get("masks", [])
    masks, names = [], []
    for i, entry in enumerate(entries):
        m = np.asarray(Image.open(work / entry["file"]).convert("L")) > 127
        masks.append(m)
        # Nom SÉMANTIQUE quand CLIP a pu étiqueter la zone (« véhicule »,
        # « ciel »), descriptif sinon (plan, position, couleur). Dans les deux
        # cas on garde plan et taille : ce sont eux qui situent le calque dans
        # la pile.
        base = mask_utils.describe(m, rgb, i, len(entries))
        lab = (entry.get("label") or "").strip()
        names.append(f"{lab} · {base}" if lab else base)
    if not masks:
        raise ToolError(
            "No usable region found. Try more probe points, or a lower "
            "minimum area.")
    return _layers_to_files(src, masks, names, stamp, want_psd, want_png, log)


def masks_to_layers(image, masks: list, names: list[str] | None = None,
                    want_psd: bool = True, want_png: bool = False,
                    log: Callable[[str], None] | None = None) -> list[Path]:
    """MANUEL : assemble des zones choisies à la main (clics SAM successifs).

    Les masques arrivent déjà segmentés : aucun modèle n'est chargé ici.
    """
    if not masks:
        raise ToolError("No region selected — click on the image first to "
                        "create layers.")
    src = _to_src(image, "layers")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = names or [f"Zone {i + 1}" for i in range(len(masks))]
    return _layers_to_files(src, masks, names, stamp, want_psd, want_png, log)


ENHANCE_STYLES = ("generic", "krea2", "xanax")


# --------------------------------------------------------------------------- #
#  Améliorateur de prompt (petit LLM instruct)
# --------------------------------------------------------------------------- #
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
        raise ToolError("The prompt improver is not installed (the “✨ "
                        "Improve” section of the generation tab).")
    if not (prompt or "").strip():
        raise ToolError("Enter a prompt to enhance first.")
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
    _run_tool(cmd, log, "Prompt improvement failed (see the log).",
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
        raise ToolError("The improver returned no text (see the log).")
    return out


# --------------------------------------------------------------------------- #
#  Image → prompt (modèle de vision-langage)
# --------------------------------------------------------------------------- #
# Trois intentions RÉELLEMENT différentes, pas trois réglages du même curseur :
#   full  -> refaire cette image ailleurs (sujet + style)
#   style -> appliquer CE rendu à un AUTRE sujet (style seul, sujet interdit)
#   plain -> savoir ce qu'il y a dedans, en français courant côté lecture
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
    agrandir avec un MODÈLE plutôt qu'en Lanczos), `use_controlnet`.

    `esrgan_model` garde son nom pour ne pas casser les appelants, mais il
    accepte les deux familles du catalogue : les ESRGAN GGUF lus par sd.cpp ET
    les agrandisseurs modernes (DAT, SPAN, PLKSR…) lus par spandrel. C'est le
    FICHIER qui décide du moteur, jamais l'appelant — la même règle que dans
    l'onglet « 🔼 Enlarge », et pour la même raison : l'utilisateur choisit un
    modèle, pas une implémentation."""
    if not upscale_is_installed():
        raise ToolError("The creative SDXL upscale is not installed "
                        "(“Install” button in Toolkit → Upscale).")
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
    # UNE SEULE passe, jamais deux. Enchaîner un ESRGAN sur sa propre sortie est
    # le meilleur moyen de fabriquer les artefacts et l'aliasing qu'on cherche à
    # éviter : la 2ᵉ passe prend les hautes fréquences INVENTÉES par la 1ʳᵉ pour
    # du détail réel et les ré-accentue, ce qui transforme un léger ringing en
    # marches d'escalier franches sur les diagonales. Si une passe ne suffit pas
    # à atteindre la cible, le runner complète en Lanczos : c'est plus doux,
    # mais c'est propre — et c'est précisément le rôle du raffinage SDXL qui
    # suit de redonner le détail. Une base molle se rattrape, une base crénelée
    # non : SDXL fige les créneaux au lieu de les corriger.
    #
    # Quand le facteur du modèle DÉPASSE la cible (un ×4 pour un ×2), la
    # réduction faite ensuite par le runner est un suréchantillonnage : c'est le
    # cas le plus propre possible, l'aliasing est moyenné à la baisse.
    inp = src
    if esrgan_model:
        from .. import registry
        from . import generate as gen_engine
        factor = registry.upscaler_factor(esrgan_model)
        engine = registry.upscaler_engine(esrgan_model)
        try:
            if log:
                kind = ("modern (spandrel)" if engine == "spandrel"
                        else "ESRGAN (sd.cpp)")
                log(f"Pre-enlargement “{esrgan_model}” — {kind}, ×{factor}, "
                    "one pass…")
                if factor > float(scale):
                    log(f"[usdu] ×{factor} for a ×{scale:g} target: the "
                        "downscale that follows acts as supersampling "
                        "(free anti-aliasing).")
                elif factor < float(scale):
                    # Honnêteté : le runner complète en Lanczos, donc la base
                    # sera plus douce — mais pas crénelée, et SDXL la reprend.
                    log(f"[usdu] ×{factor} < target ×{scale:g}: the rest is "
                        "filled in with Lanczos (a softer base, which the "
                        "SDXL refine pass re-details). For crisp line art "
                        "from the start, pick a ×"
                        f"{int(-(-float(scale) // 1))} model "
                        "or aim for a lower factor.")
            if engine == "spandrel":
                #  Les modernes ignorent `repeats` par construction : leur
                #  facteur vient du modèle, et enchaîner un réseau sur sa
                #  propre sortie est précisément ce qui fabrique les escaliers
                #  qu'on veut éviter. Une passe, toujours.
                inp = modern_upscale(src, esrgan_model, log=log)
            else:
                inp = gen_engine.upscale_image(src, esrgan_model, repeats=1,
                                               log=log)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"[usdu] pre-enlargement failed ({exc}) → Lanczos "
                    "fallback.")
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
    _run_tool(cmd, log, "The creative SDXL upscale failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "usdu", stamp)


