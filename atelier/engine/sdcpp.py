"""Exécuteur stable-diffusion.cpp : construction et lancement des commandes sd-cli."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .. import settings


class EngineError(RuntimeError):
    pass


# Registre des process sd-cli en cours, pour pouvoir les annuler.
_ACTIVE: set[subprocess.Popen] = set()
_LOCK = threading.Lock()
_CANCELLED = False


def cancel_active() -> str:
    """Termine tous les process sd-cli en cours (bouton « Annuler »)."""
    global _CANCELLED
    with _LOCK:
        procs = list(_ACTIVE)
    if not procs:
        return "Aucune génération en cours."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Génération annulée."


@dataclass
class GenRequest:
    diffusion_model: Path | None = None   # modèle de diffusion seul (GGUF flow)
    vae: Path | None = None
    model_path: Path | None = None        # checkpoint complet -> -m
    text_encoder: Path | None = None       # --llm (modèles à encodeur LLM)
    # --llm_vision : projecteur vision (mmproj) de l'encodeur — permet à
    # Qwen3-VL de « voir » l'image de référence (édition Krea 2 / Ostris Edit).
    llm_vision: Path | None = None
    t5xxl: Path | None = None              # --t5xxl (FLUX.1, etc.)
    clip_l: Path | None = None             # --clip_l
    uncond_model: Path | None = None
    extra_flags: list[str] = field(default_factory=list)
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    schedule: str = ""          # vide = laisser le scheduler par défaut du modèle
    flow_shift: float = 0.0     # 0 = auto (ne pas passer --flow-shift)
    width: int = 1024
    height: int = 1024
    seed: int = -1
    batch_count: int = 1
    init_image: Path | None = None     # img2img classique (-i + --strength)
    strength: float = 0.6
    # Masque d'inpainting (blanc = à régénérer, noir = à conserver). L'option
    # est DÉTECTÉE sur le binaire (mask_flag) : ignorée s'il ne la connaît pas.
    mask_image: Path | None = None
    # édition (-r / --ref-image, Flux.2) : un chemin OU une liste (multi-référence)
    ref_image: "Path | list[Path] | None" = None
    lora_dir: Path | None = None       # --lora-model-dir
    preview_path: Path | None = None   # aperçu temps réel (--preview proj)
    flags: dict[str, bool] = field(default_factory=dict)
    gpu_index: int | None = None
    # EXPÉRIMENTAL : place l'encodeur de texte sur un autre GPU (ex. 1080 Ti)
    # via --backend te=cudaX. None = encodeur sur le GPU principal / RAM.
    encoder_gpu_index: int | None = None
    # EXPÉRIMENTAL : répartition auto du modèle sur tous les GPU (--auto-fit).
    # Prioritaire sur le split d'encodeur (auto-fit remplace --backend).
    auto_fit: bool = False
    split_mode: str = ""               # --split-mode : "layer" | "row" (vide=défaut)
    # Accélération par cache (docs/caching.md) : réutilise les calculs entre pas.
    cache_mode: str = ""               # easycache | dbcache | taylorseer | …
    cache_option: str = ""             # ex. "threshold=0.2"


# --------------------------------------------------------------------------- #
#  Découverte des options réellement supportées par LE binaire installé.
#
#  L'orthographe des options bouge d'une version de sd.cpp à l'autre (et notre
#  build maison peut différer de l'officielle). Plutôt que de coder en dur un
#  nom d'option et d'échouer à l'exécution, on lit « sd-cli -h » une fois et on
#  s'adapte. Coût : un lancement de quelques millisecondes, mis en cache.
# --------------------------------------------------------------------------- #
_OPTS_CACHE: dict[tuple, frozenset] = {}


def supported_options(sd_cli: Path | None) -> frozenset:
    """Ensemble des options longues (« --xxx ») acceptées par le binaire."""
    if not sd_cli or not Path(sd_cli).is_file():
        return frozenset()
    p = Path(sd_cli)
    try:
        key = (str(p), p.stat().st_mtime_ns, p.stat().st_size)
    except OSError:
        return frozenset()
    if key in _OPTS_CACHE:
        return _OPTS_CACHE[key]
    import re
    text = ""
    try:
        # -h sort parfois sur stderr et/ou avec un code de retour non nul.
        r = subprocess.run([str(p), "-h"], capture_output=True, text=True,
                           timeout=30, errors="replace")
        text = (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:  # noqa: BLE001
        text = ""
    opts = frozenset(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", text))
    _OPTS_CACHE[key] = opts
    return opts


# Orthographes possibles de l'option « image de masque », par ordre de préférence.
_MASK_CANDIDATES = ("--mask-image", "--mask-img", "--mask")


def mask_flag(sd_cli: Path | None) -> str | None:
    """Nom de l'option de masque supportée, ou None si le binaire n'en a pas.

    Le masque de sd.cpp est appliqué PENDANT l'échantillonnage : blanc = zone à
    (re)générer, noir = zone conservée. Sans masque, sd.cpp en fabrique un tout
    blanc — c'est-à-dire « repeins tout », le comportement img2img normal."""
    opts = supported_options(sd_cli)
    if not opts:
        return None
    return next((c for c in _MASK_CANDIDATES if c in opts), None)


# Options qui n'existent que sur les binaires récents. On ne les envoie que si
# CE binaire les connaît : sinon sd-cli s'arrête sur un argument inconnu, et
# l'utilisateur récolte une erreur illisible au lieu d'une image.
_OPTIONAL_FLAGS = ("--diffusion-conv-direct", "--vae-conv-direct")


def _flag_args(flags: Mapping[str, bool],
               sd_cli: Path | None = None) -> list[str]:
    mapping = {
        "diffusion_fa": "--diffusion-fa",
        "offload_to_cpu": "--offload-to-cpu",
        "vae_tiling": "--vae-tiling",
        "clip_on_cpu": "--clip-on-cpu",
        "vae_on_cpu": "--vae-on-cpu",
        # Convolution directe : évite le gros tampon intermédiaire d'im2col.
        "conv_direct_diffusion": "--diffusion-conv-direct",
        "conv_direct_vae": "--vae-conv-direct",
    }
    wanted = [opt for key, opt in mapping.items() if flags.get(key)]
    risky = [o for o in wanted if o in _OPTIONAL_FLAGS]
    if risky:
        known = supported_options(sd_cli)
        if known:
            wanted = [o for o in wanted
                      if o not in _OPTIONAL_FLAGS or o in known]
        else:
            # Binaire non interrogeable : on s'abstient plutôt que de parier.
            wanted = [o for o in wanted if o not in _OPTIONAL_FLAGS]
    return wanted


def _require(*paths: Path | None) -> None:
    for p in paths:
        if p is not None and not Path(p).is_file():
            raise EngineError(
                f"Fichier requis introuvable : {p}\n"
                "Téléchargez le modèle depuis l'onglet Catalogue de modèles.")


def _ref_list(ref) -> list[Path]:
    """Normalise ref_image (chemin unique ou liste) en liste de chemins."""
    if ref is None:
        return []
    if isinstance(ref, (list, tuple)):
        return [Path(r) for r in ref if r]
    return [Path(ref)]


def build_gen_cmd(sd_cli: Path, req: GenRequest, output: Path) -> list[str]:
    refs = _ref_list(req.ref_image)
    _require(req.model_path, req.diffusion_model, req.vae, req.text_encoder,
             req.llm_vision, req.t5xxl, req.clip_l, req.uncond_model,
             req.init_image, req.mask_image, *refs)

    cmd: list[str] = [str(sd_cli), "--mode", "img_gen"]
    if req.model_path:
        # Checkpoint complet : CLIP + VAE inclus.
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

    cmd += list(req.extra_flags)
    cmd += ["-p", req.prompt]
    if req.negative and req.cfg_scale > 1.0:
        cmd += ["-n", req.negative]

    cmd += [
        "--cfg-scale", f"{req.cfg_scale}",
        "--steps", f"{req.steps}",
        "--sampling-method", req.sampler,
        "-W", f"{req.width}", "-H", f"{req.height}",
        "-s", f"{req.seed}", "-b", f"{req.batch_count}",
    ]
    if req.schedule:
        cmd += ["--scheduler", req.schedule]
    if req.flow_shift and req.flow_shift > 0:
        cmd += ["--flow-shift", f"{req.flow_shift}"]
    if req.init_image:
        cmd += ["-i", str(req.init_image), "--strength", f"{req.strength}"]
        # Masque : uniquement si CE binaire connaît l'option (sinon sd.cpp
        # planterait sur un argument inconnu — on dégrade en img2img simple).
        if req.mask_image:
            mf = mask_flag(sd_cli)
            if mf:
                cmd += [mf, str(req.mask_image)]
    for r in refs:
        # Édition d'image (Flux.2) : pilotée par le prompt, sans strength.
        # Plusieurs « -r » = édition multi-référence (combine les images).
        cmd += ["-r", str(r)]
    if req.lora_dir:
        cmd += ["--lora-model-dir", str(req.lora_dir)]

    if req.preview_path:
        cmd += ["--preview", "proj", "--preview-path", str(req.preview_path),
                "--preview-interval", "1"]
    # Accélération par cache (opt-in) : saute des calculs quasi identiques entre
    # pas. Nécessite un sd-cli récent (update-engine.bat si flag inconnu).
    if req.cache_mode:
        cmd += ["--cache-mode", req.cache_mode]
        if req.cache_option:
            cmd += ["--cache-option", req.cache_option]
    cmd += _flag_args(req.flags, sd_cli)
    # Multi-GPU. auto-fit répartit TOUT le modèle sur les GPU visibles (prioritaire,
    # remplace --backend) ; sinon, split d'encodeur : diffusion+VAE sur le GPU
    # principal, encodeur (te) sur l'autre. Ordre CUDA par bus PCI forcé via env.
    if req.auto_fit:
        cmd += ["--auto-fit"]
        if req.split_mode:
            cmd += ["--split-mode", req.split_mode]
    elif (req.encoder_gpu_index is not None
            and req.encoder_gpu_index != req.gpu_index):
        g = req.gpu_index if req.gpu_index is not None else 0
        e = req.encoder_gpu_index
        cmd += ["--backend",
                f"diffusion=cuda{g},vae=cuda{g},te=cuda{e}"]
    cmd += ["-o", str(output), "-v"]
    return cmd


def build_convert_cmd(sd_cli: Path, input_model: Path, output_model: Path,
                      qtype: str) -> list[str]:
    """Conversion/quantification d'un modèle en GGUF (sd.cpp --mode convert).

    Lit un checkpoint/safetensors/diffusion et le ré-écrit dans la quant `qtype`
    (ex. « q4_k », « q8_0 »). 100% CPU, aucune diffusion : c'est une simple
    transformation des poids. Voir docs/quantization_and_gguf.md."""
    return [str(sd_cli), "-M", "convert", "-m", str(input_model),
            "-o", str(output_model), "--type", qtype, "-v"]


def build_upscale_cmd(sd_cli: Path, init_image: Path, upscale_model: Path,
                      output: Path, repeats: int = 1,
                      offload: bool = True) -> list[str]:
    """Upscale ESRGAN natif sd.cpp (--mode upscale). Déterministe, 100% GPU.

    `repeats` applique le modèle plusieurs fois (ex. un modèle ×2 appliqué 2 fois
    = ×4). Aucun prompt/diffusion : c'est un réseau ESRGAN GGUF."""
    _require(upscale_model, init_image)
    cmd = [str(sd_cli), "--mode", "upscale", "-i", str(init_image),
           "--upscale-model", str(upscale_model)]
    if repeats and repeats > 1:
        cmd += ["--upscale-repeats", str(int(repeats))]
    if offload:
        cmd.append("--offload-to-cpu")
    cmd += ["-o", str(output), "-v"]
    return cmd


def run(cmd: list[str], log: Callable[[str], None] | None = None,
        gpu_index: int | None = None, all_gpus: bool = False) -> None:
    global _CANCELLED
    env = None
    if all_gpus:
        # Split multi-GPU (encodeur sur un 2e GPU) : tous les GPU visibles, et
        # ordre CUDA par bus PCI pour que cudaN corresponde à l'index nvidia-smi.
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif gpu_index is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    if log:
        log("$ " + " ".join(_q(c) for c in cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
                            encoding="utf-8", errors="replace")
    with _LOCK:
        _ACTIVE.add(proc)
    # On garde la fin de la sortie pour diagnostiquer les crashs de sd-cli
    # (l'assert GGML n'apparaît que quelques lignes avant la mort du process).
    tail: deque[str] = deque(maxlen=100)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.rstrip("\n")
            tail.append(s)
            if log:
                log(s)
        code = proc.wait()
    finally:
        with _LOCK:
            _ACTIVE.discard(proc)
    if _CANCELLED:
        raise EngineError("Interrompu par l'utilisateur.")
    if code != 0:
        raise EngineError(_diagnose_failure(code, cmd, tail))


def _diagnose_failure(code: int, cmd: list[str], tail: "deque[str]") -> str:
    """Transforme un code de sortie brut de sd-cli en message actionnable.

    Le cas le plus fréquent est l'assert GGML de reshape (`ggml_nelements(a) ==
    ne0*ne1*ne2`) : dimensions de tenseur incompatibles. Avec un LoRA, c'est
    quasi toujours un LoRA entraîné pour une autre base (ex. Krea 2 « full » vs
    Turbo) ; sinon c'est une résolution qui ne respecte pas la grille du modèle.
    """
    reshape_assert = any("GGML_ASSERT(ggml_nelements(a) ==" in ln for ln in tail)
    if reshape_assert:
        has_lora = "--lora-model-dir" in cmd or any("<lora:" in c for c in cmd)
        if has_lora:
            return (
                "❌ Crash pendant l'application d'un LoRA (formes de tenseurs "
                "incompatibles).\n"
                "Ce LoRA n'est pas compatible avec le modèle sélectionné — "
                "souvent un LoRA entraîné pour une autre base (ex. Krea 2 "
                "« full » alors que vous utilisez Krea 2 Turbo).\n"
                "→ Réessayez sans ce LoRA, ou utilisez le modèle pour lequel "
                "il a été entraîné.")
        return (
            "❌ sd-cli a planté sur un reshape de tenseur (dimensions "
            "incompatibles).\n"
            "Vérifiez que la résolution respecte la grille du modèle "
            "(multiple de 64 px pour Krea, 32 px pour Flux.2).\n"
            f"(code de sortie {code})")
    return f"sd-cli s'est terminé avec le code {code}."


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


def unique_output(prefix: str, ext: str = "png") -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ms = int(time.time() * 1000) % 1000
    return settings.OUTPUT_DIR / f"{prefix}-{stamp}-{ms:03d}.{ext}"


def collect_outputs(output: Path, batch_count: int) -> list[Path]:
    if batch_count <= 1 and output.is_file():
        return [output]
    found = sorted(output.parent.glob(f"{output.stem}_*{output.suffix}"))
    return found or ([output] if output.is_file() else [])
