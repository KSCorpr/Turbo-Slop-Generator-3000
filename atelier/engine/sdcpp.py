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


def supports_video(sd_cli: Path | None) -> bool:
    """Ce binaire sait-il générer de la vidéo (« -M vid_gen ») ?

    Détecté au lieu d'être supposé : les binaires officiels récents l'ont, mais
    un build plus ancien (ou maison, taillé pour l'image) ne l'a pas, et sd-cli
    échouerait alors sur une option inconnue avec un message illisible."""
    return "--video-frames" in supported_options(sd_cli)


def missing_options(sd_cli: Path | None, wanted: "list[str]") -> list[str]:
    """Parmi `wanted`, les options que CE binaire ne connaît pas.

    Chaque famille de modèles vidéo arrive avec ses propres options (LTX a
    « --embeddings-connectors », MiniMax-H3 « --ref-video »…), et un même
    binaire peut connaître l'une sans l'autre selon sa date. Le catalogue
    déclare donc ce dont chaque modèle a besoin, et on le vérifie ici plutôt
    que de deviner à partir d'un numéro de version."""
    opts = supported_options(sd_cli)
    if not opts:
        return list(wanted or [])
    return [o for o in (wanted or []) if o not in opts]


def mask_flag(sd_cli: Path | None) -> str | None:
    """Nom de l'option de masque supportée, ou None si le binaire n'en a pas.

    Le masque de sd.cpp est appliqué PENDANT l'échantillonnage : blanc = zone à
    (re)générer, noir = zone conservée. Sans masque, sd.cpp en fabrique un tout
    blanc — c'est-à-dire « repeins tout », le comportement img2img normal."""
    opts = supported_options(sd_cli)
    if not opts:
        return None
    return next((c for c in _MASK_CANDIDATES if c in opts), None)


def _flag_args(flags: Mapping[str, bool]) -> list[str]:
    mapping = {
        "diffusion_fa": "--diffusion-fa",
        "offload_to_cpu": "--offload-to-cpu",
        "vae_tiling": "--vae-tiling",
        "clip_on_cpu": "--clip-on-cpu",
        "vae_on_cpu": "--vae-on-cpu",
    }
    return [opt for key, opt in mapping.items() if flags.get(key)]


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
    cmd += _flag_args(req.flags)
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


# --------------------------------------------------------------------------- #
#  Vidéo — LTX-2.3 (sd.cpp « -M vid_gen », docs/ltx2.md)
# --------------------------------------------------------------------------- #
#  Contraintes du modèle, appliquées ICI plutôt que dans l'UI pour qu'aucun
#  appelant ne puisse les contourner. Elles DIFFÈRENT d'une famille à l'autre —
#  LTX-2.3 veut des paquets de 8 images + 1, MiniMax-H3 des paquets de 17 + 5 —
#  donc elles sont paramétrées, pas codées en dur :
#   · largeur/hauteur alignées sur 32 px. sd.cpp divise ENTIÈREMENT par le
#     facteur du VAE : demander 720 px rend silencieusement 704 px ;
#   · nombre d'images sur la grille « step·k + base ». sd.cpp réaligne tout seul,
#     mais autant afficher à l'utilisateur ce qu'il obtiendra vraiment.
VAE_SCALE = 32
FRAME_STEP = 8      # LTX-2.3 : 8k+1
FRAME_BASE = 1


def snap_size(value: int, align: int = VAE_SCALE) -> int:
    """Aligne une dimension VERS LE HAUT sur `align` (minimum `align`).

    Vers le haut, et non au plus proche : c'est ce que documente MiniMax-H3, et
    ça ne perd jamais de contenu. Sur des formats déjà alignés (ceux proposés
    par l'onglet), les deux reviennent au même."""
    align = max(1, int(align))
    v = -(-int(value) // align) * align
    return max(align, v)


def snap_frames(value: int, step: int = FRAME_STEP,
                base: int = FRAME_BASE) -> int:
    """Aligne un nombre d'images VERS LE HAUT sur la grille « step·k + base »."""
    step = max(1, int(step))
    base = max(1, int(base))
    n = max(base, int(value))
    k = -(-(n - base) // step)
    return k * step + base


@dataclass
class VidRequest:
    diffusion_model: Path
    vae: Path                       # VAE vidéo
    audio_vae: Path                 # VAE audio -> bande-son dans le .webm
    text_encoder: Path              # --llm (Gemma-3-12B ou Qwen3-VL-32B)
    # --embeddings-connectors : propre à LTXAV. MiniMax-H3 n'en a pas, d'où
    # l'option facultative plutôt qu'un champ obligatoire.
    connectors: Path | None = None
    llm_vision: Path | None = None  # tour vision séparée, si le dépôt la sépare
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    schedule: str = ""
    width: int = 1280
    height: int = 704
    frames: int = 33
    fps: int = 24
    seed: int = -1
    # Conditionnement : image de départ (i2v) et image de fin (flf2v).
    init_image: Path | None = None
    end_image: Path | None = None
    # Images de RÉFÉRENCE (-r) : MiniMax-H3 Ref2VA, pour garder un personnage
    # d'un plan à l'autre. Exclusif avec init/end (contrainte du modèle).
    ref_images: list[Path] = field(default_factory=list)
    # Grille du modèle (voir snap_size / snap_frames).
    size_align: int = VAE_SCALE
    frame_step: int = FRAME_STEP
    frame_base: int = FRAME_BASE
    # Générateur aléatoire imposé par le modèle ("cpu" pour MiniMax-H3). Vide =
    # défaut du moteur.
    rng: str = ""
    # Reprise haute résolution par upscaler LATENT ×2 (LTX spatial upscaler).
    hires_upscaler: Path | None = None
    hires_steps: int = 4
    flags: dict[str, bool] = field(default_factory=dict)
    gpu_index: int | None = None
    encoder_gpu_index: int | None = None
    auto_fit: bool = False
    split_mode: str = ""


def build_vid_cmd(sd_cli: Path, req: VidRequest, output: Path) -> list[str]:
    """Commande LTX-2.3 (« -M vid_gen »). La sortie doit être un .webm : c'est
    le seul conteneur que sd.cpp sait muxer AVEC l'audio généré."""
    _require(req.diffusion_model, req.vae, req.audio_vae, req.text_encoder,
             req.connectors, req.llm_vision, req.init_image, req.end_image,
             req.hires_upscaler, *req.ref_images)

    cmd: list[str] = [
        str(sd_cli), "-M", "vid_gen",
        "--diffusion-model", str(req.diffusion_model),
        "--vae", str(req.vae),
        "--audio-vae", str(req.audio_vae),
        "--llm", str(req.text_encoder),
    ]
    if req.connectors:
        cmd += ["--embeddings-connectors", str(req.connectors)]
    if req.llm_vision:
        cmd += ["--llm_vision", str(req.llm_vision)]
    cmd += ["-p", req.prompt]
    # Comme en image : à CFG 1.0 le modèle est distillé et ignore le négatif.
    if req.negative and req.cfg_scale > 1.0:
        cmd += ["-n", req.negative]
    cmd += [
        "--cfg-scale", f"{req.cfg_scale}",
        "--steps", f"{req.steps}",
        "--sampling-method", req.sampler,
        "-W", f"{snap_size(req.width, req.size_align)}",
        "-H", f"{snap_size(req.height, req.size_align)}",
        "--video-frames",
        f"{snap_frames(req.frames, req.frame_step, req.frame_base)}",
        "--fps", f"{int(req.fps)}",
        "-s", f"{req.seed}",
    ]
    if req.schedule:
        cmd += ["--scheduler", req.schedule]
    if req.rng:
        cmd += ["--rng", req.rng]
    # Image de départ : « -i » en image → vidéo comme en début → fin (c'est la
    # même option --init-img) ; « --end-img » n'a de sens qu'avec les deux.
    # Les images de RÉFÉRENCE sont un chemin exclusif (contrainte MiniMax-H3 :
    # Ref2VA ne se combine ni avec --init-img ni avec --end-img).
    if req.ref_images:
        for r in req.ref_images:
            cmd += ["-r", str(r)]
    elif req.init_image:
        cmd += ["-i", str(req.init_image)]
        if req.end_image:
            cmd += ["--end-img", str(req.end_image)]
    if req.hires_upscaler:
        # L'upscaler latent est désigné par son NOM sans extension, cherché dans
        # le dossier passé à --hires-upscalers-dir (doc ltx2.md).
        cmd += ["--hires",
                "--hires-upscalers-dir", str(Path(req.hires_upscaler).parent),
                "--hires-upscaler", Path(req.hires_upscaler).stem,
                "--hires-steps", f"{int(req.hires_steps)}"]
    cmd += _flag_args(req.flags)
    if req.auto_fit:
        cmd += ["--auto-fit"]
        if req.split_mode:
            cmd += ["--split-mode", req.split_mode]
    elif (req.encoder_gpu_index is not None
            and req.encoder_gpu_index != req.gpu_index):
        g = req.gpu_index if req.gpu_index is not None else 0
        cmd += ["--backend",
                f"diffusion=cuda{g},vae=cuda{g},te=cuda{req.encoder_gpu_index}"]
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
