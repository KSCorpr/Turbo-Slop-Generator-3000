"""Inventaire de tout ce qui s'installe sur le disque : moteurs, modèles,
add-ons du Toolkit, LoRA, sorties, temporaires.

Sert au gestionnaire « 🧹 Gestion & nettoyage » : lister ce qui occupe de la
place, avec sa taille, et permettre de le supprimer sélectivement. Aucun
chemin hors du dossier du projet n'est jamais listé ni supprimé.
"""
from __future__ import annotations

import shutil
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import addons, registry, settings
from .engine import tools, trellis


@dataclass
class Item:
    key: str
    label: str
    category: str
    paths: list[Path] = field(default_factory=list)
    note: str = ""
    protected: bool = False        # supprimable, mais on prévient (données perso)

    @property
    def size(self) -> int:
        return sum(path_size(p) for p in self.paths)

    @property
    def installed(self) -> bool:
        return any(p.exists() for p in self.paths)


def path_size(p: Path, *, seen=None) -> int:
    """Taille d'un fichier ou d'un dossier (récursif), 0 s'il n'existe pas."""
    try:
        if p.is_file():
            st=p.stat()
            key=(st.st_dev,st.st_ino)
            if seen is not None:
                if key in seen:
                    return 0
                seen.add(key)
            return st.st_size
        if p.is_dir():
            total = 0
            pending = [p]
            while pending:
                folder = pending.pop()
                try:
                    with os.scandir(folder) as entries:
                        for entry in entries:
                            try:
                                if entry.is_dir(follow_symlinks=False):
                                    # Windows junctions can lead back to a parent.
                                    if not getattr(os.path, "isjunction", lambda _: False)(entry.path):
                                        pending.append(entry.path)
                                elif entry.is_file(follow_symlinks=False):
                                    st=entry.stat(follow_symlinks=False)
                                    key=(st.st_dev,st.st_ino)
                                    if seen is None or key not in seen:
                                        total += st.st_size
                                        if seen is not None:
                                            seen.add(key)
                            except OSError:
                                continue
                except OSError:
                    continue
            return total
    except OSError:
        pass
    return 0


def human(n: int) -> str:
    if n <= 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            # Les octets s'écrivent en entier : « 512.0 B » n'a pas de sens.
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _engine_sdcpp_paths() -> list[Path]:
    """Binaire + DLL de sd.cpp dans bin/ (hors sous-dossier trellis)."""
    if not settings.BIN_DIR.is_dir():
        return []
    out = []
    for p in settings.BIN_DIR.iterdir():
        if p.name.lower() == "trellis":
            continue
        out.append(p)
    return out


def _model_dirs_for(model: registry.BaseModel) -> list[Path]:
    """Dossiers de dépôt HF utilisés par un modèle du catalogue."""
    dirs: list[Path] = []
    for repo in sorted(registry.model_repos(model)):
        if not repo:
            continue
        d = settings.model_repo_dir(repo)
        if d not in dirs:
            dirs.append(d)
    return dirs


def items(prefs: dict | None = None) -> list[Item]:
    """Tout ce qui peut occuper de la place, groupé par catégorie."""
    prefs = prefs or settings.load_prefs()
    out: list[Item] = []

    # --- Moteurs -----------------------------------------------------------
    out.append(Item(
        "engine_sdcpp", "stable-diffusion.cpp engine (sd-cli + DLLs)", "Engines",
        _engine_sdcpp_paths(),
        note="Reinstallable: install.bat / update.bat."))
    out.append(Item(
        "engine_trellis", "trellis.cpp engine (image → 3D)", "Engines",
        [trellis.TRELLIS_BIN_DIR],
        note="Reinstallable: the “Image → 3D” tab."))
    for name, label in (("seedvr2", "SeedVR2 classic"), ("seedvr2-convrot", "SeedVR2 ConvRot")):
        folder = addons.root(name)
        out.append(Item(f"engine_{name}", f"{label} runtime", "Engines",
                        [folder / part for part in (".venv", "source", "comfy", "installed.json", "ffmpeg")],
                        note="Reinstallable: SeedVR2 → Install / repair."))

    # --- Modèles du catalogue ---------------------------------------------
    for name, label in (("svg", "SVG tracing / LayerTracer"), ("hivg", "HiVG"), ("omnisvg", "OmniSVG"),
                        ("sam2", "SAM 2.1"), ("qwen-layered", "Qwen-Image-Layered")):
        folder = addons.root(name)
        out.append(Item(f"engine_{name}", f"{label} runtime", "Engines",
                        [folder / part for part in (".venv", "source", "installed.json", "constraints.txt")],
                        note=("Reinstallable: Toolkit → Layers." if name in {"sam2", "qwen-layered"}
                              else "Reinstallable: Tools & 3D → SVG & layers.")))
    for name, label, parts in (("sam2", "SAM 2.1 small weights (~184 MB)", ("model",)),
                               ("qwen-layered", "Qwen-Image-Layered Q4 weights (~30 GB)", ("model", "quantized"))):
        out.append(Item(name+"_models", label, "Models", [addons.root(name) / part for part in parts],
                        note="Reinstallable: Toolkit → Layers → Install / repair."))
    out.append(Item("hivg_models", "HiVG-3B-Base weights (~8.14 GB)", "Models",
                    [addons.root("hivg") / "model"],
                    note="Reinstallable: SVG & layers → Install / repair HiVG."))
    out.append(Item("omnisvg_models", "OmniSVG 1.1 4B weights & processor (~7.71 GB)", "Models",
                    [addons.root("omnisvg") / part for part in ("model", "processor")],
                    note="Reinstallable: SVG & layers → Install / repair OmniSVG."))
    # Deux modèles peuvent vivre dans les MÊMES dossiers de dépôt (variantes
    # d'une même famille partageant encodeur ou VAE). Un dossier n'est listé
    # qu'une fois — sinon on compterait sa taille deux fois — mais l'élément
    # porte alors le nom de TOUS les modèles concernés : supprimer les efface
    # tous, et il faut que ce soit écrit avant le clic, pas découvert après.
    seen: dict[Path, int] = {}          # dossier -> index de l'élément porteur
    shared_names: dict[int, list[str]] = {}
    for m in registry.load_base_models(prefs):
        model_dirs = _model_dirs_for(m)
        new = [d for d in model_dirs if d not in seen]
        if new:
            idx = len(out)
            out.append(Item(f"model_{m.id}", f"Model — {m.name}", "Models",
                            new,
                            note="Downloadable again: Model catalog."))
            shared_names[idx] = [m.name]
            for d in new:
                seen[d] = idx
        elif model_dirs:
            # Entièrement contenu dans des dossiers déjà listés : on l'ajoute au
            # nom de l'élément porteur au lieu de le faire disparaître.
            idx = seen[model_dirs[0]]
            shared_names[idx].append(m.name)
    for idx, names in shared_names.items():
        if len(names) > 1:
            out[idx].label = "Models — " + " + ".join(names)
            out[idx].note = ("⚠️ Folders SHARED by these models: deleting "
                             "them removes them all. Downloadable again: "
                             "Model catalog.")

    # --- Autres modèles ----------------------------------------------------
    out.append(Item("upscalers", "Upscalers ESRGAN (GGUF)", "Models",
                    [registry.upscalers_dir()],
                    note="Optional pre-upscalers used inside Toolkit → SDXL creative upscale."))
    out.append(Item("seedvr2_models", "SeedVR2 weights (shared classic / ConvRot)", "Models",
                    [addons.root("seedvr2") / "models"],
                    note="Downloads on first restoration. Both runtimes share this cache."))
    out.append(Item("trellis_models", "TRELLIS.2 / Pixal3D weights (GGUF)",
                    "Models", [trellis.MODELS_DIR],
                    note="Reinstallable: the “Image → 3D” tab."))

    # --- Add-ons du Toolkit (PyTorch) -------------------------------------
    out += [
        Item("tool_depth", "Toolkit — Depth", "Toolkit add-ons",
             [tools.DEPTH_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_bg", "Toolkit — Background removal", "Toolkit add-ons",
             [tools.BG_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_sam", "Toolkit — SAM cut-out", "Toolkit add-ons",
             [tools.SAM_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_enhance", "Prompt improver (LLM)", "Toolkit add-ons",
             [tools.ENHANCE_MODEL_DIR], note="Reinstallable in one click."),
        # `clip` et `describe` manquaient à cet inventaire : le modèle
        # image → prompt pèse 7,5 Go et n'apparaissait nulle part dans ce que
        # l'utilisateur peut voir ou libérer.
        Item("tool_clip", "Toolkit — CLIP labelling (layers)", "Toolkit add-ons",
             [tools.CLIP_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_describe", "Toolkit — Image → prompt (~7.5 GB)",
             "Toolkit add-ons", [tools.DESCRIBE_MODEL_DIR],
             note="Reinstallable in one click."),
        Item("tool_face", "Toolkit — Face restoration", "Toolkit add-ons",
             [tools.FACE_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_adetailer", "Toolkit — ADetailer detectors (YOLOv8)",
             "Toolkit add-ons", [tools.ADETAILER_DIR],
             note="Reinstallable in one click (~12 MB)."),
        Item("tool_upscale", "Toolkit — Creative SDXL upscale", "Toolkit add-ons",
             [tools.UPSCALE_DIR], note="Includes ControlNet and custom checkpoints."),
    ]

    # --- Données utilisateur (prudence) -----------------------------------
    out += [
        Item("loras", "Installed LoRAs", "Your data", [settings.LORA_DIR],
             note="⚠️ Your LoRA files (Civitai imports included).",
             protected=True),
        Item("custom", "Custom models (models/custom)", "Your data",
             [settings.CUSTOM_DIR],
             note="⚠️ Files you dropped in or converted by hand.", protected=True),
        Item("outputs", "Generated images & 3D (outputs/)", "Your data",
             [settings.OUTPUT_DIR],
             note="⚠️ Your creations.", protected=True),
        Item("tmp", "Temporary files (tmp/)", "Your data",
             [settings.TMP_DIR],
             note="Preview caches, working files and the interface's image "
                  "cache. Safe with the application CLOSED; while it is "
                  "running, images already on screen turn into broken icons "
                  "until the page is reloaded."),
    ]
    return out


def by_key(key: str, prefs: dict | None = None) -> Item | None:
    return next((i for i in items(prefs) if i.key == key), None)


def total_size(prefs: dict | None = None) -> int:
    seen=set()
    return sum(path_size(p,seen=seen) for i in items(prefs) for p in i.paths)


def reclaimable_size(path):
    """Bytes freed only when this deletion removes the last hard link."""
    path=Path(path)
    counts={}
    candidates=[path] if path.is_file() else []
    if path.is_dir():
        for folder,dirs,files in os.walk(path,followlinks=False):
            dirs[:]=[d for d in dirs if not Path(folder,d).is_symlink() and
                     not getattr(os.path,"isjunction",lambda _:False)(Path(folder,d))]
            candidates.extend(Path(folder,f) for f in files)
    for file in candidates:
        if file.is_symlink():
            continue
        try:
            st=file.stat()
            key=(st.st_dev,st.st_ino)
            size,links,n=counts.get(key,(st.st_size,st.st_nlink,0))
            counts[key]=(size,links,n+1)
        except OSError:
            continue
    return sum(size for size,links,n in counts.values() if n >= links)


def delete(keys: list[str], prefs: dict | None = None) -> tuple[list[str], int]:
    """Supprime les éléments désignés. Retourne (messages, octets libérés).

    Sécurité : on ne supprime QUE des chemins situés sous le dossier du projet.
    """
    msgs: list[str] = []
    freed = 0
    root = settings.ROOT.resolve()
    for key in keys or []:
        item = by_key(key, prefs)
        if item is None:
            msgs.append(f"• {key}: unknown, ignored.")
            continue
        if not item.installed:
            msgs.append(f"• {item.label}: nothing to delete.")
            continue
        size = item.size
        ok = 0
        for p in item.paths:
            try:
                rp = p.resolve()
            except OSError:
                continue
            if root not in rp.parents and rp != root:
                msgs.append(f"• {item.label}: path outside the project ignored ({p}).")
                continue
            if rp == root:
                continue
            try:
                reclaimable=reclaimable_size(rp)
                if rp.is_dir():
                    shutil.rmtree(rp, ignore_errors=True)
                elif rp.exists():
                    rp.unlink()
                if rp.exists():
                    msgs.append(f"• {item.label}: some files remain in {p.name}.")
                    continue
                freed += reclaimable
                ok += 1
            except OSError as exc:
                msgs.append(f"• {item.label}: failed on {p.name} ({exc}).")
        if ok:
            msgs.append(f"✓ {item.label} removed ({human(size)} logical file size; shared data may remain).")
    # Les dossiers de base doivent continuer d'exister.
    settings.ensure_dirs()
    return msgs, freed
