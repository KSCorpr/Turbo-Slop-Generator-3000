"""Inventaire de tout ce qui s'installe sur le disque : moteurs, modèles,
add-ons du Toolkit, LoRA, sorties, temporaires.

Sert au gestionnaire « 🧹 Gestion & nettoyage » : lister ce qui occupe de la
place, avec sa taille, et permettre de le supprimer sélectivement. Aucun
chemin hors du dossier du projet n'est jamais listé ni supprimé.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import registry, settings
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


def path_size(p: Path) -> int:
    """Taille d'un fichier ou d'un dossier (récursif), 0 s'il n'existe pas."""
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        pass
    return 0


def human(n: int) -> str:
    if n <= 0:
        return "—"
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or unit == "To":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} To"


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
    for comp in model.components:
        repo = getattr(comp, "repo", None)
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
        "engine_sdcpp", "Moteur stable-diffusion.cpp (sd-cli + DLL)", "Moteurs",
        _engine_sdcpp_paths(),
        note="Réinstallable : install.bat / update-engine.bat."))
    out.append(Item(
        "engine_trellis", "Moteur trellis.cpp (image → 3D)", "Moteurs",
        [trellis.TRELLIS_BIN_DIR],
        note="Réinstallable : onglet « Image → 3D »."))

    # --- Modèles du catalogue ---------------------------------------------
    seen: set[Path] = set()
    for m in registry.load_base_models(prefs):
        dirs = [d for d in _model_dirs_for(m) if d not in seen]
        seen.update(dirs)
        if dirs:
            out.append(Item(f"model_{m.id}", f"Modèle — {m.name}", "Modèles",
                            dirs,
                            note="Re-téléchargeable : Catalogue de modèles."))

    # --- Autres modèles ----------------------------------------------------
    pid_dirs: list[Path] = []
    for comp in registry.pid_components():
        d = settings.model_repo_dir(comp.repo)
        if d not in pid_dirs and d not in seen:
            pid_dirs.append(d)
    if pid_dirs:
        out.append(Item("pid", "PiD — décodeur/upscaler ×4 (NVIDIA)", "Modèles",
                        pid_dirs, note="Réinstallable : onglet de génération."))
    out.append(Item("upscalers", "Upscalers ESRGAN (GGUF)", "Modèles",
                    [registry.upscalers_dir()],
                    note="Réinstallable : Toolkit → Agrandir."))
    out.append(Item("trellis_models", "Modèles trellis 3D (GGUF, ~10 Go)",
                    "Modèles", [trellis.MODELS_DIR],
                    note="Réinstallable : onglet « Image → 3D »."))

    # --- Add-ons du Toolkit (PyTorch) -------------------------------------
    out += [
        Item("tool_depth", "Toolkit — Profondeur", "Add-ons Toolkit",
             [tools.DEPTH_MODEL_DIR], note="Réinstallable en 1 clic."),
        Item("tool_bg", "Toolkit — Sans arrière-plan", "Add-ons Toolkit",
             [tools.BG_MODEL_DIR], note="Réinstallable en 1 clic."),
        Item("tool_sam", "Toolkit — Détourage SAM", "Add-ons Toolkit",
             [tools.SAM_MODEL_DIR], note="Réinstallable en 1 clic."),
        Item("tool_enhance", "Améliorateur de prompt (LLM)", "Add-ons Toolkit",
             [tools.ENHANCE_MODEL_DIR], note="Réinstallable en 1 clic."),
        Item("tool_upscale", "Toolkit — Upscale créatif SDXL", "Add-ons Toolkit",
             [tools.UPSCALE_DIR], note="Inclut ControlNet et checkpoints perso."),
        Item("tool_seedvr2", "Toolkit — Restauration SeedVR2", "Add-ons Toolkit",
             [tools.SEEDVR2_DIR],
             note="Code d'inférence + poids 1.4B. Réinstallable en 1 clic."),
    ]

    # --- Données utilisateur (prudence) -----------------------------------
    out += [
        Item("loras", "LoRA installés", "Vos données", [settings.LORA_DIR],
             note="⚠️ Vos fichiers LoRA (dont imports Civitai).",
             protected=True),
        Item("custom", "Modèles perso (models/custom)", "Vos données",
             [settings.CUSTOM_DIR],
             note="⚠️ Fichiers déposés/convertis à la main.", protected=True),
        Item("outputs", "Images & 3D générés (outputs/)", "Vos données",
             [settings.OUTPUT_DIR],
             note="⚠️ Vos créations.", protected=True),
        Item("tmp", "Fichiers temporaires (tmp/)", "Vos données",
             [settings.TMP_DIR],
             note="Sans risque : caches d'aperçu et fichiers de travail."),
    ]
    return out


def by_key(key: str, prefs: dict | None = None) -> Item | None:
    return next((i for i in items(prefs) if i.key == key), None)


def total_size(prefs: dict | None = None) -> int:
    return sum(i.size for i in items(prefs))


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
            msgs.append(f"• {key} : inconnu, ignoré.")
            continue
        if not item.installed:
            msgs.append(f"• {item.label} : rien à supprimer.")
            continue
        size = item.size
        ok = 0
        for p in item.paths:
            try:
                rp = p.resolve()
            except OSError:
                continue
            if root not in rp.parents and rp != root:
                msgs.append(f"• {item.label} : chemin hors projet ignoré ({p}).")
                continue
            if rp == root:
                continue
            try:
                if rp.is_dir():
                    shutil.rmtree(rp, ignore_errors=True)
                elif rp.exists():
                    rp.unlink()
                ok += 1
            except OSError as exc:
                msgs.append(f"• {item.label} : échec sur {p.name} ({exc}).")
        if ok:
            freed += size
            msgs.append(f"✓ {item.label} — {human(size)} libéré(s).")
    # Les dossiers de base doivent continuer d'exister.
    settings.ensure_dirs()
    return msgs, freed
