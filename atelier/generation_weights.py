"""Installed catalog variants for generation; no Hub requests or downloads."""
from __future__ import annotations

import fnmatch
from copy import deepcopy
from pathlib import Path

from . import registry, settings, weight_precision

AUTO = "__auto__"
LOCAL = "__local__:"
ROLES = ("diffusion", "text_encoder", "vae")


def _automatic_model(model_id, prefs):
    prefs = deepcopy(prefs)
    files = prefs.setdefault("model_files", {})
    selected = files.setdefault(model_id, {})
    for role in (*ROLES, "model"):
        selected.pop(role, None)
    return registry.get_base_model(model_id, prefs)


def _sources(comp):
    repo = comp.catalog_repo or comp.repo
    pattern = comp.catalog_template or comp.base_glob()
    pattern = pattern.replace("{quant}", "*").replace("{enc_quant}", "*")
    sources = [(repo, name, "") for name in comp.selectable] if comp.selectable else [(repo, pattern, "")]
    sources.extend((v.repo, v.template, v.label) for v in comp.alternatives)
    return sources


def _approved(filename, pattern, role):
    if not isinstance(filename, str):
        return False
    parts = Path(filename).parts
    return (not any(p.startswith(".") or p == ".." for p in parts)
            and Path(filename).suffix.lower() in {".gguf", ".safetensors", ".ckpt", ".pt", ".pth"}
            and (role != "text_encoder" or "mmproj" not in filename.lower())
            and fnmatch.fnmatchcase(filename, pattern))


def _local_files():
    return sorted(p for p in settings.CUSTOM_DIR.glob("*") if p.is_file()
                  and p.suffix.lower() in {".gguf", ".safetensors", ".sft", ".pth", ".ckpt"})


def installed(model):
    """Approved (filename, path, label) choices, scanning each local repo once."""
    repos, result = {}, {}
    for comp in model.components:
        role = "diffusion" if comp.role == "model" else comp.role
        if role not in ROLES:
            continue
        choices = {}
        for repo, pattern, label in _sources(comp):
            if repo not in repos:
                root = settings.model_repo_dir(repo)
                repos[repo] = [(p.relative_to(root).as_posix(), p)
                               for p in root.rglob("*") if p.is_file()]
            for filename, path in repos[repo]:
                if _approved(filename, pattern, role):
                    choices.setdefault(filename, (filename, path, label))
        result[role] = sorted(choices.values(), key=lambda row: row[0])
    return result


def menus(model_id):
    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    automatic = _automatic_model(model_id, prefs)
    choices = installed(model) if model else {}
    saved = (prefs.get("model_files") or {}).get(model_id) or {}
    custom = (prefs.get("generation_custom_files") or {}).get(model_id) or {}
    local_files = _local_files()
    profile = weight_precision.profile(prefs)
    menus = []
    for role in ROLES:
        comp = next((c for c in model.components
                     if c.role == role or (role == "diffusion" and c.role == "model")), None) if model else None
        auto_comp = next((c for c in automatic.components
                          if comp and c.role == comp.role), None) if automatic else None
        current = registry.resolve_component_path(auto_comp) if auto_comp else None
        default = f"Automatic — {current.name}" if current else "Automatic — download weights in Models"
        menu = [(default, AUTO)]
        for filename, path, label in choices.get(role, []):
            detail = weight_precision.describe(filename, profile, label=label,
                                               size=path.stat().st_size, role=role)
            menu.append((f"{detail} · {path.stat().st_size / 1e9:.2f} GB — {filename}", filename))
        if comp:
            menu.extend((f"Local / custom — {p.name}", LOCAL + p.name) for p in local_files)
        value = custom.get(comp.role) or saved.get(comp.role, AUTO) if comp else AUTO
        if value != AUTO and value not in {name for _, name in menu}:
            if (isinstance(value, str) and value.startswith(LOCAL)
                    or any(_approved(value, pattern, role) for _, pattern, _ in _sources(comp))):
                menu.append((f"Not installed — {value} (download in Models)", value))
            else:
                value = AUTO
        menus.append(dict(choices=menu, value=value, visible=comp is not None))
    return menus


def save_selection(model_id, role, value):
    save_selections(model_id, {role: value})


def save_selections(model_id, selections):
    """Share per-model choices with Models without touching other preferences."""
    prefs = settings.load_prefs()
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    choices = installed(model)
    files = deepcopy(prefs.get("model_files") or {})
    custom = deepcopy(prefs.get("generation_custom_files") or {})
    local_choices = {LOCAL + path.name for path in _local_files()}
    custom_selected = custom.setdefault(model_id, {})
    selected = files.setdefault(model_id, {})
    for role, value in selections.items():
        key = "model" if role == "diffusion" and any(c.role == "model" for c in model.components) else role
        if value in local_choices:
            custom_selected[key] = value
            continue
        custom_selected.pop(key, None)
        if value and value != AUTO and value not in {name for name, _, _ in choices.get(role, [])}:
            raise ValueError("This variant is not installed. Download it in Models, then refresh variants.")
        if value and value != AUTO:
            selected[key] = value
        else:
            selected.pop(key, None)
    if not selected:
        files.pop(model_id, None)
    if not custom_selected:
        custom.pop(model_id, None)
    prefs["model_files"] = files
    prefs["generation_custom_files"] = custom
    settings.save_prefs(prefs)


def resolve(model_id, selections):
    """Snapshot exact paths for this request, even if saved choices change later.

    Automatic keeps the usual quant-tolerant fallback. An explicit selection
    must still exist and never falls back to a different installed weight.
    Local / custom entries are resolved separately from approved catalog files.
    """
    prefs = settings.load_prefs()
    model = _automatic_model(model_id, prefs)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    choices = installed(model)
    local_choices = {LOCAL + path.name: path for path in _local_files()}
    result = {}
    for role, value in zip(ROLES, selections):
        if value and value != AUTO:
            path = local_choices.get(value) or next(
                (path for name, path, _ in choices.get(role, []) if name == value), None)
            if path is None or not path.is_file():
                raise ValueError(f"Selected {role.replace('_', ' ')} variant is not installed: {value}. "
                                 "Download it in Models, then refresh variants.")
        else:
            comp = next((c for c in model.components
                         if c.role == role or (role == "diffusion" and c.role == "model")), None)
            path = registry.resolve_component_path(comp) if comp else None
        result[role] = path
    return result
