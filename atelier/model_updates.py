"""Read-only checks of installed HF artifacts; never replace weights on update."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re

from . import addons, model_store, settings
from .fileio import atomic_write_text

SKIP = model_store.SKIP | {".huggingface"}
EXTENSIONS = model_store.WEIGHTS | {".json", ".txt", ".py", ".model", ".jinja"}


def _allowed(path):
    path = Path(path)
    roots = [settings.MODELS_DIR, settings.ROOT / "tools_repo", settings.LORA_DIR]
    return not path.is_symlink() and any(
        path.resolve().is_relative_to(root.resolve())
        and not any(part in SKIP for part in path.resolve().relative_to(root.resolve()).parts)
        for root in roots)


def _read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def record(path, *, repo, filename, revision, commit, etag):
    """Persist provenance after a verified transfer/reuse, without hashing again."""
    from filelock import FileLock
    path = Path(path)
    if not path.is_file() or not _allowed(path):
        return
    settings.USERDATA_DIR.mkdir(parents=True, exist_ok=True)
    index = settings.USERDATA_DIR / "model-downloads.json"
    with FileLock(str(index) + ".lock"):
        data = _read(index)
        data[str(path.resolve())] = dict(repo=repo, filename=filename,
            revision=revision, commit=commit, etag=etag,
            fingerprint=model_store.fingerprint(path))
        atomic_write_text(index, json.dumps(data, ensure_ascii=False))


def _roots():
    """Historic download locations as well as the image catalog's shared repos."""
    base = settings.ROOT / "tools_repo"
    entries = []
    if settings.MODELS_DIR.is_dir():
        for path in settings.MODELS_DIR.iterdir():
            if path.is_dir() and "__" in path.name:
                owner, name = path.name.split("__", 1)
                entries.append((path, owner + "/" + name, "main"))
    for name, repo in (("depth", "depth-anything/Depth-Anything-V2-Small-hf"),
        ("bg", "briaai/RMBG-1.4"), ("sam", "facebook/sam-vit-base"),
        ("clip", "openai/clip-vit-base-patch32"), ("enhance", "Qwen/Qwen2.5-3B-Instruct"),
        ("describe", "Qwen/Qwen2.5-VL-3B-Instruct")):
        entries.append((base / name / "model", repo, "main"))
    entries += [(base / "sam2/model", addons.SAM2_REPO, addons.SAM2_REVISION),
        (base / "qwen-layered/model", addons.QWEN_LAYERED_REPO, addons.QWEN_LAYERED_REVISION),
        (base / "qwen-layered/quantized", addons.QWEN_LAYERED_GGUF_REPO, addons.QWEN_LAYERED_GGUF_REVISION),
        (base / "omnisvg/model", "OmniSVG/OmniSVG1.1_4B", addons.OMNISVG_REVISION),
        (base / "omnisvg/processor", "Qwen/Qwen2.5-VL-3B-Instruct", addons.OMNISVG_PROCESSOR_REVISION),
        (base / "hivg/model", "xingxm/HiVG-3B-Base", addons.HIVG_REVISION),
        (base / "upscale/vae", "madebyollin/sdxl-vae-fp16-fix", "main"),
        (base / "upscale/controlnet", "xinsir/controlnet-tile-sdxl-1.0", "main")]
    from . import moss_dmd, taesd
    return entries + [(moss_dmd.path().parent, moss_dmd.REPO, moss_dmd.REVISION)] + [
        (spec.path().parent, spec.repo, spec.revision) for spec in taesd.SPECS.values()]


def _files(root):
    if not root.is_dir() or root.is_symlink():
        return
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP and not Path(folder, d).is_symlink()
                   and not getattr(os.path, "isjunction", lambda _: False)(Path(folder, d))]
        for name in files:
            path = Path(folder, name)
            if path.suffix.lower() in EXTENSIONS and path.is_file() and _allowed(path):
                yield path


def discover():
    """Recover old installations via known roots, HF metadata and verified hashes."""
    entries = {}
    hashes = {}
    for sha, files in model_store._read().items():
        for name, fingerprint in files.items():
            hashes[name] = (sha, fingerprint)

    def add(path, repo, filename, revision="main", **extra):
        path = Path(path)
        if not path.is_file() or not _allowed(path):
            return
        key = str(path.resolve())
        old = entries.get(key, {})
        entry = dict(path=key, repo=repo, filename=filename, revision=revision, **extra)
        cached = hashes.get(key)
        if cached and cached[1] == model_store.fingerprint(path):
            entry.setdefault("etag", cached[0])
        else:
            # Standard HF local_dir metadata: commit, etag, timestamp. Trust
            # only an unchanged/older payload; later local edits are unknown.
            root = path
            for _ in Path(filename).parts:
                root = root.parent
            metadata = root / ".cache/huggingface/download" / (filename + ".metadata")
            try:
                lines = metadata.read_text(encoding="utf-8").splitlines()
                if len(lines) >= 2 and path.stat().st_mtime_ns <= metadata.stat().st_mtime_ns:
                    entry.setdefault("etag", lines[1])
                    entry.setdefault("commit", lines[0])
            except OSError:
                pass
        entries[key] = {**old, **entry}

    for root, repo, revision in _roots():
        for path in _files(root):
            add(path, repo, path.relative_to(root).as_posix(), revision)
    # VOSR's bundle uses nested repo paths, whereas the DMD LoRA is renamed.
    from . import vosr, moss_dmd
    for filename in vosr.FILES:
        add(vosr.model_dir() / filename, vosr.MODEL_REPO, filename, vosr.MODEL_REVISION)
    add(moss_dmd.path(), moss_dmd.REPO, moss_dmd.FILENAME, moss_dmd.REVISION)
    for key, spec in addons.SEED_MODELS.items():
        add(addons.root("seedvr2") / "models" / spec, "AInVFX/SeedVR2_comfyUI", spec)
    for spec in addons.SEED_CONVROT_MODELS.values():
        add(addons.root("seedvr2") / "models" / spec, "ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8", spec)
    add(addons.root("seedvr2") / "models/ema_vae_fp16.safetensors", "numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors")
    add(settings.ROOT / "tools_repo/upscale/sd_xl_base_1.0.safetensors",
        "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors")
    trellis_root = settings.MODELS_DIR / "trellis"
    for path in _files(trellis_root):
        if path.name.startswith("pixal3d_"):
            add(path, "vegax87/Pixal3D", path.name)
        else:
            add(path, "ilintar/trellis2-gguf", path.relative_to(trellis_root).as_posix())
    # Explicit download provenance takes priority, preserving requested branch
    # vs pinned snapshot commit. Changed local files never reuse the old etag.
    for name, entry in _read(settings.USERDATA_DIR / "model-downloads.json").items():
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) and entry[k]
                                                for k in ("repo", "filename", "revision")):
            continue
        path = Path(name)
        if not path.is_file() or not _allowed(path):
            continue
        if entry.get("fingerprint") == model_store.fingerprint(path):
            add(path, entry["repo"], entry["filename"], entry["revision"],
                etag=entry.get("etag"), commit=entry.get("commit"))
        else:
            add(path, entry["repo"], entry["filename"], entry["revision"], local_modified=True)
            entries[str(path.resolve())].pop("etag", None)
    for path in [*model_store.weight_files(), *(_files(settings.LORA_DIR) or [])]:
        if path.suffix.lower() in model_store.WEIGHTS and str(path.resolve()) not in entries:
            entries[str(path.resolve())] = dict(path=str(path.resolve()), repo=None, filename=path.name)
    return list(entries.values())


def _field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def check(log=print):
    from huggingface_hub import HfApi, constants
    settings.configure_hf_env()
    entries = discover()
    result = []
    api = HfApi()
    cache = {}
    for item in entries:
        repo = item["repo"]
        if repo is None:
            result.append({**item, "status": "unmapped"})
            continue
        pinned = bool(re.fullmatch(r"[a-fA-F0-9]{40,64}", item["revision"]))
        revision = "main" if pinned else item["revision"]
        key = (repo, revision)
        if key not in cache:
            if constants.HF_HUB_OFFLINE:
                cache[key] = RuntimeError("Hugging Face offline mode")
            else:
                log(f"[models] Checking {repo}...")
                try:
                    info = api.model_info(repo, revision=revision, files_metadata=True, timeout=15)
                    cache[key] = {file.rfilename: file for file in info.siblings or []}
                except Exception as exc:
                    cache[key] = exc
        remote = cache[key]
        row = {**item, "pinned": pinned}
        if isinstance(remote, Exception):
            row.update(status="unavailable", detail=str(remote))
        elif item["filename"] not in remote:
            row.update(status="missing_upstream")
        else:
            file = remote[item["filename"]]
            etag = _field(_field(file, "lfs"), "sha256") or _field(file, "blob_id")
            row.update(remote_etag=etag, remote_size=_field(file, "size"))
            local = Path(item["path"])
            if etag and (not item.get("etag") or len(item["etag"]) != len(etag)):
                # Older installs have no transfer manifest. Identify once from
                # the actual local bytes; subsequent checks use the fingerprint.
                # HF LFS uses SHA-256; ordinary Git blobs use Git's SHA-1 framing.
                log("[models] Identifying local file once: " + str(local))
                before = model_store.fingerprint(local)
                digest = hashlib.sha256() if len(etag) == 64 else hashlib.sha1()
                if len(etag) == 40:
                    digest.update(f"blob {local.stat().st_size}\0".encode())
                with local.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                if before == model_store.fingerprint(local):
                    item["etag"] = digest.hexdigest()
                    row["etag"] = item["etag"]
                    record(local, repo=repo, filename=item["filename"], revision=item["revision"],
                           commit=item.get("commit"), etag=item["etag"])
            if item.get("etag") and item["etag"] == etag:
                row.update(status="up_to_date")
            elif item.get("local_modified"):
                row.update(status="local_modified")
            elif not item.get("etag") or not etag:
                row.update(status="unverified")
            else:
                row.update(status="pinned_update" if pinned else "update_available")
        result.append(row)
    counts = {status: sum(row["status"] == status for row in result) for status in
              ("up_to_date", "update_available", "pinned_update", "unverified", "local_modified",
               "unavailable", "missing_upstream", "unmapped")}
    for row in result:
        if row["status"] != "up_to_date":
            size = row.get("remote_size")
            suffix = f" ({size / 1e9:.2f} GB)" if size is not None else ""
            name = f"{row['repo']}/{row['filename']}" if row.get("repo") else row["path"]
            log(f"[models] {row['status']}: {name}{suffix}")
    log(f"[models] {len(result)} installed file(s): " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    log("[models] Available updates: use Models > Install / repair or Download selected image weights. "
        "Pinned toolkit versions require a compatible runtime update.")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "counts": counts, "files": result}
    target = settings.USERDATA_DIR / "model-update-report.json"
    atomic_write_text(target, json.dumps(report, ensure_ascii=False, indent=2))
    log("[models] Report: " + str(target))
    return report
