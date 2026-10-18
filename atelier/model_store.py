"""Reuse identical immutable weights, preserving every engine's existing paths.

The index owns no weight copies: live files are hard-linked on the same volume.
Deleting one model therefore cannot remove another model's independent path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from . import settings
from .fileio import atomic_write_text

WEIGHTS = {".gguf", ".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".sft", ".onnx"}
SKIP = {".venv", ".git", ".cache", "source", "comfy", "python", "offload", "__pycache__"}


def roots():
    return [settings.MODELS_DIR.resolve(), (settings.ROOT / "tools_repo").resolve()]


def managed(path):
    path = Path(path)
    resolved = path.resolve()
    return (path.suffix.lower() in WEIGHTS and not path.is_symlink() and
            any(resolved.is_relative_to(root) and not any(part in SKIP for part in resolved.relative_to(root).parts)
                for root in roots()))


def fingerprint(path):
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino]


def digest(path, check=lambda: None):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            check()
            result.update(chunk)
    return result.hexdigest()


def _index_path():
    return settings.USERDATA_DIR / "shared-weights.json"


def _read():
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {sha: {p: fp for p,fp in files.items() if isinstance(p,str) and isinstance(fp,list) and len(fp) == 4}
                for sha,files in data.items() if isinstance(sha,str) and len(sha) == 64 and isinstance(files,dict)}
    except (OSError, ValueError):
        return {}


def _lock():
    from filelock import FileLock
    settings.USERDATA_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(str(settings.USERDATA_DIR / "shared-weights.lock"))


def _write(index):
    atomic_write_text(_index_path(), json.dumps(index, ensure_ascii=False))


def known(path, sha, index=None):
    """A cached hash is valid only while the complete file fingerprint matches."""
    try:
        return (index if index is not None else _read()).get(sha, {}).get(str(Path(path).resolve())) == fingerprint(path)
    except (OSError, AttributeError):
        return False


def _source(sha, size, index):
    entries = index.get(sha, {})
    if not isinstance(entries, dict):
        return None
    for name in list(entries):
        path = Path(name)
        try:
            if managed(path) and known(path, sha, index) and (size is None or path.stat().st_size == size):
                return path
        except OSError:
            continue
    return None


def _link(source, dest, *, allow_copy=False):
    """Replace only after a complete new alias exists; leave originals on error."""
    source, dest = Path(source), Path(dest)
    if dest.exists() and os.path.samefile(source, dest):
        return "shared"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name("."+dest.name+"."+uuid.uuid4().hex+".shared")
    try:
        try:
            os.link(source, temporary)
            kind = "shared"
        except OSError:
            if not allow_copy:
                return "separate"
            shutil.copyfile(source, temporary)
            kind = "local copy"
        os.replace(temporary, dest)
        return kind
    finally:
        temporary.unlink(missing_ok=True)


def reuse(sha, size, dest):
    """Use a SHA-256-identified file before starting another network transfer."""
    dest = Path(dest)
    if not managed(dest):
        return False
    with _lock():
        index = _read()
        source = _source(sha, size, index)
        if source is None:
            return False
        try:
            kind = _link(source, dest, allow_copy=True)
        except OSError:
            return False
        index.setdefault(sha, {})[str(dest.resolve())] = fingerprint(dest)
        _write(index)
        print(f"[models] Reused verified local weight ({kind}): {dest.name}", flush=True)
        return True


def remember(path, sha=None):
    path = Path(path)
    if not path.is_file() or not managed(path):
        return
    with _lock():
        index = _read()
        if sha is None:
            name = str(path.resolve())
            sha = next((key for key,files in index.items() if files.get(name) == fingerprint(path)), None)
        sha = sha or digest(path)
        source = _source(sha, path.stat().st_size, index)
        if source is not None:
            _link(source, path)
        index.setdefault(sha, {})[str(path.resolve())] = fingerprint(path)
        _write(index)


def weight_files():
    visited = set()
    for root in roots():
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(folder, d).is_symlink()
                             and not getattr(os.path, "isjunction", lambda _: False)(Path(folder, d)))
            for name in sorted(files):
                path = Path(folder, name)
                if managed(path) and path.resolve() not in visited:
                    visited.add(path.resolve())
                    yield path


def consolidate(log=print, check=lambda: None):
    """Hash existing weights explicitly; never scan/hash at application startup."""
    from .transfers import transfer_lock
    saved = linked = separate = 0
    # The same lock ordering is used by downloads: transfer, then index.
    with transfer_lock(), _lock():
        index = {}
        inode_hashes = {}
        for number, path in enumerate(weight_files(), 1):
            check()
            before = fingerprint(path)
            identity = tuple(before[2:])
            sha = inode_hashes.get(identity)
            if sha is None:
                log(f"Checking {number}: {path.name} ({before[0]/1e9:.2f} GB)")
                sha = digest(path, check)
                if before != fingerprint(path):
                    log(f"Skipped a file changed during checking: {path.name}")
                    continue
                inode_hashes[identity] = sha
            source = _source(sha, before[0], index)
            if source is not None and not os.path.samefile(source, path):
                old_links = path.stat().st_nlink
                try:
                    mode = _link(source, path)
                except OSError as exc:
                    log(f"Kept separate: {path.name} ({exc})")
                    mode = "separate"
                if mode == "shared":
                    if old_links == 1:
                        saved += before[0]
                    linked += 1
                    log(f"Shared identical bytes: {path.name}")
                else:
                    separate += 1
                    log(f"Kept separate (hard links unavailable or different volumes): {path.name}")
            index.setdefault(sha, {})[str(path.resolve())] = fingerprint(path)
            _write(index)
    return {"linked_files": linked, "saved_bytes": saved, "separate_files": separate}
