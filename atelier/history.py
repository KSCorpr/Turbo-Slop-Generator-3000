"""Paged local image history. Only the displayed page is decoded.

The originals remain in outputs/. Disposable thumbnails live in tmp/ and
never replace downloads. Opening the studio does not scan the history.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import settings

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
PAGE_SIZE = 24
_lock = threading.Lock()
_cached: tuple = ()
_entries: tuple = ()


@dataclass(frozen=True)
class Output:
    name: str
    modified_ns: int
    size: int


def resolve(name: str) -> Path:
    """Do not turn a browser-supplied name into an arbitrary file download."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Select an image from the gallery.")
    root = settings.OUTPUT_DIR.resolve()
    path = root / name
    if (path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file()
            or path.resolve().parent != root):
        raise ValueError("This image is no longer available in outputs/.")
    return path


def scan() -> tuple[Output, ...]:
    global _cached, _entries
    root = settings.OUTPUT_DIR
    try:
        stamp = root.stat().st_mtime_ns
    except OSError:
        return ()
    # Directory mtime detects arrivals/deletions. The short TTL also notices
    # in-place edits, which do not change a directory's mtime.
    key = (str(root.resolve()), stamp, int(time.monotonic() // 2))
    with _lock:
        if key == _cached:
            return _entries
        found = []
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        st = entry.stat(follow_symlinks=False)
                        found.append(Output(entry.name, st.st_mtime_ns, st.st_size))
                    except OSError:
                        continue
        except OSError:
            return ()
        _entries = tuple(sorted(found, key=lambda e: (e.modified_ns, e.name), reverse=True))
        _cached = key
        return _entries


def page(query: str = "", number: int = 1) -> tuple[list[Output], int, int]:
    query = (query or "").strip().casefold()
    found = [entry for entry in scan() if query in entry.name.casefold()]
    pages = max(1, (len(found) + PAGE_SIZE - 1) // PAGE_SIZE)
    number = max(1, min(int(number or 1), pages))
    start = (number - 1) * PAGE_SIZE
    return found[start:start + PAGE_SIZE], len(found), number


def thumbnail(entry: Output) -> str | None:
    try:
        path = resolve(entry.name)
        cache = settings.TMP_DIR / "history-thumbnails"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{path}:{entry.modified_ns}:{entry.size}".encode()).hexdigest()[:32]
        dest = cache / f"{key}.png"
        if dest.is_file():
            return str(dest)
        with Image.open(path) as source:
            source.draft("RGB", (448, 448))
            image = ImageOps.exif_transpose(source)
            image.thumbnail((448, 448), Image.Resampling.LANCZOS, reducing_gap=3)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            # Atomic publish: concurrent page requests never read half a PNG.
            fd, temporary = tempfile.mkstemp(dir=cache, suffix=".png")
            try:
                with os.fdopen(fd, "wb") as stream:
                    image.save(stream, format="PNG", compress_level=3)
                os.replace(temporary, dest)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return str(dest)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def prune_thumbnails(keep: int = 512) -> None:
    """Bound the disposable cache; never touch originals or other temp files."""
    cache = settings.TMP_DIR / "history-thumbnails"
    try:
        files = sorted(cache.glob("*.png"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for path in files[keep:]:
            try:
                path.unlink()
            except OSError:
                pass
    except OSError:
        pass


def details(name: str) -> dict:
    path = resolve(name)
    data = {}
    for suffix in (".json", ".txt"):
        sidecar = path.with_suffix(suffix)
        try:
            if sidecar.resolve().parent != path.parent or sidecar.stat().st_size > 1_048_576:
                continue
            raw = sidecar.read_text(encoding="utf-8")
            if suffix == ".json":
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data.update(parsed)
            else:
                data["text"] = raw
                # Read the existing sidecar format as well as new JSON files.
                data.setdefault("prompt", re.split(r"\n(?:Negative prompt:|Model:|Steps:|Date:)", raw, maxsplit=1)[0])
                model = re.search(r"^Model: .*\(([^()]+)\)$", raw, re.M)
                if model:
                    data.setdefault("model_id", model.group(1))
        except (OSError, ValueError, UnicodeError):
            continue
    data["path"] = str(path)
    data["name"] = name
    return data
