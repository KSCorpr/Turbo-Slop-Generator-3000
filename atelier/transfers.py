"""Téléchargements HTTP séquentiels, plafonnés et reprenables.

Les fichiers incomplets restent distincts des poids utilisables. Un verrou
commun au projet sérialise également les installeurs lancés dans un autre Python.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import re
import socket
import time
from pathlib import Path

from . import settings
from .fileio import atomic_write_text

CHUNK_SIZE = 64 * 1024


def download_limit() -> float:
    try:
        value = float(settings.load_prefs().get("download_limit_mbps", 5))
        return max(0.0, value) if math.isfinite(value) else 5.0
    except (TypeError, ValueError):
        return 5.0


class RateLimiter:
    """Cadence les lectures, avec au plus deux petits blocs de crédit."""
    def __init__(self):
        self.next_read = time.monotonic()
        self.last_limit = None

    def wait(self, size: int = CHUNK_SIZE) -> None:
        limit = download_limit()
        if not limit:
            self.next_read = time.monotonic()
            self.last_limit = 0
            return
        now = time.monotonic()
        if limit != self.last_limit:
            self.next_read = now
            self.last_limit = limit
        duration = size / (limit * 1_000_000)
        # Créditer le temps de lecture/écriture et les dépassements du timer.
        # Auparavant, chaque lecture réseau s'ajoutait au délai du plafond :
        # une source lente était donc ralentie une seconde fois.
        # Deux blocs de crédit suffisent à amortir un timer Windows de 15 ms
        # sans accumuler une grosse rafale après une pause de plusieurs minutes.
        self.next_read = max(now - 2 * duration, self.next_read) + duration
        # Réagir aux modifications du plafond sans dormir plusieurs minutes.
        while True:
            delay = self.next_read - time.monotonic()
            if delay <= 0.000001:
                break
            time.sleep(min(0.25, delay))
            changed = download_limit()
            if changed != limit:
                self.next_read = time.monotonic()
                self.last_limit = changed
                return


def receive_buffer_size() -> int | None:
    limit = download_limit()
    if not limit:
        return None
    # Environ 100 ms au débit choisi, avec des bornes raisonnables. Empêche
    # une lecture cadencée de laisser Windows recevoir des Mo en avance.
    # Ce compromis peut réduire le débit sur un chemin à très forte latence.
    return min(2 * 1024 * 1024, max(CHUNK_SIZE, int(limit * 100_000)))


def download_session():
    """Borne le tampon AVANT connect(), sans modifier le réseau du PC."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection

    buffer_size = receive_buffer_size()
    if buffer_size is None:
        return requests.Session()
    options = [*HTTPConnection.default_socket_options,
               (socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)]

    class DownloadAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
            kwargs["socket_options"] = options
            return super().init_poolmanager(connections, maxsize, block=block, **kwargs)

        def proxy_manager_for(self, proxy, **kwargs):
            kwargs["socket_options"] = options
            return super().proxy_manager_for(proxy, **kwargs)

    session = requests.Session()
    session.mount("http://", DownloadAdapter())
    session.mount("https://", DownloadAdapter())
    return session


def transfer_lock():
    from filelock import FileLock
    settings.USERDATA_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(str(settings.USERDATA_DIR / "model-download.lock"))


def _verify(path: Path, size: int, etag: str) -> None:
    if path.stat().st_size != size:
        raise RuntimeError("Incomplete download; retry to resume it.")
    # Pour les gros poids LFS/Xet, le Hub expose leur SHA-256 comme ETag.
    if re.fullmatch(r"[0-9a-f]{64}", etag):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != etag:
            path.unlink(missing_ok=True)
            raise RuntimeError("Download checksum mismatch; retry the download.")


def hf_hub_download(*, repo_id: str, filename: str, local_dir: str,
                    revision: str = "main", **kwargs) -> str:
    settings.configure_hf_env()
    from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    if constants.HF_HUB_OFFLINE or not download_limit():
        from huggingface_hub import hf_hub_download as hub_download
        with transfer_lock():
            return hub_download(repo_id=repo_id, filename=filename,
                                local_dir=local_dir, revision=revision, **kwargs)
    from tqdm.auto import tqdm

    root = Path(local_dir).resolve()
    dest = (root / filename).resolve()
    if not dest.is_relative_to(root):
        raise ValueError("Download filename must stay inside local_dir.")
    with transfer_lock():
        url = hf_hub_url(repo_id, filename, revision=revision,
                         endpoint=constants.ENDPOINT)
        meta = get_hf_file_metadata(url, token=kwargs.get("token"))
        if meta.size is None or not meta.etag or not meta.commit_hash:
            raise RuntimeError("Hub download metadata is incomplete.")
        if dest.is_file() and dest.stat().st_size == meta.size:
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".download")
        marker = partial.with_name(partial.name + ".etag")
        identity = f"{repo_id}\n{meta.etag}\n{meta.size}"
        if not marker.is_file() or marker.read_text() != identity:
            partial.unlink(missing_ok=True)
            atomic_write_text(marker, identity)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > meta.size:
            partial.unlink()
            offset = 0
        partial.touch(exist_ok=True)
        if offset < meta.size:
            limit = download_limit()
            print(f"[download] HTTP, one stream, Xet disabled; "
                  f"limit {limit:g} MB/s ({limit * 8:g} Mbit/s); "
                  f"receive buffer {receive_buffer_size()} bytes", flush=True)
            # URL à commit immuable : un changement de main ne mélange jamais
            # deux versions, même si le transfert est repris le lendemain.
            url = hf_hub_url(repo_id, filename, revision=meta.commit_hash,
                             endpoint=constants.ENDPOINT)
            headers = build_hf_headers(token=kwargs.get("token"))
            headers["Accept-Encoding"] = "identity"
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with download_session() as session, session.get(
                    url, headers=headers, stream=True,
                    timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code == 200:
                    offset = 0  # serveur sans Range : recommencer proprement
                elif response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != meta.size:
                        raise RuntimeError("Invalid resume response from server.")
                else:
                    raise RuntimeError("Unexpected HTTP download response.")
                limiter = RateLimiter()
                started = time.monotonic()
                with partial.open("ab" if offset else "wb") as target, tqdm(
                        total=meta.size, initial=offset, unit="B", unit_scale=True,
                        desc=filename) as progress:
                    chunks = response.iter_content(chunk_size=CHUNK_SIZE)
                    while True:
                        limiter.wait(min(CHUNK_SIZE, max(0, meta.size - target.tell())))
                        chunk = next(chunks, None)
                        if chunk is None:
                            break
                        target.write(chunk)
                        progress.update(len(chunk))
                        if target.tell() > meta.size:
                            raise RuntimeError("Download exceeds expected file size.")
                elapsed = max(time.monotonic() - started, 0.001)
                speed = (partial.stat().st_size - offset) / elapsed / 1_000_000
                print(f"[download] File data: {speed:.2f} MB/s "
                      f"({speed * 8:.1f} Mbit/s). Verifying the file; "
                      "the model transfer has ended.", flush=True)
        _verify(partial, meta.size, meta.etag)
        partial.replace(dest)
        marker.unlink(missing_ok=True)
        return str(dest)


def snapshot_download(*, repo_id: str, local_dir: str,
                      allow_patterns=None, ignore_patterns=None, **kwargs) -> str:
    settings.configure_hf_env()
    from huggingface_hub import HfApi, constants
    if constants.HF_HUB_OFFLINE or not download_limit():
        from huggingface_hub import snapshot_download as hub_snapshot
        with transfer_lock():
            return hub_snapshot(repo_id=repo_id, local_dir=local_dir,
                                allow_patterns=allow_patterns,
                                ignore_patterns=ignore_patterns,
                                max_workers=1, **kwargs)
    revision = kwargs.pop("revision", "main")
    info = HfApi().model_info(repo_id, revision=revision, token=kwargs.get("token"))
    def matches(name, patterns):
        if isinstance(patterns, str):
            patterns = [patterns]
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns or [])
    for file in info.siblings or []:
        name = file.rfilename
        if allow_patterns is not None and not matches(name, allow_patterns):
            continue
        if matches(name, ignore_patterns):
            continue
        hf_hub_download(repo_id=repo_id, filename=name, local_dir=local_dir,
                        revision=info.sha, **kwargs)
    return str(Path(local_dir))
