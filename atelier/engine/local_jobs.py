"""Cancellable multi-stage local jobs, including their child processes."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from contextlib import contextmanager

from .. import settings


class LocalJob:
    def __init__(self):
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._proc = None
        self._busy = False

    @contextmanager
    def session(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("A task is already running in this workspace.")
            self._busy = True
            self.cancelled.clear()
        try:
            yield self
        finally:
            with self._lock:
                self._busy = False

    def check(self):
        if self.cancelled.is_set():
            raise RuntimeError("Task cancelled. Completed files have been kept.")

    @staticmethod
    def _stop(proc):
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()

    def cancel(self):
        with self._lock:
            if not self._busy:
                return "No task running."
            self.cancelled.set()
            proc = self._proc
        if proc:
            self._stop(proc)
        return "Stopping the task…"

    def run(self, command, log=print, *, cwd=None, env=None):
        self.check()
        command = [str(part) for part in command]
        log("$ " + subprocess.list2cmdline(command))
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        with self._lock:
            self.check()
            proc = subprocess.Popen(command, cwd=cwd or settings.ROOT,
                                    env=env or settings.child_env(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    bufsize=1, **options)
            self._proc = proc
        tail = []
        try:
            for line in proc.stdout:
                tail.append(line.rstrip())
                tail = tail[-12:]
                log(line.rstrip())
            code = proc.wait()
            self.check()
            if code:
                raise RuntimeError(f"{command[0]} exited with code {code}.\n" + "\n".join(tail))
        finally:
            self._stop(proc)
            proc.stdout.close()
            with self._lock:
                self._proc = None
