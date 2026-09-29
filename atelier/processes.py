"""Reap command-line workers, including when their output consumer fails."""
from __future__ import annotations

import subprocess


def close_worker(proc: subprocess.Popen) -> None:
    """Do not leave a GPU process running after an exception in its log reader."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
    except ProcessLookupError:
        # The child may have exited between poll() and terminate().
        pass
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
