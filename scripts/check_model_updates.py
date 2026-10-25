#!/usr/bin/env python3
"""Check installed model files against Hugging Face metadata; no weight download."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from atelier.model_updates import check
    try:
        check()
    except Exception as exc:
        print("[models] Check unavailable: " + str(exc))
        sys.exit(1)
