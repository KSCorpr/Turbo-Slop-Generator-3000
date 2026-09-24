"""Fichiers attendus par trellis.cpp 0.8 pour TRELLIS.2 et Pixal3D."""
from __future__ import annotations

from pathlib import Path

# Les décodeurs et DINOv3 existent déjà dans chaque variante de TRELLIS.2.
TRELLIS_FILES = (
    "ss_flow.gguf", "shape_flow_512.gguf", "shape_flow_1024.gguf",
    "tex_flow_512.gguf", "tex_flow_1024.gguf", "ss_dec.gguf",
    "shape_dec.gguf", "tex_dec.gguf", "dinov3.gguf", "birefnet.gguf",
)
PIXAL_512 = ("pixal3d_ss_flow.gguf", "pixal3d_shape_flow_512.gguf",
             "pixal3d_naf.gguf")
PIXAL_1024 = PIXAL_512 + ("pixal3d_shape_flow_1024.gguf",
                           "pixal3d_tex_flow_1024.gguf")
SHARED_512 = ("ss_dec.gguf", "shape_dec.gguf", "dinov3.gguf",
              "birefnet.gguf")
SHARED_1024 = SHARED_512 + ("tex_dec.gguf",)


def pixal_files(resolution: int) -> tuple[str, ...]:
    return PIXAL_512 if int(resolution) == 512 else PIXAL_1024


def shared_files(resolution: int) -> tuple[str, ...]:
    return SHARED_512 if int(resolution) == 512 else SHARED_1024


def all_present(directory: Path, names: tuple[str, ...]) -> bool:
    return all((directory / name).is_file() for name in names)
