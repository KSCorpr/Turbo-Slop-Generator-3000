#!/usr/bin/env python3
"""CPU COLMAP calibration in an isolated environment; no CUDA toolkit needed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def reconstruct(work: Path, sequential: bool = False):
    import pycolmap
    images = work / "images"
    database = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    print("Extracting image features (CPU)…", flush=True)
    pycolmap.extract_features(str(database), str(images),
                              sift_options={"max_image_size": 1600, "max_num_features": 8192},
                              device=pycolmap.Device.cpu)
    print("Matching camera views…", flush=True)
    matcher = pycolmap.match_sequential if sequential else pycolmap.match_exhaustive
    matcher(str(database), device=pycolmap.Device.cpu)
    print("Solving camera positions…", flush=True)
    maps = pycolmap.incremental_mapping(str(database), str(images), str(sparse))
    if not maps:
        raise RuntimeError("No reconstruction: capture more overlapping, sharp views of a static subject.")
    key, model = max(maps.items(), key=lambda item: item[1].num_reg_images())
    registered = model.num_reg_images()
    total = len(list(images.iterdir()))
    if registered < 4:
        raise RuntimeError(f"Only {registered}/{total} cameras aligned; at least four are required.")
    print(f"Aligned {registered}/{total} images. Unaligned images will not contribute.", flush=True)
    if registered < total * 0.7:
        print("WARNING: less than 70% of views aligned. The reconstruction may be incomplete.", flush=True)
    dataset = work / "dataset"
    pycolmap.undistort_images(str(dataset), str(sparse / str(key)), str(images),
                              undistort_options={"max_image_size": 1600})
    (work / "alignment.json").write_text(json.dumps({
        "registered": registered, "total": total, "points": model.num_points3D(),
        "component": key, "components": len(maps),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    reconstruct(args.work, args.sequential)
