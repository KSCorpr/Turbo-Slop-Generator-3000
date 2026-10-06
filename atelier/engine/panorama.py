"""Perspective views to a 360×180 equirectangular projection.

Convention: yaw 0 is +Z, positive yaw turns right toward +X; pitch is up.
The horizon is the middle row and yaw 0 the middle column. Input views must
share a camera centre. An orbit around an object cannot be mapped this way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class View:
    name: str
    yaw: float
    pitch: float


SCENE_VIEWS = tuple(View(f"yaw-{yaw:03d}", yaw, 0) for yaw in range(0, 360, 45)) + (
    View("zenith", 0, 90), View("nadir", 0, -90))
OBJECT_VIEWS = SCENE_VIEWS[:8] + (
    View("high-angle", 0, 45), View("low-angle", 0, -20))
FOV = 100.0  # square perspective images; overlap for blending


def basis(view: View):
    yaw, pitch = math.radians(view.yaw), math.radians(view.pitch)
    forward = np.array((math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                        math.cos(yaw) * math.cos(pitch)), dtype=np.float32)
    right = np.array((math.cos(yaw), 0, -math.sin(yaw)), dtype=np.float32)
    up = np.cross(forward, right)
    return forward, right, up


def _bilinear(pixels, x, y):
    h, w = pixels.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    nx, ny = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
    dx, dy = (x - ix)[..., None], (y - iy)[..., None]
    return ((pixels[iy, ix] * (1 - dx) + pixels[iy, nx] * dx) * (1 - dy)
            + (pixels[ny, ix] * (1 - dx) + pixels[ny, nx] * dx) * dy)


def equirectangular(images, *, width=4096, fov=FOV, check=None,
                    progress=None) -> Image.Image:
    """Reproject the ten ordered SCENE_VIEWS, feathering their overlaps.

    This corrects projection and blends pixels; it cannot correct different
    geometry invented independently by a generative model.
    """
    width = int(width)
    if width < 64 or width > 8192 or width % 2:
        raise ValueError("Panorama width must be even and between 64 and 8192.")
    if not 95 <= float(fov) <= 120:
        raise ValueError("Use a square field of view between 95 and 120 degrees.")
    if len(images) != len(SCENE_VIEWS):
        raise ValueError("Provide all 10 fixed-camera scene views, including zenith and nadir.")
    pixels = []
    for im in images:
        if im.width != im.height:
            raise ValueError("Each perspective view must be square.")
        pixels.append(np.asarray(im.convert("RGB"), dtype=np.float32))
    height = width // 2
    output = np.empty((height, width, 3), dtype=np.uint8)
    longitude = ((np.arange(width, dtype=np.float32) + .5) / width * 2 - 1) * np.pi
    tangent = math.tan(math.radians(float(fov) / 2))
    frames = [(p, *basis(v)) for p, v in zip(pixels, SCENE_VIEWS)]
    for start in range(0, height, 64):
        if check:
            check()
        end = min(start + 64, height)
        latitude = (.5 - (np.arange(start, end, dtype=np.float32) + .5) / height) * np.pi
        cos_lat = np.cos(latitude)[:, None]
        rays = np.stack((cos_lat * np.sin(longitude)[None, :],
                         np.broadcast_to(np.sin(latitude)[:, None], (end - start, width)),
                         cos_lat * np.cos(longitude)[None, :]), axis=-1)
        total = np.zeros_like(rays)
        weights = np.zeros(rays.shape[:2], dtype=np.float32)
        for p, forward, right, up in frames:
            depth = rays @ forward
            denom = np.maximum(depth, 1e-6) * tangent
            u, v = (rays @ right) / denom, (rays @ up) / denom
            edge = np.maximum(np.abs(u), np.abs(v))
            weight = np.clip((1 - edge) / .2, 0, 1) * np.maximum(depth, 0) ** 4
            x = (u + 1) * p.shape[1] / 2 - .5
            y = (1 - v) * p.shape[0] / 2 - .5
            total += _bilinear(p, x, y) * weight[..., None]
            weights += weight
        if np.any(weights <= 0):
            raise ValueError("The supplied camera geometry leaves uncovered directions.")
        output[start:end] = np.clip(total / weights[..., None] + .5, 0, 255).astype(np.uint8)
        if progress:
            progress(end, height)
    return Image.fromarray(output)
