#!/usr/bin/env python3
"""Agrandissement déterministe par un modèle MODERNE, via spandrel.

Pourquoi ce runner existe alors qu'il y a déjà l'ESRGAN natif : sd.cpp
n'implémente qu'UNE architecture de super-résolution, RRDBNet (ESRGAN, 2018).
Tout le catalogue d'upscalers de l'app est donc la même architecture, et son
défaut — champ de vision étroit, halo sur les traits nets, grain inventé sur
les aplats — n'est pas un réglage à trouver, c'est ce que fait ce réseau.

spandrel charge 42 architectures sous licence permissive (DAT, ATD, DRCT, HAT,
SPAN, PLKSR, MoSR, RGT, SwinIR, Compact…). Il est DÉJÀ installé : c'est la
dépendance de la restauration de visages. Le coût de ce chemin est donc un
fichier de poids, pas un environnement de plus.

Aucun prompt, aucune diffusion : à graine égale ou non, la même image donne
toujours exactement le même résultat. C'est le point — un upscale génératif
« invente » et c'est ce qui donne l'aspect peinture.
"""
import argparse
import gc
import sys
from pathlib import Path

# Le Python embarqué de Windows n'ajoute pas le dossier du script au chemin.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _device import pick_device                                  # noqa: E402
from atelier.engine.tiling import tile_for_vram, upscale_tiled   # noqa: E402


def _load_source(path: Path):
    """Ouvre l'image en respectant son orientation EXIF, alpha conservé à part."""
    from PIL import Image, ImageOps
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).copy()


def _free_vram_gb(torch, device: str) -> float | None:
    if device != "cuda":
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


def _save(result, source, output: Path, expected: tuple[int, int]) -> None:
    """Écrit le PNG en préservant alpha et profil ICC, sans EXIF périmé.

    L'écriture passe par un temporaire : une sortie tronquée par un Ctrl-C
    ressemble à une image valide jusqu'à ce qu'on l'ouvre.
    """
    import os
    import uuid
    from PIL import Image

    if result.size != expected:
        raise ValueError(f"got {result.size}, expected {expected}")
    out = result.convert("RGB")
    if "A" in source.getbands() or "transparency" in source.info:
        # L'alpha n'est pas passé dans le réseau (3 canaux) : on l'agrandit
        # séparément. Lanczos y est sans risque — un masque n'a pas de texture
        # à inventer, seulement un bord à suivre.
        alpha = source.convert("RGBA").getchannel("A")
        out.putalpha(alpha.resize(expected, Image.Resampling.LANCZOS))
    tmp = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    try:
        out.save(tmp, format="PNG", icc_profile=source.info.get("icc_profile"))
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tile", type=int, default=0,
                    help="0 = decide from free VRAM; a single pass when it fits")
    ap.add_argument("--full-precision", action="store_true",
                    help="force fp32 (slower, more memory, always exact)")
    args = ap.parse_args()

    import numpy as np
    import torch
    import spandrel
    from spandrel import ImageModelDescriptor, ModelLoader

    # ESRGAN vit dans le paquet principal ; les architectures non commerciales
    # (CodeFormer, SRFormer…) sont dans extra_arches et ne sont PAS chargées
    # ici : cet outil sert des images qu'on peut vendre.
    descriptor = ModelLoader().load_from_file(args.model)
    if not isinstance(descriptor, ImageModelDescriptor):
        sys.exit("This file is not an image super-resolution model.")
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        sys.exit(f"Only RGB models are supported (this one is "
                 f"{descriptor.input_channels}→{descriptor.output_channels}).")

    scale = int(descriptor.scale)
    device = pick_device(torch)
    arch = descriptor.architecture.name
    half = (device == "cuda" and descriptor.supports_half
            and not args.full_precision)
    dtype = torch.float16 if half else torch.float32
    descriptor.to(device=device, dtype=dtype).eval()

    source = _load_source(Path(args.input))
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    tile = args.tile if args.tile > 0 else tile_for_vram(
        w, h, scale, _free_vram_gb(torch, device))
    print(f"{arch} ×{scale} on {device} ({'fp16' if half else 'fp32'})",
          flush=True)
    print(f"{w}×{h} → {w * scale}×{h * scale}", flush=True)

    def predict(block: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(block.transpose(2, 0, 1)))
        t = t.unsqueeze(0).to(device=device, dtype=dtype)
        with torch.inference_mode():
            out = descriptor(t)
        return out[0].float().cpu().numpy().transpose(1, 2, 0)

    def log(msg: str) -> None:
        print(msg, flush=True)

    # Échelle de repli. L'ordre suit ce que chaque marche COÛTE : la précision
    # ne change pas l'image (elle la rend seulement exacte), une tuile plus
    # petite ajoute des coutures, le CPU coûte des minutes. On sacrifie donc
    # dans cet ordre-là, et jamais la taille demandée.
    while True:
        try:
            result = upscale_tiled(rgb, predict, scale=scale, tile=tile,
                                   log=log)
            break
        except ArithmeticError:
            # NaN : cette architecture ment sur son support du fp16.
            if dtype is torch.float32:
                sys.exit("The model produced non-finite pixels even in fp32.")
            log("[upscale] non-finite pixels in fp16 — retrying in fp32.")
            dtype = torch.float32
            descriptor.to(dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            if device != "cuda":
                raise
            if tile == 0:
                tile = 512
            elif tile > 128:
                tile //= 2
            else:
                sys.exit("Out of GPU memory even at 128 px tiles. "
                         "Close what else uses the card, or upscale in two "
                         "halves.")
            log(f"[upscale] out of GPU memory — retrying with {tile} px tiles.")
        # Hors du bloc except : les tenseurs de la trace y seraient encore
        # référencés, et le cache ne se libérerait pas.
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    from PIL import Image
    out = Image.fromarray((result * 255.0).round().astype(np.uint8))
    _save(out, source, Path(args.output), (w * scale, h * scale))
    print(f"Saved {w * scale}×{h * scale}: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
