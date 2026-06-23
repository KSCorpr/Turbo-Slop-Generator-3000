#!/usr/bin/env python3
"""Upscale créatif moderne (façon Magnific) : SDXL + ControlNet Tile via diffusers.

Pré-agrandit l'image (Lanczos), puis raffine PAR TUILES en img2img guidé par un
ControlNet Tile : chaque tuile est ancrée sur le contenu agrandi, donc les tuiles
voisines restent cohérentes et le recollage (fondu cosinus) est sans couture.

Lancé en sous-process pour ne pas verrouiller les DLL de torch dans Gradio.
Offload CPU activé -> tient sur 11-12 Go de VRAM (plus lent, mais robuste).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _feather(h, w, fade, np):
    fy = np.ones(h, np.float32)
    fx = np.ones(w, np.float32)
    f = max(1, min(fade, h // 2, w // 2))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32))
    fy[:f] = ramp; fy[-f:] = ramp[::-1]
    fx[:f] = ramp; fx[-f:] = ramp[::-1]
    return (fy[:, None] * fx[None, :])[:, :, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)   # checkpoint SDXL (1 fichier)
    ap.add_argument("--controlnet", required=True)   # dossier ControlNet Tile
    ap.add_argument("--vae", required=True)           # dossier VAE fp16-fix
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--creativity", type=float, default=0.35)  # strength img2img
    ap.add_argument("--cn-scale", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tile", type=int, default=1024)   # résolution native SDXL
    ap.add_argument("--overlap", type=int, default=160)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    try:
        from diffusers import (AutoencoderKL, ControlNetModel,
                               DPMSolverMultistepScheduler,
                               StableDiffusionXLControlNetImg2ImgPipeline)
    except ImportError:
        sys.exit("diffusers manquant. Réinstallez l'outil depuis l'onglet Upscale.")

    if not torch.cuda.is_available():
        print("[upscale] ATTENTION : pas de CUDA, exécution CPU (très lent).")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("[upscale] chargement SDXL + ControlNet Tile…", flush=True)
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        args.base_model, controlnet=controlnet, vae=vae, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    if torch.cuda.is_available():
        try:
            pipe.enable_vae_tiling()
        except Exception:  # noqa: BLE001
            pass
        free, _ = torch.cuda.mem_get_info()
        if free >= int(9.5 * 1024 ** 3):
            # Assez de VRAM libre -> modèle résident sur le GPU (BIEN plus rapide :
            # pas de rechargement par tuile).
            pipe.to("cuda")
            print(f"[upscale] VRAM libre {free/1e9:.1f} Go -> modèle résident GPU "
                  "(rapide).", flush=True)
        else:
            # VRAM serrée -> offload CPU (un module à la fois, plus lent).
            pipe.enable_model_cpu_offload()
            print(f"[upscale] VRAM libre {free/1e9:.1f} Go -> offload CPU "
                  "(plus lent).", flush=True)

    prompt = args.prompt or ("high quality, sharp focus, fine intricate details, "
                             "crisp textures, photorealistic, 8k, masterpiece")
    negative = ("blurry, soft, low quality, jpeg artifacts, oversaturated, "
                "deformed, oversmooth, plastic")

    img = Image.open(args.input).convert("RGB")
    tw = max(512, int(round(img.width * args.scale / 8)) * 8)
    th = max(512, int(round(img.height * args.scale / 8)) * 8)
    # Plafond : au-delà, le nombre de tuiles (donc le temps) et la taille du
    # fichier explosent. On limite le côté long à 4096 px.
    cap = 4096
    if max(tw, th) > cap:
        r = cap / max(tw, th)
        tw = max(512, int(round(tw * r / 8)) * 8)
        th = max(512, int(round(th * r / 8)) * 8)
        print(f"[upscale] cible plafonnée à {tw}x{th} (max {cap}px).", flush=True)
    print(f"[upscale] {img.width}x{img.height} -> {tw}x{th} (Lanczos puis raffinage)…",
          flush=True)
    # Pré-agrandissement + léger affinage : donne des bords nets au ControlNet
    # (qui sinon reproduit le flou du Lanczos).
    base = img.resize((tw, th), Image.LANCZOS).filter(
        ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))

    def _sharpen(im):
        return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))

    common = dict(prompt=prompt, negative_prompt=negative,
                  num_inference_steps=int(args.steps), strength=float(args.creativity),
                  guidance_scale=float(args.cfg),
                  controlnet_conditioning_scale=float(args.cn_scale))

    def refine(tile_img):
        return pipe(image=tile_img, control_image=tile_img, **common).images[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_upscaled.png")

    # Petite cible -> une seule passe.
    if max(tw, th) <= args.tile:
        print("[upscale] raffinage en 1 passe…", flush=True)
        res = _sharpen(refine(base).resize((tw, th)))
        res.save(dest)
        print(f"[upscale] image écrite : {dest}", flush=True)
        return

    # Grande cible -> tuiles avec recouvrement + fondu cosinus.
    t = max(512, (args.tile // 8) * 8)
    step = max(64, t - args.overlap)
    ys = list(range(0, max(1, th), step))
    xs = list(range(0, max(1, tw), step))
    total = len(ys) * len(xs)
    print(f"[upscale] {total} tuiles de {t}px…", flush=True)

    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw, 1), np.float32)
    n = 0
    for y in ys:
        for x in xs:
            y2, x2 = min(y + t, th), min(x + t, tw)
            y1, x1 = max(0, y2 - t), max(0, x2 - t)
            n += 1
            print(f"[upscale]   tuile {n}/{total}…", flush=True)
            tile_img = base.crop((x1, y1, x2, y2))
            res = refine(tile_img).resize((x2 - x1, y2 - y1))
            arr = np.asarray(res, np.float32)
            mask = _feather(y2 - y1, x2 - x1, args.overlap, np)
            acc[y1:y2, x1:x2] += arr * mask
            wsum[y1:y2, x1:x2] += mask

    final = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
    _sharpen(Image.fromarray(final)).save(dest)
    print(f"[upscale] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
