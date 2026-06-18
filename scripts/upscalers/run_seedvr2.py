#!/usr/bin/env python3
"""Runner SeedVR2 — exécution IN-PROCESS avec correctif Windows.

Pourquoi in-process et pas un simple sous-appel du script officiel :
  1. Le script officiel force `dist.init_process_group(backend="nccl")`, or
     NCCL n'existe pas sous Windows -> on monkeypatch vers le backend "gloo".
  2. Sa boucle de sauvegarde zippe sur `fps_lists`, vide pour une image seule,
     donc aucune image n'est écrite. On reproduit ici proprement le chemin image.

On réutilise les composants du module officiel (transforms, VAE, DiT, étape de
génération) — on ne réimplémente pas le modèle, juste l'orchestration image.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _patch_distributed_for_windows():
    """Force le backend gloo (mono-GPU) là où NCCL est indisponible (Windows)."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    import torch.distributed as dist
    _orig = dist.init_process_group

    def _patched(*a, **k):
        nccl_ok = hasattr(dist, "is_nccl_available") and dist.is_nccl_available()
        if sys.platform == "win32" or not nccl_ok:
            k["backend"] = "gloo"
            a = tuple(x for x in a if x != "nccl")
        return _orig(*a, **k)

    dist.init_process_group = _patched


def _install_flash_attn_shim():
    """Fournit un `flash_attn` minimal basé sur PyTorch SDPA.

    flash-attn ne se compile pas sous Windows ; SeedVR2 ne s'en sert que pour
    `flash_attn_varlen_func`. On reproduit ce calcul (attention par segment
    défini par cu_seqlens) avec scaled_dot_product_attention.
    """
    import types
    try:
        import flash_attn  # vrai paquet déjà là ?
        return
    except Exception:
        pass

    import torch
    import torch.nn.functional as F

    def _seg(q, k, v, scale, causal):  # (s,h,d) -> SDPA (h,s,d) -> (s,h,d)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            dropout_p=0.0, is_causal=causal, scale=scale)
        return out.transpose(0, 1)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                               max_seqlen_q=None, max_seqlen_k=None,
                               dropout_p=0.0, softmax_scale=None, causal=False,
                               **kwargs):
        cq = cu_seqlens_q.tolist()
        ck = cu_seqlens_k.tolist()
        outs = []
        for i in range(len(cq) - 1):
            outs.append(_seg(q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]],
                             v[ck[i]:ck[i + 1]], softmax_scale, causal))
        return torch.cat(outs, dim=0)

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                        causal=False, **kwargs):  # (b,s,h,d)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=causal, scale=softmax_scale)
        return o.transpose(1, 2)

    m = types.ModuleType("flash_attn")
    m.flash_attn_varlen_func = flash_attn_varlen_func
    m.flash_attn_func = flash_attn_func
    m.__version__ = "0.0.0-sdpa-shim"
    sys.modules["flash_attn"] = m
    print("[SeedVR2] flash-attn absent -> shim PyTorch SDPA activé "
          "(plus lent mais fonctionne sous Windows).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    repo = Path(args.repo_dir).resolve()
    # Le script officiel utilise des chemins relatifs (./ckpts, ./configs_3b,
    # ./pos_emb.pt …) -> cwd ET sys.path doivent pointer la racine du dépôt.
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    print(f"[SeedVR2] dépôt : {repo}", flush=True)

    # Dossier d'entrée propre (une seule image).
    in_dir = Path(args.output_dir).resolve().parent / "seedvr2_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    for old in in_dir.glob("*"):
        try:
            old.unlink()
        except OSError:
            pass
    img_name = Path(args.input).name
    shutil.copy(args.input, in_dir / img_name)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_distributed_for_windows()
    _install_flash_attn_shim()

    import torch
    # Résolution cible = taille de l'image × facteur (NaResize vise sqrt(h*w)).
    try:
        from PIL import Image
        with Image.open(args.input) as im:
            w0, h0 = im.size
    except Exception:
        w0, h0 = 1280, 720
    res_w = max(16, int(w0) * max(1, args.scale))
    res_h = max(16, int(h0) * max(1, args.scale))
    seed = args.seed if args.seed and args.seed >= 0 else random.randint(0, 2**31 - 1)
    print(f"[SeedVR2] entrée {w0}x{h0} -> cible ~{res_w}x{res_h} (×{args.scale}), "
          f"seed={seed}", flush=True)

    # Import du module officiel SANS déclencher son argparse (__main__).
    sys.argv = ["inference_seedvr2_3b.py"]
    print("[SeedVR2] chargement du modèle (peut être long)…", flush=True)
    import importlib
    mod = importlib.import_module("projects.inference_seedvr2_3b")

    runner = mod.configure_runner(1)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1  # défaut SeedVR2-3B
    runner.configure_diffusion()
    mod.set_seed(seed, same_across_ranks=True)

    dev = mod.get_device()
    video_transform = mod.Compose([
        mod.NaResize(resolution=(res_h * res_w) ** 0.5, mode="area",
                     downsample_only=False),
        mod.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        mod.DivisibleCrop((16, 16)),
        mod.Normalize(0.5, 0.5),
        mod.Rearrange("t c h w -> c t h w"),
    ])

    # Lecture image -> TCHW (t=1), normalisée.
    img = mod.read_image(str(in_dir / img_name)).unsqueeze(0) / 255.0
    cond = video_transform(img.to(dev))
    input_videos = [cond]
    ori_lengths = [cond.size(1)]
    cond_latents = [cond]  # image (t=1) : cut_videos est un no-op

    print("[SeedVR2] encodage VAE…", flush=True)
    runner.dit.to("cpu")
    runner.vae.to(dev)
    cond_latents = runner.vae_encode(cond_latents)
    runner.vae.to("cpu")
    runner.dit.to(dev)

    text_pos = torch.load("pos_emb.pt")
    text_neg = torch.load("neg_emb.pt")
    text_embeds = {"texts_pos": [text_pos.to(dev)], "texts_neg": [text_neg.to(dev)]}

    print("[SeedVR2] génération…", flush=True)
    samples = mod.generation_step(runner, text_embeds, cond_latents=cond_latents)
    runner.dit.to("cpu")
    sample = samples[0]

    # Color-fix (optionnel ; color_fix.py absent du dépôt -> ignoré).
    inp = input_videos[0]
    inp = (mod.rearrange(inp[:, None], "c t h w -> t c h w")
           if inp.ndim == 3 else mod.rearrange(inp, "c t h w -> t c h w"))
    if getattr(mod, "use_colorfix", False):
        sample = mod.wavelet_reconstruction(sample.to("cpu"),
                                            inp[: sample.size(0)].to("cpu"))
    else:
        sample = sample.to("cpu")

    sample = (mod.rearrange(sample[:, None], "t c h w -> t h w c")
              if sample.ndim == 3 else mod.rearrange(sample, "t c h w -> t h w c"))
    sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
    sample = sample.to(torch.uint8).numpy()

    out_path = out_dir / img_name
    mod.mediapy.write_image(str(out_path), sample[0] if sample.shape[0] == 1
                            else sample.squeeze(0))
    print(f"[SeedVR2] image écrite : {out_path}", flush=True)


if __name__ == "__main__":
    main()
