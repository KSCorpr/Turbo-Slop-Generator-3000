#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  sam     -> Segment Anything (facebook/sam-vit-base) : extraction d'objet au clic.
  enhance -> Qwen2.5-3B-Instruct : améliore un prompt brut (LLM).
  upscale -> SDXL base + VAE fp16-fix : upscale créatif tuilé (Ultimate SD Upscale).

Réutilise les helpers torch CUDA de _torch_setup (build adaptée au GPU,
sans verrouiller de DLL). Lançable depuis l'interface ou en ligne :
    python scripts/setup_tools.py depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atelier import settings  # noqa: E402
from _torch_setup import ensure_torch_cuda, pin_numpy, sh  # noqa: E402

# Modèle léger (~100 Mo) : rapide, tourne même sur Pascal (GTX 10xx) et en CPU.
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# RMBG-1.4 (~176 Mo) : code self-contained (pur torch), poids sur HF.
BG_REPO = "briaai/RMBG-1.4"
# Segment Anything (base, ~375 Mo) via transformers, depuis HF.
SAM_REPO = "facebook/sam-vit-base"
# Améliorateur de prompt : petit LLM instruct (~6 Go fp16), tourne en sous-process.
ENHANCE_REPO = "Qwen/Qwen2.5-3B-Instruct"
# Upscale créatif tuilé : SDXL base (1 fichier) + VAE fp16-fix + ControlNet Tile
# (optionnel, verrouille la structure pour pousser la créativité sans dériver).
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0.safetensors"
VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"
CN_TILE_REPO = "xinsir/controlnet-tile-sdxl-1.0"


def install_depth():
    model_dir = settings.ROOT / "tools_repo" / "depth" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5", "pillow"])
    print(f"\nTéléchargement du modèle de profondeur ({DEPTH_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DEPTH_REPO, local_dir=str(model_dir))
    pin_numpy()  # transformers peut réintroduire NumPy 2 -> on re-fige
    print("\n[OK] Depth Anything V2 installé (profondeur + normales).")


def install_bg():
    model_dir = settings.ROOT / "tools_repo" / "bg" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5",
        "scikit-image", "pillow"])
    print(f"\nTéléchargement du modèle de suppression d'arrière-plan ({BG_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=BG_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] RMBG-1.4 installé. Disponible dans l'onglet Toolkit.")


def install_sam():
    model_dir = settings.ROOT / "tools_repo" / "sam" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5", "pillow"])
    print(f"\nTéléchargement de Segment Anything ({SAM_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=SAM_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] Segment Anything installé. Disponible dans l'onglet Toolkit.")


def install_enhance():
    model_dir = settings.ROOT / "tools_repo" / "enhance" / "model"
    ensure_torch_cuda()
    print("Installation de transformers + accelerate…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45,<5",
        "accelerate", "safetensors"])
    print(f"\nTéléchargement de l'améliorateur de prompt ({ENHANCE_REPO}, ~6 Go)…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=ENHANCE_REPO, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "tokenizer*", "vocab*", "merges*"])
    pin_numpy()
    print("\n[OK] Améliorateur de prompt installé. Bouton « ✨ Améliorer ».")


def _hf_fetch(fn, desc: str, manual_url: str, dest) -> None:
    """Téléchargement HF avec 3 tentatives + diagnostic réseau actionnable.

    Sur un poste d'entreprise, l'erreur huggingface_hub « cannot find the
    requested files / check your connection » cache presque toujours un proxy
    obligatoire, une inspection SSL ou un huggingface.co filtré — on l'explique
    au lieu de laisser le traceback brut."""
    import time
    for attempt in range(3):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  [!] échec ({type(exc).__name__}) — nouvel essai "
                      f"dans {wait} s…")
                time.sleep(wait)
            else:
                print(f"\n[X] Téléchargement impossible : {desc}")
                print("    Causes fréquentes sur un réseau d'entreprise :")
                print("    • proxy obligatoire → définissez HTTPS_PROXY="
                      "http://proxy:port avant de lancer run.bat ;")
                print("    • inspection SSL → REQUESTS_CA_BUNDLE="
                      "chemin\\vers\\ca-entreprise.pem ;")
                print("    • huggingface.co filtré → téléchargez à la main :")
                print(f"      {manual_url}")
                print(f"      et placez le fichier dans : {dest}")
                raise


def install_upscale():
    base = settings.ROOT / "tools_repo" / "upscale"
    ensure_torch_cuda()
    print("Installation de diffusers + accelerate…")
    sh([sys.executable, "-m", "pip", "install", "diffusers>=0.30,<0.32",
        "transformers>=4.45,<5", "accelerate", "safetensors", "omegaconf", "pillow"])
    from huggingface_hub import hf_hub_download, snapshot_download
    # Dossier où déposer des checkpoints SDXL perso (sélectionnables dans l'UI).
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)

    # MODE HORS-LIGNE : chaque composant DÉJÀ présent est conservé (aucun
    # téléchargement). Sur un réseau qui bloque huggingface.co (entreprise),
    # téléchargez les fichiers depuis un poste qui atteint HF et déposez-les
    # aux emplacements indiqués — puis relancez : l'install les détecte et saute.
    base_ck = base / "sd_xl_base_1.0.safetensors"
    vae_ok = (base / "vae").is_dir() and any((base / "vae").glob("*.safetensors"))
    cn_ok = (base / "controlnet").is_dir() and \
        any((base / "controlnet").glob("*.safetensors"))

    if base_ck.is_file():
        print(f"  [OK] checkpoint SDXL déjà présent ({SDXL_FILE}) — pas de "
              "téléchargement.")
    else:
        print(f"\nTéléchargement du checkpoint SDXL ({SDXL_REPO}/{SDXL_FILE}, "
              "~6,6 Go)…")
        _hf_fetch(lambda: hf_hub_download(repo_id=SDXL_REPO, filename=SDXL_FILE,
                                          local_dir=str(base)),
                  f"checkpoint SDXL ({SDXL_FILE})",
                  f"https://huggingface.co/{SDXL_REPO}/resolve/main/{SDXL_FILE}",
                  base)

    if vae_ok:
        print("  [OK] VAE fp16-fix déjà présente — pas de téléchargement.")
    else:
        print(f"\nTéléchargement de la VAE fp16-fix ({VAE_FIX_REPO})…")
        _hf_fetch(lambda: snapshot_download(repo_id=VAE_FIX_REPO,
                                            local_dir=str(base / "vae"),
                                            allow_patterns=["*.json", "*.safetensors"]),
                  "VAE fp16-fix",
                  f"https://huggingface.co/{VAE_FIX_REPO}/tree/main",
                  base / "vae")

    # ControlNet Tile : OPTIONNEL (l'upscale marche sans). Un échec ici n'arrête
    # pas l'install.
    if cn_ok:
        print("  [OK] ControlNet Tile déjà présent — pas de téléchargement.")
    else:
        print(f"\nTéléchargement du ControlNet Tile ({CN_TILE_REPO}, ~2,5 Go, "
              "optionnel)…")
        try:
            _hf_fetch(lambda: snapshot_download(repo_id=CN_TILE_REPO,
                                                local_dir=str(base / "controlnet"),
                                                allow_patterns=["*.json", "*.safetensors"]),
                      "ControlNet Tile",
                      f"https://huggingface.co/{CN_TILE_REPO}/tree/main",
                      base / "controlnet")
        except Exception:  # noqa: BLE001
            print("  [!] ControlNet Tile non installé (optionnel) — l'upscale "
                  "créatif fonctionne sans, décochez « ControlNet » dans l'UI.")

    pin_numpy()
    print("\n[OK] Upscale créatif SDXL installé "
          "(onglet Toolkit → Upscale créatif).")


# --------------------------------------------------------------------------- #
#  SeedVR2 — upscale de RESTAURATION en un pas (image fixe)
#
#  Il n'existe ni paquet pip ni pipeline diffusers pour SeedVR2 : le code
#  d'inférence de référence est celui du dépôt numz, qui fournit un
#  « inference_cli.py » explicitement documenté comme utilisable SANS ComfyUI.
#  On clone donc ce dépôt et on appelle son CLI en sous-process — on n'installe
#  ni ne lance ComfyUI, et on ne recopie pas non plus des centaines de lignes
#  d'architecture qu'il faudrait ensuite maintenir à la main.
# --------------------------------------------------------------------------- #
SEEDVR2_GIT = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
SEEDVR2_W_REPO = "lvladikov/SeedVR2-1.4B"
SEEDVR2_DIT = "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors"

# Config d'architecture du 1.4B (distillation 6 blocs du 7B). Fournie par
# l'auteur des poids, dimensions recoupées tenseur par tenseur avec le
# checkpoint. Le dépôt d'inférence ne connaît que le 3B et le 7B.
SEEDVR2_CFG_1_4B = """\
# SeedVR2-1.4B (6-layer distillation) — architecture config.
# configs_7b/main.yaml avec la PROFONDEUR changée, rien d'autre : le modèle est
# une tranche de 6 blocs du 7B, donc toutes les dimensions sont celles du 7B.
#   vid_dim 3072  <- blocks.0.ada.vid.attn_gate a la forme [3072]
#   txt_in_dim 5120 <- txt_in.weight a la forme [3072, 5120]
#   heads 24 x head_dim 128 = 3072 = vid_dim
#   num_layers 6  <- le checkpoint contient blocks.0 a blocks.5, et rien d'autre
__object__:
  path: projects.video_diffusion_sr.train
  name: VideoDiffusionTrainer

dit:
  model:
    __object__:
      path: "dit_7b.nadit"
      name: "NaDiT"
      args: "as_params"
    vid_in_channels: 33
    vid_out_channels: 16
    vid_dim: 3072
    vid_out_norm: fusedrms
    txt_in_dim: 5120
    txt_in_norm: fusedln
    txt_dim: ${.vid_dim}
    emb_dim: ${eval:'6 * ${.vid_dim}'}
    heads: 24
    head_dim: 128
    expand_ratio: 4
    norm: fusedrms
    norm_eps: 1.0e-05
    ada: single
    qk_bias: False
    qk_norm: fusedrms
    patch_size: [1, 2, 2]
    num_layers: 6
    mm_layers: 6
    mlp_type: swiglu
    msa_type: None
    block_type: ${eval:'${.num_layers} * ["mmdit_sr"]'}
    window: ${eval:'${.num_layers} * [(4,3,3)]'}
    window_method: ${eval:'${.num_layers} // 2 * ["720pwin_by_size_bysize","720pswin_by_size_bysize"]'}
    rope_type: mmrope3d
    rope_dim: 128
  compile: False
  gradient_checkpoint: True
  fsdp:
    sharding_strategy: _HYBRID_SHARD_ZERO2
"""

# Sélection d'architecture dans le dépôt amont : un ternaire d'une seule ligne.
# On l'étend au 1.4B. Le motif est distinctif ; s'il disparaît (mise à jour
# amont), on le DIT au lieu de patcher au hasard — le 3B et le 7B continuent de
# fonctionner sans ce correctif.
_SEEDVR2_NEEDLE = ("'./configs_7b' if \"7b\" in dit_model else './configs_3b'")
_SEEDVR2_PATCH = ("('./configs_1_4b' if (\"1.4b\" in dit_model.lower() or "
                  "\"6l\" in dit_model.lower()) else "
                  "'./configs_7b' if \"7b\" in dit_model else './configs_3b')")


def _seedvr2_enable_1_4b(repo: Path) -> bool:
    """Ajoute la config 1.4B et enseigne au dépôt à la sélectionner."""
    cfg = repo / "configs_1_4b"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "main.yaml").write_text(SEEDVR2_CFG_1_4B, encoding="utf-8")
    print(f"[OK] Config 1.4B ecrite : {cfg / 'main.yaml'}")

    target = repo / "src" / "core" / "model_configuration.py"
    if not target.is_file():
        print(f"[!] {target} introuvable — le 1.4B ne sera pas selectionnable.")
        return False
    src = target.read_text(encoding="utf-8")
    if "configs_1_4b" in src:
        print("[OK] Selection 1.4B deja en place (rien a faire).")
        return True
    if _SEEDVR2_NEEDLE not in src:
        print("[!] Le motif de selection d'architecture a change en amont : "
              "le 1.4B ne sera pas selectionnable.")
        print("    Les modeles 3B / 7B officiels restent utilisables.")
        return False
    target.write_text(src.replace(_SEEDVR2_NEEDLE, _SEEDVR2_PATCH, 1),
                      encoding="utf-8")
    print("[OK] Selection d'architecture etendue au 1.4B.")
    return True


def install_seedvr2():
    base = settings.ROOT / "tools_repo" / "seedvr2"
    repo = base / "repo"
    models = base / "models"
    models.mkdir(parents=True, exist_ok=True)

    ensure_torch_cuda()

    # 1) code d'inference (clone superficiel, ou mise a jour)
    if (repo / ".git").is_dir():
        print("Mise a jour du code d'inference SeedVR2…")
        sh(["git", "-C", str(repo), "fetch", "--depth", "1", "origin"])
        sh(["git", "-C", str(repo), "reset", "--hard", "origin/HEAD"])
    else:
        print(f"Recuperation du code d'inference ({SEEDVR2_GIT})…")
        repo.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--depth", "1", SEEDVR2_GIT, str(repo)])

    # 2) dependances — torch/torchvision sont deja gerees par ensure_torch_cuda,
    #    on ne les laisse SURTOUT pas etre reinstallees par pip (build CPU).
    reqs_file = repo / "requirements.txt"
    reqs = []
    if reqs_file.is_file():
        for line in reqs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            if name.lower() in ("torch", "torchvision", "torchaudio"):
                continue
            reqs.append(line)
    if reqs:
        print("Installation des dependances SeedVR2…")
        sh([sys.executable, "-m", "pip", "install", *reqs])

    # 3) poids du 1.4B (le VAE est recupere automatiquement au 1er lancement)
    print(f"\nTelechargement des poids 1.4B ({SEEDVR2_W_REPO})…")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=SEEDVR2_W_REPO, filename=SEEDVR2_DIT,
                        local_dir=str(models))
    print(f"[OK] {Path(p).name} ({Path(p).stat().st_size / 2**30:.2f} Go)")

    # 4) rendre le 1.4B selectionnable
    ok = _seedvr2_enable_1_4b(repo)

    pin_numpy()
    print("\n[OK] SeedVR2 installe (onglet Toolkit -> Restauration).")
    if ok:
        print("    Modele 1.4B pret. Le VAE (~0,5 Go) sera telecharge")
        print("    automatiquement au tout premier agrandissement.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "sam", "enhance", "upscale",
                                     "seedvr2"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()
    elif args.tool == "bg":
        install_bg()
    elif args.tool == "sam":
        install_sam()
    elif args.tool == "enhance":
        install_enhance()
    elif args.tool == "upscale":
        install_upscale()
    elif args.tool == "seedvr2":
        install_seedvr2()


if __name__ == "__main__":
    main()
