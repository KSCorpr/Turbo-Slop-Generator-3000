#!/usr/bin/env python3
"""SONDE MiniMax-H3 : est-ce que la vidéo est jouable sur cette machine ?

Ce script ne fait PAS partie de l'application et n'ajoute rien à l'interface.
Il répond à deux questions, et à elles seules, avant d'écrire la moindre ligne
d'interface :

  1. Le modèle tient-il dans la VRAM de cette carte ?
  2. La LoRA Turbo (4 pas) s'applique-t-elle ?

La deuxième est celle qui décide. sd.cpp sait appliquer les LoRA « lightx2v »
aux modèles vidéo — c'est documenté pour Wan 2.2, avec `--steps 4`. Mais ce
n'est PAS documenté pour MiniMax-H3, et sans la LoRA on reste au nombre de pas
de base : c'est la différence entre quelques minutes et une demi-heure par
clip.

Le jeu de poids choisi est le plus léger publié (Q2_K_M « pruned »), parce que
la seule chose qu'on veut savoir ici est si ça passe. Compter ~26 Go de
téléchargement.

    python scripts/try_minimax.py --check       # vérifie tout, ne télécharge rien
    python scripts/try_minimax.py --download    # télécharge le jeu léger
    python scripts/try_minimax.py --run         # lance les deux passes

Rien n'est écrit dans le catalogue de modèles : les fichiers vont dans
`models/`, et se suppriment depuis « Gestion & nettoyage » comme le reste.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atelier import hardware, settings  # noqa: E402
from atelier.engine import sdcpp  # noqa: E402

# --------------------------------------------------------------------------- #
#  Les poids — noms relevés dans docs/minimax_h3.md et sur les dépôts, pas
#  devinés. Le « pruned Q2_K_M » de ref2va est le plus petit jeu publié.
# --------------------------------------------------------------------------- #
WEIGHTS = [
    ("diffusion", "leejet/MiniMax-H3-GGUF",
     "minimax_h3_ref2va_pruned-Q2_K_M.gguf", 6.72),
    ("encodeur",  "leejet/MiniMax-H3-GGUF",
     "qwen3vl_32b_minimax_h3-Q2_K_M.gguf", 13.1),
    ("VAE vidéo", "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors", 5.21),
    ("VAE audio", "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_audio_vae_fp32.safetensors", 0.605),
]

# La LoRA Turbo : c'est elle qu'on teste. 4 pas au lieu du régime de base.
TURBO = ("LoRA Turbo", "lightx2v/Minimax-h3-Turbo",
         "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors", 1.0)

# Options communes, reprises TELLES QUELLES de docs/minimax_h3.md. On n'invente
# rien ici : c'est la ligne de la documentation, avec nos chemins.
#
# `--video-frames` suit la grille 17k+5 imposée par le modèle (5, 22, 39, 56…).
# 22 images à 24 fps ≈ 0,9 s : c'est un ESSAI, pas un film — on veut savoir si
# ça démarre et à quelle vitesse, pas produire un clip.
BASE_ARGS = [
    "-M", "vid_gen", "--cfg-scale", "1.0", "-v",
    "-W", "864", "-H", "480",
    "--diffusion-fa", "--offload-to-cpu", "--rng", "cpu",
    "--fps", "24", "--video-frames", "22",
]


# --------------------------------------------------------------------------- #
#  OÙ METTRE L'ENCODEUR — la question qui décide, et sa réponse dépend du
#  nombre de cartes.
#
#  Qwen3-VL 32B réclame 12 845 Mio pour son tampon de calcul. Une RTX 3060 en
#  a 12 288 en tout : carte vide, ça ne rentre pas. Mesuré, pas supposé.
#
#  `--offload-to-cpu` n'y suffit pas : il range les POIDS en RAM mais le calcul
#  revient sur le GPU. Il faut placer l'encodeur lui-même, via `--backend`.
#  Trois placements, du plus rapide au plus sûr :
#
#   1. RÉPARTI SUR PLUSIEURS CARTES — `te=cuda0&cuda1`. sd.cpp découpe les
#      blocs de l'encodeur en tranches proportionnelles à la mémoire libre de
#      chaque carte. Une 3060 (12 Go) et une 1080 Ti (11 Go) font 23 Go
#      cumulés : l'encodeur y tient, et il tourne sur GPU.
#   2. SUR LE PROCESSEUR — `te=cpu`. Marche partout, mais encoder un 32B sur
#      processeur coûte plusieurs minutes.
#   3. Le même, plus un budget VRAM pour le reste du graphe.
#
#  `--clip-on-cpu` ferait la même chose que (2) mais est déprécié en amont : il
#  ne fait plus que préfixer `te=cpu`.
# --------------------------------------------------------------------------- #
def placements(gpus) -> list[tuple[str, list[str]]]:
    """Placements à essayer, du plus rapide au plus sûr."""
    out: list[tuple[str, list[str]]] = []
    cuda = [g for g in gpus if not g.is_apple]
    if len(cuda) >= 2:
        total = sum(g.vram_gb for g in cuda)
        devices = "&".join(f"cuda{g.index}" for g in cuda)
        out.append((f"encodeur réparti sur {len(cuda)} cartes "
                    f"({total:.0f} Go cumulés)",
                    ["--backend", f"te={devices}"]))
    out.append(("encodeur sur le processeur", ["--backend", "te=cpu"]))
    out.append(("encodeur sur le processeur + budget VRAM",
                ["--backend", "te=cpu", "--max-vram", "-1"]))
    return out

# Signature d'un manque de mémoire dans la sortie du moteur : c'est ce qui
# déclenche la relance, et ce qui permet de nommer le vrai coupable.
_OOM = ("out of memory", "cudamalloc failed", "failed to allocate")

# Le modèle choisi est ref2va — REFERENCE-to-video : il part d'une image et
# doit la recevoir via `-r`. Le prompt la désigne par « <Picture 1> », comme
# dans l'exemple de la documentation. Sans référence, ce modèle-là n'a rien à
# quoi se raccrocher.
PROMPT = ("Use the subject from <Picture 1> as the main character, keeping its "
          "appearance and identity consistent. Cinematic slow orbit shot, "
          "natural light, smooth motion. Add calm ambient background music.")

# Taille du jeu complet, pour prévenir AVANT de lancer 26 Go de téléchargement.
NEEDED_GB = sum(w[3] for w in WEIGHTS) + TURBO[3]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _ram_gb() -> float:
    """RAM totale, sans dépendance : Windows par ctypes, Linux par /proc."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _S(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _S()
            st.dwLength = ctypes.sizeof(_S)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1e9
        import os
        return (os.sysconf("SC_PAGE_SIZE")
                * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
#  1. Vérifications préalables
# --------------------------------------------------------------------------- #
def check() -> bool:
    """Tout ce qui peut échouer AVANT de télécharger 26 Go.

    Volontairement SANS arrêt au premier échec : quand on prépare une machine,
    savoir d'un coup qu'il manque le moteur ET 10 Go de disque évite deux
    allers-retours. On rend le verdict à la fin.
    """
    ok = True
    log("── Vérifications ──────────────────────────────────────────")

    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        log("✗ binaire sd-cli introuvable → lancez install.bat")
        log("  (les options du moteur ne peuvent pas être vérifiées)")
        ok = False
    else:
        log(f"✓ moteur : {sd_cli}")
        # Le point qui échoue le plus probablement : le support MiniMax-H3 date
        # du 4 août 2026. Un binaire plus ancien ne connaît ni « vid_gen » ni
        # « --audio-vae », et le dira par un message d'usage illisible.
        opts = sdcpp.supported_options(sd_cli)
        missing = [f for f in ("--audio-vae", "--llm", "--video-frames",
                               "--fps", "--offload-to-cpu", "--diffusion-fa",
                               "--lora-model-dir") if f not in opts]
        if missing:
            log(f"✗ options inconnues du moteur : {', '.join(missing)}")
            log("  → moteur antérieur au 4 août 2026, lancez update-engine.bat")
            ok = False
        else:
            log("✓ le moteur connaît vid_gen, l'audio et les LoRA")

    # TOUTES les cartes, pas seulement la meilleure : la VRAM cumulée décide
    # du placement de l'encodeur, et n'afficher qu'une carte cache l'option la
    # plus rapide sur une machine qui en a deux.
    gpus = hardware.detect_gpus()
    if not gpus:
        log("✗ aucun GPU détecté — inutile d'essayer en CPU")
        ok = False
    else:
        total = sum(g.vram_gb for g in gpus)
        log(f"✓ {len(gpus)} carte(s), {total:.0f} Go de VRAM cumulée :")
        for g in gpus:
            log(f"    cuda{g.index}  {g.label()}")
        if len(gpus) >= 2 and total >= 14:
            log("  → l'encodeur peut être RÉPARTI sur les cartes : c'est le "
                "placement le plus rapide, essayé en premier.")
        else:
            log("  → l'encodeur (12,5 Gio) ne tient sur aucune carte seule : "
                "il tournera sur le processeur.")

    # L'encodeur tourne sur le processeur : ses 12,5 Gio sont désormais un
    # besoin de RAM, pas de VRAM. C'est devenu le vrai plancher.
    ram = _ram_gb()
    if ram:
        log(f"{'✓' if ram >= 24 else '✗'} RAM : {ram:.0f} Go "
            "(l'encodeur en réclame ~13 à lui seul)")
        ok = ok and ram >= 24
    else:
        log("• RAM : indéterminée — il en faut au moins 24 Go")

    free = shutil.disk_usage(settings.MODELS_DIR).free / 1e9
    log(f"{'✓' if free > NEEDED_GB + 5 else '✗'} disque libre : "
        f"{free:.0f} Go (il en faut ~{NEEDED_GB:.0f})")
    ok = ok and free > NEEDED_GB + 5

    log()
    log("Poids à télécharger :")
    for role, repo, name, gb in WEIGHTS + [TURBO]:
        dest = settings.model_repo_dir(repo) / name
        mark = "déjà là" if dest.is_file() else f"{gb:.1f} Go"
        log(f"  {role:<12} {name:<52} {mark}")
    return ok


# --------------------------------------------------------------------------- #
#  2. Téléchargement
# --------------------------------------------------------------------------- #
def fetch(items) -> dict[str, Path]:
    settings.configure_hf_env()
    from huggingface_hub import hf_hub_download

    out = {}
    for role, repo, name, gb in items:
        local_dir = settings.model_repo_dir(repo)
        dest = local_dir / name
        if dest.is_file():
            log(f"  ✓ déjà présent : {role}")
        else:
            log(f"  ↓ {role} ({gb:.1f} Go) : {repo}/{name}")
            dest = Path(hf_hub_download(repo_id=repo, filename=name,
                                        local_dir=str(local_dir)))
        out[role] = dest
    return out


# --------------------------------------------------------------------------- #
#  3. Les deux passes
# --------------------------------------------------------------------------- #
def pick_reference(explicit: str | None) -> Path | None:
    """L'image de référence : celle donnée, sinon la dernière image produite.

    Reprendre une de vos propres sorties évite d'avoir à en chercher une, et
    rend l'essai plus parlant : on voit tout de suite si le modèle garde le
    sujet ou l'invente.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    imgs = [p for p in settings.OUTPUT_DIR.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    return max(imgs, key=lambda p: p.stat().st_mtime) if imgs else None


def build_cmd(sd_cli: Path, paths: dict[str, Path], turbo: bool,
              out_file: Path, ref: Path,
              placement: list[str] | None = None) -> list[str]:
    prompt = PROMPT
    extra: list[str] = []
    if turbo:
        # Syntaxe LoRA de sd.cpp : le nom SANS extension, dans le prompt, plus
        # le dossier où chercher. C'est la forme documentée pour Wan 2.2.
        lora = paths["LoRA Turbo"]
        prompt = f"{PROMPT}<lora:{lora.stem}:1>"
        extra = ["--lora-model-dir", str(lora.parent), "--steps", "4"]
    return [
        str(sd_cli),
        "--diffusion-model", str(paths["diffusion"]),
        "--vae", str(paths["VAE vidéo"]),
        "--audio-vae", str(paths["VAE audio"]),
        "--llm", str(paths["encodeur"]),
        "-p", prompt,
        "-r", str(ref),
        *BASE_ARGS, *extra,
        *(placement or []),
        "-o", str(out_file),
    ]


def out_path(turbo: bool) -> Path:
    """Un fichier par passe : les comparer côte à côte est tout l'intérêt de
    l'essai, et un nom commun ferait écraser le premier par le second."""
    return settings.OUTPUT_DIR / (
        f"minimax_test_{'turbo' if turbo else 'base'}.webm")


def _launch(cmd: list[str]) -> tuple[int, float, bool]:
    """Lance le moteur en RELAYANT sa sortie, et dit si elle parle de mémoire.

    On relaie ligne à ligne plutôt que de capturer en silence : une passe dure
    des minutes, et regarder une fenêtre muette n'apprend rien. Mais on garde
    de quoi reconnaître un manque de mémoire, pour relancer avec un budget au
    lieu de rendre un simple « échec ».
    """
    oom = False
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        low = line.lower()
        if any(sig in low for sig in _OOM):
            oom = True
    proc.wait()
    return proc.returncode, time.time() - start, oom


def run_pass(sd_cli: Path, paths: dict[str, Path], turbo: bool,
             ref: Path, ladder: list[tuple[str, list[str]]]) -> str | None:
    """Essaie les placements jusqu'à ce que l'un passe. Rend celui qui a marché.

    On descend l'échelle SEULEMENT sur un manque de mémoire : sur une autre
    erreur, réessayer ailleurs coûte plusieurs minutes pour rien et brouille le
    diagnostic.
    """
    name = "AVEC la LoRA Turbo (4 pas)" if turbo else "SANS LoRA (pas de base)"
    out_file = out_path(turbo)

    for i, (label, extra) in enumerate(ladder, 1):
        cmd = build_cmd(sd_cli, paths, turbo, out_file, ref, placement=extra)
        log()
        log(f"── {name} · placement {i}/{len(ladder)} : {label} ──")
        log(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        log()

        code, took, oom = _launch(cmd)

        if code == 0 and out_file.is_file():
            log(f"✓ {out_file.name} — {out_file.stat().st_size / 1e6:.1f} Mo "
                f"en {took / 60:.1f} min — placement : {label}")
            return label
        if code == 0:
            log(f"✗ terminé sans erreur mais aucun fichier produit ({took:.0f} s)")
            return None

        log(f"✗ échec (code {code}) après {took / 60:.1f} min")
        if not oom:
            log("  Ce n'est PAS un manque de mémoire : changer de placement "
                "n'y changerait rien. Le message du moteur ci-dessus est la "
                "vraie piste.")
            return None
        if i == len(ladder):
            log("  À court de mémoire sur TOUS les placements possibles.")
            log("  → cette machine ne suffit pas pour ce modèle, même au plus "
                "petit quant publié.")
            return None
        log("  Manque de mémoire → on essaie le placement suivant…")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="vérifie moteur, GPU et disque, sans rien télécharger")
    ap.add_argument("--download", action="store_true",
                    help="télécharge le jeu léger (~26 Go)")
    ap.add_argument("--run", action="store_true",
                    help="lance les deux passes (télécharge si nécessaire)")
    ap.add_argument("--no-turbo", action="store_true",
                    help="ne faire que la passe sans LoRA")
    ap.add_argument("--ref", default=None,
                    help="image de référence (défaut : la dernière de outputs/)")
    args = ap.parse_args()
    if not (args.check or args.download or args.run):
        args.check = True

    settings.ensure_dirs()
    if not check():
        log()
        log("→ Corrigez les points ✗ ci-dessus avant d'aller plus loin.")
        return 1
    if args.check and not (args.download or args.run):
        log()
        log("→ Tout est en place. `--download` pour récupérer les poids.")
        return 0

    items = WEIGHTS if args.no_turbo else WEIGHTS + [TURBO]
    log()
    log("── Téléchargement ─────────────────────────────────────────")
    paths = fetch(items)
    if args.download and not args.run:
        log()
        log("→ Poids en place. `--run` pour lancer les deux passes.")
        return 0

    sd_cli = settings.find_sd_cli()
    ref = pick_reference(args.ref)
    if ref is None:
        log()
        log("✗ aucune image de référence. Le modèle ref2va en exige une :")
        log("  générez une image (elle ira dans outputs/), ou passez --ref "
            "chemin/vers/image.png")
        return 1
    log(f"  référence : {ref}")
    ladder = placements(hardware.detect_gpus())
    base_ok = run_pass(sd_cli, paths, turbo=False, ref=ref, ladder=ladder)
    turbo_ok = None if args.no_turbo else run_pass(
        sd_cli, paths, turbo=True, ref=ref,
        # Une fois un placement trouvé, on ne recommence pas l'escalade : la
        # seconde passe teste la LoRA, pas la mémoire.
        ladder=[(l, e) for l, e in ladder if l == base_ok] or ladder)

    log()
    log("── Verdict ────────────────────────────────────────────────")
    log(f"  modèle sur cette machine : {'OUI' if base_ok else 'NON'}")
    if base_ok:
        log(f"  placement retenu         : {base_ok}")
    if not args.no_turbo:
        log(f"  LoRA Turbo appliquée     : {'OUI' if turbo_ok else 'NON'}")
        if base_ok and not turbo_ok:
            log("  → sans la LoRA, comptez le régime de pas complet : c'est "
                "jouable mais lent.")
    # Ne rien annoncer qui n'existe pas : la version précédente invitait à
    # comparer deux fichiers alors que les deux passes avaient échoué.
    made = [p for p in (out_path(False), out_path(True)) if p.is_file()]
    if made:
        log("  Produit : " + ", ".join(p.name for p in made) + " (outputs/)")
    else:
        log("  Aucun fichier produit.")
    return 0 if base_ok else 1


if __name__ == "__main__":
    sys.exit(main())
