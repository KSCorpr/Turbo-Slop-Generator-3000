"""Détection matérielle (GPU NVIDIA + RAM) et profils d'optimisation.

Déduit automatiquement les bons réglages pour stable-diffusion.cpp selon la
génération de la carte (Pascal -> Blackwell) et la mémoire disponible.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from .i18n import t


# --------------------------------------------------------------------------- #
#  Détection
# --------------------------------------------------------------------------- #
@dataclass
class Gpu:
    index: int
    name: str
    vram_gb: float
    arch: str          # pascal | turing | ampere | ada | blackwell | unknown
    tensor_cores: bool


def _arch_from_name(name: str) -> tuple[str, bool]:
    """Déduit l'architecture et la présence de tensor cores depuis le nom."""
    n = name.upper()
    # RTX 50xx
    if re.search(r"RTX\s?50\d\d", n):
        return "blackwell", True
    # RTX 40xx
    if re.search(r"RTX\s?40\d\d", n):
        return "ada", True
    # RTX 30xx
    if re.search(r"RTX\s?30\d\d", n):
        return "ampere", True
    # RTX 20xx / TITAN RTX / Quadro RTX
    if re.search(r"RTX\s?20\d\d", n) or "TITAN RTX" in n:
        return "turing", True
    # GTX 16xx (Turing sans tensor cores grand public)
    if re.search(r"GTX\s?16\d\d", n):
        return "turing", False
    # GTX 10xx / TITAN X(p)
    if re.search(r"GTX\s?10\d\d", n) or "TITAN X" in n:
        return "pascal", False
    # Datacenter récents
    if any(x in n for x in ("H100", "H200", "B100", "B200", "GB200")):
        return "blackwell", True
    if any(x in n for x in ("A100", "A40", "A6000", "A5000", "A4000")):
        return "ampere", True
    if any(x in n for x in ("L40", "L4", "RTX 6000 ADA", "RTX 5000 ADA")):
        return "ada", True
    return "unknown", True


@lru_cache(maxsize=1)
def detect_gpus() -> tuple[Gpu, ...]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus: list[Gpu] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            vram = float(parts[2]) / 1024.0  # Mio -> Gio
        except ValueError:
            continue
        arch, tc = _arch_from_name(parts[1])
        gpus.append(Gpu(idx, parts[1], round(vram, 1), arch, tc))
    return tuple(gpus)


@lru_cache(maxsize=1)
def detect_ram_gb() -> float:
    """RAM système totale en Gio (Windows + Linux, sans dépendance externe)."""
    import platform
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / (1024 ** 3), 1)
        # Linux / autres POSIX
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# --------------------------------------------------------------------------- #
#  Profil d'optimisation
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    gpu: Gpu | None
    ram_gb: float
    quant: str                 # quant recommandé pour le modèle de diffusion
    enc_quant: str             # quant recommandé pour l'encodeur de texte
    diffusion_fa: bool
    offload_to_cpu: bool
    vae_tiling: bool
    clip_on_cpu: bool
    vae_on_cpu: bool
    notes: list[str] = field(default_factory=list)

    def flags(self) -> dict[str, bool]:
        return {
            "diffusion_fa": self.diffusion_fa,
            "offload_to_cpu": self.offload_to_cpu,
            "vae_tiling": self.vae_tiling,
            "clip_on_cpu": self.clip_on_cpu,
            "vae_on_cpu": self.vae_on_cpu,
        }


def _quant_for_vram(vram: float) -> str:
    if vram < 8:
        return "Q4_K_S"
    if vram < 12:
        return "Q4_K_M"
    if vram < 16:
        return "Q5_K_M"
    if vram < 24:
        return "Q6_K"
    return "Q8_0"


def _enc_quant_for_ram(ram: float) -> str:
    if ram < 16:
        return "Q4_K_M"
    if ram < 32:
        return "Q6_K"
    return "Q8_0"


# Échelle des quantifications, de la plus légère (rapide) à la plus lourde (qualité).
QUANT_LADDER = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
                "Q6_K", "Q8_0"]


def _shift_quant(q: str, delta: int) -> str:
    """Décale une quant de `delta` crans dans l'échelle (borné)."""
    try:
        i = QUANT_LADDER.index(q)
    except ValueError:
        return q
    return QUANT_LADDER[max(0, min(len(QUANT_LADDER) - 1, i + delta))]


# Générations RTX grand public. `bias` décale la quant : < 0 privilégie la
# VITESSE (cartes anciennes / VRAM serrée), > 0 la QUALITÉ (cartes récentes).
GENERATIONS: dict[str, dict] = {
    "gtx10": {"label": "GTX 10xx (Pascal)", "arch": "pascal", "bias": -1,
              "typical_vram": 8.0,
              "note": "Pascal (GTX 10xx / 1080 Ti) : pas de tensor cores, "
                      "flash-attention désactivé (peu efficace). Quant légère "
                      "pour compenser ; encodeur déchargé en RAM."},
    "rtx20": {"label": "RTX 20xx (Turing)", "arch": "turing", "bias": 0,
              "typical_vram": 8.0,
              "note": "Turing : flash-attention OK, pas d'accélération fp8 "
                      "(sd.cpp calcule en fp16). VRAM souvent serrée → quant "
                      "légère pour rester rapide."},
    "rtx30": {"label": "RTX 30xx (Ampere)", "arch": "ampere", "bias": 0,
              "typical_vram": 12.0,
              "note": "Ampere : bf16 natif, bon équilibre. Quant selon la VRAM."},
    "rtx40": {"label": "RTX 40xx (Ada)", "arch": "ada", "bias": 1,
              "typical_vram": 16.0,
              "note": "Ada : très rapide, grande marge VRAM → on monte d'un cran "
                      "de qualité."},
    "rtx50": {"label": "RTX 50xx (Blackwell)", "arch": "blackwell", "bias": 1,
              "typical_vram": 16.0,
              "note": "Blackwell : architecture récente + grosse VRAM → qualité "
                      "élevée."},
}


def generation_profile(gen_key: str, vram_gb: float | None = None,
                       ram_gb: float | None = None) -> Profile:
    """Profil d'optimisation CURATÉ pour une génération de carte RTX.

    S'appuie sur la VRAM réelle (si fournie) pour la quant et les flags mémoire,
    et applique un léger biais qualité/vitesse propre à la génération.
    """
    spec = GENERATIONS.get(gen_key) or GENERATIONS["rtx30"]
    vram = vram_gb if (vram_gb and vram_gb > 0) else spec["typical_vram"]
    ram = ram_gb if (ram_gb and ram_gb > 0) else detect_ram_gb()
    quant = _shift_quant(_quant_for_vram(vram), spec["bias"])
    enc_quant = _enc_quant_for_ram(ram or 16.0)
    return Profile(
        gpu=None, ram_gb=ram or 0.0, quant=quant, enc_quant=enc_quant,
        # Flash-attention : à partir de Turing. Désactivé sur Pascal (GTX 10xx).
        diffusion_fa=(spec.get("arch") != "pascal"),
        offload_to_cpu=(vram < 16),
        vae_tiling=(vram <= 12),
        clip_on_cpu=(vram < 8),
        vae_on_cpu=(vram < 6),
        notes=[t(spec["note"]),
               t("VRAM {vram} Go → diffusion {quant}, encodeur {enc}.").format(
                   vram=f"{vram:.0f}", quant=quant, enc=enc_quant)],
    )


def auto_profile(gpu_index: int | None = None) -> Profile:
    """Construit un profil d'optimisation à partir du matériel détecté."""
    gpus = detect_gpus()
    ram = detect_ram_gb()
    notes: list[str] = []

    gpu: Gpu | None = None
    if gpus:
        if gpu_index is not None:
            gpu = next((g for g in gpus if g.index == gpu_index), None)
        if gpu is None:
            gpu = max(gpus, key=lambda g: g.vram_gb)  # par défaut : la plus grosse
        if len(gpus) > 1:
            notes.append(
                t("{n} GPU détectés — calcul épinglé sur #{idx} ({name}). "
                  "Modifiable dans Réglages.").format(
                    n=len(gpus), idx=gpu.index, name=gpu.name))

    if gpu is None:
        notes.append(t("Aucun GPU NVIDIA détecté : mode CPU (très lent). "
                       "Vérifiez les pilotes / nvidia-smi."))
        return Profile(None, ram, "Q4_K_M", "Q4_K_M",
                       diffusion_fa=False, offload_to_cpu=True, vae_tiling=True,
                       clip_on_cpu=True, vae_on_cpu=True, notes=notes)

    vram = gpu.vram_gb
    quant = _quant_for_vram(vram)
    enc_quant = _enc_quant_for_ram(ram)

    # Flash attention : à partir de Turing (RTX 20xx). Désactivé sur Pascal.
    fa = gpu.arch in ("turing", "ampere", "ada", "blackwell")
    if gpu.arch == "pascal":
        notes.append(t("Carte Pascal (GTX 10xx) : flash-attention désactivé "
                       "(peu efficace), génération plus lente."))

    profile = Profile(
        gpu=gpu, ram_gb=ram, quant=quant, enc_quant=enc_quant,
        diffusion_fa=fa,
        offload_to_cpu=(vram < 16),
        vae_tiling=(vram <= 12),
        clip_on_cpu=(vram < 8),
        vae_on_cpu=(vram < 6),
        notes=notes,
    )

    notes.append(t("VRAM {vram} Go ({arch}) -> diffusion en {quant}.").format(
        vram=f"{vram:.0f}", arch=gpu.arch, quant=quant))
    if ram:
        notes.append(t("RAM {ram} Go -> encodeur de texte en {enc} "
                       "(déchargé en RAM, sans coût VRAM).").format(
                        ram=f"{ram:.0f}", enc=enc_quant))
    if vram < 10:
        notes.append(t("VRAM serrée : préférez des résolutions ≤ 768 px et une "
                       "quantification plus basse (la génération sera plus "
                       "lente)."))
    return profile


def summary_text() -> str:
    """Petit résumé lisible du matériel détecté (pour l'UI)."""
    gpus = detect_gpus()
    ram = detect_ram_gb()
    if not gpus:
        return t("⚠️ Aucun GPU NVIDIA détecté · RAM {ram} Go").format(
            ram=f"{ram:.0f}")
    lines = [t("RAM système : **{ram} Go**").format(ram=f"{ram:.0f}"), "",
             t("**GPU détectés :**")]
    for g in gpus:
        tc = t("tensor cores") if g.tensor_cores else t("sans tensor cores")
        lines.append(f"- #{g.index} — {g.name} · {g.vram_gb:.0f} Go · "
                     f"{g.arch} ({tc})")
    return "\n".join(lines)
