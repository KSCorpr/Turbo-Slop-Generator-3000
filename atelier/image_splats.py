"""Pinned single-image splat models. No Torch import in the studio process."""
from __future__ import annotations

from dataclasses import dataclass
import json

from . import addons, hardware, model_store, settings, transfers

BACKENDS = {"triposplat": "TripoSplat", "idol": "Idol"}
SOURCES = {
    "triposplat": ("VAST-AI-Research/TripoSplat", "d8db9e018b413dd9c4a9fe22463781bf98e8e68d"),
    "idol": ("speridlabs/idol-sdk", "4702a7f00850e34b6e83a2170cc123c3e9570744"),
}
TRIPO_REV = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
IDOL_REV = "2731c2291fc65a797c5cb627a5e5526fc1490902"
DINO_REPO = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DINO_REV = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
FLUX_REPO = "black-forest-labs/FLUX.2-dev"
FLUX_REV = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
FLUX_SHA = "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"
ADAPTER_VERSION = 1


@dataclass(frozen=True)
class Artifact:
    key: str
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str = ""

    def path(self):
        return settings.model_repo_dir(self.repo) / self.filename


TRIPO = (
    Artifact("denoiser", "VAST-AI/TripoSplat", TRIPO_REV, "diffusion_models/triposplat_fp16.safetensors", 741106994, "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20"),
    Artifact("decoder", "VAST-AI/TripoSplat", TRIPO_REV, "vae/triposplat_vae_decoder_fp16.safetensors", 576148442, "ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8"),
    Artifact("dino", "VAST-AI/TripoSplat", TRIPO_REV, "clip_vision/dino_v3_vit_h.safetensors", 1681247696, "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53"),
    Artifact("flux", "VAST-AI/TripoSplat", TRIPO_REV, "vae/flux2-vae.safetensors", 336213556, FLUX_SHA),
    Artifact("bg", "VAST-AI/TripoSplat", TRIPO_REV, "background_removal/birefnet.safetensors", 444473596, "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"),
)
IDOL = (
    Artifact("config", "speridlabs/idol", IDOL_REV, "config.json", 1335),
    Artifact("denoiser", "speridlabs/idol", IDOL_REV, "denoiser.safetensors", 1482258308, "ded2cacf671bf3978ebc62729087abdbd318ef4752e6805b4c9fc8ffa51fe716"),
    Artifact("decoder", "speridlabs/idol", IDOL_REV, "decoder.safetensors", 203171696, "9da4d7676bfcadb1c3fdfbff796e6257dadf82d26ec05be7717c79b2b13a1c4e"),
    Artifact("decoder_gs", "speridlabs/idol", IDOL_REV, "decoder_gs.safetensors", 949083036, "3087eb8ee5763206a53e4078acf2a1bf382dab5147d407ec13a2cf39c04cc544"),
    Artifact("bg", "speridlabs/idol", IDOL_REV, "birefnet.safetensors", 444466856, "44bbc97b041f879975d0d6d538b8e66a25eb04d1f49aab40697544cdd97eaa8e"),
    Artifact("dino_config", DINO_REPO, DINO_REV, "config.json", 744),
    Artifact("dino", DINO_REPO, DINO_REV, "model.safetensors", 3362432800, "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"),
    Artifact("flux_config", FLUX_REPO, FLUX_REV, "vae/config.json", 770),
    Artifact("flux", FLUX_REPO, FLUX_REV, "vae/diffusion_pytorch_model.safetensors", 336213556, FLUX_SHA),
)


def artifacts(backend):
    if backend not in BACKENDS:
        raise ValueError("Choose TripoSplat or Idol.")
    return TRIPO if backend == "triposplat" else IDOL


def root():
    return addons.root("image-splats")


def selected_gpu_index():
    selected = settings.generation_gpu_index(settings.load_prefs())
    if selected is not None:
        return selected
    cards = hardware.detect_gpus()
    gpu = max(cards, key=lambda g: (g.vram_gb, g.tensor_cores)) if cards else None
    return gpu.index if gpu else None


def source(backend):
    artifacts(backend)
    return root() / "sources" / backend


def marker(backend):
    artifacts(backend)
    return root() / (backend + "-installed.json")


def identity(backend):
    return {"source": SOURCES[backend][1], "revisions": sorted({s.revision for s in artifacts(backend)}),
            "adapter": ADAPTER_VERSION}


def file_ready(spec):
    try:
        return spec.path().is_file() and spec.path().stat().st_size == spec.size
    except OSError:
        return False


def ready(backend):
    artifacts(backend)
    try:
        info = json.loads(marker(backend).read_text(encoding="utf-8"))
        return (info == identity(backend) and addons.python("image-splats").is_file()
                and (source(backend) / ("triposplat.py" if backend == "triposplat" else "idol/pipeline.py")).is_file()
                and all(file_ready(s) for s in artifacts(backend)))
    except (OSError, ValueError):
        return False


def validate(backend, check=lambda: None):
    paths = {}
    for spec in artifacts(backend):
        check()
        if not file_ready(spec):
            raise RuntimeError(f"Incomplete {BACKENDS[backend]} weights: {spec.filename}. Use Install / repair.")
        if spec.sha256 and not model_store.known(spec.path(), spec.sha256):
            if model_store.digest(spec.path(), check) != spec.sha256:
                raise RuntimeError(f"Checksum mismatch: {spec.filename}. Use Install / repair.")
            model_store.remember(spec.path(), spec.sha256)
        paths[spec.key] = str(spec.path())
    return paths


def check_encoder_access():
    """Check the actual pinned gated files before any multi-GB Idol download."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    for spec in (s for s in IDOL if s.key in {"dino", "flux"}):
        try:
            get_hf_file_metadata(hf_hub_url(spec.repo, spec.filename, revision=spec.revision))
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            raise RuntimeError("Idol needs Hugging Face access. Accept the terms at "
                               f"https://huggingface.co/{DINO_REPO} and https://huggingface.co/{FLUX_REPO}, "
                               "then set HF_TOKEN or log in with hf auth login and retry. "
                               "No large Idol weights were downloaded.") from exc


def download_models(backend, log=print):
    if backend == "idol":
        check_encoder_access()
    for spec in artifacts(backend):
        log(f"{BACKENDS[backend]}: {spec.filename} ({spec.size / 1e9:.2f} GB)")
        transfers.hf_hub_download(repo_id=spec.repo, filename=spec.filename, revision=spec.revision,
                                  local_dir=str(settings.model_repo_dir(spec.repo)))
    validate(backend)
