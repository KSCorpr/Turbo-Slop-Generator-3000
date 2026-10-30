"""Pinned tiny autoencoders for sd.cpp previews or explicit fast final decoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import model_store, settings, transfers

MODES = (("Projection preview / original VAE (default)", "off"),
         ("TAE preview only / original VAE for final image", "preview"),
         ("TAE fast encoding & final decoding (reduced detail)", "full"))


@dataclass(frozen=True)
class TinyAutoencoder:
    key: str
    name: str
    repo: str
    filename: str
    revision: str
    size: int
    sha256: str
    families: tuple[str, ...]

    def path(self):
        return settings.model_repo_dir(self.repo) / self.filename


SPECS = {
    "taef2": TinyAutoencoder("taef2", "TAEF2 · Flux.2 / MOSS / Fibo 2", "madebyollin/taef2",
        "taef2.safetensors", "bd244ebfe4398c84fbf312a2f1c868c676b500e3",
        10_723_524, "701d31c0ecbef3b609fe236e2e3a65ffb17f3a94237acdd771e8ea3194df557a",
        ("flux2", "moss_image", "fibo2", "fibo2_turbo")),
    "taeqi2_1": TinyAutoencoder("taeqi2_1", "TAEQI2.1 · Qwen Image 2.1 RGBA", "madebyollin/taeqi2_1",
        "taeqi2_1.safetensors", "5d62f11fee0068b4df7bd699ddab685e32174010",
        30_591_264, "6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e", ("qwen21",)),
    # HF mirror verified byte-for-byte against madebyollin/taehv:
    # safetensors/taew2_1.safetensors, Git blob ec1a2be453a250aa406e912bb630c34463ba0b2f.
    "taew2_1": TinyAutoencoder("taew2_1", "TAEW2.1 · Krea 2 Turbo / Realism", "OzzyGT/taew2_1",
        "taew2_1.safetensors", "4f23bfa06fbb9bc3331d70db8c1dfc5eb2cd1441",
        22_642_902, "04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f",
        ("krea2", "krea2_realism")),
}


def for_model(model):
    return next((s for s in SPECS.values() if model and model.family in s.families), None)


def selected_mode(model_id, prefs=None):
    prefs = settings.load_prefs() if prefs is None else prefs
    return (prefs.get("taesd_by_model") or {}).get(model_id, "off")


def save_mode(model_id, mode):
    if mode not in {value for _, value in MODES}:
        raise ValueError("Choose a listed TAE mode.")
    prefs = settings.load_prefs()
    modes = dict(prefs.get("taesd_by_model") or {})
    modes[model_id] = mode
    prefs["taesd_by_model"] = modes
    settings.save_prefs(prefs)


def ready(spec):
    try:
        return spec.path().is_file() and spec.path().stat().st_size == spec.size
    except OSError:
        return False


def model_ready(model, prefs=None):
    from . import registry

    if model is None:
        return False
    mode = selected_mode(model.id, prefs)
    spec = for_model(model)
    if mode != "off" and (spec is None or not ready(spec)):
        return False
    return all(mode == "full" and c.role == "vae" for c in registry.missing_components(model))


def validate(spec):
    if not ready(spec):
        raise ValueError(f"Download TAESD / {spec.name} in Models first (Install / repair).")
    path = spec.path()
    if not model_store.known(path, spec.sha256):
        if model_store.digest(path) != spec.sha256:
            raise ValueError(f"{spec.name}: checksum mismatch. Use Install / repair in Models.")
        model_store.remember(path, spec.sha256)
    return path


def install(key, log=print):
    spec = SPECS[key]
    log(f"Downloading / verifying {spec.name} ({spec.size / 1e6:.2f} MB)…")
    transfers.hf_hub_download(repo_id=spec.repo, filename=spec.filename,
                              revision=spec.revision, local_dir=str(spec.path().parent))
    validate(spec)
    log("Tiny autoencoder installed. In Images → Encoder, VAE & local files, choose a TAE mode.")


def require(model, mode, sd_cli, vae_override=None):
    from . import registry
    from .engine import sdcpp

    if mode not in {value for _, value in MODES}:
        raise sdcpp.EngineError("Choose a listed TAE mode.")
    if mode == "off":
        return None
    spec = for_model(model)
    if spec is None:
        raise sdcpp.EngineError(f"{model.name}: no validated tiny autoencoder in this catalog.")
    known = sdcpp.supported_options(sd_cli)
    if "--taesd" not in known or (mode == "preview" and "--taesd-preview-only" not in known):
        raise sdcpp.EngineError("This TAE mode needs a newer sd.cpp engine. Run update.bat.")
    if vae_override:
        comp = next((c for c in model.components if c.role == "vae"), None)
        allowed = {registry.resolve_component_path(comp)} if comp else set()
        if comp:
            allowed.update(settings.model_repo_dir(v.repo) / v.template for v in comp.alternatives)
        if Path(vae_override).resolve() not in {p.resolve() for p in allowed if p}:
            raise sdcpp.EngineError("TAE needs the catalog VAE architecture. Select Automatic or a catalog "
                                    "VAE variant, or switch TAE off for a custom VAE.")
    try:
        return validate(spec)
    except (OSError, ValueError) as exc:
        raise sdcpp.EngineError(str(exc)) from exc


def info(model):
    spec = for_model(model)
    if spec is None:
        return "No validated TAE for this model. Keep the original VAE."
    return (f"{spec.name} · {spec.size / 1e6:.2f} MB. Download TAESD / {spec.name} in Models. "
            "Preview only keeps the original VAE for the final image. Full mode also changes encoding/decoding "
            "and may soften fine detail; it does not reduce diffusion steps.")
