"""Banc d'essai reproductible des placements GPU et des variantes Krea 2."""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import diagnostics, hardware, registry, settings
from .engine import generate


BENCHMARK_SEED = 424242
BENCHMARK_PROMPT = (
    "editorial photograph of a red enamel coffee maker on a blue workbench, "
    "soft window light, legible label TEST 42, realistic materials")


@dataclass(frozen=True)
class Placement:
    key: str
    label: str
    prefs_patch: dict


def _selected_gpu(prefs: dict, gpus: tuple[hardware.Gpu, ...]) -> hardware.Gpu | None:
    wanted = prefs.get("gpu_index")
    selected = next((g for g in gpus if g.index == wanted), None)
    return selected or (max(gpus, key=lambda g: g.vram_gb) if gpus else None)


def placement_candidates(prefs: dict | None = None,
                         gpus: tuple[hardware.Gpu, ...] | None = None
                         ) -> list[Placement]:
    """Scénarios réellement comparables sur la machine, sans Auto-Fit."""
    prefs = prefs or settings.load_prefs()
    gpus = gpus if gpus is not None else hardware.detect_gpus()
    main = _selected_gpu(prefs, gpus)
    if main is None:
        return []
    base_flags = hardware.auto_profile(main.index).flags()
    base_flags["vae_tiling"] = True
    staged = {**base_flags, "offload_to_cpu": True,
              "clip_on_cpu": False, "vae_on_cpu": False}
    resident = {**base_flags, "offload_to_cpu": False,
                "clip_on_cpu": False, "vae_on_cpu": False}
    common = {"auto_optimize": False, "gpu_index": main.index,
              "auto_fit": False, "split_mode": "layer"}
    out = [Placement(
        "single-staged", f"{main.name} seule · poids en RAM",
        {**common, "encoder_gpu_index": None, "params_backend": "",
         "flags": staged})]
    secondary = next((g for g in gpus if g.index != main.index), None)
    if secondary is not None:
        mapping = (f"diffusion=cuda{main.index},vae=cuda{main.index},"
                   f"te=cuda{secondary.index}")
        out.extend([
            Placement(
                "dual-resident",
                f"{main.name} diffusion · {secondary.name} encodeur résident",
                {**common, "encoder_gpu_index": secondary.index,
                 "params_backend": mapping, "flags": resident}),
            Placement(
                "dual-staged",
                f"{main.name} diffusion · {secondary.name} calcul · poids en RAM",
                {**common, "encoder_gpu_index": secondary.index,
                 "params_backend": "*=cpu", "flags": staged}),
        ])
    return out


def _merge_prefs(base: dict, patch: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if key == "flags" and isinstance(value, dict):
            merged["flags"] = {**merged.get("flags", {}), **value}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _benchmark_prefs(base: dict, patch: dict) -> dict:
    merged = _merge_prefs(base, patch)
    # Un benchmark mesure le placement, pas le cache entre pas.
    merged["cache_mode"] = ""
    merged["cache_option"] = ""
    merged["cache_by_model"] = {}
    merged["max_vram"] = ""
    merged["stream_layers"] = False
    return merged


def _monitor_peak(stop: threading.Event, peak: dict[int, float]) -> None:
    while not stop.wait(0.2):
        for idx, used in hardware.used_vram_gb().items():
            peak[idx] = max(peak.get(idx, 0.0), used)


def _run_case(model_id: str, prefs: dict, label: str,
              log: Callable[[str], None] | None = None) -> dict:
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        return {"label": label, "ok": False, "error": f"Modèle inconnu : {model_id}"}
    if not registry.model_is_ready(model):
        missing = [c.role for c in registry.missing_components(model)]
        return {"label": label, "ok": False,
                "error": "Fichiers manquants : " + ", ".join(missing)}
    d = model.defaults
    peak = hardware.used_vram_gb()
    stop = threading.Event()
    watcher = threading.Thread(target=_monitor_peak, args=(stop, peak), daemon=True)
    watcher.start()
    started = time.perf_counter()
    if log:
        log(f"[benchmark] {label}")
    try:
        outputs = generate.generate(
            model_id=model_id, prompt=BENCHMARK_PROMPT, negative="",
            steps=min(4, int(d.get("steps", 4))), cfg_scale=float(d.get("cfg_scale", 1.0)),
            width=512, height=512, seed=BENCHMARK_SEED, batch_count=1,
            sampler=d.get("sampler", "euler"), schedule=d.get("scheduler", "auto"),
            flow_shift=float(d.get("flow_shift", 0.0) or 0.0),
            save_prompt=False, prefs_override=prefs, log=log)
        elapsed = time.perf_counter() - started
        return {"label": label, "ok": True, "seconds": round(elapsed, 3),
                "peak_used_vram_gb": peak,
                "outputs": [str(p) for p in outputs]}
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "ok": False,
                "seconds": round(time.perf_counter() - started, 3),
                "peak_used_vram_gb": peak, "error": str(exc)}
    finally:
        stop.set()
        watcher.join(timeout=1)


def _pick_model(prefs: dict) -> str | None:
    models = {m.id: m for m in registry.load_base_models(prefs)}
    for wanted in ("krea2-turbo", "flux2-klein-9b", "krea2-raw"):
        model = models.get(wanted)
        if model and registry.model_is_ready(model):
            return wanted
    return next((m.id for m in models.values() if registry.model_is_ready(m)), None)


def _write_report(prefix: str, payload: dict) -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = settings.OUTPUT_DIR / f"{prefix}-{stamp}.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return dest


def run_hardware_benchmark(model_id: str | None = None,
                           log: Callable[[str], None] | None = None) -> Path:
    base = settings.load_prefs()
    selected = model_id or _pick_model(base)
    if not selected:
        raise RuntimeError("Aucun modèle installé : téléchargez Krea 2 Turbo ou Flux.2.")
    modes = placement_candidates(base)
    if not modes:
        raise RuntimeError("Aucun GPU NVIDIA détecté.")
    results = []
    for mode in modes:
        prefs = _benchmark_prefs(base, mode.prefs_patch)
        result = _run_case(selected, prefs, mode.label, log)
        result.update({"key": mode.key, "prefs_patch": mode.prefs_patch})
        results.append(result)
    successes = [r for r in results if r.get("ok")]
    winner = min(successes, key=lambda r: r["seconds"]) if successes else None
    payload = {
        "schema": 1, "kind": "hardware-placement", "seed": BENCHMARK_SEED,
        "model_id": selected, "system": diagnostics.system_report(),
        "results": results,
        "recommended_mode": winner.get("key") if winner else None,
        "recommended_prefs_patch": winner.get("prefs_patch") if winner else None,
    }
    return _write_report("hardware-benchmark", payload)


def compare_krea_variants(log: Callable[[str], None] | None = None) -> Path:
    base = settings.load_prefs()
    results = []
    for model_id, label in (("krea2-turbo", "Krea 2 Turbo GGUF"),
                            ("krea2-turbo-int8", "Krea 2 Turbo INT8 ConvRot")):
        results.append({"model_id": model_id,
                        **_run_case(model_id, base, label, log)})
    successes = [r for r in results if r.get("ok")]
    fastest = min(successes, key=lambda r: r["seconds"]) if successes else None
    payload = {
        "schema": 1, "kind": "krea-quant-comparison", "seed": BENCHMARK_SEED,
        "system": diagnostics.system_report(), "results": results,
        "fastest_model": fastest.get("model_id") if fastest else None,
        "quality_review_required": True,
    }
    return _write_report("krea-gguf-vs-int8", payload)


def apply_recommendation(report_path: str | Path) -> str:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    patch = data.get("recommended_prefs_patch")
    if not isinstance(patch, dict) or not patch:
        raise RuntimeError("Ce rapport ne contient aucun profil valide à appliquer.")
    prefs = _merge_prefs(settings.load_prefs(), patch)
    settings.save_prefs(prefs)
    return str(data.get("recommended_mode") or "profil mesuré")
