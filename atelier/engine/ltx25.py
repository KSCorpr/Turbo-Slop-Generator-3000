"""LTX 2.5 dev via sd.cpp, with the convolutional VAE and dedicated encoder."""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from .. import settings
from . import sdcpp, video
from .local_jobs import LocalJob

JOB = LocalJob()
REPO = "vantagewithai/LTX-2.5-GGUF"
OFFICIAL = "Lightricks/LTX-2.5"
DIFFUSIONS = {
    "Q3_K_M": "dev/ltx-2.5-22b-dev-transformer-Q3_K_M.gguf",
    "Q4_K_M": "dev/ltx-2.5-22b-dev-transformer-Q4_K_M.gguf",
}
ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
ENCODER_QUANT = "text_encoders/gemma4-12b-with-proj-ltx-2.5-Q4_0.gguf"
VAE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
SIZES = ((512, 288), (768, 448), (960, 544))
FRAMES = (33, 65, 97, 121)


def weights(quant="Q3_K_M"):
    if quant not in DIFFUSIONS:
        raise ValueError("Choose a listed LTX quantization.")
    return (settings.model_repo_dir(REPO) / DIFFUSIONS[quant],
            settings.model_repo_dir(OFFICIAL) / ENCODER_QUANT,
            settings.model_repo_dir(OFFICIAL) / VAE)


def ready(quant="Q3_K_M"):
    return all(p.is_file() and p.stat().st_size for p in weights(quant))


def build_command(cli, paths, prompt, output, *, size=(512, 288), frames=33,
                  seed=-1, first=None, last=None):
    if not prompt.strip():
        raise ValueError("Describe the scene and its movement first.")
    if tuple(size) not in SIZES or frames not in FRAMES:
        raise ValueError("Choose a listed LTX resolution and duration.")
    if last and not first:
        raise ValueError("A last frame also requires a first frame.")
    diffusion, encoder, vae = paths
    if "ltx-2.5-video-vae-conv" not in Path(vae).name:
        raise ValueError("LTX 2.5 in sd.cpp requires the convolutional VAE.")
    sdcpp._require(*paths, first, last)
    known = sdcpp.supported_options(cli)
    needed = {"--video-frames", "--auto-fit", "--max-vram", "--llm"}
    if not needed <= known or (last and "--end-img" not in known):
        raise RuntimeError("Update sd.cpp from System before using LTX 2.5.")
    req = sdcpp.GenRequest(auto_fit=True, max_vram=sdcpp.max_vram_arg("auto"),
                          flags={"diffusion_fa": "--diffusion-fa" in known,
                                 "vae_tiling": "--vae-tiling" in known})
    command = [str(cli), "-M", "vid_gen", "--diffusion-model", str(diffusion),
               "--llm", str(encoder), "--vae", str(vae), "-p", prompt,
               "--steps", "20", "--cfg-scale", "3.0", "--sampling-method", "euler",
               "-W", str(size[0]), "-H", str(size[1]), "--video-frames", str(frames),
               "--fps", "24", "-s", str(seed), *sdcpp.memory_args(cli, req)]
    if first:
        command += ["-i", str(first)]
    if last:
        command += ["--end-img", str(last)]
    return command + ["-o", str(output), "-v"]


def generate(prompt, quant="Q3_K_M", size="512x288", frames=33, seed=-1,
             first=None, last=None, log=print):
    cli = settings.find_sd_cli()
    if cli is None:
        raise RuntimeError("Install or update sd.cpp from System first.")
    seed = secrets.randbelow(2**31) if int(seed) < 0 else int(seed)
    output = sdcpp.unique_output("ltx-2.5", "avi")
    command = build_command(cli, weights(quant), prompt, output,
                            size=tuple(map(int, size.split("x"))), frames=int(frames),
                            seed=int(seed), first=first, last=last)
    env = settings.child_env(settings.generation_gpu_index(settings.load_prefs()))
    with JOB.session():
        JOB.run(command, log, env=env)
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("LTX produced no video. See the generation log.")
        final = output.with_suffix(".mp4")
        JOB.run([video._ffmpeg_exe(log), "-y", "-nostdin", "-i", output,
                 "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", final], log)
        if not final.is_file() or not final.stat().st_size:
            raise RuntimeError(f"MP4 encoding failed; the AVI is preserved at {output}.")
        output.unlink()
        final.with_suffix(".json").write_text(json.dumps({
            "model": "LTX 2.5 dev", "quant": quant, "prompt": prompt,
            "resolution": size, "frames": frames, "seed": seed,
            "steps": 20, "cfg": 3.0, "fps": 24,
        }, indent=2), encoding="utf-8")
    return final
