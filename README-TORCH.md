# Test7000 — the same application, everything through PyTorch

This branch answers a question you cannot settle by reading: **what would this
tool look like if every feature went through PyTorch and diffusers instead of
stable-diffusion.cpp?** Same tabs, same presets, same model ids, same output
folder. Only the engine underneath changes.

It is an experiment, and it is meant to be judged on measurements, not on
principle. What follows is what it actually costs.

---

## The short version

| | stable-diffusion.cpp | PyTorch / diffusers |
|---|---|---|
| Z-Image Turbo on disk | **6.6 GB** (Q8_0 GGUF) | **32.9 GB** (repo is fp32) |
| Install to add | nothing — a binary ships with the app | **several GB** of Python packages |
| Quantization | downloaded ready made | done at load time (bitsandbytes) |
| On a 12 GB card | Q8_0, resident | int8, modules take turns |
| Flux.2 Klein / Krea 2 | ungated GGUF mirrors | **gated** repos: licence + token |
| Samplers | 21 | 14 map; 7 fall back to Euler |
| ESRGAN GGUF upscalers | yes | no — use the modern upscalers |
| Image → 3D (trellis) | yes | **not ported** |
| Convert to GGUF | yes | meaningless here |

Nothing above is an opinion. The sizes come from the Hugging Face API, the
gating from `/api/models/<id>`, the pipeline classes from each repo's
`model_index.json` and from what diffusers 0.40.0 exports, and the sampler
count from constructing all 336 menu combinations against the real library.

---

## Install

```
setup-torch-engine.bat
```

Several gigabytes, once. Then **Settings → 🔧 Expert → Generation engine**, or
`set TURBOSLOP_BACKEND=torch` before `run.bat` if you would rather not touch
your preferences — the environment variable wins over everything, so you can
run two copies of the app side by side, one per engine, and compare.

`setup-torch-engine.bat --check` prints what is installed and what would
change, without installing anything.

### It moves the Toolkit's stack too

This is the one real risk of the branch, and it is deliberate.

The Toolkit add-ons run on `torch 2.4.1` and `diffusers 0.33.1`, and those two
pins hold each other up: 2.4.1 was chosen to cover Pascal through Ada, and its
`infer_schema` cannot read the `X | None` annotations that diffusers ≥ 0.35
uses. The three models in the catalog need **diffusers ≥ 0.36** — it is written
in each repo's `model_index.json`. No version satisfies both, so this branch
moves the whole stack up. Moving torch also dissolves the constraint that
justified the old pin, which is why it is possible at all.

**Pascal survives.** The `cu126` wheels still carry `5.0;6.0;7.0;…` (checked in
pytorch's own `.ci/manywheel/build_cuda.sh` through 2.12) and an `sm_60` cubin
loads on an `sm_61` card, so a GTX 1080 Ti keeps working. **Blackwell moves to
`cu128`**, which starts at `7.0`; the installer picks the index from the card it
detects, exactly as `_torch_setup.py` already did.

---

## What runs, and how

### Memory: the planner replaces the GGUF ladder

There is no quantized file to download here. A diffusers repo is published in
its own precision — often fp32 — and two things are decided at load time:

**Precision.** Three rungs, best to leanest: bf16 (2 bytes per parameter), int8
(1), NF4 (~0.5). The backend is bitsandbytes; it is the only one of the two
candidates publishing a `win_amd64` wheel (torchao does not).

**Placement.** Everything on the card; or modules taking turns
(`enable_model_cpu_offload`, one sub-model resident at a time, so the peak is
the largest rather than the sum); or weights streamed layer by layer
(`enable_sequential_cpu_offload`, fits in almost nothing, an order of magnitude
slower).

One rule ties them together: **keep the best precision that avoids layer
streaming.** That third mode is a safety net, not a working mode — dropping a
precision rung costs a little quality, staying there costs a factor of ten.

On the two cards this project targets, Z-Image Turbo lands on **int8 with
modules taking turns**: 6.2 GB for the transformer inside a budget of 8.5 GB
(11 GB card) or 9.5 GB (12 GB card). A 24 GB card keeps bf16 and stays
resident. An unknown size — a gated repo, so not measurable — never buys the
fast path: it takes per-module offload, which works wherever the fast path
would have.

### Samplers: 14 of 21, and the other 7 say so

The menu is sd.cpp's. diffusers does not cut the problem the same way, so the
translation is partial by nature and the only honest thing is to say which is
which — exact, approximate (with the difference written), or absent.

Absent falls back to Euler **with a line in the log**. A menu that accepts
everything and applies something else is worse than a short menu: you believe
you tried.

Two things the real library taught that reading the docs could not. Confronted
with diffusers 0.40, **110 of the 336 combinations raised on construction** —
unknown arguments, options that only exist on the diffusion schedulers, a DPM
variant rejecting its own default. The kwargs are now filtered against the
target class's actual signature and what gets dropped is logged; all 336 build,
and a test runs the whole matrix. And the scheduler is rebuilt with
`from_config(pipe.scheduler.config)`, never constructed fresh: a model's
published config carries its own training constants, and starting from class
defaults would silently replace them with someone else's.

### Modes are per model, not per app

sd.cpp offered text-to-image, img2img, editing and inpainting on everything.
diffusers publishes a class per model per mode, and they differ:

| | text→image | img2img | edit | inpaint | negative prompt |
|---|---|---|---|---|---|
| **Z-Image Turbo** | ✓ | ✓ | — | ✓ | ✓ |
| **Krea 2 Turbo** | ✓ | — | — | — | ✓ |
| **Flux.2 Klein 9B** | ✓ | — | ✓ | ✓ | — |

Krea 2 has no `Img2Img` class upstream, so "starting image" is meaningless on
it — the app says so and renders text-to-image rather than pretending. Flux.2
Klein takes `image=` on its main pipeline, which *is* the multi-reference edit
mode, and exposes no `negative_prompt` at all.

That table drives real behaviour: the outpaint tab asks the engine whether it
can inpaint **with a mask for this model** instead of asking a binary whether
it knows an option.

---

## What was ported, and what was not

**Outpaint needed nothing.** It already called `generate()` with a starting
image and a mask, so it changed engine when the seam did. That is what a single
point of passage is worth — and it is what made the other three visible.

**The HD pass** is rebuilt in the open: enlarge, then img2img on the loaded
pipeline. Same sequence as `--hires`, but visible, and the factor is no longer
bounded by the engine's ability to cut its own graph. "Latent" has nothing to
upscale outside sd.cpp, so it becomes Lanczos and says so; naming a modern
upscaler uses it as the base, which lets a lower denoise do the job.

**ADetailer** changes detector format, and cannot not: sd.cpp wants a converted
`.safetensors` because it will not execute a pickle, Ultralytics reads only the
original `.pt`. Same names, same source repo, different file. Detection is a
separate function from the redraw so "it found nothing" can be said, and each
patch is blended back through a feathered mask.

**GGUF upscalers refuse**, and name where to go. Those weights are read by
sd.cpp alone, they are RRDBNet from 2018, and the modern upscalers already run
on PyTorch through spandrel — which is why they were added in the first place.

**Image → 3D is not ported.** trellis.cpp is a second native engine with its own
binary and its own weights; porting it means the PyTorch TRELLIS, which needs
custom CUDA extensions compiled on the machine. That is a different project, not
a different backend, and pretending otherwise would have meant a tab that fails
at the first click.

**Convert to GGUF is left alone.** It converts weights *for* sd.cpp; it has no
meaning on this engine and still works if you switch back.

---

## Reading a result three days later

Both engines write into the same output folder, so the `.txt` beside each image
carries an `Engine:` line and the placement that produced it. Without it a
comparison is impossible to make after the fact — which is the whole point of
the branch.

The system report (**System → Manage → Diagnose**) gains a `torch_engine`
section: what is installed, what is missing by name, and which repositories are
actually on disk.
