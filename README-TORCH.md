# Test7000 — the same application, everything through PyTorch

This branch answers a question you cannot settle by reading: **what would this
tool look like if every feature went through PyTorch and diffusers instead of
stable-diffusion.cpp?** Same tabs, same presets, same model ids, same output
folder. Only the engine underneath changes.

> **This document was rewritten after a correction.** The first version claimed
> the PyTorch path meant re-downloading every model as a full diffusers
> repository — 33 GB for a model that weighs 6.6 on disk. That was wrong.
> **diffusers reads GGUF**, and it reads exactly the files this app already
> downloads for stable-diffusion.cpp. Everything below reflects that; the
> figures were re-checked against the real files, not remembered.

---

## The short version

| | stable-diffusion.cpp | PyTorch / diffusers |
|---|---|---|
| Model files | GGUF, quantized by VRAM | **the same files** |
| Extra disk to switch | — | **none** |
| Install to add | nothing — a binary ships with the app | **several GB** of Python packages |
| Z-Image Turbo on a 12 GB card | Q8_0, resident | Q8_0, **resident** |
| Z-Image Turbo on an 11 GB card | Q8_0, resident | Q8_0, modules take turns |
| Hugging Face token | never | once, for Flux.2 Klein's **config** (a few KB) |
| Krea 2 | full support | **does not run yet** — its transformer loads, the rest does not |
| Samplers | 21 | 14 map; 7 fall back to Euler |
| ESRGAN GGUF upscalers | yes | no — use the modern upscalers |
| Image → 3D (trellis) | yes | **not ported** |
| Convert to GGUF | yes | meaningless here |

Everything above was read from a source. Sizes and gating from the Hugging Face
API; pipeline classes from each repo's `model_index.json` and from what
diffusers 0.40.0 exports; tensor names from the **headers of the actual GGUF
files** the app installs; the sampler count from constructing all 336 menu
combinations against the real library.

---

## Why the GGUF path works — the checks

Four things had to be true, and each was verified rather than assumed:

1. **`GGUFQuantizationConfig` exists in diffusers 0.40.0**, and its dequantizer
   covers 23 GGML types including `Q2_K` → `Q8_0` — the whole ladder this app
   uses.
2. **`ZImageTransformer2DModel` and `Flux2Transformer2DModel` are registered**
   in `SINGLE_FILE_LOADABLE_CLASSES`, each with its conversion function.
   `Krea2Transformer2DModel` is not — so this branch writes that converter
   itself (see below).
3. **The tensor names match.** Read straight out of the GGUF headers:
   leejet's Z-Image file carries `context_refiner.0.attention.qkv.weight`,
   `cap_embedder.1.weight`; unsloth's Flux.2 Klein carries `double_blocks.0.
   img_attn.norm.query_norm.scale`, `final_layer.linear`. Those are exactly the
   keys the two converters rename.
4. **The text encoders load too.** `qwen3` is in transformers' GGUF table, and
   the encoder GGUF's header carries `general.architecture = qwen3` with the
   full architecture (36 blocks, width 2560, head counts) and the vocabulary.

What does *not* come from the GGUF is a few megabytes of metadata — the
architecture config, the tokenizer and the scheduler — taken from the model's
reference repository. The tokenizer could have been rebuilt from the GGUF, but a
rebuilt tokenizer can differ in ways you would only notice in the render.

---

## Install

```
setup-torch-engine.bat
```

Several gigabytes of **Python packages**, once. **No model is re-downloaded.**
Then **Settings → 🔧 Expert → Generation engine**, or `set
TURBOSLOP_BACKEND=torch` before `run.bat` if you would rather not touch your
preferences — the environment variable wins over everything, so you can run two
copies of the app side by side, one per engine, and compare on identical files.

`setup-torch-engine.bat --check` prints what is installed and what would
change, without installing anything.

### It moves the Toolkit's stack too

This is the one real risk of the branch, and it is deliberate.

The Toolkit add-ons run on `torch 2.4.1` and `diffusers 0.33.1`, and those pins
hold each other up: 2.4.1 was chosen to cover Pascal through Ada, and its
`infer_schema` cannot read the `X | None` annotations diffusers ≥ 0.35 uses. The
models need **diffusers ≥ 0.36** — it is written in each repo's
`model_index.json`. No version satisfies both, so this branch moves the whole
stack up; moving torch also dissolves the constraint that justified the old pin.

**Pascal survives.** The `cu126` wheels still carry `5.0;6.0;7.0;…` (checked in
pytorch's own `.ci/manywheel/build_cuda.sh` through 2.12) and an `sm_60` cubin
loads on an `sm_61` card, so a GTX 1080 Ti keeps working. **Blackwell moves to
`cu128`**, which starts at `7.0`; the installer picks the index from the card it
detects, exactly as `_torch_setup.py` already did.

### The one token

Two models want a Hugging Face account, and they want it for very different
reasons. Nothing on the native engine ever does: every repository in the GGUF
catalog is open, deliberately.

**Flux.2 Klein** needs it for a few kilobytes. Its weights come from Unsloth's
open GGUF, already on your disk; what is gated is `transformer/config.json`,
which only Black Forest Labs publishes. diffusers' own fallback is no help: it
points at `black-forest-labs/FLUX.2-dev`, a different model, also gated.

**Krea 2** needs it for its **pipeline settings** — the scheduler, which
encoder layers are tapped, whether the model is distilled — which only
`krea/Krea-2-Turbo` publishes. Not for its weights: those load from the GGUF
you already have. It does not run on this engine yet for other reasons; see
the section below.

There is **no command line to run**. This app ships a portable Python with no
console and no PATH, so `huggingface-cli login` is not a thing its user has.
Instead:

1. Create a **read** token at
   [huggingface.co → Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Paste it into **Settings → 🌍 Theme and accounts → Hugging Face token**. It
   applies immediately — no restart.
3. Open each model's page and accept its licence:
   [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
   [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo).
4. Press **🔍 Check what is still missing**. It reports, per model, whether the
   token is valid, whether the licence is accepted, and which page to open —
   because those three failures are all a bare `401` otherwise.

The token is never written into a report: the diagnostic export lists only the
preference keys that affect execution, and this is not one of them.

---

## What runs, and how

### Memory: the ladder already decided

There is nothing to quantize at load time on the normal path. The file on disk
is already `Q5_K_M` or `Q8_0`, chosen by the same VRAM ladder as for
stable-diffusion.cpp, and the sizes are **read** rather than estimated — which
matters, because a `_K_M` file mixes precisions per layer and a bytes-per-
parameter estimate is wrong precisely where it counts.

What is left is placement: everything on the card; or modules taking turns
(`enable_model_cpu_offload`, one sub-model resident at a time, so the peak is
the largest rather than the sum); or weights streamed layer by layer
(`enable_sequential_cpu_offload`, fits in almost nothing, an order of magnitude
slower — a safety net, not a working mode).

With Z-Image Turbo at Q8_0 (transformer 6.6 GB + encoder 2.5 + VAE 0.3 = 9.4):

* **12 GB (RTX 3060)** — everything resident. The fast path.
* **11 GB (RTX 2080 Ti)** — 9.4 does not fit alongside the compute reserve, but
  the largest part (6.6) does: modules take turns.
* **8 GB** — nothing fits; it streams, and says that a lower quantization rung
  in Settings would be the better answer.

The full-precision ladder (bf16 → int8 → NF4 via bitsandbytes) still exists, but
only serves the repo path — that is, Krea 2.

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
can inpaint **with a mask for this model** instead of asking a binary whether it
knows an option.

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

**GGUF upscalers refuse**, and name where to go. Those weights are RRDBNet from
2018 with no diffusers loader, and the modern upscalers already run on PyTorch
through spandrel — which is why they were added in the first place.

**Image → 3D is not ported.** trellis.cpp is a second native engine with its own
binary and its own weights; porting it means the PyTorch TRELLIS, which needs
custom CUDA extensions compiled on the machine. That is a different project, not
a different backend, and pretending otherwise would have meant a tab that fails
at the first click.

**Convert to GGUF is left alone.** It converts weights *for* sd.cpp; it has no
meaning on this engine and still works if you switch back.

---

## Krea 2: the converter that was missing, and the three walls after it

I first wrote that Krea 2 meant "download the whole model". That was wrong, and
worth unpicking because the correction is instructive.

diffusers does not register `Krea2Transformer2DModel` for single-file loading.
But an absent converter does not mean an unreadable file — only that nobody
wrote the mapping. So it was written, and **checked exhaustively**: 432 tensors
in the real `krea2_turbo-Q5_K_M.gguf`, 430 keys in the diffusers class, a
one-to-one mapping with every shape matching, nothing unmapped, nothing
invented, nothing landing twice. The check replays offline from a recorded
header in `tests/fixtures/`, so CI runs it without the 13 GB.

The two leftover tensors are the interesting part. `last.up.weight` and
`last.down.weight`, both 6144×6144, appear in **neither engine's** final layer —
diffusers' `Krea2FinalLayer` and sd.cpp's `KreaLastLayer` both hold exactly a
modulation table, a norm and a projection. The file's own metadata says what
they are: `egg_w: 6144`, `egg_h: 6144`, `egg_c: 1`,
`egg_format: chw_m1p1_flat` — a 6144×6144 single-channel image, flattened, that
whoever packaged the GGUF hid inside it. Roughly fifty megabytes of easter egg
in every quantization, that neither engine reads. The converter drops them **by
name** and raises on anything else it does not recognise: a key that vanishes
silently is how a model ends up loading cleanly and rendering subtly wrong.

So the transformer — the bulk of the weight — is settled. Three things around
it are not, and the app names them one by one instead of saying "download the
model":

* **the text encoder.** Krea 2 reads its prompt with Qwen3-VL-4B, a *vision*-
  language model. transformers converts GGUF for `qwen2`, `qwen3` and
  `qwen3_moe`, but not `qwen3vl` — which is what the installed file's header
  declares. The encoder would have to come as safetensors from
  `Qwen/Qwen3-VL-4B-Instruct` (~8 GB, and **open**).
* **the VAE.** The pipeline wants an `AutoencoderKLQwenImage`, and that class
  has no single-file entry — only `AutoencoderKLWan` does. The
  `wan_2.1_vae.safetensors` already on disk is not loadable as it stands.
* **the pipeline settings.** Scheduler, tapped encoder layers, distilled or
  not: published only by the gated `krea/Krea-2-Turbo`. Reconstructing them
  from memory would replace the model's own constants with mine, silently —
  the same mistake as building a scheduler from class defaults.

Krea 2 therefore refuses **before loading anything**, listing those three. On
this engine it is a text-to-image model on paper and unavailable in practice;
on stable-diffusion.cpp it works as it always has, which is one switch away in
Settings.

## The benchmark measures the one open question

Precision is settled at download time, so there is nothing to compare there. The
sd.cpp profiles — params backend, split mode, auto-fit — are options of a binary
that is not running; offering them would give identical runs and a ranking drawn
from noise.

What is genuinely open is the **compute reserve**: the 2.5 GiB the planner
leaves free for the CUDA context, attention buffers, latents and the desktop.
It is an estimate, it decides on its own between "everything resident" and
"modules take turns", and it has no reason to be right on every machine. So the
benchmark offers exactly two profiles — the planned placement, and the same
model forced fully resident — and only when the two actually differ.

## Reading a result three days later

Both engines write into the same output folder, so the `.txt` beside each image
carries an `Engine:` line and the placement that produced it. Without it a
comparison is impossible to make after the fact — which is the whole point of
the branch.

The system report (**System → Manage → Diagnose**) gains a `torch_engine`
section: what is installed, what is missing by name, and which models are
actually on disk.
