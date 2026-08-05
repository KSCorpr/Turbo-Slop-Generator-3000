# 🟢 Turbo Slop Generator 3000

> **Adaptatif au matériel.** Par défaut l'app utilise la **meilleure carte
> NVIDIA détectée** et s'y adapte (quant de diffusion selon la VRAM, encodeur
> déchargé dans la RAM, flash-attention, offload, VAE tiling). Deux options
> avancées dans **Réglages** : les **presets par génération de carte** (GTX 10xx
> → RTX 50xx, en 1 clic) et le **multi-GPU** (choix du GPU de génération, split
> de l'encodeur sur une 2e carte, ou auto-fit qui répartit le modèle sur toutes
> les cartes).

A **local**, modern, lightweight image-generation studio for artists, built on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** (native
CUDA, GGUF). Generate with **Flux.2 Klein 9B** and **Krea 2 Turbo**, with an
on-demand model catalog, automatic optimization for your RTX card, LoRA, native
resolution presets, saved styles, an AI prompt enhancer, multi-reference image
editing, two upscalers, **video generation with sound (LTX-2.3, MiniMax-H3)**,
and a utility toolkit.

No ComfyUI, no node spaghetti — just a clean web UI.

> **Credits & honesty.** All the heavy lifting — the inference engine, GGUF
> support, CUDA kernels — comes from **Leejet’s
> [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)**. This
> project is just a friendly local UI on top of it; full credit and thanks to
> Leejet and the sd.cpp contributors.
>
> This GUI was **vibe-coded with [Claude](https://claude.ai/code)** (Anthropic) —
> built iteratively in plain language rather than hand-written line by line. Treat
> it accordingly: it’s a hobby tool, not battle-tested production software. Read
> the code, test before relying on it, and report anything that breaks.

| Tab | What it does |
|---|---|
| 🟣 **Flux.2 Klein** | fast (4 steps) · text-to-image & **multi-reference image editing** · presets, styles, LoRA |
| ⚡ **Krea 2 Turbo** | fast photorealism (8 steps, GGUF, Qwen3-VL encoder, WAN 2.1 VAE) |
| 📚 **Model Catalog** | hardware-aware recommendations, on-demand download / delete |
| 🧰 **Toolkit** | depth · background removal · click-to-cutout (SAM) · ESRGAN upscale · creative SDXL upscale |
| 🎬 **Video** | **LTX-2.3** and **MiniMax-H3** · text→video, image→video, first→last frame, reference-conditioned · **with generated soundtrack** · native sd.cpp `-M vid_gen`, no PyTorch |
| 🧊 **Image → 3D** | image → textured 3D mesh (GLB) via **trellis.cpp** (TRELLIS.2, native CUDA, no PyTorch) · one-shot (frees VRAM) · **f16/q8/q4 weight variants** (~16.5 / 9.9 / 6 GB) · in-browser 3D preview |
| 🔧 **Convert to GGUF** | quantize any checkpoint / safetensors / diffusion model to a lighter GGUF (CPU, `sd --mode convert`) so it fits your card |
| 🧹 **Manage & help** | disk inventory of everything downloaded (engines, models, add-ons, your data) with sizes · selective uninstall with confirmation · **in-app documentation of every option** |
| ⚙️ **Settings** | detected hardware, quantization, optimizations (auto profile per detected GPU / manual override) |

---

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [Generation options](#generation-options)
- [Hardware & optimization](#hardware--optimization)
- [Upscaling](#upscaling)
- [Outpaint](#outpaint)
- [Video](#video)
- [Toolkit](#toolkit)
- [Managing disk space & uninstalling](#managing-disk-space--uninstalling)
- [Sharing on your LAN](#sharing-on-your-lan)
- [Distributing a portable package](#distributing-a-portable-package)
- [Models & sources](#models--sources)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Install

### Windows (RTX cards)
```bat
install.bat      ::  portable Python + dependencies + GGUF engine (CUDA)
run.bat          ::  launch the UI at http://127.0.0.1:7860
```

### Linux
```bash
./install.sh
./run.sh
```

> The install does **not** download any models. You fetch them on demand from the
> **Model Catalog** tab (like a media library). Everything stays inside the
> project folder.

Generation runs through stable-diffusion.cpp (no heavy PyTorch for image
generation). **PyTorch is only installed on demand** for the optional Toolkit
tools (depth, background removal, SAM, prompt enhancer, creative SDXL upscale),
each via its own one-click installer.

### Updating by copy-paste — run maintenance afterwards
If you update by extracting the repo ZIP over your existing folder (keeping
`python/`, `bin/`, `models/`…), copy-paste **adds and overwrites files but never
deletes** the ones removed upstream — they linger as orphans, and stale
`__pycache__` can confuse Python. After each copy-paste update, run:
```bat
maintenance.bat      ::  Windows   (./maintenance.sh on Linux/Mac)
```
It deletes obsolete files, purges `__pycache__` and `tmp/`, then verifies that
everything compiles, the model catalog is valid, and the dependencies + `sd-cli`
engine are present. It never touches `models/`, `loras/`, `outputs/`, `userdata/`,
`python/` or `bin/`.

**What a copy-paste update does and does not refresh**

| | In the ZIP? | Refreshed by copy-paste |
| --- | --- | --- |
| App code, `config/models.yaml`, docs | yes | **yes** |
| Engine binary (`bin/`) | no | no — run `update-engine.bat` only when a new sd.cpp feature is needed |
| Models, LoRAs, outputs, prefs (`models/`, `loras/`, `outputs/`, `userdata/`) | no | no — kept, which is the point |
| Toolkit add-ons (`tools_repo/`) | no | **no — and this one bites** |

That last row matters. Some add-ons are not just downloaded weights: **SeedVR2
patches the inference code it clones** (architecture config, model selection,
pinned dependency versions). When a release changes *how an add-on installs*,
updating the app leaves the installed add-on frozen in its old state, and you
only find out at the next use. So: **after updating, if an add-on misbehaves,
re-run its one-click installer** — weights already on disk are not re-downloaded.

`maintenance` now checks this for you and names exactly what is stale, so you
don't have to guess.

---

## Quick start

1. Run `install.bat` / `./install.sh`, then `run.bat` / `./run.sh`.
2. Open the **Model Catalog** tab and download **Flux.2 Klein 9B** (or **Krea 2
   Turbo**). Quantization is picked automatically for your VRAM/RAM.
3. Go to the model's **generation tab**, type a prompt, click **Generate**.
4. (Optional) Install the **prompt enhancer** and click **✨ Enhance prompt** to
   turn a rough idea into a detailed English prompt.

---

## Generation options

Every generation tab exposes the same controls.

### Prompt & system style
- **Prompt** — your description. For **edit models** (Flux.2 Klein) describe the
  *modification* to apply to the reference image.
- **✨ Enhance prompt (AI)** — see [Prompt enhancer](#prompt-enhancer-ai). Note
  that a **style preset translates nothing** — it is a prefix glued in front of
  your text, so writing in French leaves you with French plus an English header.
  The enhancer is what translates and shapes the prompt, and it is **given the
  active preset as a constraint**: it describes the subject without adding
  camera, lens, lighting or processing wording that would contradict the style.
  Pick the preset first, then enhance.
- **Negative prompt** — shown only for models that support it (CFG > 1). Distilled
  models run at CFG 1.0 and ignore it.
- **System / style prefix** (accordion) — a prefix prepended to every prompt. Save
  reusable styles to a dropdown (persisted in `userdata/`). Styles are **global**:
  saved once, available in every generation tab. A few are **bundled** with the
  app in `config/style_presets.json` — they survive updates and cannot be
  deleted, but saving a style under the same name creates your own version, which
  takes precedence; deleting that restores the original. Two separate buttons,
  so nothing is lost by accident: **✖️ Stop applying** only detaches the style
  from the next generations and keeps the preset, while **🗑️ Delete this preset
  (permanent)** erases it and asks for a second click to confirm. Bundled today:
  **📷 France provinciale 1995-2005 (amateur)**, a transcription of a
  "mundane amateur snapshot, provincial France, always overcast, no
  post-processing" brief.
- **📷 Krea 2 photo styles** (accordion) — a bundled bank of **139 stackable
  photographic styles** (quality, lighting, lens, film stock, mood…), grouped by
  category in a **multi-select** dropdown. Your prompt subject is inserted into
  each selected style (`{prompt}` template), and multiple styles chain their
  descriptions after the subject so you can combine axes. Style negatives are
  merged into the negative field **only when the model uses them** (CFG > 1);
  on distilled CFG 1.0 models they are dropped. Bank © *ghleg* — MIT
  ([aoleg/Photographic-styles-and-wildcards-for-Krea-2](https://github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2)),
  shipped as `config/krea2_styles.csv`.
- **🎨 Krea artistic styles** (accordion) — a bundled bank of **397 stackable
  artistic styles** (anime, cartoon, comics, drawing, photography, design,
  digital painting, painting), grouped by category in a **multi-select**
  dropdown. These have no `{prompt}` and no negatives: the style description is
  appended after your subject. They stack with each other **and** with the photo
  styles above. A **🎲 wildcard** button rolls one at random. Shipped as
  `config/krea2_art_styles.csv` — a Krea style collection supplied by the user;
  **provenance/license to be confirmed** (no license was attached to the source).

### Reference image / image-to-image
The accordion adapts to the model family:
- **Flux.2 Klein (edit model)** — **Multi-reference editing**: load the image to
  edit plus up to **2 extra reference images**, and describe the change or the
  combination in the prompt (e.g. *“put the character from image 1 into the scene
  of image 2”*). Each image is passed to the engine as a separate `-r` flag. No
  strength slider — editing is prompt-driven. Output aspect follows your image.
  An **🧩 Outpaint** slider (experimental) extends the canvas and lets the model
  fill the new borders — describe the extension in the prompt.
- **Krea 2 Turbo — ✏️ Edit mode (Ostris Edit)** — check **Mode édition** to pass
  the image as a **context reference** (style transfer, subject reference, edits)
  instead of an img2img starting point. This requires a **Krea 2 edit LoRA**
  (e.g. HF repo [`ostris/krea2_turbo_style_reference`](https://huggingface.co/ostris/krea2_turbo_style_reference) —
  add it via the LoRA panel) and a **recent sd.cpp engine** (`update-engine.bat`;
  needs the `Krea2OstrisEdit` + `--llm_vision` support from July 2026). The
  Qwen3-VL **vision projector (mmproj)** downloads automatically with the model
  and is only loaded in edit mode.
- **Other models (img2img)** — load a **reference / starting image** and set the
  **transformation strength** — low (0.2–0.4) keeps the reference's structure,
  high (0.7–1.0) reinvents it.

### LoRA
Drop `.safetensors` / `.gguf` files into **`loras/`**, then pick up to **2** with
their weights. The `<lora:name:weight>` syntax is forwarded to the engine. Use
**↻ Refresh** after adding files, **✖ Clear** to reset. You can also **import a
LoRA from Civitai** in one click — paste the model URL (or version ID) into the
LoRA accordion; gated models need a Civitai token (Settings).

### Local / custom files
To use a model downloaded elsewhere, drop the file(s) into **`models/custom/`**
and select them as *Diffusion / VAE / Encoder (local)*. Empty = use the catalog
model. **✖ Clear custom fields** resets the selection.

### Resolution presets (native per model)
Each model offers **formats aligned with its training resolutions** (it renders
best on these):
- **Flux.2 Klein** — ~1 MP, 32-px grid: 1024², 1248×832, 1184×880, 1392×752,
  1568×672… (+ a 2K option).
- **Krea 2** — 1024 family (multiples of 64): 1024², 1216×832, 1152×896,
  1344×768… (+ a 2K option).

Pick a ratio from the dropdown, or choose **Custom (sliders)** for free width /
height (256–2048, step 16). Loading a reference image auto-fits width/height to
its aspect.

### Sampler / scheduler / steps
- **Preset** — vetted combos per model (e.g. Flux.2 Klein → 4 steps / CFG 1.0 /
  euler + simple). Selecting one fills sampler, scheduler, steps and CFG.
- **Sampler** — all samplers supported by sd.cpp (euler, dpm++2m, res_multistep…).
- **Scheduler (sigmas)** — auto (model default), karras, simple, exponential…
- **Steps** — diffusion steps. Distilled models need few (4–8).
- **CFG** — guidance. **1.0 = no guidance** (normal for distilled Flux). Values
  other than 1.0 are experimental on distilled models.
- **Flow shift** — leave at **0 (auto)**: the model picks the right value for the
  resolution. Too low (1–2) leaves grain/noise at high resolution; ~3–4 reinforces
  structure.

### Seed & batch
- **Seed** — `-1` = random. The used seed is shown under each result and written
  to the sidecar file.
- **Images** — batch count (1–8).

### Output
- **Merged preview & results** — the live preview shows in the gallery during
  generation, then the final images replace it (one view).
- **Seed** — the selected image's seed shows in a copy-button box; **Reuse this
  seed** drops it back into the seed field. Clearing the seed field resets it to -1.
- **Send to Toolkit** — push the selected image straight into a Toolkit tool
  (depth, background removal, SAM, ESRGAN or creative upscale).
- **Saved prompts** — every image gets an A1111-style `.txt` sidecar in
  `outputs/` with the prompt, negative, model, sampler/scheduler, seed and size.

## Hardware & optimization

### Automatic optimization
The app detects your GPU (via `nvidia-smi`) and RAM, then chooses on its own:
- **diffusion quantization** by VRAM
  (`<8 GB → Q4_K_S`, `8–12 → Q4_K_M`, `12–16 → Q5_K_M`, `16–24 → Q6_K`, `≥24 → Q8_0`);
- **encoder quantization** by RAM (the text encoder is offloaded to RAM, so it
  costs no VRAM);
- **flags**: flash-attention (Turing / RTX 20xx and newer), CPU offload, VAE
  tiling, CLIP/VAE on CPU — enabled progressively as VRAM gets tighter;
- Pascal cards (GTX 10xx) → flash-attention disabled automatically (it’s slow there).

Multi-GPU: the largest card is used by default, changeable in **Settings**
(see [Multi-GPU](#multi-gpu)). Everything is overridable manually
(uncheck auto-optimization).

These map to stable-diffusion.cpp flags: `--diffusion-fa` (CUDA: faster + less
VRAM), `--offload-to-cpu` (saves VRAM with no speed loss), `--vae-tiling`,
`--clip-on-cpu`, `--vae-on-cpu`, plus GGUF quantization.

### Manual settings
With auto unchecked you control quant (diffusion / encoder), the GPU, and each
flag (flash attention, CPU offload, VAE tiling, CLIP on CPU, VAE on CPU). A custom
Hugging Face endpoint (mirror) can also be set.

### Per-generation presets (1 click)
Below the auto toggle, one button per RTX generation (**GTX 10xx → RTX 50xx**)
applies a **curated profile** for that card in one click: it reads the real VRAM
of the selected GPU, picks the diffusion quant (with a small speed/quality bias
per generation — lighter on Pascal/Turing, higher on Ada/Blackwell), sets the
encoder quant from RAM, and toggles the memory flags (flash-attention off on
Pascal, VAE tiling / CPU offload on tighter cards). This unchecks auto and fills
the manual fields, so you can still tweak afterwards.

### Multi-GPU
When **two or more GPUs** are detected, a **Multi-GPU** accordion appears with a
single mutually-exclusive strategy:
- **Single card (recommended)** — everything on the generation GPU, text encoder
  offloaded to RAM. The most reliable.
- **Text encoder on the 2nd card** — diffusion + VAE stay on the main GPU, the
  text encoder runs on the other card (`--backend …,te=cudaN`). Frees VRAM on the
  main card for the diffusion model.
- **Auto-fit** — sd.cpp spreads diffusion / encoder / VAE across all cards
  (`--auto-fit`). ⚠️ forces everything into VRAM (disables CPU offload) → can OOM
  on big-encoder models (Flux.2 Klein); reserve it for models that fit in the
  combined VRAM.

A separate **prompt-enhancer GPU** can also be chosen (the text LLM runs there;
image generation and SDXL upscale always stay on the generation GPU). The GPU
picker and these strategies live in **Settings**.

### Samplers & schedulers
The built-in **presets follow the official sd.cpp docs** (`docs/flux2.md`,
`docs/krea2.md`): **Euler** sampler with the **scheduler left to the engine
default** (the docs never force one), at each model's documented steps/CFG
(Flux.2 Klein 4 steps · CFG 1.0; Krea 2 Turbo 8 steps · CFG 1.0). The dropdowns
still expose
the full sd.cpp list for manual experimentation — newer samplers like **DPM++ 2M
SDE** and schedulers like **Flux.2 / Flux / Beta** are there to try, but the
presets stay on the documented defaults. New entries need a recent engine
(`update-engine.bat`).

### Cache acceleration (experimental)
**Settings → 🗃️ Cache acceleration** exposes sd.cpp's step-caching
([`caching.md`](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md)):
`easycache`, `dbcache`, `taylorseer`, `cache-dit` or `spectrum`, plus a free-form
option (e.g. `threshold=0.2`). It reuses near-identical computations across
diffusion steps. Honest note: it pays off mostly above ~10 steps — on 4–8-step
distilled models the gain is small and artifacts are possible, hence **off by
default**. Requires a recent engine (`update-engine.bat`).

### One engine, warm reloads
Generation always runs through **one-shot `sd-cli`** — the single engine mode.
It gives the **live step preview**, and reload speed is handled by the OS: after
the first load of a session, the model files sit in the **disk cache (RAM)**, so
subsequent loads take seconds. (Earlier experimental engines — a resident
`sd-server` and a ComfyUI backend — were removed: the server couldn't do live
preview and ComfyUI proved too fragile. One engine, no mode switch, no surprise.)

### Engine binary: official or self-built (CI)
By default `update-engine.bat` downloads the **official** prebuilt binary from
[`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
releases — the simplest, always-works path.

Optionally, the project can build **its own** engine binary via **GitHub
Actions**, with no dev tools on your machine:
- The workflow **`.github/workflows/build-sdcpp.yml`** (Actions tab → *Build
  sd.cpp (Windows CUDA)* → *Run workflow*) clones sd.cpp, builds a Windows CUDA
  binary compiled **only for this project's cards** (arch `61;75;86` =
  GTX 1080 Ti + RTX 2080 Ti + RTX 3060 — leaner, sometimes faster than the
  generic release), bundles the CUDA runtime DLLs, and publishes it to a moving
  `engine-latest` release.
- **`update-engine-ci.bat`** then installs *that* binary
  (`scripts/get_sdcpp.py --source ours`) instead of the official one.

Why self-build: **day-0** access to new sd.cpp features, arch-tuned binaries,
the ability to **pin a known-good commit** (workflow input `sd_ref`), or to
apply engine **patches** when needed. Everything heavy happens in CI — your
machine only ever downloads a ready binary.

### Updating the engines

| Script | Updates |
|---|---|
| `update-engine.bat` | **sd.cpp** — latest official prebuilt binary (image generation) |
| `update-engine-ci.bat` | **sd.cpp** — our own CI build (arch-tuned, can carry PRs) |
| `update-trellis.bat` | **trellis.cpp** — latest official Windows CUDA build (Image → 3D) |

Each one replaces only the **engine binary** (the previous version is removed
first, so DLLs from two releases never mix). **Models are never
re-downloaded** — including the ~16 GB trellis 3D set; reinstall those from the
**Image → 3D** tab if ever needed.

### Interface language & theme
**Settings → 🌐 Langue / Language** switches the UI between **French** and
**English**; **🎨 Thème** switches between **Light** and **Dark**. Both are saved
to `userdata/` and applied on **restart** (`run.bat` / `run.sh`) — Gradio builds
the interface once at launch. On first launch a bilingual language chooser is
shown at the top. Engine logs and progress hints stay in French.

---

## Upscaling

Three complementary upscalers live under **Toolkit**, in increasing order of
invention: ESRGAN interpolates, SeedVR2 restores, SDXL hallucinates.

### 🔼 Simple (ESRGAN, native sd.cpp)
Deterministic ESRGAN upscale via sd.cpp `--mode upscale`: **100% GPU, no PyTorch,
no prompt**. One-click downloads **all** models from
[`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf)
(2x-ESRGAN, RealESRGAN_x4plus, 4xUltrasharpV10, 4x_foolhardy_Remacri…). Pick a
model (×2/×4 depending on its name); **Repeat ×2** chains two passes (a ×2 model
twice = ×4). Best for a clean, faithful enlargement.

**Line art, comics and illustration.** The model matters more than the settings.
Photo-trained models (Remacri, Nomos, UltraSharp…) learned natural texture: on a
flat colour area they hallucinate grain, and along a clean ink line they ring.
The dropdown therefore tags each model — 🎨 **drawing / anime** vs 📷 **photo** —
lists the drawing ones first and preselects one, instead of defaulting to
whatever sorted first alphabetically.

**Bring your own models.** sd.cpp loads most `.pth` files directly, so the picker
accepts `.gguf`, `.pth` and `.safetensors`: drop a file in the upscalers folder,
hit **↻ Refresh**, and it appears. That opens the whole
[OpenModelDB](https://openmodeldb.info) catalog — filter on *anime* / *manga* /
*cartoon* for line art. GGUF still loads faster and avoids executing a pickle, and
you can convert one yourself with
`sd-cli --mode convert --model x.pth --output x.gguf`. Note that sd.cpp only
implements the **ESRGAN (RRDBNet)** architecture for image upscaling, so newer
SPAN / DAT / Compact models will not load.

### 🎯 Restoration (SeedVR2 1.4B)

**One diffusion step, no prompt, no text encoder.** SeedVR2 reconstructs
plausible detail — skin, fabric, foliage, text — instead of smoothing like ESRGAN
or inventing like the creative upscale. Its sweet spot is **×2 to ×4**; quality
degrades past that.

Small and fast: **2.9 GB** of weights. Fixed sampler settings (steps 1, cfg 1,
euler), so the only control that really matters is the **target short-side
resolution**.

> **VRAM.** The model card's "~4.6 GB peak" is for a 512→2048 job. What actually
> drives memory is the *output* size: SeedVR2 resizes the input to the target
> first, then the VAE encodes at that size — and its causal 3-D convolutions
> replicate the frame along the temporal axis, which is what blows up. On
> 11–12 GB, an untiled run OOMs around 1440 px. **VAE tiling is therefore on by
> default** and the target defaults to 1080 px. If you still run out, lower the
> target, then drop the tile size to 256. The tile slider is capped at 512 on
> purpose: the VAE self-attends over the whole tile, so cost grows as O(n²) —
> bigger tiles are both slower *and* heavier.

> **How it is wired.** There is no pip package and no `diffusers` pipeline for
> SeedVR2. The reference inference code is the
> [numz repository](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler),
> which ships as a ComfyUI node **but provides an `inference_cli.py` explicitly
> documented as usable without ComfyUI**. The installer clones that repo and we
> call its CLI as a subprocess — **ComfyUI is never installed or run**, and we do
> not copy hundreds of lines of model architecture that would then need
> maintaining. Dependencies are ordinary pip packages (einops, omegaconf,
> diffusers, peft, rotary_embedding_torch…); **no apex, flash-attn or triton** —
> those are optional accelerators and the code falls back to PyTorch SDPA.

> **Shared pins across every add-on.** All PyTorch tools live in the *same*
> embedded Python, so a tool installed with a looser constraint silently
> replaces the version another one needs — and the breakage only surfaces at the
> next use of the *other* tool, as an unreadable import-time error. Every
> installer therefore applies one shared set, `_PINS` in
> `scripts/setup_tools.py`:
>
> | Pin | Why |
> | --- | --- |
> | `diffusers==0.33.1` | satisfies SeedVR2's `>=0.33.1` and predates `attention_dispatch.py`, whose `X \| None` annotations torch 2.4.1's `infer_schema` cannot parse |
> | `numpy>=1.24,<2` | required by the torch 2.4.1 / torchvision 0.19 base |
> | `transformers>=4.45,<4.50` | lower bound for the four tools that use it; upper bound set by torch — recent 4.5x import `DTensor` from the *public* `torch.distributed.tensor`, which only exists from torch 2.5 (`<5` alone was not enough, and 5.x additionally drags in `huggingface-hub` 1.x) |
>
> The numpy pin is passed **inside** the same `pip install` rather than
> re-applied afterwards. That lets pip's resolver pick an `opencv-python` build
> compatible with numpy 1.x by itself, instead of installing the newest and then
> breaking its dependency — no need to guess where opencv started requiring
> numpy ≥ 2. `maintenance` checks every pin against what is actually installed
> and names the add-on to reinstall if one drifted.
>
> The SeedVR2 installer ends with a **smoke test** that replays the exact import
> chain (`diffusers` → `loaders` → `transformers` → `torch.distributed`) in a
> subprocess and prints the four version numbers. Version incompatibilities in
> this stack surface only at import, as long unreadable tracebacks — this way
> they surface during installation, not ten minutes into an upscale.

The **1.4B** weights come from
[`lvladikov/SeedVR2-1.4B`](https://huggingface.co/lvladikov/SeedVR2-1.4B) — a
6-block distillation of the 7B teacher. Upstream only knows the 3B and 7B, so the
installer writes a `configs_1_4b/main.yaml` — **derived at install time from the
repo's own `configs_7b/main.yaml`**, changing only `num_layers` / `mm_layers` to
6 (the derived `block_type` / `window` / `window_method` fields are OmegaConf
interpolations on `${.num_layers}`, so they follow by themselves). Deriving
rather than shipping a frozen copy matters: a static config goes stale as
upstream's `NaDiT` gains parameters, which surfaces as
`NaDiT.__init__() missing 1 required positional argument`. It also extends one
line in
`src/core/model_configuration.py` that picks the config directory. That edit is
**idempotent and verified**: if the upstream pattern ever changes, the installer
refuses to patch, says so, and the official 3B/7B models keep working.

A weight file is only selectable if it sits in **the exact folder upstream
scans**. Without ComfyUI, `get_base_cache_dir()` returns the *relative* path
`./models/SEEDVR2`, resolved against the **current working directory**, and
`--dit_model`'s list of valid choices is built from that folder when argparse is
constructed. So two things must line up: the weights live in
`tools_repo/seedvr2/models/SEEDVR2/`, and the CLI is launched with
`cwd=tools_repo/seedvr2`. The installer asks the cloned repo itself where that
folder is rather than hard-coding it, and migrates weights from the older
location instead of re-downloading 2.9 GB.

The VAE (~0.5 GB) is fetched automatically on the first upscale.

### ✨ Creative (SDXL, *Ultimate SD Upscale*)
Creative, Magnific-style upscale: pre-enlarge, then **refine tile by tile** with
SDXL img2img at low denoise. The model stays **resident** on the GPU so tiles are
fast; overlapping tiles are blended with a cosine feather for seamless joins, with
a **real-time preview**. Invents fine detail. This is an A1111-free re-implementation
(plain img2img, no ControlNet). PyTorch + diffusers (~7 GB: SDXL base + VAE
fp16-fix), installed in one click.

Controls:
- **SDXL model** — use the bundled SDXL Base 1.0, or drop your own SDXL
  checkpoint (`.safetensors`) into `tools_repo/upscale/checkpoints/` and pick it.
  **VAE** choice: external fp16-fix (recommended, avoids black images) or the
  checkpoint's **built-in VAE**.
- **Pre-upscale** — base enlargement before the SDXL tile refine: **Lanczos**
  (default) or any installed **ESRGAN** model (sharper, real detail). The number
  of ESRGAN passes is **computed from the target**: a ×2 model asked for a ×4
  result runs twice, because landing *below* the target would force the runner to
  finish in Lanczos — reintroducing exactly the blur the ESRGAN was there to
  avoid. Overshooting is harmless (the following downscale is sharp), so passes
  are only capped by an 8192 px ceiling on the intermediate image; when that
  ceiling stops it short, the log says so instead of quietly going soft.
- **Presets** — a dropdown that sets **the whole recipe**, not just a prompt:
  prompt, negative prompt, creativity, CFG, steps, structure locking and
  pre-upscaler. A line under the menu states what it just applied. See
  [Upscaling illustrations](#upscaling-illustrations-without-interpolation)
  below.
- **Creativity (denoise)** — 0.15 faithful → 0.75 inventive.
- **Negative prompt** — what SDXL is forbidden to add. The default is
  photo-oriented; on drawings it is what keeps grain and photo texture off the
  flat color areas.
- **🔒 ControlNet Tile** (optional) — conditions each tile on the source so you can
  push creativity higher **without drifting** from the original structure (the
  Magnific trick). Toggle + a *ControlNet fidelity* slider appear once it's
  installed (`xinsir/controlnet-tile-sdxl-1.0`, ~2.5 GB, included in the
  installer). Without it, it's plain low-denoise img2img — already very good.
- **Scale** — ×1.5 to ×8 (up to ~8K, capped at 8192 px). High factors mean many
  tiles → slow, and ~1–2 GB system RAM for the final assembly; VRAM stays constant
  (tiled).
- **Steps / tile**, **CFG**, **tile size** (640–1280).
- On < 12 GB VRAM, the model is automatically CPU-offloaded to avoid OOM.

> Use the right tool: **ESRGAN** is fast/faithful/deterministic; **SDXL creative**
> is slower but adds invented detail.

#### Upscaling illustrations without interpolation

A photo-oriented upscale does three specific things to a drawing, and all three
have to be fixed together — which is why this is a preset and not a prompt:

1. **the base interpolates.** Lanczos does not add information, it averages
   pixels: linework goes soft and flat fills go mushy. Only an ESRGAN *trained on
   drawings* actually enlarges line art;
2. **the default negative prompt does not defend flat areas**, so SDXL happily
   lays photo grain and material texture over them;
3. **denoise around 0.40 redraws the linework**, which then wobbles — lines stop
   being the same lines.

The **🖍️ Illustration / comics — crisp linework, no interpolation** preset sets:
a **drawing ESRGAN** as the base (auto-picked from what you have installed,
preferring `RealESRGAN_x4plus_anime_6B`), a negative prompt aimed at
photorealism/grain/halos, **denoise 0.18**, **CFG 4.0**, and **ControlNet Tile at
0.85** so the structure is locked. At that point SDXL is no longer redrawing
anything — it only cleans up what the ESRGAN produced.

If you have **no drawing upscaler installed**, the preset says so explicitly and
warns that the base will stay on Lanczos: download the upscaler pack from the
**🔼 Upscale** tab first, otherwise the preset cannot do its main job.

For painted or brushwork illustration, **🎨 Painted illustration / concept art**
is the looser variant (denoise 0.35, ControlNet 0.7) — it keeps some material,
which is the point there.

---

## Outpaint

The **🖼️ Outpaint** tab extends an image **left, right, up, down — or all
around**, Midjourney-style.

> **Use it with an *edit* model** — Flux.2 Klein or Boogu Edit. This is not a
> preference, it is what makes the feature work at all.

**Why the model has to be an edit model.** An edit model receives the enlarged
canvas as a **reference image** (`-r`): its image conditioning tells it what the
scene actually contains, and an **auto-generated extension instruction** tells it
what to do with it. A plain text-to-image model gets none of that — in img2img it
only sees a noised latent, so it does not know what it is continuing and
**reinvents instead of extending**. No amount of tuning strength, feather or edge
fill fixes that; it is a limitation of the method, not a setting. The img2img path
is kept as a fallback so nothing is blocked, not because it produces good output.

That auto-generated instruction is what "no prompt" means here: you write
nothing, but the model still receives a precise directive naming which sides were
extended and telling it to continue perspective, lighting, palette and style
without touching the original or duplicating subjects.

**The pipeline** (`atelier/engine/outpaint.py`):

1. the canvas is enlarged in the chosen directions (snapped to 16 px, capped at
   2048 px per side — margins shrink proportionally if the cap is hit);
2. the new area is pre-filled: **neutral grey** for an edit model (an obviously
   empty zone reads as "fill this"; a fake backdrop would mislead it), or an
   edge-stretch/mirror fill for the img2img fallback, which has nothing else to
   go on;
3. **edit path** — canvas as `-r` plus the instruction, no strength, no mask.
   **Fallback path** — canvas as `-i` with a strength, plus an inpainting mask
   *if* the installed `sd-cli` has a mask option. That flag is **discovered at
   runtime** by parsing `sd-cli -h` (`sdcpp.supported_options` /
   `sdcpp.mask_flag`) rather than hard-coded, since its spelling varies between
   versions and between official and self-built binaries;
4. the result's **tone is matched back to the original** (per-channel mean and
   standard deviation, measured on the overlap region);
5. the **original is composited back on top**, with a feather at the seam.

Step 4 is not cosmetic. The model re-renders the *whole* canvas with punchier
contrast and saturation; pasting the untouched original back on top then leaves a
visibly duller rectangle in the middle, which a feather cannot hide because the
mismatch is global, not local. The fix corrects the *new* area toward the
original, never the reverse.

**Controls**

| Control | What it does |
| --- | --- |
| **Direction** | left / right / top / bottom / horizontal / vertical / all around |
| **Extension per side** | fraction of the original added to each chosen side (0.25 = +25 %); the resulting size is previewed live |
| **Model** | any installed model; sampler, CFG and steps follow its catalog defaults |
| **Prompt** | *optional* — leave empty for a neutral extension, fill it only to steer what appears in the new area |
| **Edge fill** | **Neutral grey** (default on an edit model) — the empty zone is unambiguous. **Blurred stretch** replicates the border outward, carrying color but no shape. **Mirror** gives perfect continuity on regular patterns, but reflects any subject near the edge and the model turns that reflection into a second real object — uniform backgrounds only |
| **Generation strength** | fallback path only, and greyed out on an edit model (which is driven by the instruction, not by a strength). High = invents freely; low = stays close to the pre-fill |
| **Feather** | width of the blend at the seam. Note it blends a band of roughly **2× its value** *inside* the original's border — that is what makes the seam disappear. **0 = hard paste**, original strictly untouched everywhere |
| **Tone match** | 0–1, how strongly the new area is pulled onto the original's contrast and color. Lower it only if the correction over-corrects on an unusual image |
| **Seed** | -1 = random; a fixed value replays the same extension |

**♻️ Re-extend the result** reloads the output as the new input, so extensions
can be chained (right, then up, …). Results land in `outputs/` with a `.txt`
sidecar recording model, seed and settings. Generation tabs have a
**🖼️ Send selection to Outpaint** button.

> The old **🧩 Centered outpaint** slider inside the Flux.2 tab is a different,
> experimental thing: symmetric only, edit-capable models only, prompt-driven.
> The dedicated tab supersedes it.

---

## Video

The **🎬 Video** tab generates short clips **with a generated soundtrack**, from
text or from images, read natively by sd.cpp (`-M vid_gen`). No PyTorch, no
ComfyUI — the same engine that generates images.

> ⚠️ **This is by far the heaviest thing in the app.** The weights live in RAM
> and stream to the GPU (`--offload-to-cpu`), which is the only reason a 22 B+
> model runs on an 11-12 GB card. Budget **32 GB of RAM for LTX-2.3 and 48-64 GB
> for MiniMax-H3**, and expect **several minutes per clip** — that is not a
> crash.

**Two families**, and the tab reconfigures itself around whichever you pick —
available modes, frame-count steps, frame rate and the ×2 refine pass all differ:

| | **LTX-2.3** (Lightricks) | **MiniMax-H3** |
| --- | --- | --- |
| Download | ~25 GB | ~35 GB |
| Prompt encoder | Gemma-3-12B (7.3 GB) | **Qwen3-VL-32B** (18.2 GB) |
| Audio | yes | yes, **stereo**, generated in the *same* diffusion pass |
| Frame counts | 8k+1 (33, 41, 49…) | 17k+5 (5, 22, 39, 56…) |
| Frame rate | free | **24 fps, forced** by the model |
| ×2 latent refine | yes | no |
| Reference conditioning | no | yes (Ref2VA variant) |

**Modes** — which ones appear depends on the model, because the constraint is
the model's, not the interface's:

| Mode | What you give it | What it does |
| --- | --- | --- |
| 📝 **Text → video** | a prompt | generates the clip from scratch |
| 🖼️ **Image → video** | a prompt + one image | animates that still |
| 🎞️ **First → last** | a prompt + two images | generates the in-between |
| 🎭 **Reference** | a prompt + 1-2 reference images | keeps that character/object across the shot (**MiniMax-H3 Ref2VA only**) |

**The four catalog entries**

- **LTX-2.3 Distilled 22B** — 8 steps, CFG 1.0. **The one to start with on
  11-12 GB.** Distilled at CFG 1.0, so like Flux.2 Klein it ignores the negative
  prompt and the field is hidden.
- **LTX-2.3 Dev 22B** — ~20 steps at CFG 6.0, negative prompt active. Finer, but
  roughly **2.5× slower** at equal settings.
- **MiniMax-H3 FL2VA** — text→video, image→video, first→last. Video and
  **stereo sound** come out of one packed diffusion transformer, so the audio is
  synchronised by construction rather than bolted on afterwards.
- **MiniMax-H3 Ref2VA** — reference-conditioned instead: give it images of a
  character and ask for it to stay the same. In exchange it accepts **neither a
  first nor a last frame** — a model constraint, which is why the tab hides
  those inputs rather than letting you fail into it. Name the reference from
  inside the prompt (*"use the cat from &lt;Picture 1&gt;, keep its appearance
  consistent"*), otherwise the model sees the image but has no instruction
  attached to it.

Within each family the encoder and VAEs are **shared**, so the second variant is
a much smaller download than the first.

> **The engine must be recent enough.** MiniMax-H3 landed in stable-diffusion.cpp
> well after LTX-2.3, so an `sd-cli` that happily runs LTX can still be unable to
> load MiniMax. The tab checks the **actual binary** for the options each model
> needs and tells you to run `update-engine.bat` **before** you download tens of
> gigabytes of weights.

**Writing the prompt.** English works best, and **describing the motion matters
as much as describing the scene** — both the subject's movement and the camera's
(*"camera slowly pushing in"*, *"handheld, drifting left"*). A prompt that only
describes a still image tends to produce a nearly still clip.

**Format, duration and the grids that constrain them.** LTX only produces sizes
that are **multiples of 32 px** and frame counts of the form **8k+1** (33, 41,
49…). Ask for 720 px and sd.cpp integer-divides it down to 704 without telling
you. So the tab offers only aligned formats, takes the duration **in seconds**,
and prints — under the sliders, *before* you click — exactly what will come out:
`→ 704×384 · 33 images à 24 i/s · 1.4 s`. That readout is the contract; there is
no surprise after the fact.

**🔍 Detail ×2** runs the official **LTX spatial latent upscaler** between a
low-resolution pass and a refine pass, doubling the output. Sharper, but clearly
slower and hungrier — leave it off until the clip is otherwise right. It is
downloaded with the model as an optional component; if it is missing the box is
simply ignored and the log says so.

**Output.** A `.webm` in `outputs/` with **video and audio muxed together** (LTX
generates the soundtrack; sd.cpp writes it into the same file), plus a `.txt`
sidecar recording prompt, model, mode, size, steps, CFG and seed.

**If the tab says the engine is too old.** Video needs an `sd-cli` that knows
`-M vid_gen`. The tab **checks the installed binary** (it parses `sd-cli -h`
rather than assuming) and tells you to run **`update-engine.bat`** if the option
is absent. LTX-2.3 has been supported by stable-diffusion.cpp since May 2026.

**Start small.** 704×384 over 2 seconds, distilled, no ×2 refine. Once that
produces something you like, raise one thing at a time — resolution *or*
duration *or* the refine pass. Raising all three at once on a 11-12 GB card is
the reliable way to get an out-of-memory error after ten minutes of waiting.

---

## Toolkit

One-click installable utilities (models pulled from Hugging Face, run as
subprocesses so torch DLLs never lock the UI process):

- **Depth** — *Depth Anything V2* (depth map).
- **Background removal** — *RMBG-1.4* (cutout → transparent PNG; non-commercial
  license).
- **Click-to-cutout (SAM)** — *Segment Anything* (`facebook/sam-vit-base`): click
  an object, extract it to a transparent PNG.
- **Upscale (ESRGAN)**, **Restore (SeedVR2)** and **Creative upscale (SDXL)** —
  see [Upscaling](#upscaling).

### Prompt enhancer (AI)
The **✨ Enhance prompt** button (in each generation tab) runs a small instruct
LLM (*Qwen2.5-3B-Instruct*, PyTorch ~6 GB, one-click install) that rewrites your
idea into a detailed **English** prompt (subject, lighting, composition, style).
The model is loaded then unloaded per call → **no VRAM conflict** with generation.
It outputs only the enhanced prompt, injected straight into the prompt field. The
system prompt **detects intent from keywords** (medium/style/subject/mood) and
keeps the output medium-coherent; a **strength** selector (Light / Medium / Strong)
controls how far it expands. Krea 2 uses a Krea-specific system prompt.

**Several proposals at once.** Pick 1, 2 or 4 under **🎨 Enhancement options**;
they are produced in a *single* model load (`num_return_sequences`), so four cost
barely more than one. They appear under the prompt field and clicking one puts it
in the prompt. A grey line under the menus states in plain words what the button
will do before you press it.

**The active style preset is passed in as a constraint**, so the LLM writes *with*
it rather than against it — no camera, lens, lighting or processing wording that
would contradict the style, and no restating of the prefix itself.

---

## Sharing on your LAN

Colleagues can generate from their **Mac/PC** using **your** machine and its GPU,
without installing anything — just a link in a browser.

1. On your PC, run **`run-lan.bat`** (instead of `run.bat`).
2. The address to share is printed, e.g. `http://192.168.1.42:7860`.
3. Colleagues on the **same Wi-Fi/network** open it in their browser. That’s it.

Options:
- **Password**: `run-lan.bat --auth name:password` (prompted on connect).
- **Firewall**: on first launch Windows may ask to allow Python — accept (private
  networks). Otherwise allow port 7860 in the firewall.

> Generations run **on your PC**: don’t turn it off during use. One generation is
> processed at a time (automatic queue).

---

## Distributing a portable package

To share the GUI so friends install nothing:

1. On a machine where **everything already works** (Python + engine in `bin\`),
   run **`make_portable.bat`**.
2. It produces `Turbo-Slop-Generator-3000-portable.zip` with the code, the **portable
   Python** and the **engine** — but no models.
3. Friends **unzip** and run **`run.bat`**. No GitHub download needed: they only
   fetch the **models** from the Model Catalog tab (via Hugging Face).

This works even if someone’s network filters GitHub — the engine is already in the
ZIP. To update the GGUF engine later, run **`update-engine.bat`**.

---

## Models & sources

`config/models.yaml` is the single source of truth (sources, defaults, presets).
Quantization tokens (`{quant}` for diffusion, `{enc_quant}` for the encoder) are
resolved from your hardware; the downloader picks the closest matching file.

**Flux.2 Klein 9B** (family `flux2`, edit model)
- diffusion — [`leejet/FLUX.2-klein-9B-GGUF`](https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF) (distilled, 4 steps, CFG 1.0)
- VAE — [`Comfy-Org/flux2-klein-9B`](https://huggingface.co/Comfy-Org/flux2-klein-9B) (`flux2-vae.safetensors`)
- text encoder — [`unsloth/Qwen3-8B-GGUF`](https://huggingface.co/unsloth/Qwen3-8B-GGUF) (official Qwen3-8B, via `--llm`, offloaded to RAM)

**Krea 2 Turbo** (family `krea2`, sd.cpp)
- diffusion — [`realrebelai/KREA-2_GGUFs`](https://huggingface.co/realrebelai/KREA-2_GGUFs) (`TURBO/…`, 8 steps, CFG 1.0)
- text encoder — [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (official Qwen3-VL-4B-Instruct, via `--llm`, offloaded to RAM)
- VAE — [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) (`wan_2.1_vae.safetensors`)

**LTX-2.3 22B — video** (family `ltx2`, `kind: video`, sd.cpp `-M vid_gen`)
Two variants sharing their encoder and latent upscaler, so the second one is
almost free to add:
- diffusion — [`unsloth/LTX-2.3-GGUF`](https://huggingface.co/unsloth/LTX-2.3-GGUF) — `distilled-1.1/…` (8 steps, CFG 1.0) or `ltx-2.3-22b-dev-…` (~20 steps, CFG 6.0)
- video VAE + **audio VAE** — same repo, `vae/…_video_vae.safetensors` and `vae/…_audio_vae.safetensors` (the audio VAE is what puts a soundtrack in the `.webm`)
- embeddings connectors — same repo, `text_encoders/…_embeddings_connectors.safetensors` (via `--embeddings-connectors`)
- prompt encoder — [`unsloth/gemma-3-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF) (`Q4_K_M`, via `--llm`, offloaded to RAM)
- spatial latent upscaler (optional) — [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`), used by **🔍 Detail ×2**

About **25 GB** on disk per variant-set (14 GB diffusion at Q4_K_M, 7.3 GB
encoder, 1.8 GB VAEs, 2.3 GB connectors, 1 GB upscaler). The encoder quant is
**pinned to Q4_K_M rather than following `{enc_quant}`**: on a 64 GB machine the
ladder would fetch Q8_0, i.e. 12.5 GB for a *prompt* encoder, with no visible
gain on the video — the sd.cpp docs use a ~7 GB Q4 as well. See
[Video](#video) for how to actually use it.

**MiniMax-H3 22B — video + stereo audio** (family `minimax_h3`, `kind: video`)
Two DiT variants, sharing encoder and VAEs:
- diffusion — [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) — `minimax_h3_fl2va_pruned-…` (first/last-frame) or `minimax_h3_ref2va_pruned-…` (reference-conditioned). The **pruned** weights are used: 11.4 GB at Q4_K_M against 18.8 GB for the full DiT, which is the difference between usable and not on a 11-12 GB card
- prompt encoder — same repo, `qwen3vl_32b_minimax_h3-…` — **Qwen3-VL-32B** truncated to 50 language layers, vision tower **included in the GGUF** (so no separate `--llm_vision` to pass). 18.2 GB at Q4_K_M; the repo only publishes Q4_K_M and Q2_K_M, so the ladder lands on Q4_K_M and you can drop to **Q2_K_M (13.1 GB)** by forcing the encoder quantization in Settings if RAM is tight
- video VAE + **audio VAE** — [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) (`vae/minimax_h3_video_vae_fp16` 5.2 GB, `vae/minimax_h3_audio_vae_fp32` 0.6 GB)

About **35 GB** per variant-set, essentially all of it resident in RAM during
generation — hence the 48 GB RAM floor. The 5.2 GB video VAE is the VRAM peak,
which is why VAE tiling is forced on. The documented invocation also pins
`--rng cpu`, which the catalog carries as a per-model field rather than a global
setting.

**Boogu Image Edit Turbo 10B** (family `boogu`, instruction editing, Apache 2.0)
- diffusion — [`realrebelai/Boogu-Image-Edit-Turbo_GGUFs`](https://huggingface.co/realrebelai/Boogu-Image-Edit-Turbo_GGUFs) (distilled, 4 steps, CFG 1.0)
- text encoder — [`Qwen/Qwen3-VL-8B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) (via `--llm`) + its `mmproj` vision projector (via `--llm_vision`, editing only)
- VAE — [`Comfy-Org/Boogu-Image`](https://huggingface.co/Comfy-Org/Boogu-Image) (`vae/flux1_vae_bf16.safetensors`, 168 MB — the Flux.1 VAE, from an ungated mirror rather than the license-gated `FLUX.1-dev`)

This is the one model here built for **instruction editing** — *"remove the car"*,
*"make the background a beach"* — rather than text-to-image. Distilled, so it
keeps the 4-step / CFG 1.0 speed profile of Flux.2 Klein. About **17 GB** on disk
(7.4 GB diffusion at Q4_1, or 8.6 GB at Q5_1 on a 12 GB card; 8.7 GB encoder
offloaded to RAM; 0.75 GB vision projector; 0.17 GB VAE).

> **⚠️ These weights predate the 2026-07-08 hotfix — a deliberate choice.** That
> hotfix addressed *"severe image quality degradation and poor performance on
> removal and other editing tasks"* (revisions `hotfix-1k-20260708` /
> `hotfix-1k5-20260708`), but **no GGUF of the fixed revision has been published**
> — both Edit-Turbo GGUF repos predate it (realrebelai 07-01, chfm 07-06), and
> the only hotfix-derived community upload is `mxfp8`, which sd.cpp cannot load
> (checked 2026-07-29). The 4-step speed was worth the trade here.
>
> **If object removal disappoints, two ways out.** Both reuse the same encoder,
> mmproj and VAE — only the DiT changes:
> 1. the **non-Turbo Edit**, never flagged as defective: grab a
>    `boogu-edit-dit-*.gguf` from
>    [`realrebelai/Boogu-Image-Edit_GGUFs`](https://huggingface.co/realrebelai/Boogu-Image-Edit_GGUFs)
>    and point **📂 Local files → diffusion** at it in the same tab (then set
>    25–50 steps at CFG 2–5, per the official Model Zoo);
> 2. the **fixed bf16 weights** on `Comfy-Org/Boogu-Image`
>    (`boogu_image_edit_turbo_hotfix_1k_20260708_bf16.safetensors`, 20.6 GB),
>    quantized locally from the **🔧 Convert to GGUF** tab.

Two more things to know: CFG 1.0 means the **negative prompt is ignored** (the
field stays hidden, as on Flux.2 Klein and Krea 2), and the sd.cpp docs show only
a **single** `-r` reference image for Boogu where Flux.2 takes three — slots 2–3
remain usable but undocumented.

**Upscalers** — [`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) (ESRGAN), `stabilityai/stable-diffusion-xl-base-1.0` + `madebyollin/sdxl-vae-fp16-fix` (creative).

To delete a model, use **🗑️ Delete** in the Model Catalog — shared files
(encoders/VAEs used by another model) are preserved.

---

## Project layout

```
app.py                       # Gradio entry point
config/models.yaml           # catalog: sources, defaults, presets (source of truth)
atelier/
  settings.py                # paths + persisted preferences (userdata/)
  hardware.py                # GPU/RAM detection + optimization profiles
  registry.py                # catalog, file resolution, status, recommendations
  downloader.py              # on-demand Hugging Face downloads
  styles.py                  # system-prompt / style presets
  engine/
    sdcpp.py                 # build/run sd-cli commands (gen, edit, upscale, LoRA)
    generate.py              # generation pipeline (model + hardware + LoRA) + ESRGAN upscale
    outpaint.py              # directional outpaint: canvas plan, mirror fill, composite-back
    video.py                 # LTX-2.3 video pipeline (-M vid_gen): components, 32px/8k+1 grids
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, enhancer, SDXL upscale)
  ui/
    theme.py                 # light theme + CSS
    generate_tab.py · library_tab.py · toolkit_tab.py · outpaint_tab.py · video_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # downloads the stable-diffusion.cpp binary
  _torch_setup.py            # shared PyTorch-CUDA install helpers
  setup_tools.py             # installs PyTorch tools (depth, bg, sam, enhance, upscale)
  tools/run_*.py             # inference runners (subprocess: depth, rembg, sam, enhance, usdu)
```

---

## Managing disk space & uninstalling

The **🧹 Manage & help** tab is the single place to see what the app has
downloaded and to reclaim space. It lists every item with its **size**, grouped
into four categories:

| Category | What's in it |
|---|---|
| **Engines** | `sd-cli` (stable-diffusion.cpp) and the trellis.cpp 3D binary |
| **Toolkit add-ons** | depth, background removal, SAM, prompt enhancer, creative SDXL upscale |
| **Your data** ⚠️ | LoRAs, custom models, generated images/3D, temp files |

Tick what you want to remove, tick **“I confirm”**, then delete. Sizes refresh
after each operation.

- Everything outside **Your data** is **re-downloadable** from inside the app
  (Model Catalog, one-click installers).
- Items marked **⚠️** are *yours* — LoRAs, hand-placed/converted models and your
  generated images. Deleting them is not recoverable from the app.
- **Temp files** (`tmp/`) are always safe to clear.
- Deletion is restricted to paths **inside the project folder** — the tool
  never touches anything elsewhere on your disk, and base folders are recreated
  right after.

### Moving models to another drive

Models don't have to live inside the project folder. In **🧹 Manage & help →
📁 Models location** you can point them at any absolute path — typically a fast
**NVMe** (much quicker loads) or a bigger drive:

- **📦 Move models here** — transfers the existing files, then saves the new
  location. Same drive = instant; across drives = a real copy (can take a
  while). Free space is checked first, and if any file fails to move the
  location is **not** switched.
- **🔗 Point here without moving** — reuse a folder that already contains your
  models (e.g. shared with another install).
- **↩️ Back to the project folder** — restore the default `models/`.

The setting is stored as `models_dir` in `userdata/preferences.json` and applies
on **restart** (several modules resolve the path at import time).

**Moving only *some* items** — a second accordion lets you relocate **selected
items** (e.g. the ~16 GB trellis 3D set) to another drive and leaves a
**junction/symlink** behind. The app keeps finding them at the original path, so
there is **no setting and no restart** involved; a **↩️ Bring back** button undoes
it. Handy to keep image models on the NVMe while parking bulky ones elsewhere.

> The destination drive must stay connected — relocated items become unreachable
> if it isn't. On Windows the link is a directory *junction*, which does not
> require administrator rights.

> Repeatedly loading models does **not** wear out an SSD — flash endurance is
> consumed by *writes* (TBW), not reads. Putting models on your NVMe is exactly
> what it's for.

The same tab carries **📖 in-app documentation for every option** of the app
(generation, settings/hardware, Image → 3D, convert, toolkit, network &
maintenance) — the fastest way to know what a slider actually does.

---

## Troubleshooting

- **“sd-cli binary not found”** → re-run `install.bat`, or download the engine
  manually. On a portable Windows install there is no global `python`; use the
  embedded one:
  `python\python.exe scripts\get_sdcpp.py --variant cuda`
  (fetches the **win-cuda12** build *and* the **cudart** runtime side by side).
- **“No NVIDIA GPU detected”** → check drivers / `nvidia-smi`.
- **Model shows “to download”** → Model Catalog tab → **Download**.
- **Out of memory** → Settings: lower the quantization, enable offload/tiling, or
  reduce the resolution. For very tight setups, try a per-generation preset.
- **A Toolkit tool runs on CPU (very slow)** → its installer prints
  `CUDA: True/False`; if False, fix NVIDIA drivers and reinstall the tool.
- **Creative SDXL upscale OOM** → lower the scale or tile size (it auto-offloads
  under 12 GB, but a huge target can still exceed memory).
- **`ConnectionResetError [WinError 10054]` in the console** → harmless, and
  silenced since. It is a known Python bug on Windows (bpo-39010): when the
  browser drops a connection (F5, tab closed, cancelled image load), asyncio's
  Proactor loop calls `socket.shutdown()` on an already-dead socket and prints
  `Exception in callback`. The request is already finished server-side. `app.py`
  now intercepts it *and* completes the cleanup the raised exception used to
  skip — the socket was actually leaking, which is why Python also logged
  `unclosed transport`. Only connection errors are caught; anything else still
  propagates.
- **A Toolkit add-on fails at import (`DTensor`, `diffusers`, numpy…)** → a
  shared package drifted. Run `maintenance.bat`: it names the offending package,
  then reinstall that add-on. The SeedVR2 installer also ends with a smoke test
  that catches this at install time.

---

## Acknowledgments

This project is just glue around other people's hard work. Heartfelt thanks to
everyone below — all credit for the models and tools goes to their original
authors. Please read and respect each model's own license on its page.

### Engine & framework
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** —
  **leejet** & contributors. The inference engine this whole project rests on.
- **[Gradio](https://github.com/gradio-app/gradio)** — the web UI.
- **[PyTorch](https://pytorch.org)**, **[Hugging Face](https://huggingface.co)**
  `transformers` / `diffusers` / `huggingface_hub` — the optional Toolkit tools.

### Image models
- **Flux.2 Klein** — base model by **Black Forest Labs**; GGUF by
  [leejet](https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF); VAE by
  [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-9B); text encoder
  **Qwen3-8B** by **Alibaba / Qwen team**
  ([official GGUF by Unsloth](https://huggingface.co/unsloth/Qwen3-8B-GGUF)).
- **Krea 2** — base model by **Krea AI**; GGUF by
  [realrebelai](https://huggingface.co/realrebelai/KREA-2_GGUFs); text encoder
  **Qwen3-VL-4B-Instruct** by **Alibaba / Qwen team**
  ([official GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF));
  **WAN 2.1** VAE by **Alibaba / Wan team**, repackaged by
  [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged).

### Upscalers
- **ESRGAN models (GGUF)** collected by
  [wbruna](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) — including
  **Real-ESRGAN** (Xintao Wang et al., Tencent ARC) and community models
  (UltraSharp, foolhardy Remacri, Nomos, LSDIR, NickelbackFS, StarSample…). Credit
  to each upstream author; see the repo for individual sources/licenses.
- **Creative upscale (Ultimate SD Upscale style):** **SDXL** by
  [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0);
  fp16-fix VAE by [Ollin Boer Bohan / madebyollin](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix);
  **ControlNet Tile** by [xinsir](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0).
  The tiled-redraw method is inspired by **Ultimate SD Upscale**
  ([Coyote-A](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)).

### Toolkit
- **Depth Anything V2** —
  [depth-anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) team.
- **RMBG-1.4** background removal — **BRIA AI**
  ([briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4), **non-commercial** license).
- **Segment Anything** — **Meta AI**
  ([facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)).
- **Prompt enhancer** — **Qwen2.5-3B-Instruct** by **Alibaba / Qwen team**
  ([Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)).
- The **Krea prompting guide** informed the Krea 2 enhancer system prompt.

### Built with
- **[Claude](https://claude.ai/code)** (Anthropic) — vibe-coded iteratively in
  natural language.

This is an independent, non-commercial hobby project, **not affiliated with or
endorsed by** any of the above. If you are an author and want a credit corrected
or removed, please open an issue.
