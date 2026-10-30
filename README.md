# Turbo Slop Generator 3000 · Studio

**Removed modules:** Z-Image Turbo, Flux.3 Klein, SDXL creative upscale,
ADetailer, Prompt improver, 10 views / 360°, SVG & layers, Layers and Faces.
They no longer appear in the UI or download catalog. Existing dedicated files
are listed under **System → Manage → Removed features** when present; cleanup
is optional. Shared encoders and VAEs used by active models are retained, as
are generated outputs. SDXL custom checkpoints are marked as personal data.

**TAESD / tiny autoencoders:** optional per-model TAE previews or fast final
encoding/decoding, with shared downloads in Models. Preview-only keeps the
original final VAE; full mode may soften detail. Supports the seven catalog
image models. [Install and choose a mode](docs/taesd.md).

**Choose weights before generating:** every Images workspace now offers an
installed model variant selector, plus encoder and VAE choices. Selections are
saved per model; exact chosen files are passed to generation. Install more
variants in Models, then refresh the generation selectors.

**GPU-aware weight choices:** Models now explains INT8, BF16, FP16, FP32 and
GGUF precision separately from file formats. Load sizes for memory suggestions
on your selected GPU. Krea 2 Turbo offers INT8/BF16 Safetensors
alongside GGUF; saved selections and shared files are retained.
[Precision and memory advice](docs/model-catalog.md#gpu-and-weight-precision).

**Installed model updates:** `update.bat` now checks installed Hugging Face
files after updating the code and engines. `update.bat --models` runs only
that check. It reports changed files and pinned toolkit versions, preserving
the installed weights. [How model checks work](docs/model-updates.md).

**Fibo 2.0:** native **Dev** generation/editing (20 steps, CFG 5) and
**Edit Turbo** reference editing (4 steps, CFG 1), with INT8 ConvRot weights,
shared Flux.2 VAE / Qwen3-VL 4B components and structured JSON captions.
Run `update.bat`, download the chosen variant in **Models**, then use its
**Images** tab. Requires sd.cpp build 985+. Weights are CC BY-NC 4.0;
commercial use requires a BRIA agreement. [Setup and limits](docs/fibo2.md).

**Local media engines:** SeedVR2, VOSR 2.0, LTX 2.5 and COLMAP + Brush are available on
`main` and included in the Studio download below. See
[the setup guide and GPU validation limits](docs/media-engines.md).

**VOSR 2.0:** install the package from **Models**, then open **Tools & 3D →
VOSR 2.0 upscale** for tiled ×1 / ×2 / ×4 image restoration. It uses its own
CUDA 12.6 environment and one 6.97 GB weight bundle. See [VOSR controls and
validation limits](docs/vosr.md).

**Studio — available on `main`.** A calmer, responsive
workspace with six navigation tabs, a local image gallery, lighter previews,
and one shared GPU queue. [Release notes and verification](docs/STUDIO_RELEASE.md).

[Download Studio](https://github.com/KSCorpr/Turbo-Slop-Generator-3000/archive/refs/heads/main.zip)
and extract it into a folder, then use the normal installers below.
With Git:

```bash
git clone --branch main --single-branch https://github.com/KSCorpr/Turbo-Slop-Generator-3000.git turbo-slop-studio
cd turbo-slop-studio
```

> **It adapts to your hardware.** By default the app uses the **best NVIDIA card
> it detects** (or the **Apple Silicon GPU** on a Mac) and tunes itself to it:
> diffusion quantization from VRAM, text encoder offloaded to system RAM,
> flash-attention, CPU offload, VAE tiling. Two advanced options live in
> **Settings**: the **per-generation card profiles** (GTX 10xx → RTX 50xx, one
> click) and **multi-GPU** (which card generates, and whether the text encoder
> moves to a second card). Weight placement itself — including letting sd.cpp
> plan it — is **measured** by the benchmark rather than asked as a question.

> **The interface is in English**, and only in English. It used to be written in
> French and translated through a dictionary; that layer is gone and the source
> strings are English. See [Interface language & theme](#interface-language--theme).

A **local**, modern, lightweight image-generation studio for artists, built on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** (native
CUDA, GGUF). Generate with **Flux.2 Klein 9B**, **Qwen Image 2.1 Turbo**,
**MOSS SFT**, **Fibo 2 Dev / Edit Turbo** and **Krea 2 Turbo / Realism**, with an
on-demand model catalog, automatic optimization for your RTX card, LoRA, native
resolution presets, saved styles, multi-reference image editing, SeedVR2 and
VOSR restoration, and a utility toolkit.

A clean web UI without a node editor. SeedVR2 ConvRot uses Comfy internals
in its isolated command-line runtime; image generation still uses sd.cpp.
A future replacement of sd.cpp is described in [the backend migration plan](docs/backend-migration.md).

**Krea 2 Realism:** [RudySen's full checkpoint](https://huggingface.co/RudySen/Krea2-Realism-Checkpoint)
is available in **Models → Krea 2 Realism** and **Images → Krea 2 Realism**.
Choose **INT8 ConvRot (~13.32 GB, default)** or the **original FP8 (~13.14 GB)**;
the V3 realism LoRA is already merged into these weights. The existing Krea 2
Wan 2.1 VAE, Qwen3-VL-4B encoder and optional vision projector are reused.
Defaults follow the author: **Euler + Simple, 6 steps, CFG 1, 1152×1792 (~2 MP)**.
Smaller Krea resolutions remain available. Both checkpoints exceed a 12 GB
card's VRAM before compute buffers; keep **automatic system placement** enabled
to use RAM. File size is not a speed or memory-use guarantee.
Update with `update.bat` / `./update.sh`; the app uses build **968+** as its
supported baseline for this checkpoint. INT8 ConvRot needs an official build
using sd.cpp's patched GGML; a custom build with `SD_USE_UPSTREAM_GGML=ON` cannot
load it. The author's W4A8 `asym_w4a8_int8` format is unsupported by sd.cpp and
is excluded from the file menu. Actual GPU output and speed remain unverified.

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

**Six main tabs** keep the core workflow visible. Choose a model inside
**Images**, write a prompt, then press **Generate** or **Ctrl/Cmd+Enter**. Size,
quality, seed and batch stay close to the prompt; detailed sampling, styles and
LoRA settings expand when needed.

| Tab | What it does |
|---|---|
| **Images** | Flux.2 Klein, Qwen Image 2.1 Turbo, [MOSS-Image SFT](docs/moss-image.md), Fibo 2, Krea 2 and Xanax; sampling controls follow each model's requirements. |
| **Video** | LTX 2.5 text/image-to-video generation and AVI recovery. |
| **Tools & 3D** | Image-to-prompt, depth, cutouts, upscaling, restoration, Outpaint and textured 3D. |
| **Gallery** | Browse local outputs, search filenames, download originals and reuse prompts; 24 thumbnails per page. |
| **Models** | Searchable catalog for image models, Toolkit, video and 3D; shared weights and on-demand downloads. [Model storage guide](docs/model-catalog.md). |
| **System** | Settings, disk management, help and GGUF conversion. |

---

## Table of contents

- [Install](#install)
  - [Updating the app itself](#updating-the-app-itself)
  - [Maintenance](#maintenance)
- [Quick start](#quick-start)
- [Model catalog](#model-catalog)
- [Qwen Image 2.1 Turbo](#qwen-image-21-turbo)
- [Generation options](#generation-options)
- [Xanax tab](#xanax-tab)
- [Hardware & optimization](#hardware--optimization)
  - [Samplers & schedulers](#samplers--schedulers)
  - [Multi-GPU](#multi-gpu)
  - [One shot per image, and nothing kept warm](#one-shot-per-image-and-nothing-kept-warm)
  - [Updating the engines](#updating-the-engines)
  - [Interface language & theme](#interface-language--theme)
- [Upscaling](#upscaling)
- [Outpaint](#outpaint)
- [Text / Image → 3D](#text--image--3d)
- [Toolkit](#toolkit)
  - [Image → prompt](#-image--prompt)
- [Convert to GGUF](#convert-to-gguf)
- [Managing disk space & uninstalling](#managing-disk-space--uninstalling)
- [Why it only listens on 127.0.0.1](#why-it-only-listens-on-127001)
- [Distributing a portable package](#distributing-a-portable-package)
- [Models & sources](#models--sources)
- [Project layout](#project-layout)
- [Gradio version](#gradio-version)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Install

### Windows (RTX cards)
```bat
install.bat      ::  portable Python + dependencies + GGUF engine (CUDA)
run.bat          ::  launch the UI at http://127.0.0.1:7860
update.bat       ::  the ONLY update button: code, cleanup, both engines
```
Windows and Linux use the **CUDA** build; macOS uses the **Metal** one. The
engine variant is derived from the platform, so no flag to remember.

### Linux (NVIDIA)
```bash
./install.sh
./run.sh
```

### macOS (Apple Silicon)
```bash
./install.sh
./run.sh
```
The same script; it detects the platform and fetches the **Metal** build of
stable-diffusion.cpp instead of the CUDA one. What differs on a Mac:

- **Unified memory, not VRAM.** The GPU addresses the same memory as the CPU, so
  the app reports "~75% of RAM addressable by the GPU" rather than inventing a
  VRAM figure. **RAM offload is switched off** — offloading to a memory the GPU
  is already using saves nothing and only adds copies.
- **The encoder budget is tighter than on PC.** On a PC the text encoder is
  offloaded to system RAM and competes with nothing; here it shares one pool with
  the diffusion model, so the encoder quantization is picked from what's left
  rather than from total RAM.
- **Toolkit add-ons run on MPS** (Metal) via PyTorch, with
  `PYTORCH_ENABLE_MPS_FALLBACK` set so an operator MPS lacks drops to CPU instead
  of killing the run.
- **🧊 Text / Image → 3D is unavailable.** trellis.cpp ships a Windows-CUDA binary only —
  no Apple Silicon build, no Metal path. The tab says so instead of offering an
  installer that would find nothing. SeedVR2 ConvRot requires NVIDIA CUDA and is unavailable on MPS.
- **16 GB is the realistic floor**, 24 GB+ comfortable. Intel Macs are not
  supported: upstream publishes no Intel build.

> The install does **not** download any models. You fetch them on demand from the
> **Model Catalog** tab (like a media library). Everything stays inside the
> project folder.

Generation runs through stable-diffusion.cpp (no heavy PyTorch for image
generation). **PyTorch is only installed on demand** for the optional Toolkit
tools (depth, background removal, SAM and image → prompt),
each via its own one-click installer.

### Updating the app itself

**`update.bat` is the way** (`./update.sh` on Linux/Mac). It downloads the
current code from the release channel in `config/update-channel.json` and applies
it in place. Studio follows **main**; releases without that file also use
**main**. In a Git checkout, the same launcher fetches and fast-forwards the
current tracking branch without switching branches or using ZIP extraction.
If local code blocks the update (for example after ZIP updates over a clone),
it automatically saves that code in a Git stash before continuing. Only modified
code and untracked files colliding with the incoming version are saved;
unrelated untracked files and data folders stay in place. Double-click the same
launcher for Git and ZIP installs; no manual `git pull` or stash command is needed.
The launcher uses portable Python when present, then an existing project venv,
then Python from PATH.
Do not run `git stash pop` afterwards: it would reapply the old code.
Diverged histories, detached commits and unfinished Git operations require
manual resolution; the updater never forces a reset. The code step leaves
your models, outputs, settings, portable Python and installed engines alone;
the later engine step updates sd.cpp in `bin/` with a rollback copy.

**It is the only update button, and it does four things in order.** Updating
the code alone was never enough. There used to be `update-engine.bat`, because
the new code expects an `sd-cli` option the installed engine has never heard of;
`update-trellis.bat` for the 3D engine; and `maintenance.bat` to erase what a
removed feature left behind. **Four buttons, in an order nothing announced,
three of which everyone forgot** — so: an engine a release behind, a folder that
only grows, and failures whose cause was an update half-done weeks earlier.

There is one now:

1. **the code**, from GitHub;
2. **the cleanup** — what removed features left behind, then a check that
   everything compiles and the catalog holds together. It *measures* first,
   shows the total, and **asks** before deleting anything. Answer no (or run it
   with no terminal attached) and nothing is deleted; the total stays on screen;
3. **the sd.cpp engine** (images *and* video), brought in line with that code;
4. **the 3D engine** (trellis.cpp) — the binary only, and **only if it is
   already installed**. The ~16 GB of 3D weights are never re-downloaded, and
   someone who has never opened the 3D tab is not handed a download they did
   not ask for.

**The order is the point.** Everything after step 1 has to run on the *new*
code: it is the code that declares which options `sd-cli` must know and which
files a removed feature left behind. And the two engines come last because they
are the only steps that touch the network — everything sayable offline has
already been said by the time a connection drops.

```bat
update.bat              ::  all four, in order
update.bat --clean      ::  the cleanup only
update.bat --engine     ::  the sd.cpp engine only
update.bat --trellis    ::  the 3D engine only
update.bat --code       ::  the code only
update.bat --check      ::  show what the code would change, write nothing
update.bat --rollback   ::  undo the last code update
```

Those four flags are what replaced the three separate launchers: doing one part
without the others is a real need, it just did not deserve four files.

In a Git checkout, `--check` compares cached refs without fetching or changing
files. `--rollback` is reserved for ZIP installs; Git code backups remain in
`git stash list` for inspection and manual recovery.

Steps 2–4 run in **one fresh process**, not an import — `update.bat` has just
rewritten `maintenance.py` on disk, and an import would run the version it just
replaced: the old table of removed features, the old list of expected engine
capabilities. One call rather than three, because `maintenance.py` already
sequences the cleanup before the engines; splitting it here would replay its
checks three times and make their order depend on this file. They are skipped
entirely when there is nothing to tidy: after `--check` (which promises to write
nothing) and after a failed update that rolled itself back.

What it does that dropping a ZIP over the folder cannot:

- it writes **only files that actually differ**, and lists them;
- it **deletes what disappeared** from the project — but only files it installed
  itself, tracked in `userdata/app-update.json`. A file it never wrote is not
  its business;
- it **backs up everything it replaces**, so `--rollback` undoes the update;
- it **refuses a suspicious download** (a proxy's HTML error page is a perfectly
  readable "zip") and any archive that does not contain the app;
- if the updated code **does not compile**, it restores the previous version by
  itself;
- it purges `__pycache__`, because a `.pyc` of a deleted module stays importable.

Close the app first: Windows cannot replace a file that is open.

`maintenance.bat`, `update-engine.bat` and `update-trellis.bat` are **gone**.
An installed copy still has them sitting in its folder — extracting a ZIP over a
folder never deletes anything — so they are declared in `REMOVED_FEATURES` and
the first `update.bat` deletes them. That matters: left there, they still work
and still do a quarter of the job, without ever saying so.

### Maintenance
If you instead update by extracting the repo ZIP over your existing folder
(keeping `python/`, `bin/`, `models/`…), copy-paste **adds and overwrites files
but never deletes** the ones removed upstream — they linger as orphans, and
stale `__pycache__` can confuse Python. After each copy-paste update, run:
```bat
update.bat --clean      ::  Windows   (./update.sh --clean on Linux/Mac)
```
It deletes the **code** of removed features, purges `__pycache__` and `tmp/`,
then verifies that everything compiles, the model catalog is valid, and the
dependencies + `sd-cli` engine are present. It never touches `models/custom/`,
`loras/`, `outputs/`, `userdata/`, `python/` or `bin/`.

**The engine updates separately from the code, and that is the trap.** Copying
the repo over the old one does not touch `bin/`, so the app can start asking for
a `sd-cli` option the installed binary has never heard of — and you find out when
a tab fails. Maintenance therefore checks the engine's *capabilities*, not just
its presence: it parses `sd-cli -h` against the options the current code actually
needs, and names the feature rather than the flag ("`--hires` missing" tells
nobody anything; "the HD tab will not work" does). To fix everything in one go:

```bat
update.bat               ::  code, cleanup, sd.cpp engine, 3D engine
```

`update.bat --engine` does just the sd.cpp one. Updates are transactional: the
archive is unpacked into a staging directory, checked with `sd-cli -h`, and only
then swapped into `bin/`. The last known-good engine stays in
`.engine-previous/`; use `rollback-engine.bat` (or `./rollback-engine.sh`) to
restore it immediately. A failed download, unsafe archive or failed smoke test
leaves the installed engine untouched.

**Orphan modules are found generically.** Beyond the hand-declared
`REMOVED_FEATURES`, maintenance walks the import graph from `app.py` and
`scripts/`, and reports any module under `atelier/` that nothing reaches. That
catches leftovers from versions nobody remembered to declare. It reports rather
than deletes: a dynamically loaded module would show up here wrongly.

**Data left behind by removed features is measured, not deleted.** When a feature
goes away it leaves gigabytes on disk — downloaded weights, cloned repos, model
folders no longer in the catalog. Erasing those silently is not the script's call,
so it reports each one with its size and a single recoverable total:

```bat
update.bat --clean                     ::  shows the total, then asks
python scripts\maintenance.py --purge  ::  deletes without asking
```

Three things are checked, and none of them relies on a hand-kept list of files:
orphan **add-ons** in `tools_repo/` are whatever no longer matches an add-on in
the code, orphan **models** are whatever the catalog no longer references, and
removed features declare their own leftovers in `REMOVED_FEATURES` at the top of
`scripts/maintenance.py` — adding an entry there is the only step needed when
something is dropped.

**What a copy-paste update does and does not refresh**

| | In the ZIP? | Refreshed by copy-paste |
| --- | --- | --- |
| App code, `config/models.yaml`, docs | yes | **yes** |
| Engine binary (`bin/`) | no | no — `update.bat` brings it in line as step 3 |
| Models, LoRAs, outputs, prefs (`models/`, `loras/`, `outputs/`, `userdata/`) | no | no — kept, which is the point |
| Toolkit add-ons (`tools_repo/`) | no | **no — and this one bites** |

That last row matters. An add-on is not just a folder of weights: its installer
also pins Python package versions **shared with every other add-on**. When a
release changes *how an add-on installs*, updating the app leaves the installed
add-on frozen in its old state, and you only find out at the next use. So:
**after updating, if an add-on misbehaves, re-run its one-click installer** —
weights already on disk are not re-downloaded.

`maintenance` reports leftovers from removed features and tells you exactly how
much disk they hold, so you don't have to guess.

It also **reports any file from the update manifest that has gone missing** —
the diagnostic that was absent the day a module vanished and the app stopped
starting with an `ImportError` naming the module but not the cause.

---

## Quick start

1. Run `install.bat` / `./install.sh`, then `run.bat` / `./run.sh`.
2. Open the **📚 Model Catalog** tab and download **Flux.2 Klein 9B** (or **Krea
   2 Turbo**). Quantization is picked automatically for your VRAM/RAM.
3. Go to the model's **generation tab**, type a prompt, click **🎨 Generate**.

---

## Model catalog

**📚 Model Catalog** is the media library: nothing is downloaded at install
time, everything is fetched on demand from here.

Each entry shows whether the model is ready (● installed · ○ to download).
Click **Load weights and sizes** for the actual filenames and file sizes
published by Hugging Face, then select the diffusion and text encoder weights
for that model. The selections persist across restarts. With **Automatic**, the
target quantization comes from VRAM (diffusion) and RAM (encoder); if that rung
does not exist, the downloader chooses the nearest lower one and says so.
An already installed compatible encoder is reused in Automatic mode even if
the preferred quantization has changed, avoiding another multi-GB copy. An
explicit weight choice always takes precedence.

The seven catalog image models use four encoder families:
Qwen3-8B (Flux.2), Qwen3-VL-8B (Qwen Image 2.1 Turbo), Qwen3-VL-4B (Krea 2 / Fibo),
and Qwen3.5-9B (MOSS). Similar names do not make their
weights interchangeable. Compatible files from the *same* Hugging Face repo
are stored only once and shared by any model that uses them.

- **⬇️ Download** fetches every component the model needs (diffusion, VAE, text
  encoder, and the vision projector for edit models) into `models/`, with a
  live log.
- **🗑️ Delete** removes a model's own files and **keeps shared ones**: an
  encoder or VAE used by another installed model is never taken out from under
  it.
- The image catalog is `config/models.yaml` (sources, defaults and presets).

The tools that live under 🧰 Toolkit are **not** here: each one carries its own
one-click installer in its own tab, because each pulls a different Python
dependency set.

### Qwen Image 2.1 Turbo

Run **`update.bat`** (`./update.sh` on Linux/Mac), then download **Qwen Image
2.1 Turbo** in **Models**. The catalog uses the actual Turbo checkpoint from
[`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models):
**INT8 ConvRot** by default (~7.26 GB), with **BF16** selectable (~14.2 GB).
The earlier Qwen 2.1 diffusion checkpoint is not a Turbo checkpoint. Old saved
diffusion selections are therefore not reused. The Qwen 2.1 VAE and Qwen3-VL-8B
GGUF encoder remain compatible and are reused when installed; the vision
projector is needed only for editing. Older Qwen Image and Wan VAEs are incompatible.

Turbo uses **8 steps, CFG 1, Euler** and the exact sigma sequence in
[Qwen's model configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json):
`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`,
followed by terminal zero for sd.cpp. These controls are fixed in the interface.
The engine must support `--sigmas`; an older binary displays an update instruction.
Official build 896 or newer is needed for transparent inputs, and build 901 or
newer for per-step live previews. These checks do not replace a GPU image-quality
test of the new Turbo weights.

Start at **1024×1024** on 11–12 GB cards. Up to **three reference images** use
native editing through `-r`, with no editing LoRA or strength slider. Image sizes
follow a 32-pixel grid. For transparency, request an RGBA image with transparent
background and save PNG. Check the **Qwen Research** license before commercial use.

## Generation options

Generation tabs share prompt, styles, size, sampler and preview controls.

### Prompt & system style
- **Prompt** — your description. For **edit models** (Flux.2 Klein, Qwen 2.1 Turbo) describe the
  *modification* to apply to the reference image.
- **Negative prompt** — hidden on the current catalog models at CFG 1.0,
  where it has no effect. Custom models can expose it when guidance is enabled.
- **System / style prefix** (🎨 Styles → *Custom preset*) — a prefix prepended
  to every prompt. Save
  reusable styles to a dropdown (persisted in `userdata/`). Styles are **global**:
  saved once, available in every generation tab. A few are **bundled** with the
  app in `config/style_presets.json` — they survive updates and cannot be
  deleted, but saving a style under the same name creates your own version, which
  takes precedence; deleting that restores the original. Two separate buttons,
  so nothing is lost by accident: **✖️ Stop applying** only detaches the style
  from the next generations and keeps the preset, while **🗑️ Delete this preset
  (permanent)** erases it and asks for a second click to confirm. Bundled today:
  **📷 Provincial France 1995-2005 (amateur)**, a transcription of a
  "mundane amateur snapshot, provincial France, always overcast, no
  post-processing" brief.
- **📷 Krea 2 photo styles** (🎨 Styles → *Photo*) — a bundled bank of **139 stackable
  photographic styles** (quality, lighting, lens, film stock, mood…), grouped by
  category in a **multi-select** dropdown. Your prompt subject is inserted into
  each selected style (`{prompt}` template), and multiple styles chain their
  descriptions after the subject so you can combine axes. Style negatives are
  merged into the negative field **only when the model uses them** (CFG > 1);
  on distilled CFG 1.0 models they are dropped. Bank © *ghleg* — MIT
  ([aoleg/Photographic-styles-and-wildcards-for-Krea-2](https://github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2)),
  shipped as `config/krea2_styles.csv`.
- **🎨 Krea artistic styles** (🎨 Styles → *Artistic*) — a bundled bank of **397 stackable
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
- **Qwen Image 2.1 Turbo** — native multi-reference editing via `-r` as above; the
  Qwen3-VL-8B vision projector is required for editing, no LoRA. Uploaded PNG
  alpha is preserved for transparent image editing.
- **Krea 2 Turbo — ✏️ Edit mode (Ostris Edit)** — tick **✏️ Edit mode** to pass
  the image as a **context reference** (style transfer, subject reference, edits)
  instead of an img2img starting point. This requires a **Krea 2 edit LoRA**
  (e.g. HF repo [`ostris/krea2_turbo_style_reference`](https://huggingface.co/ostris/krea2_turbo_style_reference) —
  add it via the LoRA panel) and a **recent sd.cpp engine** (`update.bat`;
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

- **Qwen 2.1 Turbo** — a 32-px grid with 1024-class presets and native 2K formats
  (2048², 2400×1792, 2752×1536…).

Pick a ratio from the dropdown, or choose **Custom (sliders)** for free width /
height (256–2048, step 16 for most tabs; 256–2816, step 32
for Qwen). Loading a reference image auto-fits width/height to its aspect.

### Sampler / scheduler / steps
- **Preset** — vetted combos per model (e.g. Flux.2 Klein → 4 steps / CFG 1.0 /
  euler + simple). Selecting one fills sampler, scheduler, steps and CFG.
- **Sampler** — all samplers supported by sd.cpp (euler, dpm++2m, res_multistep…).
  Each entry is **annotated for the model of the current tab** — ⭐ recommended,
  no mark = usable, △ poorly suited, ⚠️ discouraged — and the card below the menu
  spells out what the selected one does, its ✅ upside and its ❌ downside.
- **Scheduler (sigmas)** — auto (model default), karras, simple, exponential…
  Annotated and documented the same way.
- **📖 Why half of this menu is useless here** — a fold-out that explains the
  verdicts from the model's own properties. See
  [Samplers & schedulers](#samplers--schedulers) for the full reasoning.
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
- **Live preview** — each sampling step writes its own preview and updates the
  image above the gallery. Qwen Image 2.1 Turbo requires sd.cpp build 901 or newer for its
  RGBA latent preview (`update.bat`). Final images appear in the gallery when
  generation finishes.
- **Seed** — the selected image's seed shows in a copy-button box; **Reuse this
  seed** drops it back into the seed field. Clearing the seed field resets it to -1.
- **Send to Toolkit** — push the selected image straight into a Toolkit tool
  (depth, background removal or SAM).
- **Saved prompts** — every image gets an A1111-style `.txt` sidecar in
  `outputs/` with the prompt, negative, model, sampler/scheduler, seed and size.

## Xanax tab

Choose **Krea 2 Turbo** or **Flux.2 Klein**, then enter an English sentence
about your day. Xanax adds its fixed amateur-photo style: provincial France,
1995–2005, overcast daylight, cheap camera and no post-processing. It does not
rewrite or translate your input. The random-day button supplies an example.
The generated prompt and seed are shown with the result.

## Hardware & optimization

### Automatic optimization
The app detects your GPU (via `nvidia-smi`) and RAM, then chooses on its own:
- **diffusion quantization** by VRAM
  (`<8 GB → Q4_K_S`, `8–12 → Q4_K_M`, `12–16 → Q5_K_M`, `16–24 → Q6_K`, `≥24 → Q8_0`);
- **encoder quantization** by RAM (the text encoder is offloaded to RAM, so it
  costs no VRAM);
- **flags**: flash-attention (Turing / RTX 20xx and newer), CPU offload, VAE
  tiling, CLIP/VAE on CPU — enabled progressively as VRAM gets tighter;
- Pascal cards (GTX 10xx) → flash-attention disabled automatically (it's slow there).

The engine log also reports **how fast your models are read**. sd-cli already
prints the read time and the buffer size, on two separate lines; the app does
the division and, below 300 MB/s, says plainly that the models sit on a
mechanical drive and what that costs per image. The case that prompted it read
8.2 GB at **105 MB/s** — 65 s lost on every single image.

Multi-GPU: the largest card is used by default (tensor cores break ties), changeable in **Settings**
(see [Multi-GPU](#multi-gpu)).

These map to stable-diffusion.cpp flags: `--diffusion-fa` (CUDA: faster + less
VRAM), `--offload-to-cpu`, `--vae-tiling`, plus GGUF quantization. On a recent
engine the app also separates **where computation runs** (`--backend`) from
**where weights live** (`--params-backend`). This matters on a second GPU behind
a slow PCIe link: an encoder can keep its weights on that GPU instead of staging
them from RAM. The old CLIP/VAE-on-CPU flags remain compatibility fallbacks.

### Automatic system settings by model
**Settings → System settings** starts on **Automatic for each model**. At every
generation, the app checks the selected model's installed image weights, the
VRAM actually free, the output size and any reference image. When the image
weights fit with compute headroom, it keeps diffusion and VAE on the GPU and
keeps the text encoder's weights in RAM while its computation runs on the GPU.
When they do not fit, it asks a recent sd.cpp to place the modules automatically
with a VRAM budget. Older engines use RAM staging and layer streaming if the
respective options exist. The system choices table shows a preview for every
installed model; the available VRAM is measured again on each run. Installing
a model or changing a GGUF weight can therefore change its system plan without
touching Settings.

**Balanced** needs no adjustment. The optional **More memory headroom / More
detail** priority shifts the automatic GGUF target one notch. An explicit GGUF
choice in the Model Catalog still wins. The app cannot make a model with
insufficient RAM or disk space fit; automatic placement can be slower when
weights have to leave VRAM.

A short **"Something specific going wrong?"** block maps symptoms to actions,
and two of its three answers deliberately point *elsewhere*: "too slow" is the
step count and image size in the generation tab, "images look dull" is the
prompt and the styles. Pretending everything is solved in Settings is what sent
people hunting through checkboxes in the first place.

**There is no Save button.** Every control applies immediately and says so, next
to itself. A Save button is one more chance to wonder whether the change was
taken into account — and the theme already saved itself, which made the rest
ambiguous.

**Manual / measured configuration** restores saved expert flags, quantization
and GPU placement. The benchmark still compares configurations without saving
them and its **Apply the fastest** button switches to Manual (see
[Measured hardware profile](#measured-hardware-profile)).

Everything sd.cpp exposes for deliberate overrides lives under the folded
**🔧 Expert** section. Changing any Expert control switches to Manual;
returning to Automatic preserves the previous manual settings for later.

### Per-generation presets
`hardware.GENERATIONS` still holds a curated profile per RTX generation (**GTX
10xx → RTX 50xx**): real VRAM of the selected GPU, diffusion quant with a small
speed/quality bias per generation, encoder quant from RAM, memory flags
(flash-attention off on Pascal, VAE tiling / CPU offload on tighter cards). It is
no longer a row of five buttons in the interface — the three-notch choice covers
the same ground with one decision instead of five. `generation_profile()`
remains available for diagnosing and comparing older hardware presets;
Automatic uses the detected GPU, RAM and the selected model's installed files.

### Machine profiles — manual overrides
Two measured machine profiles remain available when **Manual** is selected and
the corresponding cards are present. The new automatic mode does its placement
per installed image model, without applying one fixed profile to every model:

| Profile | Cards | What it sets that the automatic profile cannot guess |
|---|---|---|
| 🏠 **Home** | RTX 3060 12 GB + GTX 1080 Ti | the Pascal card **never encodes generation text**; it runs Image → prompt and holds weights |
| 🏢 **Office** | RTX 2080 Ti | the encoder's **weights stay in RAM**, and the engine gets a **compute budget** |

Both are measurements, not preferences. Getting the first wrong costs **38 s of
prompt encoding per image** (see below). Getting the second wrong produces
`cannot make enough memory available … workspace capacity check` part-way
through an image: 11 GB has to hold the diffusion weights *and* room to compute,
and without `--max-vram` the engine cuts its graph with no target to aim at.

A profile appears only when its cards are detected — a button offering a profile
for hardware you do not have is a button that lies. VRAM is checked as well as
the name: an RTX 3060 Ti 8 GB is not an RTX 3060 12 GB, however close the names
look.

### Multi-GPU
When **two or more GPUs** are detected, a **Multi-GPU** accordion appears with a
single mutually-exclusive strategy:
- **Single card (recommended)** — everything on the generation GPU, text encoder
  offloaded to RAM. The most reliable.
- **Text encoder on the 2nd card** — diffusion + VAE stay on the main GPU, the
  text encoder runs **and keeps its weights** on the other card (`--backend` +
  `--params-backend …,te=cudaN`). Frees VRAM on the main card without repeatedly
  staging the encoder through system RAM.
  **Refused automatically when that second card has no tensor cores** (Pascal,
  GTX 16xx) while the generation card does. Measured cost of getting this wrong:
  **38 s of prompt encoding per image** on a GTX 1080 Ti, against ~1 s on the
  RTX 3060 next to it — consumer Pascal runs fp16 at 1/64 of its fp32 rate, and
  prompt encoding is one big fp16 matmul. The card is still excellent at
  *holding* weights; it is bad at computing on them. In that case the encoder's
  weights go to RAM and its computation to the generation card
  (`--params-backend` sets residency, not where work happens), which costs no
  VRAM at all. The benchmark can still measure the refused placement on purpose.
- **Text encoder computed on the 2nd card, weights in RAM** — a compatibility
  fallback kept as a measurable option. It can win on unusual topologies, but
  normally loses to resident weights because it crosses PCIe repeatedly.
- **Auto-fit** — `--auto-fit on`. **It is no longer a multi-GPU feature**, and
  this section used to say it was. Current sd.cpp picks **one** GPU for
  diffusion / encoder / VAE *computation* — the one with the most free memory —
  then places the *weights* on that GPU, in RAM, on another GPU or on disk, in
  that order, according to what fits. On a single 11–12 GB card that tiered
  placement is exactly the problem we otherwise hand-tune, which makes it worth
  measuring rather than assuming.
  Two things follow. It is **on by default** when the flag is omitted, so the
  app sends `off` explicitly when you have not asked for it. And an explicit
  `--backend` or `--params-backend` **disables it regardless of argument
  order** — so the encoder-split placement, which sets one,
  silently win over it. That is why the app never sends both.
  It remains memory-aware, not topology-aware: on a mismatched pair or a PCIe
  x4 secondary slot, benchmark it instead of assuming that more aggregate VRAM
  means more speed.
  **So the benchmark measures it**, on every machine including single-GPU ones,
  as the profile `single-autofit` — “placement left to sd.cpp”. It sits
  *second* in the list, behind the current behaviour: profiles separated by
  less than the measured noise are resolved in favour of the earlier one, so a
  tie can never silently change how your machine runs. If it wins, **Apply the
  measured profile** turns it on; nothing here is a new checkbox to understand.
  One consequence inside the app: under auto-fit the out-of-memory retry cannot
  be “put the text encoder in RAM” any more — that shortcut is translated
  upstream into a `--backend` assignment, which *disables auto-fit*, so the
  retry would have relaunched the command that had just failed. It now deepens
  the budget instead (`--max-vram -3`, i.e. free memory minus 3 GiB), which
  makes the planner move a whole module down its own ladder.
  **The flag's shape changed upstream**: it was a bare switch and now requires
  `on` or `off`. Sent bare to a recent engine it would swallow the next
  argument as its value, so the binary's help text is parsed rather than its
  version guessed. `--split-mode` is no longer sent alongside it either —
  upstream states that auto-fit does not select per-layer/row multi-GPU
  computation, so the option had no effect there.

A separate **Image → prompt GPU** can also be chosen (the vision-language model runs there;
image generation always stays on the generation GPU). The GPU
picker and these strategies live in **Settings**.

### Single image to Gaussian splats

**TripoSplat and Idol** are available in **Tools & 3D → Image → splats** and
in the Models catalog. Import one object image, install the selected model,
then export a Gaussian PLY (plus SPLAT for TripoSplat or camera JSON for Idol).
These are generated Gaussian objects, not polygon meshes. Idol requires prior
Hugging Face access to its DINOv3 and FLUX.2 image encoders. Both use an isolated
CUDA runtime; the main installation stays light.
See [installation, GPU precision, downloads and validation limits](docs/image-splats.md).

### Samplers & schedulers
The **Quality preset** menu keeps the existing baseline settings first and adds
**28 experimental sampler / scheduler combinations across all nine image
models**. Each alternative names its sampler, scheduler and step count, with
a short explanation displayed with the menu. Selecting a preset applies its
sampler, scheduler, steps and CFG together and restores the native automatic
flow shift, clearing a previous manual shift. No extra weights are needed.
See the [per-model preset table and comparison advice](docs/sampling-presets.md).

Alternatives are comparison starting points, not measured quality upgrades.
Keep the seed, prompt and resolution unchanged to compare them. Heun and Res 2S
usually need more network evaluations per step than Euler. The recommended
presets and benchmark defaults are unchanged; new engine features may require
`update.bat`.

Qwen Image 2.1 Turbo keeps **CFG 1, eight steps and the exact official custom
sigma sequence** for every preset. Its experimental presets change only the
sampler (Heun, Res 2S or DPM++ 2M); the generic scheduler and flow shift remain
fixed. Outpaint and the benchmark retain their official Euler settings.
The comparison below concerns **Flux.2 Klein and Krea 2 Turbo**.

#### Why most of the menu does not apply here
Three properties of the two models compared below — read off sd.cpp itself —
decide almost every verdict, and they rule out whole families at once:

1. **They are flow-matching models.** sd.cpp runs both in `FLUX_FLOW_PRED` /
   `FluxFlowDenoiser` (Krea 2 with a flow shift of 1.15). **Karras** and
   **Exponential**, which gain a lot on SD 1.5 / SDXL, were designed for EDM
   epsilon-prediction diffusion: their sigma spread does not match this
   trajectory.
2. **They are distilled at CFG 1.0.** There is no guidance to correct, so the
   whole **CFG++** family has nothing to do — and the negative prompt is ignored
   whichever sampler you pick.
3. **They run in very few steps** (4 for Flux.2 Klein, 8 for Krea 2 Turbo).
   **Ancestral/stochastic** methods re-inject noise that never gets reconverged;
   **multistep** methods need a history of evaluations that barely has time to
   exist.

**LCM** and **TCD** are not general-purpose options either: they are the
samplers of models distilled *by those methods*, which neither of ours is.

#### The verdicts, at a glance
⭐ recommended · ✓ usable · △ poorly suited · ⚠️ discouraged
*(in the menu itself, "usable" simply carries no mark)*

| Sampler | Flux.2 Klein (4 steps) | Krea 2 Turbo (8 steps) |
| --- | :---: | :---: |
| `euler` | ⭐ | ⭐ |
| `res_2s` · `euler_ge` | ✓ | ✓ |
| `heun` · `dpm++2m` · `dpm++2mv2` · `ipndm` · `ipndm_v` · `res_multistep` | △ | ✓ |
| `dpm2` · `ddim_trailing` · `lms` | △ | △ |
| `dpm++2m_sde` · `dpm++2m_sde_bt` · `er_sde` | ⚠️ | △ |
| `euler_a` · `dpm++2s_a` · `lcm` · `tcd` · `euler_cfg_pp` · `euler_a_cfg_pp` | ⚠️ | ⚠️ |

| Scheduler | Flux.2 Klein | Krea 2 Turbo |
| --- | :---: | :---: |
| `auto` (engine default) | ⭐ | ⭐ |
| `flux2` | ⭐ | △ |
| `discrete` | ✓ | ⭐ |
| `smoothstep` · `sgm_uniform` · `simple` · `logit_normal` | ✓ | ✓ |
| `ays` · `gits` · `kl_optimal` | △ | ✓ |
| `flux` · `beta` · `bong_tangent` | △ | △ |
| `karras` · `exponential` | ⚠️ | △ |
| `lcm` | ⚠️ | ⚠️ |

`auto` means *don't pass `--scheduler`*, so the engine picks: `flux2` for
Flux.2 Klein, `discrete` for Krea 2 — which is why those two are also marked ⭐
on their own model.

**What is worth actually trying:** on Flux.2 Klein, almost nothing beyond
`euler` — 4 steps leave no room, and `res_2s` is the only other one accurate
without a history to build. On Krea 2 Turbo the margin is wider: `res_multistep`,
`dpm++2m` and the `ays` scheduler (designed for small step budgets) deserve a
side-by-side run at a fixed seed.

These verdicts are **reasoned from the models' properties, not measured on a
benchmark**. They say where to aim your experiments, not what your eye will
prefer. The source of truth is [`atelier/sampling.py`](atelier/sampling.py);
`tests/test_sampling_docs.py` checks every key against the engine's own list so
a typo can't reach the menu.

### Cache acceleration (experimental)
**Settings → 🗃️ Cache acceleration** exposes sd.cpp's step-caching
([`caching.md`](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md)):
`easycache`, `dbcache`, `taylorseer`, `cache-dit` or `spectrum`, plus a free-form
option (e.g. `threshold=0.2`). It reuses near-identical computations across
diffusion steps. Honest note: it pays off mostly above ~10 steps — on 4–8-step
distilled models the gain is small and artifacts are possible, hence **off by
default**. Requires a recent engine (`update.bat`).

### Measured hardware profile
**Settings → 🧪 Measure this machine** runs a fixed 512×512 / 4-step / seed
424242 generation through every sensible placement: main GPU with RAM staging,
**placement left to sd.cpp** (`--auto-fit on`), and — with a second card —
encoder resident on the second GPU plus the staged dual-GPU path. The first two
run on any machine, so a single-card PC has something to compare as well; that
is the whole reason auto-fit is here rather than behind a checkbox.

**How it measures matters more than what it measures.** Each placement gets one
**discarded warm-up run** followed by **three timed runs**, and the reported
figure is the **median**, not a single sample. Without the warm-up, the first
placement tested pays the cold-disk cost of reading ~9 GB of GGUF while the
later ones hit the OS page cache — the ranking would then tell you the order the
tests ran in, not which placement is faster. Peak VRAM is reported **over the
baseline measured just before each run**, so what the desktop already occupied
does not land in the number.

A winner is only declared when it beats the runners-up by **more than the
measured spread** (max − min across the timed runs). When two placements sit
inside the noise, the report keeps the **simpler** one rather than pretending to
split them. The log streams while the test runs and **⏹️ Stop** ends it cleanly
after the run in progress. Running the test never alters your preferences —
applying the profile is a separate, explicit click.

### Direct convolution (memory)

**⚡ Acceleration (advanced)** also exposes `--diffusion-conv-direct` and
`--vae-conv-direct`. They swap the convolution algorithm: instead of **im2col**
— which unfolds the image into a large matrix before multiplying — the
convolution is computed directly. im2col is fast, but that intermediate buffer
is big; computing directly removes it.

What can honestly be promised: **less memory**. Speed depends on tensor shapes —
sometimes better, sometimes worse. **Measure it**, don't tick it on principle.
The clear case for turning it on is when you are close to running out of memory.

Both are recent sd.cpp options: the app **checks the installed binary** and
simply omits them if it doesn't know them, so an older engine cannot break on an
unknown argument. `update.bat` to get them.

> **What about SageAttention or Triton?** They cannot be added, and it is not a
> matter of build flags. Both live in the **PyTorch** ecosystem: Triton is a
> kernel compiler driven from Python and JIT-compiled at runtime, and
> SageAttention is a pip package whose CUDA/Triton kernels hook into PyTorch's
> attention. stable-diffusion.cpp is C/C++ on GGML — no Python, no PyTorch, no
> runtime that could host a Triton kernel, and therefore no hook point. Getting
> quantized attention here would mean **reimplementing it as a ggml CUDA
> kernel** upstream, not flipping an option. What sd.cpp already has on that
> front is flash attention (`--diffusion-fa`, compiled in by default via
> `GGML_CUDA_FA`), plus GGUF quantization and the step caches above — which is
> where the actual speedups live.

### One shot per image, and nothing kept warm
Generation runs through **one-shot `sd-cli`**: the engine starts, reads the
model, makes the image and exits. That is the whole model now, and there is no
second path.

There was one. This section used to describe a **resident engine** — `sd-server`
from the same sd.cpp archive, keeping the model loaded and answering local HTTP
requests, so the second image started straight at sampling. The measurement that
motivated it is real and still worth knowing: on one 250 s Krea 2 Turbo image,
**80 s went to re-reading the model** and **38 s to encoding the prompt**,
against 117 s of actual sampling — paid again for every image.

**It is removed anyway, and the cost was not in the server.** Keeping a model
resident means every other part of the project has to remember to make it give
the card back before taking it: `sd-cli` runs, every Toolkit tool, the 3D tab,
the benchmark's own baseline. Miss one and you get an out-of-memory error whose
cause is somewhere else entirely, at a moment that has nothing to do with it.
On top of that it served only a narrow slice — LoRAs, the HD pass, step caches
and auto-fit all fell back to `sd-cli` — and it cost the **live step preview**,
which is the thing you actually watch while a slow image renders.

The reload cost is better attacked where it lives: a **lighter quantization**
reads fewer bytes, and keeping the **text encoder on a card that is fast at
fp16** removes the 38 s rather than hiding it (see
[Machine profiles](#machine-profiles--one-button-per-known-tower)).

(The earlier ComfyUI backend stays removed too: too fragile.)

### Engine binary: official or self-built (CI)
By default the engine step downloads the **official** prebuilt binary from
[`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
releases — the simplest, always-works path.

The engine binary comes from the **official upstream release**, and only from
there. The project used to publish its own CI build (arch-tuned for this
project's cards, `update-engine-ci.bat`); it was removed. It packaged a single
executable and therefore silently lacked `sd-server`, and maintaining a second
build chain to save a few percent was not worth its cost — nor the class of bug
where the two engines differ.

### Updating the engines

**The code and the engines used to update separately, and that was the trap.**
They do not any more: `update.bat` is one button for all of it (see
[Updating the app itself](#updating-the-app-itself)), and each part is still
reachable alone.

| Command | Updates |
|---|---|
| `update.bat` | **everything**, in order: code, cleanup, sd.cpp, 3D |
| `update.bat --engine` | **sd.cpp** only — checks the latest official prebuilt release and installs it if newer (images and video) |
| `update.bat --trellis` | **trellis.cpp** only — checks the latest official Windows build (Image → 3D), and only if already installed |

The engine steps replace only the **engine binary**. sd.cpp updates are
first validated outside `bin/`, then swapped atomically; the previous working
build is retained for one-command rollback (`rollback-engine.bat` /
`./rollback-engine.sh`) and an `engine-manifest.json` records its exact commit,
archive checksum and supported options. DLLs from two releases therefore never
mix, while a broken download can never destroy the working install. **Models
are never re-downloaded** — including the ~10 GB trellis 3D set; reinstall
those from the **🧊 Text / Image → 3D** tab if ever needed.

The trellis.cpp updater likewise extracts and checks the new archive before
replacing the installed binary. It remembers the release and archive, skips
an unchanged version, and keeps the working CUDA build if a new release has
not yet published the archive for that card. The 3D model weights stay in place.

**When do you need to update the engine?** When a tab tells you to, or when a
new model arrives. The updater compares the installed build with the latest
official release even if `sd-cli -h` exposes all the options the app knows:
model support can change without adding a command-line flag. Running
`update.bat` with no flag checks it as part of the full update.

The 3D step picks its archive from your card — see [Text / Image → 3D](#text--image--3d).

### Keeping the interface shallow
Two rules, enforced by `tests/test_ui_shape.py` rather than by good intentions:

- **No accordion lives inside another accordion** — the threshold is zero, not
  "reasonable". A fold inside a fold costs two clicks to reveal one option, and
  the second click is never announced. Where several sections must coexist, they
  are **tabs**: all of them visible without opening anything. That is why the
  536-style bank is now 🎨 Styles → three tabs (*Custom preset* · *Photo* ·
  *Artistic*), and why **Settings** is four tabs rather than a stack of six
  folds.
- **One-shot blocks disappear once they are done** — the seven "⚙️ Install …"
  panels are hidden on a machine where the tool is already installed, leaving a
  small **Reinstall / repair** button in their place. The state that matters is
  the one you live in, not the first launch.

Measured on the real rendered tree of a machine where every tool is already
installed: **62 visible accordions before, 33 after**, and accordion nesting
from **1 level to 0**. Part of that drop is simply the removal of a generation
tab; the rest is the two rules above.

### Interface language & theme
**The interface is English, and there is no language setting.** There used to be
one: the source strings were French and `atelier/i18n.py` held a French → English
dictionary that `t()` looked them up in. That layer had two costs. A string added
to a new tab was French until someone remembered to add a row to the table, so
the English half was only ever as good as the last person to update it. And the
table had to be kept in step with the code by hand, which is exactly the kind of
bookkeeping that quietly stops happening.

So the strings themselves are English now, and the table is gone.
`atelier/i18n.py` survives as a seam — `t()`, `to_source()` and
`translate_blocks()` are identity functions — so the ~40 call sites read the
same and a real language layer could come back at that one point. Engine logs,
progress lines and error messages are English too; there is nothing left that
speaks French to the user.

`tests/test_i18n.py` enforces it rather than trusting anyone's diligence: it
fails on any French string literal reaching the interface, detecting them by
accents, by a list of French words chosen to collide with no ordinary English
word, and by the metric units (`Go` / `Mo` / `Ko` / `To`) matched
case-sensitively so `To use` and `Go high resolution` stay legal. A short list
of accented English loanwords (`café`, `fête`, `naïve`…) is excepted rather than
weakening the accent rule.

**`scripts/` is inside that scope, and was the gap.** Those files look like
build plumbing, but `setup_tools.py`, `get_sdcpp.py` and `get_trellis.py` run as
subprocesses whose stdout is piped **into the Gradio log box** — their French
was on screen, one click on any Install button away, while a guard scoped to
`atelier/` reported everything clean. The 346 strings they held are translated
and the guard now covers them.

**Code comments and docstrings are still French** and deliberately out of scope:
they are the source's own language, not the interface's, and nobody reads them
from the app.

**Theme** — **Settings → 🌍 Theme and accounts** switches between **Light** and
**Dark**. It is saved to `userdata/` and applied on **restart** (`run.bat` /
`run.sh`), because Gradio builds the interface once at launch.

---

## Upscaling

Use **Tools & 3D → SeedVR2 upscale** or **VOSR 2.0 upscale** for restoration.
See [SeedVR2 and media engines](docs/media-engines.md) and [VOSR](docs/vosr.md).

## Outpaint

The **🖼️ Outpaint** tab extends an image **left, right, up, down — or all
around**, Midjourney-style.

> **Use it with an *edit* model** — Flux.2 Klein. This is not a
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

## Text / Image → 3D

**🧰 Tools → 🧊 Text / Image → 3D** turns a **prompt or an image** into a
**3D mesh** (GLB) through
**[trellis.cpp](https://github.com/pwilkin/trellis.cpp)** (TRELLIS.2 or Pixal3D) — a native
CUDA binary on GGML, **no PyTorch**.

### Text → 3D is a chain, not another model

TRELLIS only reads images. So a prompt is turned into one first, with the
generation model you already have, and that image is handed over. This is not a
shortcut on our side — it is what trellis.cpp's own documentation means by
text-to-3D: *"optionally driven end-to-end from a text prompt with
stable-diffusion.cpp producing the input image"*.

Two consequences shape the tab:

- **The intermediate image is shown, not hidden.** It lands in the image field
  before the mesh starts. That is where things go wrong — a tight crop, a cast
  shadow, a background — and a single button that chained everything would make
  you pay for a full mesh to find out. Seeing it costs one look; re-rolling it
  costs seconds, where the mesh costs minutes.
- **The prompt is rewritten**, and that is not decoration. TRELLIS reconstructs
  *one* object: it cuts out the subject, processes the image as a square, and
  assumes what it sees is the whole thing. A tight crop makes it invent what is
  off-frame; a cast shadow becomes geometry; a background becomes noise on the
  mesh. So your subject is wrapped in terms that each answer one of those
  failures — *one object only, entire object visible in frame, plain flat
  neutral background, no cast shadow, centred* — and the image is generated
  **square at 1024²**, because TRELLIS would square it anyway and it is better
  to choose the framing than to let it crop.

  The template is in **Advanced options**, editable and visible. Empty it and
  your prompt goes through untouched.

One thing the tab handles for you: the **resident 3D server holds the card**.
If it is running, it is stopped before the image is generated — otherwise you
would get an out-of-memory error whose cause is the 3D tab itself, which is
exactly the kind of failure nobody traces back.

### Image → 3D

Give it a sharp image of a **single object** on a simple background;
background removal is automatic.

> ⛔ **Windows CUDA only.** There is no Apple Silicon build and no Metal path.
> The tab says so on a Mac instead of offering an installer that would find
> nothing.

**Installation is two downloads**, both from the tab: the **binary** (~700 MB)
into `bin/trellis/`, and a **set of GGUF weights**
([`ilintar/trellis2-gguf`](https://huggingface.co/ilintar/trellis2-gguf)) into
`models/trellis/`. Three weight variants, and the intuition here is backwards:

| Variant | Size | Speed |
|---|---|---|
| **f16** (default) | ~16.5 GB | **the fastest** when it fits |
| **q8** | ~9.9 GB | slower |
| **q4** | ~6 GB | slower still |

In ggml, quantized weights are **dequantized on the fly at every computation**,
and a 3D workload does not amortize that overhead the way a long diffusion run
does. So quantizing here buys **memory, not speed** — which is the point when it
is what makes 1024/1536 reachable at all. You can install several and switch at
generation time.

**Stay at 512 below 16 GB of VRAM.** The 1024/1536 cascade is documented for a
16 GB card. Below that it does not fail cleanly — it **degrades the computation**
and returns a mesh made of blobs. No setting works around it: trellis has neither
offload nor tiling. To gain quality *without* touching the resolution, raise the
**UV atlas** (2048/4096) and the **decimation**: a well-textured 512 mesh beats a
botched 1024 one, at almost no VRAM cost.

### Pixal3D (trellis.cpp 0.8.0+)

Select **Pixal3D** in the same 3D tab. Its installer checks the binary and
fetches the [GGUF files recommended by trellis.cpp](https://github.com/pwilkin/trellis.cpp/blob/v0.8.0/docs/pixal3d/README.md)
from [`vegax87/Pixal3D`](https://huggingface.co/vegax87/Pixal3D).
Click **Check Pixal3D GGUF file sizes** to see the published sizes first.

| Download choice | Pixal3D files added | Output |
|---|---:|---|
| **512** | 3 GGUF, about 5.5 GB | Geometry only: no 512 texture checkpoint exists |
| **1024** | 5 GGUF total, about 11 GB | Textured GLB; at least 16 GB VRAM recommended |

The Pixal3D flows are stored once in `models/trellis/`. Its decoders, DINOv3
and BiRefNet are reused from the selected TRELLIS.2 weight variant. For q4 or
q8, local hard links make the Pixal3D flows available beside the selected
decoders without another disk copy. An external models drive must support hard
links (NTFS on Windows); select f16 if it does not.

The **horizontal FOV** describes the camera of the input photograph. Leave it
empty for Pixal3D's default 49.13°. Try another value if the generated geometry
drifts from the object's silhouette. trellis.cpp does not implement the
reference's MoGe-2 camera estimation.

**How the server is driven.** The Windows release ships `trellis-server.exe`
and a CLI. The app **starts the server, posts the
image, takes the GLB and stops it**, so all the VRAM is released afterwards
(the low-VRAM strategy). A **⚡ Resident server** checkbox keeps it alive
instead: the next objects skip the ~30 s model reload, but the VRAM stays
occupied — stop it before generating images. Changing a *launch* setting
(resolution, decimation, atlas, GPU, texture) restarts the server automatically,
because those are not renegotiable per request. The seed, background removal,
model family and Pixal3D camera can change per request without a reload.

**Controls**

| Control | What it does |
|---|---|
| **Object to model** | a prompt — leave empty to use an image instead |
| **Model used for the image** | any generation model you have downloaded |
| **Input image** | one object, simple background |
| **3D model family** | TRELLIS.2 or Pixal3D; 512 Pixal3D produces geometry only |
| **Pixal3D camera FOV** | horizontal field of view; empty = 49.13° |
| **Pad to square** | TRELLIS processes its input as a square — without this a non-square image comes out **distorted** |
| **Geometry resolution** | 512 (recommended, ≤ 12 GB) · 1024 · 1536 |
| **Decimation — legacy cluster grid** | a GRID size, not a face count, and upstream marks it legacy. Left at 0 you get the current default: quadric simplify to 150K faces at 512 (300K at 1024), then weld, hole-fill and a stray-component pass |
| **UV atlas size** | texture resolution — the cheapest quality gain |
| **Background removal** | **Automatic** (default), always BiRefNet, or threshold — see below |
| **WebP textures** (Advanced) | on by default upstream; untick for PNG textures and a GLB any viewer opens |
| **Geometry only** | skip the texture, faster |
| **Card used for 3D** | passed to the engine as its own `--gpu N` |
| **Seed** | shown under the result, to replay the same object |

**Background removal has three states, and only two can be typed.** trellis.cpp
declares `int birefnet = -1` — 1 BiRefNet, 0 threshold, **-1 auto** — but its
parser is `p.birefnet = (strcmp(v, "birefnet") == 0) ? 1 : 0`. No string yields
-1: **auto is only reachable by not passing the flag at all**, and writing
`--bg-removal auto` gives you the *threshold*, the opposite of what you asked
for. The app therefore treats automatic as an absence, on the command line and
in the resident server's form field alike.

Automatic is worth having: if the image already carries an alpha channel it is
kept as-is and BiRefNet is skipped entirely. The Toolkit's own **✂️ Background
removal** produces exactly such PNGs — without this, a cutout you chose was
re-matted by a second, different neural cutout costing ~13 s.

And the threshold keyer is a trap rather than a fast path: it reads bright
specular highlights (`min(RGB) ≥ 232`) as background, and the flow then
**generates holes right where the highlight was** — upstream names the helmet
crest and axe edge of its own test asset. It stays available, labelled.

**WebP textures.** The GLB's textures are WebP by default, through the
`EXT_texture_webp` glTF extension — which not every viewer or marketplace
reads. Untick it in *Advanced options* for PNG and a file anything can open, at
the cost of size.

The result is previewed in the browser and written to `outputs/`. Generation
tabs carry a **🧊 Send selection to Image → 3D** button.

**When it fails, the message is the real cause.** If the server dies
mid-generation the HTTP connection is cut and `requests` raises a bare
`ConnectionResetError`, which says nothing useful. The app captures the server's
own last lines and reports those instead — see the `no kernel image` entry in
[Troubleshooting](#troubleshooting), which is the one failure that used to be a
dead end and no longer is.

---

## Toolkit

One-click installable utilities (models pulled from Hugging Face, run as
subprocesses so torch DLLs never lock the UI process):

| Sub-tab | What it does | Add-on |
|---|---|---|
| **📝 Image → prompt** | hand it an image, get the prompt back ([below](#-image--prompt)) | Qwen2.5-VL-3B, ~7.5 GB |
| **🌐 Depth** | depth map — *Depth Anything V2* | ~100 MB |
| **✂️ Background removal** | cutout → transparent PNG — *RMBG-1.4* (**non-commercial**) | ~176 MB |
| **🪄 Cut out (SAM)** | click an object, extract it — *Segment Anything* (SAM 2.1 small or legacy SAM 1) | ~184 MB / ~375 MB |

Every generation tab has a **Send to Toolkit** control that pushes the selected
image straight into one of these, so there is no file to find and re-upload.

### 📝 Image → prompt
Give it an image, get back the prompt that would recreate it — then send that
prompt straight into a generation tab with one button.

**This is not captioning, and the difference is the whole point.** A vision
model left to itself says *"a photo of a cat on a sofa"*. That is a sentence
*about* the image; pasted into the Prompt field it produces something flat.
A prompt is an *instruction*: it names the medium, the light, the lens, the
palette and the framing, because those are the words a diffusion model actually
responds to. The system prompt therefore bans caption formulas outright, and the
lead-ins the model emits anyway ("This image shows…", "Sure, here is a
prompt:…") are stripped from the answer — the instruction lowers their
frequency, it does not reach zero, and one is enough to pollute the field.

Output is always **English**, whatever the interface language: that is what the
models were trained on.

**Repetition is the normal failure of this format, and it is handled in three
places.** A comma-separated keyword list never tells the model it is finished —
nothing in the grammar signals an end — so it loops: a real run produced a good
opening and then repeated *"high heels, fashion, modern, wet, rain"* until the
token budget ran out, 130 fragments for 50 useful ones. The fixes, in order of
how much they can be relied on:

1. the token budget is **matched to the requested length** (200 tokens for
   110 words, not 320 — that 65% of slack is exactly what the model fills with
   restatements);
2. `repetition_penalty` and `no_repeat_ngram_size=6` are passed to the
   generator, and the system prompt forbids restating an idea already written;
3. **the answer is deduplicated afterwards.** This is the one that guarantees
   the result rather than hoping for it: the output *is* a comma-separated
   list, so identical fragments can be dropped without losing anything. Order
   is preserved (the opening carries the subject, the light, the lens), and
   `wet pavement` collapses with `the wet pavement`. Prose answers — the plain
   description mode — are never deduplicated, since a comma there is grammar,
   not a separator.

**A second failure showed up in use, and it is worse than looping.** On a
photograph, the model finished with *"oil painting style, brushstrokes visible,
canvas texture evident"* — three fragments contradicting everything before
them, and enough on their own to make the diffusion model paint instead of
photograph. Same lesson, same shape of fix:

- the medium is now asked for **first**, not last. The drift happens at the
  *end* of a generation, when there is nothing real left to say; naming the
  medium while the model is still looking at the image pins it down;
- and it is **enforced afterwards**: whichever medium is established first wins,
  and later fragments from an incompatible vocabulary are dropped. First wins
  because the model describes what it sees before it starts confabulating — in
  the real case `depth of field` came nine fragments before `oil painting
  style`. Markers are deliberately unambiguous (`canvas texture` is a marker,
  bare `texture` is not): the cost of a wrong match is deleting a legitimate
  fragment.

**And the sampling was wrong for the job.** Describing is not creating. The
runner sampled at `temperature=0.7`, which is literally asking the model to pick
a less likely token now and then — on a description, that means inventing. It
reported *"dark red shoes"* for black ones. A single proposal is now decoded
**greedily**, with no sampling at all: the image does not vary, so there is
nothing to gain from varying the answer. Sampling only comes back when several
proposals are requested, and then at a low temperature. This is a fix at the
cause; a wrong colour cannot be detected from the text afterwards, and the
README will not pretend otherwise.

A tail cut off by the token limit is dropped too, but only when the runner
*knows* it was cut — the model emitted no end token. It is never guessed from
the text: `shallow dep` and `shallow` are indistinguishable without a
dictionary, and trimming a legitimate fragment is worse than leaving a stub.

Three modes, which ask for genuinely different things:

| Mode | What it writes | What it is for |
|---|---|---|
| 📸 **Recreate this image** | subject, setting, light, composition, colours, medium — 60–110 words | reproducing or varying an image on your own models |
| 🎨 **Style only** | medium, technique, light, palette, contrast, grain — **and not one word about the subject** | applying that look to a completely different subject |
| 🔍 **Plain description** | two or three sentences, no prompt jargon | knowing what is in an image |

The style mode's rule is strict on purpose: if the image is a red car in Rome,
neither "car" nor "Rome" may appear. A style that names its subject is not
transposable, and the mode would have no reason to exist.

**Model**: **Qwen2.5-VL-3B-Instruct** (~7.5 GB), loaded then unloaded per call.
It runs on the separately selected Image → prompt GPU when configured. It needs `transformers>=4.49` — the
`Qwen2_5_VLForConditionalGeneration` class does not exist in 4.48, verified —
so the installer pins that range for this add-on only. ⚠️ **Qwen Research
licence: non-commercial**.

## Convert to GGUF

**⚙️ System → 🔧 Convert to GGUF** quantizes a model you got from somewhere else
— a **checkpoint / safetensors / diffusion** file — into a lighter **GGUF** that
fits on your card. It is **100% CPU** (no diffusion involved): a few minutes
depending on the size and the disk, done once, and then you reuse the GGUF.

1. Drop your model into **`models/custom/`**, then hit **↻ Refresh**.
2. Pick the quantization.
3. **Convert** — the GGUF is written into the same `models/custom/` folder and
   becomes selectable as a **local model** in the generation tabs (through their
   *Refresh local files* button).

| Quantization | Bits | Notes |
|---|---|---|
| `f16` | 16 | no loss, and no saving either |
| `q8_0` | 8 | near lossless, larger |
| `q5_1` | 5 | **recommended** — the size/quality compromise |
| `q4_1` | 4 | light |
| `q4_0` | 4 | the lightest |

⚠️ sd.cpp only implements `q8_0 / q5_1 / q5_0 / q4_1 / q4_0 / f16` **when
converting** — there are **no k-quants** here. That is a limit of the engine's
conversion path, not of the app; the k-quants you see in the catalog were
produced upstream.

---

## Why it only listens on 127.0.0.1

The app binds the **loopback address and nothing else**. There is no flag to
change that: `--listen`, `--share`, `--auth` and `--host` are gone, and so is
`run-lan.bat`.

It did offer LAN sharing — colleagues on the same Wi-Fi opening
`http://192.168.1.42:7860` in a browser and generating on your GPU. Removing it
is not a security posture, it is an honest accounting of what serving costs. An
interface that accepts connections has to answer three questions, and none of
them has a good default:

- **who may connect** — the password was a `--auth user:password` typed on a
  command line, in a project whose whole point is that the user never opens one;
- **what the firewall should allow** — Windows asks once, about "Python", and
  the answer sticks for every other Python program on the machine;
- **what a viewer can read** — `allowed_paths` is what the browser may fetch by
  path. On loopback that is a display detail. Open to the network it becomes a
  permission, granted to anyone who knows the URL.

Nobody was using it. Keeping it meant keeping three answers that were all "it
depends", so the feature went instead.

A copy already installed still has `run-lan.bat` sitting in its folder —
extracting a ZIP over a folder never deletes anything. `update.bat` removes it
(it is declared in `REMOVED_FEATURES`), which is one more reason the cleanup is
now part of the update rather than a button beside it.

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
ZIP. To update the GGUF engine later, run **`update.bat`**.

---

## Models & sources

`config/models.yaml` is the single source of truth (sources, defaults, presets).
Quantization tokens (`{quant}` for diffusion, `{enc_quant}` for the encoder) are
resolved from your hardware; the downloader picks the closest matching file.

**Flux.2 Klein 9B** (family `flux2`, edit model)
- diffusion — [`unsloth/FLUX.2-klein-9B-GGUF`](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) (distilled, 4 steps, CFG 1.0)
- VAE — [`Comfy-Org/flux2-klein-9B`](https://huggingface.co/Comfy-Org/flux2-klein-9B) (`flux2-vae.safetensors`)
- text encoder — [`unsloth/Qwen3-8B-GGUF`](https://huggingface.co/unsloth/Qwen3-8B-GGUF) (official Qwen3-8B, via `--llm`, offloaded to RAM)

**Qwen Image 2.1 Turbo** (family `qwen21`, generation + edit)
- diffusion — [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models) (`qwen_image_2.1_turbo_int8_convrot.safetensors`, selectable Turbo BF16; 8 steps / CFG 1 / official sigmas)
- VAE — [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (`qwen_image_2.1_vae_bf16.safetensors`)
- text encoder and optional editing projector — [`Qwen/Qwen3-VL-8B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) (`--llm`, `--llm_vision`)

**Krea 2 Turbo** (family `krea2`, sd.cpp)
- diffusion — [`vantagewithai/Krea-2-Turbo-GGUF`](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF) (8 steps, CFG 1.0)
- text encoder — [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (official Qwen3-VL-4B-Instruct, via `--llm`, offloaded to RAM)
- VAE — [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) (`wan_2.1_vae.safetensors`)

> Why the Klein source changed too: the previous repo
> (`leejet/FLUX.2-klein-9B-GGUF`) publishes **only Q4_0 and Q8_0**. Measured
> against the real fallback logic, *every* card ended up on Q4_0 (5.6 GB) — a
> 12 GB RTX 3060 asking for `Q5_K_M` and a 16 GB RTX 4080 asking for `Q6_K`
> alike. The current repo carries `Q2_K` → `Q8_0` with genuinely distinct sizes
> (`Q5_K_S` 6.94 GB ≠ `Q5_K_M` 7.02 GB), same base model and same filenames, so
> the targeted quant actually exists. On 12 GB that is a whole quality tier
> recovered: 7.02 GB instead of 5.62 GB.

> Why the Turbo source changed: the previous repo (`realrebelai/KREA-2_GGUFs`,
> `TURBO/` folder) had no `Q5_K_M` and no `Q2_K`. On a 12 GB card the auto
> profile asked for `Q5_K_M` and silently fell back to `Q5_K_S`. The current
> repo carries the full ladder from `Q2_K` (4.9 GB) to `Q8_0`, so the targeted
> quant actually exists — and small cards finally get a rung. Already-downloaded
> files are untouched; the change only affects new downloads.
>
> Honest footnote: in both `vantagewithai` repos the `_K_S` and `_K_M` files of
> a given bit width are **byte-identical in size** (Q5_K_S and Q5_K_M are both
> 8.87 GB). So what the switch really buys is the missing rungs — `Q2_K` at the
> bottom, and no more "quantization adjusted" fallback notice — not a smaller
> or larger file at Q5.


**Upscalers** — [ESRGAN GGUF](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) and Spandrel modern upscalers.

To delete a model, use **🗑️ Delete** in the Model Catalog — shared files
(encoders/VAEs used by another model) are preserved.

---

## Project layout

```
app.py                       # Gradio entry point + presentation() (theme/CSS/head)
config/
  models.yaml                # catalog: sources, defaults, presets (source of truth)
  style_presets.json         # bundled system-prompt styles
  krea2_styles.csv           # 139 photographic styles (© ghleg, MIT)
  krea2_art_styles.csv       # 397 artistic styles
atelier/
  settings.py                # paths + persisted preferences (userdata/)
  hardware.py                # GPU/RAM detection + optimization profiles
  quant.py                   # the quantization ladder
  registry.py                # catalog, file resolution, status, recommendations
  downloader.py              # on-demand Hugging Face / Civitai downloads
  styles.py                  # system-prompt / style presets
  sampling.py                # samplers & schedulers: verdicts and their reasoning
  benchmark.py               # "measure this machine": placements, medians, spread
  inventory.py               # what is on disk, with sizes, by category
  storage.py                 # moving models to another drive (+ junctions/symlinks)
  imgcheck.py                # image-display diagnostic (cache, MIME, last upload)
  diagnostics.py             # machine report for bug traces
  net.py                     # finds a free port for the local interface
  i18n.py                    # identity seam — the interface is English (see above)
  engine/
    sdcpp.py                 # build/run sd-cli commands, error typing, DiskWatch
    generate.py              # generation pipeline (model + hardware + LoRA)
    outpaint.py              # directional outpaint: canvas plan, fill, tone match, composite
    trellis.py               # trellis.cpp server: image → GLB, transient or resident
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, faces…)
    masks.py                 # mask cleanup and layer naming — pure numpy
    psd.py                   # PSD writer — pure Python, no compiled dependency
    vocab.py                 # CLIP scene vocabulary as DATA, for layer labelling
  ui/
    theme.py                 # theme + CSS
    widgets.py               # shared image/gallery button lists (Gradio 6)
    preview.py               # data: URI fallback preview
    generate_tab.py · xanax_tab.py (hard-wired style) · library_tab.py
    toolkit_tab.py · outpaint_tab.py · threed_tab.py
    settings_tab.py · manage_tab.py · convert_tab.py
scripts/
  get_sdcpp.py               # downloads the stable-diffusion.cpp binary
  get_trellis.py             # downloads the trellis.cpp binary + GGUF weights
  update_app.py              # update.bat: the four steps, in order (code → clean → engines)
  maintenance.py             # steps 2-4: purge, orphan detection, engine capabilities
                             #   (no launcher of its own: update.bat --clean runs it)
  _torch_setup.py            # shared PyTorch-CUDA install helpers
  setup_tools.py             # installs the PyTorch add-ons
  tools/_device.py           # CUDA / Metal-MPS / CPU picker shared by the runners
  tools/run_*.py             # inference runners (depth, rembg, sam, describe,
                             #   restoration, video, splats)
tests/                       # 400+ unit tests, run with `python -m unittest discover -s tests`
```

---

## Managing disk space & uninstalling

The **🧹 Manage & help** tab is the single place to see what the app has
downloaded and to reclaim space. It lists every item with its **size**, grouped
into four categories:

| Category | What's in it |
|---|---|
| **Engines** | `sd-cli` (stable-diffusion.cpp) and the trellis.cpp 3D binary |
| **Models** | each catalog model separately, the ESRGAN upscaler pack, the ~10 GB trellis 3D set |
| **Toolkit add-ons** | depth, background removal, SAM and image → prompt |
| **Your data** ⚠️ | LoRAs, custom models, generated images/3D, temp files |

Tick what you want to remove, tick **“I confirm”**, then delete. Sizes refresh
after each operation.

- Active tools outside **Your data** are **re-downloadable** from inside the app
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

### What else lives in Manage & help

- **📖 In-app documentation for every option** — generation tabs, settings and
  hardware, Outpaint, Image → 3D, Convert to GGUF, the Toolkit, and network &
  maintenance. It is the fastest way to know what a slider actually does, and it
  is written to be read *while* the slider is in front of you.
- **🩺 Diagnose image display** — the three-layer test for the broken-image-icon
  problem, described in [Troubleshooting](#troubleshooting). Press the button
  instead of opening the browser console.
- **🌐 Updating & maintenance** — what `update.bat` does, step by step, in the
  same place as the settings that need it.

---

## Gradio version

The app targets **Gradio 6** (`gradio>=6.0,<7` in `requirements.txt`) and will
not run on 5.x — the 6.0 release removed parameters it used. The cleanup
names the problem if an old version is still installed, rather than letting it
surface as a `TypeError` while the interface is being built.

What the move changed, and why it is worth knowing:

- **`show_download_button` / `show_copy_button` are gone**, replaced by a
  `buttons=[…]` list. The catch is the new default for an image:
  `["download", "share", "fullscreen"]`. That **share** button posts to Hugging
  Face Spaces Discussions — meaningless in an app running on your own machine,
  and it never appeared under Gradio 5 locally. Every image and gallery here
  therefore declares its buttons explicitly, from the named lists in
  [`atelier/ui/widgets.py`](atelier/ui/widgets.py); `tests/test_gradio6.py`
  fails if a component is added without them.
- **Theme, CSS and `<head>` moved** from the `Blocks(...)` constructor to
  `launch(...)`. Forget them and nothing breaks — the interface simply renders
  unstyled — so they are grouped in `app.presentation()` and a test checks they
  reach `launch()`.
- **`launch(show_api=…)` is gone**; API visibility is now set per event
  listener. Nothing to expose here anyway.
- The install is **118 MB lighter** (200 MB → 82 MB for the `gradio` package),
  which matters for the portable Windows folder.

What it did **not** change: the cross-volume upload race described below is
present in Gradio 6 exactly as in 5.50 — same `os.rename`, same fallback to a
background copy, just moved to another module. The fix in `app.py` is still
required.

## Troubleshooting

- **“sd-cli binary not found”** → re-run `install.bat`, or download the engine
  manually. On a portable Windows install there is no global `python`; use the
  embedded one:
  `python\python.exe scripts\get_sdcpp.py --variant cuda`
  (fetches the **win-cuda12** build *and* the **cudart** runtime side by side).
- **“No NVIDIA GPU detected”** → check drivers / `nvidia-smi`.
- **An image shows as a broken-image icon** — the browser got a response, just
  not an image.
  **The main cause, found and fixed, was self-inflicted.** Pinning the image
  cache inside the project (see below) put it on a different drive from
  `%TEMP%` on any machine whose project does not live on `C:`. Gradio
  moves an uploaded file with `os.rename`, which cannot cross volumes; it then
  falls back to copying **in a background task** — *after* already answering
  with the final path. The browser asks for the image immediately,
  `FileResponse` puts the size of the **partial** file in `Content-Length`,
  then keeps reading as the copy grows it, and h11 cuts the response with
  `LocalProtocolError: Too much data for declared Content-Length`. Truncated
  response, broken icon — while the tool itself gets the complete file once the
  copy finishes, which is why the image was *used* correctly but never
  *displayed*. Measured on an 11 MB upload: the endpoint returned a path to a
  **0 KB** file, and the next request announced 65 536 bytes and delivered
  65 536 of 11 234 505. The upload's temp file is now created next to the
  cache, so `os.rename` is an in-place rename again and there is nothing left
  to copy afterwards. `tests/test_upload_volume.py` fails if that ever stops
  being true.
  Two earlier causes had already been closed off. Gradio does not serve images
  from where they live; it copies them into a cache and serves *that*, and the
  cache defaulted to `%TEMP%` — a folder Windows Storage Sense, disk cleanup and
  antivirus all consider fair game. When the copy vanishes the request 404s and
  the image breaks, intermittently, which is what makes it maddening to
  diagnose. The cache is now pinned to `tmp/gradio` inside the project. And
  anything the app hands over *by path* rather than through that cache — a
  generated image sent to a tool, a live preview, a mask — used to get a **403**,
  because `allowed_paths` was never declared at launch; `outputs/` and `tmp/` are
  now declared (and only those: that list is what the browser may fetch by
  path). A third was closed off later: the cache was pinned with
  `os.environ.setdefault`, so a `GRADIO_TEMP_DIR` already present in your
  environment — left by another Gradio app or an old install — silently took
  precedence and put the cache back in `%TEMP%`, reintroducing the exact bug the
  line exists to prevent. It is now set outright.
  **If it still happens, don't open the browser console** — go to
  **⚙️ System → 🧹 Manage & help → 🩺 Diagnose image display** and press the
  button. It answers in three layers, because the first two can pass while
  imports still break:
  1. **The chain itself** — writes a file into the cache and serves it over
     HTTP, naming the causes that actually bite on Windows: a cache on a
     **network or removable drive**, a **full disk**, an **unwritable folder**,
     a **path too long** for the Win32 API, and an **antivirus** locking each
     new file (visible as an abnormal read-back time).
  2. **One tile per format** (PNG, JPEG, WEBP, GIF, BMP) with the MIME type
     Python resolves for each. On Windows `mimetypes` initialises from the
     **registry**, so a missing or hijacked entry makes Gradio serve that
     format as a download and the browser shows nothing — which hits *some*
     extensions and spares the rest, exactly the shape of a bug that breaks
     imports while the general test passes. The tile that fails names the
     format.
  3. **Your last imported file**, displayed, with what it really is: size,
     dimensions, actual format. This is the one that settles it. If the file
     is not listed at all, the **upload** is failing, not the display. If it
     is listed, the report calls out a **truncated file**, an **extension that
     does not match the content** (a `.png` that is really a JPEG), and images
     so large the browser gives up decoding them.

  Meanwhile, the two tools you drive by **clicking on the image** — 🪄 Cut out
  (SAM) and 🧩 Layers in manual mode — carry a **🖼️ Fallback preview** panel
  (folded, so it costs nothing when the normal thumbnail works). It shows the
  same image as a `data:` URI: the pixels travel inside the page, so there is
  no request, no path, no MIME type and no filename involved — it cannot fail
  for any of the reasons above. It is for *seeing*, not for clicking: the click
  still happens on the component itself.
  Note that emptying `tmp/` from **Manage & clean** *while the app is running*
  legitimately breaks already-displayed images until you reload the page.
- **Model shows “to download”** → Model Catalog tab → **Download**.
- **Out of memory** → Settings: lower the quantization, enable offload/tiling, or
  reduce the resolution. For very tight setups, try a per-generation preset.
- **A Toolkit tool runs on CPU (very slow)** → its installer prints
  `CUDA: True/False`; if False, fix NVIDIA drivers and reinstall the tool.

- **The console fills with `StarletteDeprecationWarning: 'HTTP_422_...' is
  deprecated`** → noise from Gradio's own code, one line per queued request, so
  a single generation buries the console. Nothing to fix on our side and it will
  go away with a Gradio update; it is filtered out. It escaped the existing
  filter for a precise reason: `StarletteDeprecationWarning` subclasses
  **`UserWarning`**, not `DeprecationWarning`. The new filter is deliberately
  narrow — silencing every `UserWarning` from Gradio would also have hidden
  *"A function returned too many output values"*, which is exactly what revealed
  that nine **Stop** buttons computed a cancellation message and threw it away
  (`outputs=None`). Pressing Stop now writes that message where you can see it:
  the status line, or appended to the log rather than replacing it.
- **`ConnectionResetError [WinError 10054]` in the console** → harmless, and
  silenced since. It is a known Python bug on Windows (bpo-39010): when the
  browser drops a connection (F5, tab closed, cancelled image load), asyncio's
  Proactor loop calls `socket.shutdown()` on an already-dead socket and prints
  `Exception in callback`. The request is already finished server-side. `app.py`
  now intercepts it *and* completes the cleanup the raised exception used to
  skip — the socket was actually leaking, which is why Python also logged
  `unclosed transport`. Only connection errors are caught; anything else still
  propagates.
- **Image → 3D: `CUDA error: no kernel image is available for execution on the
  device`** → the installed binary has no machine code for your card. Because
  CUDA errors are *sticky*, the failure surfaces on the next ggml op — usually
  `IM2COL` — which sends the diagnosis off in the wrong direction.
  For a long time this was a dead end: upstream's `CMakeLists.txt` **overrode**
  the complete architecture list its own CI passed, pinning trellis's kernels
  (`deform_conv.cu`, `decimate_qem.cu`) to `86;120` — so only RTX 30xx and 50xx
  got machine code, and updating changed nothing. The fix was to install the
  Vulkan build.
  **v0.6.0 (19 Aug 2026) fixed it upstream**: the pin became a mere default
  (`if(NOT CMAKE_CUDA_ARCHITECTURES)`), and the release now ships **two** CUDA
  packages — verified in `.github/workflows/release.yml` at that tag:
  `cuda` (CUDA 13.1) builds `75;80;86;89;90;120` — **Turing and newer** — and
  `cuda12` (CUDA 12.9) builds `60;61;70` — **Pascal and Volta**. So an RTX
  2080 Ti and a GTX 1080 Ti each have their package now.
  The installer picks the archive **from your card**, and **Vulkan stays the
  fallback** for anything neither list covers (and when the card cannot be
  identified). The only substitution allowed is *toward* Vulkan: swapping
  `cuda` for `cuda12` would install a binary built for architectures the card
  does not have. Press **⬆️ Update the binary** in the 3D tab; the ~10 GB of
  models are not re-downloaded. Force a choice with
  `python scripts/get_trellis.py --binary --force --backend cuda12`.
  The app surfaces the diagnosis itself: when the trellis server dies
  mid-generation the HTTP connection is cut and `requests` raises a bare
  `ConnectionResetError`, which says nothing — the real cause is captured from
  the server's output and reported instead.
- **A Toolkit add-on fails at import (`DTensor`, `diffusers`, numpy…)** → a
  shared package drifted. All add-ons share one Python, so the last installer to
  run decides the versions. Run `update.bat --clean`: it names the offending
  package, then reinstall that add-on.

---

## Acknowledgments

This project is just glue around other people's hard work. Heartfelt thanks to
everyone below — all credit for the models and tools goes to their original
authors. Please read and respect each model's own license on its page.

### Engine & framework
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** —
  **leejet** & contributors. The inference engine this whole project rests on.
- **[Gradio](https://github.com/gradio-app/gradio)** — the web UI (6.x).
- **[PyTorch](https://pytorch.org)**, **[Hugging Face](https://huggingface.co)**
  `transformers` / `diffusers` / `huggingface_hub` — the optional Toolkit tools.

### Image models
- **Flux.2 Klein** — base model by **Black Forest Labs**; GGUF by
  [Unsloth](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) (the app moved
  off `leejet/FLUX.2-klein-9B-GGUF`, which publishes only Q4_0 and Q8_0 — see
  [Models & sources](#models--sources)); VAE by
  [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-9B); text encoder
  **Qwen3-8B** by **Alibaba / Qwen team**
  ([official GGUF by Unsloth](https://huggingface.co/unsloth/Qwen3-8B-GGUF)).
- **Krea 2** — base model by **Krea AI**; GGUF by
  [vantagewithai](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF)
  (previously [realrebelai](https://huggingface.co/realrebelai/KREA-2_GGUFs),
  which was missing rungs of the ladder); text encoder
  **Qwen3-VL-4B-Instruct** by **Alibaba / Qwen team**
  ([official GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF));
  **WAN 2.1** VAE by **Alibaba / Wan team**, repackaged by
  [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged).
- **TRELLIS.2 / trellis.cpp** — the native engine by
  [pwilkin](https://github.com/pwilkin/trellis.cpp); GGUF weights by
  [ilintar](https://huggingface.co/ilintar/trellis2-gguf); TRELLIS by
  **Microsoft Research**.

### Upscalers
- **ESRGAN models (GGUF)** collected by
  [wbruna](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) — including
  **Real-ESRGAN** (Xintao Wang et al., Tencent ARC) and community models
  (UltraSharp, foolhardy Remacri, Nomos, LSDIR, NickelbackFS, StarSample…). Credit
  to each upstream author; see the repo for individual sources/licenses.
### Toolkit
- **Depth Anything V2** —
  [depth-anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) team.
- **RMBG-1.4** background removal — **BRIA AI**
  ([briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4), **non-commercial** license).
- **Segment Anything** — **Meta AI**
  ([facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)).
- **Image → prompt** — **Qwen2.5-VL-3B-Instruct** by **Alibaba / Qwen team**
  ([Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)),
  Qwen Research licence (non-commercial).
### Built with
- **[Claude](https://claude.ai/code)** (Anthropic) — vibe-coded iteratively in
  natural language.

This is an independent, non-commercial hobby project, **not affiliated with or
endorsed by** any of the above. If you are an author and want a credit corrected
or removed, please open an issue.
# New media tools: SeedVR2, LTX 2.5 and Gaussian Splatting

**Tools & 3D → SeedVR2 upscale** restores images and videos locally.
Choose **3B ConvRot W4A8** (default, ~2 GB diffusion weights) on an RTX GPU,
or the classic GGUF runtime on GTX 10xx. Install / repair prepares the selected
runtime separately and tests its native quantized GPU operations. INT8 and 7B
variants are also available. ConvRot uses **PyTorch CUDA 13.0** in its isolated
environment, with a driver compatibility check before installation. If an
earlier installation used CUDA 12.6, update the app, select a ConvRot model and
use **Install / repair SeedVR2** to replace that runtime. Installing the CUDA
Toolkit alone does not change PyTorch's CUDA build.
Both runtimes share the model/VAE download cache.
W4A8 uses the standard loader, without DisTorch2 or TensorRT VAE engines.
Its speed on your hardware remains to be measured.
Choose **×2 / ×4** to calculate the size from the source, or **Custom** to set
the short edge manually. **Added detail strength** mixes the resized original
with the AI result: 0% is a simple upscale without loading SeedVR2, 100% keeps
the full restoration, and intermediate values reduce the restoration changes.
**Video → LTX 2.5** generates video from text or first/last frames.
**Tools & 3D → Capture → splats** reconstructs static objects and scenes from
overlapping photographs or video, with a PLY download and interactive preview.
Each workspace has its own installation button. SeedVR2 and COLMAP use isolated
Python environments. Start with the small presets on 11–12 GB GPUs.

See [installation, capture guidance and validation limits](docs/media-engines.md).
# Débit des téléchargements

Dans **Settings → Downloads**, le plafond est de **5 Mo/s (40 Mbit/s)** par
défaut. Il s'applique aux modèles du catalogue et aux modèles téléchargés par
les installeurs d'outils, et peut être changé pendant un transfert. `0` enlève
le plafond. Choisir une valeur inférieure au débit de la connexion garde une
marge pour les autres usages ; l'application ne mesure pas automatiquement la
capacité de la ligne. Les transferts HTTP sont séquentiels, sans Xet. Les
fichiers Hub interrompus sont repris au prochain essai ; les poids ne sont
publiés sur disque qu'après vérification de leur taille et, lorsque le Hub le
fournit, de leur SHA-256. Cette limite ne s'applique pas aux paquets pip, aux
clones Git ou aux téléchargements internes de logiciels tiers.
