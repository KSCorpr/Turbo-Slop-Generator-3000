# 🟢 Turbo Slop Generator 3000

> **Adaptatif au matériel.** Par défaut l'app utilise la **meilleure carte
> NVIDIA détectée** (ou le **GPU Apple Silicon** sur Mac) et s'y adapte (quant de diffusion selon la VRAM, encodeur
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
editing, three upscalers, and a utility toolkit.

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

**Six root tabs**, arranged by what they do rather than by what they are: what
*produces* an image stays at the root, what *retouches* one lives under **Tools**,
what administers the machine lives under **System**. (Eleven root tabs used to
overflow into a `…` menu, which made Manage and Settings invisible at a glance —
so the grouping is not decoration.)

| Tab | What it does |
|---|---|
| 🟣 **Flux.2 Klein** | fast (4 steps) · text-to-image & **multi-reference image editing** · presets, styles, LoRA |
| ⚡ **Krea 2 Turbo** | fast photorealism (8 steps, GGUF, Qwen3-VL encoder, WAN 2.1 VAE) |
| 💊 **Xanax** | one sentence → **one photo** · style **hard-wired**, nothing to configure · model picker for either engine |
| 📚 **Model Catalog** | hardware-aware recommendations, on-demand download / delete |
| 🧰 **Tools** | **Toolkit** (depth · background removal · click-to-cutout (SAM) · ESRGAN · **HD**, the native sd.cpp highres fix with no tiles · SeedVR2 · creative SDXL upscale) · **Outpaint** · **Image → 3D** (textured GLB via **trellis.cpp**, native CUDA, no PyTorch) |
| ⚙️ **System** | **Settings** (detected hardware, quantization, optimizations) · **Manage & help** (disk inventory with sizes, selective uninstall, in-app documentation of every option) · **Convert to GGUF** |

---

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [Generation options](#generation-options)
- [Xanax tab](#xanax-tab)
- [Hardware & optimization](#hardware--optimization)
- [Upscaling](#upscaling)
- [Outpaint](#outpaint)
- [Toolkit](#toolkit)
- [Managing disk space & uninstalling](#managing-disk-space--uninstalling)
- [Sharing on your LAN](#sharing-on-your-lan)
- [Distributing a portable package](#distributing-a-portable-package)
- [Models & sources](#models--sources)
- [Project layout](#project-layout)
- [Gradio version](#gradio-version)
- [Trying MiniMax-H3 video (probe)](#trying-minimax-h3-video-probe)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Install

### Windows (RTX cards)
```bat
install.bat      ::  portable Python + dependencies + GGUF engine (CUDA)
run.bat          ::  launch the UI at http://127.0.0.1:7860
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
- **🧊 Image → 3D is unavailable.** trellis.cpp ships a Windows-CUDA binary only —
  no Apple Silicon build, no Metal path. The tab says so instead of offering an
  installer that would find nothing. Everything else works.
- **16 GB is the realistic floor**, 24 GB+ comfortable. Intel Macs are not
  supported: upstream publishes no Intel build.

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
maintenance.bat --all    ::  purge + engine update  (./maintenance.sh --all)
```

`--update-engine` alone does just the engine. The engine download runs as a
subprocess, so a network failure is reported rather than taking maintenance down
with it.

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
maintenance.bat --purge      ::  actually deletes them (./maintenance.sh --purge)
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
| Engine binary (`bin/`) | no | no — run `update-engine.bat` only when a new sd.cpp feature is needed |
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
- **Merged preview & results** — the live preview shows in the gallery during
  generation, then the final images replace it (one view).
- **Seed** — the selected image's seed shows in a copy-button box; **Reuse this
  seed** drops it back into the seed field. Clearing the seed field resets it to -1.
- **Send to Toolkit** — push the selected image straight into a Toolkit tool
  (depth, background removal, SAM, ESRGAN or creative upscale).
- **Saved prompts** — every image gets an A1111-style `.txt` sidecar in
  `outputs/` with the prompt, negative, model, sampler/scheduler, seed and size.

## Xanax tab

**💊 Xanax** takes **one sentence about your day and returns one photo**. Pick
the model with a radio button (Krea 2 Turbo or Flux.2 Klein); everything else is
compiled in. No style dropdown, no system-prompt box, no preset menu — **the
style cannot be changed**. That is the point of the tab; the normal generation
tabs are there when you want to tune something.

The fixed style: amateur snapshot, provincial France, 1995–2005, cheap
point-and-shoot, ordinary people, unstaged, always overcast, no grain, no filter,
no post-processing, **4:3** on each model's native grid (1184×880 for Flux.2,
1152×896 for Krea 2).

### Write a diary line, not an image description
This is the part that decides whether the result works:

> **not** *"a man waiting for the bus outside a supermarket"*
> **but** *"j'ai mangé chez Flunch avec Mamie"*, *"journée pas terrible mais
> j'ai pu aller acheter des clopes"*

A description is already framed — it says what to show, so the model centres the
subject and composes it. A diary line does not say what to show: the place, the
hour and the bystanders have to be worked out from it, and what comes back looks
like a photo taken in passing. Which is the whole aesthetic.

The **🎲 A random day** button fills the box from a bank of ready-made banal
sentences — as much to show the register expected as to unblock you. Edit the
line it gives you and generate.

### Why the enhancer matters more here
The style is an English prefix glued in front of your text — on its own it
translates nothing, and the image model has never heard of Flunch. With the
enhancer installed, the checkbox turns your sentence into *what the photo would
show*: the self-service cafeteria with its plastic trays and fluorescent
ceiling, the tobacconist's red sign.

It runs on **its own system prompt** (`xanax`), separate from the two used by
the normal tabs. Those ask for a named lens, a lighting setup and a composition —
applied here they would produce a *good* photograph, which is exactly this tab's
failure mode. The Xanax one bans photographic craft outright, forbids
beautifying, and keeps the output to two or three flat sentences: a long prompt
makes the model compose.

Without the enhancer the tab says so, and your sentence is sent **as is** —
write in English then, and say what is visible rather than what you did.

The exact prompt that produced the image is shown under it, a fixed seed replays
the same photo, and the usual `.txt` sidecar lands next to it in `outputs/`.


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
still expose the full sd.cpp list for manual experimentation, **annotated per
model** — with a description card and a fold-out rationale right in the tab.
New entries need a recent engine (`update-engine.bat`).

#### Why most of the menu does not apply here
Three properties of our two models — read off sd.cpp itself, not guessed —
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
| `euler_a` · `dpm++2s_a` · `euler_cfg_pp` · `euler_a_cfg_pp` · `lcm` · `tcd` | ⚠️ | ⚠️ |

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
default**. Requires a recent engine (`update-engine.bat`).

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
unknown argument. `update-engine.bat` to get them.

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

The workflow also exposes **`GGML_CUDA_FORCE_MMQ`** as an opt-in input (off by
default): it replaces cuBLAS with ggml's mmq kernels for quantized matmuls. The
effect depends on architecture and quantization — it can help or hurt. It is
there **to be measured on your own cards**, not enabled on principle. Everything
else worth setting at build time is already set: CUDA architectures targeted at
your GPUs, and flash-attention kernels (`GGML_CUDA_FA`) which ggml compiles by
default.

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

Four complementary tools live under **Toolkit**, in increasing order of
invention: ESRGAN enlarges, **HD** re-denoises with your own model, SeedVR2
restores, SDXL hallucinates.

### 🔼 Simple (ESRGAN, native sd.cpp)
Deterministic ESRGAN upscale via sd.cpp `--mode upscale`: **100% GPU, no PyTorch,
no prompt**. One-click downloads **all** models from
[`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf)
(2x-ESRGAN, RealESRGAN_x4plus, 4xUltrasharpV10, 4x_foolhardy_Remacri…). Pick a
model (×2/×4 depending on its name); **Repeat ×2** chains two passes (a ×2 model
twice = ×4). Best for a clean, faithful enlargement.

**Tile size — where ESRGAN artifacts actually come from.** sd.cpp runs the RRDB
network on **128 px tiles** by default (25% overlap, smootherstep blend). A RRDB
decides how hard to sharpen from what it can see, and on 128 px it sees almost
nothing: two neighbouring tiles treat the same line differently. The feather
softens the seam but cannot reconcile two contradictory decisions — that is the
micro-staircasing on diagonals and the grain that changes character from square
to square. The app therefore sizes the tile itself, from your VRAM (1024 px at
16 GB+, 832 at 11 GB, 640 at 8 GB, 512 otherwise) and, when the image fits under
that cap, asks for **a tile at least as large as the image** — sd.cpp then takes
its untiled path and there is no seam at all, by construction. If the wider tile
runs out of memory the run is retried once at sd.cpp's 128 px and the log says
so. On an older `sd-cli` that predates `--upscale-tile-size`, the option is
simply not sent and the log tells you to update the engine.

**Repeating is worse than it looks.** Chaining passes runs the network on its
*own output*: pass 2 mistakes the high frequencies pass 1 invented for real
detail and sharpens them again, turning mild ringing into hard stair-steps. A ×4
model always beats a ×2 model run twice.

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

### 🚀 HD (native sd.cpp *highres fix*)

The one that has no seams, because it never cuts the image up.

sd.cpp gained a native highres fix, and `sd_img_gen_params_t` carries both an
init image and the hires block — so a single `sd-cli` command does the whole
job: a very light img2img at the source size, then the enlargement (latent,
Lanczos or one of your ESRGAN models), then a **second denoise pass over the
entire image** at the final size. No PyTorch, no SDXL, **no tiles**.

Two things follow from that, and they are the reason this tab exists:

- **there is no seam to hide.** The creative SDXL upscale refines 1024 px tiles
  and blends them; a feather can smooth a border but cannot make two tiles agree
  about what they are drawing. Here the second pass sees the whole scene at once,
  so the question does not arise;
- **your model does the redrawing.** Krea 2 or Flux.2 add detail in the style
  they already know, instead of an SDXL from 2023 reinterpreting it.

Controls: the **factor**, the **added detail** (the hires denoise — the only
setting that really matters: 0.2 stays very close to the source, 0.5+ frankly
reinvents the material), the **intermediate enlarger**, and an optional short
description. The first img2img pass is fixed at a deliberately negligible
strength — sd.cpp computes `t_enc = steps × strength`, so it amounts to a single
step at very low sigma; it cannot be removed (the hires pass hangs off a
generation) so it is made harmless instead.

Sizes are aligned **up** to 16 px — the common divisor of every family's grid,
which leaves the app's own resolutions (1184×880, 1152×896…) untouched where a
64 px grid would move them. sd.cpp then aligns further up to its real multiple
if it needs to, and the log reports the size that actually came out rather than
the one that was requested. The final side is capped at 3072 px: past that the
model is outside its training scale and starts repeating patterns. Hitting the
cap lowers the *factor*, never the framing, and says so.

**VRAM is the real limit here, not the side length.** Refusing to tile has a
price: the second pass allocates a compute buffer proportional to the pixel
count, *on top of* the model weights already resident on the card. A measured
example — Krea 2 at 2304×1792 (4.13 Mpx) asks for a 4.62 GiB buffer while 8.4 GB
of weights are loaded: 13 GB total, which no 11–12 GB card can serve. So the
factor is budgeted rather than capped by a magic number: usable VRAM minus the
diffusion file's size on disk, divided by ~1200 bytes per pixel (the constant
comes from that same failure). Concretely, with Krea 2 at **Q5_K_M** (8.4 GB) an
11 GB card tops out near **×1.25** and a 12 GB card near **×1.5**; dropping to
**Q4_K_M** restores a full **×2** on 12 GB. A lighter quantization buys HD
factor.

**Segmented execution lifts that ceiling** (`--max-vram`). By default sd.cpp
reserves its compute graph *in one block*: if the block does not fit, the run
dies. Given a budget it is allowed to **cut the graph** to fit instead. A
negative value is the interesting one — the engine measures **free** VRAM at
launch and reserves the given margin, so `-1` means "take what is free, keep
1 GiB back", which adapts to the card *and* to whatever already occupies it.
The HD tab therefore turns it on regardless of the Settings preference, because
HD is where all-or-nothing allocation breaks; when it is active the pixel budget
above is **dropped rather than half-relaxed**, since keeping it would throttle
exactly what was just made possible. Cutting the graph costs memory round-trips,
so it is **slower** — which is why ordinary generation leaves it off by default
and Settings exposes it (auto / hard cap / per-device, plus `--stream-layers`)
for people who would rather wait than not get the image at all.

Neither the budget nor the segmentation is the safety net. **An out-of-memory
failure is caught, the factor is stepped down 20% and the run is retried** (twice
at most), and the log states what it settled on. sd-cli failures are now typed:
only a genuine VRAM error is retried, because it is the only one where trying
something smaller can succeed — everything else would fail identically. When even
the smallest attempt fails, the message names the two ways out (lower the factor,
or use the tiled ESRGAN → SDXL path, which fits in far less VRAM) instead of the
old bare "sd-cli exited with code 1".

Requires a recent engine. On an `sd-cli` that predates `--hires` the tab says so
and points at `update-engine.bat` instead of silently producing a plain image.

### 🌱 Restore (SeedVR2 3B)
Diffusion restoration/upscale using the standalone
[`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
engine, pinned to a known commit and installed in an isolated Python 3.12
environment. It restores natural detail more convincingly than ESRGAN while
staying closer to the source than the creative SDXL mode. The Q8 and Q4 GGUF
weights download automatically on first use.

The dedicated **RTX 3060 12 GB + GTX 1080 Ti** preset keeps computation on the
RTX 3060 and uses the GTX 1080 Ti as an offload device. This is deliberate on a
PCIe x4 secondary slot: it avoids continuously splitting matrix operations
between mismatched GPUs. Start with **Q8, 2048 px, 16 swapped blocks, 1024 px
VAE tiles**. If memory runs out, try 24 then 32 blocks, or switch to Q4.

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
  (default) or any installed **ESRGAN** model (sharper, real detail). It runs
  **exactly one pass, never two**. Chaining is what manufactures the artifacts
  and aliasing this pass is supposed to avoid, and a soft base is the cheaper
  mistake: SDXL's refine puts detail back onto a soft base, but it *freezes*
  stair-stepped edges instead of fixing them. So when the model's factor falls
  short of the target the runner finishes in Lanczos and the log says so — and
  when the factor **overshoots** (a ×4 model for a ×2 target) the downscale that
  follows is free supersampling, which is the cleanest case available. Picking a
  pre-upscaler whose factor is *above* your target is the single best setting
  here.
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

> Use the right tool: **ESRGAN** is fast and deterministic; **SeedVR2** restores
> plausible detail with limited drift; **SDXL creative** is slower and explicitly
> invents detail.

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

## Toolkit

One-click installable utilities (models pulled from Hugging Face, run as
subprocesses so torch DLLs never lock the UI process):

- **Depth** — *Depth Anything V2* (depth map).
- **Background removal** — *RMBG-1.4* (cutout → transparent PNG; non-commercial
  license).
- **Click-to-cutout (SAM)** — *Segment Anything* (`facebook/sam-vit-base`): click
  an object, extract it to a transparent PNG.
- **Layers (PSD)** — decompose an image into layers, see below.
- **Upscale (ESRGAN)**, **HD** (native sd.cpp highres fix), **Restore (SeedVR2)**
  and **Creative upscale (SDXL)** — see [Upscaling](#upscaling).

### Layers → PSD

Cuts an image into layers and writes a **PSD** (and/or separate transparent
PNGs), either by letting SAM sweep the image or by **clicking the areas
yourself**. Same add-on as click-to-cutout — nothing extra to download.

**Read this before using it: the layers are flat cut-outs.** Move an object and
you reveal a hole, because the background behind it never existed. This is for
masking, retouching a region or exporting an element — *not* for recomposing a
scene. Doing it properly would mean inpainting behind every layer, one diffusion
pass each; that is a deliberate omission, not an oversight.

**Writing the PSD is done in-house, and that was the surprise.** No usable
library exists: [`psd-tools`](https://pypi.org/project/psd-tools/1.9.28) reads
well but "does not support editing of layer structure, such as adding or removing
a layer", and [`pytoshop`](https://pypi.org/project/pytoshop/) — the only writer
— is from 2018 and no longer builds on a modern Python (verified: its `setup.py`
fails against current setuptools). The format is documented and needs nothing but
`struct`, so `atelier/engine/psd.py` writes it directly: ~180 lines, pure Python,
no compiled dependency, runs as-is in the portable Python. Two decisions carry
the file size — each layer is **cropped to its bounding box** (a PSD stores the
layer position, so keeping the full canvas for a 200 px object would multiply the
weight by twenty) and channels use the format's native **PackBits RLE**. A
512×384 three-layer file lands at 44 KB instead of 1 MB uncompressed. The output
is validated against `psd-tools` in the test suite — an independent reader is the
only honest way to check you produced a valid file rather than one that merely
pleases you.

**The two real difficulties are not in the plumbing.** SAM segments *appearance*,
not meaning: on a photo it happily returns forty to eighty nested masks — a
shirt, a button, a fold, a reflection — and, worse, "zones" made of specks
scattered across the whole frame. And SAM gives **no depth order**: that comes
from Depth Anything V2 when it is installed — median depth under each mask, far
to near. Without it, large areas go to the back, which is an approximation and is
labelled as one rather than presented as a result.

**Large images used to be the wall, and it was an ordering problem.** The two
operations that decide the cut — *are these the same zone?* and *do they
touch?* — are **quadratic** in the number of zones, and they were running at
full resolution. Measured on a 4096×4096 image: one IoU costs **77 ms**, so 150
zones spend **14 minutes** just de-duplicating, plus 70 s on adjacency. Two
changes fixed it. Each mask is reduced **once** to a 256×256 *coverage grid*
(the fraction of lit pixels per block, 262 KB instead of 16.8 MB) and every
comparison happens there — 1770 IoU pairs drop from 136 s to **0.07 s**, and the
approximation stays within 0.02 of the exact value, far from the 0.75 duplicate
threshold. And near-duplicates are now dropped **before** cleaning rather than
after: a 24-point sweep probes the image 576 times and returns the sky or the
tarmac dozens of times over, while cleaning costs 0.7–1.8 s *per mask*. Filtering
300 raw masks went from **3.4 minutes to 26 seconds**, and duplicates are
rejected as SAM produces them, so memory stays near 400 MB instead of 5 GB.

**Everything useful happens in the cleanup**, in `atelier/engine/masks.py` —
written in plain numpy rather than pulling in scipy or OpenCV, since the
segmentation add-on is heavy enough and these operations are a few dozen lines.
Connected components use a union-find over per-row *runs*, not pixels: a
million-pixel union-find in Python would take seconds, while the number of runs
is in the thousands (18 ms on a 1200×900 mask).

- **Split into connected pieces.** A "layer" made of thirty specks in the four
  corners is not a layer, it is noise no editor can use. Each piece becomes its
  own zone, and pieces under the minimum area vanish.
- **Fill interior holes**, which is what gave the cut-outs their swiss-cheese
  look. A notch *open to the edge* is kept — it is part of the silhouette.
- **Keep contained zones.** Being inside a larger mask does not mean redundant,
  it means *in front*: the car on the road, the figure against a wall, the
  window on a façade. An earlier coverage filter deleted exactly those, and it
  was the single worst behaviour of the first version.
- **Merge only what actually touches.** With CLIP installed, pieces sharing a
  label are glued back into one object — body, door and wheel become a car.
  Adjacency is measured **pixel to pixel**, on a reduced grid. It used to be
  measured on *bounding boxes*, and that was catastrophic: on a wide photo a
  car's box covers half the frame, so a chain of boxes linked the car, the
  spray at the far end and the fence in the background into a single 25.7%
  "vehicle" layer. Two more guards came with the fix — an uncertain label never
  merges (propagating a coin toss glues unrelated things), and no merged group
  may exceed 35% of the image, because at that size it is a background, not an
  object.
- **Disjoint layers.** Fronts are subtracted from backs, so showing every layer
  reproduces the source image exactly and no pixel is painted twice — verified
  in the tests by compositing the PSD back and comparing to the original.
  Cutting drops layers from the *middle* of the list, so it now reports which
  ones survived: the caller used to truncate its label list from the end, which
  shifted every name after the first casualty — the car took the wall's name.
  Layers reduced to a thin fringe by whatever sits in front are dropped too.
- **Feathered edges** (1 px), because a binary mask pasted as-is has the
  staircase border that gives automatic cut-outs away.
- **Names you can read.** "Zone 9 — 0.48%" teaches nobody anything; layers are
  named from what is already known about them — depth band, position, dominant
  colour, size: *"foreground · bottom · orange — 3.8%"*. Colour matching weights
  lightness over hue, otherwise charcoal grey gets called dark green.

**With CLIP installed, the tool stops seeing shapes and starts seeing objects.**
It is an optional add-on (`openai/clip-vit-base-patch32`, ~600 MB, one click, on
the same footing as Depth) and it does three things that change the result, not
just the labels:

- **It merges the pieces of one object.** SAM returns "body", "door" and "wheel"
  as three masks. Labelled *vehicle* and adjacent, they become **one layer** —
  what a human calls a car. Adjacency is required: two cars at opposite ends of
  the frame share a label but are not the same object.
- **It discards what is nothing.** A flat fill, a patch of blur, a meaningless
  fragment. A zero-shot classifier cannot say "nothing" — forced to choose, it
  labels a blurry piece of asphalt *car*. So the vocabulary carries deliberate
  **junk categories** whose only job is to absorb those, plus a margin test:
  a label that wins by a hair is a coin toss, not information.
- **It orders the stack by meaning** when Depth is not installed — sky at the
  back because it is the sky, not because it is large. Each vocabulary entry
  declares a typical depth for exactly this.

The vocabulary lives in `atelier/engine/vocab.py` as **data, not code**: that is
the file to edit when labels land wide, with no change to the pipeline. Each
entry carries several phrasings, because CLIP scores an image against a *text* —
"a car" and "a racing car seen head-on" do not score alike on the same crop, and
the best variant wins. Crops are fed as the bounding box with margin, with the
outside of the mask **faded toward neutral grey** rather than cut to black: a raw
box drowns a thin object in its surroundings, while a black cut-out strips the
context CLIP was trained on.

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


**Upscalers** — [`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) (ESRGAN), [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (SeedVR2), `stabilityai/stable-diffusion-xl-base-1.0` + `madebyollin/sdxl-vae-fp16-fix` (creative).

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
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, enhancer, SDXL upscale)
  ui/
    theme.py                 # light theme + CSS
    generate_tab.py · xanax_tab.py (hard-wired style) · library_tab.py · toolkit_tab.py
    outpaint_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # downloads the stable-diffusion.cpp binary
  _torch_setup.py            # shared PyTorch-CUDA install helpers
  setup_tools.py             # installs PyTorch tools (depth, bg, sam, enhance, upscale)
  tools/_device.py           # CUDA / Metal-MPS / CPU picker shared by the runners
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

## Gradio version

The app targets **Gradio 6** (`gradio>=6.0,<7` in `requirements.txt`) and will
not run on 5.x — the 6.0 release removed parameters it used. `maintenance.bat`
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

## Trying MiniMax-H3 video (probe)

sd.cpp gained **day-one MiniMax-H3 support on 4 August 2026** — video *and*
synchronised audio, natively, no ComfyUI. Whether it is usable on an 11–12 GB
card is a different question, so there is a probe rather than a feature:

```
update-engine.bat          :: the engine must be newer than 4 Aug 2026
try-minimax.bat            :: checks engine, GPU and disk — downloads nothing
try-minimax.bat --download :: the lightest published set, ~26 GB
try-minimax.bat --run      :: two passes, timed
```

It adds nothing to the interface and writes nothing to the model catalog. It
answers the only two questions that decide whether a video tab is worth
building: **does the model fit in this card's VRAM**, and **does the 4-step
Turbo LoRA apply**. The second is the real unknown — sd.cpp documents applying
lightx2v turbo LoRAs to Wan 2.2 with `--steps 4`, but not to MiniMax-H3, and
without it you are at the full step count.

The weights: diffusion `ref2va_pruned-Q2_K_M` (6.7 GB), text encoder
**Qwen3-VL 32B** `Q2_K_M` (13.1 GB — this is the real obstacle), video VAE
(5.2 GB), audio VAE (0.6 GB), Turbo LoRA (~1 GB). They land in `models/` and are
removable from **Manage & clean** like everything else.

**Where the text encoder goes is the question that decides everything.** The
first real run failed on it: Qwen3-VL 32B asks for a **12 845 MiB** compute
buffer, and an RTX 3060 has 12 288 MiB in total — it does not fit on an empty
card, and no smaller quant is published. `--offload-to-cpu` does not help,
because it parks the *weights* in RAM while the computation still happens on
the GPU; the encoder itself has to be placed, with `--backend`.

The probe therefore walks a **ladder of placements**, fastest first, and only
steps down when the engine actually reports running out of memory:

1. **Split across cards** — `te=cuda0&cuda1`. sd.cpp cuts the encoder's blocks
   into ranges proportional to each device's free memory, so a 12 GB card and
   an 11 GB card pool into 23 GB and the encoder runs *on GPU*. Only offered
   when two or more cards are present, and it uses their real indices.
2. **On the processor** — `te=cpu`. Works anywhere, but encoding a 32B model on
   CPU costs minutes. (`--clip-on-cpu` does the same and is deprecated upstream
   — it now merely prepends `te=cpu`.)
3. The same, plus a VRAM budget (`--max-vram -1`) for the rest of the graph.

On an error that is *not* a memory problem, it says so rather than burning
another multi-minute pass. The report names the placement that worked, and
`--check` lists **every** card with the pooled total — a machine with two GPUs
has an option a single-card summary hides. When the encoder does end up on the
CPU, **~13 GB of RAM** becomes the real floor, so that is checked before
anything is downloaded.

`ref2va` is a *reference*-to-video model, so it needs a source image: the probe
takes the most recent one in `outputs/`, or `--ref path/to/image.png`. The two
passes run the same generation with and without the LoRA and write two separate
`.webm` files so you can compare them. Community reports put a 3060 12 GB at
roughly 7–19 minutes for a 5–10 s clip, so the probe deliberately asks for
22 frames (~0.9 s) — the point is to learn whether it starts and how fast, not
to produce a clip.

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
  now declared (and only those: on `--listen` that list is what the machine
  exposes). A third was closed off later: the cache was pinned with
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
- **Creative SDXL upscale OOM** → lower the scale or tile size (it auto-offloads
  under 12 GB, but a huge target can still exceed memory).
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
  device`** → the **upstream CUDA build only covers RTX 30xx and RTX 50xx**, and
  updating will not change that. Its `CMakeLists.txt` pins trellis's own CUDA
  kernels (`deform_conv.cu`, `decimate_qem.cu`) with
  `set_target_properties(trellis_core PROPERTIES CUDA_ARCHITECTURES "86;120")`,
  which **overrides** the complete list its own CI passes
  (`75;80;86;89;90;120`). So sm_75 (RTX 20xx), sm_89 (RTX 40xx), sm_80 (A100)
  and sm_90 (H100) get no machine code. Because CUDA errors are *sticky*, the
  failure surfaces on the next ggml op — usually `IM2COL` — which sends the
  diagnosis off in the wrong direction. Checked across every published tag:
  `75` has been in the CI list since the very first release, so "your binary is
  old" was never the explanation.
  **The fix is the Vulkan build**, which compiles nothing per-architecture and
  is, tellingly, the only one of the two the upstream CI does *not* mark
  `experimental` on Windows. The installer now picks the backend **from your
  card** — CUDA only for sm_86/sm_120, Vulkan otherwise, and Vulkan too when the
  card cannot be identified, because a backend that works everywhere beats a
  faster one that works on two models. Press **⬆️ Update the binary** in the 3D
  tab; the ~10 GB of models are not re-downloaded. Force it either way with
  `python scripts/get_trellis.py --binary --force --backend vulkan`.
  The app surfaces the diagnosis itself: when the trellis server dies
  mid-generation the HTTP connection is cut and `requests` raises a bare
  `ConnectionResetError`, which says nothing — the real cause is captured from
  the server's output and reported instead.
- **A Toolkit add-on fails at import (`DTensor`, `diffusers`, numpy…)** → a
  shared package drifted. All add-ons share one Python, so the last installer to
  run decides the versions. Run `maintenance.bat`: it names the offending
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
- **SeedVR2 3B** standalone integration by
  [numz](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), using the
  upstream Q8/Q4 GGUF models and low-VRAM block swapping.
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
