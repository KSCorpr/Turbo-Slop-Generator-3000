# 🟢 Turbo Slop Generator 3000

> **Mono-GPU par choix.** L'app utilise la **meilleure carte NVIDIA détectée**
> et s'y adapte automatiquement (quant de diffusion selon la VRAM, encodeur
> déchargé dans la RAM, flash-attention, offload, VAE tiling). Les stratégies
> multi-GPU (split d'encodeur, auto-fit) et les presets par génération de
> carte ont été retirés pour garder l'outil simple et prévisible.

A **local**, modern, lightweight image-generation studio for artists, built on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** (native
CUDA, GGUF). Generate with **Flux.2 Klein 9B** and **Krea 2 Turbo**, with an
on-demand model catalog, automatic optimization for your RTX card, LoRA, native
resolution presets, saved styles, an AI prompt enhancer, multi-reference image
editing, two upscalers, and a utility toolkit.

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
| ⚙️ **Settings** | detected hardware, quantization, optimizations (auto profile per detected GPU / manual override) |

---

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [Generation options](#generation-options)
- [Hardware & optimization](#hardware--optimization)
- [Upscaling](#upscaling)
- [Toolkit](#toolkit)
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
- **✨ Enhance prompt (AI)** — see [Prompt enhancer](#prompt-enhancer-ai).
- **Negative prompt** — shown only for models that support it (CFG > 1). Distilled
  models run at CFG 1.0 and ignore it.
- **System / style prefix** (accordion) — a prefix prepended to every prompt. Save
  reusable styles to a dropdown (persisted in `userdata/`).
- **📷 Krea 2 photo styles** (accordion) — a bundled bank of **139 stackable
  photographic styles** (quality, lighting, lens, film stock, mood…), grouped by
  category in a **multi-select** dropdown. Your prompt subject is inserted into
  each selected style (`{prompt}` template), and multiple styles chain their
  descriptions after the subject so you can combine axes. Style negatives are
  merged into the negative field **only when the model uses them** (CFG > 1);
  on distilled CFG 1.0 models they are dropped. Bank © *ghleg* — MIT
  ([aoleg/Photographic-styles-and-wildcards-for-Krea-2](https://github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2)),
  shipped as `config/krea2_styles.csv`.

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

### High-resolution output (PiD)
**🚀 High-resolution output (PiD ×4)** in the generation panel decodes each
result through NVIDIA's **Pixel Diffusion Decoder** ([PiD](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/pid.md)):
a 4-step pixel-space diffusion decode that enlarges **×4** (→ 2048 px). Tick it
and every generated image comes out directly in high resolution — no separate
upscaler step. This uses the **Flux.2 PiD** checkpoint (`--vae-format flux2`),
matching the Flux.2 backbone + VAE of the generation, which cuts artifacts versus
the flux1 variant; it **reuses the Flux.2 VAE already downloaded** for generation.
Note sd.cpp exposes PiD as a re-encode pipeline (RGB → VAE latent → PiD decode),
which is its intended mode here, so it adds one encode round-trip rather than
tapping the generation latent directly. First use downloads the PiD weights
(decoder + Gemma-2-2B encoder, one click; the Flux.2 VAE is shared). The output
is 2048 px (the checkpoint is trained for a fixed ×4 from 512 px — other ratios
produce artifacts). Official weights are under the **NSCLv1 non-commercial**
license.

---

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

Multi-GPU: the largest card is used by default, changeable in **Settings**.
Everything is overridable manually (uncheck auto-optimization).

These map to stable-diffusion.cpp flags: `--diffusion-fa` (CUDA: faster + less
VRAM), `--offload-to-cpu` (saves VRAM with no speed loss), `--vae-tiling`,
`--clip-on-cpu`, `--vae-on-cpu`, plus GGUF quantization.

### Manual settings
With auto unchecked you control quant (diffusion / encoder), the GPU, and each
flag (flash attention, CPU offload, VAE tiling, CLIP on CPU, VAE on CPU). A custom
Hugging Face endpoint (mirror) can also be set.

### Single GPU
The app targets a **single GPU** (the best detected card): multi-GPU strategies
(encoder split, auto-fit, GPU picker) are removed for simplicity. The auto
profile adapts to the detected card — diffusion quant by VRAM (11 GB → Q4_K_M,
12 GB → Q5_K_M, more → higher), text encoder quant by system RAM (Q8_0 from
32 GB, offloaded to RAM), flash-attention from Turing on, CPU offload and VAE
tiling on ≤ 12 GB cards.

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

### Interface language & theme
**Settings → 🌐 Langue / Language** switches the UI between **French** and
**English**; **🎨 Thème** switches between **Light** and **Dark**. Both are saved
to `userdata/` and applied on **restart** (`run.bat` / `run.sh`) — Gradio builds
the interface once at launch. On first launch a bilingual language chooser is
shown at the top. Engine logs and progress hints stay in French.

---

## Upscaling

Two complementary upscalers live under **Toolkit**.

### 🔼 Simple (ESRGAN, native sd.cpp)
Deterministic ESRGAN upscale via sd.cpp `--mode upscale`: **100% GPU, no PyTorch,
no prompt**. One-click downloads **all** models from
[`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf)
(2x-ESRGAN, RealESRGAN_x4plus, 4xUltrasharpV10, 4x_foolhardy_Remacri…). Pick a
model (×2/×4 depending on its name); **Repeat ×2** chains two passes (a ×2 model
twice = ×4). Best for a clean, faithful enlargement.

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
  (default) or any installed **ESRGAN** model (sharper, real detail).
- **Prompt presets** — a dropdown of ready-made prompts (Sharp & faithful / Add
  detail / Realistic skin / Nature / Architecture / Illustration / Maximum detail
  / Soft & clean) that fills the prompt **and** sets a matching creativity level.
- **Creativity (denoise)** — 0.15 faithful → 0.75 inventive.
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

---

## Toolkit

One-click installable utilities (models pulled from Hugging Face, run as
subprocesses so torch DLLs never lock the UI process):

- **Depth** — *Depth Anything V2* (depth map).
- **Background removal** — *RMBG-1.4* (cutout → transparent PNG; non-commercial
  license).
- **Click-to-cutout (SAM)** — *Segment Anything* (`facebook/sam-vit-base`): click
  an object, extract it to a transparent PNG.
- **Upscale (ESRGAN)** and **Creative upscale (SDXL)** — see [Upscaling](#upscaling).

### Prompt enhancer (AI)
The **✨ Enhance prompt** button (in each generation tab) runs a small instruct
LLM (*Qwen2.5-3B-Instruct*, PyTorch ~6 GB, one-click install) that rewrites your
idea into a detailed **English** prompt (subject, lighting, composition, style).
The model is loaded then unloaded per call → **no VRAM conflict** with generation.
It outputs only the enhanced prompt, injected straight into the prompt field. The
system prompt **detects intent from keywords** (medium/style/subject/mood) and
keeps the output medium-coherent; a **strength** selector (Light / Medium / Strong)
controls how far it expands. Krea 2 uses a Krea-specific system prompt.

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
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, enhancer, SDXL upscale)
  ui/
    theme.py                 # light theme + CSS
    generate_tab.py · library_tab.py · toolkit_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # downloads the stable-diffusion.cpp binary
  _torch_setup.py            # shared PyTorch-CUDA install helpers
  setup_tools.py             # installs PyTorch tools (depth, bg, sam, enhance, upscale)
  tools/run_*.py             # inference runners (subprocess: depth, rembg, sam, enhance, usdu)
```

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
