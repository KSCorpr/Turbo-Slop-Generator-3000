# 🟢 GEN.Ai Image Workshop

A **local**, modern, lightweight image-generation studio for artists, built on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** (native
CUDA, GGUF). Generate with **Flux.2 Klein 9B** and **Krea 2 Turbo**, with an
on-demand model catalog, automatic optimization for your RTX card, LoRA, native
resolution presets, saved styles, an AI prompt enhancer, multi-reference image
editing, two upscalers, video, and a utility toolkit.

No ComfyUI, no node spaghetti — just a clean web UI.

| Tab | What it does |
|---|---|
| 🟣 **Flux.2 Klein** | fast (4 steps) · text-to-image & **multi-reference image editing** · presets, styles, LoRA |
| ⚡ **Krea 2 Turbo** | fast photorealism (8 steps, GGUF, Qwen3-VL encoder, WAN 2.1 VAE) |
| 📚 **Model Catalog** | hardware-aware recommendations, on-demand download / delete |
| 🧰 **Toolkit** | depth · background removal · click-to-cutout (SAM) · ESRGAN upscale · creative SDXL upscale |
| 🎬 **Video (LTX-2.3)** | text→video, image→video, first→last frame (⚠️ 22B, very heavy) |
| ⚙️ **Settings** | detected hardware, quantization, optimizations (auto / manual / per-generation presets) |

---

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [Generation options](#generation-options)
- [Hardware & optimization](#hardware--optimization)
- [Upscaling](#upscaling)
- [Toolkit](#toolkit)
- [Video (LTX-2.3)](#video-ltx-23)
- [Sharing on your LAN](#sharing-on-your-lan)
- [Distributing a portable package](#distributing-a-portable-package)
- [Models & sources](#models--sources)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

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

### Reference image / image-to-image
The accordion adapts to the model family:
- **Flux.2 Klein (edit model)** — **Multi-reference editing**: load the image to
  edit plus up to **2 extra reference images**, and describe the change or the
  combination in the prompt (e.g. *“put the character from image 1 into the scene
  of image 2”*). Each image is passed to the engine as a separate `-r` flag. No
  strength slider — editing is prompt-driven. Output aspect follows your image.
- **Other models (img2img)** — load a starting image and set the
  **transformation strength** (low = close to the original, high = reinvented).

### LoRA
Drop `.safetensors` / `.gguf` files into **`loras/`**, then pick up to **2** with
their weights. The `<lora:name:weight>` syntax is forwarded to the engine. Use
**↻ Refresh** after adding files, **✖ Clear** to reset.

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
- **Live preview** — updates in real time during generation.
- **Gallery** — results, captioned with their seed.
- **Saved prompts** — every image gets an A1111-style `.txt` sidecar in
  `outputs/` with the prompt, negative, model, sampler/scheduler, seed and size.

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

### One-click per-generation presets
**Settings** has four buttons that apply a curated profile for your card’s
generation: **RTX 20xx** (Turing), **RTX 30xx** (Ampere), **RTX 40xx** (Ada),
**RTX 50xx** (Blackwell). The preset is keyed to the selected GPU’s **actual
VRAM** (quantization + offload + VAE tiling), with a slight **speed** bias on
older generations and a **quality** bias on newer ones. Handy for switching fast
between machines (e.g. 2080 Ti → `Q4_K_M`, 4090 → `Q8_0`).

> Note: for GGUF models, the fp8 hardware on 40xx/50xx isn’t used by sd.cpp (it
> computes in fp16/bf16). The real differentiators across generations are VRAM,
> flash-attention and the quant bias above — not a magic fp8 speedup.

### Manual settings
With auto unchecked you control quant (diffusion / encoder), the GPU, and each
flag (flash attention, CPU offload, VAE tiling, CLIP on CPU, VAE on CPU). A custom
Hugging Face endpoint (mirror) can also be set.

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
- **Creativity (denoise)** — 0.15 faithful → 0.6 inventive.
- **Scale** — ×1.5 to ×4.
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
It outputs only the enhanced prompt, injected straight into the prompt field.

---

## Video (LTX-2.3)

Text→video, image→video, and first→last frame via sd.cpp (`-M vid_gen`).
⚠️ **22B diffusion + Gemma-3-12B encoder = very heavy** (≥16 GB ideal). On 11–12 GB:
use a low quant (`Q3_K`/`Q2_K`) + offload, and expect several minutes per clip.
Start small (640×360, 25 frames).

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
2. It produces `GEN-Ai-Image-Workshop-portable.zip` with the code, the **portable
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
- text encoder — [`unsloth/Qwen3-8B-GGUF`](https://huggingface.co/unsloth/Qwen3-8B-GGUF) (via `--llm`, offloaded to RAM)

**Krea 2 Turbo** (family `krea2`)
- diffusion — [`realrebelai/KREA-2_GGUFs`](https://huggingface.co/realrebelai/KREA-2_GGUFs) (`TURBO/…`, 8 steps, CFG 1.0)
- text encoder — [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (via `--llm`, offloaded to RAM)
- VAE — [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) (`wan_2.1_vae.safetensors`)

**Video** — [`unsloth/LTX-2.3-GGUF`](https://huggingface.co/unsloth/LTX-2.3-GGUF) + [`unsloth/gemma-3-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF).
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
    sdcpp.py                 # build/run sd-cli commands (gen, edit, upscale, video, LoRA)
    generate.py              # generation pipeline (model + hardware + LoRA) + ESRGAN upscale + video
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, enhancer, SDXL upscale)
  ui/
    theme.py                 # light theme + CSS
    generate_tab.py · library_tab.py · toolkit_tab.py · video_tab.py · settings_tab.py
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
