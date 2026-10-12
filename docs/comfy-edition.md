# Comfy edition

This edition runs ComfyUI locally behind TurboSlop's existing controls.
Qwen Turbo and SDXL need no sd-cli binary. The studio starts the owned
process automatically, waits for its readiness check, submits internal
workflows, streams previews and imports the resulting images into the gallery.
Models remain cached between compatible jobs when memory allows. Other GPU
tools stop that process before they run so its allocations are returned.

## Install and use

Download the `comfy-edition` branch into a separate folder to try it alongside
the classic edition. `install.bat` or `./install.sh` installs the GUI, an
isolated full Python 3.12 runtime for Comfy, and the classic runner used by
the remaining models. No image weights are downloaded by the base installer.

For an existing checkout of this branch: open **System → Settings → Local
image engine (optional settings)** and click **Install / repair Comfy**.
The default **Image engine** is **Comfy local**. Changing engines requires
restarting TurboSlop so the catalogue and encoder choices refresh.

Download **Qwen Image 2.1 Turbo** or **SDXL Base 1.0** from **Models**.
An existing Toolkit SDXL checkpoint is reused, and Toolkit installation
also reuses a catalogue SDXL download. Qwen's diffusion and RGBA VAE weights
are shared with the classic edition. Its classic GGUF text encoder cannot
be loaded by the native Comfy workflow: download the native
`qwen3vl_8b_int8_convrot.safetensors` encoder (about 8.7 GB), or the larger
BF16 variant. The native encoder includes the vision path for references.
No separate GGUF vision projector is needed for Comfy.

Use the normal **Images → Qwen Image 2.1 Turbo** or **Images → SDXL** tab.
Toolkit SDXL upscale uses the same local engine with optional ControlNet
Tile, tiled decoding and overlap blending. Install its weights using the
Toolkit installer if necessary.

## Scope

| Workflow | Engine in this edition |
| --- | --- |
| Qwen 2.1 Turbo text generation, up to three references, RGBA | Local Comfy |
| SDXL generation, img2img, masks, safetensors LoRAs | Local Comfy |
| Outpaint with Qwen / SDXL | Local Comfy, existing canvas/compositing controls |
| Tiled SDXL upscale and optional ControlNet Tile | Local Comfy |
| Flux.2 Klein, Krea 2, Z-Image, Xanax | Existing classic runner |
| ADetailer, GGUF conversion and native HD | Existing classic runner |
| SeedVR2 / ConvRot, video and 3D | Existing separate engines |

Qwen Turbo always uses its official eight steps, CFG 1, Euler and distilled
sigmas. Seamless patterns, custom flow shift and native SDCPP HD do not have
Comfy implementations yet. Select the classic engine for those workflows.
ADetailer with Qwen still needs its classic GGUF encoder/projector components.
Use safetensors LoRAs and native safetensors encoders with Comfy; this edition
does not install third-party GGUF loader extensions.

## Hardware and updates

Comfy core is pinned to **v0.39.0**, commit
`b0b743566f65daafc423b4fea8a2fbda94b3384a`, with Torch 2.13.0,
torchvision 0.28.0 and Comfy Kitchen 0.2.37. Its dependencies stay out of
the Gradio environment. The installer selects CUDA 13 for tensor-core GPUs
with a sufficiently recent NVIDIA driver (580 or newer), CUDA 12.6
compatibility mode for older drivers, MPS for Apple Silicon, or CPU when no
GPU is detected. The CUDA installer verifies an actual tensor operation,
not just device detection.

**CUDA 12.6 compatibility mode does not enable the latest optimized INT8
CUDA kernels.** It uses standard attention, low VRAM mode and CPU offload;
speed and memory use may be worse. CUDA 13 with a compatible driver is the
recommended path for recent RTX cards. The compatibility path has not been
tested on the user's RTX 3060 or GTX 1080 Ti. CPU verifies the installation
but is too slow for normal use with large Qwen models.

`update.bat` / `./update.sh` stays on this branch, updates the studio and
brings an already installed Comfy runtime to the release pinned by that
code. A new upstream Comfy release is adopted by changing the core/dependency
pins and passing verification; arbitrary new models still need catalogue
entries and a tested internal workflow. This is not an unattended nightly
upgrade. If runtime installation fails, the UI reports that it needs repair
instead of treating it as ready; the classic runner remains selectable.

Prompts and images go only to the owned process on **127.0.0.1**, using HTTP
and WebSocket communication on an automatically selected local port. It
does not attach to another Comfy installation or accept remote engine URLs.
The node editor is never opened. Generation runs offline once dependencies
and weights have been downloaded. Installation and model downloads need
internet access.

## Measure on your machine

In the same Settings panel, select Qwen Turbo or SDXL and click **Compare
Classic / Comfy on my PC**. Download both engines' required weights first.
It measures a first image and three subsequent images at 1024×1024 using
the full model preset. The seed sequence is identical between engines but
changes on every image so Comfy must sample, rather than return a cached
latent. The JSON report includes timings, whole-card sampled VRAM and saved
image paths. No engine preference is changed automatically.

Compare the images too: the encoder formats differ, and timings alone do
not prove equivalent visual quality. First-image measurements include
process/model startup but do not reset the filesystem cache. The existing
**Measure classic GPU placement** button still measures classic placement
flags only.

## Validation

Unit tests cover workflow routing, the distilled Qwen schedule, local-only
requests, subscription-before-submit, lost preview connections, RGBA,
cancel/error handling, process reuse and release, sidecars for completed
batch images, tile boundaries and benchmark cache avoidance. Browser tests
exercise the existing studio and the SDXL tab with simulated inference.

The installer launches the real pinned Comfy server to validate native node
contracts for Qwen, references, masks, LoRAs, SDXL and ControlNet, then
executes an RGBA load/save roundtrip through its actual submission protocol.
CI performs this check on Linux and Windows without model weights.
**Actual Qwen/SDXL model inference, visual quality and GPU speed remain to
be validated on hardware with the downloaded weights.** No measured GPU
speedup is claimed by this branch.
