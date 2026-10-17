# SVG & layers

Open **Tools & 3D → SVG & layers**. Installation is explicit; opening the tab
does not download weights or load Torch. The two install buttons create separate
Python 3.12 environments under `tools_repo/svg`, `tools_repo/hivg` and
`tools_repo/omnisvg`, without
changing the main interface or sd.cpp. Install/repair and inference share the
studio GPU queue; Stop terminates the child process tree.

| Mode | Input | Runs on | Result |
| --- | --- | --- | --- |
| VTracer | One image | CPU | Traced colors and contours |
| HiVG image | One image | Selected CUDA GPU, CPU offload as needed | Reconstructed SVG |
| HiVG text | Description | Selected CUDA GPU, CPU offload as needed | Generated SVG |
| OmniSVG 1.1 4B | One image or description | Selected CUDA GPU + RAM, CPU offload or CPU only | Selectable SVG variants |
| LayerTracer 4-panel | Prepared 2 × 2 drawing sequence | CPU | Base and three difference layers |
| LayerTracer 9-panel | Prepared 3 × 3 drawing sequence | CPU | Base, seven difference layers and outlines |

Every successful run saves `result.svg`, a bounded PNG preview, `manifest.json`
and `vectors.zip` under `outputs/svg-*`. LayerTracer's ZIP also contains each
individual SVG layer; the combined SVG contains named `<g>` groups. These are
vector shapes, without an embedded bitmap. Generated SVG is validated before
rendering; active content, external references and embedded images are rejected.
SVG downloads open in external vector editors. In-app previews are PNG renders.
An error or cancellation before export does not create a completed output folder.

## OmniSVG 1.1 4B and controls

Use **Install / repair OmniSVG 4B**, then select its image or text mode.
The pinned [OmniSVG 1.1 4B checkpoint](https://huggingface.co/OmniSVG/OmniSVG1.1_4B)
occupies **7,694,002,862 bytes**, plus about 12 MB of Qwen tokenizer/processor
assets. Only those Qwen assets are downloaded, never the base Qwen weights.
Source and model revisions are pinned in `atelier/addons.py`. Downloads use the
shared Hugging Face transfer settings. Runtime/source disk use is additional.
The CPU decoder uses upstream's token mapping, path tensors and
DeepSVG serialization. Three eager notebook/Cairo/video imports are removed
from the pinned DeepSVG copy; PNG previews use resvg without a Windows Cairo DLL.
The source license remains in `tools_repo/omnisvg/source/LICENSE` (Apache 2.0).

The model is created on the meta device with its final 197,000-token vocabulary,
then the OmniSVG state dictionary is memory-mapped and assigned with strict
key/shape checking. This avoids loading Qwen base weights or resizing live GPU
embeddings. Transformers **4.51.3** matches the checkpoint's parameter layout;
the source checkpoint's **825 tensor names/shapes** were checked against this
configuration. CUDA 12.6 / Torch 2.7.1 is selected for Pascal/Turing/Ampere/Ada;
Blackwell uses CUDA 12.8 / Torch 2.8.0, with matching torchvision versions.

**Check OmniSVG memory** runs an isolated diagnostic on the selected generation
GPU, without loading weights. The same check runs again before each generation.

- **Auto:** place weights on GPU and RAM with Accelerate; retain 2 GiB inside
  the Torch budget for activations/cache. Disk offload is disabled.
- **CPU offload:** retain weights in RAM and move modules to the selected GPU
  when used. This reduces persistent VRAM use and can be substantially slower.
- **CPU only:** explicit FP32 inference. CUDA failure never silently selects it.
- **VRAM budget:** 0 means currently free VRAM minus 1.5 GiB of external
  headroom. A manual value can lower this limit, never exceed current headroom.
  Torch's allocator limit covers weights, activations and cache; CUDA/driver
  allocations outside Torch are separate. A GPU budget below 4 GiB is rejected.
- **RAM budget:** 0 means available RAM minus 4 GiB for the OS and other apps.
  A manual value is clamped to that availability. The conservative loading
  admission estimate is about **16.3 GiB** for GPU modes or **23.5 GiB** for CPU
  FP32. This is an estimate and placement budget, not an OS-enforced hard quota.

Generation controls include token count (256–4,096), seed, temperature, top-p,
top-k, repetition penalty, KV cache, source proportions and a shared generation
time budget (30–1,800 seconds, excluding model loading). Transformers checks the
time budget after a token step; it is not a hard process deadline. Stop terminates
the whole subprocess tree, including during loading.

Request **1–4 variants**. They run sequentially, without upstream's extra hidden
candidates or simultaneous batches. Seeds increment per attempt. An incomplete
token sequence (missing EOS), invalid SVG or blank render is rejected. Completed
variants survive a generation-time limit if at least one is valid; user Stop or
an inference failure discards the unfinished job. The ZIP includes all completed
candidate SVGs/PNGs, settings, memory plan and peak Torch VRAM in `manifest.json`.
Select any completed variant to preview/download it separately. The default
`result.svg` and `preview.png` refer to variant 1.

Image inputs keep their proportions with white padding by default, without
automatic background removal. Turning this off stretches the image to the
upstream 448 × 448 input canvas. Pixel difference compares the normalized source
with each rendered candidate: lower means closer pixel colors, **not** verified
lettering, semantic fidelity or overall quality. Always compare the source.

Validation covers strict loading and real two-token inference on a tiny Qwen
model, the pinned official SVG decoder and actual PNG/ZIP exports. The full 4B
model's image quality, speed and GPU memory behavior on RTX 3060/Windows remain
unverified here; no claim of a guaranteed 12 GB fit is made.

## HiVG

[HiVG-3B-Base](https://huggingface.co/xingxm/HiVG-3B-Base) uses the official
[HiVG inference and SVG detokenizer](https://github.com/ximinng/HiVG), not a
generic Qwen text prompt. Both checkpoint and code revisions are pinned in
`atelier/addons.py`. Model weights alone occupy **8.14 GB (decimal)**; installation
also needs disk space for Torch, dependencies, source and download caches.
Downloads use Turbo Slop's shared Hugging Face transfer settings and can resume.

The runner uses Transformers, SDPA and local files only. No vLLM, FlashAttention
build or remote model code is required. CUDA loading reserves 2 GiB of currently
free VRAM and 2 GiB of available system RAM; Accelerate can place weights in RAM
or temporary disk storage. This budget is for weights: activation/KV memory also
depends on image complexity and token count. Default output budget is 2,048 tokens
(256–4,096 selectable), input processing is capped at 512 × 512 pixels, and each
job exits to release its model. CPU inference uses FP32 and requires at least
19 GiB free system RAM; it can be slow. CUDA needs a compatible driver for the
isolated PyTorch CUDA 12.8 build.

HiVG reconstructs illustrations. It can change lettering, contours and fine
details; compare with the input. Use CPU tracing when following the source is
more important than generative simplification. An unfinished SVG token sequence
is reported as a failure rather than silently exported as a repaired result.
End-to-end HiVG GPU generation has not been validated on an RTX 3060 here.

## LayerTracer: supported stage and limits

This integration runs the **official sequence-to-SVG functions** from
[showlab/LayerTracer](https://github.com/showlab/LayerTracer), pinned at
`8e089f92795b4f32dcae8f7dbbd5ced20395dad7`. The MIT-licensed source and LICENSE are
downloaded during the CPU engine installation. We do not install its training
stack or replace it with VTracer under the same name.

The input must show successive stages of the **same drawing**, aligned in equal
cells with no gutters. Four panels are read clockwise: top left, top right,
bottom right, bottom left. Nine panels are read row by row: the first contains
outlines, the second base colors, and the rest successive drawing additions.
The wrapper includes the last 8 → 9 difference, which upstream's `main()` omits.
Empty difference layers are retained. The algorithm follows upstream's grayscale
difference detection and morphology: small additions and equal-luminance color
changes can be lost. Arbitrary photo grids are not valid drawing sequences.

**FLUX image/text → drawing sequence is not included.** Upstream's image-to-sequence
script loads FLUX.1 directly onto CUDA without an offload path, and the upstream
merge workflow reports about 50 GB of system memory. It cannot be offered as a
working end-to-end RTX 3060 12 GB path without separate memory/offload work and
GPU validation. Prepare a grid externally, or use HiVG/VTracer for a single image.

SVG runtimes and HiVG weights appear separately in **System → Manage & clean**.
Existing `outputs/` protection also covers SVG exports; normal maintenance keeps
both installed add-on folders.
