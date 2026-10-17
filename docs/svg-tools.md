# SVG & layers

Open **Tools & 3D → SVG & layers**. Installation is explicit; opening the tab
does not download weights or load Torch. The two install buttons create separate
Python 3.12 environments under `tools_repo/svg` and `tools_repo/hivg`, without
changing the main interface or sd.cpp. Install/repair and inference share the
studio GPU queue; Stop terminates the child process tree.

| Mode | Input | Runs on | Result |
| --- | --- | --- | --- |
| VTracer | One image | CPU | Traced colors and contours |
| HiVG image | One image | Selected CUDA GPU, CPU offload as needed | Reconstructed SVG |
| HiVG text | Description | Selected CUDA GPU, CPU offload as needed | Generated SVG |
| LayerTracer 4-panel | Prepared 2 × 2 drawing sequence | CPU | Base and three difference layers |
| LayerTracer 9-panel | Prepared 3 × 3 drawing sequence | CPU | Base, seven difference layers and outlines |

Every successful run saves `result.svg`, a bounded PNG preview, `manifest.json`
and `vectors.zip` under `outputs/svg-*`. LayerTracer's ZIP also contains each
individual SVG layer; the combined SVG contains named `<g>` groups. These are
vector shapes, without an embedded bitmap. Generated SVG is validated before
rendering; active content, external references and embedded images are rejected.
SVG downloads open in external vector editors. In-app previews are PNG renders.
An error or cancellation before export does not create a completed output folder.

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
