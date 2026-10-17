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

Generation controls include token count (256–16,384; default 4,096), seed, temperature, top-p,
top-k, repetition penalty, KV cache, source proportions and a shared generation
time budget (0–7,200 seconds, excluding model loading). **0 means no time limit**
and is the default; token limits and Stop still apply. A stopping criterion checks
an enabled time budget after each token step; it is not a hard process deadline. Stop terminates
the whole subprocess tree, including during loading.

Request **1–4 variants**. They run sequentially, without upstream's extra hidden
candidates or simultaneous batches. Seeds increment per attempt. A sequence
without EOS can retain the prefix ending at its last actual path-color token;
the unfinished tail is discarded. It must still decode into nonempty, valid
vector paths with matching colors. Invalid SVG and blank renders are rejected.
These drawings are explicitly labeled **PARTIAL** in the UI, filenames
(`NN.partial.svg` / `.png`), SVG description and manifest. They can be missing
shapes: valid XML does not establish that generation finished or followed the source.

Valid complete or partial variants survive an enabled generation-time limit;
user Stop or an inference failure discards the unfinished job. The ZIP includes
candidate SVGs/PNGs, original numeric token sequences, settings, memory plan and
peak Torch VRAM in `manifest.json`. Select a variant to preview/download it
separately. Default `result.svg` and `preview.png` prefer the first complete
variant, falling back to the first explicitly marked partial drawing.

The original 2,048-token / 300-second defaults could stop a slow or complex
generation before its real EOS token **196999**, even when memory admission and
model placement succeeded. The 4B training token configuration and inference
code agree on this EOS. Only a real model-generated EOS marks completion.
Upstream inference wraps even unfinished output with BOS/EOS for serialization;
Turbo Slop uses that decoder but labels such output partial and drops its unfinished tail.
Progress reports generated tokens, elapsed time and tokens/second about every
10 seconds. Completion reports distinguish EOS, token limit, time limit, both
limits or an unknown stop. The user's RTX 3060 log confirmed successful loading
with all weights on GPU (`{"": 0}`); a later log confirmed a 4,096-token stop
without EOS. No GPU output-quality claim follows from successful model loading.

If all attempts fail, the UI offers `diagnostics.zip` instead of losing the
generation. It contains `diagnostic.json` with per-attempt stop statistics and
`generated-token-ids-NN.json` containing the numeric output of each attempt.
The diagnostic folder is under `outputs/svg-diagnostic-*`. To recover a previous
failed run, select an OmniSVG mode, import its `diagnostics.zip` in **Saved
OmniSVG diagnostic ZIP**, then click **Recover SVG from saved tokens · CPU**.
This runs only the installed official CPU decoder and renderer: no model weights,
processor, CUDA memory admission or new inference. If the saved output contains
no recoverable paths it still fails; recovery does not complete missing shapes
or solve the model's failure to emit EOS. It needs no model download or engine
reinstall. Successful exports also record their attempt statistics in
`manifest.json`. The saved slow processor is selected
explicitly with `use_fast=False` to retain preprocessing parity and avoid the
default-change warning. These changes need an application update only.

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
depends on image complexity and token count. Default output budget is 4,096 tokens
(256–16,384 selectable for either AI SVG engine),
input processing is capped at 512 × 512 pixels, and each
job exits to release its model. CPU inference uses FP32 and requires at least
19 GiB free system RAM; it can be slow. CUDA needs a compatible driver for the
isolated PyTorch CUDA 12.8 build.

HiVG reconstructs illustrations. It can change lettering, contours and fine
details; compare with the input. Use CPU tracing when following the source is
more important than generative simplification. An unfinished SVG token sequence
is reported as a failure rather than silently exported as a repaired result.
The runner uses deterministic greedy generation instead of overriding the
checkpoint's near-zero-temperature preset with sampling at temperature 0.7.
It retains the checkpoint EOS tokens and also stops at the tokenizer's single
`</svg>` token. `dtype` replaces the deprecated `torch_dtype` loading argument.
Complete native SVG XML is accepted when the model emits XML instead of its
atomic token format; no missing drawing commands or closing tags are invented.

**HiVG remains experimental.** Its checkpoint vocabulary contains 500 `<SEG_N>`
tokens, but the public release has no matching segment-to-atomic-token dictionary
or expansion step. This is documented in the open upstream
[segment-decoding issue](https://github.com/ximinng/HiVG/issues/5). If any such
tokens are generated, the runner refuses to silently discard their geometry and
recommends OmniSVG or VTracer. Raising the token budget cannot supply this missing
dictionary. The original traceback alone cannot distinguish missing output from
an upstream detokenizer failure: the official decoder can also lose the outer
closing tag when an atomic path is missing its own closing token.

Failures in output validation preserve a downloadable `diagnostics.zip` under
`outputs/svg-diagnostic-*`, containing `raw-output.txt` and `diagnostic.json`.
The report distinguishes a hard token limit, chat EOS, SVG completion or an
unknown stop; it records generated token count and the upstream decoder error.
The UI displays **Failed** and offers that ZIP, with no successful SVG/preview.
Raw diagnostics stay local. No reinstall or model download is needed for these
runner/UI changes: update the application, then retry.
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
