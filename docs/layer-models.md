# SAM 2.1 and Qwen-Image-Layered

Open **Tools & 3D → Toolkit → Cut out (SAM)** or **Layers**. Installation is
explicit; opening these tools does not download weights or import Torch.
Install/repair and inference use the shared GPU queue. Each add-on has its own
Python 3.12 environment and exits after inference to release its models.
The selected generation GPU is passed through `CUDA_VISIBLE_DEVICES`.

## SAM 2.1 small

Click **Install / repair SAM 2.1 small**, then choose it in **Segmentation model**
or **SAM segmentation model**. The default is SAM 2.1; the previous SAM 1 base
remains selectable with its existing installation.

The pinned [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small)
checkpoint occupies **184,305,280 bytes**. Only the Transformers image weights
and processor/configuration files are downloaded, without a second `.pt` copy.
The original repo's `sam2_video` configuration is adapted to `Sam2Config` by
retaining its image sub-configurations; loading checks all **357 image tensors**
for missing, unexpected or mismatched keys. This is the real SAM 2.1 image
model, without Meta's optional CUDA extension or a compiler.

- **Cut out:** click a point, inspect the highlighted mask, then extract a PNG.
- **Layers → Manual:** click objects to accumulate masks; undo/clear before export.
- **Layers → Automatic:** use the sweep density, minimum area and layer count
  controls. Image embeddings are computed once and reused across probe batches.

Both layer modes reuse the existing cleanup, deduplication and partitioning.
CLIP naming/merging and depth ordering remain optional. Without them, names and
stacking order are geometric approximations. SAM preserves source pixels; it
does not reconstruct the hidden background. New image uploads and backend
changes clear prior selections. The PNG export option now also produces a
downloadable ZIP; PSD remains available.

SAM 2.1 small inference was exercised on CPU with the supplied poster: 36 probes
yielded 8 cleaned regions. This confirms execution, not perfect semantic cuts.
Click-guided segmentation is often more useful than a dense automatic sweep
for overlapping figures, lettering and buildings. GPU behavior on Windows is
not benchmarked here.

## Qwen-Image-Layered

In **Layers**, choose **Qwen-Image-Layered — generative decomposition**, then
**Install / repair Qwen-Image-Layered Q4**. This uses the official Diffusers
pipeline with a pinned [Unsloth Q4_K_M conversion](https://huggingface.co/unsloth/Qwen-Image-Layered-GGUF)
of [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered).

Model weights occupy **30,083,002,824 bytes**: the selected GGUF transformer
(13,244,770,944), the official Qwen vision/text encoder and the RGBA VAE.
Tokenizer/processor files, Python runtime and download cache need additional
space. The installer excludes the approximately 40.9 GB BF16 transformer and
all other GGUF variants. Downloads use the app's shared transfer settings and
pinned revisions in `atelier/addons.py`; model downloads do not occur at inference.
This layered model is separate from the image-generation model Qwen 2.1 Turbo.

Controls:

| Control | Behavior |
| --- | --- |
| Layer count | 2–8 generated RGBA layers; default 4 |
| Steps | 1–100; official default 50 |
| Seed | Reproducible noise seed, default 777 |
| Resolution bucket | 640 recommended, or 1024; preserves source proportions with dimensions rounded by upstream |
| Guidance | 1–10, default 4; official CFG normalization enabled |
| Description | Optional description of the whole image; it does not assign objects to specific layers |
| Memory | Sequential CPU offload by default; whole-model offload or explicit CPU only |
| VRAM/RAM budgets | 0 uses current availability minus headroom; manual values can lower that allowance |

**Check Qwen layer memory** loads no weights. Admission reserves 4 GiB of system
RAM and 1.5 GiB of VRAM outside the job. The sequential path requires about
**43.4 GiB of RAM budget** for CPU-held weights, a temporary loading allowance
and runtime headroom. It requires at least 4 GiB of VRAM budget. This is an
estimate and a Torch allocator cap, not a guarantee of completion or an
OS-enforced RAM quota. Memory is checked again before loading.

Whole-model offload needs at least **18.4 GiB of VRAM budget** and is rejected
on a 12 GB card. Explicit FP32 CPU inference needs substantially more RAM and
can be very slow. CUDA failure does not silently select CPU. Sequential offload
transfers individual modules repeatedly and can also be very slow over PCIe;
more layers and resolution 1024 increase activation and attention memory.
The default uses native Diffusers **leaf-level group offloading** on the
transformer, text encoder and VAE, with streaming disabled. Accelerate's
`enable_sequential_cpu_offload()` loses GGUF metadata in this runtime
([upstream issue](https://github.com/huggingface/diffusers/issues/14691)), so that
broken path is not used.
Start with 2–4 layers at 640 on a 12 GB GPU. Stop/Cancel terminates the process
tree, including during loading. Failed/cancelled jobs do not publish partial
export folders.

Each successful run saves `outputs/layers-qwen-*` with:

- `layer-NN.png`: full-canvas RGBA layers, preserving generated alpha.
- `layers.psd`: editable raster layers in the original model order.
- `composite.png`: the actual alpha composition of those generated layers.
- `source.png`: the original source for comparison, not a fake reconstructed background.
- `manifest.json`: order, dimensions, controls, model revisions and memory plan.
- `layers.zip`: all of the above, including the PSD.

The UI shows the composite, individual PNGs in a gallery and PSD/ZIP downloads.
The model does not supply reliable object names: exported names are neutral
`Layer NN`. Rename/reorder them in your editor. To decompose a layer further,
upload its PNG as a new source. Generated shapes, colors and lettering may
change; the model does not recover the original authoring document.

These are **bitmap layers**, not SVG paths or editable text. Vectorize selected
PNG layers separately with the existing VTracer tool, and reconstruct text in
Illustrator/Inkscape when editable typography matters.

## Validation and limits

Pinned runtime: Transformers 4.57.6, Diffusers 0.37.1, gguf 0.17.1. NVIDIA
Ampere/Turing uses Torch 2.7.1 / CUDA 12.6; Blackwell uses Torch 2.8.0 / CUDA 12.8,
with matching torchvision. Neither changes the application's Torch environment.

The actual GGUF header's **1,934 tensor names/shapes** were checked against the
official Layered transformer on the meta device, including its 3D layer RoPE and
additional timestep condition. Tests run real GGUF loading, a tiny Qwen layered
transformer/VAE with two denoising steps, RGBA output, and PSD recomposition with
an independent reader. A Q4_K block also exercises native leaf offload hooks
and dequantization on CPU without losing quantization metadata. These CPU tests
do not measure CUDA transfers. Tiny random-weight tests do not establish visual quality.
The complete 30 GB Qwen checkpoint was not run on RTX 3060/Windows here;
speed, maximum layer count/resolution and reconstruction fidelity remain unverified.

Installed runtimes and model weights appear separately in **System → Manage &
clean**. Normal maintenance recognizes both add-on folders; normal application
updates preserve model downloads and exports.
