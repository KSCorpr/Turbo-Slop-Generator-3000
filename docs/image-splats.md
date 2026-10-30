# TripoSplat and Idol: one image to a Gaussian object

Update with `update.bat`, then open **Tools & 3D → Image → splats**. Choose the
model, use **Install / repair selected splat model**, import an object image,
and press **Generate splats**. Both are also in **Models → Video & 3D**.

These models generate hidden sides from one image. Their outputs are Gaussian
splats, **not polygon meshes**. Use TRELLIS / Pixal3D for mesh generation, or
Capture → splats for reconstruction from overlapping photographs or a video.
A close-fitting transparent cut-out can help TripoSplat; background removal
otherwise runs automatically. The preprocessed image is shown after generation.

| | TripoSplat | Idol |
| --- | --- | --- |
| Input | One image, optional useful alpha | One image; render/generated or photograph crop modes |
| Default | 20 steps, CFG 3, shift 3, 131,072 Gaussians | 50 steps, CFG 3, shift 3, 131,072 Gaussians |
| Density choices | 32,768–262,144 | 32,768–1,048,576 |
| Exports | Standard 3DGS PLY + binary SPLAT | Standard 3DGS PLY + predicted camera JSON |
| Core license | MIT code and weights | Apache-2.0 code and core weights; separate encoder licenses apply |
| Selected downloads | ~3.78 GB | ~6.78 GB, including DINOv3 and the FLUX.2 VAE |

## Idol's gated encoders

Before installing Idol, accept the terms with your Hugging Face account at:

- [DINOv3 ViT-H+/16](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) — only the VAE is fetched.

Make your local account's token available through `HF_TOKEN` or Hugging Face's
`hf auth login`. The installer checks the pinned gated files before downloading
large Idol weights. It does not accept agreements or obtain account access for
you, and it does not put tokens in generation arguments, metadata or logs.

No full FLUX.2 diffusion model, text encoder or DINO training repository is
needed. Inference uses only installed files and runs with HF offline mode.

## Windows and GPUs

Both use one isolated Python 3.12 environment under `tools_repo/image-splats`.
The main Gradio environment does not acquire Torch, Transformers or Diffusers.
The installer selects Torch 2.6.0 / torchvision 0.21.0 with CUDA 12.4 for
Pascal, Turing and Ampere, or Torch 2.7.1 / torchvision 0.22.1 with CUDA 12.8 for
Blackwell. It probes the selected generation GPU, native SDPA and torchvision's
native deformable convolution before downloading large model files. No CUDA
compiler, flash-attn, xFormers, Triton or ComfyUI server is needed.

Ampere and newer cards use native BF16 image encoders. Pascal / Turing use
FP32 image encoders, retaining the upstream FP16 denoiser/decoder paths. This
avoids requiring BF16 from an RTX 2080 Ti or GTX 1080 Ti, but uses more memory
and can be much slower. The models require an NVIDIA CUDA GPU; a failed GPU
probe is reported rather than silently running a very slow CPU generation.

TripoSplat stages background removal, image encoding, diffusion and Gaussian
decoding, moving each completed stage off the GPU. Idol runs the eager SDK
without `torch.compile`, batched-CFG compilation or automatic FlashAttention:
each generation uses a fresh process, so compilation would be paid again.
Stopping terminates the child process; exiting releases its GPU allocations.
An active resident TRELLIS process is stopped before generation to free VRAM.

The authors report about 7.2 GB peak for Idol's native GPU configuration. This
is not a measurement on the user's GPU and does not cover the higher-memory
FP32 encoder fallback. Start at 32k or 131k Gaussians on 11–12 GB cards. More
Gaussians increase decode time, output size and preview-rendering cost.

## Storage, updates and outputs

All checkpoints live in the shared model directory, respecting its custom
location and normal rate-limited/resumable downloads. Source and weight
revisions are pinned. Only the five required TripoSplat artifacts are fetched;
the unused 3D encoder checkpoint is excluded. Idol fetches nine files: its
config/four weights, DINO config/weight, and FLUX.2 VAE config/weight.

The TripoSplat and original FLUX.2 VAE payloads have the same SHA-256 and can
reuse storage through the shared weight store. Other existing VAE variants or
DINO formats are not substituted by name. Core weights and image encoders
appear in cleanup and the installed-model update checker; updates never
silently replace pinned files.

Each generation preserves its source, preprocessed input, exports and
`request.json` in `outputs/image-splats-*.work/`. The request records the actual
seed, parameters, code revision and exact checkpoint paths. PLY headers, point
counts, complete binary payloads and finite float values are checked before
success is reported. Tripo's PLY opacity inverse is clamped like Idol's writer,
so a saturated alpha does not become an infinite logit; splat generation is
unchanged. The built-in Gaussian viewer is a preview; exported PLY
retains all splat data. No textured GLB, mesh conversion or per-step 3D preview
is claimed.
Use **Unload 3D preview** to release the browser's preview resources, retaining
all exports. **Show 3D preview** reloads the last object without another neural
generation. Dense splats can also be opened in a dedicated Gaussian viewer.

## Validation and limits

Unit tests cover pinned downloads, gated-access failure, corruption, readiness,
request snapshots, CUDA device mapping, offline inference, cancellation,
Gaussian validation and UI dispatch. Browser journeys use simulated generation
for both models and exercise the actual PLY viewer and download controls.

`tests/verify_image_splats_runtime.py` runs actual reduced upstream DiTs, octree
and Gaussian decoders on CPU, validates both PLY writers and Tripo's SPLAT
writer, and checks the Idol VAE adapter's BF16 token parity / FP32 fallback with
a real small FLUX.2 VAE. This validates tensor APIs and serialization without
loading trained checkpoints. Full trained CUDA inference and performance on
RTX 3060 / RTX 2080 Ti / GTX 1080 Ti still require validation on those GPUs.

Sources: [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat),
[Idol SDK](https://github.com/speridlabs/idol-sdk),
[Idol model card](https://huggingface.co/speridlabs/idol).
