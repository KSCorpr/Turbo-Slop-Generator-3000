# TripoSplat, Idol and Idol Turbo: one image to a Gaussian object

Update with `update.bat`, then open **Tools & 3D → Image → splats**. Choose the
model, use **Install / repair selected splat model**, import an object image,
and press **Generate splats**. All three are also in **Models → Video & 3D**.

These models generate hidden sides from one image. Their outputs are Gaussian
splats, **not polygon meshes**. Use TRELLIS / Pixal3D for mesh generation, or
Capture → splats for reconstruction from overlapping photographs or a video.
A close-fitting transparent cut-out can help TripoSplat; background removal
otherwise runs automatically. The preprocessed image is shown after generation.

| | TripoSplat | Idol | Idol Turbo |
| --- | --- | --- | --- |
| Input | One image, optional useful alpha | One image; render/generated or photograph crop modes | Same as Idol |
| Default | 20 steps, guidance 3, shift 3 | 50 steps, guidance 3, shift 3 | **5 steps**, distilled guidance 0, fixed shift 3 |
| Density choices | 32,768–262,144 | 32,768–1,048,576 | 32,768–1,048,576 |
| Default density | 131,072 | 131,072 | 131,072 |
| Exports | Standard 3DGS PLY + binary SPLAT | Standard 3DGS PLY + predicted camera JSON | Same as Idol |
| Core license | MIT code and weights | Apache-2.0; separate encoder licenses apply | Apache-2.0; same encoder licenses |
| Selected downloads | ~3.78 GB | ~6.78 GB, including DINOv3 and the FLUX.2 VAE | Same Idol files + **117.5 MB** adapter |

## Idol Turbo

Select **Idol Turbo**, then install it once. Existing verified Idol files are
reused in place, so an installed Idol only needs the 117.5 MB
`idol_turbo_lora.safetensors` adapter. No duplicate base model or extra text
encoder is downloaded. The original Idol remains a separate selection.

The recommended preset uses **5 Euler steps**; the exposed range is **4–8**.
Guidance is already distilled into the LoRA, so there is one denoiser evaluation
per step. The SDK's guidance strength is **0**, equivalent to disabling the
additional CFG pass (ComfyUI describes this as CFG 1). Guidance and time shift
are locked to the author's recipe in Turbo and become editable again on the
other models.

Two rank-32 experts use alpha 128: Slow at shifted `t >= 0.5`, Fast below.
The time grid uses shift 3. At five steps, this means four Slow evaluations and
one Fast; at four steps the grid stays entirely in the Slow phase. The adapter
uses native SDPA and TF32 during sampling and restores the previous settings
even after an error. Compilation stays disabled. All adapter files are loaded
locally; tensor names, shapes, FP32 dtype and finite values are checked.

Some objects can look softer than the original 50-step Idol output. Try five
or eight steps, or switch back to Idol for comparison. Fewer denoiser calls do
not remove background removal, image encoding or Gaussian decoding costs; no
end-to-end speedup on consumer GPUs is claimed.

## Idol's gated encoders

Before a first install of Idol or Idol Turbo, accept the terms with your Hugging Face account at:

- [DINOv3 ViT-H+/16](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) — only the VAE is fetched.

Make your local account's token available through `HF_TOKEN` or Hugging Face's
`hf auth login`. The installer checks the pinned gated files before downloading
large Idol weights. It does not accept agreements or obtain account access for
you, and it does not put tokens in generation arguments, metadata or logs.
An encoder whose installed weight has the expected size and SHA-256 is reused
without another gated network check. Missing or corrupt encoders still require
valid account access before large downloads begin.

No full FLUX.2 diffusion model, text encoder or DINO training repository is
needed. Inference uses only installed files and runs with HF offline mode.

## Windows and GPUs

All three use one isolated Python 3.12 environment under `tools_repo/image-splats`.
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
Turbo adds exactly one pinned adapter from `akhaliq/idol-turbo` revision
`45cdd41f1f4d6d70496daa7d619f57ad79307a37`; its older v4 and ComfyUI split
checkpoints are not fetched. Its adapter has a separate cleanup entry; deleting
that entry retains the shared Idol weights and encoders. Deleting the shared
Idol weights makes both Idol variants require repair.

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
for all three models and exercise Turbo's locked controls, switching back to
Idol, the actual PLY viewer and download controls.

`tests/verify_image_splats_runtime.py` runs actual reduced upstream DiTs, octree
and Gaussian decoders on CPU, validates both PLY writers and Tripo's SPLAT
writer, and checks the Idol VAE adapter's BF16 token parity / FP32 fallback with
a real small FLUX.2 VAE. This validates tensor APIs and serialization without
loading trained checkpoints. Full trained CUDA inference and performance on
RTX 3060 / RTX 2080 Ti / GTX 1080 Ti still require validation on those GPUs.

`tests/verify_idol_turbo_runtime.py` compares a reduced native FP16 Idol DiT
against the author's pinned `turbo.py`: exact tensor parity at 4/5/8 steps,
one evaluation per step, expert switching, original-model routing and settings
restoration after interruption. It rejects malformed adapters before mutation.
Its optional header/config check validates all 448 real checkpoint tensor
names/shapes against the full native denoiser on the meta device. These checks
use random reduced weights, not full trained CUDA generation.

Sources: [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat),
[Idol SDK](https://github.com/speridlabs/idol-sdk),
[Idol model card](https://huggingface.co/speridlabs/idol),
[Idol Turbo model card and adapter source](https://huggingface.co/akhaliq/idol-turbo/tree/45cdd41f1f4d6d70496daa7d619f57ad79307a37).
