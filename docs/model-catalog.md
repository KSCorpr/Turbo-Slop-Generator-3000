# Unified Models catalog

Every Images workspace has a **Model variant (installed weights)** selector
above the resolution controls. Choose an installed GGUF or approved
INT8 / BF16 / FP16 variant for that specific model before generating.
**Encoder, VAE & local files** contains the corresponding component choices.
Menus show precision, format, actual disk size and filenames without contacting
Hugging Face. Install additional variants in Models, then click
**Refresh installed variants** or return to the image model's tab.

Selections are saved per model and shared with the Models catalog. Automatic
uses the usual hardware target and compatible installed-weight fallback.
An explicitly selected file must still be installed: generation reports a
missing file instead of silently substituting another variant. Each request
captures its file paths before starting; later menu changes do not alter that
generation. Files placed in `models/custom` appear as **Local / custom** entries
in the same menus; select the matching component and architecture. These local
selections are saved separately from catalog download choices.

Open **Models** to install image-generation weights and the models used by the
Toolkit, layers/SVG tools, video and 3D. Filter by usage or search, then choose
a package. The page renders one image-weight form, not one copy per model.
Switching images restores that model's saved weight choices.

- **Image generation:** select exact diffusion/encoder/VAE weights, or keep
  automatic choices. **Load weights and sizes** performs the metadata request
  only when clicked. A component table identifies dependencies shared with
  other image models. Delete retains files required by another image model.
- **Toolkit:** depth/normals, background removal, image descriptions, prompt
  enhancement, CLIP region naming, legacy SAM, face restoration, SDXL creative
  upscale, ADetailer, ESRGAN and modern Spandrel upscalers.
- **Layers & SVG:** SAM 2.1, Qwen-Image-Layered, VTracer/LayerTracer CPU runtime,
  HiVG and OmniSVG. Runtime-only packages are explicitly identified.
- **Video & 3D:** ConvRot and classic SeedVR2, LTX 2.5, TRELLIS.2, Pixal3D and
  Brush. Choose the requested variant before installing. SeedVR2's selected
  DiT and shared VAE can now be downloaded here before the first inference.

Toolkit installation buttons remain available as contextual shortcuts; they
use the same weight paths and transfer helpers. Image description and
multi-view captioning share one Qwen-VL model. Prompt enhancement and Xanax
share their language model. Generation and conversion constraints remain those
of each engine: the catalog does not make their checkpoint formats interchangeable.
Some large tools also install a separate Python/CUDA runtime. Those versions
are retained independently to avoid dependency conflicts.

**Download limit (MB/s; 0 = unlimited)** is the existing global transfer setting,
also available under System. Installation, image download, deletion and
consolidation use the shared job queue. **Stop download / installation** stops
catalog subprocess trees, or aborts a managed HTTP transfer between chunks.
Interrupted Hub weights retain their separate resumable partial files.
Refresh status after an install; tool pages already opened may need a refresh
or application restart to show newly available controls.

## GPU and weight precision

Image packages show a device-specific table for **INT8, BF16, FP16, FP32**
and GGUF quantizations. The generation GPU selected in System is used, including
on multi-GPU PCs. CUDA compute capability reported by the driver takes precedence
over architecture names. Native BF16 is identified on Ampere and newer NVIDIA
GPUs; Turing/Pascal can load supported BF16 files through sd.cpp's conversion
path, but are not presented as native BF16 devices. Unknown and CPU-only devices
are explicitly identified instead of assuming support.

GGUF and Safetensors are containers, not compute precisions. An INT8 ConvRot
checkpoint does not promise end-to-end INT8 arithmetic. The backend chooses
kernels, dequantization, intermediate precision and accumulation. Likewise,
FP32 weight storage is not an automatic image-quality improvement.

**Load weights and sizes** lists only files approved for the chosen model,
with their storage type, format, actual download size, installation state and
memory advice. A **suggested for GPU memory** label uses a conservative heuristic:
16-bit weights with native support and room under 70% of total GPU memory, then
INT8, then the profile's GGUF target or a GGUF fitting that budget. No suggestion
is made when sizes or GPU capability are unknown. Encoders retain their existing
RAM-aware target. This heuristic is not a measured VRAM peak or speed/quality
benchmark; the generation placement planner remains responsible for offload.
Weights alone approaching/exceeding GPU memory receive an explicit offload note.

INT8/BF16 choices for MOSS, Qwen 2.1 Turbo, Fibo 2 and Krea Realism are visible
before any metadata query. Krea 2 Turbo and Z-Image Turbo also offer their
Comfy-Org INT8 ConvRot/BF16 Safetensors alongside the existing GGUF source.
Cross-repository choices persist across restarts, download and generation;
loading sizes still shows the original GGUF variants after selecting Safetensors.
FP16/FP32 files are offered only where published and matched by the approved
catalog sources; unavailable precisions are not fabricated or converted on disk.
Suggestions label a choice; they do not overwrite saved selections or download
additional precision variants automatically.

Backend references: [Krea 2](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/krea2.md),
[Z-Image](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md),
[NVIDIA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

## Reusing identical weights

The transfer layer indexes completed weight files by **SHA-256**, not by their
names. A Hub LFS checksum can therefore reuse the exact file even when another
repository or tool expects it under a different name/path. It reuses the local
file before another HTTP body transfer. Content variants of the same size are
kept distinct. Downloads replace files atomically, preserving aliases used by
another model.

On the same filesystem, identical files are joined with hard links. Both
engine paths remain valid and the payload occupies disk space once. This
requires a filesystem supporting hard links, such as NTFS on Windows; it does
not need administrator privileges or symbolic links. Across volumes, a verified
local copy can avoid another download but still consumes separate disk space.
Unsupported hard links and copies retained during consolidation are reported.

No permanent extra weight copy is created by the index. `userdata/shared-weights.json`
records file fingerprints and hashes; missing or changed files are not reused
from stale records. Losing this index does not lose model weights. Sharing
assumes weights are immutable: updates should replace files, rather than edit
their bytes in place.

## Existing installations

Click **Consolidate existing identical weights** once to hash weights under the
configured model directory and `tools_repo`. This is an explicit disk-reading
operation that can take several minutes on large collections. It does not run
at startup. Stop is checked while hashing. Python environments, source trees,
temporary offload folders and download caches are excluded.

Only byte-identical completed weight files are linked. Unique files are retained.
Different GGUF quantizations, Safetensors/PyTorch serializations and precision
variants are separate assets. Existing unused formats are not guessed and
deleted automatically. In the Toolkit's Transformers installations, new
downloads now select one native checkpoint format (Safetensors preferred,
PyTorch fallback), along with required configs/tokenizers/Python model code.
They exclude ONNX, TF, redundant `.pth` copies and alternate exports. SDXL VAE
and Tile ControlNet likewise request only the checkpoint actually loaded.

System storage totals count shared file data once. Individual item rows show
logical file sizes, which can overlap. Deleting one hard-link alias does not
claim its bytes freed until the last link is removed. Independent engine paths
remain usable when another model's alias is deleted.

## Validation

Tests cover real temporary-file hard links, modified weights, atomic replacement,
last-link deletion accounting, cancellation, and unsupported-link fallback.
Local HTTP tests verify reuse across different repository paths without a
second GET, different bytes with the same size, and safe resume after stopping.
UI tests cover model-specific saved weights, package selection, the shared queue,
lazy metadata, and the existing browser configuration-size budget. No pretrained
model or GPU inference changes are required by this catalog/storage update.
