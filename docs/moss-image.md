# MOSS-Image SFT

Update TurboSlop and its engine with `update.bat` (or `./update.sh`), restart,
then open **Models**, search **MOSS**, and download the selected image weights.
Generate under **Images → MOSS-Image SFT**. Missing required files keep the
model marked as incomplete. No separate PyTorch environment is installed.

## Weights and memory

| Component | Default source/file | Download size |
| --- | --- | --- |
| Diffusion | `Comfy-Org/MOSS-Image` / `diffusion_models/moss_image_sft_int8_convrot.safetensors` | 5,413,698,958 bytes |
| Text encoder | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 bytes |
| Tokenizer | `OpenMOSS-Team/MOSS-Image-Base` / `tokenizer/tokenizer.json` | 19,989,325 bytes |
| VAE | Existing Flux.2 VAE, `split_files/vae/flux2-vae.safetensors` | 336,211,292 bytes, reused if installed |

Sizes and filenames were verified on the Hub on 18 October 2026. The VAE in
`Comfy-Org/MOSS-Image/vae/flux2-vae.safetensors` has the same SHA-256 as the
historical Flux.2 catalog file:
`868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`.
The existing local path is retained to avoid another copy/download. The small
VAE is a different checkpoint and is not substituted silently.

The catalog also offers the SFT BF16 diffusion checkpoint (10,164,655,808 bytes)
and exact GGUF encoder quantizations through **Load weights and sizes**. The
automatic encoder default is Q4_K_M; an explicit manual encoder quantization
or an exact per-model file selection takes priority. A compatible Qwen3.5-9B
encoder already installed can be reused. Qwen3 and Qwen3-VL encoders from
other image models are not interchangeable with this one.

On a 12 GB RTX, start at 1024×1024 with **System automatic mode**. The existing
placement planner measures the actual weight files: when the diffusion/VAE
fit, it keeps their weights on the generation GPU and the text weights in RAM,
with text computation on the generation GPU. It can use additional offload
when the selected weights or resolution exceed the available VRAM. Weight
download size is not a peak VRAM estimate. No timing or GPU inference result
is claimed by this integration.

## Engine and sampling

The current Comfy-Org weights require **sd.cpp build 975 or newer**, which adds
split Q/K/V tensor support. Build 973 introduced the architecture, but is not
sufficient for all currently published weights. Known older official builds
are rejected before resolving/loading the model. A manually installed engine
must expose `--tokenizer`; the external tokenizer is passed in both ordinary
generation and ADetailer.

The official preset is **Euler, 50 steps, CFG 3, Auto scheduler, flow shift 0
(automatic)**. sd.cpp supplies the native time shift of 5 and skips DiT layer
22 in the negative CFG branch. Dimensions must be multiples of 16. Negative
prompts can affect the image at CFG 3. This SFT checkpoint is not distilled.

Community DMD [8-step adapters](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
and [Rapid-MOSS 4-step adapters](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora)
exist. Their application, precision and sampling requirements need validation
with the native backend. This integration does not download or enable them,
and does not claim that reducing Steps reproduces their acceleration/quality.
The adapter author's older 8-step instructions are explicitly deprecated;
they should not be treated as the current loader contract.

## Layout prompts

Natural-language prompts work as usual. For layout control, paste a complete
JSON object into **Prompt**, using the official format below. Set output
dimensions to those used to design the boxes. Coordinates are pixels relative
to the canvas center, ordered `[y_min, x_min, y_max, x_max]`.

```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A cream travel poster with a turquoise mountain and a large NICE title."
  },
  "items": [
    {"label": "text", "caption": "Large black title", "text": "NICE", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Turquoise mountain illustration", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

This example uses 1024×1024. JSON prompts are validated before inference and
passed unchanged: prepended presets, photo styles and artistic styles are
skipped. Incorporate the desired style in the JSON captions. Outputs remain
raster images; the JSON describes composition rather than editable layers.

## Verification and sources

CPU tests cover catalog selection, exact dependency paths, shared VAE reuse,
required tokenizer, engine-version/capability guards, text/image-to-image CLI
arguments, ADetailer, 3060 placement planning with sparse weight files, sampling
defaults, and unchanged/invalid JSON through the actual UI callback. They do
not load pretrained weights or verify image quality on a GPU.

- [OpenMOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
- [Native sd.cpp MOSS guide](https://github.com/leejet/stable-diffusion.cpp/blob/95b3090664d019fcf8fa032a44d1d4793ba244b8/docs/moss_image.md)
- [Split QKV support in build 975](https://github.com/leejet/stable-diffusion.cpp/commit/95b3090664d019fcf8fa032a44d1d4793ba244b8)
- [Comfy-Org weights](https://huggingface.co/Comfy-Org/MOSS-Image)
- [Qwen3.5-9B GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
