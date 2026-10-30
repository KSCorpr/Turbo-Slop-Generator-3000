# Tiny autoencoders (TAESD)

Run `update.bat`, restart, then search **TAESD** in **Models** and install the
package matching your image model. Each architecture has one shared, pinned
Safetensors file. No Python neural runtime or weight conversion is needed.

| Image models | Tiny autoencoder | Download |
|---|---|---:|
| Flux.2 Klein, MOSS SFT, Fibo 2 Dev / Edit Turbo | TAEF2 | 10.72 MB |
| Qwen Image 2.1 Turbo (RGBA) | TAEQI2.1 | 30.59 MB |
| Krea 2 Turbo / Realism (Wan 2.1 VAE) | TAEW2.1 | 22.64 MB |

Open **Images → your model → Encoder, VAE & local files → Tiny autoencoder
(TAESD)**. The choice is saved independently for each model:

* **Projection preview / original VAE (default)** keeps the existing lightweight
  projection previews and original final VAE. No TAE file is loaded.
* **TAE preview only / original VAE for final image** uses `--preview tae`,
  `--taesd PATH` and `--taesd-preview-only`, retaining `--vae PATH`. TAE is a
  learned preview decoder rather than the cheaper color projection. The final
  image still uses the selected original or texture-fix VAE. TAE is not loaded
  for jobs that do not request previews.
* **TAE fast encoding & final decoding** replaces the original VAE for encoding
  and final decoding. It passes `--taesd PATH` and omits `--vae`. The native VAE
  can be absent in this mode. This trades reconstruction detail for lower VAE
  cost, and can also change reference-image encoding. It does not reduce the
  diffusion step count. Use preview-only to retain the original final decoder.

TAE weights are chosen by the model's VAE architecture, not just its number of
latent channels. Qwen 2.1 uses its dedicated RGBA network; it cannot use the
older Qwen/Wan TAE. The experimental Qwen *Tinier* weights are not exposed by
this integration. Custom VAE overrides are refused when TAE is active because
their architecture has not been established by this catalog.

Downloads use the common resumable transfer, rate limit and checksum store.
The installed file's SHA-256 is verified before loading, with fingerprinted
caching. TAEs appear in storage management and installed-model update checks.
Shared families reuse the same path. The Krea TAE's pinned HF mirror was
verified byte-for-byte against the official GitHub Safetensors file.

The automatic memory planner counts both VAE and TAE in preview-only mode,
and only the TAE in full mode. Export metadata records the TAE filename and
mode. A missing download, damaged file or engine lacking the required options
stops generation with an actionable message; modes do not silently fall back.

Sources:

* [sd.cpp TAESD guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/taesd.md)
* [Tiny AutoEncoder model families](https://github.com/madebyollin/taesd)
* [TAEHV / Wan weights](https://github.com/madebyollin/taehv)
* [sd.cpp Krea 2 VAE integration](https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model/integrations/krea2.cpp)

Command construction and pipeline/UI behavior are covered by automated tests.
All four real, pinned encoders and decoders were also run on CPU with 64×64
inputs using the upstream Python implementations, including Qwen's four RGBA
channels and the single-frame Wan architecture. The native sd.cpp build 1017
was checked for the required CLI options. These checks do not measure full
diffusion inference speed or quality on an RTX 3060.
