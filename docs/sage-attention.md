# Native SageAttention

After `update.bat`, open **System → Settings** and enable **SageAttention (CUDA)**.
The setting is opt-in and is saved immediately. It does not switch Automatic
system setup to Manual or get overwritten by machine/memory profiles.
Use **Refresh acceleration support** after updating the engine without restarting.

Eligible hardware: NVIDIA compute capability **8.0 or newer**. RTX 3060 is
eligible; RTX 2080 Ti, GTX 1080 Ti, CPU and Apple Metal are not. Driver-reported
compute capability takes precedence over the architecture fallback. Old engines
without `--sage-attn` and known non-CUDA engine archives leave the option inactive.
A compatible CLI flag alone does not prove that a manually built binary contains
the required CUDA kernels: upstream requires CUDA 12+ and its patched GGML.

The switch applies to sd.cpp image generation (including reference editing,
Xanax, outpaint and text-to-3D's initial image) and LTX 2.5. It does not affect
SeedVR2, VOSR, TripoSplat, Idol or TRELLIS, which use separate engines.
No Python SageAttention, Torch or Triton package is installed for this switch.

The generation log reports whether a requested SageAttention run is active or
why standard attention remains in use. Image TXT/JSON sidecars record the enabled
state. FlashAttention remains available for operations that SageAttention cannot
handle; sd.cpp gives SageAttention priority for eligible diffusion operations.

With **mixed-GPU auto-fit**, sd.cpp chooses diffusion compute placement itself.
If an incompatible GPU is visible (for example a 1080 Ti beside a 3060), TurboSlop
keeps standard attention for that request rather than assume the newer card will
run every SageAttention operation. It preserves the memory-placement settings.
LTX pins its selected generation card, so its mono-GPU auto-fit stays eligible.

SageAttention quantizes attention and changes numerical results. A fixed seed
can produce a different image. Compare output quality and full request time at
the same resolution, seed and sampler. Short sequences may not be faster.

Validation covers real application request construction for all seven image
models in Automatic and Manual modes, native LTX command routing, unsupported
hardware/engine fallbacks, saved settings and UI callbacks. Full CUDA inference,
image-quality comparisons and speed on RTX 3060 have not been measured in the
CPU-only development environment.

[Upstream requirements and limitations](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sage_attention.md)
