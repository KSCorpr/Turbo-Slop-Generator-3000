# TripoSplat: one image to a Gaussian object

Open **Tools & 3D → Image → splats** and install TripoSplat. The five public
weight files total about 3.78 GB. Import one object image, choose the Gaussian
count, then select **Generate splats**. The default uses 20 steps, guidance 3,
shift 3 and 131,072 Gaussians. A transparent cutout can help. TripoSplat
removes the background and shows the prepared image after generation.

TripoSplat infers hidden sides and exports a Gaussian PLY and binary SPLAT.
Use TRELLIS.2 / Pixal3D for meshes or **Capture → splats** for reconstruction
from overlapping photographs. **Unload 3D preview** frees viewer resources.
**Show 3D preview** reopens the saved result.

## Runtime and storage

TripoSplat uses an isolated Python 3.12 environment under
`tools_repo/image-splats`. The installer probes CUDA, native SDPA and
torchvision deformable convolution before downloading weights. Ampere and
newer GPUs use BF16 encoders. Pascal and Turing use FP32 for those encoders
and FP16 for diffusion and decoding. The FP32 path uses more memory.

The installer pins source and weights and uses the shared model store.
Generation runs offline on the selected CUDA GPU. Each output folder
contains the original input, prepared image, exports and `request.json`.
The app checks the PLY count, payload length and finite values before
reporting success.

Idol and Idol Turbo were retired because their encoders required an account.
Existing weights can be managed from **System → Storage & help**.

## Validation

The unit and browser tests cover installation routing, local requests,
cancellation, preview and exports. The browser uses simulated inference.
`tests/verify_image_splats_runtime.py` also exercises a reduced upstream
model and its PLY/SPLAT serializers on CPU. Full trained CUDA performance
depends on the selected GPU and resolution.

Source: [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat).
