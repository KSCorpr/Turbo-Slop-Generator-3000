# MOSS DMD header fixture

`moss_dmd_header.json.gz` contains only the tensor header of
`lodestones/MOSS-Image-SFT-8-steps-dmd/lora_latest.safetensors` at revision
`f95df83b5e90ae91ea412434b3f9b7f1c8c6314f` (Apache-2.0). It contains names,
F32 dtypes, shapes and byte offsets for 756 tensors, with no tensor payload.
The original model file SHA-256 is
`be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452`.

The tests construct sparse dummy payloads to check parsing and integration
without downloading pretrained weights or requiring Torch/CUDA.

## VOSR checkpoint headers

`vosr_headers.json.gz` contains only the 849 DiT and 186 VAE tensor headers,
`VOSR2/args.json` and the dedicated 2D VAE config from `CSWRY/VOSR` revision
`f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3`. No pretrained tensor payload is
included. Source architecture commit: `ylchen333/ComfyUI-VOSR2` at
`6a21810b3f4f1a0ffc477c55330db0ac03c671f0` (Apache-2.0 package).
`verify_vosr_runtime.py` uses it to check exact checkpoint compatibility on meta.
