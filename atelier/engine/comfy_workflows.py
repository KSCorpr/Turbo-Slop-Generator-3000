"""Small native Comfy workflows, filled by the existing TurboSlop controls."""
from __future__ import annotations

TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
TILED = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
SAMPLERS = {"euler_a": "euler_ancestral", "dpm++2m": "dpmpp_2m",
            "dpm++2m_v2": "dpmpp_2m", "dpm++2s_a": "dpmpp_2s_ancestral"}


class Workflow:
    def __init__(self):
        self.nodes = {}
        self.output = ""

    def add(self, kind: str, **inputs) -> list:
        key = str(len(self.nodes) + 1)
        self.nodes[key] = {"class_type": kind, "inputs": inputs}
        return [key, 0]

    def finish(self, image, prefix: str):
        self.output = self.add("SaveImage", images=image, filename_prefix=prefix)[0]
        return self


def qwen(*, diffusion: str, encoder: str, vae: str, prompt: str, width: int,
         height: int, seed: int, prefix: str, references=(), init=None,
         mask=None, loras=()) -> Workflow:
    g = Workflow()
    model = g.add("UNETLoader", unet_name=diffusion, weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=encoder, type="qwen_image", device="default")
    vae_node = g.add("VAELoader", vae_name=vae)
    for filename, strength in loras:
        model = g.add("LoraLoaderModelOnly", model=model, lora_name=filename,
                      strength_model=float(strength))
    inputs = {f"images.image_{i}": g.add("TurboSlopLoadRGBA", image=image)
              for i, image in enumerate(references, 1)}
    conditioning = g.add("TextEncodeQwenImage21", clip=clip, prompt=prompt,
                          negative_prompt="", vae=vae_node, resolution=0, **inputs)
    positive, negative = conditioning, [conditioning[0], 1]
    latent = ([conditioning[0], 2] if references else
              g.add("TurboSlopEmptyQwen21", width=width, height=height))
    if init:
        image = g.add("TurboSlopLoadRGBA", image=init)
        latent = g.add("VAEEncodeTiled", pixels=image, vae=vae_node, **TILED)
    if mask:
        noise = g.add("LoadImageMask", image=mask, channel="red")
        latent = g.add("SetLatentNoiseMask", samples=latent, mask=noise)
    sigmas = g.add("ManualSigmas", sigmas=TURBO_SIGMAS)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    sampled = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=seed,
                    cfg=1.0, positive=positive, negative=negative, sampler=sampler,
                    sigmas=sigmas, latent_image=latent)
    image = g.add("VAEDecodeTiled", samples=sampled, vae=vae_node, **TILED)
    return g.finish(image, prefix)


def sdxl(*, checkpoint: str, prompt: str, negative: str, width: int, height: int,
         seed: int, steps: int, cfg: float, prefix: str, init=None, mask=None,
         strength=1.0, vae=None, controlnet=None, cn_scale=0.6,
         sampler="euler", scheduler="normal", loras=()) -> Workflow:
    g = Workflow()
    loaded = g.add("CheckpointLoaderSimple", ckpt_name=checkpoint)
    model, clip, vae_node = loaded, [loaded[0], 1], [loaded[0], 2]
    if vae:
        vae_node = g.add("VAELoader", vae_name=vae)
    for filename, weight in loras:
        lora = g.add("LoraLoader", model=model, clip=clip, lora_name=filename,
                     strength_model=float(weight), strength_clip=float(weight))
        model, clip = lora, [lora[0], 1]
    positive = g.add("CLIPTextEncode", text=prompt, clip=clip)
    neg = g.add("CLIPTextEncode", text=negative, clip=clip)
    if init:
        image = g.add("LoadImage", image=init)
        latent = g.add("VAEEncodeTiled", pixels=image, vae=vae_node, **TILED)
        if controlnet:
            cn = g.add("ControlNetLoader", control_net_name=controlnet)
            applied = g.add("ControlNetApplyAdvanced", positive=positive, negative=neg,
                            control_net=cn, image=image, strength=float(cn_scale),
                            start_percent=0.0, end_percent=1.0, vae=vae_node)
            positive, neg = applied, [applied[0], 1]
    else:
        latent = g.add("EmptyLatentImage", width=width, height=height, batch_size=1)
    if mask:
        noise = g.add("LoadImageMask", image=mask, channel="red")
        latent = g.add("SetLatentNoiseMask", samples=latent, mask=noise)
    sampled = g.add("KSampler", model=model, positive=positive, negative=neg,
                    latent_image=latent, seed=seed, steps=int(steps), cfg=float(cfg),
                    sampler_name=SAMPLERS.get(sampler, sampler),
                    scheduler="normal" if scheduler in (None, "", "auto") else scheduler,
                    denoise=float(strength) if init else 1.0)
    image = g.add("VAEDecodeTiled", samples=sampled, vae=vae_node, **TILED)
    return g.finish(image, prefix)
