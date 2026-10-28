# Sampler / scheduler comparison presets

The **Quality preset** menu in each Images workspace keeps the existing presets
first, including the recommended configuration. It adds the alternatives below,
with the step count in brackets and an **Experimental** explanation when selected. These are
curated starting points based on the solver, available step budget and native
sd.cpp schedules, not GPU image-quality benchmarks or publisher recommendations.

| Model | Added sampler / scheduler pairs (steps) | CFG |
| --- | --- | --- |
| Flux.2 Klein 9B | Res 2S / Flux.2 (6); Euler / SGM Uniform (8); Euler / Beta (8) | 1 |
| Flux.3 Image Klein | Res Multistep / Discrete (16); Heun / Discrete (16); Euler / Beta (20) | 1 |
| Qwen Image 2.1 Turbo | Heun / official Turbo sigmas (8); Res 2S / official Turbo sigmas (8); DPM++ 2M / official Turbo sigmas (8) | 1 |
| MOSS-Image SFT | Euler / Simple (50); ER SDE / Bong Tangent (40); DPM++ 2M / Beta (40); Res Multistep / SGM Uniform (40) | 3 |
| Fibo 2.0 Dev | Heun / Discrete (20); Res Multistep / Discrete (24); Euler / Beta (24) | 5 |
| Fibo 2.0 Edit Turbo | Res 2S / Discrete (4); Euler / SGM Uniform (6); Euler / Beta (8) | 1 |
| Krea 2 Turbo | Res Multistep / Discrete (12); DPM++ 2M / SGM Uniform (12); Euler / Beta (12) | 1 |
| Krea 2 Realism | Res Multistep / Simple (8); DPM++ 2M / SGM Uniform (8); Euler / Bong Tangent (8) | 1 |
| Z-Image Turbo | Res Multistep / Discrete (12); DPM++ 2M / SGM Uniform (12); Euler / Beta (12) | 1 |

Selecting a preset applies its sampler, scheduler, steps and CFG together. It
also clears a stale manual flow shift, restoring **0 (auto)** so the native
model shift can be used where the selected scheduler supports it. Alternative
schedulers do not necessarily apply shift in the same way as the native one.
Prompt, seed, resolution, model weights and reference images are kept as chosen.

## What to compare

- **Res 2S or Heun:** second-order corrections, generally with more model
  evaluations per step than Euler. Compare both image quality and elapsed time;
  matching the number of steps does not match the compute budget.
- **Res Multistep or DPM++ 2M:** deterministic solvers that use previous steps.
  Most alternatives give them a longer step budget to build their history.
- **SGM Uniform, Beta or Bong Tangent:** different sigma spacing, potentially
  changing composition and texture. The Beta preset uses sd.cpp's default Beta
  parameters. Bong Tangent is a schedule, not a sampler.
- **ER SDE:** a stochastic solver. Its noise trajectory can change texture;
  the same seed is a useful comparison input, not a guarantee of equivalent
  noise or pixels across solvers, devices or engine versions.

Start by retaining the same prompt, seed, dimensions, weights and references.
Compare with the first baseline preset. Higher step counts and more expensive
solvers do not guarantee better results on distilled checkpoints.

## Model constraints

**Qwen Turbo:** the explicit sequence remains
`1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0`.
The scheduler field displays Auto but does not create a generic schedule: these
sigmas take precedence. Only the sampler changes, and the Images sampler menu
is limited to the four listed solvers. Eight steps, CFG 1 and no flow shift
remain enforced. Alpha output and reference editing keep their existing paths.

**MOSS:** these four alternatives use the ordinary SFT model and no DMD adapter.
The existing **DMD fast (8 steps)** preset remains Euler / Auto, CFG 1 and LoRA
strength 0.7 applied at runtime. Selecting a standard or experimental preset
after DMD disables automatic adapter injection for the next request. A DMD
adapter manually selected in the separate LoRA controls is still a user choice.

**Fibo Edit Turbo:** every alternative still requires at least one reference.
All distilled model alternatives retain CFG 1; the negative prompt remains
inactive. Fibo Dev keeps CFG 5 and ordinary MOSS keeps CFG 3.

No LCM, TCD, CFG++ or ComfyUI-only scheduler names are added as presets. No
preview, VAE, model precision, download, cache or memory-placement setting is
changed. The source of truth is [`config/models.yaml`](../config/models.yaml).

## Native references

- [sd.cpp sampler and scheduler names](https://github.com/leejet/stable-diffusion.cpp/blob/master/src/stable-diffusion.cpp)
- [Flux.2](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md),
  [Flux.3](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md),
  [Krea 2](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/krea2.md),
  [Z-Image](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md)
- [Qwen 2.1 Turbo: fixed sigmas and independently configurable solver](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md#turbo)
- [MOSS](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/moss_image.md),
  [Fibo 2.0](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/fibo2.md)
