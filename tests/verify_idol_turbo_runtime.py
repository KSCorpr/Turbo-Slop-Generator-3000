"""Compare the adapter to the pinned author's sampler on a real reduced Idol DiT.

Run in the image-splats runtime (CPU works), without trained weights or encoders:
  python tests/verify_idol_turbo_runtime.py --idol-source SDK --reference-turbo turbo.py
Optional --checkpoint-header HEADER --config CONFIG validates the real checkpoint's
448 tensor names/shapes against the full denoiser on the meta device (no VRAM).
Reference: akhaliq/idol-turbo@45cdd41f1f4d6d70496daa7d619f57ad79307a37.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idol-source', type=Path, required=True)
    parser.add_argument('--reference-turbo', type=Path, required=True)
    parser.add_argument('--checkpoint-header', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.idol_source))
    import torch
    from safetensors.torch import save_file
    from tensordict import TensorDict
    import idol.pipeline as api
    from idol.models import LatentSeqMMFlowModel
    from idol.models import layers
    from scripts.tools.run_image_splats import apply_idol_turbo
    torch.set_num_threads(2)
    assert sha256(args.reference_turbo.read_bytes()).hexdigest() == 'd607930134bc1375b5b9d652a2bab8ca0b38f7eceef973686a5dba96b6b2d264'
    spec = importlib.util.spec_from_file_location('reference_turbo', args.reference_turbo)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    def targets(model):
        for group in ('noise_refiner', 'context_refiner', 'blocks'):
            for i, block in enumerate(getattr(model, group)):
                for suffix, linear in [('attn.qkv', block.attn.qkv), ('attn.out', block.attn.out),
                                       ('mlp.mlp.0', block.mlp.mlp[0]), ('mlp.mlp.2', block.mlp.mlp[2])]:
                    yield f'{group}.{i}.{suffix}', linear

    if args.checkpoint_header:
        header = json.loads(args.checkpoint_header.read_text())
        config = json.loads(args.config.read_text())
        with torch.device('meta'):
            full = LatentSeqMMFlowModel(**config['denoiser']['args'])
        expected = {}
        for name, linear in targets(full):
            for phase in ('slow', 'fast'):
                expected[name + '.' + phase + '_A'] = [32, linear.in_features]
                expected[name + '.' + phase + '_B'] = [linear.out_features, 32]
        assert set(header) == set(expected) and len(expected) == 448
        for key, shape in expected.items():
            assert header[key]['shape'] == shape and header[key]['dtype'] == 'F32', key
        print('Real checkpoint header: all 448 tensor names/shapes match the full native Idol architecture.')

    torch.manual_seed(42)
    model = LatentSeqMMFlowModel(q_token_length=8, in_channels=4, cam_channels=5, model_channels=64,
                                cond_channels=8, cond2_channels=4, num_blocks=1, num_refiner_blocks=2,
                                num_heads=4, num_head_channels=16, mlp_ratio=2).eval().requires_grad_(False)
    state = {}
    for name, linear in targets(model):
        for phase in ('slow', 'fast'):
            state[name + '.' + phase + '_A'] = torch.randn(32, linear.in_features) * .01
            state[name + '.' + phase + '_B'] = torch.randn(linear.out_features, 32) * .01
    noise = TensorDict({'latent': torch.randn(1, 8, 4), 'camera': torch.randn(1, 1, 5)}, [1], device='cpu')
    cond = TensorDict({'feature1': torch.randn(1, 5, 8), 'feature2': torch.randn(1, 5, 4)}, [1], device='cpu')
    original_sampler = api.sample_flow
    original_backend = layers.ATTN_BACKEND
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
        path = Path(directory) / 'adapter.safetensors'
        save_file(state, path)
        upstream = SimpleNamespace(denoiser=copy.deepcopy(model))
        with patch('huggingface_hub.hf_hub_download', return_value=str(path)):
            reference.patch_pipeline_for_turbo(upstream)
        reference_sampler = api.sample_flow
        api.sample_flow = original_sampler
        adapted = SimpleNamespace(denoiser=copy.deepcopy(model))
        apply_idol_turbo(adapted, path)
        adapted_sampler = api.sample_flow
        # With shift 3, the four-step grid ends at t=0.5: all four use slow.
        for steps in (4, 5, 8):
            expected = reference_sampler(upstream.denoiser, noise.clone(), cond, None, steps=steps, verbose=False)
            phases = []
            def record_phase(module, inputs):
                phases.append(module.blocks[0].attn.qkv.active)
            handle = adapted.denoiser.register_forward_pre_hook(record_phase)
            actual = adapted_sampler(adapted.denoiser, noise.clone(), cond, None, steps=steps, verbose=False)
            handle.remove()
            assert all(torch.equal(actual[k], expected[k]) for k in noise.keys()), steps
            assert len(phases) == steps, phases
            assert set(phases) == ({'slow'} if steps == 4 else {'slow', 'fast'}), phases
            if steps == 5:
                assert phases == ['slow'] * 4 + ['fast'], phases
            assert layers.ATTN_BACKEND == original_backend
            assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
        print('Native reduced FP16 DiT: exact upstream parity at 4/5/8 steps; one evaluation/step; expert phases verified.')
        # The unpatched model still uses the original sampler in this process.
        expected = original_sampler(model, noise.clone(), cond, None, steps=5, cfg_strength=0, verbose=False)
        actual = adapted_sampler(model, noise.clone(), cond, None, steps=5, cfg_strength=0, verbose=False)
        assert all(torch.isfinite(actual[k]).all() for k in noise.keys())
        assert all(torch.equal(actual[k], expected[k]) for k in noise.keys())
        with patch.object(adapted.denoiser, 'forward', side_effect=RuntimeError('interrupted')):
            try:
                adapted_sampler(adapted.denoiser, noise, cond, None, verbose=False)
            except RuntimeError as exc:
                assert str(exc) == 'interrupted'
            else:
                raise AssertionError('Expected interruption')
        assert layers.ATTN_BACKEND == original_backend
        assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
        print('Standard Idol routing and attention/TF32 restoration after interruption verified.')
        for damage in ('missing', 'shape', 'dtype', 'nonfinite'):
            broken = dict(state)
            key = next(iter(broken))
            if damage == 'missing':
                del broken[key]
            elif damage == 'shape':
                broken[key] = broken[key][:1]
            elif damage == 'dtype':
                broken[key] = broken[key].half()
            else:
                broken[key] = torch.full_like(broken[key], float('nan'))
            save_file(broken, path)
            pristine = copy.deepcopy(model)
            try:
                apply_idol_turbo(SimpleNamespace(denoiser=pristine), path)
            except ValueError:
                assert isinstance(pristine.blocks[0].attn.qkv, torch.nn.Linear)
            else:
                raise AssertionError('Accepted invalid adapter: ' + damage)
        print('Malformed adapter names, shapes, dtypes and non-finite tensors rejected before mutation.')
    api.sample_flow = original_sampler


if __name__ == '__main__':
    main()
