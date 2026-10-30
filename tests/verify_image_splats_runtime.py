"""Real upstream tensor / PLY contracts on CPU, without model weights or GPU.

python tests/verify_image_splats_runtime.py --triposplat-source SOURCE --idol-source SOURCE
Runs small random-weight DiTs and decoders; this is not full trained inference.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-source", required=True, type=Path)
    parser.add_argument("--idol-source", required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.triposplat_source), str(args.idol_source)]
    import torch
    from tensordict import TensorDict
    import triposplat
    import model as tripo_model
    from idol.models import LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
    from idol.conditioning import ImageEncoder
    from diffusers.models import AutoencoderKLFlux2
    from atelier.engine.image_splats import validate_ply
    from scripts.tools.run_image_splats import idol_encoder_class, encoder_precision, prepare_tripo_export
    torch.set_num_threads(2)
    assert encoder_precision((7, 5)) == "fp32" and encoder_precision((6, 1)) == "fp32"
    assert encoder_precision((8, 6)) == "bf16"
    cond = {'feature1': torch.randn(1, 5, 8), 'feature2': torch.randn(1, 5, 4)}
    sample = {'latent': torch.randn(1, 8, 4), 'camera': torch.randn(1, 1, 5)}
    common = dict(q_token_length=8, in_channels=4, cam_channels=5, model_channels=64,
                  cond_channels=8, cond2_channels=4, num_blocks=1, num_refiner_blocks=2,
                  num_heads=4, num_head_channels=16, mlp_ratio=2)
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
        t = tripo_model.LatentSeqMMFlowModel(**common, out_channels=4).eval()
        t_out = t(sample, torch.tensor([500.]), cond)
        assert t_out['latent'].shape == sample['latent'].shape and torch.isfinite(t_out['latent']).all()
        i = LatentSeqMMFlowModel(**common).eval()
        i_out = i(TensorDict(sample, [1]), torch.tensor([500.]), TensorDict(cond, [1]))
        assert i_out['camera'].shape == sample['camera'].shape and torch.isfinite(i_out['latent']).all()
        print('TripoSplat and Idol: real reduced DiT forward passes succeeded.')
        # Exercise both actual decoder implementations and their PLY serializers.
        cfg = dict(num_gaussians=32, scaling_bias=0.004, opacity_bias=0.1,
                   lr={'_xyz':1.0,'_features_dc':1.0,'_opacity':1.0,'_scaling':1.0,'_rotation':0.1},
                   **{'3d_filter_kernel_size':0.0009, 'perturbe_size':1.5})
        latent = torch.randn(1, 8, 4)
        idol_octree = OctreeProbabilityFixedlenDecoder(model_channels=64, cond_channels=4, num_blocks=1, num_heads=4, mlp_ratio=2)
        points = OctreeProbabilityFixedlenDecoder.sample(idol_octree, latent, num_points=1, level=2)
        idol_gs = GaussianFixedlenDecoder(in_channels=3, model_channels=64, cond_channels=4, num_blocks=1,
                                         num_heads=4, mlp_ratio=2, representation_config=cfg)
        gaussian = idol_gs.to_representation(points, idol_gs(points, latent))[0]
        idol_ply = Path(temp) / 'idol.ply'; gaussian.save_ply(str(idol_ply)); validate_ply(idol_ply, 32)
        # Tripo's aggregate decoder expects explicit architecture options.
        oct_args = dict(triposplat.OCTREE_DECODER_ARGS)
        oct_args.update(model_channels=64, cond_channels=4, num_blocks=1, num_heads=4, mlp_ratio=2)
        gs_args = dict(triposplat.GS_DECODER_ARGS)
        gs_args.update(model_channels=64, cond_channels=4, num_blocks=1, num_heads=4, mlp_ratio=2)
        decoder = tripo_model.OctreeGaussianDecoder(oct_args, gs_args)
        decoder._MAX_VOXEL_LEVEL = 2
        g = decoder.decode(latent, 32)
        g._opacity[0] = 100  # exercise saturated alpha, not only easy random output
        prepare_tripo_export(g, torch)
        tripo_ply = Path(temp) / 'tripo.ply'; g.save_ply(str(tripo_ply)); validate_ply(tripo_ply, 32)
        g.save_splat(str(Path(temp) / 'object.splat'))
        assert (Path(temp) / 'object.splat').stat().st_size == 32 * 32
        print('Real octree and Gaussian decoders: finite PLY + Tripo SPLAT exports succeeded.')
        # Real tiny FLUX.2 VAE verifies the adapter's BF16 token packing exactly.
        vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), layers_per_block=1,
                                 latent_channels=32, norm_num_groups=32).eval().to(torch.bfloat16)
        native = ImageEncoder.__new__(ImageEncoder); native.flux2_vae = vae
        adapted_type = idol_encoder_class({}, torch, torch.bfloat16)
        adapted = adapted_type.__new__(adapted_type); adapted.flux2_vae = vae
        pixels = torch.rand(1, 3, 32, 32)
        torch.manual_seed(42); expected = native._flux2_tokens(pixels)
        torch.manual_seed(42); actual = adapted._flux2_tokens(pixels)
        assert torch.equal(actual, expected), (actual - expected).abs().max()
        vae.float()
        torch.manual_seed(42); fp32 = adapted._flux2_tokens(pixels)
        assert fp32.shape == actual.shape and torch.isfinite(fp32).all()
        print('Idol VAE adaptation: exact BF16 token parity and finite FP32 fallback succeeded.')


if __name__ == '__main__':
    main()
