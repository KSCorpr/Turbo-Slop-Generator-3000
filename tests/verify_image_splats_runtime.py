"""Real upstream tensor / PLY contracts on CPU, without model weights or GPU.

python tests/verify_image_splats_runtime.py --triposplat-source SOURCE
Runs small random-weight DiTs and decoders; this is not full trained inference.
"""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-source", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.triposplat_source))
    import torch
    import triposplat
    import model as tripo_model
    from atelier.engine.image_splats import validate_ply
    from scripts.tools.run_image_splats import encoder_precision, prepare_tripo_export
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
        latent = torch.randn(1, 8, 4)
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
if __name__ == '__main__':
    main()
