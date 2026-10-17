"""Optional real CPU checks in the isolated SAM 2 / Diffusers runtime."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('diffusers') is not None
if AVAILABLE:
    try:
        from transformers import Sam2Model
    except ImportError:
        AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Optional isolated SAM 2 / Diffusers validation dependencies')
class LayerInferenceTests(unittest.TestCase):
    def test_real_gguf_loading_and_layered_denoising_export_rgba_and_readable_psd(self):
        import torch
        import numpy as np
        import gguf
        from PIL import Image
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        from diffusers import (QwenImageLayeredPipeline, QwenImageTransformer2DModel,
                               AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig)
        from atelier import addons
        from atelier.engine import layer_decompose as layers
        from scripts.tools import run_qwen_layered as runner
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'model/transformer').mkdir(parents=True);(root/'quantized').mkdir()
            original=QwenImageTransformer2DModel(in_channels=64,out_channels=16,num_layers=1,
                attention_head_dim=16,num_attention_heads=2,joint_attention_dim=16,axes_dims_rope=(4,6,6),
                use_additional_t_cond=True,use_layer3d_rope=True)
            original.save_config(root/'model/transformer')
            path=root/'quantized'/addons.QWEN_LAYERED_GGUF
            writer=gguf.GGUFWriter(str(path),'qwen_image')
            for key,tensor in original.state_dict().items():writer.add_tensor(key,tensor.detach().numpy())
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            loaded=QwenImageTransformer2DModel.from_single_file(str(path),config=str(root/'model'),
                subfolder='transformer',local_files_only=True,torch_dtype=torch.float32,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32))
            self.assertTrue(all(torch.equal(value,loaded.state_dict()[key]) for key,value in original.state_dict().items()))
            vae=AutoencoderKLQwenImage(base_dim=8,z_dim=16,dim_mult=[1,2,4,4],num_res_blocks=1,input_channels=4)
            config=Qwen2_5_VLConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,rope_scaling={'type':'mrope','mrope_section':[1,1,2]},
                vision_config={'depth':1,'hidden_size':16,'intermediate_size':32,'num_heads':2,'out_hidden_size':16})
            encoder=Qwen2_5_VLForConditionalGeneration(config)
            pipe=QwenImageLayeredPipeline(transformer=loaded,vae=vae,text_encoder=encoder,
                tokenizer=None,processor=None,scheduler=FlowMatchEulerDiscreteScheduler())
            # The hooks are real; only accelerator discovery is substituted
            # because both transfer endpoints in this CPU check are CPU.
            with patch('torch.accelerator.current_accelerator',return_value=torch.device('cpu')):
                runner.apply_leaf_offload(pipe,torch.device('cpu'))
            source=root/'source.png';Image.new('RGBA',(64,64),'red').save(source)
            prompt=root/'description.txt';prompt.write_text('fixture')
            args=runner.parser().parse_args(['--addon-root',str(root),'--input',str(source),'--output',str(root/'out'),
                '--prompt-file',str(prompt),'--layers','2','--steps','2','--cfg','1'])
            class EmbeddingFixture:
                def __call__(self,**kwargs):
                    return pipe(**{**kwargs,'negative_prompt':None,'prompt_embeds':torch.zeros(1,4,16),
                                   'prompt_embeds_mask':torch.ones(1,4,dtype=torch.long)})
            with patch.object(runner,'load_pipeline',return_value=EmbeddingFixture()), \
                 patch('diffusers.pipelines.qwenimage.pipeline_qwenimage_layered.calculate_dimensions',return_value=(64,64)):
                runner.generate(args,{'mode':'cpu','dtype':'float32'})
            with Image.open(root/'out/layer-01.png') as image:
                self.assertEqual(image.mode,'RGBA')
            report=json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(len(report['layers']),2)
            self.assertEqual(report['size'],[64,64])
            # Force rendering from actual layer data, rather than reading only
            # the stored flattened composite from the PSD header.
            try:
                from psd_tools import PSDImage
            except ImportError:
                return
            psd=PSDImage.open(root/'out/layers.psd')
            self.assertEqual(len(psd),2)
            with Image.open(root/'out/composite.png') as image:
                expected=np.asarray(image.convert('RGBA')).astype('int16')
            actual=np.asarray(psd.composite(force=True).convert('RGBA')).astype('int16')
            # Pillow rounds intermediate alpha blends; the floating-point PSD
            # compositor truncates at final conversion to 8 bits.
            self.assertLessEqual(int(np.abs(actual-expected).max()),2)

    def test_q4_k_metadata_survives_real_leaf_offload_hooks_and_dequantization(self):
        import torch
        import gguf
        from types import SimpleNamespace
        from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
        from scripts.tools.run_qwen_layered import apply_leaf_offload
        model=GGUFLinear(256,8,compute_dtype=torch.float32)
        # A valid zero-valued Q4_K block: 256 logical values / 144 bytes.
        model.weight=GGUFParameter(torch.zeros((8,144),dtype=torch.uint8),
                                   quant_type=gguf.GGMLQuantizationType.Q4_K)
        pipe=SimpleNamespace(transformer=model,text_encoder=torch.nn.Linear(8,8),vae=torch.nn.Linear(8,8))
        with patch('torch.accelerator.current_accelerator',return_value=torch.device('cpu')):
            apply_leaf_offload(pipe,torch.device('cpu'))
        for _ in range(2):
            with torch.inference_mode():result=model(torch.ones(1,256))
            self.assertTrue(torch.equal(result,torch.zeros(1,8)))
            self.assertEqual(model.weight.quant_type,gguf.GGMLQuantizationType.Q4_K)

    @unittest.skipUnless(os.environ.get('TURBOSLOP_SAM2_MODEL'),'Optional pinned SAM 2.1 small checkpoint')
    def test_real_sam2_image_checkpoint_and_cached_embeddings(self):
        import torch
        from PIL import Image, ImageDraw
        from scripts.tools import run_sam2
        torch.set_num_threads(4)
        model,processor=run_sam2.load(Path(os.environ['TURBOSLOP_SAM2_MODEL']),'cpu')
        image=Image.new('RGB',(160,120),'white');ImageDraw.Draw(image).rectangle((40,30,120,90),fill='red')
        masks,scores=run_sam2.predict(model,processor,image,[[80,60]],'cpu')
        with torch.inference_mode():
            embeddings=model.get_image_embeddings(processor(images=image,return_tensors='pt')['pixel_values'])
        cached,cached_scores=run_sam2.predict(model,processor,image,[[80,60]],'cpu',embeddings)
        self.assertEqual(tuple(masks.shape),(1,3,120,160))
        self.assertTrue(torch.equal(masks,cached))
        self.assertTrue(torch.allclose(scores,cached_scores))
        chosen=masks[0,int(scores[0].argmax())]
        self.assertTrue(chosen[60,80])
        self.assertFalse(chosen[0,0])


if __name__=='__main__':
    unittest.main()
