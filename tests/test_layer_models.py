"""Layer model admission, isolated dispatch, exports and install boundaries."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from atelier import addons, inventory, settings
from atelier.engine import layer_decompose as layers, tools
from scripts import setup_layers
from scripts.tools import run_qwen_layered as qwen


class LayerModelTests(unittest.TestCase):
    def test_controls_reject_nonfinite_unknown_fractional_and_unlisted_values(self):
        for change in ({"layers": 3.5}, {"layers": 9}, {"steps": True}, {"cfg": float('nan')},
                       {"resolution": 800}, {"memory": "auto"}, {"ram": -1}, {"arbitrary": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                layers.arguments(change)

    def test_ram_and_vram_admission_honor_actual_availability_and_explicit_modes(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.mem_get_info.return_value = (11*qwen.GIB, 12*qwen.GIB)
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.get_device_name.return_value = "RTX 3060"
        plan = qwen.memory_plan(torch, 60*qwen.GIB, {**layers.DEFAULTS, "vram": 8, "ram": 50})
        self.assertEqual(plan["gpu_budget"], 8*qwen.GIB)
        self.assertEqual(plan["ram_budget"], 50*qwen.GIB)
        self.assertEqual(plan["dtype"], "bfloat16")
        for options, ram in (({"memory":"model"},60), ({"ram":16},60), ({},32)):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                qwen.memory_plan(torch,ram*qwen.GIB,{**layers.DEFAULTS,**options})
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError,"CUDA is unavailable"):
            qwen.memory_plan(torch,60*qwen.GIB,layers.DEFAULTS)
        cpu = qwen.memory_plan(torch,100*qwen.GIB,{**layers.DEFAULTS,"memory":"cpu"})
        self.assertEqual(cpu["dtype"],"float32")

    def test_local_rgba_export_preserves_alpha_order_and_all_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            back = Image.new("RGBA",(64,64),"red")
            front = Image.new("RGBA",back.size)
            ImageDraw.Draw(front).rectangle((16,16,48,48),fill=(0,0,255,128))
            expected = Image.alpha_composite(back,front)
            layers.export_layers([back,front],root,{"engine":"Qwen-Image-Layered"})
            self.assertEqual(Image.open(root/"composite.png").tobytes(),expected.tobytes())
            self.assertEqual(Image.open(root/"layer-02.png").getpixel((32,32))[3],128)
            self.assertEqual((root/"layers.psd").read_bytes()[:4],b"8BPS")
            metadata=json.loads((root/"manifest.json").read_text())
            self.assertEqual(metadata["order"],"background to foreground")
            with zipfile.ZipFile(root/"layers.zip") as archive:
                self.assertTrue({"layers.psd","layer-01.png","layer-02.png","composite.png","manifest.json"}.issubset(archive.namelist()))
            with self.assertRaisesRegex(ValueError,"RGBA"):
                layers.export_layers([back.convert('RGB'),front],root,{})

    def test_qwen_dispatch_keeps_exports_outside_child_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            calls=[]
            def child(command, log, cwd, env):
                calls.append((list(map(str,command)),Path(cwd),env))
                layers.export_layers([Image.new('RGBA',(32,32),'red'),Image.new('RGBA',(32,32),'blue')],
                                     Path(cwd)/'export',{})
            with patch.object(addons,'ready',return_value=True),patch.object(settings,'OUTPUT_DIR',root), \
                 patch.object(layers.JOB,'run',side_effect=child),patch.object(layers,'_gen_gpu_index',return_value=1):
                output=layers.run(Image.new('RGB',(32,32)),{'layers':2},'a couple')
            command,work,env=calls[0]
            self.assertEqual(command[0],str(addons.python('qwen-layered')))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
            self.assertIn('--memory',command)
            self.assertNotIn('a couple',command)
            self.assertFalse(work.exists())
            self.assertTrue((output/'layers.psd').exists())

    def test_sam2_cutout_dispatch_uses_its_own_python_and_selected_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def child(command,*args,**kwargs):
                self.assertEqual(command[0],str(addons.python('sam2')))
                self.assertEqual(command[command.index('--backend')+1],'sam2')
                self.assertEqual(command[command.index('--model-dir')+1],str(addons.root('sam2')/'model'))
                self.assertEqual(kwargs['gpu_index'],1)
                dest=Path(command[command.index('--output-dir')+1]);dest.mkdir(exist_ok=True)
                Image.new('RGBA',(32,32),'red').save(dest/'source_sam.png')
            with patch.object(addons,'ready',return_value=True),patch.object(settings,'TMP_DIR',root), \
                 patch.object(settings,'OUTPUT_DIR',root),patch.object(tools,'_run_tool',side_effect=child), \
                 patch.object(tools,'_gen_gpu_index',return_value=1):
                cut,_=tools.sam_segment(Image.new('RGB',(32,32)),10,10,backend='sam2')
                self.assertTrue(cut.is_file())

    def test_sam_png_exports_are_downloadable_as_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.png';Image.new('RGB',(32,32),'red').save(source)
            with patch.object(settings,'OUTPUT_DIR',root):
                files=tools._layers_to_files(source,[np.ones((32,32),bool)],['object'],'test',False,True)
            bundle=next(p for p in files if p.suffix=='.zip')
            with zipfile.ZipFile(bundle) as archive:self.assertEqual(len(archive.namelist()),2)

    def test_installer_downloads_only_the_selected_quant_and_does_not_mark_failure_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            downloads=[]
            def download(**kwargs):
                downloads.append(kwargs)
                if len(downloads)==2:raise RuntimeError('simulated interrupted download')
            with patch.object(addons,'root',return_value=root),patch.object(setup_layers,'environment',return_value='python'), \
                 patch.object(setup_layers,'pip'),patch.object(setup_layers.hardware,'detect_gpus',return_value=[]), \
                 patch.object(setup_layers.transfers,'snapshot_download',side_effect=download):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    setup_layers.install('qwen-layered')
            self.assertNotIn('transformer/*',downloads[0]['allow_patterns'])
            self.assertIn('transformer/config.json',downloads[0]['allow_patterns'])
            self.assertEqual(downloads[1]['revision'],addons.QWEN_LAYERED_GGUF_REVISION)
            self.assertEqual(downloads[1]['allow_patterns'],[addons.QWEN_LAYERED_GGUF,'README.md'])
            self.assertFalse((root/'installed.json').exists())

    def test_inventory_lists_models_separately_from_isolated_runtimes(self):
        items={item.key:item for item in inventory.items()}
        self.assertEqual(items['sam2_models'].paths,[addons.root('sam2')/'model'])
        self.assertEqual(items['qwen-layered_models'].paths,[addons.root('qwen-layered')/'model',addons.root('qwen-layered')/'quantized'])
        self.assertNotIn(addons.root('qwen-layered')/'model',items['engine_qwen-layered'].paths)


if __name__ == '__main__':
    unittest.main()
