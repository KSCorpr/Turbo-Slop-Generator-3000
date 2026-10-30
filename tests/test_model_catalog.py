"""Real file sharing, byte identity, catalog coverage and shared download UI."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch, Mock

from atelier import addons, inventory, model_catalog, model_store, settings, transfers


class SharedWeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for name,value in {"ROOT":self.root,"MODELS_DIR":self.root/'models',"USERDATA_DIR":self.root/'user'}.items():
            self.stack.enter_context(patch.object(settings,name,value))

    def file(self,name,data=b'weights'):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path

    def test_existing_models_and_toolkit_share_only_identical_bytes_and_keep_paths(self):
        a=self.file('models/repo/encoder.safetensors')
        b=self.file('tools_repo/qwen-layered/model/text_encoder/model.safetensors')
        c=self.file('tools_repo/other/model/encoder.safetensors',b'newvers')
        result=model_store.consolidate(log=lambda _:None)
        self.assertEqual(result['linked_files'],1)
        self.assertEqual(result['saved_bytes'],len(b'weights'))
        self.assertTrue(a.samefile(b));self.assertFalse(a.samefile(c))
        a.unlink();self.assertEqual(b.read_bytes(),b'weights')
        self.assertEqual(model_store.consolidate(log=lambda _:None)['saved_bytes'],0)

    def test_hard_link_failure_keeps_unique_contents_and_does_not_claim_savings(self):
        a=self.file('models/a.gguf');b=self.file('models/b.gguf')
        with patch.object(model_store.os,'link',side_effect=OSError('different volumes')):
            result=model_store.consolidate(log=lambda _:None)
        self.assertEqual(result['saved_bytes'],0)
        self.assertEqual(result['separate_files'],1)
        self.assertFalse(a.samefile(b))
        self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_an_already_linked_inode_is_hashed_once_and_not_counted_as_reclaimed_space(self):
        a=self.file('models/a.gguf');b=self.root/'models/b.gguf';os.link(a,b)
        with patch.object(model_store,'digest',wraps=model_store.digest) as hashed:
            result=model_store.consolidate(log=lambda _:None)
        self.assertEqual(hashed.call_count,1)
        self.assertEqual(result['saved_bytes'],0)

    def test_runtime_binaries_and_source_are_excluded_and_stale_records_are_rejected(self):
        self.file('tools_repo/sam2/.venv/lib/runtime.bin')
        self.file('tools_repo/sam2/source/test.bin')
        a=self.file('models/a.gguf');sha=model_store.digest(a)
        model_store.remember(a)
        a.write_bytes(b'changed')
        self.assertFalse(model_store.reuse(sha,len(b'weights'),self.root/'models/b.gguf'))
        self.assertEqual(list(model_store.weight_files()),[a])

    def test_cancel_before_hashing_does_not_modify_files(self):
        a=self.file('models/a.gguf');b=self.file('models/b.gguf')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            model_store.consolidate(log=lambda _:None,check=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))
        self.assertFalse(a.samefile(b))

    def test_replacing_linked_weight_is_atomic_and_does_not_modify_the_other_model(self):
        a=self.file('models/a.gguf');b=self.file('models/b.gguf')
        model_store.consolidate(log=lambda _:None)
        replacement=self.file('models/new.gguf',b'replacement')
        os.replace(replacement,a)
        self.assertEqual(b.read_bytes(),b'weights')
        self.assertEqual(a.read_bytes(),b'replacement')

    def test_unique_storage_totals_and_delete_report_do_not_double_count_shared_files(self):
        a=self.file('models/a.gguf');b=self.root/'models/b.gguf';os.link(a,b)
        items=[inventory.Item('a','A','Models',[a]),inventory.Item('b','B','Models',[b])]
        with patch.object(inventory,'items',return_value=items),patch.object(settings,'ensure_dirs'):
            self.assertEqual(inventory.total_size(),len(b'weights'))
            _,freed=inventory.delete(['a'])
            self.assertEqual(freed,0);self.assertTrue(b.exists())
            _,freed=inventory.delete(['b'])
            self.assertEqual(freed,len(b'weights'))


class CatalogTests(unittest.TestCase):
    def test_all_addons_and_toolkit_models_have_one_catalog_entry(self):
        packages=model_catalog.packages();ids=[p.id for p in packages]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue({'tool:depth','tool:bg','tool:sam','tool:describe','tool:upscalers','tool:modern','seed:classic','seed:convrot',
                         'video:ltx','3d:trellis','3d:pixal'}.issubset(ids))
        self.assertTrue({'addon:'+name for name in addons.ADDON_NAMES
                         if not name.startswith('seedvr2') and name not in {'image-splats','svg','hivg','omnisvg','qwen-layered'}}.issubset(ids))
        self.assertTrue({'splat:triposplat', 'splat:idol'}.issubset(ids))
        self.assertNotIn('Layers & SVG', model_catalog.CATEGORIES)
        self.assertEqual([p.id for p in packages if p.id == 'addon:sam2'], ['addon:sam2'])

    def test_tool_install_routes_through_existing_script_and_generation_gpu(self):
        with patch.object(model_catalog.JOB,'run') as run,patch.object(model_catalog,'ready',return_value=True), \
             patch.object(settings,'generation_gpu_index',return_value=1):
            model_catalog.install('tool:depth',log=lambda _:None)
        command=run.call_args.args[0]
        self.assertEqual(command[-1],'depth')
        self.assertTrue(str(command[-2]).endswith('setup_tools.py'))
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'1')
        with patch.object(model_catalog.JOB,'run') as run,self.assertRaises(ValueError):
            model_catalog.install('addon:omnisvg','arbitrary')
        run.assert_not_called()

    def test_seed_downloads_selected_weights_and_reuses_one_vae_location(self):
        for package,variant in (('seed:classic','3b'),('seed:convrot','3b-w4a8')):
            with patch.object(model_catalog.JOB,'run'),patch.object(model_catalog,'ready',return_value=True), \
                 patch.object(transfers,'hf_hub_download') as download:
                model_catalog.install(package,variant,log=lambda _:None)
            self.assertEqual(download.call_count,2)
            self.assertEqual(download.call_args.kwargs['filename'],'ema_vae_fp16.safetensors')
            self.assertEqual(download.call_args.kwargs['local_dir'],str(addons.root('seedvr2')/'models'))

    def test_transformers_snapshot_chooses_one_checkpoint_format_and_pins_revision(self):
        from types import SimpleNamespace
        names=['model.safetensors','pytorch_model.bin','onnx/model.onnx','tf_model.h5','model.pth',
               'config.json','briarmbg.py','tokenizer.json']
        info=SimpleNamespace(sha='fixed',siblings=[SimpleNamespace(rfilename=n) for n in names])
        with patch('huggingface_hub.constants.HF_HUB_OFFLINE', False), \
             patch('huggingface_hub.HfApi') as api,patch.object(transfers,'snapshot_download') as snapshot:
            api.return_value.model_info.return_value=info
            transfers.model_snapshot_download(repo_id='test/model',local_dir='unused')
        chosen=snapshot.call_args.kwargs
        self.assertEqual(chosen['revision'],'fixed')
        self.assertIn('briarmbg.py',chosen['allow_patterns'])
        self.assertIn('model.safetensors',chosen['allow_patterns'])
        self.assertNotIn('pytorch_model.bin',chosen['allow_patterns'])
        self.assertNotIn('model.pth',chosen['allow_patterns'])


if __name__ == '__main__':
    unittest.main()
