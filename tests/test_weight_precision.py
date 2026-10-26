"""Precision suggestions must respect device capability, files and memory."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from atelier import downloader, hardware, registry, settings, weight_precision as precision


def gpu_profile(arch='ampere', vram=12, cc='8.6'):
    return SimpleNamespace(gpu=hardware.Gpu(1, 'Test GPU', vram, arch, arch != 'pascal', cc),
                           ram_gb=64)


class PrecisionTests(unittest.TestCase):
    def test_format_and_storage_are_separate_and_unlabelled_files_stay_unknown(self):
        for filename, expected in [('model-Q8_0.gguf', 'Q8_0'), ('model-F16.gguf', 'FP16'),
                                   ('model-BF16.gguf', 'BF16'), ('model-F32.gguf', 'FP32'),
                                   ('model_int8_convrot.safetensors', 'INT8'),
                                   ('model_w4a8.safetensors', 'INT4 / W4A8'),
                                   ('model.safetensors', 'unspecified')]:
            self.assertEqual(precision.storage_type(filename), expected)
        self.assertEqual(precision.storage_type('model.safetensors', 'BF16'), 'BF16')
        self.assertEqual(precision.storage_type('bf16-folder/model.safetensors'), 'unspecified')

    def test_reported_capability_takes_priority_and_unknown_is_not_supported(self):
        for arch, cc, expected in [('ampere', '7.5', False), ('turing', '8.6', True),
                                   ('turing', '', False), ('ampere', '', True),
                                   ('unknown', '', None), ('unknown', 'bad', None)]:
            self.assertIs(precision.native_bf16(gpu_profile(arch, cc=cc).gpu), expected)
        self.assertIsNone(precision.native_bf16(None))
        self.assertIn('No native BF16', precision.gpu_advice('BF16', gpu_profile('turing', cc='7.5')))
        self.assertIn('CPU', precision.gpu_advice('FP16', SimpleNamespace(gpu=None)))

    def test_memory_suggestion_changes_with_vram_without_recommending_native_bf16_on_turing(self):
        comp = registry.Component('diffusion', 'test/repo', 'model_int8.safetensors', None)
        candidates = [('model_int8.safetensors', 5_400_000_000, ''),
                      ('model_bf16.safetensors', 10_200_000_000, '')]
        self.assertEqual(precision.suggest(comp, candidates, gpu_profile()), candidates[0][0])
        self.assertEqual(precision.suggest(comp, candidates, gpu_profile(vram=24)), candidates[1][0])
        self.assertEqual(precision.suggest(comp, candidates, gpu_profile('turing', 24, '7.5')), candidates[0][0])
        self.assertIsNone(precision.suggest(comp, [(name, None, '') for name, _, _ in candidates], gpu_profile()))
        self.assertIsNone(precision.suggest(comp, candidates, SimpleNamespace(gpu=None)))
        self.assertIsNone(precision.suggest(comp, candidates, gpu_profile(vram=4)))

    def test_weight_size_does_not_claim_total_vram_or_integer_compute(self):
        note = precision.describe('model_bf16.safetensors', gpu_profile(), size=16_000_000_000)
        self.assertIn('offload needed', note)
        self.assertIn('not guaranteed', precision.gpu_advice('INT8', gpu_profile()))
        for dtype in ('FP16', 'FP32'):
            self.assertIn(dtype, precision.model_advice(registry.get_base_model('moss-image-sft', {})))

    def test_encoder_keeps_ram_aware_target_and_explicit_gpu_is_used(self):
        comp = registry.Component('text_encoder', 'test/repo', 'encoder-{enc_quant}.gguf', 'Q4_K_M')
        rows = [('encoder-Q4_K_M.gguf', 5_000_000_000, ''), ('encoder-F16.gguf', 18_000_000_000, '')]
        self.assertEqual(precision.suggest(comp, rows, gpu_profile()), rows[0][0])
        with patch.object(settings, 'generation_gpu_index', return_value=1), \
             patch.object(hardware, 'auto_profile', return_value=gpu_profile()) as auto:
            precision.profile({'gpu_index': 1})
        auto.assert_called_once_with(1)


class CrossFormatTests(unittest.TestCase):
    def model(self, selected=None):
        prefs = {'model_files': {'krea2-turbo': {'diffusion': selected}}} if selected else {}
        return registry.get_base_model('krea2-turbo', prefs)

    def test_scanning_selected_safetensors_keeps_original_ggufs_and_excludes_other_models(self):
        selected = 'diffusion_models/krea2_turbo_bf16.safetensors'
        model = self.model(selected)
        self.assertIsNotNone(model)
        comp = next(c for c in model.components if c.role == 'diffusion')
        self.assertEqual(comp.repo, 'Comfy-Org/Krea-2')
        self.assertEqual(comp.requested(), selected)
        calls = []
        def info(repo_id, files_metadata):
            calls.append(repo_id)
            files = {'vantagewithai/Krea-2-Turbo-GGUF': [('krea2_turbo-Q5_K_M.gguf', 7_000_000_000)],
                     'Comfy-Org/Krea-2': [(selected, 20_000_000_000),
                         ('diffusion_models/krea2_turbo_int8_convrot.safetensors', 10_000_000_000),
                         ('diffusion_models/krea2_raw_bf16.safetensors', 20_000_000_000)],
                     'Qwen/Qwen3-VL-4B-Instruct-GGUF': [('Qwen3VL-4B-Instruct-Q4_K_M.gguf', 3_000_000_000)]}
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=s, lfs=None)
                                              for n, s in files.get(repo_id, [])])
        with patch('huggingface_hub.HfApi') as api, patch.object(precision, 'profile', return_value=gpu_profile()):
            api.return_value.model_info.side_effect = info
            choices = downloader.weight_choices(model)['diffusion']
        values = [value for _, value in choices]
        self.assertIn(selected, values)
        self.assertIn('krea2_turbo-Q5_K_M.gguf', values)
        self.assertEqual(len(values), 3)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertIn('offload needed', next(label for label, value in choices if value == selected))

    def test_non_gguf_selection_reaches_model_resolution_and_can_return_to_gguf(self):
        selected = 'diffusion_models/krea2_turbo_bf16.safetensors'
        from atelier.engine.generate import resolve_model_files
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'MODELS_DIR', Path(tmp)):
            model = self.model(selected)
            for comp in model.components:
                path = settings.model_repo_dir(comp.repo) / comp.requested()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            self.assertEqual(resolve_model_files(model)['diffusion'],
                             settings.model_repo_dir('Comfy-Org/Krea-2') / selected)
            gguf = next(c for c in self.model().components if c.role == 'diffusion')
            self.assertEqual(gguf.repo, 'vantagewithai/Krea-2-Turbo-GGUF')
            self.assertIsNotNone(gguf.token)

    def test_initial_menus_offer_non_gguf_without_network_and_preserve_saved_choices(self):
        from atelier.ui.library_tab import _weight_menu
        selected = 'diffusion_models/krea2_turbo_bf16.safetensors'
        with patch.object(precision, 'profile', return_value=gpu_profile()):
            menu, value = _weight_menu(self.model(selected), 'diffusion')
            self.assertEqual(value, selected)
            self.assertIn('INT8 ConvRot', str(menu))
            self.assertIn(selected, [v for _, v in menu])
            menu, value = _weight_menu(self.model(selected), 'diffusion', [('GGUF', 'krea2_turbo-Q4_K_M.gguf')])
            self.assertEqual(value, selected)
            self.assertIn(selected, [v for _, v in menu])
        for model_id in ('moss-image-sft', 'fibo2-dev', 'qwen-image-2.1-turbo'):
            menu, _ = _weight_menu(registry.get_base_model(model_id, {}), 'diffusion')
            self.assertTrue(any('bf16.safetensors' in value for _, value in menu))


if __name__ == '__main__':
    unittest.main()
