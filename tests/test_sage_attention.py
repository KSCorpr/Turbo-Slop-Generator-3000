"""SageAttention eligibility, real request routing, persistence and GPU isolation."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from atelier import hardware, registry, sage_attention as sage, settings
from atelier.engine import generate, sdcpp
from atelier.ui import settings_tab

RTX = hardware.Gpu(2, 'RTX 3060', 12, 'ampere', True, '8.6')
PASCAL = hardware.Gpu(0, 'GTX 1080 Ti', 11, 'pascal', False, '6.1')
TURING = hardware.Gpu(1, 'RTX 2080 Ti', 11, 'turing', True, '7.5')
OPTIONS = {'--sage-attn', '--sigmas', '--tokenizer', '--params-backend', '--max-vram',
           '--auto-fit', '--video-frames', '--llm', '--diffusion-fa', '--vae-tiling'}


class SageAttentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key in ('MODELS_DIR', 'USERDATA_DIR', 'TMP_DIR', 'BIN_DIR', 'CUSTOM_DIR', 'LORA_DIR', 'OUTPUT_DIR'):
            path = self.root / key
            path.mkdir()
            self.stack.enter_context(patch.object(settings, key, path))
        self.stack.enter_context(patch.object(settings, 'PREFS_FILE', settings.USERDATA_DIR / 'prefs.json'))
        self.stack.enter_context(patch.object(hardware, 'detect_gpus', return_value=(PASCAL, TURING, RTX)))
        self.stack.enter_context(patch.object(hardware, 'detect_ram_gb', return_value=64))
        self.stack.enter_context(patch.object(hardware, 'free_vram_gb', return_value=10.5))
        self.stack.enter_context(patch.object(sdcpp, 'supported_options', return_value=OPTIONS))
        self.stack.enter_context(patch.object(sdcpp, 'binary_help', return_value='--auto-fit on|off --sigmas minimax-h3-image'))
        self.cli = settings.BIN_DIR / 'sd-cli'
        self.cli.touch()
        self.stack.enter_context(patch.object(settings, 'find_sd_cli', return_value=self.cli))
        self.manifest = settings.BIN_DIR / 'engine-manifest.json'
        self.manifest.write_text('{"tag":"master-1088-32d516c","archive":"sd-bin-linux-cuda12-x64.zip"}')
        settings.save_prefs({'system_mode': 'auto', 'auto_gpu_index': RTX.index,
                             'gpu_index': RTX.index, 'sage_attention': True})

    def test_reported_compute_capability_overrides_architecture_and_unknowns_are_conservative(self):
        for gpu in (RTX, replace(RTX, arch='unknown'), replace(RTX, compute_cap='')):
            self.assertTrue(sage.gpu_supported(gpu))
        for gpu in (PASCAL, TURING, None, replace(RTX, compute_cap='7.5'),
                    replace(RTX, arch='unknown', compute_cap='invalid'),
                    replace(RTX, arch='apple')):
            self.assertFalse(sage.gpu_supported(gpu))

    def test_cli_support_is_not_enough_for_old_cards_and_mixed_autofit(self):
        self.assertTrue(sage.availability(self.cli, RTX.index)[0])
        for gpu in (PASCAL, TURING):
            self.assertFalse(sage.availability(self.cli, gpu.index)[0])
        self.assertFalse(sage.availability(self.cli, RTX.index, all_gpus=True)[0])
        with patch.object(hardware, 'detect_gpus', return_value=(RTX,)):
            self.assertTrue(sage.availability(self.cli, RTX.index, all_gpus=True)[0])
        with patch.object(sdcpp, 'supported_options', return_value=set()):
            self.assertIn('update.bat', sage.availability(self.cli, RTX.index)[1])
        self.manifest.write_text('{"archive":"sd-bin-win-vulkan-x64.zip"}')
        self.assertFalse(sage.availability(self.cli, RTX.index)[0])

    def test_opt_in_and_low_level_commands_cannot_enable_unsupported_gpu(self):
        self.assertFalse(settings.DEFAULT_PREFS['sage_attention'])
        log = []
        self.assertFalse(sage.requested(self.cli, TURING.index, log=log.append))
        self.assertIn('inactive', log[0])
        with patch.object(sage, 'availability', side_effect=AssertionError('disabled must not probe')):
            self.assertFalse(sage.requested(self.cli, RTX.index, {'sage_attention': False}))
        for gpu in (RTX, TURING, PASCAL):
            req = sdcpp.GenRequest(sage_attn=True, gpu_index=gpu.index, flags={'diffusion_fa': True})
            args = sdcpp.memory_args(self.cli, req)
            self.assertEqual('--sage-attn' in args, gpu == RTX)
            self.assertIn('--diffusion-fa', args)

    def test_every_image_model_passes_sage_in_auto_and_manual_modes_and_records_metadata(self):
        reference = self.root / 'reference.png'
        Image.new('RGB', (32, 32)).save(reference)
        output = settings.OUTPUT_DIR / 'out.png'
        Image.new('RGB', (32, 32)).save(output)
        for model in registry.load_base_models(settings.load_prefs()):
            for comp in model.components:
                path = settings.model_repo_dir(comp.repo) / comp.requested().replace('*', 'fixture')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            for mode in ('auto', 'manual'):
                prefs = settings.load_prefs()
                prefs.update(system_mode=mode, auto_optimize=True,
                             gpu_index=None if mode == 'manual' else RTX.index)
                settings.save_prefs(prefs)
                with self.subTest(model=model.id, mode=mode), \
                     patch.object(sdcpp, 'run') as run, \
                     patch.object(sdcpp, 'collect_outputs', return_value=[output]):
                    generate.generate(model.id, 'A poster', '', model.defaults['steps'], model.defaults['cfg_scale'],
                                      1024, 1024, 42, 1,
                                      ref_image=[reference] if model.defaults.get('requires_reference') else None)
                    self.assertIn('--sage-attn', run.call_args.args[0])
                    self.assertEqual(run.call_args.kwargs['gpu_index'], RTX.index)
                    self.assertTrue(json.loads(output.with_suffix('.json').read_text())['sage_attention'])
                    self.assertIn('SageAttention: enabled', output.with_suffix('.txt').read_text())

    def test_ui_toggle_preserves_auto_mode_and_refreshes_without_overwriting_saved_opt_in(self):
        with gr.Blocks() as demo:
            settings_tab.build_settings_tab()
        toggle = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', '') == 'sage-attention')
        self.assertTrue(toggle.interactive)
        self.assertTrue(toggle.value)
        apply = next(f.fn for f in demo.fns.values() if f.name == 'apply_sage_attention')
        refresh = next(f.fn for f in demo.fns.values() if f.name == 'refresh_sage_support')
        apply(False)
        self.assertEqual(settings.load_prefs()['system_mode'], 'auto')
        apply(True)
        prefs = settings.load_prefs()
        prefs['auto_gpu_index'] = TURING.index
        settings.save_prefs(prefs)
        state = refresh()
        self.assertFalse(state['interactive'])
        self.assertFalse(state['value'])
        self.assertTrue(settings.load_prefs()['sage_attention'])
        with self.assertRaises(gr.Error):
            apply(True)
        self.assertTrue(any(not f.queue and f.name == 'apply_sage_attention' for f in demo.fns.values()))
