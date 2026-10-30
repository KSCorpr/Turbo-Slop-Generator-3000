"""Image-to-splats: pinned installs, offline inference, real PLY validation and UI routing."""
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np
from PIL import Image

from atelier import addons, hardware, image_splats as models, inventory, model_catalog, settings
from atelier.engine import image_splats as engine
from atelier.engine.local_jobs import LocalJob
from atelier.ui import image_splats_tab, widgets
from scripts import setup_image_splats
from scripts.tools.run_image_splats import encoder_precision


def write_ply(path, count=32768, nonfinite=False):
    fields = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    header = ('ply\nformat binary_little_endian 1.0\n' + f'element vertex {count}\n' +
              ''.join('property float ' + f + '\n' for f in fields) + 'end_header\n')
    rows = np.zeros((count, len(fields)), dtype='<f4')
    # A realistic, bounded fixture. Thousands of unit-radius splats at the
    # same pixel cause extreme overdraw in a software WebGL browser.
    rows[:, :3] = np.random.default_rng(42).uniform(-0.4, 0.4, (count, 3))
    rows[:, 10:13] = np.log(0.002)
    rows[:, 13] = 1
    if nonfinite:
        rows[0, 0] = np.inf
    Path(path).write_bytes(header.encode() + rows.tobytes())


class ManifestTests(unittest.TestCase):
    def test_only_required_files_are_pinned_and_common_vae_is_identical(self):
        self.assertEqual(len(models.TRIPO), 5)
        self.assertEqual(len(models.IDOL), 9)
        self.assertFalse(any('vae_encoder' in s.filename for s in models.TRIPO))
        for spec in models.TRIPO + models.IDOL:
            self.assertEqual(len(spec.revision), 40)
            if spec.filename.endswith('.safetensors'):
                self.assertEqual(len(spec.sha256), 64)
        self.assertEqual(next(s.sha256 for s in models.TRIPO if s.key == 'flux'),
                         next(s.sha256 for s in models.IDOL if s.key == 'flux'))
        self.assertEqual(encoder_precision((6, 1)), 'fp32')
        self.assertEqual(encoder_precision((7, 5)), 'fp32')
        self.assertEqual(encoder_precision((8, 6)), 'bf16')

    def test_both_catalog_entries_route_to_selected_model_and_generation_gpu(self):
        for backend in models.BACKENDS:
            key = 'splat:' + backend
            self.assertIn(key, [v for _, v in model_catalog.choices('Video & 3D', models.BACKENDS[backend])])
            with patch.object(model_catalog.JOB, 'run') as run, patch.object(model_catalog, 'ready', return_value=True), \
                 patch.object(settings, 'generation_gpu_index', return_value=1):
                model_catalog.install(key, log=lambda _: None)
            self.assertEqual(run.call_args.args[0][-1], backend)
            self.assertTrue(str(run.call_args.args[0][-2]).endswith('setup_image_splats.py'))
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '1')

    def test_gated_access_denial_precedes_large_downloads(self):
        from huggingface_hub.errors import GatedRepoError
        with patch('huggingface_hub.get_hf_file_metadata', side_effect=GatedRepoError('denied')), \
             patch.object(models.transfers, 'hf_hub_download') as download:
            with self.assertRaisesRegex(RuntimeError, 'Accept the terms'):
                models.download_models('idol', log=lambda _: None)
            download.assert_not_called()

    def test_auto_gpu_and_cuda_wheels_follow_selected_hardware(self):
        cards = (hardware.Gpu(0,'GTX 1080 Ti',11,'pascal',False),
                 hardware.Gpu(1,'RTX 2080 Ti',11,'turing',True))
        with patch.object(settings, 'generation_gpu_index', return_value=None), \
             patch.object(hardware, 'detect_gpus', return_value=cards):
            self.assertEqual(models.selected_gpu_index(), 1)
            self.assertIn('+cu124', setup_image_splats.torch_packages()[0])
        with patch.object(settings, 'generation_gpu_index', return_value=None), \
             patch.object(hardware, 'detect_gpus', return_value=(hardware.Gpu(0,'RTX 5090',32,'blackwell',True),)):
            self.assertIn('+cu128', setup_image_splats.torch_packages()[0])


class IsolatedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        for key in ('ROOT','MODELS_DIR','OUTPUT_DIR','TMP_DIR','USERDATA_DIR'):
            p = self.root / key; p.mkdir()
            self.stack.enter_context(patch.object(settings, key, p))
        self.stack.enter_context(patch.object(settings, 'PREFS_FILE', settings.USERDATA_DIR / 'preferences.json'))
        self.stack.enter_context(patch.object(settings, 'generation_gpu_index', return_value=1))
        self.stack.enter_context(patch.object(engine, 'JOB', LocalJob()))
        self.stack.enter_context(patch.object(engine.trellis, 'resident_is_running', return_value=False))
        self.data = {}
        for name in ('TRIPO', 'IDOL'):
            specs = []
            for spec in getattr(models, name):
                payload = (spec.repo + '/' + spec.filename).encode()
                self.data[(spec.repo, spec.filename)] = payload
                specs.append(replace(spec, size=len(payload), sha256=sha256(payload).hexdigest() if spec.sha256 else ''))
            self.stack.enter_context(patch.object(models, name, tuple(specs)))

    def installed(self, backend):
        py = addons.python('image-splats'); py.parent.mkdir(parents=True, exist_ok=True); py.touch()
        src = models.source(backend) / ('triposplat.py' if backend == 'triposplat' else 'idol/pipeline.py')
        src.parent.mkdir(parents=True, exist_ok=True); src.touch()
        for spec in models.artifacts(backend):
            spec.path().parent.mkdir(parents=True, exist_ok=True)
            spec.path().write_bytes(self.data[(spec.repo, spec.filename)])
        models.marker(backend).write_text(json.dumps(models.identity(backend)))

    def test_ready_requires_exact_source_marker_all_weights_and_hash_integrity(self):
        for backend in models.BACKENDS:
            self.assertFalse(models.ready(backend)); self.installed(backend)
            self.assertTrue(models.ready(backend)); models.validate(backend)
            spec = next(s for s in models.artifacts(backend) if s.sha256)
            spec.path().write_bytes(b'x' * spec.size)
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                models.validate(backend)
            spec.path().unlink(); self.assertFalse(models.ready(backend))

    def test_downloads_retain_the_shared_store_and_never_download_unused_encoders(self):
        seen = []
        def transfer(**kwargs):
            seen.append(kwargs)
            spec = next(s for s in models.TRIPO if s.filename == kwargs['filename'])
            spec.path().parent.mkdir(parents=True, exist_ok=True)
            spec.path().write_bytes(self.data[(spec.repo, spec.filename)])
            return str(spec.path())
        with patch.object(models.transfers, 'hf_hub_download', side_effect=transfer):
            models.download_models('triposplat', log=lambda _: None)
        self.assertEqual(len(seen), 5)
        for kwargs in seen:
            self.assertEqual(kwargs['revision'], models.TRIPO_REV)
            self.assertEqual(kwargs['local_dir'], str(settings.model_repo_dir('VAST-AI/TripoSplat')))

    def test_native_requests_snapshot_rgba_unicode_input_and_select_offline_gpu_paths(self):
        source = self.root / 'objet_é_日本.png'
        Image.new('RGBA', (37, 53), (17,92,154,80)).save(source)
        for backend in models.BACKENDS:
            self.installed(backend)
            captured = []
            def run(command, log, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                captured.append((request, kwargs['env']))
                with Image.open(request['source']) as image:
                    self.assertEqual(image.mode, 'RGBA'); self.assertEqual(image.getpixel((0,0)), (17,92,154,80))
                    image.save(Path(request['output']) / 'input.png')
                write_ply(Path(request['output']) / 'object.ply')
            with patch.object(engine.JOB, 'run', side_effect=run):
                result = engine.generate(source, backend, 32768, 20, 3, 3, 'real', -1, log=lambda _: None)
            request, env = captured[0]
            self.assertTrue(result.is_file())
            self.assertEqual(request['backend'], backend); self.assertEqual(request['count'], 32768)
            self.assertEqual(request['paths'], {s.key: str(s.path()) for s in models.artifacts(backend)})
            self.assertGreaterEqual(request['seed'], 0)
            self.assertEqual(env['HF_HUB_OFFLINE'], '1'); self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1'); self.assertEqual(env['ATTN_BACKEND'], 'sdpa')
            self.assertEqual(request['code_commit'], models.SOURCES[backend][1])

    def test_invalid_controls_and_missing_input_stop_before_any_process(self):
        for args in [('triposplat', 1048576,20,3,3,'real',42), ('idol',12345,20,3,3,'real',42),
                     ('idol',32768,0,3,3,'real',42), ('idol',32768,20,float('nan'),3,'real',42)]:
            with self.assertRaises(ValueError): engine.validate_controls(*args)
        with patch.object(engine.JOB, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Import an image'):
                engine.generate(None)
            run.assert_not_called()

    def test_invalid_truncated_or_nonfinite_ply_is_never_published_as_success(self):
        path = self.root / 'object.ply'
        write_ply(path); engine.validate_ply(path, 32768)
        with self.assertRaisesRegex(RuntimeError, 'requested'):
            engine.validate_ply(path, 65536)
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            engine.validate_ply(path, 32768)
        write_ply(path, nonfinite=True)
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            engine.validate_ply(path, 32768)

    def test_cancel_reaches_the_active_process_and_next_job_starts_cleanly(self):
        job = engine.JOB
        with job.session():
            with patch.object(job, '_stop') as stop:
                job._proc = object(); job.cancel(); stop.assert_called_once()
                job._proc = None
                with self.assertRaisesRegex(RuntimeError, 'cancelled'): job.check()
        with job.session(): job.check()

    def test_install_probe_failure_never_writes_a_ready_marker_or_fetches_weights(self):
        with patch.object(setup_image_splats, 'environment', return_value=self.root / 'python'), \
             patch.object(setup_image_splats, 'pip'), patch.object(setup_image_splats, '_install_source'), \
             patch.object(setup_image_splats, 'run', side_effect=RuntimeError('GPU probe failed')), \
             patch.object(models, 'download_models') as download:
            with self.assertRaisesRegex(RuntimeError, 'GPU probe failed'):
                setup_image_splats.install('triposplat')
            download.assert_not_called(); self.assertFalse(models.marker('triposplat').exists())

    def test_ui_switch_limits_the_actual_model_controls_and_routes_preview_and_downloads(self):
        with gr.Blocks() as demo: image_splats_tab.build_image_splats_tab()
        fn = next(f for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'generate_image_splats')
        self.assertEqual(fn.concurrency_id, widgets.GPU_QUEUE['concurrency_id'])
        for backend, maximum, steps in [('triposplat',262144,20), ('idol',1048576,50)]:
            controls = image_splats_tab.backend_controls(backend)
            self.assertEqual(max(v for _,v in controls[0]['choices']), maximum)
            self.assertEqual(controls[1]['value'], steps)
        output = self.root / 'result'; output.mkdir()
        write_ply(output / 'object.ply'); Image.new('RGB',(32,32)).save(output / 'input.png')
        (output / 'request.json').write_text('{}')
        with patch.object(engine, 'generate', return_value=output / 'object.ply') as run:
            rows = list(fn.fn('source.png','idol',32768,50,3,3,'real',42))
        self.assertEqual(run.call_args.args[1], 'idol')
        self.assertEqual(rows[-1][0], 'Completed.')
        self.assertEqual(rows[-1][1], str(output / 'object.ply'))
        self.assertIn(str(output / 'request.json'), rows[-1][3])
        self.assertFalse(any(p.endswith('.splat') for p in rows[-1][3]))


if __name__ == '__main__': unittest.main()
