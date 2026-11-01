"""Real Settings browser journey with simulated GPU/CLI capabilities, no models."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gradio as gr
from playwright.sync_api import sync_playwright, expect
import app
from atelier import hardware, net, settings
from atelier.engine import sdcpp
from atelier.ui.settings_tab import build_settings_tab
from tests.test_sage_attention import RTX, TURING, PASCAL, OPTIONS


def main():
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        for key in ('MODELS_DIR', 'USERDATA_DIR', 'TMP_DIR', 'BIN_DIR', 'CUSTOM_DIR', 'LORA_DIR', 'OUTPUT_DIR'):
            path = root / key
            path.mkdir()
            stack.enter_context(patch.object(settings, key, path))
        stack.enter_context(patch.object(settings, 'PREFS_FILE', settings.USERDATA_DIR / 'prefs.json'))
        stack.enter_context(patch.object(hardware, 'detect_gpus', return_value=(PASCAL, TURING, RTX)))
        stack.enter_context(patch.object(hardware, 'detect_ram_gb', return_value=64))
        stack.enter_context(patch.object(sdcpp, 'supported_options', return_value=OPTIONS))
        cli = settings.BIN_DIR / 'sd-cli'
        cli.touch()
        stack.enter_context(patch.object(settings, 'find_sd_cli', return_value=cli))
        settings.save_prefs({'system_mode': 'auto', 'auto_gpu_index': RTX.index})
        with gr.Blocks() as demo:
            with gr.Tabs():
                build_settings_tab()
        demo.queue()
        port = net.find_free_port(7862)
        demo.launch(server_name='127.0.0.1', server_port=port, inbrowser=False,
                    prevent_thread_lock=True, quiet=True, ssr_mode=False, **app.presentation())
        try:
            with sync_playwright() as browser:
                chrome = browser.chromium.launch(headless=True)
                page = chrome.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}/?__theme=light', wait_until='domcontentloaded')
                page.get_by_text("SageAttention (CUDA) — optional", exact=True).click()
                toggle = page.locator('#sage-attention input[type=checkbox]')
                expect(toggle).to_be_enabled()
                toggle.check()
                expect(page.locator('#sage-attention-status')).to_contain_text('enabled')
                assert settings.load_prefs()['sage_attention']
                assert settings.load_prefs()['system_mode'] == 'auto'
                picker = page.get_by_label('Card used for generating', exact=True)
                picker.click()
                page.get_by_role('option', name='#1', exact=False).click()
                expect(toggle).to_be_disabled()
                expect(toggle).not_to_be_checked()
                expect(page.locator('#sage-attention')).to_contain_text('RTX 2080 Ti')
                assert settings.load_prefs()['sage_attention']
                picker.click()
                page.get_by_role('option', name='#2', exact=False).click()
                expect(toggle).to_be_enabled()
                expect(toggle).to_be_checked()
                assert settings.load_prefs()['system_mode'] == 'auto'
                page.screenshot(path='/tmp/sage-browser-supported.png')
                toggle.uncheck()
                expect(page.locator('#sage-attention-status')).to_contain_text('disabled')
                assert not settings.load_prefs()['sage_attention']
                page.reload(wait_until='domcontentloaded')
                page.get_by_text("SageAttention (CUDA) — optional", exact=True).click()
                expect(toggle).not_to_be_checked()
                assert not errors, errors
                chrome.close()
            print(json.dumps({'page_errors': errors, 'toggle_persisted': True,
                              'auto_mode_preserved': True, 'unsupported_gpu_disabled': True}))
        finally:
            demo.close()


if __name__ == '__main__':
    main()
