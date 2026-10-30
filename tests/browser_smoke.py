"""Exercise the real UI with a simulated engine; no models/GPU/downloads.

python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python tests/browser_smoke.py --screenshots tmp/ui-check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from PIL import Image, ImageDraw
from playwright.sync_api import expect, sync_playwright

import app
from atelier import registry, settings
from atelier.engine import generate as engine
from atelier.engine import seedvr2
from atelier.engine import image_splats as splat_engine
from atelier.ui import media_tabs
from tests.test_image_splats import write_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    expect.set_options(timeout=15000)
    calls = []
    seed_calls = []
    seed_installs = []
    splat_calls = []
    cancelled = threading.Event()
    interrupted = threading.Event()
    engine_started = threading.Event()
    active, peak = 0, 0
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        folder = Path(directory)
        screenshots = (args.screenshots or folder / "screenshots").resolve()
        screenshots.mkdir(parents=True, exist_ok=True)
        for name in ("OUTPUT_DIR", "TMP_DIR", "USERDATA_DIR", "MODELS_DIR", "CUSTOM_DIR", "LORA_DIR", "BIN_DIR"):
            path = folder / name.lower()
            path.mkdir()
            stack.enter_context(patch.object(settings, name, path))
        stack.enter_context(patch.object(settings, "PREFS_FILE", settings.USERDATA_DIR / "preferences.json"))

        sample = Image.new("RGBA", (1024, 768), (234, 231, 222, 255))
        draw = ImageDraw.Draw(sample)
        draw.rectangle((110, 110, 560, 650), fill="#253b3e")
        draw.ellipse((365, 180, 880, 690), fill="#be5633")
        for i in range(30):
            path = settings.OUTPUT_DIR / f"fixture-{i:02d}.png"
            sample.save(path)
            path.with_suffix(".txt").write_text(
                "A studio browser test\nModel: Flux (flux2-klein-9b)\nSteps: 4, Seed: 42", encoding="utf-8")

        def generate(**kwargs):
            nonlocal active, peak
            cancelled.clear()
            active += 1
            peak = max(peak, active)
            calls.append(kwargs["prompt"])
            engine_started.set()
            try:
                if "Cancel request" in kwargs["prompt"] and kwargs.get("preview_path"):
                    # Keep this simulated inference alive until Stop. A 750 ms
                    # normal fixture can finish before a slow CI browser clicks.
                    if not cancelled.wait(15):
                        raise RuntimeError("The browser did not request cancellation.")
                    interrupted.set()
                    raise RuntimeError("Interrupted by the user.")
                for step in range(3):
                    if cancelled.wait(0.25):
                        interrupted.set()
                        raise RuntimeError("Interrupted by the user.")
                    if kwargs.get("preview_path"):
                        sample.save(str(kwargs["preview_path"]).replace("%03d", f"{step:03d}"))
                    kwargs["log"](f"{step + 1}/{kwargs['steps']}")
                path = settings.OUTPUT_DIR / f"simulated-{len(calls)}.png"
                sample.resize((kwargs["width"], kwargs["height"])).save(path)
                path.with_suffix(".txt").write_text(
                    kwargs["prompt"] + "\nModel: Flux (flux2-klein-9b)\nSteps: 4, Seed: 42", encoding="utf-8")
                return [path]
            finally:
                active -= 1

        def cancel():
            cancelled.set()
            return "Generation cancelled."

        def restore_seed(source, model, resolution, seed, log, *, scale=None, detail_strength=100, attention="auto"):
            if not source:
                raise ValueError("Import an image or a video first.")
            seed_calls.append({"source": str(source), "model": model, "scale": scale, "details": detail_strength,
                               "attention": attention})
            path = settings.OUTPUT_DIR / "seedvr2-simulated.png"
            with Image.open(source) as im:
                target = seedvr2.output_resolution(Path(source), resolution, scale)
                ratio = target / min(im.size)
                im.convert("RGB").resize(tuple(int(edge * ratio) for edge in im.size)).save(path)
            log("Simulated SeedVR2 restoration completed.")
            return path

        stack.enter_context(patch.object(engine, "generate", generate))
        stack.enter_context(patch.object(engine, "cancel", cancel))
        def generate_splats(source, backend, count, steps, cfg, shift, mode, seed, log):
            if not source:
                raise ValueError("Import an image first.")
            splat_calls.append((backend, count, steps, mode))
            folder = settings.OUTPUT_DIR / ("splat-simulated-" + str(len(splat_calls)))
            folder.mkdir()
            write_ply(folder / "object.ply", int(count))
            sample.save(folder / "input.png")
            (folder / "request.json").write_text(json.dumps({"backend":backend}))
            if backend == "triposplat":
                (folder / "object.splat").write_bytes(b"fixture" * 32)
            else:
                (folder / "camera.json").write_text('{}')
            return folder / "object.ply"
        stack.enter_context(patch.object(splat_engine, "generate", generate_splats))
        stack.enter_context(patch.object(seedvr2, "restore", restore_seed))
        def install_seed(name, job, log, quant=None, *, optimizations=False, precision=None):
            seed_installs.append((name, optimizations, precision))
            log("Simulated optimization installation completed.")

        stack.enter_context(patch.object(media_tabs, "install_action", install_seed))
        start = time.perf_counter()
        demo = app.build_app().queue(max_size=16)
        build_seconds = time.perf_counter() - start
        port = app.net.find_free_port(7861)
        demo.launch(server_name=app.HOST, server_port=port, inbrowser=False,
                    prevent_thread_lock=True, ssr_mode=False, quiet=True,
                    allowed_paths=settings.served_paths(), **app.presentation())
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                def record_error(error):
                    errors.append(error.stack)
                    print(error.stack, flush=True)
                page.on("pageerror", record_error)
                started = time.perf_counter()
                page.goto(f"http://127.0.0.1:{port}/?__theme=light", wait_until="domcontentloaded")
                expect(page.get_by_role("tab", name="Images", exact=True)).to_be_visible()
                prompt = page.locator(".studio-prompt textarea:visible")
                expect(prompt).to_be_visible()
                ready_seconds = time.perf_counter() - started
                # Exercise every experimental choice through the real dropdown:
                # the explanation must track the selected preset, including Qwen.
                model_tabs = {
                    "flux2-klein-9b": "Flux.2 Klein",
                    "qwen-image-2.1-turbo": "Qwen Image 2.1 Turbo", "moss-image-sft": "MOSS-Image SFT",
                    "fibo2-dev": "Fibo 2.0 Dev", "fibo2-edit-turbo": "Fibo 2.0 Edit Turbo",
                    "krea2-turbo": "Krea 2 Turbo", "krea2-realism": "Krea 2 Realism",
                }
                preset_checks = 0
                for model in registry.load_base_models({}):
                    page.get_by_role("tab", name=model_tabs[model.id], exact=True).click()
                    picker = page.locator(f'[id="{model.id}-sampling-preset"] input')
                    note = page.locator(f'[id="{model.id}-sampling-preset"]')
                    for preset in model.presets:
                        if not preset.get("experimental"):
                            continue
                        picker.click()
                        page.get_by_role("option", name=preset["name"], exact=True).click()
                        expect(note).to_contain_text(preset["note"])
                        preset_checks += 1
                    picker.click()
                    page.get_by_role("option", name=model.presets[0]["name"], exact=True).click()
                    expect(note).to_contain_text("Baseline preset")
                page.get_by_role("tab", name="Flux.2 Klein", exact=True).click()
                prompt.scroll_into_view_if_needed()
                page.screenshot(path=str(screenshots / "studio-light.png"))

                # Optional media workspaces must mount correctly and report
                # missing inputs without launching installers or GPU engines.
                page.get_by_role("tab", name="Tools & 3D", exact=True).click()
                page.get_by_role("tab", name="SeedVR2 upscale", exact=True).click()
                expect(page.get_by_role("button", name="Install / repair SeedVR2", exact=True)).to_be_visible()
                expect(page.locator("#seedvr2-attention")).not_to_be_visible()
                expect(page.get_by_role("button", name="Install optimizations", exact=True)).not_to_be_visible()
                page.get_by_role("button", name="Install / repair SeedVR2", exact=True).click()
                expect(page.locator("#seedvr2-status")).to_contain_text("Completed.")
                assert seed_installs == [("seedvr2-convrot", False, "w4a8")], seed_installs
                page.locator("#seedvr2-model input").click()
                page.get_by_role("option", name="Classic 3B · GGUF Q8 · GTX compatible", exact=True).click()
                page.get_by_role("button", name="Install optimizations", exact=True).click()
                expect(page.locator("#seedvr2-status")).to_contain_text("Completed.")
                assert seed_installs == [("seedvr2-convrot", False, "w4a8"), ("seedvr2", True, None)], seed_installs
                expect(page.locator("#seedvr2-attention")).to_be_visible()
                page.locator("#seedvr2-model input").click()
                page.get_by_role("option", name="3B · ConvRot W4A8 · RTX · ~2 GB weights", exact=True).click()
                expect(page.locator("#seedvr2-attention")).not_to_be_visible()
                page.get_by_role("button", name="Restore", exact=True).click()
                expect(page.get_by_text("Failed: Import an image or a video first.", exact=True)).to_be_visible()
                seed_source = folder / "brutalisme…é_日本.jpg"
                sample.convert("RGB").save(seed_source)
                page.locator("#seedvr2-source-file input[type=file]").set_input_files(str(seed_source))
                expect(page.locator("#seedvr2-source-preview img")).to_be_visible()
                expect(page.locator("#seedvr2-source-info")).to_contain_text("1024 × 768")
                expect(page.locator("#seedvr2-size-info")).to_contain_text("2048 × 1536")
                page.locator("#seedvr2-scale").get_by_role("radio", name="Custom", exact=True).check()
                expect(page.locator("#seedvr2-resolution input")).to_be_visible()
                expect(page.locator("#seedvr2-size-info")).to_contain_text("1440 × 1080")
                page.locator("#seedvr2-scale").get_by_role("radio", name="×4", exact=True).check()
                expect(page.locator("#seedvr2-resolution input")).not_to_be_visible()
                expect(page.locator("#seedvr2-size-info")).to_contain_text("4096 × 3072")
                page.locator("#seedvr2-detail-strength input[type=number]").fill("35")
                page.locator("#seedvr2-detail-strength input[type=number]").press("Tab")
                page.wait_for_function("document.querySelector('#seedvr2-source-preview img')?.naturalWidth > 0")
                assert not seed_calls, "Upload triggered restoration instead of preview"
                page.get_by_role("button", name="Restore", exact=True).click()
                expect(page.locator("#seedvr2-status")).to_contain_text("Completed.")
                expect(page.locator("#seedvr2-restored-image img")).to_be_visible()
                page.wait_for_function("document.querySelector('#seedvr2-restored-image img')?.naturalWidth === 4096")
                assert len(seed_calls) == 1, seed_calls
                assert seed_calls[0]["scale"] == 4 and seed_calls[0]["details"] == 35, seed_calls
                assert seed_calls[0]["attention"] == "auto", seed_calls
                assert seed_calls[0]["model"] == "3b-w4a8", seed_calls
                page.screenshot(path=str(screenshots / "studio-seedvr2.png"))

                page.get_by_role("tab", name="Capture → splats", exact=True).click()
                page.get_by_role("button", name="Reconstruct", exact=True).click()
                expect(page.get_by_text("Failed: Import either photographs or one video, then clear the other input.", exact=True)).to_be_visible()
                page.get_by_role("button", name="Refresh saved captures", exact=True).click()
                page.screenshot(path=str(screenshots / "studio-capture.png"))

                page.get_by_role("tab", name="Video", exact=True).click()
                page.get_by_role("tab", name="LTX 2.5", exact=True).click()
                expect(page.get_by_role("button", name="Generate LTX video", exact=True)).to_be_visible()
                expect(page.get_by_role("button", name="Download / prepare LTX 2.5", exact=True)).to_be_visible()
                page.screenshot(path=str(screenshots / "studio-ltx25.png"))
                page.get_by_role("tab", name="Images", exact=True).click()

                # Keyboard generation must produce exactly one request.
                prompt.fill("Shortcut request")
                prompt.press("Control+Enter")
                expect(page.locator(".studio-status:visible").first).to_contain_text("Done")
                assert calls == ["Shortcut request"], calls
                assert not list(settings.TMP_DIR.glob("preview_*.png"))

                # Programmatic hand-offs must work without a manual tab click.
                page.get_by_role("button", name="🖼️ Send the selection to Outpaint", exact=True).click()
                expect(page.get_by_role("tab", name="Tools & 3D", exact=True)).to_have_attribute("aria-selected", "true")
                expect(page.get_by_role("tab", name=re.compile("Outpaint"))).to_have_attribute("aria-selected", "true")
                expect(page.locator("img:visible").first).to_be_visible()

                page.get_by_role("tab", name="Gallery", exact=True).click()
                expect(page.locator("#history-grid img")).to_have_count(24)
                page.get_by_role("button", name="Next", exact=True).click()
                expect(page.locator("#history-grid img")).to_have_count(8)
                page.locator("#history-grid img").first.click()
                expect(page.locator("#history-prompt textarea")).to_have_value("A studio browser test")
                page.screenshot(path=str(screenshots / "studio-gallery.png"))
                expect(page.get_by_role("button", name="Send original to upscale", exact=True)).to_have_count(0)

                page.get_by_role("tab", name="Images", exact=True).click()
                page.get_by_role("tab", name="Gallery", exact=True).click()
                expect(page.locator("#history-grid img")).to_have_count(24)
                page.locator("#history-grid img").first.click()
                page.get_by_role("button", name="Use prompt in Images", exact=True).click()
                expect(prompt).to_have_value("Shortcut request")

                # Advanced styles should become usable after their lazy load.
                page.get_by_role("button", name=re.compile("Styles — presets")).click()
                expect(page.get_by_role("tab", name=re.compile("Artistic"))).to_be_visible()
                page.get_by_role("tab", name=re.compile("Artistic")).click()
                page.get_by_role("button", name="🎲 Random (wildcard)", exact=True).click()
                # Closing/reopening must not clear the selected styles.
                page.get_by_role("button", name=re.compile("Styles — presets")).click()

                # All model workspaces still exist and use the new navigation.
                for title in ("Qwen Image 2.1 Turbo", "Krea 2 Turbo", "MOSS-Image SFT", "Flux.2 Klein"):
                    page.get_by_role("tab", name=title, exact=True).click()
                    expect(prompt).to_be_visible()

                # Stop executes outside the inference queue.
                engine_started.clear()
                prompt.fill("Cancel request")
                page.get_by_role("button", name="Generate", exact=True).click()
                expect(page.locator(".studio-status:visible").first).to_contain_text("Loading")
                assert engine_started.wait(10), "Simulated inference never started"
                page.get_by_role("button", name="Stop", exact=True).click()
                expect(page.locator(".studio-status:visible").first).to_contain_text(re.compile("cancelled|Error"))
                assert interrupted.wait(5), "Stop did not interrupt the worker"
                assert active == 0 and peak == 1, (active, peak)

                page.get_by_role("tab", name="Tools & 3D", exact=True).click()
                for removed in ("10 views / 360°", "SVG & layers", "🧩 Layers", "🙂 Faces", "🖐️ Details", "✨ SDXL upscale"):
                    assert page.get_by_role("tab", name=removed, exact=True).count() == 0, removed

                for title in ("Video", "Tools & 3D", "Models", "System", "Images"):
                    page.get_by_role("tab", name=title, exact=True).click()
                    expect(page.get_by_role("tab", name=title, exact=True)).to_have_attribute("aria-selected", "true")

                page.goto(f"http://127.0.0.1:{port}/?__theme=dark", wait_until="domcontentloaded")
                expect(prompt).to_be_visible()
                page.screenshot(path=str(screenshots / "studio-dark.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_function("document.documentElement.scrollWidth <= window.innerWidth + 1")
                assert page.locator("#studio-nav").bounding_box()["width"] >= 350
                page.screenshot(path=str(screenshots / "studio-mobile.png"))
                # A fresh browser page keeps the GPU-heavy 3D journey separate
                # from the long image/video history journey above.
                page.close()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on("pageerror", record_error)
                page.goto(f"http://127.0.0.1:{port}/?__theme=light", wait_until="domcontentloaded")
                page.get_by_role("tab", name="Tools & 3D", exact=True).click()
                page.get_by_role("tab", name="Image → splats", exact=True).click()
                page.get_by_role("button", name="Generate splats", exact=True).click()
                expect(page.locator("#image-splats-status")).to_contain_text("Import an image first")
                splat_source = folder / "objet_é_日本.png"
                sample.save(splat_source)
                page.locator("#image-splats-source input[type=file]").set_input_files(str(splat_source))
                expect(page.locator("#image-splats-source img")).to_be_visible()
                for model_name in ("TripoSplat", "Idol"):
                    page.locator("#image-splats-model input").click()
                    page.get_by_role("option", name=model_name, exact=True).click()
                    if model_name == "Idol":
                        expect(page.locator("#image-splats-info")).to_contain_text("HF_TOKEN")
                    page.locator("#image-splats-count input").click()
                    page.get_by_role("option", name="32,768", exact=True).click()
                    page.get_by_role("button", name="Generate splats", exact=True).click()
                    expect(page.locator("#image-splats-status")).to_contain_text("Completed.")
                    expect(page.locator("#image-splats-files")).to_contain_text(re.compile(r"object\s*\.ply"))
                    expect(page.locator("#image-splats-files")).to_contain_text(re.compile(r"request\s*\.json"))
                assert [call[0] for call in splat_calls] == ["triposplat", "idol"], splat_calls
                assert [call[2] for call in splat_calls] == [20, 50], splat_calls
                page.screenshot(path=str(screenshots / "studio-image-splats.png"))
                page.get_by_role("button", name="Unload 3D preview", exact=True).click()
                expect(page.locator("#image-splats-status")).to_contain_text("preview unloaded")

                page.goto(f"http://127.0.0.1:{port}/?__theme=dark", wait_until="domcontentloaded", timeout=60000)
                expect(page.get_by_role("tab", name="Images", exact=True)).to_be_visible()
                assert not errors, errors
                browser.close()
            print(json.dumps({"build_seconds": round(build_seconds, 3), "browser_ready_seconds": round(ready_seconds, 3),
                              "config_bytes": len(json.dumps(demo.config, default=str)), "page_errors": errors,
                              "simulated_generations": len(calls), "peak_concurrent_generations": peak,
                              "experimental_presets_checked": preset_checks,
                              "simulated_image_splats": splat_calls,
                              "simulated_seedvr2_restorations": len(seed_calls),
                              "screenshots": str(screenshots)}, indent=2))
        finally:
            demo.close()


if __name__ == "__main__":
    main()
