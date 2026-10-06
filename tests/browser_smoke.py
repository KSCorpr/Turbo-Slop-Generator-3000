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
from atelier import settings
from atelier.engine import generate as engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    expect.set_options(timeout=15000)
    calls = []
    cancelled = threading.Event()
    interrupted = threading.Event()
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
            try:
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

        stack.enter_context(patch.object(engine, "generate", generate))
        stack.enter_context(patch.object(engine, "cancel", cancel))
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
                page.screenshot(path=str(screenshots / "studio-light.png"))

                # Optional media workspaces must mount correctly and report
                # missing inputs without launching installers or GPU engines.
                page.get_by_role("tab", name="Tools & 3D", exact=True).click()
                page.get_by_role("tab", name="SeedVR2 upscale", exact=True).click()
                expect(page.get_by_role("button", name="Install / repair SeedVR2", exact=True)).to_be_visible()
                page.get_by_role("button", name="Restore", exact=True).click()
                expect(page.get_by_text("Failed: Import an image or a video first.", exact=True)).to_be_visible()
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
                expect(page.locator("#history-grid img")).to_have_count(7)
                page.locator("#history-grid img").first.click()
                expect(page.locator("#history-prompt textarea")).to_have_value("A studio browser test")
                page.screenshot(path=str(screenshots / "studio-gallery.png"))
                page.get_by_role("button", name="Send original to upscale", exact=True).click()
                expect(page.get_by_role("tab", name="🔼 Upscale", exact=True)).to_have_attribute("aria-selected", "true")

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
                for title in ("Qwen Image 2.1", "Ming Image", "Krea 2 Turbo", "Z-Image Turbo", "Flux.2 Klein"):
                    page.get_by_role("tab", name=title, exact=True).click()
                    expect(prompt).to_be_visible()

                # Stop executes outside the inference queue.
                prompt.fill("Cancel request")
                page.get_by_role("button", name="Generate", exact=True).click()
                expect(page.locator(".studio-status:visible").first).to_contain_text("Loading")
                page.get_by_role("button", name="Stop", exact=True).click()
                expect(page.locator(".studio-status:visible").first).to_contain_text(re.compile("cancelled|Error"))
                assert interrupted.wait(5), "Stop did not interrupt the worker"
                assert active == 0 and peak == 1, (active, peak)

                # A full ten-view batch must stream, download and convert in
                # the real browser using the same simulated image engine.
                page.get_by_role("tab", name="Tools & 3D", exact=True).click()
                page.get_by_role("tab", name="10 views / 360°", exact=True).click()
                page.get_by_role("button", name="Generate 10 views", exact=True).click()
                expect(page.get_by_text("Failed: Import a source image first.", exact=True)).to_be_visible()
                expect(page.get_by_text("Also create an equirectangular 360×180 panorama", exact=True)).not_to_be_visible()
                page.get_by_text("360 scene — camera rotates in place", exact=True).click()
                expect(page.get_by_text("Also create an equirectangular 360×180 panorama", exact=True)).to_be_visible()
                source_file = folder / "input.png"
                sample.save(source_file)
                page.locator("#views-source input[type=file]").set_input_files(str(source_file))
                expect(page.locator("#views-source img")).to_be_visible()
                start_calls = len(calls)
                page.get_by_role("button", name="Generate 10 views", exact=True).click()
                expect(page.get_by_text("Completed.", exact=True)).to_be_visible(timeout=30000)
                assert len(calls) == start_calls + 10, calls
                assert active == 0 and peak == 1, (active, peak)
                batches = list(settings.OUTPUT_DIR.glob("views-scene-*"))
                directory = next(p for p in batches if p.is_dir())
                assert len(list(directory.glob("[0-9][0-9]-*.png"))) == 10
                page.screenshot(path=str(screenshots / "studio-multiview.png"))
                page.get_by_text("Advanced and conversion tools", exact=True).click()
                page.locator("#views-import input[type=file]").set_input_files(
                    [str(p) for p in sorted(directory.glob("[0-9][0-9]-*.png"))])
                expect(page.get_by_text("10-nadir.png", exact=True)).to_be_visible()
                page.get_by_role("button", name="Convert views to 360 PNG", exact=True).click()
                expect(page.locator("#views-log textarea")).to_have_value(re.compile("Reprojecting the fixed-camera views"))
                expect(page.get_by_text("Completed.", exact=True)).to_be_visible(timeout=30000)
                assert len(calls) == start_calls + 10, "Conversion launched image generation"
                assert list(settings.OUTPUT_DIR.glob("views-converted-*/equirectangular-360.png"))
                page.screenshot(path=str(screenshots / "studio-panorama.png"))

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
                assert not errors, errors
                browser.close()
            print(json.dumps({"build_seconds": round(build_seconds, 3), "browser_ready_seconds": round(ready_seconds, 3),
                              "config_bytes": len(json.dumps(demo.config, default=str)), "page_errors": errors,
                              "simulated_generations": len(calls), "peak_concurrent_generations": peak,
                              "screenshots": str(screenshots)}, indent=2))
        finally:
            demo.close()


if __name__ == "__main__":
    main()
