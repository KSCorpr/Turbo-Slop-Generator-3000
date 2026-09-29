# Studio preview · 1.2.0rc1

This release lives only on `codex/studio-performance-release`. The original
`main` branch remains unchanged. Install this branch in its own folder using
the launchers documented in the README.

## What changed

- Six main tabs: Images, Video, Tools & 3D, Gallery, Models and System.
- New copper, charcoal and warm-white theme, responsive controls, keyboard
  focus indicators and reduced-motion support. No remote fonts are required.
- Prompt, format, quality, seed and batch are immediately accessible;
  advanced controls remain available in expandable panels.
- Ctrl/Cmd+Enter launches the visible image workspace exactly once.
- Gallery reads existing `outputs/` images, pages 24 at a time, caches small
  thumbnails, downloads full originals and reuses saved prompts.
- New generations also save structured JSON parameters beside the existing
  text sidecar, so prompts can be recovered without parsing display text.
- Styles and disk inventory load on demand; preferences are cached with file
  change detection. Preview images are capped at 720 pixels on their longest
  side; generated originals retain their requested resolution.
- All inference entry points share one GPU queue to prevent overlapping model
  jobs from competing for VRAM. Stop bypasses the queue, while the current job
  drains and releases its worker before the next job starts.
- Subprocess cleanup now also handles exceptions in log consumers. Temporary
  generation inputs use unique names and are removed after completion.
- The web UI uses tested Gradio/client versions. Optional GPU tool environments
  remain separate.

## Measurements and verification

The serialized UI configuration fell from **1,928,875 bytes to approximately
1,335,113 bytes (31% smaller)** on the same empty installation and dependency
versions. This measures configuration payload, not GPU generation throughput.

The unit suite contains **562 tests: 561 passed, one platform-dependent skip**
on Python 3.12/Linux. It includes regression checks for original-image
preservation, thumbnail transparency, path boundaries, cached preferences,
worker cleanup, shared concurrency, lazy styles and startup work.

Browser journeys run against the actual Gradio UI with a **simulated engine**
and synthetic image fixtures. They passed in Chromium with no JavaScript
errors, covering keyboard generation, cancellation, Outpaint and upscale
transfers, gallery pagination, prompt reuse, model navigation and a 390-pixel
mobile viewport. Run the checks locally:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m playwright install chromium
python tests/browser_smoke.py --screenshots tmp/ui-check
python -m pip check
```

GitHub Actions runs the unit suite on Windows and Linux and the browser suite
on Linux. Browser screenshots are attached as an Actions artifact.

Real CUDA/Metal inference, installed model weights, video and 3D generation
require a compatible local machine and have not been benchmarked in this
workspace. Use the existing hardware benchmark in System before choosing
production settings. No GPU speedup is claimed from these UI measurements.

## Updates and promotion

ZIP installs follow `config/update-channel.json`, which explicitly names this
preview branch. They will not silently replace the preview with `main`.
Git checkouts update with `git pull --ff-only`; the ZIP code updater refuses to
overwrite them. Engine-only maintenance commands are still available.

Before deliberately promoting this release to `main`, remove the preview
channel file or change its branch to `main`, and update the README download
link. This task does not merge or deploy the branch.
