# Studio · 1.2.0rc1

The Studio interface is promoted from `codex/studio-performance-release` to
`main`. Use the installers and main-branch download documented in the README.

## What changed

- Six main tabs: Images, Video, Tools & 3D, Gallery, Models and System.
- New copper, charcoal and warm-white theme, responsive controls, keyboard
  focus indicators and reduced-motion support. No remote fonts are required.
- Prompt, format, quality, seed and batch are immediately accessible;
  advanced controls remain available in expandable panels.
- Ctrl/Cmd+Enter launches the visible image workspace exactly once.
- Gallery reads existing `outputs/` images, pages 24 at a time, caches small
  thumbnails, downloads full originals and reuses saved prompts.
- Gallery refresh checks actual file metadata, so additions, deletions and
  replacements are detected even when Windows directory timestamps lag.
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

The original Studio preview passed **561 of 562 tests, with one
platform-dependent skip**, on Python 3.12/Linux. Promotion adds a regression
check for replaced gallery images and makes the file-arrival test reproduce
unchanged directory timestamps. The suite also checks original-image
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

## Updates

ZIP installs follow `config/update-channel.json`, which now explicitly names
`main`. An existing Studio preview ZIP installation moves to this channel on
its next code update; subsequent updates follow `main`.
Git checkouts use the same update launcher: it fetches and fast-forwards the
current upstream branch, without ZIP extraction or a branch switch. Local code
blocking the update is saved automatically in a Git stash before continuing.
No Git command or extra flag is required. The launcher detects portable Python,
an existing project venv, or Python from PATH. Models, outputs, settings and unrelated untracked files
are excluded. The stash remains available; it is never automatically reapplied.
Engine-only maintenance commands remain available. Git `--check` compares cached
refs without fetching; ZIP rollback does not operate on Git checkouts.

The README download and clone commands also point to `main`.
