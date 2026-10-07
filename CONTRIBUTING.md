# Contributing to Subsample

This file is for people changing Subsample's code, and subsystem.co does not
publish it. For using Subsample, see [https://subsystem.co/subsample](https://subsystem.co/subsample).

## Tests

For working on Subsample itself (everything in the README works from a plain
`pip install` - no clone needed): clone the repo, install editable with the
dev extras, and run the suite.

```bash
git clone https://github.com/simonholliday/subsample.git && cd subsample
pip install -e ".[dev]"
pytest
```

Renders and the analysis of captures run in worker processes (see
`docs/architecture.md`).
- **A test that patches what a render or an analysis runs, or reads what one
  writes,** builds its `TransformProcessor` or `SampleProcessor` with
  `processes=False`, which keeps the work on threads.
- **A test that needs a worker process to run a step of its own** takes it from
  `tests/render_steps.py`. That module registers its steps wherever it is
  imported, the worker included.
- **The shared worker pools stay up between tests,** so their workers start
  once. `tests/test_parallelism.py` stops them around each of its tests, which
  check that this process can still fork.

## Type checking

Same setup as [Tests](#tests):

```bash
mypy
```

That checks the package, the tests and the maintainer scripts: the folders
listed under `[tool.mypy]` in `pyproject.toml`.

## On GitHub

Every push and pull request runs the tests and the type check on Linux and
macOS, under Python 3.12 and 3.14 (`.github/workflows/tests.yml`). The
workflow also lists the system libraries a fresh machine needs first.

## Maintainer scripts

The scripts in `scripts/` are maintainer tools and need a repo
checkout: `measure_midi_latency.py` and `measure_handler_timing.py` (the
latency guards described under [Measuring latency](#measuring-latency)),
`regen_previews_png.py` (regenerate preview thumbnails after a format bump),
and `extract_gm_drums.py` (regenerate the shipped GM reference fingerprints
from a SoundFont).

### Measuring latency

Two included scripts measure the [software parts](https://subsystem.co/subsample/guide/playing-live#where-the-delay-comes-from) on your own hardware. MIDI
dispatch:

```bash
python scripts/measure_midi_latency.py --count 1000
```

and per-note handling (selection, variant lookup and render):

```bash
python scripts/measure_handler_timing.py
```
