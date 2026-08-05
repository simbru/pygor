# Pygor test suite

## Running it

```bash
# from pygor/
python src/pygor/test/run_tests.py         # everything
python src/pygor/test/run_tests.py -k strf # one topic
pytest                                     # same thing, config is in pyproject.toml
```

`run_tests.py` forwards its arguments to pytest and exists because the README
and `run_test.bat` point at it. Everything else — test paths, the per-test
timeout, the markers — is configured under `[tool.pytest.ini_options]` in
`pyproject.toml`.

pytest and pytest-timeout come from the dev dependency group, so `uv sync`
installs them.

## Test data

The integration tests read `examples/strf_demo_data.h5`, which is downloaded
separately (see the main README). Tests that need it carry the `demo_data`
marker and `conftest.py` skips them when the file is absent, so a fresh clone
still runs the unit tests.

```bash
pytest -m "not demo_data"   # unit tests only, no recording needed
```

## Running unattended

Three things used to stop the suite finishing on its own, and the fixes are
worth knowing about before adding tests:

- **napari windows.** Anything that opens a viewer blocks until a human closes
  it, and under a headless session Qt aborts the process outright. GUI methods
  are now tagged with `@pygor.core.gui.interactive`; the smoke tests filter on
  that marker, and `STRF` no longer generates `_by_channel` copies of them.
  **Tag any new GUI method with the decorator** — the tests will not find it by
  name.
- **stdin prompts.** `run_bootstrap()` asked for confirmation on stdin. It now
  raises when there is no terminal, so a batch script gets an error instead of
  hanging.
- **matplotlib.** `conftest.py` forces the Agg backend and closes figures
  between tests, so `plt.show()` is a no-op.

Per-test timeout is 300s. A test that hits it has hung, and the run says so
instead of sitting there.

## Layout

| File | What it covers |
|---|---|
| `conftest.py` | Headless setup, demo-data skipping, shared objects |
| `helpers.py` | Demo-data path, and the introspection behind the smoke tests |
| `test_Core.py` | `Core` against the demo recording |
| `test_STRF.py` | `STRF` against the demo recording |
| `test_edge_cases.py` | Extrema timing and spatial overlap on synthetic arrays |
| `test_analyses_import.py` | Dynamic class discovery in `pygor.load` |
| `test_experiment_fetch.py` | `Experiment.fetch` against mocked recordings |
| `test_ipl.py`, `test_core_ipl_depths.py` | IPL depth estimation |
| `test_osds_tuning.py` | Direction and orientation tuning metrics |
| `test_export_*.py` | H5 export round-trips |

Fixtures for the demo recording come in two forms. `core` and `strf` are loaded
once per session and must be treated as read-only; `fresh_core` and
`fresh_strf` give a test its own copy to mutate.

## The method smoke tests

`test_method_is_wired_up` calls every public no-argument method on `Core` and
`STRF`, and `test_roi_method_is_wired_up` does the same for methods whose only
required argument is an ROI index. Both fail on `AttributeError`, `NameError`
and `UnboundLocalError` — the signature of a rename that missed a call site —
and accept any other exception that carries a message, since a method refusing
its input is doing its job.

They do not check that any answer is correct. They exist because this codebase
delegates heavily to submodules, and four such delegating calls were found
pointing at functions that had been renamed.

`KNOWN_BROKEN` in `test_STRF.py` lists methods that fail on the demo recording
for real reasons. They run as `xfail`, so the suite stays green while the debt
stays visible, and each one flips to `XPASS` the moment it is fixed.

## Adding tests

- Mark anything that reads the demo recording with `@pytest.mark.demo_data`.
- Write output to `tmp_path`, never next to the source file.
- Tag new GUI methods with `@pygor.core.gui.interactive`.
