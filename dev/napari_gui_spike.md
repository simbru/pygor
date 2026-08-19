# napari GUI spike

Feasibility probe for a single-window GUI over pygor recordings, on branch
`napari-gui-spike`. Not wired into the package's public API yet.

## What exists

```
src/pygor/gui/
    __init__.py          lazy `launch` re-export
    launch.py            builds one viewer: layers + docks
    roi_bridge.py        IGOR mask <-> napari Labels conversion
    widgets/
        traces.py        matplotlib trace plot, follows ROI selection
        actions.py       magicgui buttons -> thread_worker
src/pygor/test/test_gui_spike.py   headless tests (Qt offscreen)
```

Try it:

```python
import pygor.load
from pygor.gui import launch

rec = pygor.load.Core("recording.h5")
viewer = launch(rec)        # block=True from a plain script
```

## Feasibility findings

Confirmed working, headless, against a stub recording:

- One viewer holds the image stack, comparison stacks, average projection
  and ROIs as a Labels layer.
- Selecting a label redraws the trace plot for that ROI. Label `n` maps to
  trace row `n - 1` in both mask conventions.
- ROIs are navigated from the trace dock: prev/next buttons, a spinbox, and
  the `[` / `]` keys. Selection stays in sync with the Labels layer in both
  directions, so napari's picker mode (`5` or `L`, then click) also drives
  the trace plot. "Centre view" moves the camera to the selected ROI and
  marks it with a crosshair Points layer, restoring the layer-list
  selection so the Labels layer stays in picker mode.
- The viewer's frame slider drives a cursor on the trace plot. Off by
  default: a full matplotlib redraw of a 20684-point trace costs ~110 ms,
  so following the frame stuttered badly. The cursor is now an animated
  artist blitted over a cached background, at ~0.2 ms per update, but the
  toggle stays off since most inspection does not need it.
- magicgui builds the action buttons from type annotations; `thread_worker`
  keeps segmentation and projection off the GUI thread.
- Edited labels convert back to an IGOR-style mask and go through
  `update_rois`, so ROI drawing and analysis share one window.

Environment already has everything: napari 0.7.1, magicgui 0.10.2,
qtpy 2.4.3, matplotlib 3.11.0.

## Deliberate differences from the existing `view_*` methods

The current `pygor/core/gui/methods.py` classes each construct their own
`napari.Viewer` and call `napari.run()`, and `gui_template.py` overrides
`viewer.window._qt_window.closeEvent` to block until close. This spike keeps
the viewer non-blocking and returns it, so results land on the recording
object rather than on window close. Private napari attributes are avoided;
`viewer.window.dock_widgets` is the public accessor (`_dock_widgets` is
deprecated and warns).

## Open items

- `calculate_image_average` returns None when a recording has no
  repetitions, so the Average layer is absent for such recordings. Expected,
  not a failure.
- Analysis-type subclasses (STRF, OSDS, moving bars, ...) get no
  type-specific docks yet. STRF in particular wants its own panel reusing
  `pygor/strf/plotting`.
- No progress reporting from long jobs; `thread_worker` supports `yielded`
  for that but the underlying methods do not yield.
- Trace dock redraws the whole axis on every selection change. That is one
  ~110 ms draw per ROI change, acceptable when clicking through ROIs but
  the obvious next target if it starts to feel sluggish. Decimating the
  trace to roughly the axis width would fix it at the source.
- Overlap with `view_stack_rois`, `draw_rois` and `view_images_interactive`
  is unresolved. If this direction is kept, those should either delegate to
  `launch` or be dropped.

## Test run

```
QT_QPA_PLATFORM=offscreen uv run --with pytest --with pytest-qt \
    python -m pytest src/pygor/test/test_gui_spike.py -q
```

`test_launch_against_real_recording` skips unless a recording is available.
Point it at one with `PYGOR_TEST_H5`:

```
PYGOR_TEST_H5=/path/to/recording.h5 QT_QPA_PLATFORM=offscreen \
    uv run --with pytest --with pytest-qt \
    python -m pytest src/pygor/test/test_gui_spike.py -q
```

Verified against `raw_h5/control/2023-11-14_0_0_SWN_200_Colours.h5`
(20684 frames, 64x128, 11 ROIs): layers built, label 1 resolved to its
`traces_znorm` row. 7 passed.
