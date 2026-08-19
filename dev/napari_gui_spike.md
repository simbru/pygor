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
- The viewer's frame slider drives a cursor on the trace plot.
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

- Not tested against a real recording. `examples/strf_demo_data.h5` is not
  checked in, so `test_launch_against_real_recording` skips here. Run the
  suite where that file exists before trusting the layer/trace wiring.
- Analysis-type subclasses (STRF, OSDS, moving bars, ...) get no
  type-specific docks yet. STRF in particular wants its own panel reusing
  `pygor/strf/plotting`.
- No progress reporting from long jobs; `thread_worker` supports `yielded`
  for that but the underlying methods do not yield.
- Trace dock redraws the whole axis on every selection change. Fine at this
  size, worth revisiting if more panels are added.
- Overlap with `view_stack_rois`, `draw_rois` and `view_images_interactive`
  is unresolved. If this direction is kept, those should either delegate to
  `launch` or be dropped.

## Test run

```
QT_QPA_PLATFORM=offscreen uv run --with pytest --with pytest-qt \
    python -m pytest src/pygor/test/test_gui_spike.py -q
6 passed, 1 skipped
```
