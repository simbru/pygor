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
- Selecting a label redraws the trace plot for that ROI.

### ROI numbering

Three numberings are in play and they do not all line up:

| where | background | first ROI | example with 21 ROIs |
| --- | --- | --- | --- |
| IGOR mask (`recording.rois`) | `1` | `-1` | `-1` … `-21` |
| napari label (ROIs layer, spinbox) | `0` | `1` | `1` … `21` |
| trace row (`traces_raw`, `traces_znorm`) | n/a | `0` | `0` … `20` |

`extract_traces` emits rows in ROI id order (`-1`, `-2`, ...) and packs
them with no gaps. Labels can have gaps: erasing an ROI removes its id
from the mask but every later ROI keeps its label. So `label - 1` is only
the right row while the labels run contiguously from 1. `roi_bridge`
resolves the row by position in `roi_ids_in_order(recording.rois)`
instead, and reports no trace for a label with no row.
- ROIs are navigated from the trace dock: prev/next buttons, a spinbox, and
  the `,` / `.` keys. A "New ROI" button (`n`) selects the next unused
  label, so painting or polygon-drawing starts a fresh ROI instead of
  adding to the currently selected one; napari has no built-in binding for
  this. The spinbox bound accounts for a pending label that has no pixels
  and no trace yet.
- An "Auto-new" toggle, on by default, advances to the next free label
  after every completed stroke, for drawing many blob ROIs in a row. It
  also moves off an occupied label when a drawing mode is entered, since
  the selection starts on ROI 1 for trace inspection and the first stroke
  would otherwise extend that ROI. It hangs off the
  Labels layer's `paint` event, which napari emits from
  `_commit_staged_history` when a stroke's undo history is committed on
  mouse release, so it fires once per stroke rather than once per mouse
  move. Erase strokes are skipped. Leave it off to build one ROI from
  several strokes. Selection stays in sync with the Labels layer in both
  directions, so napari's picker mode (`5` or `L`, then click) also drives
  the trace plot. "Centre view" moves the camera to the selected ROI and
  marks it with a white cross Points layer sized to a thirtieth of the
  image width, so it scales with the field of view. The layer-list
  selection is restored after adding it, so the Labels layer stays in
  picker mode.
- The viewer's frame slider drives a cursor on the trace plot. Off by
  default: a full matplotlib redraw of a 20684-point trace costs ~110 ms,
  so following the frame stuttered badly. The cursor is now an animated
  artist blitted over a cached background, at ~0.2 ms per update, but the
  toggle stays off since most inspection does not need it.
- magicgui builds the action buttons from type annotations; `thread_worker`
  keeps segmentation and projection off the GUI thread.
- Edited labels convert back to an IGOR-style mask and go through
  `update_rois`, so ROI drawing and analysis share one window. Extraction
  reads `recording.rois` rather than the layer, so Extract traces syncs the
  layer across first; without that, hand-drawn ROIs were silently missing
  from `traces_raw` and `traces_znorm`. The explicit push button remains,
  and both skip the write when the mask is unchanged.
- The Labels layer starts with `preserve_labels` and `contiguous` on, so
  painting does not eat into ROIs already placed and filling stays within
  the region under the cursor.
- Deleting the ROI layer is recoverable. napari offers no way to make a
  layer undeletable, so the layer is treated as disposable instead: the
  docks resolve it by membership in `viewer.layers` rather than holding a
  reference, unpushed edits are written to the recording on the `removing`
  event, and "Restore ROI layer" rebuilds it from `recording.rois` and
  rebinds the docks. Without the membership check a deleted layer still
  accepted edits, since it stays alive as a Python object — the failure
  was silent.

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

## napari Labels controls, for reference

- **n edit dim** — how many dimensions a paint or fill stroke reaches
  across. The ROI mask is 2D, so this stays at 2 and each stroke edits the
  plane. It matters for 3D label volumes, where 3 would paint a sphere
  through neighbouring slices.
- **contiguous** — restricts the fill bucket to the connected run of pixels
  under the cursor. Unticked, filling recolours every pixel carrying that
  label anywhere in the image.
- **preserve labels** — when on, painting only writes into background and
  leaves existing ROIs untouched. Useful for drawing up against ROIs
  already placed without eating into them. Toggle with `B`.

## Open items

- `calculate_image_average` returns None when a recording has no
  repetitions, so the Average layer is absent for such recordings. Expected,
  not a failure.
- Analysis-type subclasses (STRF, OSDS, moving bars, ...) get no
  type-specific docks yet. STRF in particular wants its own panel reusing
  `pygor/strf/plotting`.
- No progress reporting from long jobs; `thread_worker` supports `yielded`
  for that but the underlying methods do not yield.
- Trace dock redraws the whole axis on every ROI change, ~110 ms. Accepted
  as-is: it is only paid on selection, not per frame. Decimating the trace
  to roughly the axis width is the fix if that ever changes.
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
