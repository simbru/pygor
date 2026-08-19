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
- Selecting a label redraws the trace plot for that ROI, in that ROI's own
  colour, read back from the layer with `get_color`.
- ROI labels use a palette built in `pygor/gui/colors.py` rather than
  napari's default, which includes desaturated entries that read as grey
  against the greyscale stack. `gist_rainbow` is sampled once per ROI and
  resampled whenever the count changes, so the colours stay as far apart
  as the ROI count allows and run in a predictable order. Existing ROIs do
  change colour as the count grows, which is the trade for even spacing.
  `gist_rainbow` rather than `rainbow`: it is fully saturated across its
  whole range, where `rainbow` drops to 0.36 saturation in its cyan-green
  region. napari maps label `i` to `colors[1 + (i - 1) % n]`, so index 0
  holds the transparent background and the samples line up with the labels.
  An earlier golden-ratio palette was dropped: its smallest hue gaps land
  on Fibonacci lags, and lag 8 was both Fibonacci and a multiple of the
  4-tone cycle, so every pair 8 apart collided at ~20 degrees.
- A "ROI numbers" Points layer draws each ROI's number at its centroid in
  white, sitting above the labels. It follows strokes, segmentation and
  pushes, and can be hidden with its visibility toggle. napari's text has
  no outline field (`anchor`, `blending`, `color`, `rotation`, `scaling`,
  `size`, `string`, `translation`, `visible`), so a black border round the
  numbers is not available. Faking one with offset duplicate text would
  need the offsets in data coordinates, which do not scale with zoom, so
  the outline thickness would change as the view is zoomed.

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
- Deleting a default layer is recoverable. napari offers no way to make a
  layer undeletable, so the layers are treated as disposable instead: the
  docks resolve the ROI layer by membership in `viewer.layers` rather than
  holding a reference, unpushed edits are written to the recording on the
  `removing` event, and "Restore default layers" rebuilds whatever is
  missing from the recording and rebinds the docks. Without the membership
  check a deleted layer still accepted edits, since it stays alive as a
  Python object — the failure was silent.
- "Lock default layers", on by default, puts a deleted default layer
  straight back. The restore is queued with `QTimer.singleShot` rather than
  run inside the removal event: `viewer.close()` empties the layer list one
  layer at a time, so an inline restore re-added each layer as it was
  removed and the close never terminated. A queued callback simply never
  runs during teardown. Layers the user adds themselves are ignored.

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

## Plot panel

`pygor/gui/widgets/plot.py` owns the window's per-ROI matplotlib axis and
the ROI navigation that drives it. Its View selector carries one entry for
now and hides itself until there is a second, which is where the per-ROI
detail views in `dev/gui_workflow_map.md` belong: RF maps per channel,
temporal kernels, tuning functions.

Population-level plots are not here. The metric histogram sits under the
population table instead, since a distribution is read alongside the
values it summarises.

The Average view draws the mean across stimulus repetitions with the
repetitions themselves behind it, capped at `_MAX_TRIALS_DRAWN` and
reporting how many of how many are shown — a noise recording can carry
nearly two thousand loops, which would be an unreadable smear.

Note that `compute_snippets_and_averages` returns snippets shaped
`(n_rois, n_loops, snippet_length)` and averages `(n_rois, snippet_length)`,
not the `(snippet_length, n_loops, n_rois)` its docstring claims. Confirmed
by checking which axis of the snippets reduces to the averages.

## IPL depth

`pygor/gui/ipl.py` puts the two boundary polylines in the main viewer as
Shapes layers, rather than in the separate viewer `NapariDepthPrompt`
opens. Depths are computed on request instead of on window close, so the
boundaries can be adjusted and recomputed without reopening anything, and
the result lands straight on the recording via `update_ipl_depths`.

The last shape drawn on a layer wins, so redrawing a boundary supersedes
the previous attempt without deleting it first. Once computed, the
population panel switches to the IPL depth metric and shows its
histogram, since a depth you cannot see is not much use. Note that
`calculate_ipl_depths` returns percentages outside 0–100 for ROIs beyond
the boundary pair, where `estimate_ipl_depths` clips.

## Preprocessing panel

`pygor/gui/widgets/preprocessing.py` runs the steps that alter the stack
itself: `preprocess` (light artifact, X flip, detrending) and `register`
(motion correction), plus `reset_images` to undo both. Widget defaults are
seeded from `params._defaults` so the panel agrees with the config rather
than hardcoding its own.

Both steps *replace* `recording.images` rather than editing in place, so
the layers built earlier hold the old array and would show stale pixels.
`refresh_image_layers` repoints them afterwards, and picks up the backup
stacks, which only exist once something destructive has run.

The config stores `normalization` as the string `"None"`, while `register`
wants a real `None`; the panel converts it.

Preprocessing runs on H5 recordings and the mechanism is verified there,
but H5 exports are already preprocessed by IGOR, so it is semantically a
second application. Registration is wired but has not been run against a
real recording — the local stacks are 20684 frames and too slow for a
check. Both want raw `.smp`/`.smh` data to be exercised properly.

## Population panel

`pygor/gui/widgets/population.py` shows one metric across every ROI, so a
cell can be picked out of the population rather than stepped past. The
selected metric drives three things at once: a sortable table, a
histogram under it marking where the selected ROI falls in that ROI's
colour, and optionally the colour of the ROIs themselves. Table and ROI
layer selection track each other in both directions, and the histogram can
be hidden when the table alone is wanted.

`pygor/gui/metrics.py` holds the registry. A `MetricSpec` carries a cheap
`applies` check, a `compute` callable and an `expensive` flag, because
metrics differ from trace sources in three ways: they are computed rather
than looked up, their cost ranges from free to a thousand permutations,
and which exist depends on the analysis class. `compute_metric` drops
anything that raises or that does not return exactly one value per ROI,
since a misaligned metric would mislabel every cell silently.

Colouring by metric swaps the identity palette for a `DirectLabelColormap`
over the value range, drawing non-finite ROIs transparent; unticking
restores the identity palette.

Still missing, in the order `dev/gui_workflow_map.md` suggests: a per-ROI
detail panel dispatched on analysis type, persisted QC flags, knobs that
recompute a metric, and multi-recording context.

## Menus

`pygor/gui/menus.py` adds a **Pygor** menu to napari's own menu bar:

- **Analysis** — segment, extract traces, correlation projection. Same
  methods the dock buttons call, so behaviour cannot diverge.
- **ROIs** — new / next / previous, push to recording, restore layers.
- **View** — checkable items mirroring the dock checkboxes and the ROI
  number layer's visibility, in both directions.
- **Parameters...** and **Save recording as...**

The dock keeps the buttons used constantly; the menu carries the rest so
the panel does not have to grow. Dock actions were closures inside the
magicgui factories, so they were lifted to `run_segmentation`,
`run_extraction` and `run_correlation_projection` for both to share.

The parameter editor is docked from launch rather than opened as a
top-level window on demand. `params.edit(blocking=False)` returns a widget
the caller must keep alive, and dropping it let Python collect the window
the moment it appeared — pressing the button did nothing visible. Docking
hands ownership to Qt, and the menu entry raises the existing dock.

`ParamEditorWidget` groups its rows into collapsible sections by the first
part of each dotted path, rather than listing sixty-odd `section.name`
rows flat. Filtering hides sections left empty and expands those that
match.

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
