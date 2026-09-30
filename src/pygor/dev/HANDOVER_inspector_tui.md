# Handover: `pygor-tui` blank-slate inspector

Working doc for a fresh agent. NOT for commit (LLM meta files stay out of the repo).
Related auto-memory: `project_workbench_tui.md` (Textual gotchas),
`project_review_cockpit.md` (the proofreading cockpit this shares screens with).

Built 2026-09-22. Status: **done and tested, uncommitted.**

## What this is

`pygor-tui` used to require a recording path and drop you straight into the
re-segmentation screen. It is now a blank-slate terminal inspector: start it with
nothing, browse to a recording, look at it — stack in time, ROI overlays, the trace
under the cursor, the trigger train — and only then re-segment if you decide to.

Three rounds of work, in the order they were asked for:

1. **Blank slate + reader.** Optional path argument; a directory (or nothing) opens a
   file browser. One shared reader decides what counts as a recording.
2. **Interactive inspection.** Frame scrubbing, a probe that follows the mouse or the
   arrow keys, and the trace under the probe drawn as braille (fast) or matplotlib.
3. **Pin + triggers.** Click to pin the probe; triggers drawn on the trace's x axis,
   adaptively, with a statistics line that is the real diagnostic.
4. **Click-to-probe + time zoom** (2026-09-23). User: pinning is pointless if a click
   sets the probe. Hover-follow and the pin are gone; the probe moves on click or
   arrows only, which frees the pointer to read the trace (hover over it = time and
   value in the status line). Trace zooms in time: `+`/`-`, wheel over the trace
   (anchored at the pointer), `[`/`]` pan, `0` reset. Window is stored as fractions
   (`probe.clamp_window`) so it survives moving the probe. Min width 8 samples.
5. **Hand-drawn ROIs** (2026-09-23). `d` from the inspector opens `DrawScreen`
   (`draw_screen.py`): disk (`c`, `+`/`-` radius) or lasso (`l`; drag = freehand
   stroke, click or `space` = one vertex), `enter` adds, `x` deletes the ROI under the
   cursor, `u` undo, `p` mean/correlation, `S` save. Native TUI chosen over a napari
   hand-off (works over SSH). **User's rule: existing ROIs win overlaps** — the new
   shape only takes unclaimed pixels (green in the preview, red where blocked); redraw
   = delete then draw. Mask rules are pure functions in `roi_edit.py`. Preview is PIL,
   pixel-exact (`imaging.edit_preview`, 18 ms on the 64x128 file). Edits live on a
   copy; save goes through `standalone.apply_mask`: compact ids to -1..-n, re-extract
   traces (3.2 s on the long file, in a worker), recompute snippets/averages only if
   present. Arrow keys only in this screen (`l` is the lasso), shift+arrows move ten.
6. **Review sweep** (2026-09-23). Stale thread workers now drop their results
   (`post_if_current`); markup escaped everywhere file/error text is shown; edits
   blocked while saving. Save keeps per-ROI data aligned: `roi_edit.reconcile` finds
   untouched survivors, deletions go through the object's own `keep_rois` (drops
   STRFs/keys, clears caches; `ipl_depths` subset by hand since keep_rois skips it),
   new ROIs get NaN STRF/quality/depth rows. `n_colours` is inferred when a saved
   object reloads with it None (the gb_ablated file does). `.presegment` is written on
   the first save only: it is the original. Lasso stroke ends on any move with no
   button held (Textual any-motion mode reports button 0), so a missed release
   cannot leave it stuck.

## State of the tree

Branch `offsets-pairwise` (unrelated name — the work was done on whatever was checked
out). **Everything is uncommitted.** Nothing else on that branch was touched.

```
?? src/pygor/tui/reader.py          126 lines   new
?? src/pygor/tui/browse_screen.py   539 lines   new
?? src/pygor/tui/probe.py           179 lines   new
?? src/pygor/tui/sparkline.py       208 lines   new
?? src/pygor/test/test_tui_browse.py 856 lines  new — 57 tests
 M src/pygor/tui/imaging.py        +178 -~20
 M src/pygor/tui/standalone.py     +193 -~84
 M src/pygor/tui/napari_launcher.py +62 -~20
```

Suite: **696 passed, 7 skipped, 14 xfailed** — run three consecutive times clean
(there was one flake; it was a real bug, see Gotchas).

## How to run and test

```bash
cd /home/simen/Documents/Git_repos/2p_analysis
uv run pygor-tui                      # browse the working directory
uv run pygor-tui data/gb_ablated      # browse there
uv run pygor-tui <file>.recording.h5  # straight to the inspector
uv run pygor-tui <file>.smp --n-colours 4

# full suite (from pygor/)
PYTHONPATH=src python -m pytest src/pygor/test -q
PYTHONPATH=src python -m pytest src/pygor/test/test_tui_browse.py -q
```

It is a full-screen app, so it cannot be driven through a tool call — headless
testing goes through Textual's pilot (`app.run_test()`), which the test file does
throughout. Test recordings used during the build:

- `data/gb_ablated/0_0_GBabl_SWN_200_ColourSwitcher_RGBUV.recording.h5` — saved
  object, 49576 frames, 64x128, 76 ROIs, 15616 triggers. The stress case.
- `pygor/examples/strf_demo_data.h5` — raw IGOR export, 10000 frames, 4 ROIs.
- `pygor/examples/FullFieldFlash_5_colour_demo.smp` — raw ScanM, unsegmented, 45
  triggers. The sparse-trigger and no-ROIs case.

## Keys (inspector)

| key | does |
|---|---|
| `p` | cycle view: rois, correlation, labels, **stack** |
| `,` `.` | scrub one frame (switches to the stack view if not there) |
| `<` `>` | scrub 25 |
| click | move the probe to that pixel |
| mouse over trace | time and value under the pointer, in the status line |
| `+` `-`, wheel over trace | zoom the trace in time (wheel anchors at the pointer) |
| `[` `]` | pan the zoomed window by half its width |
| `0` | whole trace again |
| arrows, `hjkl` | move probe one pixel; `HJKL` ten |
| `t` | trace style: braille <-> matplotlib panel |
| `v` | napari, detached subprocess |
| `r` | reprocess screen (the cockpit's, bound to this recording) |
| `esc` | back to the browser, or quit if launched on a file |

## Architecture

```
standalone.py      main() -> build_app(target, caps, n_colours) -> StandaloneApp
                   make_reprocess_screen(recording, source, caps)   [factory]
                   APP_CSS                                          [all screens]
browse_screen.py   RecordingTree   filter_paths -> dirs + reader.is_recording
                   BrowseScreen    picks a file, reads nothing until enter
                   InspectScreen   the whole interactive surface
                   TracePane       PanelView + show_text (braille)
reader.py          kind/is_recording/load_recording/summarise
probe.py           pixel_at, at_pixel, roi_ids, trigger_summary,
                   triggers_of, display_range
sparkline.py       Canvas, plot, framed, event_row, BLOCKS, DOT_BITS
imaging.py         + draw_marker, frame_preview (PIL), trace_figure
                   + marker=/marker_colour=/projection= on the existing renderers
napari_launcher.py + spawn(), _add_rois(), --n-colours, uses the shared reader
```

The screens are handed callables, never a dataset binding — that is what lets
`ReprocessScreen` serve both the cockpit and this tool. Keep that.

## Design decisions — do not re-litigate

- **The inspector is read-only.** `r` is one key away but it is a different screen, so
  "check this recording" and "change this recording" never share a keystroke. The
  explicit-file form (`pygor-tui file.h5`) now lands on the inspector too, not on the
  reprocess screen as it used to. That was a deliberate consistency call; the user was
  told and did not object.
- **One reader, three callers.** The browser, the path argument and the napari
  hand-off all go through `reader.load_recording`, so what the browser lists is exactly
  what the other two can open. Before this, the hand-off only called `load_object` and
  raised `No recording groups found` on any raw file.
- **`.smh` is deliberately unlisted.** Core accepts it, but listing both halves of a
  ScanM pair shows every recording twice. The browser offers the `.smp`.
- **Braille for anything that tracks the pointer.** A PNG panel is re-transmitted to
  the terminal in full on every repaint. Braille is text. This is also the only thing
  that works in `--graphics none` and over a slow SSH link.
- **Stack scrubbing goes through PIL, not matplotlib.** 3 ms versus 40 ms.
- **Display range is computed once over the whole stack**, from a ~40-frame subsample
  (`probe.display_range`). Per-frame scaling makes the tissue pulse while scrubbing,
  which reads as a signal that is not there.
- **ROI row lookup is positional, not arithmetic.** `probe.roi_ids` sorts the present
  negative ids and takes `.index()`. `abs(id) - 1` is wrong the moment a mask has a
  gap (delete ROI -3 and -1, -2, -4 occupy three rows), and a trace attributed to the
  wrong cell is worse than no trace.
- **Triggers: ticks when resolvable, density bar when not.** At 5 Hz over an hour there
  are 15616 of them, ~130 per dot column. `sparkline.event_row` switches at
  `n <= cols`. The statistics line (`probe.trigger_summary`) carries the information a
  picture cannot: count, median interval, MAD, gaps (> 1.5x median) and shorts
  (< 0.5x). A dropped trigger shows there *and* as a hole in the bar.

## Gotchas that bit us

**Textual** (also folded into the `project_workbench_tui` memory):

- `remove_children()` completes **asynchronously**. A pane that swaps its contents must
  mount the replacement with a **class, never an id** — the outgoing widget is still
  mounted when the new one arrives, and the duplicate id wedges the message pump. It
  surfaces only as `WaitForScreenTimeout` in a pilot test, never as an exception. Cost
  us an hour.
- **A widget reference does not survive a redraw that re-mounts it.** The detached one
  reports a zero region. In the app that meant `region.contains()` failed and the click
  was silently dropped — an occasional click that does nothing, with no way to tell why.
  Fixed two ways: `image_widget()` filters on `is_attached`, and `pixel_under()` keeps
  `self._panel_region` as a fallback for the remount window. In tests it meant
  `pilot.click(stale_widget)` landed at screen (0,0), hit the header icon and opened the
  command palette — hence `live_widget()` in the test file, which re-queries.
- `capabilities.probe()` returns mode `"none"` whenever stdout is not a tty, which it
  never is under pytest. To exercise the image widget and the mouse path, build
  `Capabilities(mode="halfcell", ...)` directly — see `TestHover` in the test file.
- `pilot.click/hover` refuse an offset outside the visible screen region; the panel can
  extend past it, so use the widget centre, not its far corner (`centre()` helper).

**pygor:**

- `roi_figure` used to crash on an unsegmented recording (`mask=None` then
  `recording.rois` on a `None` recording). Raw ScanM files therefore rendered
  "preview failed" where a perfectly good projection belonged. Fixed — it now draws the
  projection alone.
- `averages`/`snippets` default to `np.nan`, not `None`, on Core. A bare `is None`
  check reports an absent average as present. `reader.summarise` handles it.
- Recomputing `np.mean(images, axis=0)` per crosshair move was 86 ms on the 49576-frame
  file — the whole cost of the redraw. `recording_preview` now takes `projection=` and
  `InspectScreen` caches it (invalidated after a reprocess).

## Measured cost (49576 x 64 x 128 recording, local terminal)

| path | time | bytes transmitted |
|---|---|---|
| `frame_preview` (scrub) | 2.8 ms | 21 KiB |
| `recording_preview` rois + crosshair | 86 ms | 56 KiB |
| `trace_figure` with 15616 triggers | 88 ms | 34 KiB |
| braille `framed` + trigger row | 3.3 ms | 0 |
| `at_pixel` probe lookup | 0.1 ms | 0 |

Pointer-driven PNG repaints are coalesced by `InspectScreen.REDRAW_INTERVAL` (0.12 s);
braille is never throttled.

## Known limits

- **Probe precision is one character cell** — about two pixels on a 64-row scan in a
  30-row pane. Terminal limit, not tunable. Arrow keys are exact.
- Mouse reporting is on while the app runs, so drag-to-select text over the image does
  not work in that pane.
- At mid zoom (~50x on the hour-long file) the trigger density bar aliases into a
  regular `█▆█▆` pattern: bins of 1-2 triggers each. Harmless, but it looks like
  structure.
- The density bar's leftmost column often reads low simply because triggers start
  slightly after t=0.
- The correlation view computes `compute_correlation_projection()` on first view and
  takes a few seconds on a long recording. Threaded, so the UI stays live, but the pane
  sits empty meanwhile.
- The napari hand-off has only been tested as far as the `spawn()` call — no window was
  opened on the user's display during the build. Worth one manual check.

## Next steps, if it gets picked up again

- **A frame cursor on the trace** while in the stack view, so scrubbing and the trace
  are visibly the same axis. Natural pair: click on the trace to jump the stack there.
- **Multi-ROI comparison** — keep several probes, overlay their traces.
- Register the entry point properly: `pygor-tui` is already in `[project.scripts]`, but
  a fresh clone needs `uv sync` before the console script exists.
- The cockpit's `ProofreadApp.open_napari` still has its own copy of the spawn logic;
  `napari_launcher.spawn()` was deliberately not wired into it to avoid touching the
  cockpit. Worth unifying when someone is next in that file.
