# What the inspector scripts actually do

Read of `Analysis_scripts/RF_DS_story/rf_inspector.py` (156 lines),
`dsi_inspector.py` (240 lines) and the `pool_results.py` registry, to work
out what the napari GUI is competing with. Both inspectors are `# %%` cell
scripts, which is to say: interactive tools written without a GUI to put
them in.

## The loop they both run

1. **Pick one FOV** from `pool_results.FOVS`, a hand-maintained list of
   dicts keyed by `uid`, each naming an OSDS, an FFF and an SWN file plus
   its stimulus geometry (`dir_num`, `n_colours`, `osds_trigger`, ...).
2. **Load two or more recordings for that FOV** from cache with
   `load_object`, and cross-reference them: `rf_inspector` takes quality
   indices from the OSDS recording and applies them to the STRF
   recording's ROIs, guarded by `osds.num_rois == n_roi`.
3. **Compute a population mask** — `valid_rf` from per-channel PCA
   centroids, or `dsi_pval < 0.05` from a permutation test.
4. **Look at the distribution** — histograms of QI and RF amplitude split
   by the mask, to see what the rejected cells have in common.
5. **Sweep the threshold that produced the mask** — `threshold_sd` over
   `[2.0, 1.5, 1.0, 0.75, 0.5]`, or `K_SD` over `[1.5, 2.0, 2.5, 3.0]`,
   counting survivors at each. A flat curve means weak data, a steep one
   means the cut is doing the damage.
6. **Prototype an alternative statistic** and diff it against the current
   one — `rf_inspector`'s SNR-based validity prints `recovered` and `lost`
   ROI ids against the `cs_seg` version.
7. **Step through individual ROIs** by editing a variable (`ROI = 15`,
   `FOV_IDX = 0`, `METRIC = "range"`) and re-running the cell.
8. **Look at that one ROI in detail** — per-channel collapsed RF maps with
   both centroid estimates overlaid, the temporal kernel at the peak
   pixel, or the tuning function with its permutation null distribution.
9. **Drop into a viewer** — the last two lines of `rf_inspector` are
   `swn.view_stack_rois(roi_indices=ROI, labels=False)` and
   `swn.play_multichrom_strf(ROI)`.

So: **population view → pick a suspicious cell → per-ROI detail → adjust a
threshold → recompute → repeat.** Step 9 is where the current GUI starts,
which is to say it covers the end of the loop and none of the rest.

## What the GUI already covers

Loading one recording, stepping ROI by ROI, seeing a trace per ROI,
drawing and editing ROIs, segmentation, trace extraction. That is the
work *before* analysis — getting ROIs onto a recording.

## What it does not

1. **No population view.** Nothing shows per-ROI metrics across the
   recording, so there is no way to pick which ROI to look at. Both
   inspectors spend their first half on exactly this. Biggest gap: it is
   how a person decides to type `ROI = 15`.
2. **No per-ROI detail beyond the raw trace.** STRF work needs per-channel
   RF maps and temporal kernels; OSDS work needs the tuning function and
   the permutation null. The trace dock shows neither.
3. **No parameter sweep with recompute.** Steps 5 and 6 are the actual
   intellectual work and the GUI cannot do them at all.
4. **No QC verdict.** Nothing marks an ROI accepted or rejected, and
   nothing persists such a mark.
5. **Single recording only.** Every inspector session loads an SWN *and*
   an OSDS for the same FOV and reads across them. `launch()` takes one
   recording.
6. **No FOV registry.** `FOVS` is a literal in a script; the GUI opens
   whatever path it was handed.

## What this says about the parameter panel

The current Parameters dock is the right home for the ~60 `AnalysisParams`
pipeline defaults, and a flat table is fine for those — they are set once
and left alone.

The parameters the inspectors actually touch are a different animal, and
should not go in that table:

| script | knob | values tried |
| --- | --- | --- |
| `rf_inspector` | `threshold_sd` | 2.0, 1.5, 1.0, 0.75, 0.5 |
| `rf_inspector` | `K_SD` | 1.5, 2.0, 2.5, 3.0 |
| `rf_inspector` | `MIN_PIX` | 3 |
| `dsi_inspector` | `METRIC` | range, max, peak, auc, ... |
| `dsi_inspector` | phase window | ON edge / OFF edge |

There are only a handful per analysis type, they are always swept rather
than set, and the point of changing one is to see the population mask move
in response. So the useful thing is not a typed widget per parameter but a
small per-analysis panel where changing a knob recomputes the mask and
recolours the ROIs — the sweep in step 5, done visually.

That is worth building only once a population view exists to show the
effect, which is why the ordering below puts it fourth.

## Suggested order

1. **Population panel** — a per-ROI table or scatter of whatever metrics
   the analysis type provides, selection wired to the viewer both ways.
   Turns the viewer into an inspector and is the prerequisite for the rest.
2. **Per-ROI detail panel**, dispatched on analysis type. STRF: channel RF
   maps plus kernels, reusing `pygor/strf/plotting`. OSDS: tuning function
   and permutation null. This is the figure both scripts build by hand.
3. **QC flags** on the recording, persisted through `save_object`, so a
   curation pass survives the session.
4. **Analysis knobs with recompute**, per the table above.
5. **Multi-recording context** — an FOV holding SWN + OSDS + FFF together,
   which also raises where the registry should live.

## Igor, afterwards

`OS_hdf5Export_custom.ipf` in this repo is 76 lines and only handles HDF5
export; the Baden-lab `OS_*` pipeline is not here and would need pointing
at. Worth reading for one question the Python scripts cannot answer,
because they are batch: what the manual curation step looked like — what a
person clicked to accept or reject a cell, and what that decision was
recorded as. That maps directly onto item 3.
