# Handover: unanchored (STA-independent) spatial snippet analysis

Working doc for a fresh agent. NOT for commit (LLM meta files stay out of the repo).
Full project history is in the auto-memory `project_snippet_covariance_analysis.md`.

## The goal

Write a new version of the event-triggered "snippet" analysis (currently in
`pygor/src/pygor/dev/snippet_analysis.py`) that derives the **spatial position of
the ON and OFF response components from the events + all pixels' snippets**, WITHOUT
borrowing anything from the linear STA (`obj.strfs`). The output we want: for a given
ROI, where in space is the cell ON-driven vs OFF-driven, estimated from the events
themselves — no peak-pixel assumption.

## Why the current method is wrong (the thing to fix)

The current spatial position is **100% inherited from the linear STRF**, not derived
from the events:

1. `peak_pixel_neighbourhood(obj, roi)` = `argmax(|obj.strfs[roi]|)` over time → one
   pixel `(px,py)`. This is the ONLY spatial estimate in the whole pipeline.
2. `extract_snippets_spatial(..., center=(px,py), crop=...)` centres the crop on that
   pixel → any kernel blob is centred by construction.
3. `extract_snippets(..., pixels=<5px cross at (px,py)>)` + `split_updown` sort events
   by the pre-event contrast **at that pixel** → up/down are ON/OFF at one location.

Consequence: for a spatially-opponent (Gabor) RF the two lobes sit at *different*
pixels, but the analysis only "sees" the peak lobe; the offset lobe never organizes
the split and averages toward zero. So up/down overlap at the centre, and the method
is parasitic on the linear analysis it was meant to go beyond.

**Key insight:** the *temporal* event detection is fine and NOT the problem — events
are defined from the ROI's calcium trace (a legitimate response signal). Only the
*spatial* anchoring is broken. Keep event detection; replace the spatial estimator.

## Environment & how to run/test

- Repo: `/home/simen/Documents/Git_repos/2p_analysis` (uv workspace). Run with
  `uv run python ...` from the repo root.
- The dev file is a `# %%` percent-script with IPython magics, so it can't be plain
  `exec`'d. Pattern used for headless testing:
  ```python
  src = open(DEVFILE).read().splitlines()
  code = "\n".join(l for l in src if not l.strip().startswith("%"))
  # or cut at the "Demo: run end-to-end" marker to get functions only, then exec.
  ns = {}; exec(code_functions_only, ns)
  ```
- Headless: `matplotlib.use("Agg")`, save PNGs to a scratch dir, then Read them to
  eyeball. Interactive figures/movies don't render in a script.
- Example recording (single white channel):
  `/mnt/data/Igor analyses/OSDS/251104 OSDS/0_1_SWN_200_White.recording.h5`
- Stimulus noise array:
  `/home/simen/Noise_npy_arrs/9deg_200_SINGLEcolour_30000x6x10_0.25_1.npy`
- Load: `obj = pygor.load.STRF.load_object(path)`.
- Loading + one STC-null run is slow-ish; a full functions exec + a few ROIs is ~30-60s.

## Data conventions & gotchas (READ THESE — each one bit us)

- **x/y axis swap (critical).** `obj.strfs[roi]` is `(time, x, y)` with `x`=24, `y`=40.
  The noise array is `noise_array[iy, ix, pattern]` — X and Y are FLIPPED vs the STRF.
  A strf pixel `(i_x, i_y)` maps to `noise_array[i_y, i_x, pattern]`.
  `extract_snippets_spatial` already returns `(x, y)`-ordered snippets matching
  `obj.strfs`; keep that convention so your maps line up.
- **Frame-precise stimulus mapping — reuse verbatim.** `build_frame_to_pattern`
  replicates `pygor/src/pygor/strf/calculate_strf.py` lines ~471-513 EXACTLY: walk
  `obj.triggertimes_frame`, hold each noise pattern until the next trigger, cap gaps at
  `max_frames_per_trigger=8`, cycle the pattern index INCLUDING the `+=1` on the
  out-of-bounds `continue` (pattern phase depends on it). Do not reinvent this.
- **Lag convention.** `taus = arange(n_f_filter) + (1 - n_f_filter_past)`; negative =
  past; snippet at `event_frame + tau`. `frame_duration = obj.linedur_s *
  obj.images.shape[1]` = 0.064 s. `n_f_filter = obj.strfs.shape[1]` (62).
  `strf_window_from_obj(obj)` returns `(frame_duration, n_f_filter, n_f_filter_past)`
  (n_f_filter_past=31, symmetric split — the split isn't stored, derived from
  `strf_dur_ms/2`, empirically validated: RF energy peaks ~lag 27/62 ≈ -0.26s, causal).
- **Contrast = value − 0.5** (baseline 0.5). Binary noise → contrast ±0.5.
- **Binary-stimulus discretization.** Any single-window contrast scalar is DISCRETE
  (mean of ~N ±0.5 samples → ~N+1 levels). A GMM/clustering on such a key fits the
  quantization grid, not biology. Avoid scalar-key clustering for structure discovery.
- **Edge artifact (bites centroid/peak finding).** The noise stimulus does NOT tile the
  full FOV; border pixels are held at baseline → degenerate. ALWAYS edge-crop
  (`edge_crop=2`) before any centroid / peak-pixel / variance computation, or a
  near-constant border pixel wins `argmax`. We hit this repeatedly: use max-VARIANCE
  (over lags) pixels, edge-masked — not `argmax(|mean|)` (a constant border pixel has
  huge |mean|, ~zero variance).
- **Bidirectional cells.** SyGCaMP8m bipolars are graded, non-spiking, respond BELOW
  baseline. `detect_events(sign='pos')` finds depolarising events only; `sign='neg'`
  finds hyperpolarising. To characterise BOTH ON and OFF spatial components you may
  need both event signs (or think carefully about what one sign gives you).
- **Temporal offset vs the STA (accepted, don't "fix").** Events are detected on the
  calcium DERIVATIVE (`np.diff`); pygor's STA is weighted by the calcium LEVEL. The
  derivative leads the level, so event-triggered kernels sit at a systematically
  different (earlier) lag than `obj.strfs`, consistently across ROIs. This is a
  reference difference, not an off-by-one; the user has chosen to leave it as-is.
- **Timescales.** Imaging 15.625 Hz (64 ms/frame); stimulus 5 Hz (~3.1 frames/update);
  GCaMP8m τ≈80 ms (~1.25 frames). Calcium is NOT slow relative to the frame rate here.
- **Memory.** Full-frame per-event snippets are `(n_ev, 62, 24, 40)` float32 ≈ 236 MB at
  ~1000 events. `extract_snippets_spatial` has a `max_gib` guard. Time-crop lags
  (`time_crop_lags`) to shrink; covariance over the full field is high-dim — regularize.

## What to reuse (functions in snippet_analysis.py — don't reinvent)

- `build_frame_to_pattern(obj, noise_array, ...)` → `(frame_to_pattern, trigger_start,
  n_f_relevant)`. Frame-precise stim map. REUSE.
- `strf_window_from_obj(obj)` → `(frame_duration, n_f_filter, n_f_filter_past)`.
- `detect_events(obj, roi, threshold=1.75, sign='pos', trigger_start, n_f_relevant,
  use_znorm=True)` → `(event_frames, event_amps)`. Temporal event detector. REUSE.
- `extract_snippets_spatial(noise_array, frame_to_pattern, event_frames,
  n_f_filter_past, n_f_filter, center=None, crop=6, ...)` → `(snips (n_ev,n_lags,nx,ny),
  taus, box)`. **Pass `crop=None` for the FULL FIELD (all pixels)** — this is your raw
  material. Memory-guarded. REUSE (this is the un-anchored data source; just don't feed
  it a peak-pixel centre).
- `time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2)` → bool lag mask.
- STC machinery if you go the covariance route: `stimulus_prior_basis`, `compute_stc`,
  `stc_event` (prior-whitened covariance in a truncated-PCA subspace — the pattern for
  handling the high-dim full-field covariance efficiently).
- Do NOT reuse for spatial position: `peak_pixel_neighbourhood`, `extract_snippets`
  (1-D peak-pixel), `split_updown`, `updown_spatial` — these are the anchored path.
- pygor library helpers: `pygor.strf.spatial.collapse_3d(arr_3d, zscore)` ([t,x,y]→[x,y]
  — note: gave near-uniform maps on small kernels for us; peak-energy-lag frame was
  cleaner), `pygor.strf.contouring.contour_centroid` / `bipolar_contour` (RF centroids
  by polarity), `pygor.strf.temporal.extract_timecourse`, `pygor.plotting.play_movie_4d`
  (stack `(k,1,time,x,y)` for a k-panel synced movie), `spacetime_plot` (kymograph).

## Design space for the un-anchored spatial estimator (not prescriptive)

The un-anchored raw material = full-field per-event snippets `(n_ev, n_lags, 24, 40)`
from `extract_snippets_spatial(..., crop=None)`. Options for deriving ON/OFF position:

1. **Full-field event-triggered average (simplest, linear-but-independent).**
   `event_sta = snips.mean(0)` → `(n_lags, 24, 40)`. At the peak-energy lag, the positive
   region = ON component, negative = OFF component; take polarity-split centroids
   (edge-cropped) via `contour_centroid` / thresholded weighted centroid. This gives
   ON/OFF positions from the events, fully independent of `obj.strfs`. Caveat: it's a
   first-order reverse correlation (linear), just event-driven and un-anchored — good
   for "where are ON/OFF", not for non-linearity.

2. **Spatial decomposition of the event ensemble (NMF / ICA / PCA of pixel-timecourses).**
   Reshape to per-pixel event-triggered features and factor into spatial components.
   Can separate spatially-distinct ON and OFF subunits without assuming one centre.

3. **Event-triggered covariance / STC over the full field.** The eigen-filters localize
   multiple spatial features with no anchor; use the prior-PCA-subspace trick
   (`stimulus_prior_basis` + `compute_stc`) to keep the 24×40×lags covariance tractable.
   This is the non-linearity-preserving route.

4. **Per-pixel split, then aggregate.** Instead of splitting events by one pixel, define
   an ON/OFF membership *per pixel* (each pixel's event-triggered contrast) and build the
   ON-position and OFF-position maps from the ensemble — no single anchor.

Conceptual fork the user is weighing (unresolved — discuss): a fully un-anchored split
that reproduces the STA's spatial structure tends to BE the STA (linear, circular),
whereas keeping it a non-linearity probe means the un-anchored estimate need not match
the STA. Decide what "success" is before building: (a) recover the Gabor's offset lobes
from events independently of `obj.strfs`, or (b) find spatial event-types the STA misses.

## Validation

- **Motivating test ROI: 44** — a Gabor / spatially-opponent RF (ON lobe above, OFF lobe
  below in `obj.strfs`). The whole point: does the un-anchored method recover the two
  offset lobes at their true positions from events alone? If ON/OFF centroids come out
  offset (matching the STA lobes) WITHOUT using the STA, that's success for goal (a).
- Other ROIs: 8 / 26 / 65 (strong OFF-driven, single-lobe), 15 (near-symmetric). Use
  `threshold≈1.5–2`.
- Sanity: the full-field event STA should roughly agree with `obj.strfs[roi]` in
  spatial layout (up to the known temporal offset and the level-vs-derivative
  difference) — agreement validates the extraction; the value is that it's derived
  independently. Disagreement in POSITION is the interesting signal.
- Convergent context: across every method tried so far (peak-pixel split-axis, STC
  eigenvalues vs linear-null, cluster stability) the tested cells read as LINEAR — no
  non-linear/co-incident structure found. The pipeline is validated; the example cells
  are just linear. Don't expect fireworks; expect a correct, honest spatial estimate.

## Traps that already cost us time

- `argmax(|mean|)` for peak-pixel picks a constant border pixel → flat/garbage. Use
  max-variance-over-lags, edge-masked.
- `collapse_3d` returned near-uniform maps on the small kernels → use the peak-energy
  lag frame for spatial display instead.
- Full-D covariance `eigh` (24×40×lags)² is minutes-slow → do it in a truncated
  prior-PCA subspace (see `compute_stc`), not the raw space.
- Auto-k GMM on a scalar contrast key fits the binary-stimulus quantization, not biology.
- Forgetting the x/y swap silently transposes your spatial maps (peak-pixel 1-D checks
  still pass — validate the full 2-D layout against `obj.strfs`).
