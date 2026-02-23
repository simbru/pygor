# Coherence NaN handling — decisions needed

`compute_latency_vectors` currently returns NaN coherence for several ROIs.
Some of these are legitimate (no data), others feel arbitrary or could mask
real issues. This document lists each NaN path and proposes a stance.

## NaN sources

### 1. Too few valid pixels (`mask.sum() < 2`)
**Current behaviour:** `continue` → NaN from initialisation.
**Assessment:** Legitimate. A gradient needs at least 2 pixels. No change needed.

### 2. Zero amplitude sum (`amplitude_weighted=True`)
**Current behaviour:** `continue` → NaN.
**Assessment:** Legitimate but should probably warn. If a user explicitly
requests amplitude weighting and an ROI has zero amplitude everywhere, that's
worth flagging rather than silently returning NaN.
**Proposed:** Add `warnings.warn(f"ROI {i}: zero amplitude sum, coherence set to NaN")`.

### 3. Full-image segmentation mask (upstream in `get_strf_delta_times`)
**Current behaviour:** When segmentation covers *every* pixel (i.e. didn't
isolate anything), the entire ROI is masked → propagates to case 1.
**Assessment:** This is the most arbitrary path. The segmentation saying
"everything" is conceptually different from "nothing". Two options:
  - **(a)** Treat full-coverage segmentation as "no segmentation available" and
    fall back to unsegmented computation for that ROI.
  - **(b)** Keep current behaviour but explicitly mark these ROIs so the user
    can distinguish "no RF" from "segmentation failed".
**Decision:** TBD

### 4. NaN propagation from peak times
**Current behaviour:** `np.gradient` on NaN-containing arrays silently
propagates NaNs through the arithmetic. No `continue` is hit — NaN is
*written* into the coherence array rather than left from initialisation.
**Assessment:** This is a bug-adjacent gap. The code handles masks but not
in-band NaN values. A pixel with no detectable peak time returns NaN from
`get_strf_peak_times`, and `np.gradient` spreads that to neighbours.
**Proposed:** Add `mask &= np.isfinite(lmap)` right after mask construction
in `compute_latency_vectors`. This treats NaN peak-time pixels the same as
masked pixels, which is the only sensible interpretation.

## Summary of proposed changes

| Source | Action |
|--------|--------|
| Few valid pixels | Keep as-is |
| Zero amplitude | Add warning |
| Full segmentation | Decide on (a) vs (b) above |
| NaN in peak times | Add `np.isfinite` guard |
