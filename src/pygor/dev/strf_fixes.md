# STRF Calculation Fixes: IGOR-Python Parity

This document outlines the discrepancies between the IGOR Pro `OS_STRFs_beta.ipf` implementation and the Python `pygor` implementation, ordered by severity. Each phase should be tested independently to verify correctness.

---

## Status Overview

| Phase | Issue | Severity | Status |
|-------|-------|----------|--------|
| 0 | Correlation argument order | Critical | COMPLETED |
| 1 | Mean stimulus calculation | High | COMPLETED |
| 2 | Noise stimulus initialization (0 vs 0.5) | Medium | COMPLETED |
| 3 | Colour lookup masking | Medium | COMPLETED |
| 4 | Z-normalization before storage | High | COMPLETED |
| 5 | Output axis transpose (y,x vs x,y) | Medium | COMPLETED |
| 6 | Post-processing shape mismatch (5D vs 4D) | Medium | COMPLETED |
| 7 | Add joblib parallelization | Enhancement | COMPLETED |

---

## Phase 0: Correlation Argument Order (COMPLETED)

### Problem
scipy/numpy `correlate(a, b)` uses opposite lag convention from IGOR's `Correlate srcWave, destWave`:
- **IGOR**: `result[k] = sum(src[n] * dest[n-k])` (subtraction)
- **scipy**: `result[k] = sum(a[n] * b[n+k])` (addition)

This caused temporal reversal of STRFs.

### Fix Applied
Reversed argument order in:
- `calculate_optimized.py:68`: `correlate(dest_nodc[i], src_nodc, ...)`
- `calculate.py:62`: `np.correlate(dest_nodc, src_nodc, ...)`
- `calculate.py:82`: `np.correlate(stim_nodc, trace_nodc, ...)`

### Verification
Compare `obj.get_timecourses_dominant()` between pygor and IGOR - should now align without manual reversal.

---

## Phase 1: Mean Stimulus Calculation (HIGH PRIORITY)

### Problem
The mean stimulus (`MeanStim`) is used to normalize STRF kernels. IGOR computes it per-pixel, Python uses a constant.

**IGOR (correct)**:
```igor
// When rr == -1, CurrentTrace is all 1s (or color-filtered 0s/1s)
CurrentPX = NoiseStimulus_Frameprecision[xx][yy][frames] * CurrentTrace[frames]
MeanStim[xx][yy][colour] = V_Avg  // mean of product
```

**Python (incorrect in optimized versions)**:
```python
mean_stim[xx, yy, colour] = np.mean(current_trace)  # Ignores noise stimulus!
```

### Impact
Since `mean_stim` divides the kernel values (normalization step), using a constant instead of per-pixel values produces incorrect spatial normalization.

### Files to Modify
1. `src/pygor/strf/calculate_optimized.py` - lines 309-313
2. `src/pygor/strf/calculate_multicolour_optimized.py` - similar location

### Fix
Replace the simplified mean calculation with proper pixel-wise computation:
```python
if rr == -1:  # compute meanimage
    for xx in range(edge_crop, n_x_noise-edge_crop):
        for yy in range(edge_crop, n_y_noise-edge_crop):
            # Must compute noise_stimulus first (moved outside rr loop)
            current_px = noise_stimulus[xx-edge_crop, yy-edge_crop, :] * current_trace
            mean_stim[xx, yy, colour] = np.mean(current_px)
```

**Note**: This requires restructuring because `noise_stimulus` is currently only built inside the `rr >= 0` block. The noise stimulus mapping needs to happen BEFORE the ROI loop (or at rr==-1).

### Verification
- Compare normalized STRF amplitudes between pygor and IGOR
- Check that spatial structure matches (not just temporal)

---

## Phase 2: Noise Stimulus Initialization Value (MEDIUM PRIORITY)

### Problem
**IGOR**:
```igor
make /B/o/n=(...) NoiseStimulus_Frameprecision = 0.5
```

**Python**:
```python
noise_stimulus = np.zeros((...), dtype=np.float32)
```

### Impact
Frames not covered by any trigger will have value 0 in Python vs 0.5 in IGOR. This affects:
- Correlation at experiment boundaries
- Gaps between stimulus loops

### Files to Modify
1. `src/pygor/strf/calculate_optimized.py` - line 317
2. `src/pygor/strf/calculate_multicolour_optimized.py` - similar location
3. `src/pygor/strf/calculate.py` - if applicable

### Fix
```python
noise_stimulus = np.full((n_x_noise-edge_crop*2, n_y_noise-edge_crop*2, n_f_relevant),
                         0.5, dtype=np.float32)
```

### Verification
- Check STRF quality for experiments with gaps between loops
- Compare edge frames of STRFs

---

## Phase 3: Colour Lookup Masking (COMPLETED)

### Problem
For multi-colour stimuli, IGOR masks the trace to only include frames matching the current colour:

**IGOR**:
```igor
// Build ColourLookup (lines 234-243)
make /o/n=(nF) ColourLookup = NaN
for (ll=0;ll<nColourLoops;ll+=1)
    for (colour=0;colour<nColours;colour+=1)
        currentstartframe = triggertimes_frame[ll*(nColours*nTriggers_per_Colour)+colour*nTriggers_per_Colour]
        currentendframe = triggertimes_frame[ll*(nColours*nTriggers_per_Colour)+(colour+1)*nTriggers_per_Colour-1]
        ColourLookup[currentstartframe,currentendframe]=colour
    endfor
endfor

// Apply masking (lines 290-293)
Multithread CurrentTrace[] = (CurrentLookup[p]==colour)?(CurrentTrace[p]):(0)
```

### Fix Applied
Added to `calculate_optimized.py`:

1. **ColourLookup array construction** (after line 244):
```python
n_colour_loops = int(np.ceil(n_triggers / (n_colours * n_triggers_per_colour)))
colour_lookup = np.full(n_f, np.nan)

for ll in range(n_colour_loops):
    for colour in range(n_colours):
        start_trigger_idx = ll * (n_colours * n_triggers_per_colour) + colour * n_triggers_per_colour
        end_trigger_idx = ll * (n_colours * n_triggers_per_colour) + (colour + 1) * n_triggers_per_colour - 1
        if start_trigger_idx < n_triggers and end_trigger_idx < n_triggers:
            current_start_frame = int(triggertimes_frame[start_trigger_idx])
            current_end_frame = int(triggertimes_frame[end_trigger_idx])
            colour_lookup[current_start_frame:current_end_frame+1] = colour
```

2. **Colour masking application** (in colour loop):
```python
current_lookup = colour_lookup[trigger_start:trigger_start+n_f_relevant].copy()
current_trace = np.where(current_lookup == colour, base_trace, 0.0)
```

### Verification
- Test with multi-colour noise stimulus (RGB/RGBUV)
- Verify each colour channel shows distinct response
- For single-colour stimuli: all frames map to colour 0, so behaviour unchanged

---

## Testing Protocol

For each phase:

1. **Before**: Run notebook cell `out = obj.calculate_strf(noise_array=noise_array)` and save results
2. **Apply fix**: Modify the code
3. **After**: Re-run calculation and compare:
   - Visual comparison of STRF movies (`obj.play_strf()`)
   - Timecourse comparison (`obj.get_timecourses_dominant()`)
   - Statistical comparison (correlation coefficient between pygor and IGOR STRFs)

### Test Data
- Single colour: `2_0_SWN_200_White.smp` with corresponding IGOR H5
- Multi-colour: Need separate test file with RGB/RGBUV stimulus

---

## Phase 4: Z-normalization Before Storage (COMPLETED)

### Problem
Python was applying z-normalization to `current_filter` before storing to `strfs_concatenated`, but IGOR only z-normalizes a **copy** (`CurrentFilter_Smth`) for polarity/SD calculations.

### Fix Applied
- Store `current_filter` to `strfs_concatenated` BEFORE z-normalization
- Apply z-normalization only to `current_filter_smth` copy for SD/polarity calculations

---

## Phase 5: Output Axis Transpose (COMPLETED)

### Problem
Pygor output was `(time, y, x)` but IGOR (after load_strf rotation correction) uses `(time, x, y)`.

### Fix Applied
Changed transpose in `calculate_optimized.py:443` from `(2, 1, 0)` to `(2, 0, 1)`.

---

## Phase 6: Post-processing Shape Mismatch (COMPLETED)

### Problem
`calculate_strf` returns 5D `(n_colours, n_rois, time, x, y)` but `post_process_strf_all` expected 4D `(n_items, time, x, y)`, causing all values to be masked.

### Fix Applied
Added reshape in `strf_data.py:3875-3889` to convert 5D→4D before processing and 4D→5D after.

---

## Phase 7: Joblib Parallelization (COMPLETED)

### Enhancement
Added optional parallel processing of ROIs using joblib for improved performance on multi-core systems.

### Implementation
1. **New helper function** `_process_single_roi()` - encapsulates all per-ROI processing
2. **New parameter** `n_jobs` (default=1) - controls parallelization:
   - `n_jobs=1`: Sequential processing (original behavior)
   - `n_jobs=-1`: Use all available cores
   - `n_jobs=N`: Use N parallel workers

3. **Restructured calculation flow**:
   - Step 1: Compute `mean_stim` (must be done first, needed by all ROIs)
   - Step 2: Process ROIs (sequential or parallel based on `n_jobs`)

### Usage
```python
# Sequential (default, for debugging)
out = obj.calculate_strf(noise_array=noise_array, n_jobs=1)

# Parallel (use all cores)
out = obj.calculate_strf(noise_array=noise_array, n_jobs=-1)

# Parallel (use 4 cores)
out = obj.calculate_strf(noise_array=noise_array, n_jobs=4)
```

---

## Validation Results

**Test: Single-colour STRF calculation**
- Scale ratio (IGOR/pygor std): 1.000
- Pixel-wise correlation: 0.9995
- Max frame agreement: Both find same frame

---

## Notes

- `calculate.py` (non-optimized) has the correct mean stimulus calculation but may have other issues
- The optimized versions were simplified for speed but lost some correctness
- After all phases complete, consider adding unit tests comparing against known IGOR outputs

---

## Clarifications

### Baseline Points (100) vs Baseline_nSeconds (5 seconds)

These are **different parameters** used in different contexts:

| Parameter | Location | Usage |
|-----------|----------|-------|
| `baseline_points = 100` | `calculate_optimized.py:311` | Event counting baseline for differentiated trace normalization |
| `Baseline_nSeconds = 5` | `OS_Parameters` table | Trace z-normalization in `OS_TracesAndTriggers` (preprocessing) |

The `100 points` in Python **correctly matches IGOR** (`OS_STRFs_beta.ipf` line 269):
```igor
make /o/n=(100) CurrentTrace_DIFBase = CurrentTrace_DIF[p]
```

The 5-second baseline is applied during trace preprocessing *before* STRF calculation, not within it. In pygor, this preprocessing happens via `obj.preprocess()` before `calculate_strf()` is called.
