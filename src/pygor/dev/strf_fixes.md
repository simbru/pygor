# STRF Calculation Fixes: IGOR-Python Parity

This document outlines the discrepancies between the IGOR Pro `OS_STRFs_beta.ipf` implementation and the Python `pygor` implementation, ordered by severity. Each phase should be tested independently to verify correctness.

---

## Status Overview

| Phase | Issue | Severity | Status |
|-------|-------|----------|--------|
| 0 | Correlation argument order | Critical | COMPLETED |
| 1 | Mean stimulus calculation | High | COMPLETED |
| 2 | Noise stimulus initialization (0 vs 0.5) | Medium | COMPLETED |
| 3 | Colour lookup masking | Medium | ON HOLD |

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

## Phase 3: Colour Lookup Masking (MEDIUM PRIORITY)

### Problem
For multi-colour stimuli, IGOR masks the trace to only include frames matching the current colour:

**IGOR**:
```igor
Multithread CurrentTrace[] = (CurrentLookup[p]==colour)?(CurrentTrace[p]):(0)
```

**Python** (currently):
```python
# Create color-filtered trace from base trace (simplified for single color)
current_trace = base_trace.copy()  # No masking!
```

### Impact
Multi-colour STRF calculations will be incorrect - all colours will show the same response.

### Files to Modify
1. `src/pygor/strf/calculate_optimized.py` - need to add ColourLookup and masking
2. `src/pygor/strf/calculate_multicolour_optimized.py` - may already have this

### Fix
Need to:
1. Build `ColourLookup` array mapping frames to colours
2. Apply masking: `current_trace = np.where(colour_lookup == colour, base_trace, 0)`

### Verification
- Test with multi-colour noise stimulus
- Verify each colour channel shows distinct response

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

## Notes

- `calculate.py` (non-optimized) has the correct mean stimulus calculation but may have other issues
- The optimized versions were simplified for speed but lost some correctness
- After all phases complete, consider adding unit tests comparing against known IGOR outputs
