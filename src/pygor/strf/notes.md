# New STRF Analysis Methods - Development Notes

## Overview
This document summarizes the new analysis methods added to the STRF module during the development session. All implementations follow the project's modular organization pattern where complex logic lives in specialized submodules and STRF class methods provide simple access points.

## New Files Created

### 1. `extrema_timing.py`
**Purpose**: Analyze temporal extrema in STRFs
**Key Functions**:
- `map_extrema_timing(strfs, threshold=3.0, exclude_firstlast=(1,1))` - Core vectorized function
- `convert_timing_to_milliseconds(timing_maps, frame_rate_hz=60.0, time_offset=0)` - Unit conversion
- `map_extrema_timing_wrapper(strf_obj, ...)` - STRF object interface wrapper
- `compare_color_channel_timing_wrapper(strf_obj, ...)` - Color channel comparison wrapper

**Key Features**:
- Fully vectorized for performance with large datasets
- Handles both 3D (single cell) and 4D (multiple cells) input
- Automatic threshold masking (±3 SD by default)
- Returns simple arrays instead of complex dictionaries
- Millisecond conversion support

### 2. `spatial_alignment.py`
**Purpose**: Analyze spatial alignment and overlap between color channels
**Key Functions**:
- `compute_spatial_overlap_metrics(strf1_2d, strf2_2d, ...)` - Core overlap calculations
- `compute_centroid(spatial_map, mask=None, weighted=True)` - Centroid computation
- `compute_spatial_offset(strf1_2d, strf2_2d, ...)` - Spatial offset measurements
- `analyze_multicolor_spatial_alignment(strf_obj, roi, ...)` - Comprehensive alignment analysis
- `plot_spatial_alignment(alignment_results, ...)` - Visualization
- Wrapper functions for STRF object integration

**Key Features**:
- Multiple overlap metrics: correlation, Jaccard index, overlap coefficient
- Multiple offset methods: centroid, peak, cross-correlation
- Comprehensive multi-channel analysis
- Built-in visualization tools
- Handles different time collapse methods (peak, std, sum)

## New STRF Class Methods

### Temporal Analysis
1. **`map_extrema_timing(roi=None, threshold=3.0, ...)`**
   - Maps timing of extrema for each pixel
   - For single STRF: Returns 2D array (y, x)
   - For all STRFs: Returns 3D array (n_strfs, y, x)
   - NaN indicates pixels below threshold
   - For multicolor data, use manual indexing or multicolour_reshape() to organize by color channels

2. **`compare_color_channel_timing(roi, color_channels=(0,1), ...)`**
   - Compares timing between two color channels
   - Returns timing difference map (channel2 - channel1)

### Spatial Analysis
3. **`analyze_spatial_alignment(roi, threshold=3.0, reference_channel=0, ...)`**
   - Comprehensive spatial alignment analysis across all color channels
   - Returns n_colors × n_colors matrices for correlation, overlap, and distance
   - Holistic approach examining all channel pairs simultaneously

4. **`compute_color_channel_overlap(roi, color_channels=(0,1), ...)`**
   - Computes overlap metrics between two specific color channels
   - Returns correlation, Jaccard index, overlap coefficient, etc.

5. **`compute_spatial_offset_between_channels(roi, color_channels=(0,1), ...)`**
   - Measures spatial offset between two color channels
   - Multiple methods: centroid, peak, cross-correlation

6. **`plot_spatial_alignment(roi, threshold=3.0, ...)`**
   - Creates visualization of spatial alignment across channels
   - Returns matplotlib figure object

## Design Principles Applied

### 1. Modular Organization
- Complex logic implemented in focused submodules
- STRF class methods are simple wrappers that call submodule functions
- Follows existing project patterns (similar to `cs_seg`, `bootstrap`, etc.)

### 2. Performance Optimization
- `map_extrema_timing` is fully vectorized (no pixel-by-pixel loops)
- Uses efficient numpy operations for speed with large datasets
- Simplified return types (arrays instead of complex dictionaries)

### 3. User-Friendly Interface
- Simple method calls from STRF objects
- Consistent parameter naming across methods
- Optional millisecond conversion for timing analysis
- Flexible threshold and method options

### 4. Data Format Consistency
- Returns simple numpy arrays where possible
- NaN values consistently indicate below-threshold or invalid data
- Metadata included in dictionaries when additional context needed

## Testing Checklist

### Basic Functionality
- [ ] Load STRF data with `pygor.load.STRF("strf_demo_data.h5")`
- [ ] Test `map_extrema_timing()` on single ROI and all ROIs
- [ ] Verify timing maps have reasonable values and NaN masking
- [ ] Test millisecond conversion functionality

### Multicolor Analysis
- [ ] Test `compare_color_channel_timing()` with different color channel pairs
- [ ] Verify `analyze_spatial_alignment()` works across all channels
- [ ] Test `compute_color_channel_overlap()` metrics are reasonable
- [ ] Check `compute_spatial_offset_between_channels()` with different methods

### Visualization
- [ ] Test `plot_spatial_alignment()` produces meaningful plots
- [ ] Verify color-coded visualizations work correctly
- [ ] Check that plots handle edge cases (no signal, single channel, etc.)

### Error Handling
- [ ] Test with invalid ROI indices
- [ ] Test with non-multicolor data where multicolor methods should fail
- [ ] Verify threshold edge cases are handled properly
- [ ] Test with minimal datasets

### Performance
- [ ] Test with large datasets to verify vectorization works
- [ ] Compare timing vs. previous implementations if any exist
- [ ] Memory usage should be reasonable for typical datasets

## Usage Examples

```python
# Load STRF data
import pygor.load
strf_obj = pygor.load.STRF("strf_demo_data.h5")

# Basic timing analysis
timing_map = strf_obj.map_extrema_timing(roi=0)  # Single STRF - 2D array (y, x)
all_timing = strf_obj.map_extrema_timing()       # All STRFs - 3D array (n_strfs, y, x)
timing_ms = strf_obj.map_extrema_timing(roi=0, return_milliseconds=True)

# For multicolor data, organize by color channels manually:
import pygor.utilities
timing_reshaped = pygor.utilities.multicolour_reshape(all_timing, strf_obj.numcolour)
# timing_reshaped will be (n_colors, n_rois, y, x)

# Color channel comparison
timing_diff = strf_obj.compare_color_channel_timing(roi=0, color_channels=(0,1))

# Spatial alignment analysis - NEW MATRIX FORMAT
alignment = strf_obj.analyze_spatial_alignment(roi=0)

# Access correlation matrix (n_colors × n_colors)
corr_matrix = alignment['correlation_matrix']
print(f"Red-Green correlation: {corr_matrix[0, 1]:.3f}")
print(f"Red-Blue correlation: {corr_matrix[0, 2]:.3f}")

# Access overlap matrix (Jaccard indices)
overlap_matrix = alignment['overlap_matrix']
print(f"Red-Green overlap: {overlap_matrix[0, 1]:.3f}")

# Access distance matrix (centroid distances)
distance_matrix = alignment['distance_matrix']
print(f"Red-Green distance: {distance_matrix[0, 1]:.2f} pixels")

# Visualize matrices
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im1 = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('Correlation Matrix')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(overlap_matrix, cmap='viridis', vmin=0, vmax=1)
axes[1].set_title('Overlap Matrix')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(distance_matrix, cmap='plasma')
axes[2].set_title('Distance Matrix')
plt.colorbar(im3, ax=axes[2])

# Individual channel pair analysis (still works)
overlap = strf_obj.compute_color_channel_overlap(roi=0, color_channels=(0,1))
offset = strf_obj.compute_spatial_offset_between_channels(roi=0)

# Visualization
fig = strf_obj.plot_spatial_alignment(roi=0)
```

## Known Limitations

1. **Time collapse methods**: Currently supports 'peak', 'std', 'sum' but could be extended
2. **Multicolor requirement**: Some methods require multicolor data, properly validated
3. **Memory usage**: Large datasets with many ROIs/colors will use significant memory
4. **Edge cases**: Very noisy data or minimal signals may produce unreliable results

## Future Enhancements

1. **Additional metrics**: Could add more spatial overlap metrics (Dice coefficient, etc.)
2. **Temporal alignment**: Could extend to analyze temporal alignment between channels
3. **Batch processing**: Could add methods for processing multiple experiments
4. **Statistical testing**: Could add significance testing for alignment metrics
5. **3D visualization**: Could extend plotting to interactive 3D views

## File Structure Summary
```
src/pygor/strf/
├── extrema_timing.py          # New: Temporal extrema analysis
├── spatial_alignment.py       # New: Spatial alignment analysis
├── notes.md                   # New: This documentation
└── (existing files...)
```

All new methods integrated into `src/pygor/classes/strf_data.py` following existing patterns.