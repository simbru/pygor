#%%
import pathlib
import os


os.chdir(pathlib.Path(__file__).parent.parent.parent.parent)  # Go to pygor root

# Activate your environment with pygor installed, and import pygor.load
# pygor.load will output a message about which classes are available to you.
# This takes a moment to fetch (expect waiting for a few seconds), and will
# only output the first time it is run.
import pygor.load

 #%%`
# Path to example data - update this to your file
EXAMPLE_PATH = r".\examples\FullFieldFlash_4_colour_demo.smp"
CUSTOM_CONFIG = r".\configs\example.toml" # Specify custom config if needed

# Shrink for gaps (inline to avoid cellpose import)
from scipy.ndimage import binary_erosion as _erode, label as _label
def _shrink_rois(masks, iterations=1, min_size_to_shrink=10):
    """Shrink ROIs by erosion, but only if they're larger than min_size_to_shrink."""
    result = np.zeros_like(masks)
    for roi_id in np.unique(masks):
        if roi_id == 0:
            continue
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()
        # Only shrink if ROI is large enough
        if roi_size >= min_size_to_shrink:
            shrunk = _erode(roi_mask, iterations=iterations)
            if shrunk.any():
                result[shrunk] = roi_id
            else:
                # If erosion removes everything, keep original
                result[roi_mask] = roi_id
        else:
            # Small ROIs: keep as-is
            result[roi_mask] = roi_id
    return result

#%%
obj = pygor.load.Core(EXAMPLE_PATH, config=CUSTOM_CONFIG)

#%%
obj.preprocess(detrend=False)


#%% 
obj.register(plot=True, mode = "mirror")

obj.compute_correlation_projection()

#%%
# obj.segment_rois()
import matplotlib.pyplot as plt
# stack = obj.average_stack
stack = obj.correlation_projection
normd_stack = (stack - stack.min()) / (stack.max() - stack.min())
fig, ax = plt.subplots(1,2, figsize=(10, 3))
avg = ax[0].imshow(normd_stack, cmap='gray')
other = ax[1].imshow(normd_stack>0.5, cmap='gray')
plt.colorbar(avg, ax=ax[0])
plt.colorbar(other, ax=ax[1])


# obj.compute_traces_from_rois()


# %% ==========================================================================
# SEGMENTATION METHOD COMPARISON
# Run all methods and compare visually
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

"""
NOTES:
- Watershed (peaks) and Flood Fill (IGOR-style) give best results overall
- IGOR style needs to be able to fit ROIs more closely together (smaller gaps)

the others are not that good

Combined image (correlation × average) seems to work best as input overall

"""

# =============================================================================
# CHOOSE INPUT IMAGE HERE
# =============================================================================
INPUT_MODE = "average"  # Options: "correlation", "average", "combined"

if INPUT_MODE == "correlation":
    input_img = obj.correlation_projection.copy()
    input_name = "Correlation Projection"
elif INPUT_MODE == "average":
    input_img = obj.images.mean(axis=0)
    input_name = "Average Projection"
elif INPUT_MODE == "combined":
    # Correlation weighted by average (structural + functional)
    corr = obj.correlation_projection.copy()
    avg = obj.images.mean(axis=0)
    corr_n = (corr - corr.min()) / (corr.max() - corr.min())
    avg_n = (avg - avg.min()) / (avg.max() - avg.min())
    input_img = corr_n * avg_n
    input_name = "Correlation × Average"
else:
    raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE}")

# Normalize to 0-1 for consistent thresholding (do this BEFORE masking artifact)
corr_norm = (input_img - input_img.min()) / (input_img.max() - input_img.min())

# =============================================================================
# MASK OUT LIGHT c REGION
# =============================================================================
# The light artifact region is at the left edge of the image.
# IGOR fills pixels 0 through artifact_width (inclusive), so we mask artifact_width + 1 columns.
artifact_width = obj.params.artifact_width
artifact_fill_width = artifact_width + 1  # IGOR uses inclusive indexing

# Set artifact region to 0 AFTER normalization so it doesn't affect the scaling
corr_norm[:, :artifact_fill_width] = 0
print(f"Masked light artifact region: columns 0-{artifact_fill_width-1} (artifact_width={artifact_width})")
print(f"Using: {input_name}")

# Store results: (masks, title, description)
results = []

# -----------------------------------------------------------------------------
# Method 1: Local Maxima Seeded Watershed - Good, intuitive, simple
# Find peaks, use as seeds, watershed to boundaries
# -----------------------------------------------------------------------------
def method_watershed_peaks(img, threshold=0.4, min_distance=3, gap_pixels=1, min_size_to_shrink=10):
    """
    Watershed segmentation seeded from local maxima.

    Parameters:
    - threshold: intensity threshold for foreground
    - min_distance: minimum distance between seed peaks (controls ROI density)
    - gap_pixels: erosion iterations to create gaps between ROIs
    - min_size_to_shrink: ROIs smaller than this won't be eroded
    """
    binary = img > threshold
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))
    markers = np.zeros_like(img, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    masks = watershed(-img, markers, mask=binary)
    # Erode to create gaps between ROIs
    if gap_pixels > 0:
        masks = _shrink_rois(masks, iterations=gap_pixels, min_size_to_shrink=min_size_to_shrink)
    return masks

m1 = method_watershed_peaks(corr_norm, threshold=0.1, min_distance=1, 
                        gap_pixels=2, min_size_to_shrink=10)
results.append((m1, "1: Watershed (peaks)", "min_dist=1, gap=1px, thresh=0.15"))
    
# -----------------------------------------------------------------------------
# Method 2: IGOR-style Flood Fill (region growing)
# Grow from peaks, stop at correlation drop or size limit
# -----------------------------------------------------------------------------
def method_flood_fill(img, threshold=0.4, min_distance=3, max_size=25, drop_fraction=0.5,
                      min_gap=1):
    """
    IGOR-style: Start at peaks, grow outward.
    Stop when: pixel < peak * drop_fraction OR region > max_size

    Parameters:
    - threshold: intensity threshold for foreground
    - min_distance: minimum distance between seed peaks
    - max_size: maximum pixels per ROI
    - drop_fraction: stop growing when intensity drops below peak * drop_fraction
    - min_gap: minimum gap (in pixels) to maintain from other ROIs
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    binary = img > threshold
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    # Sort peaks by intensity (brightest first get priority)
    peak_intensities = [img[y, x] for y, x in coords]
    sorted_indices = np.argsort(peak_intensities)[::-1]
    coords = coords[sorted_indices]

    masks = np.zeros_like(img, dtype=int)
    # Buffer zone tracks pixels that are too close to existing ROIs
    buffer_zone = np.zeros_like(img, dtype=bool)

    # Structuring element for buffer dilation
    if min_gap > 0:
        struct = generate_binary_structure(2, 2)  # 8-connectivity for buffer

    # 4-connectivity neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for roi_id, (py, px) in enumerate(coords, start=1):
        # Skip if seed is in buffer zone
        if buffer_zone[py, px]:
            continue

        peak_val = img[py, px]
        stop_val = peak_val * drop_fraction

        # BFS flood fill
        visited = set()
        queue = [(py, px)]
        region = []

        while queue and len(region) < max_size:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue
            if y < 0 or y >= img.shape[0] or x < 0 or x >= img.shape[1]:
                continue
            if masks[y, x] != 0:  # already claimed by another ROI
                continue
            if buffer_zone[y, x]:  # in buffer zone of another ROI
                continue
            if img[y, x] < stop_val:
                continue
            if not binary[y, x]:
                continue

            visited.add((y, x))
            region.append((y, x))

            # Add neighbors
            for dy, dx in neighbors:
                queue.append((y + dy, x + dx))

        # Assign region to mask
        for y, x in region:
            masks[y, x] = roi_id

        # Update buffer zone around this ROI
        if min_gap > 0 and len(region) > 0:
            roi_mask = masks == roi_id
            dilated = binary_dilation(roi_mask, structure=struct, iterations=min_gap)
            buffer_zone = buffer_zone | (dilated & ~roi_mask)

    return masks

m2 = method_flood_fill(corr_norm, threshold=0.1, min_distance=1, max_size=12,
                    drop_fraction=.3, min_gap=0)
results.append((m2, "2: Flood Fill (IGOR-style)", "max=12, drop=0.3"))


# -----------------------------------------------------------------------------
# PLOTTING - Compare methods
# -----------------------------------------------------------------------------
n_methods = len(results)
n_cols = 3
n_rows = (n_methods + 1 + n_cols - 1) // n_cols  # +1 for input image

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

# Input image
axes[0].imshow(corr_norm, cmap='gray')
axes[0].set_title(f'Input: {input_name}')
axes[0].axis('off')

# Plot each method result
for i, (masks, title, desc) in enumerate(results):
    ax = axes[i + 1]
    ax.imshow(corr_norm, cmap='gray')
    n_rois = len(np.unique(masks)) - 1
    if n_rois > 0:
        masks_masked = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masks_masked, cmap='prism', alpha=0.4)
    ax.set_title(f'{title}\n{desc}\n({n_rois} ROIs)', fontsize=9)
    ax.axis('off')

# Hide unused axes
for j in range(n_methods + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

print("\n=== ROI Counts ===")
for masks, title, desc in results:
    n = len(np.unique(masks)) - 1
    print(f"{title}: {n} ROIs")

# %%
