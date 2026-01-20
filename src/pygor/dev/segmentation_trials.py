#%%
import pathlib
import os
from re import split


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


# %%
# Diagnostic: look at correlation projection statistics and try sharpening
import numpy as np
from scipy.ndimage import gaussian_laplace, gaussian_filter
import matplotlib.pyplot as plt

# correlation_projection = obj.correlation_projection
correlation_projection = obj.images.mean(axis=0)
# normalise to 0-1
correlation_projection = (correlation_projection - correlation_projection.min()) / (correlation_projection.max() - correlation_projection.min())
print(f"Correlation stats: min={correlation_projection.min():.3f}, max={correlation_projection.max():.3f}, mean={correlation_projection.mean():.3f}")

# Try different enhancements
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original
axes[0, 0].imshow(correlation_projection, cmap='gray')
axes[0, 0].set_title(f'Original corr (range: {correlation_projection.min():.2f}-{correlation_projection.max():.2f})')

# Laplacian of Gaussian (edge detection / sharpening)
log = -gaussian_laplace(correlation_projection, sigma=1)
axes[0, 1].imshow(log, cmap='gray')
axes[0, 1].set_title('Laplacian of Gaussian (sigma=1)')

# Unsharp mask (sharpen)
blurred = gaussian_filter(correlation_projection, sigma=2)
unsharp = correlation_projection + 2 * (correlation_projection - blurred)
axes[0, 2].imshow(unsharp, cmap='gray')
axes[0, 2].set_title('Unsharp mask')

# Local contrast enhancement
from skimage import exposure
equalized = exposure.equalize_adapthist(correlation_projection, clip_limit=0.03)
axes[1, 0].imshow(equalized, cmap='gray')
axes[1, 0].set_title('Adaptive histogram eq')

# Average stack for comparison
axes[1, 1].imshow(obj.average_stack, cmap='gray')
axes[1, 1].set_title('Average stack')

# Correlation * average (combine structural + functional)
combined = correlation_projection * (obj.average_stack / obj.average_stack.max())
axes[1, 2].imshow(combined, cmap='gray')
axes[1, 2].set_title('Correlation × Average')

plt.tight_layout()

# %% Try segmentation with enhanced image
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Use the enhanced image for segmentation
# Try: correlation_projection, unsharp, equalized, or combined
seg_image = equalized  # <-- change this to try different inputs

threshold = 0.4  # adjust based on which image you use
min_distance = 2

binary = seg_image > threshold
coords = peak_local_max(seg_image, min_distance=min_distance, labels=binary.astype(int))
print(f"Found {len(coords)} peaks")

markers = np.zeros_like(binary, dtype=int)
for i, (y, x) in enumerate(coords, start=1):
    markers[y, x] = i

masks = watershed(-seg_image, markers, mask=binary)

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
final_masks = _shrink_rois(masks, iterations=1)

n_rois = len(np.unique(final_masks)) - 1
print(f"Final: {n_rois} ROIs")

# Visualize
masks_masked = np.ma.masked_where(final_masks == 0, final_masks)

fig, ax = plt.subplots()
ax.imshow(seg_image, cmap='gray')
ax.imshow(masks_masked, cmap='prism', alpha=0.3)
ax.set_title(f'Segmentation on enhanced image ({n_rois} ROIs)')

# %% ==========================================================================
# SEGMENTATION METHOD COMPARISON
# Run all methods and compare visually
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import exposure
from skimage.morphology import local_maxima, h_maxima, remove_small_objects

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
INPUT_MODE = "combined"  # Options: "correlation", "average", "combined"

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

# Normalize to 0-1 for consistent thresholding
corr_norm = (input_img - input_img.min()) / (input_img.max() - input_img.min())
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

m1 = method_watershed_peaks(corr_norm, threshold=0.1, min_distance=1, gap_pixels=2, min_size_to_shrink=8)
results.append((m1, "1: Watershed (peaks)", "min_dist=2, gap=1px, thresh=0.2"))

# -----------------------------------------------------------------------------
# Method 2: H-maxima Watershed - Unituitive, doesnt seem to segment well 
# Suppress small local maxima, use remaining as seeds
# -----------------------------------------------------------------------------
def method_h_maxima_watershed(img, h=.01, threshold=0.2):
    binary = img > threshold
    # h-maxima suppresses peaks that are less than h above surroundings
    hmax = h_maxima(img, h=h)
    markers, n = label(hmax)
    masks = watershed(-img, markers, mask=binary)
    return masks

m2 = method_h_maxima_watershed(corr_norm, h=0.2, threshold=0.5)
results.append((m2, "2: H-maxima Watershed", "h=0.01, thresh=0.2"))
# -----------------------------------------------------------------------------
# Method 3: IGOR-style Flood Fill (region growing)
# Grow from peaks, stop at correlation drop or size limit
# -----------------------------------------------------------------------------
def method_flood_fill(img, threshold=0.4, min_distance=3, max_size=25, drop_fraction=0.5):
    """
    IGOR-style: Start at peaks, grow outward.
    Stop when: pixel < peak * drop_fraction OR region > max_size
    """
    binary = img > threshold
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    masks = np.zeros_like(img, dtype=int)

    for roi_id, (py, px) in enumerate(coords, start=1):
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
            if masks[y, x] != 0:  # already claimed
                continue
            if img[y, x] < stop_val:
                continue
            if not binary[y, x]:
                continue

            visited.add((y, x))
            region.append((y, x))

            # Add neighbors (4-connected)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((y + dy, x + dx))

        # Assign region to mask
        for y, x in region:
            masks[y, x] = roi_id

    return masks

m3 = method_flood_fill(corr_norm, threshold=0.1, min_distance=1, max_size=20, drop_fraction=0)
results.append((m3, "3: Flood Fill (IGOR-style)", "max_size=20, drop=0.5"))

# -----------------------------------------------------------------------------
# Method 4: Gradient-based Watershed - meh, merges a lot despite being intuitive to use 
# Use gradient magnitude to define basins
# -----------------------------------------------------------------------------
def method_gradient_watershed(img, threshold=0.4, min_distance=3):
    from skimage.filters import sobel
    binary = img > threshold
    # Gradient magnitude - high at edges
    gradient = sobel(img)
    # Seeds from correlation peaks
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))
    markers = np.zeros_like(img, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    # Watershed on gradient (finds edges)
    masks = watershed(gradient, markers, mask=binary)
    return masks

m4 = method_gradient_watershed(corr_norm, threshold=0.2, min_distance=1)
results.append((m4, "4: Gradient Watershed", "sobel edges, peak seeds"))

# -----------------------------------------------------------------------------
# PLOTTING - Compare all methods
# -----------------------------------------------------------------------------
n_methods = len(results)
n_cols = 3
n_rows = (n_methods + n_cols - 1) // n_cols + 1  # +1 for original at top

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

# First row: show the input image
axes[0].imshow(corr_norm, cmap='gray')
axes[0].set_title(f'Input: {input_name}')
axes[0].axis('off')

axes[1].imshow(corr_norm > 0.4, cmap='gray')
axes[1].set_title('Binary (thresh=0.4)')
axes[1].axis('off')

# Hide unused in first row
axes[2].axis('off')

# Plot each method result
for i, (masks, title, desc) in enumerate(results):
    ax = axes[i + 3]  # offset by 3 for first row

    # Show correlation as background
    ax.imshow(corr_norm, cmap='gray')

    # Overlay ROIs
    n_rois = len(np.unique(masks)) - 1  # exclude background (0)
    if n_rois > 0:
        masks_masked = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masks_masked, cmap='prism', alpha=0.4)

    ax.set_title(f'{title}\n{desc}\n({n_rois} ROIs)', fontsize=9)
    ax.axis('off')

# Hide any remaining empty subplots
for j in range(len(results) + 3, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('segmentation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== ROI Counts ===")
for masks, title, desc in results:
    n = len(np.unique(masks)) - 1
    print(f"{title}: {n} ROIs")

# %%
