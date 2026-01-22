#%%
import pathlib
import os

os.chdir(pathlib.Path(__file__).parent.parent.parent.parent)  # Go to pygor root

# Activate your environment with pygor installed, and import pygor.load
# pygor.load will output a message about which classes are available to you.
# This takes a moment to fetch (expect waiting for a few seconds), and will
# only output the first time it is run.
import pygor.load

#%%
# Path to example data - update this to your file
EXAMPLE_PATH = r".\examples\FullFieldFlash_4_colour_demo.smp"
CUSTOM_CONFIG = r".\configs\example.toml" # Specify custom config if needed

#%%
obj = pygor.load.Core(EXAMPLE_PATH, config=CUSTOM_CONFIG)

#%%
obj.preprocess(detrend=False)

#%%
obj.register(plot=True, mode="mirror")

obj.compute_correlation_projection()

#%%
# Quick visualization of the correlation projection
import numpy as np
import matplotlib.pyplot as plt

stack = obj.correlation_projection
normd_stack = (stack - stack.min()) / (stack.max() - stack.min())
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
avg = ax[0].imshow(normd_stack, cmap='gray')
other = ax[1].imshow(normd_stack > 0.5, cmap='gray')
plt.colorbar(avg, ax=ax[0])
plt.colorbar(other, ax=ax[1])
plt.show()

# %% ==========================================================================
# SEGMENTATION METHOD COMPARISON
# Using the new lightweight segmentation module
# ==========================================================================

from pygor.segmentation.lightweight import (
    segment_watershed,
    segment_flood_fill,
    prepare_image,
)

"""
NOTES:
- Watershed and Flood Fill (IGOR-style) are now in pygor.segmentation.lightweight
- Combined image (correlation × average) works best as input overall
- These methods automatically mask the light artifact region
"""

# =============================================================================
# PREPARE IMAGE (handles input mode and artifact masking)
# =============================================================================
INPUT_MODE = "combined"  # Options: "correlation", "average", "combined"

# prepare_image handles normalization and artifact masking automatically
img, artifact_fill_width = prepare_image(obj, input_mode=INPUT_MODE)
print(f"Using input mode: {INPUT_MODE}")
print(f"Masked artifact region: columns 0-{artifact_fill_width - 1}")

# Store results: (masks, title, description)
results = []

# =============================================================================
# Method 1: Watershed segmentation
# =============================================================================
m1 = segment_watershed(
    img,
    threshold=0.1,
    min_distance=1,
    gap_pixels=2,
    min_size_to_shrink=10,
    min_roi_size=3,
)
results.append((m1, "1: Watershed", "thresh=0.1, gap=2px"))

# =============================================================================
# Method 2: IGOR-style Flood Fill
# =============================================================================
m2 = segment_flood_fill(
    img,
    threshold=0.15,
    min_distance=1,
    max_size=20,
    drop_fraction=0.2,
    min_gap=0,
    min_roi_size=3,
)
results.append((m2, "2: Flood Fill (IGOR-style)", "thresh=0.15, max=20, drop=0.2"))

# =============================================================================
# PLOTTING - Compare methods
# =============================================================================
n_methods = len(results)
n_cols = 3
n_rows = (n_methods + 1 + n_cols - 1) // n_cols  # +1 for input image

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

# Input image
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Input: {INPUT_MODE}')
axes[0].axis('off')

# Plot each method result
for i, (masks, title, desc) in enumerate(results):
    ax = axes[i + 1]
    ax.imshow(img, cmap='gray')
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

# ALTERNATIVE: Use the high-level API via obj.segment_rois()

# This is the recommended way to use segmentation in production code:
# obj.segment_rois(mode="watershed", input_mode="combined", threshold=0.1)
# obj.segment_rois(mode="flood_fill", input_mode="combined", max_size=20)

# %%
