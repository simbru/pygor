#%%
import pathlib
import os

os.chdir(pathlib.Path(__file__).parent.parent.parent.parent)  # Go to pygor root

import pygor.load

#%%
# Path to example data - update this to your file
EXAMPLE_PATH = r".\examples\FullFieldFlash_4_colour_demo.smp"
CUSTOM_CONFIG = r".\configs\example.toml"

#%%
obj = pygor.load.Core(EXAMPLE_PATH, config=CUSTOM_CONFIG)

#%%
obj.preprocess(detrend=False)

#%%
obj.register(plot=True, mode="mirror")
obj.compute_correlation_projection()

# %% ==========================================================================
# BLOB DETECTION METHODS - TUNING PLAYGROUND
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log, blob_dog, blob_doh
from skimage.draw import disk

# =============================================================================
# CHOOSE INPUT IMAGE
# =============================================================================
INPUT_MODE = "correlation"  # Options: "correlation", "average", "combined"

if INPUT_MODE == "correlation":
    input_img = obj.correlation_projection.copy()
    input_name = "Correlation Projection"
elif INPUT_MODE == "average":
    input_img = obj.images.mean(axis=0)
    input_name = "Average Projection"
elif INPUT_MODE == "combined":
    corr = obj.correlation_projection.copy()
    avg = obj.images.mean(axis=0)
    corr_n = (corr - corr.min()) / (corr.max() - corr.min())
    avg_n = (avg - avg.min()) / (avg.max() - avg.min())
    input_img = corr_n * avg_n
    input_name = "Correlation × Average"

# Normalize to 0-1
img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
print(f"Using: {input_name}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def blobs_to_masks(blobs, img_shape, radius_multiplier=1.0):
    """Convert blob detections to mask array."""
    masks = np.zeros(img_shape, dtype=int)
    for i, (y, x, sigma) in enumerate(blobs, start=1):
        radius = max(1, int(radius_multiplier * sigma))
        rr, cc = disk((int(y), int(x)), radius, shape=img_shape)
        unclaimed = masks[rr, cc] == 0
        masks[rr[unclaimed], cc[unclaimed]] = i
    return masks

def plot_blobs(ax, img, blobs, title, radius_multiplier=1.0):
    """Plot blobs as circles on image."""
    ax.imshow(img, cmap='gray')
    for y, x, sigma in blobs:
        radius = radius_multiplier * sigma
        circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=1)
        ax.add_patch(circle)
    ax.set_title(f'{title}\n({len(blobs)} blobs)')
    ax.axis('off')

# =============================================================================
# BLOB LoG - Laplacian of Gaussian
# Most accurate, slowest. Good for well-defined blobs.
# =============================================================================
# TUNING PARAMETERS:
LOG_MIN_SIGMA = 1      # Minimum blob radius (in pixels) / sqrt(2)
LOG_MAX_SIGMA = 5      # Maximum blob radius
LOG_NUM_SIGMA = 5      # Number of sigma values to try
LOG_THRESHOLD = 0.02   # Detection threshold (lower = more blobs)

blobs_log = blob_log(img,
                     min_sigma=LOG_MIN_SIGMA,
                     max_sigma=LOG_MAX_SIGMA,
                     num_sigma=LOG_NUM_SIGMA,
                     threshold=LOG_THRESHOLD)
# LoG radius = sqrt(2) * sigma
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# =============================================================================
# BLOB DoG - Difference of Gaussian
# Faster approximation of LoG.
# =============================================================================
# TUNING PARAMETERS:
DOG_MIN_SIGMA = 1      # Minimum blob radius / sqrt(2)
DOG_MAX_SIGMA = 5      # Maximum blob radius
DOG_THRESHOLD = 0.02   # Detection threshold

blobs_dog = blob_dog(img,
                     min_sigma=DOG_MIN_SIGMA,
                     max_sigma=DOG_MAX_SIGMA,
                     threshold=DOG_THRESHOLD)
# DoG radius = sqrt(2) * sigma
blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

# =============================================================================
# BLOB DoH - Determinant of Hessian
# Fastest. Better for bright blobs on dark background.
# =============================================================================
# TUNING PARAMETERS:
DOH_MIN_SIGMA = 1      # Minimum blob radius
DOH_MAX_SIGMA = 5      # Maximum blob radius
DOH_NUM_SIGMA = 5      # Number of sigma values
DOH_THRESHOLD = 0.002  # Detection threshold (much lower than LoG/DoG)

blobs_doh = blob_doh(img,
                     min_sigma=DOH_MIN_SIGMA,
                     max_sigma=DOH_MAX_SIGMA,
                     num_sigma=DOH_NUM_SIGMA,
                     threshold=DOH_THRESHOLD)
# DoH radius ~ sigma (no sqrt(2) factor)

# =============================================================================
# COMPARE ALL THREE METHODS
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Top row: blob circles overlay
plot_blobs(axes[0, 0], img, blobs_log, f'LoG (σ={LOG_MIN_SIGMA}-{LOG_MAX_SIGMA}, t={LOG_THRESHOLD})', radius_multiplier=1.0)
plot_blobs(axes[0, 1], img, blobs_dog, f'DoG (σ={DOG_MIN_SIGMA}-{DOG_MAX_SIGMA}, t={DOG_THRESHOLD})', radius_multiplier=1.0)
plot_blobs(axes[0, 2], img, blobs_doh, f'DoH (σ={DOH_MIN_SIGMA}-{DOH_MAX_SIGMA}, t={DOH_THRESHOLD})', radius_multiplier=1.0)

# Bottom row: filled masks
masks_log = blobs_to_masks(blobs_log, img.shape, radius_multiplier=1.0)
masks_dog = blobs_to_masks(blobs_dog, img.shape, radius_multiplier=1.0)
masks_doh = blobs_to_masks(blobs_doh, img.shape, radius_multiplier=1.0)

for ax, masks, name in [(axes[1, 0], masks_log, 'LoG'),
                         (axes[1, 1], masks_dog, 'DoG'),
                         (axes[1, 2], masks_doh, 'DoH')]:
    ax.imshow(img, cmap='gray')
    n_rois = len(np.unique(masks)) - 1
    if n_rois > 0:
        masks_masked = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masks_masked, cmap='prism', alpha=0.4)
    ax.set_title(f'{name} masks ({n_rois} ROIs)')
    ax.axis('off')

plt.suptitle(f'Blob Detection Comparison - {input_name}', fontsize=12)
plt.tight_layout()
plt.savefig('blob_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Blob Counts ===")
print(f"LoG: {len(blobs_log)} blobs")
print(f"DoG: {len(blobs_dog)} blobs")
print(f"DoH: {len(blobs_doh)} blobs")

# %% ==========================================================================
# PARAMETER SWEEP - Find optimal settings
# ==========================================================================

# Sweep threshold values for LoG
thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, thresh in zip(axes, thresholds):
    blobs = blob_log(img, min_sigma=1, max_sigma=5, num_sigma=5, threshold=thresh)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    plot_blobs(ax, img, blobs, f'LoG thresh={thresh}', radius_multiplier=1.0)

plt.suptitle('LoG Threshold Sweep', fontsize=12)
plt.tight_layout()
plt.show()

# %% ==========================================================================
# SIGMA RANGE SWEEP
# ==========================================================================

sigma_ranges = [(0.5, 3), (1, 4), (1, 5), (2, 6), (2, 8), (3, 10)]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, (min_s, max_s) in zip(axes, sigma_ranges):
    blobs = blob_log(img, min_sigma=min_s, max_sigma=max_s, num_sigma=5, threshold=0.02)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    plot_blobs(ax, img, blobs, f'LoG σ={min_s}-{max_s}', radius_multiplier=1.0)

plt.suptitle('LoG Sigma Range Sweep (thresh=0.02)', fontsize=12)
plt.tight_layout()
plt.show()

# %% ==========================================================================
# BEST RESULT - Adjust parameters here after tuning
# ==========================================================================

# Your best parameters after tuning:
BEST_METHOD = "log"  # "log", "dog", or "doh"
BEST_MIN_SIGMA = 1
BEST_MAX_SIGMA = 4
BEST_THRESHOLD = 0.02

if BEST_METHOD == "log":
    best_blobs = blob_log(img, min_sigma=BEST_MIN_SIGMA, max_sigma=BEST_MAX_SIGMA,
                          num_sigma=5, threshold=BEST_THRESHOLD)
    best_blobs[:, 2] = best_blobs[:, 2] * np.sqrt(2)
elif BEST_METHOD == "dog":
    best_blobs = blob_dog(img, min_sigma=BEST_MIN_SIGMA, max_sigma=BEST_MAX_SIGMA,
                          threshold=BEST_THRESHOLD)
    best_blobs[:, 2] = best_blobs[:, 2] * np.sqrt(2)
else:
    best_blobs = blob_doh(img, min_sigma=BEST_MIN_SIGMA, max_sigma=BEST_MAX_SIGMA,
                          num_sigma=5, threshold=BEST_THRESHOLD)

best_masks = blobs_to_masks(best_blobs, img.shape, radius_multiplier=1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Circles
plot_blobs(axes[0], img, best_blobs, f'Best: {BEST_METHOD.upper()}', radius_multiplier=1.0)

# Masks
axes[1].imshow(img, cmap='gray')
n_rois = len(np.unique(best_masks)) - 1
if n_rois > 0:
    masks_masked = np.ma.masked_where(best_masks == 0, best_masks)
    axes[1].imshow(masks_masked, cmap='prism', alpha=0.4)
axes[1].set_title(f'Masks ({n_rois} ROIs)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nBest result: {len(best_blobs)} ROIs using {BEST_METHOD.upper()}")
print(f"Parameters: σ={BEST_MIN_SIGMA}-{BEST_MAX_SIGMA}, threshold={BEST_THRESHOLD}")

# %%
