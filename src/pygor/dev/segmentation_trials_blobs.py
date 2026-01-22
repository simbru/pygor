#%%
"""
Blob Detection for Synaptic Terminals using Difference of Gaussian (DoG).
"""
import pathlib
import os
import h5py
os.chdir(pathlib.Path(__file__).parent.parent.parent.parent)

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
from skimage.filters import unsharp_mask, threshold_otsu
from scipy.ndimage import label, binary_erosion

import pygor.load
from pygor.segmentation.lightweight import prepare_image

#%%
# =============================================================================
# PARAMETERS - All tunable settings in one place
# =============================================================================

EXAMPLE_PATH = r".\examples\FullFieldFlash_4_colour_demo.smp"
EXAMPLE_PATH = r"D:\Igor analyses\OSDS\251103 OSDS\0_0_SWN_200_White.smp"
EXAMPLE_PATH = r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp"
CUSTOM_CONFIG = r".\configs\example.toml"
NOISE_PATH = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"

# load up noise array
with h5py.File(NOISE_PATH, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"][:30000])  # Use subset for speed

#%%
obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG)
obj.preprocess(detrend=False)
obj.register(plot=True, mode="reflect")

#%%
obj.compute_correlation_projection(timecompress = 2, binpix = 2, force_recompute = True)
plt.imshow(obj.correlation_projection, cmap='gray')

#%%
import importlib
import pygor.segmentation.plotting
importlib.reload(pygor.segmentation.plotting)

masks = obj.segment_rois(input_mode = "combined",
                        
                        mode = "blob", roi_order = "LR", plot = True)
obj.view_stack_rois()

#%%

obj.extract_traces_from_rois()
obj.calculate_strf(noise_array=noise_array,
        sta_past_window=2, sta_future_window=0, n_jobs=-1);

obj.plot_strfs_space()

# %%

# # Image preparation
# INPUT_MODE = "combined"      # correlation × average works best
# UNSHARP_RADIUS = 1.0         # Smaller = finer sharpening (0.5-2.0)
# UNSHARP_AMOUNT = 2.5         # Strength of sharpening (1.0-5.0)

# # Blob detection
# MIN_SIGMA = 1
# MAX_SIGMA = 2
# THRESHOLD = 0.01             # Lower = catch weaker blobs
# ELIMINATE_OVERLAP = 1                  # Fraction overlap before DoG merges
# MERGE_OVERLAP = 0.6          # Merge if overlap > 30% of smaller blob (1.0 to disable)
# # Anatomy mask (exclude border regions)
# ANATOMY_THRESHOLD = 'otsu'   # 'otsu' for auto, or float (e.g., 0.1) for manual
# ANATOMY_THRESH_MULT = 0.2    # Multiply threshold by this (lower = more permissive)
# ERODE_ITERATIONS = 1         # Push boundary inward to exclude edge blobs

# # Mask creation
# RADIUS_MULTIPLIER = 1.5      # Scale factor for blob radius
# MIN_RADIUS = 1               # Prevents single-pixel ROIs


# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def normalize(img):
#     """Rescale image to [0, 1]."""
#     return (img - img.min()) / (img.max() - img.min())


# def create_anatomy_mask(img, method='otsu', thresh_mult=1.0, erode_iterations=2):
#     """
#     Create a binary mask of the anatomy region.

#     Parameters
#     ----------
#     img : ndarray
#         Input image (normalized)
#     method : str or float
#         'otsu' for automatic thresholding, or float for manual threshold
#     thresh_mult : float
#         Multiply threshold by this factor (lower = more permissive, keeps more)
#     erode_iterations : int
#         Erode mask to exclude border regions (0 to disable)

#     Returns
#     -------
#     mask : ndarray (bool)
#         True where blobs are allowed
#     """
#     if method == 'otsu':
#         thresh = threshold_otsu(img)
#     else:
#         thresh = float(method)

#     thresh *= thresh_mult
#     mask = img > thresh

#     # Erode to push boundary inward (exclude edge blobs)
#     if erode_iterations > 0:
#         mask = binary_erosion(mask, iterations=erode_iterations)

#     return mask


# def filter_blobs_by_mask(blobs, mask):
#     """
#     Keep only blobs whose center falls within the mask.

#     Parameters
#     ----------
#     blobs : ndarray
#         Array of (y, x, radius) for each blob
#     mask : ndarray (bool)
#         Binary mask where True = allowed region

#     Returns
#     -------
#     filtered_blobs : ndarray
#         Blobs with centers inside the mask
#     """
#     if len(blobs) == 0:
#         return blobs

#     centers_y = blobs[:, 0].astype(int)
#     centers_x = blobs[:, 1].astype(int)

#     # Clip to image bounds
#     centers_y = np.clip(centers_y, 0, mask.shape[0] - 1)
#     centers_x = np.clip(centers_x, 0, mask.shape[1] - 1)

#     valid = mask[centers_y, centers_x]
#     return blobs[valid]


# def detect_blobs(img, min_sigma, max_sigma, threshold, overlap, artifact_width=0):
#     """
#     Detect blobs using DoG and convert sigma to radius.
    
#     Returns array of (y, x, radius) for each blob.
#     """
#     blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma,
#                      threshold=threshold, overlap=overlap)
    
#     if len(blobs) == 0:
#         return blobs
    
#     # DoG returns sigma, convert to radius: radius = sqrt(2) * sigma
#     blobs = blobs.copy()
#     blobs[:, 2] *= np.sqrt(2)
    
#     # Filter out blobs in artifact region
#     if artifact_width > 0:
#         blobs = blobs[blobs[:, 1] >= artifact_width]
    
#     return blobs


# def blobs_to_masks(blobs, img_shape, radius_multiplier=1.0, min_radius=2, merge_overlap=0.3):
#     """
#     Convert blob detections to labeled mask array, merging overlapping blobs.
    
#     Uses scipy.ndimage.label for connected component labeling after drawing
#     all circles, then splits components based on overlap threshold.
#     """
#     if len(blobs) == 0:
#         return np.zeros(img_shape, dtype=np.int32)
    
#     centers = blobs[:, :2]
#     radii = np.maximum(min_radius, radius_multiplier * blobs[:, 2])
    
#     # Create distance grids once
#     yy, xx = np.ogrid[:img_shape[0], :img_shape[1]]
    
#     # Draw each blob as a separate temporary mask, then merge based on overlap
#     n_blobs = len(blobs)
#     blob_masks = np.zeros((n_blobs, *img_shape), dtype=bool)
    
#     for i, ((cy, cx), r) in enumerate(zip(centers, radii)):
#         blob_masks[i] = (yy - cy)**2 + (xx - cx)**2 <= r**2
    
#     # Find which blobs should merge (union-find via adjacency)
#     parent = np.arange(n_blobs)
    
#     def find(i):
#         while parent[i] != i:
#             parent[i] = parent[parent[i]]  # Path compression
#             i = parent[i]
#         return i
    
#     # Check overlap between blob pairs
#     for i in range(n_blobs):
#         for j in range(i + 1, n_blobs):
#             intersection = blob_masks[i] & blob_masks[j]
#             if not intersection.any():
#                 continue
            
#             overlap_pixels = intersection.sum()
#             smaller_area = min(blob_masks[i].sum(), blob_masks[j].sum())
            
#             if overlap_pixels / smaller_area >= merge_overlap:
#                 parent[find(i)] = find(j)
    
#     # Build final masks by group
#     masks = np.zeros(img_shape, dtype=np.int32)
#     group_labels = {}
#     current_label = 0
    
#     for i in range(n_blobs):
#         root = find(i)
#         if root not in group_labels:
#             current_label += 1
#             group_labels[root] = current_label
#         masks[blob_masks[i]] = group_labels[root]
    
#     return masks


# def plot_results(img, blobs, masks, params, anatomy_mask=None):
#     """Display input image, detected blobs, and final masks."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # Apply anatomy mask to image for display (mask out background)
#     if anatomy_mask is not None:
#         img_display = np.ma.masked_where(~anatomy_mask, img)
#     else:
#         img_display = img

#     # Input
#     axes[0].imshow(img_display, cmap='gray')
#     axes[0].set_title(f"Input: {params['input_mode']}")

#     # Blob circles
#     axes[1].imshow(img_display, cmap='gray')
#     for y, x, r in blobs:
#         circle = plt.Circle((x, y), r * params['radius_mult'],
#                            color='red', fill=False, linewidth=1)
#         axes[1].add_patch(circle)
#     axes[1].set_title(f"DoG Blobs ({len(blobs)})\n"
#                       f"σ={params['min_sigma']}-{params['max_sigma']}, "
#                       f"thresh={params['threshold']}")

#     # Masks (show full image, not masked)
#     n_rois = masks.max()
#     axes[2].imshow(img, cmap='gray')
#     if n_rois > 0:
#         masked = np.ma.masked_where(masks == 0, masks)
#         axes[2].imshow(masked, cmap='prism', alpha=0.4)
#     axes[2].set_title(f"ROI Masks ({n_rois})\nmin_r={params['min_radius']}")

#     for ax in axes:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()


# def print_roi_stats(masks):
#     """Print size statistics for ROIs."""
#     n_rois = masks.max()
#     if n_rois == 0:
#         print("No ROIs detected")
#         return
    
#     roi_sizes = np.bincount(masks.ravel())[1:]  # Skip background
#     print(f"\nROI size statistics ({n_rois} ROIs):")
#     print(f"  Min: {roi_sizes.min()} px | Max: {roi_sizes.max()} px")
#     print(f"  Mean: {roi_sizes.mean():.1f} px | Median: {np.median(roi_sizes):.1f} px")


# # =============================================================================
# # MAIN WORKFLOW
# # =============================================================================

# # Prepare and enhance image
# img, artifact_width = prepare_image(obj, input_mode=INPUT_MODE)
# img = normalize(unsharp_mask(img, radius=UNSHARP_RADIUS, amount=UNSHARP_AMOUNT))

# print(f"Input: {INPUT_MODE} | Unsharp: r={UNSHARP_RADIUS}, a={UNSHARP_AMOUNT}")
# print(f"Artifact mask: columns 0-{artifact_width - 1}")


# # Create anatomy mask to exclude border regions
# anatomy_mask = create_anatomy_mask(img, method=ANATOMY_THRESHOLD, thresh_mult=ANATOMY_THRESH_MULT, erode_iterations=ERODE_ITERATIONS)

# # Detect blobs and filter by anatomy mask
# blobs_raw = detect_blobs(img, MIN_SIGMA, MAX_SIGMA, THRESHOLD, ELIMINATE_OVERLAP, artifact_width)
# blobs = filter_blobs_by_mask(blobs_raw, anatomy_mask)
# masks = blobs_to_masks(blobs, img.shape, RADIUS_MULTIPLIER, MIN_RADIUS, MERGE_OVERLAP)

# print(f"Detected {len(blobs_raw)} blobs → filtered to {len(blobs)} (removed {len(blobs_raw) - len(blobs)} border blobs) → {masks.max()} ROIs")


# # Visualize results
# plot_results(img, blobs, masks, {
#     'input_mode': INPUT_MODE,
#     'min_sigma': MIN_SIGMA,
#     'max_sigma': MAX_SIGMA,
#     'threshold': THRESHOLD,
#     'radius_mult': RADIUS_MULTIPLIER,
#     'min_radius': MIN_RADIUS,
# }, anatomy_mask=anatomy_mask)

# print_roi_stats(masks)


# obj.rois = masks

# plt.imshow(obj.images.mean(axis=0))
# plt.imshow(obj.rois_alt, cmap='prism')

# # %%
# # Visualize anatomy mask for tuning
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Input image')
# axes[1].imshow(anatomy_mask, cmap='gray')
# axes[1].set_title(f'Anatomy mask\nthresh={ANATOMY_THRESHOLD}, erode={ERODE_ITERATIONS}')
# axes[2].imshow(img, cmap='gray')
# axes[2].contour(anatomy_mask, colors='red', linewidths=1)
# axes[2].set_title('Mask boundary overlay')
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
# # %%
