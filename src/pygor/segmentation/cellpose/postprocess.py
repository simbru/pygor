"""
Post-processing heuristics for Cellpose segmentation results.

These functions improve segmentation quality by:
- Splitting large ROIs that may contain merged objects
- Shrinking ROI boundaries to reduce signal contamination
"""

import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import binary_erosion


def split_large_rois(
    masks,
    image,
    size_multiplier=1.25,
    min_distance=1,
    min_size_after_split=4,
):
    """Split large ROIs that may contain multiple merged objects.

    Uses watershed segmentation seeded by local intensity maxima within
    candidate ROIs. ROIs larger than (median_size * size_multiplier) are
    candidates for splitting.

    Parameters
    ----------
    masks : ndarray
        Cellpose-format mask (background=0, ROIs=1,2,3...)
    image : ndarray
        Original intensity image (used to find local maxima)
    size_multiplier : float
        ROIs larger than median_size * this value are candidates
    min_distance : int
        Minimum pixel distance between peaks to consider them distinct
    min_size_after_split : int
        Minimum size for split fragments (smaller ones are discarded)

    Returns
    -------
    new_masks : ndarray
        Updated mask with split ROIs
    n_splits : int
        Number of ROIs that were split
    """
    new_masks = masks.copy()
    roi_ids = np.unique(masks)
    roi_ids = roi_ids[roi_ids > 0]  # exclude background

    if len(roi_ids) == 0:
        return new_masks, 0

    # Calculate median ROI size
    roi_sizes = [(masks == roi_id).sum() for roi_id in roi_ids]
    median_size = np.median(roi_sizes)
    size_threshold = median_size * size_multiplier

    max_label = masks.max()
    n_splits = 0

    for roi_id in roi_ids:
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()

        if roi_size <= size_threshold:
            continue

        # Extract intensity within this ROI
        roi_image = image * roi_mask

        # Find local maxima within the ROI
        coords = peak_local_max(
            roi_image,
            min_distance=min_distance,
            labels=roi_mask.astype(int),
            num_peaks_per_label=10,  # allow multiple peaks per ROI
        )

        if len(coords) <= 1:
            # Only one peak found, no split needed
            continue

        # Create markers for watershed
        markers = np.zeros_like(masks, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            markers[y, x] = i + 1

        # Run watershed on inverted image (watershed finds basins, we want peaks)
        split_labels = watershed(-roi_image, markers, mask=roi_mask)

        # Check if split produced valid fragments
        split_ids = np.unique(split_labels)
        split_ids = split_ids[split_ids > 0]

        if len(split_ids) <= 1:
            continue

        # Validate fragment sizes and relabel
        valid_splits = []
        for split_id in split_ids:
            fragment_mask = split_labels == split_id
            fragment_size = fragment_mask.sum()
            if fragment_size >= min_size_after_split:
                valid_splits.append((split_id, fragment_mask))

        if len(valid_splits) <= 1:
            # After size filtering, only one valid fragment remains
            continue

        # Clear the original ROI from new_masks
        new_masks[roi_mask] = 0

        # Assign new labels to valid fragments
        for _, fragment_mask in valid_splits:
            max_label += 1
            new_masks[fragment_mask] = max_label

        n_splits += 1

    return new_masks, n_splits


def shrink_rois(masks, iterations=1, size_threshold=30):
    """Shrink ROI boundaries by erosion.

    Uniformly shrinks all ROIs above a size threshold to reduce potential
    signal contamination from neighboring structures.

    Parameters
    ----------
    masks : ndarray
        Cellpose-format mask (background=0, ROIs=1,2,3...)
    iterations : int
        Number of pixels to erode from boundaries (0 to disable)
    size_threshold : int
        Only shrink ROIs with at least this many pixels (smaller ones kept as-is)

    Returns
    -------
    new_masks : ndarray
        Mask with shrunk ROIs
    """
    if iterations <= 0:
        return masks.copy()

    new_masks = np.zeros_like(masks)
    for roi_id in np.unique(masks):
        if roi_id == 0:
            continue
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()

        if roi_size < size_threshold:
            # Keep small ROIs unchanged
            new_masks[roi_mask] = roi_id
        else:
            # Shrink larger ROIs
            shrunk = binary_erosion(roi_mask, iterations=iterations)
            if shrunk.sum() > 0:
                new_masks[shrunk] = roi_id
            else:
                # Erosion eliminated the ROI, keep original
                new_masks[roi_mask] = roi_id
    return new_masks
