"""
Mask format conversion utilities.

Cellpose uses a specific mask format:
- Background = 0
- ROIs = 1, 2, 3, ...

Pygor uses a different format for storage:
- Background = 1
- ROIs = -1, -2, -3, ...

This module handles conversion between these formats and is shared
across all segmentation methods.
"""

import numpy as np
from scipy.ndimage import label


def to_pygor(cellpose_mask):
    """Convert Cellpose-style mask to pygor format.

    Parameters
    ----------
    cellpose_mask : ndarray
        Mask in Cellpose format (background=0, ROIs=1,2,3...)

    Returns
    -------
    pygor_mask : ndarray
        Mask in pygor format (background=1, ROIs=-1,-2,-3...)
    """
    pygor_mask = cellpose_mask.copy().astype(np.int16)

    # Convert ROIs to negative (do this first while we can identify them)
    roi_pixels = pygor_mask > 0
    pygor_mask[roi_pixels] = -pygor_mask[roi_pixels]

    # Convert background (0) to 1
    pygor_mask[pygor_mask == 0] = 1

    return pygor_mask


def from_pygor(pygor_mask):
    """Convert pygor mask to Cellpose-style format.

    Parameters
    ----------
    pygor_mask : ndarray
        Mask in pygor format (background=1, ROIs=-1,-2,-3...)

    Returns
    -------
    cellpose_mask : ndarray
        Mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
    if pygor_mask is None:
        return None

    cellpose_mask = pygor_mask.copy()
    # Convert background (1) to 0
    cellpose_mask[cellpose_mask == 1] = 0
    # Convert negative ROI labels to positive
    cellpose_mask = np.abs(cellpose_mask)

    return cellpose_mask.astype(np.uint16)


def from_pygor_relabeled(pygor_mask):
    """Convert pygor mask to Cellpose-style format with sequential relabeling.

    Like from_pygor() but ensures ROI labels are sequential (1, 2, 3, ...)
    by relabeling connected components. Useful for training data preparation.

    Parameters
    ----------
    pygor_mask : ndarray
        Mask in pygor format

    Returns
    -------
    cellpose_mask : ndarray
        Mask in Cellpose format with sequential labels
    n_rois : int
        Number of ROIs found
    """
    if pygor_mask is None:
        return None, 0

    # Create binary mask: foreground (any ROI) vs background
    binary_mask = (pygor_mask != 1).astype(np.uint8)

    # Relabel connected components to get sequential 1,2,3...
    labeled_mask, n_components = label(binary_mask)

    return labeled_mask.astype(np.uint16), n_components


def relabel_pygor_consecutive(pygor_mask, roi_order="LR"):
    """Ensure ROI labels in a pygor mask are consecutive -1, -2, -3...

    After post-processing (splitting, merging), ROI labels may have gaps.
    This function relabels them to be consecutive, optionally with spatial ordering.

    Parameters
    ----------
    pygor_mask : ndarray
        Mask in pygor format (may have non-consecutive labels)
    roi_order : str or None
        Spatial ordering mode for ROI numbering:
        - "LR": Left-to-right (x primary, y as tiebreaker) - default
        - "TB": Top-to-bottom (y primary, x as tiebreaker)
        - None: Original detection order (numerical order from np.unique)

    Returns
    -------
    relabeled_mask : ndarray
        Mask with consecutive ROI labels ordered spatially
    """
    from scipy.ndimage import center_of_mass

    unique_labels = np.unique(pygor_mask)
    unique_labels = unique_labels[unique_labels != 1]  # Remove background

    if len(unique_labels) == 0:
        return pygor_mask

    if roi_order is not None:
        # Calculate centroid for each ROI
        centroids = []
        for lbl in unique_labels:
            # center_of_mass returns (y, x) for 2D arrays
            cy, cx = center_of_mass(pygor_mask == lbl)
            centroids.append((lbl, cx, cy))

        # Sort by spatial position
        if roi_order == "LR":
            # Primary: x (left-to-right), Secondary: y (top-to-bottom)
            centroids.sort(key=lambda c: (c[1], c[2]))
        elif roi_order == "TB":
            # Primary: y (top-to-bottom), Secondary: x (left-to-right)
            centroids.sort(key=lambda c: (c[2], c[1]))

        unique_labels = np.array([c[0] for c in centroids])

    # Remap to consecutive -1, -2, -3...
    remapped = pygor_mask.copy()
    for i, old_label in enumerate(unique_labels, start=1):
        remapped[pygor_mask == old_label] = -i

    return remapped
