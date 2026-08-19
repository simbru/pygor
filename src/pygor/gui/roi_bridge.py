"""Conversion between pygor ROI masks and napari Labels layers.

Pygor stores ROI masks in IGOR convention: background is 1 and each ROI
is a distinct negative integer (-1, -2, ...). Segmentation backends may
instead return positive labels on a 0 background. napari's Labels layer
needs positive integers with 0 as background, so both conventions are
normalised here.

Trace arrays are indexed 0-based in ROI order, so napari label ``n``
always maps to trace row ``n - 1``.
"""

import numpy as np


def is_igor_style(roi_mask):
    """Return True if mask uses the IGOR negative-label convention."""
    vals = np.unique(roi_mask)
    vals = vals[np.isfinite(vals)]
    return bool(np.all(np.logical_or(vals < 0, vals == 1)))


def mask_to_labels(roi_mask):
    """Convert a pygor ROI mask to a napari-compatible label image.

    Parameters
    ----------
    roi_mask : numpy.ndarray
        ROI mask in either IGOR (negative) or positive-label convention.

    Returns
    -------
    numpy.ndarray
        Label image of dtype int32, background 0, ROIs numbered from 1.
    """
    roi_mask = np.asarray(roi_mask)
    if is_igor_style(roi_mask):
        labels = np.where(roi_mask < 0, -roi_mask, 0)
    else:
        labels = np.where(roi_mask > 0, roi_mask, 0)
    return np.nan_to_num(labels, nan=0).astype(np.int32)


def labels_to_mask(labels, igor_style=True):
    """Convert a napari label image back to a pygor ROI mask.

    Parameters
    ----------
    labels : numpy.ndarray
        Label image, background 0, ROIs numbered from 1.
    igor_style : bool
        If True, emit the IGOR convention (background 1, ROIs negative).

    Returns
    -------
    numpy.ndarray
        ROI mask in the requested convention.
    """
    labels = np.asarray(labels).astype(np.int32)
    if not igor_style:
        return labels
    mask = np.where(labels > 0, -labels, 1)
    return mask.astype(np.int32)


def label_to_trace_index(label):
    """Map a napari label value to its row in the trace arrays."""
    return int(label) - 1


def trace_index_to_label(index):
    """Map a trace row index to its napari label value."""
    return int(index) + 1
