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


def roi_ids_in_order(roi_mask):
    """Return ROI ids in the order ``extract_traces`` emits their rows.

    IGOR-style masks are ordered -1, -2, -3, ...; positive-label masks are
    ordered 1, 2, 3, ... Either way the position in this list is the row
    the ROI occupies in the trace arrays.
    """
    ids = np.unique(np.asarray(roi_mask))
    ids = ids[np.isfinite(ids)]
    if is_igor_style(roi_mask):
        ids = ids[ids < 0]
        return np.sort(ids)[::-1]
    ids = ids[ids > 0]
    return np.sort(ids)


def label_to_trace_index(label, roi_mask=None):
    """Map a napari label to its row in the trace arrays.

    Rows are packed in ROI order with no gaps, so erasing an ROI shifts
    every later row down by one. Without the mask to compare against, the
    labels are assumed contiguous and ``label - 1`` is used.

    Returns -1 when the label has no row, which covers a freshly drawn ROI
    that has not been extracted yet.
    """
    label = int(label)
    if roi_mask is None:
        return label - 1
    ids = roi_ids_in_order(roi_mask)
    target = -label if is_igor_style(roi_mask) else label
    matches = np.flatnonzero(ids == target)
    if not matches.size:
        return -1
    return int(matches[0])


def trace_index_to_label(index, roi_mask=None):
    """Map a trace row index to its napari label value."""
    index = int(index)
    if roi_mask is None:
        return index + 1
    ids = roi_ids_in_order(roi_mask)
    if index < 0 or index >= len(ids):
        return 0
    return int(abs(ids[index]))


def sync_rois_from_layer(recording, labels_layer):
    """Write layer edits back to the recording, if there are any.

    Returns True when the recording's mask was updated. Analysis reads
    ``recording.rois``, not the layer, so anything drawn by hand has to be
    pushed across before it can be measured.
    """
    if labels_layer is None:
        return False
    current = getattr(recording, "rois", None)
    igor = True if current is None else is_igor_style(current)
    mask = labels_to_mask(labels_layer.data, igor_style=igor)
    if current is not None and np.array_equal(mask, np.asarray(current)):
        return False
    recording.update_rois(mask)
    return True
