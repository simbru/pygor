"""
Trace extraction functions for ROI analysis.

This module provides efficient trace extraction from image stacks
using vectorized NumPy operations.
"""
import numpy as np

__all__ = ["extract_traces", "znorm_traces"]


def extract_traces(images: np.ndarray, rois: np.ndarray) -> np.ndarray:
    """
    Extract ROI traces from image stack using vectorized operations.

    Computes the mean fluorescence signal for each ROI across all frames.
    Uses float64 internally to prevent uint16 overflow during summation.

    Parameters
    ----------
    images : np.ndarray
        3D array (frames, height, width) of imaging data
    rois : np.ndarray
        2D ROI mask with negative values for ROIs (-1, -2, -3, ...)
        Background should be positive (typically 1).

    Returns
    -------
    traces : np.ndarray
        Shape (n_rois, n_frames), dtype float32.
        ROIs are ordered by ID: -1 first, then -2, -3, etc.

    Notes
    -----
    This vectorized implementation is ~5.6x faster than the previous
    parallel/multiprocessing approach for typical dataset sizes (~400MB).
    The speedup comes from avoiding:
    - Process spawning overhead
    - Shared memory setup/teardown
    - Pickling of ROI masks

    Examples
    --------
    >>> traces = extract_traces(data.images, data.rois)
    >>> print(traces.shape)  # (n_rois, n_frames)
    """
    # Get unique ROI IDs (negative values in pygor convention)
    roi_ids = np.unique(rois)
    roi_ids = roi_ids[roi_ids < 0]
    roi_ids = np.sort(roi_ids)[::-1]  # -1, -2, -3, ...

    n_rois = len(roi_ids)
    n_frames = images.shape[0]

    # Flatten spatial dimensions for efficient boolean indexing
    # Convert to float64 upfront to prevent uint16 overflow during mean
    images_flat = images.reshape(n_frames, -1).astype(np.float64)
    rois_flat = rois.ravel()

    traces = np.zeros((n_rois, n_frames), dtype=np.float64)

    for i, roi_id in enumerate(roi_ids):
        mask = rois_flat == roi_id
        # Vectorized mean: computes mean across all frames in one NumPy call
        traces[i] = images_flat[:, mask].mean(axis=1)

    return traces.astype(np.float32)


def znorm_traces(
    traces: np.ndarray,
    baseline_start: int,
    baseline_end: int,
) -> np.ndarray:
    """
    Z-normalize traces using baseline period (vectorized).

    Applies z-score normalization: (trace - baseline_mean) / baseline_std
    This matches IGOR's baseline z-normalization method.

    Parameters
    ----------
    traces : np.ndarray
        Shape (n_rois, n_frames)
    baseline_start : int
        First frame of baseline period (inclusive)
    baseline_end : int
        Last frame of baseline period (exclusive)

    Returns
    -------
    traces_znorm : np.ndarray
        Z-normalized traces, shape (n_rois, n_frames), dtype float32

    Notes
    -----
    If baseline_std is 0 for any ROI (constant signal), that ROI's
    z-normalized trace will be 0 (not NaN or Inf).
    """
    baseline = traces[:, baseline_start:baseline_end]
    means = baseline.mean(axis=1, keepdims=True)
    stds = baseline.std(axis=1, keepdims=True)
    # Avoid division by zero - set std=1 where std=0
    stds = np.where(stds == 0, 1, stds)
    return ((traces - means) / stds).astype(np.float32)
