"""
Registration (motion correction) for calcium imaging stacks.

This module provides frame registration using phase cross-correlation,
optimized for low-SNR calcium imaging data. The implementation uses
batch averaging to improve registration quality.

Key Features
------------
- Batch-averaged registration for improved SNR
- Phase cross-correlation with subpixel precision
- Parallel processing for large stacks
- Per-batch shift computation with interpolation to all frames

Reference
---------
Based on batch-averaged registration approach, which computes shifts
on temporally averaged chunks of the stack rather than individual frames.
This dramatically improves registration quality for noisy calcium imaging.

Note this module is largely written by Claude Code but heavily vetted by me(:
"""

import numpy as np
import warnings
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os

from scipy.ndimage import shift as scipy_shift, fourier_shift
from scipy.fft import fft2, ifft2
from skimage.registration import phase_cross_correlation


__all__ = [
    "register_stack",
    "compute_batch_shifts",
    "apply_shifts_to_stack",
]

# Detect available parallelization backends
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


def _shift_frame(frame: np.ndarray, shift_yx: np.ndarray, order: int, mode: str) -> np.ndarray:
    """Shift a single frame (helper for parallel processing)."""
    return scipy_shift(frame, shift=shift_yx, order=order, mode=mode)


def _shift_frame_fft(frame: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """Shift a single frame using FFT (faster for large frames)."""
    freq = fft2(frame)
    shifted_freq = fourier_shift(freq, shift_yx)
    return np.real(ifft2(shifted_freq))


def _compute_frame_shifts(
    n_frames: int,
    n_batches: int,
    batch_size: int,
    shifts: np.ndarray,
    interpolate: bool,
) -> np.ndarray:
    """
    Compute per-frame shifts from batch shifts.

    Parameters
    ----------
    n_frames : int
        Total number of frames
    n_batches : int
        Number of batches
    batch_size : int
        Frames per batch
    shifts : ndarray
        Per-batch shifts (n_batches, 2)
    interpolate : bool
        If True, linearly interpolate between batch centers.
        If False, use step function (all frames in batch get same shift).

    Returns
    -------
    frame_shifts : ndarray
        Per-frame shifts (n_frames, 2)
    """
    frame_shifts = np.zeros((n_frames, 2))

    if interpolate and n_batches > 1:
        # Compute batch center frame indices
        batch_centers = np.array([
            (i * batch_size + min((i + 1) * batch_size, n_frames)) // 2
            for i in range(n_batches)
        ])

        # Interpolate shifts for each axis
        frame_indices = np.arange(n_frames)
        frame_shifts[:, 0] = np.interp(frame_indices, batch_centers, shifts[:, 0])
        frame_shifts[:, 1] = np.interp(frame_indices, batch_centers, shifts[:, 1])
    else:
        # Step function: all frames in batch get same shift
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_frames)
            frame_shifts[start:end] = shifts[i]

    return frame_shifts


def _compute_projection(stack: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute projection of stack along time axis.

    Parameters
    ----------
    stack : ndarray
        3D stack (time, height, width)
    mode : str
        Projection mode: "mean", "std", "var", "median", "max"

    Returns
    -------
    projection : ndarray
        2D projection (height, width)
    """
    if mode == "mean":
        return stack.mean(axis=0)
    elif mode == "std":
        return stack.std(axis=0)
    elif mode == "var":
        return stack.var(axis=0)
    elif mode == "median":
        return np.median(stack, axis=0)
    elif mode == "max":
        return stack.max(axis=0)
    else:
        raise ValueError(
            f"Unknown projection mode: '{mode}'. "
            f"Options: 'mean', 'std', 'var', 'median', 'max'"
        )


def register_stack(
    stack: np.ndarray,
    n_reference_frames: int = 1000,
    artifact_width: int = 0,
    batch_size: int = 10,
    upsample_factor: int = 10,
    normalization: Optional[str] = None,
    order: int = 1,
    mode: str = "reflect",
    return_shifts: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
    interpolate: bool = True,
    batch_mode: str = "std",
    reference_mode: str = "mean",
    edge_crop: int = 0,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Register imaging stack using batch-averaged phase cross-correlation.

    This function performs motion correction by:
    1. Creating a stable reference from the first n_reference_frames
    2. Dividing the stack into batches and averaging each batch
    3. Computing shifts for each batch via phase cross-correlation
    4. Applying shifts to all frames in each batch

    The batch averaging dramatically improves registration quality for
    low-SNR calcium imaging data.

    Parameters
    ----------
    stack : ndarray
        3D imaging stack (time, height, width)
    n_reference_frames : int, optional
        Number of initial frames to average for reference (default: 1000)
    artifact_width : int, optional
        Number of pixels to exclude from left edge during shift computation
        (default: 0). This region is preserved but not used for registration.
        Set to match preprocessing artifact_width to exclude the light artifact.
    batch_size : int, optional
        Number of frames to average per batch (default: 10)
    upsample_factor : int, optional
        Subpixel precision factor for phase correlation (default: 10)
    normalization : str or None, optional
        Phase correlation normalization mode (default: None).
        For low-SNR data, None is recommended. Use 'phase' for high-SNR.
    order : int, optional
        Spline interpolation order for shifting (0-5, default: 1).
        Higher orders are smoother but slower. 0=nearest, 1=linear.
    mode : str, optional
        Edge handling mode for shifting (default: 'reflect').
        Options: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
    return_shifts : bool, optional
        If True, return (registered, shifts, errors) (default: False)
    parallel : bool, optional
        Use parallel processing for faster shifting (default: True).
        Uses joblib if available, otherwise ThreadPoolExecutor.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: -1)
    interpolate : bool, optional
        If True (default), linearly interpolate shifts between batch centers
        for smooth frame-to-frame transitions. If False, all frames in a
        batch get the same shift (step/square-wave behavior).
    batch_mode : str, optional
        Projection mode for batch images (default: "std").
        Options: "mean", "std", "var", "median", "max".
        Std captures morphology better and is less affected by
        temporal brightness fluctuations.
    reference_mode : str, optional
        Projection mode for reference image (default: "mean").
        Mean over many frames gives clean, stable structure.
    edge_crop : int, optional
        Pixels to crop from all edges before cross-correlation (default: 0).
        Useful to exclude edge artifacts from shift computation.
        Does not affect the output dimensions.

    Returns
    -------
    registered : ndarray
        Registered stack (same shape as input)
    shifts : ndarray (if return_shifts=True)
        Per-batch shifts (n_batches, 2) as (shift_y, shift_x)
    errors : ndarray (if return_shifts=True)
        Per-batch registration errors (lower is better)

    Examples
    --------
    >>> # Basic registration
    >>> registered = register_stack(stack)
    >>>
    >>> # Registration with shift information
    >>> registered, shifts, errors = register_stack(stack, return_shifts=True)
    >>> print(f"Mean shift: {shifts.mean(axis=0)}")
    >>> print(f"Mean error: {errors.mean():.4f}")
    >>>
    >>> # Custom parameters for faster processing
    >>> registered = register_stack(stack, batch_size=20, upsample_factor=5)

    Notes
    -----
    - For low-SNR calcium imaging, normalization=None is crucial
    - Larger batch_size trades temporal resolution for better shift estimates
    - Higher upsample_factor increases precision but slows computation
    - Preprocessing should be applied before registration to handle artifacts

    See Also
    --------
    compute_batch_shifts : Compute shifts without applying them
    apply_shifts_to_stack : Apply pre-computed shifts to stack
    """
    if artifact_width > 0:
        # Store original artifact region to restore after registration
        artifact_region = stack[:, :, :artifact_width].copy()
        stack = stack[:, :, artifact_width:]

    # Compute shifts
    shifts, errors = compute_batch_shifts(
        stack=stack,
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        normalization=normalization,
        batch_mode=batch_mode,
        reference_mode=reference_mode,
        edge_crop=edge_crop,
    )

    # Apply shifts
    registered = apply_shifts_to_stack(
        stack=stack,
        shifts=shifts,
        batch_size=batch_size,
        order=order,
        mode=mode,
        parallel=parallel,
        n_jobs=n_jobs,
        interpolate=interpolate,
    )

    # Restore original artifact region (unregistered, as it's already corrected)
    if artifact_width > 0:
        registered = np.concatenate((artifact_region, registered), axis=2)

    if return_shifts:
        return registered, shifts, errors
    else:
        return registered


def compute_batch_shifts(
    stack: np.ndarray,
    n_reference_frames: int = 1000,
    batch_size: int = 10,
    upsample_factor: int = 10,
    normalization: Optional[str] = None,
    batch_mode: str = "std",
    reference_mode: str = "mean",
    edge_crop: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute registration shifts for batches of frames.

    Parameters
    ----------
    stack : ndarray
        3D imaging stack (time, height, width)
    n_reference_frames : int, optional
        Number of initial frames to average for reference
    batch_size : int, optional
        Number of frames to average per batch
    upsample_factor : int, optional
        Subpixel precision factor
    normalization : str or None, optional
        Phase correlation normalization mode
    batch_mode : str, optional
        Projection mode for batch images (default: "std").
        Options: "mean", "std", "var", "median", "max".
        Std captures morphology better and is less affected by
        temporal brightness fluctuations.
    reference_mode : str, optional
        Projection mode for reference image (default: "mean").
        Mean over many frames gives clean, stable structure.
    edge_crop : int, optional
        Pixels to crop from all edges before cross-correlation (default: 0).
        Useful to exclude edge artifacts from shift computation.
        Does not affect the output dimensions.

    Returns
    -------
    shifts : ndarray
        Per-batch shifts (n_batches, 2) as (shift_y, shift_x)
    errors : ndarray
        Per-batch registration errors (n_batches,)

    Notes
    -----
    This function only computes shifts without applying them.
    Use apply_shifts_to_stack() to actually shift the frames.
    """
    # Handle TOML "None" string (TOML doesn't support Python None)
    if normalization in ("None", "none", "", None):
        normalization = None

    n_frames = len(stack)
    n_batches = int(np.ceil(n_frames / batch_size))

    # Create reference from initial frames
    reference = _compute_projection(stack[:n_reference_frames], reference_mode)

    # Apply edge cropping for shift computation
    if edge_crop > 0:
        reference = reference[edge_crop:-edge_crop, edge_crop:-edge_crop]

    # Compute shifts per batch
    shifts = []
    errors = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_frames)

        # Compute batch projection
        batch_proj = _compute_projection(stack[start:end], batch_mode)

        # Apply edge cropping for shift computation
        if edge_crop > 0:
            batch_proj = batch_proj[edge_crop:-edge_crop, edge_crop:-edge_crop]

        # Compute shift
        shift_yx, error, _ = phase_cross_correlation(
            reference,
            batch_proj,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )

        shifts.append(shift_yx)
        errors.append(error)

    return np.array(shifts), np.array(errors)


def apply_shifts_to_stack(
    stack: np.ndarray,
    shifts: np.ndarray,
    batch_size: int = 10,
    order: int = 1,
    mode: str = "reflect",
    parallel: bool = True,
    n_jobs: int = -1,
    interpolate: bool = True,
) -> np.ndarray:
    """
    Apply pre-computed shifts to imaging stack.

    Parameters
    ----------
    stack : ndarray
        3D imaging stack (time, height, width)
    shifts : ndarray
        Per-batch shifts (n_batches, 2) as (shift_y, shift_x)
    batch_size : int, optional
        Number of frames per batch (default: 10)
    order : int, optional
        Spline interpolation order for shifting (0-5, default: 1).
        Higher orders are smoother but slower. 0=nearest, 1=linear.
    mode : str, optional
        Edge handling mode for shifting (default: 'reflect').
        Options:
        - 'reflect': Reflects at edge, duplicating the edge pixel
        - 'constant': Pads with zeros
        - 'nearest': Extends with the nearest edge pixel value
        - 'mirror': Reflects at edge without duplicating the edge pixel
        - 'wrap': Wraps around to the opposite edge
    parallel : bool, optional
        Use parallel processing for faster shifting (default: True).
        Uses joblib if available, otherwise ThreadPoolExecutor.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: -1)
    interpolate : bool, optional
        If True (default), linearly interpolate shifts between batch centers
        for smooth frame-to-frame transitions. If False, all frames in a
        batch get the same shift (step/square-wave behavior).

    Returns
    -------
    registered : ndarray
        Registered stack (same shape as input)

    Notes
    -----
    - When interpolate=True, shifts are assigned to batch centers and linearly
      interpolated to all frames for smooth transitions
    - When interpolate=False, each batch of frames gets the same shift applied
    - Uses scipy.ndimage.shift with spline interpolation for subpixel accuracy
    """
    n_frames = len(stack)
    n_batches = len(shifts)

    # Store original data range for clipping
    original_min = stack.min()
    original_max = stack.max()

    # Build per-frame shift array
    frame_shifts = _compute_frame_shifts(n_frames, n_batches, batch_size, shifts, interpolate)

    # Choose processing path
    if parallel and n_frames > 100:
        if _HAS_JOBLIB:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_shift_frame)(stack[i], frame_shifts[i], order, mode)
                for i in range(n_frames)
            )
            registered = np.array(results)
        else:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    lambda i: _shift_frame(stack[i], frame_shifts[i], order, mode),
                    range(n_frames)
                ))
            registered = np.array(results)

        # Clip at end
        registered = np.clip(registered, original_min, original_max)

    else:
        # Sequential path
        registered = np.zeros_like(stack)

        for i in range(n_frames):
            shifted = scipy_shift(stack[i], shift=frame_shifts[i], order=order, mode=mode)
            registered[i] = np.clip(shifted, original_min, original_max)

    return registered


def transfer_rois(
    roi_mask: np.ndarray,
    ref_projection: np.ndarray,
    target_projection: np.ndarray,
    max_shift: int = 20,
    upsample_factor: int = 10,
) -> Tuple[np.ndarray, dict]:
    """
    Transfer ROI mask from one recording to another using registration.

    This function registers two recordings to each other and applies
    the resulting transform to transfer an ROI mask from the reference
    to the target recording.

    Parameters
    ----------
    roi_mask : ndarray
        2D ROI mask from reference recording (background=1, ROIs=-1,-2,...)
    ref_projection : ndarray
        Mean projection of reference recording
    target_projection : ndarray
        Mean projection of target recording
    max_shift : int, optional
        Maximum expected shift in pixels (default: 20)
    upsample_factor : int, optional
        Subpixel precision factor (default: 10)

    Returns
    -------
    shifted_mask : ndarray
        ROI mask aligned to target recording
    transform : dict
        Transform information with keys:
        - 'shift': (dy, dx) shift in pixels
        - 'error': registration error metric

    Examples
    --------
    >>> # Transfer ROIs from recording A to recording B
    >>> shifted_rois, transform = transfer_rois(
    ...     rois_a,
    ...     stack_a.mean(axis=0),
    ...     stack_b.mean(axis=0)
    ... )
    >>> print(f"Detected offset: {transform['shift']} pixels")

    Notes
    -----
    - ROI masks use negative integers for ROI labels (IGOR convention)
    - Background pixels are labeled as 1
    - Nearest-neighbor interpolation preserves integer ROI labels
    """
    # Compute shift between projections
    shift_yx, error, _ = phase_cross_correlation(
        ref_projection,
        target_projection,
        upsample_factor=upsample_factor,
        normalization=None,
    )

    # Check if shift is reasonable
    if np.abs(shift_yx).max() > max_shift:
        warnings.warn(
            f"Detected shift {shift_yx} exceeds max_shift={max_shift}. "
            f"ROI transfer may be unreliable.",
            RuntimeWarning
        )

    # Apply shift to ROI mask using nearest-neighbor to preserve labels
    shifted_mask = scipy_shift(roi_mask, shift=shift_yx, order=0, mode='constant', cval=1)

    # Build transform info
    transform = {
        'shift': tuple(shift_yx),
        'error': float(error),
    }

    return shifted_mask, transform
