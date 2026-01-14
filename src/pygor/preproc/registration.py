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


def register_stack(
    stack: np.ndarray,
    n_reference_frames: int = 1000,
    artefact_crop = 2, 
    batch_size: int = 10,
    upsample_factor: int = 10,
    normalization: Optional[str] = None,
    order: int = 1,
    mode: str = "reflect",
    return_shifts: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
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
    batch_size : int, optional
        Number of frames to average per batch (default: 10)
    upsample_factor : int, optional
        Subpixel precision factor for phase correlation (default: 10)
    normalization : str or None, optional
        Phase correlation normalization mode (default: None).
        For low-SNR data, None is recommended. Use 'phase' for high-SNR.
    order : int, optional
        Spline interpolation order for shifting (0-5, default: 1)
    mode : str, optional
        Edge handling mode for shifting (default: 'reflect')
    return_shifts : bool, optional
        If True, return (registered, shifts, errors) (default: False)
    parallel : bool, optional
        Use parallel processing with FFT-based shifting for ~2x speedup
        (default: True)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: -1)

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
    if artefact_crop is not None:
        # Store original artefact region to restore after registration
        artefact_region = stack[:, :, 0:artefact_crop].copy()
        stack = stack[:, :, artefact_crop:]

    # Compute shifts
    shifts, errors = compute_batch_shifts(
        stack=stack,
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        normalization=normalization,
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
    )

    # Restore original artefact region (unregistered, as it's already corrected)
    if artefact_crop is not None:
        registered = np.concatenate((artefact_region, registered), axis=2)

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
    n_frames = len(stack)
    n_batches = int(np.ceil(n_frames / batch_size))

    # Create reference from initial frames
    reference = stack[:n_reference_frames].mean(axis=0)

    # Compute shifts per batch
    shifts = []
    errors = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_frames)

        # Average batch
        batch_avg = stack[start:end].mean(axis=0)

        # Compute shift
        shift_yx, error, _ = phase_cross_correlation(
            reference,
            batch_avg,
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
        Only used when parallel=False.
    mode : str, optional
        Edge handling mode for shifting (default: 'reflect').
        Only used when parallel=False. Options:
        - 'reflect': Reflects at edge, duplicating the edge pixel
        - 'constant': Pads with zeros
        - 'nearest': Extends with the nearest edge pixel value
        - 'mirror': Reflects at edge without duplicating the edge pixel
        - 'wrap': Wraps around to the opposite edge
    parallel : bool, optional
        Use parallel processing with FFT-based shifting for ~2x speedup
        (default: True)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: -1)

    Returns
    -------
    registered : ndarray
        Registered stack (same shape as input)

    Notes
    -----
    - Each batch of frames gets the same shift applied
    - When parallel=True, uses FFT-based shifting (faster, ~2x speedup)
    - When parallel=False, uses spline-based shifting (original behavior)
    """
    n_frames = len(stack)
    n_batches = len(shifts)
    
    # Store original data range for clipping
    original_min = stack.min()
    original_max = stack.max()

    # Choose processing path
    if parallel and n_frames > 100:
        # Parallel path: uses FFT-based shifting (faster)
        # Build per-frame shift array for parallel processing
        frame_shifts = np.zeros((n_frames, 2))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_frames)
            frame_shifts[start:end] = shifts[i]
        
        if _HAS_JOBLIB:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_shift_frame_fft)(stack[i], frame_shifts[i]) 
                for i in range(n_frames)
            )
            registered = np.array(results)
        else:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    lambda i: _shift_frame_fft(stack[i], frame_shifts[i]),
                    range(n_frames)
                ))
            registered = np.array(results)
        
        # Clip at end
        registered = np.clip(registered, original_min, original_max)
    
    else:
        # Sequential path: spline-based shifting (original behavior)
        registered = np.zeros_like(stack)
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_frames)
            shift_yx = shifts[i]
            
            for j in range(start, end):
                shifted = scipy_shift(stack[j], shift=shift_yx, order=order, mode=mode)
                registered[j] = np.clip(shifted, original_min, original_max)

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
