"""
Core computational functions for pygor.

This module contains standalone computation functions that are called by
Core class methods. Separating these allows for cleaner code organization
and easier testing of the algorithms independently.
"""

import numpy as np


def _spatial_bin(images: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Spatially bin an image stack by averaging pixels in bin_factor x bin_factor blocks.

    IGOR implementation:
    InputDataBinDiv[floor(xx/binpix)][floor(yy/binpix)][]+=InputData[xx][yy][r]/(binpix^2)

    Parameters
    ----------
    images : np.ndarray
        3D array with shape (n_frames, height, width)
    bin_factor : int
        Binning factor (e.g., 2 means 2x2 pixel blocks become 1 pixel)

    Returns
    -------
    np.ndarray
        Binned array with shape (n_frames, ceil(height/bin_factor), ceil(width/bin_factor))
    """
    n_frames, height, width = images.shape
    new_height = int(np.ceil(height / bin_factor))
    new_width = int(np.ceil(width / bin_factor))

    # Pad to make dimensions divisible by bin_factor
    pad_height = new_height * bin_factor - height
    pad_width = new_width * bin_factor - width

    if pad_height > 0 or pad_width > 0:
        images = np.pad(images, ((0, 0), (0, pad_height), (0, pad_width)), mode='edge')

    # Reshape and average
    # Shape: (n_frames, new_height, bin_factor, new_width, bin_factor)
    reshaped = images.reshape(n_frames, new_height, bin_factor, new_width, bin_factor)
    # Average over the bin dimensions (axes 2 and 4)
    binned = reshaped.mean(axis=(2, 4))

    return binned


def compute_correlation_projection(
    images: np.ndarray,
    include_diagonals: bool = True,
    n_jobs: int = -1,
    timecompress: int = 1,
    binpix: int = 1,
) -> np.ndarray:
    """
    Compute pixel-wise temporal correlation with neighboring pixels.

    Creates a correlation map useful for visualizing functional connectivity
    in 2-photon calcium imaging data. Each pixel's correlation is computed as
    the average correlation coefficient with its immediate neighbors.

    Parameters
    ----------
    images : np.ndarray
        3D image stack with shape (n_frames, height, width).
    include_diagonals : bool, optional
        If True, uses 8-neighbor connectivity (including diagonals).
        If False, uses 4-neighbor connectivity (cardinal directions only).
        Default: True
    n_jobs : int, optional
        Number of parallel jobs to run. -1 uses all available CPUs.
        Default: -1
    timecompress : int, optional
        Temporal downsampling factor. Takes every Nth frame before computing
        correlations. This reduces noise and speeds up computation.
        IGOR implementation: "currentwave_main[]=InputData[xx][yy][p*timecompress]"
        Default: 1 (no downsampling)
    binpix : int, optional
        Spatial binning factor. Pixels are averaged in binpix x binpix
        blocks before computing correlations. The result is then expanded back
        to the original resolution.
        IGOR implementation: "InputDataBinDiv[floor(xx/binpix)][floor(yy/binpix)][]"
        Default: 1 (no binning)

    Returns
    -------
    np.ndarray
        Correlation projection with shape (height, width) matching input dimensions.
        Values range from -1 to 1, representing average correlation with neighbors.

    Notes
    -----
    Temporal compression (timecompress > 1) can significantly improve results
    for noisy data by:
    1. Reducing high-frequency noise that degrades correlation estimates
    2. Speeding up computation (linear with compression factor)
    3. Emphasizing slower calcium dynamics over noise

    Spatial binning (binpix > 1) can help by:
    1. Increasing SNR by averaging nearby pixels
    2. Dramatically speeding up computation (quadratic with binning factor)
    3. Smoothing the correlation map

    For typical 2-photon calcium imaging at ~30 Hz, timecompress=2-4 often
    gives cleaner correlation maps without losing significant signal.

    Examples
    --------
    >>> # Basic usage
    >>> corr_map = compute_correlation_projection(images)

    >>> # With temporal compression (faster, less noisy)
    >>> corr_map = compute_correlation_projection(images, timecompress=3)

    >>> # With spatial binning (much faster, smoother)
    >>> corr_map = compute_correlation_projection(images, binpix=2)

    >>> # Combined compression for fast preview
    >>> corr_map = compute_correlation_projection(images, timecompress=4, binpix=2)

    >>> # 4-connectivity only
    >>> corr_map = compute_correlation_projection(images, include_diagonals=False)
    """
    if images.ndim != 3:
        raise ValueError(f"Expected 3D array (frames, height, width), got shape {images.shape}")

    # Store original dimensions for expansion
    _, original_height, original_width = images.shape

    # Apply temporal compression if requested
    if timecompress > 1:
        # IGOR-style: take every Nth frame
        # "currentwave_main[]=InputData[xx][yy][p*timecompress]"
        images = images[::timecompress, :, :]

    # Apply spatial binning if requested
    if binpix > 1:
        images = _spatial_bin(images, binpix)

    n_frames, height, width = images.shape

    # Define neighbor offsets
    if include_diagonals:
        # 8-connectivity (all surrounding pixels)
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        # 4-connectivity (cardinal directions only)
        neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def compute_pixel_correlation(y, x):
        """Compute correlation for a single pixel with its neighbors."""
        pixel_trace = images[:, y, x]
        correlations = []

        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            # Check if neighbor is within bounds
            if 0 <= ny < height and 0 <= nx < width:
                neighbor_trace = images[:, ny, nx]
                # Compute Pearson correlation
                corr = np.corrcoef(pixel_trace, neighbor_trace)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        # Return mean correlation with neighbors
        return np.mean(correlations) if correlations else 0.0

    # Create list of all pixel coordinates
    pixel_coords = [(y, x) for y in range(height) for x in range(width)]

    # Compute correlations in parallel or sequentially
    if n_jobs == 1 or len(pixel_coords) <= 100:
        # Sequential computation for small images or when explicitly requested
        correlation_values = [compute_pixel_correlation(y, x) for y, x in pixel_coords]
    else:
        # Parallel computation using joblib
        from joblib import Parallel, delayed

        effective_frames = n_frames
        compress_msg = f" (timecompress={timecompress})" if timecompress > 1 else ""
        print(f"Computing correlation projection using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
        print(f"  Image: {height}x{width}, {effective_frames} frames{compress_msg}")

        correlation_values = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(compute_pixel_correlation)(y, x) for y, x in pixel_coords
        )

    # Reshape to 2D image
    correlation_projection = np.array(correlation_values).reshape(height, width)

    # Expand back to original resolution if spatial binning was used
    # IGOR: "Correlation_projection[][]=correlation_projection_4Compute[floor(p/binpix)][floor(q/binpix)]"
    if binpix > 1:
        # Use nearest-neighbor interpolation to expand
        expanded = np.zeros((original_height, original_width), dtype=correlation_projection.dtype)
        for y in range(original_height):
            for x in range(original_width):
                expanded[y, x] = correlation_projection[y // binpix, x // binpix]
        correlation_projection = expanded

    return correlation_projection
