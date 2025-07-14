"""
Tools for analyzing temporal extrema in STRFs
"""

import numpy as np


def map_extrema_timing(strfs, threshold=3.0, exclude_firstlast=(1, 1)):
    """
    Map the timing of extrema for each pixel across multiple STRFs.
    
    Parameters
    ----------
    strfs : ndarray
        4D array with shape (n_cells, time, y, x) or 3D array (time, y, x) for single cell
    threshold : float, optional
        Threshold in standard deviations. Default is 3.0.
    exclude_firstlast : tuple of int, optional
        Number of time points to exclude from beginning and end. Default is (1, 1).
    
    Returns
    -------
    timing_maps : ndarray
        3D array (n_cells, y, x) with time indices of absolute extrema.
        For single cell input, returns 2D array (y, x).
        NaN values indicate pixels below threshold.
    
    Notes
    -----
    Time indices are relative to the cropped array (after excluding first/last frames).
    Fully vectorized for speed with large datasets.
    """
    
    # Handle both 3D and 4D input
    if strfs.ndim == 3:
        strfs = strfs[np.newaxis, :]  # Add cell dimension
        single_cell = True
    elif strfs.ndim == 4:
        single_cell = False
    else:
        raise ValueError(f"Input must be 3D or 4D array, got {strfs.ndim}D")
    
    n_cells, n_time, n_y, n_x = strfs.shape
    
    # Crop time axis
    start_idx = exclude_firstlast[0]
    end_idx = n_time - exclude_firstlast[1]
    if start_idx >= end_idx:
        raise ValueError("exclude_firstlast removes all time points")
    
    strfs_cropped = strfs[:, start_idx:end_idx, :, :]
    
    # Vectorized threshold mask: max absolute response across time for each cell
    max_abs_response = np.max(np.abs(strfs), axis=1)  # Shape: (n_cells, y, x)
    threshold_mask = max_abs_response > threshold
    
    # Vectorized extrema finding: argmax of absolute values across time
    abs_values = np.abs(strfs_cropped)  # Shape: (n_cells, time_cropped, y, x)
    timing_maps = np.argmax(abs_values, axis=1)  # Shape: (n_cells, y, x)
    
    # Apply threshold mask
    timing_maps = timing_maps.astype(float)
    timing_maps[~threshold_mask] = np.nan
    
    # Return appropriate shape
    if single_cell:
        return timing_maps[0]  # Return 2D array for single cell
    else:
        return timing_maps


def convert_timing_to_milliseconds(timing_maps, frame_rate_hz=60.0, time_offset=0):
    """
    Convert timing indices to milliseconds relative to stimulus onset.
    
    Parameters
    ----------
    timing_maps : ndarray
        2D or 3D array of timing indices from map_extrema_timing
    frame_rate_hz : float, optional
        Frame rate in Hz. Default is 60.0.
    time_offset : int, optional
        Time offset to add (e.g., from exclude_firstlast). Default is 0.
        
    Returns
    -------
    timing_ms : ndarray
        Timing in milliseconds with same shape as input
    """
    frame_duration_ms = 1000.0 / frame_rate_hz
    return (timing_maps + time_offset) * frame_duration_ms