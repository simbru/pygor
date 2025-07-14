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


def compare_color_channel_timing_wrapper(strf_obj, roi, color_channels=(0, 1), threshold=3.0,
                                        exclude_firstlast=(1, 1), return_milliseconds=False, 
                                        frame_rate_hz=60.0):
    """
    Compare extrema timing between different color channels for a single ROI.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing multicolor data
    roi : int
        ROI index to analyze
    color_channels : tuple of int, optional
        Two color channel indices to compare. Default is (0, 1).
    threshold : float, optional
        Threshold in standard deviations. Default is 3.0.
    exclude_firstlast : tuple of int, optional
        Number of time points to exclude from beginning and end. Default is (1, 1).
    return_milliseconds : bool, optional
        If True, convert timing to milliseconds. Default is False.
    frame_rate_hz : float, optional
        Frame rate for millisecond conversion. Default is 60.0.
        
    Returns
    -------
    timing_difference : ndarray
        2D array (y, x) of timing differences (channel2 - channel1).
        NaN where either channel is below threshold.
    """
    
    if not strf_obj.multicolour:
        raise ValueError("Color channel comparison requires multicolor STRF data")
    
    # Get timing maps for both color channels
    timing1 = map_extrema_timing_wrapper(
        strf_obj, roi=roi, color_channel=color_channels[0], threshold=threshold,
        exclude_firstlast=exclude_firstlast, return_milliseconds=return_milliseconds,
        frame_rate_hz=frame_rate_hz
    )
    
    timing2 = map_extrema_timing_wrapper(
        strf_obj, roi=roi, color_channel=color_channels[1], threshold=threshold,
        exclude_firstlast=exclude_firstlast, return_milliseconds=return_milliseconds,
        frame_rate_hz=frame_rate_hz
    )
    
    # Return timing difference
    return timing2 - timing1


def map_extrema_timing_wrapper(strf_obj, roi=None, color_channel=0, threshold=3.0, 
                             exclude_firstlast=(1, 1), return_milliseconds=False, frame_rate_hz=60.0):
    """
    Map the timing of extrema for each pixel in STRF.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing the data
    roi : int, optional
        ROI index. If None, returns results for all ROIs.
    color_channel : int, optional
        Color channel index (for multicolor STRFs). Default is 0.
    threshold : float, optional
        Threshold in standard deviations. Default is 3.0.
    exclude_firstlast : tuple of int, optional
        Number of time points to exclude from beginning and end. Default is (1, 1).
    return_milliseconds : bool, optional
        If True, convert timing to milliseconds. Default is False.
    frame_rate_hz : float, optional
        Frame rate for millisecond conversion. Default is 60.0.
        
    Returns
    -------
    timing_maps : ndarray
        2D array (y, x) for single ROI or 3D array (n_rois, y, x) for all ROIs.
        Values are time indices of extrema (or milliseconds if return_milliseconds=True).
        NaN indicates pixels below threshold.
    """
    
    if roi is not None:
        # Single ROI analysis
        if strf_obj.multicolour:
            strf_idx = roi * strf_obj.numcolour + color_channel
            if strf_idx >= len(strf_obj.strfs):
                raise IndexError(f"ROI {roi}, color {color_channel} exceeds available STRFs")
            strf_3d = strf_obj.strfs[strf_idx]
        else:
            if roi >= len(strf_obj.strfs):
                raise IndexError(f"ROI {roi} exceeds available STRFs")
            strf_3d = strf_obj.strfs[roi]
        
        timing_map = map_extrema_timing(
            strf_3d, threshold=threshold, exclude_firstlast=exclude_firstlast
        )
        
        if return_milliseconds:
            timing_map = convert_timing_to_milliseconds(
                timing_map, frame_rate_hz, exclude_firstlast[0]
            )
        
        return timing_map
    
    else:
        # All ROIs analysis - vectorized
        if strf_obj.multicolour:
            # Extract specific color channel for all ROIs
            n_rois = len(strf_obj.strfs) // strf_obj.numcolour
            strf_indices = [i * strf_obj.numcolour + color_channel for i in range(n_rois)]
            strfs_subset = strf_obj.strfs[strf_indices]
        else:
            strfs_subset = strf_obj.strfs
        
        timing_maps = map_extrema_timing(
            strfs_subset, threshold=threshold, exclude_firstlast=exclude_firstlast
        )
        
        if return_milliseconds:
            timing_maps = convert_timing_to_milliseconds(
                timing_maps, frame_rate_hz, exclude_firstlast[0]
            )
        
        return timing_maps