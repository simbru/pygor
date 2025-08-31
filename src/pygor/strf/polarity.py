"""
STRF Polarity Analysis Module

This module contains functions for determining the polarity of STRF responses,
including center-surround analysis and rebound-aware polarity detection.
"""

import numpy as np
import pygor.np_ext


def get_polarities_old(strf_obj, roi=None, exclude_FirstLast=(1,1)) -> np.ndarray:
    """
    Original polarity detection method using absolute max comparison.
    
    Parameters
    ----------
    strf_obj : STRF
        STRF object containing timecourse data
    roi : int, optional
        Specific ROI to analyze (default: all ROIs)
    exclude_FirstLast : tuple, optional
        Timepoints to exclude from start and end (default: (1,1))
        
    Returns
    -------
    np.ndarray
        Polarity assignments: -1 (OFF), 1 (ON), 2 (bipolar), 0 (no signal)
    """
    # Get the time as absolute values, then get the max value
    abs_time_max = np.max(np.abs(strf_obj.get_timecourses(mask_empty=True).data), axis=2)
    
    # First find the obvious polarities
    pols = np.where(abs_time_max[:, 0] > abs_time_max[:, 1], -1, 1)
    
    # Now we check if values are close
    # We reorder, because np.isclose(a, b) assumes b is the reference 
    # and we will use the largest value as the reference
    abs_time_max_reordered = np.sort(abs_time_max, axis=1)
    outcome = np.isclose(abs_time_max_reordered[:, 0], abs_time_max_reordered[:, 1], rtol=.33, atol=.01)
    
    # If values were close, we assign 2
    pols = np.where(outcome, 2, pols)
    
    # Now we need to set values to 0 where there is no signal
    pols = pols * np.prod(np.where(abs_time_max == 0, 0, 1), axis=1)
    
    return pols


def get_polarities_new(strf_obj, roi=None, exclude_FirstLast=(1,1)) -> np.ndarray:
    """
    Improved polarity detection using signed peak values.
    
    Parameters
    ----------
    strf_obj : STRF
        STRF object containing timecourse data
    roi : int, optional
        Specific ROI to analyze (default: all ROIs)
    exclude_FirstLast : tuple, optional
        Timepoints to exclude from start and end (default: (1,1))
        
    Returns
    -------
    np.ndarray
        Polarity assignments: -1 (OFF), 1 (ON), NaN (no signal)
    """
    # Get the time as absolute values, then get the max value
    abs_time_max = np.max(np.abs(strf_obj.get_timecourses(mask_empty=True).data), axis=2)
    times_peak = pygor.np_ext.maxabs(strf_obj.get_timecourses(mask_empty=True), axis=2)
    
    # First find the obvious polarities
    pols = np.where(times_peak[:, 0] < 0, -1, 1)
    
    # If values are 0, we assign NaN
    zero_bools = np.all(times_peak == 0, axis=1)
    pols = np.where(zero_bools == True, np.nan, pols)
    
    return pols


def get_sustained_polarity_rebound_aware(timecourse):
    """
    Polarity detection that looks for sustained responses and positive rebounds
    after negative transients, avoiding fixed percentage-based cutoffs.
    
    This function handles edge cases where cells have negative transients followed
    by sustained positive responses, which should be classified as ON cells.
    
    Parameters
    ----------
    timecourse : np.ndarray
        1D array representing the temporal response
        
    Returns
    -------
    float
        Sustained response value (positive for ON, negative for OFF)
    """
    n = len(timecourse)
    if n < 5:  # Need minimum length for meaningful analysis
        return 0
    
    global_max_abs = np.max(np.abs(timecourse))
    if global_max_abs < 0.1:  # Too weak signal
        return 0
    
    # Find the global peak (most extreme value)
    global_peak_idx = np.argmax(np.abs(timecourse))
    global_peak_val = timecourse[global_peak_idx]
    
    # Look for opposite polarity rebounds after the global peak
    # This is key for catching positive rebounds after negative transients
    rebound_threshold = 0.25 * global_max_abs  # Lower threshold for rebounds
    
    best_rebound = 0
    best_rebound_score = 0
    
    # Search for rebounds in the latter half of the timecourse
    search_start = max(global_peak_idx + 1, n // 3)  # Start after peak or 1/3 through
    
    for i in range(search_start, n):
        val = timecourse[i]
        
        # Look for opposite polarity rebounds
        if (global_peak_val < 0 and val > rebound_threshold) or \
           (global_peak_val > 0 and val < -rebound_threshold):
            
            # Score based on magnitude and how sustained it is
            magnitude_score = abs(val)
            
            # Check if this value is sustained (look ahead a few points)
            sustainability_score = 1.0
            lookahead = min(3, n - i - 1)
            if lookahead > 0:
                future_vals = timecourse[i:i+lookahead+1]
                # If future values maintain similar magnitude, it's more sustained
                if all(abs(fv) > 0.5 * abs(val) for fv in future_vals):
                    sustainability_score = 1.5
            
            # Favor later rebounds (sustained response)
            time_bonus = 1.0 + 0.5 * (i - search_start) / (n - search_start) if n > search_start else 1.0
            
            total_score = magnitude_score * sustainability_score * time_bonus
            
            if total_score > best_rebound_score:
                best_rebound_score = total_score
                best_rebound = val
    
    # Decision logic: use rebound if it's substantial enough
    rebound_vs_peak_ratio = abs(best_rebound) / global_max_abs if best_rebound != 0 else 0
    
    if rebound_vs_peak_ratio > 0.3 and best_rebound_score > 0:
        # Strong rebound found - use it
        return best_rebound
    else:
        # No significant rebound - use global peak
        return global_peak_val


def get_polarities_cs(strf_obj, roi=None, exclude_FirstLast=(1,1), force_recompute=False, 
                     mode="cs_pol") -> np.ndarray:
    """
    Center-surround based polarity detection with rebound-aware analysis.
    
    This method uses center-surround segmentation to extract center timecourses,
    then applies rebound-aware polarity detection to handle complex temporal
    dynamics including transients and sustained responses.
    
    Parameters
    ----------
    strf_obj : STRF
        STRF object containing spatial-temporal data
    roi : int, optional
        Specific ROI to analyze (default: all ROIs)
    exclude_FirstLast : tuple, optional
        Timepoints to exclude from start and end (default: (1,1))
    force_recompute : bool, optional
        Force recomputation of center-surround segmentation (default: False)
    mode : str, optional
        Analysis mode: "cs_pol" or "cs_pol_extra" (default: "cs_pol")
        
    Returns
    -------
    np.ndarray
        Polarity classifications:
        - 1: ON center cells
        - -1: OFF center cells  
        - 2: Center-surround cells
        - NaN: Cells with insufficient signal
        
    Notes
    -----
    This method performs several steps:
    1. Center-surround segmentation to extract center timecourses
    2. Rebound-aware polarity detection on center responses
    3. Classification based on amplitude and covariance criteria
    4. Optional extra classification for strong vs weak center-surround
    """
    # Perform center-surround segmentation
    _, prediction_times_ROIs = strf_obj.cs_seg(force_recompute=force_recompute)
    
    # Classification parameters
    covar_thresh = -.5
    var_thresh = .2
    S_absamp_thresh = 1.5
    center_dominance_ratio = 2.0  # Center must be 2x stronger than surround for simple ON/OFF

    # Extract center and surround timecourses
    C_times = prediction_times_ROIs[:, 0, :]
    S_times = prediction_times_ROIs[:, 1, :]
    
    # Center timecourses for covariance calculation
    C_centered = C_times - C_times.mean(axis=1, keepdims=True)
    S_centered = S_times - S_times.mean(axis=1, keepdims=True)

    # Compute covariance for each pair
    CS_covariances = np.sum(C_centered * S_centered, axis=1) / (C_times.shape[1] - 1)

    # Get absolute max for each value of C and S
    C_maxabs = np.abs(pygor.np_ext.maxabs(C_times, axis=1))
    S_maxabs = np.abs(pygor.np_ext.maxabs(S_times, axis=1))

    # Get signs based on sustained response using rebound-aware approach
    C_sustained_response = np.array([get_sustained_polarity_rebound_aware(c) for c in C_times])
    C_signs = np.sign(C_sustained_response)
    
    # Initialize categories
    cat = np.where(C_signs > 0, 1, -1)
    cat = cat.astype("float")
    zerovals = C_signs == 0
    cat[zerovals] = np.nan
    
    # Only classify as center-surround if ALL criteria are met AND center doesn't dominate
    amplitude_pass_idx = S_maxabs > S_absamp_thresh
    var_pass_idx = np.var(S_times, axis=1) > var_thresh
    covariance_pass_idx = CS_covariances < covar_thresh
    center_not_dominant = C_maxabs <= center_dominance_ratio * S_maxabs
    
    # More stringent CS criteria - must pass all conditions
    cs_pass_bool = (amplitude_pass_idx & var_pass_idx & 
                   covariance_pass_idx & center_not_dominant)
    
    # Only assign center-surround if genuinely meets criteria
    cat = np.where(cs_pass_bool, 2, cat)
    
    # Set to NaN if center amplitude is too weak for reliable classification
    weak_center = C_maxabs < 1.0  # Minimum amplitude threshold
    cat = np.where(weak_center, np.nan, cat)
    
    # Extra check for strong vs weak CS (if requested)
    if mode == "cs_pol_extra":
        with np.errstate(divide='ignore', invalid='ignore'):
            lower_bound = .5
            upper_bound = 2
            
            # Calculate surround/center ratio for CS cells only
            cs_mask = cat == 2
            if np.any(cs_mask):
                surr_cent_ratio = S_maxabs[cs_mask] / C_maxabs[cs_mask]
                
                # Classify CS strength: 3 = strong CS, 4 = weak CS
                strong_cs = (surr_cent_ratio > upper_bound)
                weak_cs = (surr_cent_ratio >= lower_bound) & (surr_cent_ratio <= upper_bound)
                
                # Apply extra classification to CS cells
                cat_cs_extra = cat[cs_mask].copy()
                cat_cs_extra[strong_cs] = 3
                cat_cs_extra[weak_cs] = 4
                cat[cs_mask] = cat_cs_extra
    
    return cat


def get_polarities(strf_obj, roi=None, exclude_FirstLast=(1,1), mode="cs_pol", 
                  force_recompute=False) -> np.ndarray:
    """
    Main polarity detection function with multiple analysis modes.
    
    Parameters
    ----------
    strf_obj : STRF
        STRF object containing spatial-temporal data
    roi : int, optional
        Specific ROI to analyze (default: all ROIs) 
    exclude_FirstLast : tuple, optional
        Timepoints to exclude from start and end (default: (1,1))
    mode : str, optional
        Analysis mode:
        - "old": Original absolute max comparison
        - "new": Improved signed peak detection
        - "cs_pol": Center-surround with rebound-aware detection (default)
        - "cs_pol_extra": CS analysis with strong/weak classification
    force_recompute : bool, optional
        Force recomputation of center-surround segmentation (default: False)
        
    Returns
    -------
    np.ndarray
        Polarity classifications depend on mode:
        - old/new: -1 (OFF), 1 (ON), 2 (bipolar), 0/NaN (no signal)
        - cs_pol: -1 (OFF), 1 (ON), 2 (center-surround), NaN (weak signal)
        - cs_pol_extra: Same as cs_pol plus 3 (strong CS), 4 (weak CS)
        
    Examples
    --------
    >>> # Basic center-surround analysis
    >>> polarities = strf_obj.get_polarities()
    >>> 
    >>> # Force recomputation with debug
    >>> polarities = strf_obj.get_polarities(mode="cs_pol", force_recompute=True)
    >>>
    >>> # Extended CS analysis
    >>> polarities = strf_obj.get_polarities(mode="cs_pol_extra")
    """
    if mode == "old":
        return get_polarities_old(strf_obj, roi, exclude_FirstLast)
    elif mode == "new":
        return get_polarities_new(strf_obj, roi, exclude_FirstLast)
    elif mode in ["cs_pol", "cs_pol_extra"]:
        return get_polarities_cs(strf_obj, roi, exclude_FirstLast, force_recompute, mode)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Available: 'old', 'new', 'cs_pol', 'cs_pol_extra'")