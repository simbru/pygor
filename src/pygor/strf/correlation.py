"""Spatiotemporal correlation analysis for STRF data."""
import numpy as np
import numpy.ma as ma
from scipy.stats import spearmanr


def correlation_polarity(arr_3d):
    """
    Extract polarity from correlation with the strongest pixel.

    Returns +1 for ON pixels (positive response), -1 for OFF pixels (negative response).
    The polarity is determined by correlation with the strongest pixel, then adjusted
    based on whether the strongest pixel itself is ON or OFF.

    Parameters
    ----------
    arr_3d : np.ndarray
        3D array of shape (n_frames, height, width).

    Returns
    -------
    polarity : np.ndarray
        2D array of shape (height, width) with values +1 or -1.
    """
    n_frames, height, width = arr_3d.shape

    # Find strongest pixel (max absolute amplitude across all space and time)
    abs_arr = np.abs(arr_3d)
    strongest_flat_idx = np.argmax(abs_arr)
    t_strong, y_strong, x_strong = np.unravel_index(strongest_flat_idx, arr_3d.shape)

    # Get the sign of the strongest pixel at its peak time (ON or OFF)
    strongest_pixel_sign = np.sign(arr_3d[t_strong, y_strong, x_strong])

    # Get correlation map
    corr_map = _compute_single_correlation_map_vectorized(arr_3d)

    # Correlation gives us similarity to the reference pixel
    # Multiply by the reference pixel's sign to get actual ON/OFF polarity
    polarity = np.sign(corr_map) * strongest_pixel_sign

    # Handle zero correlation (assign +1 by default)
    polarity = np.where(polarity == 0, 1, polarity)

    if isinstance(arr_3d, ma.MaskedArray):
        polarity = ma.masked_array(polarity, mask=corr_map.mask if hasattr(corr_map, 'mask') else np.isnan(corr_map))

    return polarity


def calculate_spatiotemporal_correlation(strf_obj, roi=None, method='pearson'):
    """
    Calculate spatiotemporal correlation map for STRF data.

    Step by step:
    1. Get the strongest pixel from the STRF along the time dimension.
    2. For that pixel, extract its timecourse.
    3. Compute the correlation between that timecourse and the timecourse of every other pixel
        in the STRF.
    4. Return the resulting spatiotemporal correlation map.

    Parameters
    ----------
    strf_obj : STRF
        STRF object containing the data.
    roi : int, optional
        ROI index to analyze. If None, analyzes all ROIs.
    method : str, optional
        Correlation method: 'pearson' (default) or 'spearman'.

    Returns
    -------
    correlation_map : np.ndarray
        Shape (n_rois, height, width) if roi is None, or (height, width) if single ROI.
        Correlation values in range [-1, 1].
    """
    # Get STRF data with borders removed
    strfs = strf_obj.strfs_no_border

    # Handle ROI selection
    if roi is not None:
        strfs = strfs[roi:roi+1]  # Keep dimension for consistency

    # Select correlation function
    if method == 'pearson':
        corr_func = _pearson_corr
    elif method == 'spearman':
        corr_func = _spearman_corr
    else:
        raise ValueError(f"Unknown correlation method: {method}. Use 'pearson' or 'spearman'.")

    # Compute correlation map for each ROI
    results = []
    for strf_3d in strfs:  # strf_3d shape: (n_frames, height, width)
        corr_map = _compute_single_correlation_map(strf_3d, corr_func)
        results.append(corr_map)

    result = np.array(results)

    # Squeeze if single ROI requested
    if roi is not None:
        result = result.squeeze(axis=0)

    return result


def _compute_single_correlation_map(strf_3d, corr_func):
    """
    Compute correlation map for a single 3D STRF array.

    Parameters
    ----------
    strf_3d : np.ndarray
        3D array of shape (n_frames, height, width).
    corr_func : callable
        Correlation function that takes two 1D arrays and returns a scalar.

    Returns
    -------
    correlation_map : np.ndarray
        2D array of shape (height, width) with correlation values.
    """
    n_frames, height, width = strf_3d.shape

    # Handle masked arrays
    is_masked = isinstance(strf_3d, ma.MaskedArray)

    # Step 1: Find strongest pixel (max absolute amplitude across time)
    collapsed = np.abs(strf_3d).max(axis=0)  # (height, width)
    strongest_idx = np.argmax(collapsed)
    y_strong, x_strong = np.unravel_index(strongest_idx, collapsed.shape)

    # Step 2: Extract reference timecourse
    ref_timecourse = strf_3d[:, y_strong, x_strong]

    # If reference is masked or constant, return empty correlation map
    if is_masked and ma.is_masked(ref_timecourse):
        correlation_map = np.full((height, width), np.nan)
        return ma.masked_array(correlation_map, np.ones_like(correlation_map, dtype=bool))

    if np.std(ref_timecourse) == 0:
        correlation_map = np.full((height, width), np.nan)
        if is_masked:
            return ma.masked_array(correlation_map, np.ones_like(correlation_map, dtype=bool))
        return correlation_map

    # Step 3: Compute correlation with every pixel
    correlation_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            pixel_timecourse = strf_3d[:, y, x]

            # Handle masked pixels
            if is_masked and ma.is_masked(pixel_timecourse):
                correlation_map[y, x] = np.nan
                continue

            # Handle constant timecourses (zero std)
            if np.std(pixel_timecourse) == 0:
                correlation_map[y, x] = np.nan
                continue

            correlation_map[y, x] = corr_func(ref_timecourse, pixel_timecourse)

    # Return masked array if input was masked
    if is_masked:
        mask = np.isnan(correlation_map)
        correlation_map = ma.masked_array(correlation_map, mask)

    return correlation_map


def _pearson_corr(x, y):
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(x, y)[0, 1]


def _spearman_corr(x, y):
    """Compute Spearman rank correlation coefficient."""
    return spearmanr(x, y).correlation


def _compute_single_correlation_map_vectorized(strf_3d):
    """
    Compute Pearson correlation map using vectorized operations (fast).

    Parameters
    ----------
    strf_3d : np.ndarray
        3D array of shape (n_frames, height, width).

    Returns
    -------
    correlation_map : np.ndarray
        2D array of shape (height, width) with correlation values in [-1, 1].
    """
    n_frames, height, width = strf_3d.shape
    is_masked = isinstance(strf_3d, ma.MaskedArray)

    # Step 1: Find strongest pixel
    collapsed = np.abs(strf_3d).max(axis=0)
    strongest_idx = np.argmax(collapsed)
    y_strong, x_strong = np.unravel_index(strongest_idx, collapsed.shape)

    # Step 2: Extract reference timecourse
    ref = strf_3d[:, y_strong, x_strong]

    # Handle edge cases
    ref_std = np.std(ref)
    if ref_std == 0:
        result = np.full((height, width), np.nan)
        if is_masked:
            return ma.masked_array(result, np.ones_like(result, dtype=bool))
        return result

    # Step 3: Flatten spatial dims and compute vectorized correlation
    pixels = strf_3d.reshape(n_frames, -1)  # (n_frames, height*width)

    # Normalize reference
    ref_norm = (ref - np.mean(ref)) / ref_std

    # Normalize all pixels
    pixels_mean = np.mean(pixels, axis=0)
    pixels_std = np.std(pixels, axis=0)

    # Avoid division by zero for constant pixels
    pixels_std = np.where(pixels_std == 0, np.nan, pixels_std)
    pixels_norm = (pixels - pixels_mean) / pixels_std

    # Pearson correlation via dot product of normalized vectors
    corr_flat = np.dot(ref_norm, pixels_norm) / n_frames

    # Reshape back to 2D
    correlation_map = corr_flat.reshape(height, width)

    # Handle masked arrays
    if is_masked:
        # Get mask from first frame (assumes consistent masking across time)
        spatial_mask = strf_3d.mask[0] if strf_3d.mask.ndim > 2 else strf_3d.mask
        correlation_map = ma.masked_array(correlation_map, mask=spatial_mask | np.isnan(correlation_map))

    return correlation_map
