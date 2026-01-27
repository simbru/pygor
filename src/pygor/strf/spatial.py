"""
Tools for determining spatial properties of STRFs
"""

import numpy as np
import numpy.ma as ma
#import math
import scipy
#import skimage.measure
#import skimage.segmentation
import warnings
from sklearn.preprocessing import MinMaxScaler
import scipy.signal

# Local imports
import pygor.strf.contouring
import pygor.strf.correlation
import pygor.utilities

def centroid(arr):#
    """TODO
    Rewrite for new contour function"""
    if np.isnan(arr).all() == True:
        return np.nan, np.nan
    arr_shape = arr.shape
    if arr_shape[0] < arr_shape[1]:
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
    else:
        length = arr.shape[1]
        sum_x = np.sum(arr[0])
        sum_y = np.sum(arr[1])
    return sum_x/length, sum_y/length

def snr_gated_balance_ratio(sta_array, axis=(-2, -1), snr_threshold=5.0, noise_method='std'):
    """
    Calculate SNR-gated balance ratio for spatial opponency.
    
    Parameters:
    -----------
    sta_array : numpy array
        STA data of any shape
    axis : int, tuple of ints, or None
        Axis/axes along which to compute the metric. If None, compute over entire array.
        For shape (cell_idx, num_colours, x, y), use axis=(2,3) to compute per cell/color
    snr_threshold : float
        Minimum SNR required to compute balance ratio
    noise_method : str
        Method to estimate noise ('std' or 'periphery')
    
    Returns:
    --------
    balance_ratio : numpy array
        Balance ratios (NaN where SNR < threshold)
    snr : numpy array  
        Signal-to-noise ratios
    """
    
    # Calculate SNR
    peak_signal = np.max(np.abs(sta_array), axis=axis, keepdims=True)
    
    if noise_method == 'std':
        noise_level = np.std(sta_array, axis=axis, keepdims=True)
    else:  # Could add periphery method later
        noise_level = np.std(sta_array, axis=axis, keepdims=True)
    
    snr = peak_signal / (noise_level + 1e-10)  # Avoid division by zero
    
    # Calculate balance ratio where SNR is sufficient
    pos_sum = np.sum(np.maximum(sta_array, 0), axis=axis, keepdims=True)
    neg_sum = np.abs(np.sum(np.minimum(sta_array, 0), axis=axis, keepdims=True))
    
    # Avoid division by zero
    max_sum = np.maximum(pos_sum, neg_sum)
    min_sum = np.minimum(pos_sum, neg_sum)
    
    balance_ratio = np.where(max_sum > 0, min_sum / max_sum, 0)
    
    # Apply SNR gate
    balance_ratio = np.where(snr.squeeze() >= snr_threshold, balance_ratio.squeeze(), np.nan)
    
    return balance_ratio#, snr.squeeze()

def snr_gated_spatial_opponency(sta_array, axis=(-2, -1), snr_threshold=3.0, noise_method='std'):
    """
    Calculate SNR-gated spatial opponency (net vs total energy).
    
    Parameters same as above.
    
    Returns:
    --------
    spatial_opponency : numpy array
        Spatial opponency values (NaN where SNR < threshold)  
    snr : numpy array
        Signal-to-noise ratios
    """
    
    # Calculate SNR (same as above)
    peak_signal = np.max(np.abs(sta_array), axis=axis, keepdims=True)
    
    if noise_method == 'std':
        noise_level = np.std(sta_array, axis=axis, keepdims=True)
    else:
        noise_level = np.std(sta_array, axis=axis, keepdims=True)
    
    snr = peak_signal / (noise_level + 1e-10)
    
    # Calculate spatial opponency
    net_response = np.sum(sta_array, axis=axis, keepdims=True)
    total_energy = np.sum(np.abs(sta_array), axis=axis, keepdims=True)
    
    spatial_opponency = np.where(total_energy > 0, 
                                1 - (np.abs(net_response) / total_energy), 
                                0)
    
    # Apply SNR gate
    spatial_opponency = np.where(snr.squeeze() >= snr_threshold, 
                                spatial_opponency.squeeze(), 
                                np.nan)
    
    return spatial_opponency 

def _legacy_pixel_polarity(array_3d):
    """
    Calculate the polarity of each pixel in a 3D array based on the time series
     at each pixel location.

    Parameters
    ----------
    array_3d : numpy array
        A 3D array with shape (time, x, y).

    Returns
    -------
    polarity_array : numpy array
        A 2D array with shape (x, y) containing the polarity values for each pixel.

    Notes
    -----
    The polarity of a pixel is determined as follows:
        - If the time series at a pixel location is all zeros, the pixel polarity is set to 0.
        - If the time series is masked, the pixel polarity is also masked.
        - Otherwise, the polarity is set to 1 if the maximum value in the time 
        series occurs after the minimum value, 
        or -1 if the minimum value occurs after the maximum value.
    """
    if isinstance(array_3d, np.ma.MaskedArray) == True:
        polarity_array = np.zeros((array_3d.shape[1], array_3d.shape[2]))
        mask = np.zeros((array_3d.shape[1], array_3d.shape[2]))
        polarity_array = ma.masked_array(polarity_array, mask)
    else:
        polarity_array = np.zeros((array_3d.shape[1], array_3d.shape[2]))
    for x in range(array_3d.shape[1]):
        for y in range(array_3d.shape[2]):
            if all(array_3d[:, x, y] == 0) == True: # if time series at pix x, y is 0 all the way through
                polarity_array[x, y] = 0
            elif np.ma.is_masked(array_3d[:, x, y]) == True:
                polarity_array.mask[x, y] = True
            else:
                maxi = np.max(array_3d[:, x, y])
                mini = np.min(array_3d[:, x, y])
                where_mini = np.where(array_3d[:, x, y] == mini)
                where_maxi = np.where(array_3d[:, x, y] == maxi)
                if where_mini < where_maxi:
                    polarity_array[x, y] = -1
                if where_maxi < where_mini:
                    polarity_array[x, y] = 1
    return polarity_array

def pixel_polarity(arr_3d, exclude_PrePost = (2, 2)):
    """
    Return a map of whether minimum or maximum value comes first, expressed 
    as 1 or -1 and represents the polarity of a given pixel. 

    exclude_PrePost : tuple
        - first index: exclude n first frames 
        - second index: exclude m last frames
    """
    if np.sum(exclude_PrePost) > len(arr_3d):
        raise ValueError("exclude_PrePost is leading to exclusion of entire array.")
    max_locs = np.argmax(arr_3d[exclude_PrePost[0]:len(arr_3d)-exclude_PrePost[1]], axis = 0)
    min_locs = np.argmin(arr_3d[exclude_PrePost[0]:len(arr_3d)-exclude_PrePost[1]], axis = 0)
    bool_arr = max_locs > min_locs
    # Needs to retain mask if input array contained mask 
    if isinstance(arr_3d, np.ma.MaskedArray) == True:
        bool_arr = np.ma.array(data = bool_arr, mask = arr_3d[0].mask) # take mask from first frame
        return np.ma.where(bool_arr == 0, -1, 1)
    # Otherwise
    else:
        return np.where(bool_arr == 0, -1, 1)

def _legacy_corr_spacetime(array_3d, border = 0):
    """
    Calculate correlations between elements of a 3D NumPy array in space and time.
    
    Parameters
    ----------
    array_3d : numpy.ndarray
        A 3D NumPy array containing the elements to be correlated.
    border : int, optional
        The number of border elements to skip when calculating correlations. 
        Default is 0.
    
    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the correlations between the elements of 
        `array_3d` in space and time, normalized to be between -1 and 1.
    
    Notes
    -----
    - If `array_3d` is a masked array, the returned array will also be masked.
    - The correlations are calculated using NumPy's `np.correlate` function and 
    averaged using NumPy's `np.average` function.
    """
    if isinstance(array_3d, np.ma.MaskedArray) == True:
        correlation_map = np.zeros((array_3d.shape[1], array_3d.shape[2]))
        mask = np.zeros((array_3d.shape[1], array_3d.shape[2]))
        correlation_map = ma.masked_array(correlation_map, mask)
    else:
        correlation_map = np.zeros((array_3d.shape[1], array_3d.shape[2]))
    for x in range(border, array_3d.shape[1]-border):
        for y in range(border, array_3d.shape[2]-border):
            if np.ma.is_masked(array_3d[:, x, y]) == True:
                correlation_map.mask[x, y] = True
            if x == array_3d.shape[1] - 1 or y == array_3d.shape[2] - 1:
                correlation_map[x, y] = 0
            else:
                curr_corr_coeff = np.zeros((8))
                "Could rewrite this for efficiency"
                centre_pix   = array_3d[:, x, y]
                top_left     = array_3d[:, x-1, y+1] # determine offsets
                top_mid      = array_3d[:,x, y+1]
                top_right    = array_3d[:,x+1, y+1]
                mid_left     = array_3d[:,x-1, y]
                mid_right    = array_3d[:,x+1, y]
                bottom_left  = array_3d[:,x-1, y-1]
                bottom_mid   = array_3d[:,x, y-1]
                bottom_right = array_3d[:,x+1, y-1]

                # correlates in time (same length), so 1 diagonal will be 1s and the other will be correlation coefficient (?)
                curr_corr_coeff[0] = np.correlate(centre_pix, top_left)
                curr_corr_coeff[1] = np.correlate(centre_pix, top_mid)
                curr_corr_coeff[2] = np.correlate(centre_pix, top_right)
                curr_corr_coeff[3] = np.correlate(centre_pix, mid_left)
                curr_corr_coeff[4] = np.correlate(centre_pix, mid_right)
                curr_corr_coeff[5] = np.correlate(centre_pix, bottom_left)
                curr_corr_coeff[6] = np.correlate(centre_pix, bottom_mid)
                curr_corr_coeff[7] = np.correlate(centre_pix, bottom_right)
                
                correlation_map[x, y] = np.average(curr_corr_coeff)
    return correlation_map / np.max(correlation_map) # Normalize correlation map by its own max, to get scale between -1 or 1 and some number on the other end of polarity

def corr_spacetime(arr_3d, convolve = True, kernel_width = 3, kernel_depth = 5, 
    pix_subsample = 1, time_subsample = 2, mode = "var", corr_mode = 'constant'):
    """
    Calculate the spatial-temporal correlations of a 3D array (with time on the first axis). 
    
    The correlations are calculated by convolving the array with a 3D box kernel of 
    specified width and depth, and  optionally subsampling the array in the spatial and temporal 
    dimensions by specified factors. The correlations can be calculated as the variance (recommended), 
    standard deviationsum, average, or sum of the convolved array. The convolution 
    is performed using `scipy.ndimage.correlate` and the specified correlation mode.
    
    Parameters
    ----------
    arr_3d : ndarray
        The input 3D array (time on first axis).
    kernel_width : int, optional
        Width of the 3D box kernel. Default is 2.
    kernel_depth : int, optional
        Depth of the 3D box kernel. Default is 1.
    pix_subsample : int, optional
        Factor by which to subsample the array in the spatial dimensions. Default is 1.
    time_subsample : int, optional
        Factor by which to subsample the array in the temporal dimension. Default is 1.
    mode : str, optional
        Mode for calculating correlations. Can be "sum", "avg", "var", or "std". Default is "var".
    corr_mode : str, optional
        Mode for the convolution. Passed to `scipy.ndimage.correlate`. Default is "constant".
    
    Returns
    -------
    ndarray
        The spatial-temporal correlations of the input array.
    """
    kernel = np.expand_dims(np.ones((kernel_width, kernel_width)), axis = 0) # Needs to be same dimension (so 3d in, this should have 3 dims too)
    kernel = np.repeat(kernel, kernel_depth, axis = 0)
    # ^ potential here to define your own kernel. Didn't seem worthwhile implementing just yet. 
    if isinstance(arr_3d, np.ma.MaskedArray):
        arr_3d = np.where(arr_3d.mask, 0, arr_3d.data)
    # Define a function to get prod_funct based on the mode
    def get_prod_funct(mode):
        if mode in ["var", "variance", None]:
            return lambda x, axis: np.ma.var(x, axis=axis)
        elif mode in ["std", "stdev", "sd"]:
            return lambda x, axis: np.ma.std(x, axis=axis)
        elif mode == "sum":
            return lambda x, axis: np.ma.sum(x, axis=axis)
        elif mode in ["avg", "average"]:
            return lambda x, axis: np.ma.average(x, axis=axis)
        elif mode in ["max", "maximum"]:
            return lambda x, axis: np.ma.max(x, axis=axis)
        elif mode == "absmax":
            return lambda x, axis: np.ma.max(np.abs(x), axis=axis)
        elif mode == "corr":
            return lambda x, axis: _legacy_corr_spacetime(x)
        else:
            raise ValueError("Invalid mode specified")
    prod_funct = get_prod_funct(mode)
    # Convolves input according to kernel and returns sum of products at each location as corrs
    if convolve is True:
        # convolved_corr = scipy.ndimage.correlate(arr_3d[::time_subsample, ::pix_subsample, ::pix_subsample], weights=kernel, mode = corr_mode)
        convolved_corr = scipy.signal.fftconvolve(arr_3d[::time_subsample, ::pix_subsample, ::pix_subsample], kernel, mode = "same")
        if pix_subsample > 1: 
            convolved_corr = np.kron(convolved_corr, np.ones((time_subsample, pix_subsample, pix_subsample)))
        # Then we collapse convolved_corr by a simple mathematical operation (variance-based ones work best)
        corr_arr = prod_funct(convolved_corr, axis = 0)
        # Finally we re-scale the data to apply the same range as we had before (makes life a bit easier)
        temp_arr = prod_funct(arr_3d, axis = 0)
        min, max = np.min(temp_arr), np.max(temp_arr)
        if min == max:
            min, max = 0, 1
        scaler = MinMaxScaler(feature_range=(min, max))
        corr_arr = scaler.fit_transform(corr_arr.reshape(-1, 1)).reshape(corr_arr.shape)
    else:
        corr_arr = prod_funct(arr_3d, axis = 0)
    return corr_arr

def collapse_3d(arr_3d, zscore=True, **kwargs):
    """Collapses a 3D array by applying spatial-temporal correlation and polarity.

    This function takes in a 3D array and collapses it by multiplying the result
    of a spatial-temporal correlation operation by a polarity array. If the `zscore`
    flag is set to True, the collapsed array is also transformed by dividing it by
    the standard deviation of its border region.

    Parameters
    ----------
    arr_3d : numpy.ndarray
        A 3D array to be collapsed.
    zscore : bool, optional
        A flag indicating whether the collapsed array should be transformed by
        dividing it by the standard deviation of its border region (default is True).
    border_width : int, optional
        *Depricated*
        The width of the border region used to calculate the standard deviation
        (default is 5).

    Returns
    -------
    collapsed_to_2d : numpy.ndarray
        The collapsed array.
    """
    if zscore == True:
        # First we get the mask for which the baseline should be calculated
        if isinstance(arr_3d, np.ma.MaskedArray) == False:
            border_mask = pygor.utilities.auto_border_mask(arr_3d)
            arr_3d = np.ma.array(arr_3d, mask=border_mask)
            arr_3d = scipy.stats.zscore(arr_3d, axis=None)
        else:
            arr_3d = scipy.stats.zscore(arr_3d, axis=None)
    # Get the polarity for each pixel via correlation with strongest pixel
    polarity_array = pygor.strf.correlation.correlation_polarity(arr_3d)
    # Collapse space via temporal correlation
    corr_map = corr_spacetime(arr_3d, **kwargs)
    # Put polarity back into the correlation map
    return corr_map * polarity_array

def get_polarity_masks(arr_2d):#, negative_range = (-1, -0.5), positive_range = (0.5, 1)):
    """
    Creates two masks for the elements in a numpy array: a negative mask and a 
    positive mask.
    ----------
    Parameters:
    - STRF_arr: a numpy array containing the values to be masked.
    ----------
    Returns:
    - A tuple containing two masked arrays: a negative masked array and a 
    positive masked array.
    """
    neg_masked = np.ma.masked_where(arr_2d > 0, arr_2d)
    pos_masked = np.ma.masked_where(arr_2d < 0, arr_2d)
    return (neg_masked), (pos_masked)

def polarity_2d(arr_2d):
    """
    Determine the polarity of a 2D array regardless of signal/noise ratio.

    Parameters
    ----------
    arr_2d : numpy.ndarray
        The 2D array for which to determine the polarity.

    Returns
    -------
    pol : int
        The polarity of the array. Returns 1 if the sum of the maximum mask is 
        greater than the sum of the minimum mask, and -1 otherwise.
    """
    # Get masks centered on 0 to min & 0 to max
    neg_mask, pos_mask = get_polarity_masks(arr_2d)
    # Determine polarity by which has the most abs "weight"
    if np.abs(np.sum(neg_mask)) > np.abs(np.sum(pos_mask)):
        # If the negative masked array is the heaviest 
        pol = -1
    else:
        # If the positive masked array is the heaviest 
        pol = 1
    return pol

def rf_mask2d(arr_3d, axis = 0, level = None, mode = collapse_3d, **kwargs):
    """
    Generate a masked array for a 2D array representing a receptive field (RF).
    
    The mask is generated based on the polarity of the RF, ande is created by
    finding the contour of the maximum (positive polarity) or minimum (negative
    polarity) values in the array. The resulting mask can be applied to the
    original array to exclude values outside of the RF contour.
    
    Parameters
    ----------
    arr_2d : ndarray
        2D array representing a receptive field.
    
    Returns
    -------
    masked_array : MaskedArray
        Masked version of the input array, with values outside of the RF
        contour masked.
    """
    # Collapse 3rd dimention 
    arr_3d_collapsed = mode(arr_3d, **kwargs)
    # Create contour and mask
    _contour = pygor.strfs.conturing.contour(arr_3d_collapsed, expect_bipolar = True, **kwargs)
    (_contour_mask_neg, _contour_mask_pos) = pygor.strf.contouring.contour_mask(_contour, arr_3d_collapsed.shape, expect_bipolar = True, **kwargs)
    # Mask array with negative and positive mask
    neg_masked = np.ma.array(arr_3d_collapsed, mask = _contour_mask_neg)
    pos_masked = np.ma.array(arr_3d_collapsed, mask = _contour_mask_pos)
    arr = np.ma.array((neg_masked, pos_masked))
    return arr

# def rf_mask3d(arr_3d, axis = 0, level = None, mode = collapse_3d, **kwargs):
#     """
#     Generate a mask for a 3D array representing a spatio-temporal receptive field (STRFs).
    
#     The mask is generated by collapsing the 3D array along a specified axis using the
#     `mode` function (currenlty only 'collapse_3d' works), and then finding the contour 
#     of the resulting 2D array. The resulting mask is then applied to all 2D slices of 
#     the original 3D array along the 0th axis (time).
    
#     Parameters
#     ----------
#     arr_3d : ndarray
#         3D array representing a sequence of 2D receptive fields.
#     axis : int, optional
#         Axis along which the 3D array will be collapsed and the mask applied.
#         Default is 0.
#     mode : function, optional
#         Function used to collapse the 3D array into a 2D array. Will accept 
#         for example np.var or np.std. Default is `collapse_3d`.
#     **kwargs : optional
#         These are keyword args passed to mode(arr_3d, **kwargs) so that functions 
#         using axis = 0 (numpy) or axis = None (scipy) can be used. 

#     Returns
#     -------
#     masked_array : MaskedArray
#         Masked version of the input array, with values outside of the RF
#         contour masked in all 2D slices along the specified axis.
#     """
#     # Figure out how many frames
#     frames = arr_3d.shape[axis]
#     # Create contour and mask
#     if "mask_tup" in kwargs:
#         arr_3d_collapsed = mode(arr_3d)
#         (_contour_mask_neg, _contour_mask_pos) = kwargs["mask_tup"]
#         neg_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_neg, axis = 0), frames, axis = axis))
#         pos_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_pos, axis = 0), frames, axis = axis))
#         arr = np.ma.array((neg_masked, pos_masked))
#         return arr
#     # Collapse 3rd dimention 
#     arr_3d_collapsed = mode(arr_3d, **kwargs)
#     if level == None:
#         _contour = pygor.strf.contouring.contour(arr_3d_collapsed, expect_bipolar = True, **kwargs)
#     else:
#         _contour = pygor.strf.contouring.contour(arr_3d_collapsed, expect_bipolar = True, level = (-level, level), **kwargs)
#     (_contour_mask_neg, _contour_mask_pos) = pygor.strf.contouring.contour_mask(_contour, arr_3d_collapsed.shape, expect_bipolar = True, **kwargs)
#     # Mask array with negative and positive mask
#     neg_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_neg, axis = 0), frames, axis = axis))
#     pos_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_pos, axis = 0), frames, axis = axis))
#     arr = np.ma.array((neg_masked, pos_masked))
#     return arr

def rf_mask3d(arr_3d, axis = 0, level = None, mode = collapse_3d, **kwargs):
    """
    Generate a mask for a 3D array representing a spatio-temporal receptive field (STRFs).
    
    The mask is generated by collapsing the 3D array along a specified axis using the
    `mode` function (currenlty only 'collapse_3d' works), and then finding the contour 
    of the resulting 2D array. The resulting mask is then applied to all 2D slices of 
    the original 3D array along the 0th axis (time).
    
    Parameters
    ----------
    arr_3d : ndarray
        3D array representing a sequence of 2D receptive fields.
    axis : int, optional
        Axis along which the 3D array will be collapsed and the mask applied.
        Default is 0.
    mode : function, optional
        Function used to collapse the 3D array into a 2D array. Will accept 
        for example np.var or np.std. Default is `collapse_3d`.
    **kwargs : optional
        These are keyword args passed to mode(arr_3d, **kwargs) so that functions 
        using axis = 0 (numpy) or axis = None (scipy) can be used. 

    Returns
    -------
    masked_array : MaskedArray
        Masked version of the input array, with values outside of the RF
        contour masked in all 2D slices along the specified axis.
    """
    frames = arr_3d.shape[axis]
    arr_2d = mode(arr_3d, **kwargs)
    # get 2d masks
    all_masks = np.array(pygor.strf.contouring.bipolar_mask(arr_2d))
    # Apply mask to expanded and repeated strfs (to get negative and positive)
    strfs_expanded = np.repeat(np.expand_dims(all_masks, axis = 1), arr_3d.shape[0], axis = 1)
    # all_strfs_masked = np.ma.array(strfs_expanded, mask = all_masks, keep_mask=True)
    # return all_strfs_masked
    return strfs_expanded

def concat_masks(ma_list):
    """
    ma_list can be either list or MaskedArray of MaskedArrays, but will not work with 
    Numpy array of MaskedArrays because regular np arrays removes the mask.

    The data for arrays in the list must be the same, with only the masks differing. 
    """
    # Do some housekeeping for ease of use and making it clear what input needs to be 
    if type(ma_list) is np.ndarray:
        raise AssertionError("Input array must be either list of MaskedArrays MaskedArray of MaskedArrays. You passed a regular numpy array, which throws away any mask present.")
    # Check if all arrays have masks (by default these should be boolean)
    if np.all([np.ma.is_masked(x) for x in ma_list]) is False:
        raise AssertionError("Not all input arrays contain masks")
    # Check if all arrays in the list are contain the same data 
    if np.all([x.data == ma_list[0].data for x in ma_list]) is False: #if data for every index is the same as index 0
        raise AssertionError("Input arrays in do not contain the same data")
    # Since we know the data is the same, use the first index as our data "truth"
    data = ma_list[0].data
    # Extract masks and place them in their own array
    masks_list = np.array([x.mask for x in ma_list])
    # Now, take the product of all the masks (since each mask should be boolean) along first axis
    concat_mask = np.prod(masks_list, axis = 0)
    return np.ma.array(data = data, mask = concat_mask)

def centre_on_max(arr_3d):
    """
    Simpler, faster centring on max pixel in RF (approximating centre)
    """
    arr_2d = np.var(arr_3d, axis = 0)
    arr_2d_max = np.unravel_index(np.argmax(np.abs(arr_2d)), arr_2d.shape)
    target_pos = np.array(arr_3d.shape[1:]) / 2 # middle
    shift_by = target_pos - arr_2d_max
    shift_by = np.nan_to_num(shift_by).astype(int)
    strf_shifted = np.roll(arr_3d, shift_by, axis = (1,2))
    return strf_shifted

def centre_on_max2d(arr_2d):
    """
    Simpler, faster centring on max pixel in RF (approximating centre) for 2D arrays
    """
    arr_2d_max = np.unravel_index(np.argmax(np.abs(arr_2d)), arr_2d.shape)
    target_pos = np.array(arr_2d.shape) / 2  # middle
    shift_by = target_pos - arr_2d_max
    shift_by = np.nan_to_num(shift_by).astype(int)
    strf_shifted = np.roll(arr_2d, shift_by, axis=(0, 1))
    return strf_shifted

def centre_on_mass(arr_3d):
    """
    Simpler, faster centring on centre of mass
    """
    arr_2d = np.var(arr_3d, axis = 0)
    arr_2d_max = np.unravel_index(np.argmax(np.abs(arr_2d)), arr_2d.shape)
    target_pos = np.array(arr_3d.shape[1:]) / 2 # middle
    shift_by = target_pos - arr_2d_max
    shift_by = np.nan_to_num(shift_by).astype(int)
    strf_shifted = np.roll(arr_3d, shift_by, axis = (1,2))
    return strf_shifted