"""
Tools for determining spatial properties of STRFs
"""

import numpy as np
import numpy.ma as ma
import math
import scipy
import skimage.measure
import skimage.segmentation
import warnings
import cv2
# Local imports
import contouring
import utilities


# Global vars
#abs_criteria_global = 5     ## Rectified STRF must pass this threshold to attempt contour draw at all 
#criteria_modifier_global = 3  ## At which multiple of metric (SD) to draw contour lines
# arth_criteria_global = 1     ## Split STRF (lower, upper) must each pass this for contours to be drawn

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

def corr_spacetime(arr_3d, convolve = False, kernel_width = 1, kernel_depth = 2, 
    pix_subsample = 1, time_subsample = 1, mode = "var", corr_mode = 'constant'):
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
    if mode == "var" or mode == 'variance' or mode == None:
        prod_funct = np.ma.var
    if mode == "std" or mode == 'stdev':
        prod_funct = np.ma.std
    if mode == "sum":
        prod_funct = np.ma.sum
    if mode == "avg" or mode == 'average':
        prod_funct = np.ma.average
    # Convolves input according to kernel and returns sum of products at each location as corrs
    if convolve == True:
        convolved_corr = scipy.ndimage.correlate(arr_3d[::time_subsample, ::pix_subsample, ::pix_subsample], weights=kernel, mode = corr_mode)
        if pix_subsample > 1: 
            convolved_corr = np.kron(convolved_corr, np.ones((time_subsample, pix_subsample, pix_subsample)))
        # Then we collapse convolved_corr by a simple mathematical operation (variance-based ones work best)
        corr_arr = prod_funct(convolved_corr, axis = 0)
        # Finally we re-scale the data to apply the same range as we had before (makes life a bit easier)
        temp_arr = prod_funct(arr_3d, axis = 0)
        corr_arr = utilities.min_max_norm(corr_arr, np.min(temp_arr), np.max(temp_arr))
    else:
        corr_arr = prod_funct(arr_3d, axis = 0)
    return corr_arr

def collapse_3d(arr_3d, zscore = True, mode = "var"):
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
            # autogen_mask = utilities.auto_border_mask(arr_3d)
            border_mask = utilities.auto_border_mask(arr_3d)
            arr_3d = np.ma.array(arr_3d, mask = border_mask)
            arr_3d = scipy.stats.zscore(arr_3d, axis = None)
        else:
            arr_3d = scipy.stats.zscore(arr_3d, axis = None)
    # Get the polarity for each pixel in STRF
    polarity_array = pixel_polarity(arr_3d)
    # Collapse space via temporal correlation 
    corr_map = corr_spacetime(arr_3d, mode = mode)
    # Put polarity back into the correlation map, 
    return corr_map * polarity_array

        ## In the old version, z-score was computed something like:
        # # First we get the mask for which the baseline should be calculated
        # pre_existing_border_width = utilities.check_border(Spatial_RF.mask)
        # border_width = 10
        # border_mask = utilities.manual_border_mask(Spatial_RF.shape, border_width + pre_existing_border_width)
        # # Then pass it to an array
        # edge_masked = np.ma.array(corr_map_pol, mask = border_mask)
        # # Then compute stats
        # edge_stdev = np.ma.std(edge_masked)
        # edge_mean = np.average(edge_masked)
        # # Spatial_RF_stdev = (corr_map / edge_stdev) * polarity_array # Z-scored
        # Spatial_RF_stdev = (corr_map_pol - edge_mean) / edge_stdev # Z-scored

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

# def _contour_arithtmatic_threshold_passfail(arr_2d, criteria, metric = np.std):
#     """
#     Helper function which simply checks if input-data meets criteria for computing
#     RF mask via scikit.image.contour(). 
#     """
#     assert arr_2d.ndim == 2 # do not allow 3d arrs since would mess with 
#     # Compute the value in the present data
#     _value = metric(np.abs(arr_2d))
#     # Check if that value is above or below criteria
#     if _value > criteria:
#         return True
#     if _value <= criteria:
#         return False

# def _contour_absolute_threshold_passfail(arr_2d, criteria = abs_criteria_global):
#     """
#     Helper function which simply checks if input-data meets criteria for computing
#     RF mask via scikit.image.contour().
#     """
#     assert arr_2d.ndim == 2 # do not allow 3d arrs (must be time-collapsed)
#     # Rectify input
#     arr_2d_rectified = np.abs(arr_2d)
#     # Check if that value is above or below criteria
#     if np.max(arr_2d_rectified) > criteria:
#         return True
#     if np.max(arr_2d_rectified) <= criteria:
#         return False

# def _contour_determine_level(arr, metric = np.std, modifier = criteria_modifier_global):
#     """
#     Helper function that computes the level at which contours used for masks are
#     drawn, in accordance with skimage.measure.find_contours(arr, level = n) where
#     n is the copmuted level value. 

#     Drawn at modifier * metric (default: 3x the STDev)
#     """
#     rectified_arr = np.abs(arr)
#     level = metric(rectified_arr) * modifier
#     level_val = level
#     return level_val

# # def contour_points_QualityControl(contours_list, conotur_total_max = 5, contour_points_min = 10):
# #     """
# #     TODO
# #     - Kick out the n smallest contours (e.g., max 8 allowed, kick out from smallest upwards)
# #     """
# #     # Make a copy and turn it into an array
# #     contours_list_new = np.copy(np.array(contours_list, dtype='object'))
# #     # Check how many points each contour has
# #     contour_points_counter = np.array([n.shape[0] for n in contours_list_new])
# #     sub_counter_threshold = np.where(contour_points_counter <= contour_points_min)[0]
# #     if any(sub_counter_threshold) is True: # any as in if any is below threshold 
# #         contours_list_new = np.delete(contours_list_new, sub_counter_threshold, axis = 0)
# #         return contours_list_new
# #     else: # If all is fine just return input list as is 
# #         return contours_list


# def contour_points_QualityControl(contours_list, contour_total_max = 5, contour_points_min = 10):
#     # Make a copy and turn it into an array
#     if len(contours_list) > 1:
#         contours_list_QC = np.array(contours_list, dtype='object')
#     else:
#         contours_list_QC = contours_list
#     # else
#     # Check how many points each contour has
#     contour_points_counter = np.array([len(n) for n in contours_list_QC])
#     # Remove contours that fall below point number threshold
#     sub_counter_threshold = np.where(contour_points_counter <= contour_points_min)[0]
#     if sub_counter_threshold.size > 0: # any as in if any is below threshold
#         contours_list_QC = np.delete(contours_list_QC, sub_counter_threshold, axis = 0)
#     # Remove contours starting with smallest if more contours than contour_points_min
#     if len(contours_list_QC) > contour_total_max:
#         # Update how many points there are now 
#         contour_points_counter = [len(n) for n in contours_list_QC]
#         total_contours = len(contour_points_counter)
#         # Partition smallest k values to begining (as indeces)
#         remove_n = total_contours - contour_total_max
#         indeces_to_delete = np.argpartition(contour_points_counter, remove_n)[:remove_n]
#         contours_list_QC = np.delete(contours_list_QC, indeces_to_delete, axis = 0)
#     return contours_list_QC


# def contour_unipolar(arr_2d, abs_criteria = abs_criteria_global, **kwargs):#, force_polarity = False):
#     """
#     Accessible helper for main function 'contour()'.

#     Finds the contour of the input 2D array by first determining the polarity 
#     of the array, then masking the array accordingly, and finally wrapping contours
#     to the masked array.
    
#     Parameters:
#     ----------
#     arr_2d (ndarray): 2D array to find the contour of.
    
#     Returns:
#     ----------
#     list: List of contour points in (row, column) format.
    
#     """
#     # If level is given, pass it along
#     if "level" in kwargs:
#         warnings.warn("Keyword arg 'level' specified, ignoring threshold criteria.",  stacklevel=2)
#         if kwargs["level"] == "scikit-default":
#             contour = skimage.measure.find_contours(arr_2d)
#         else:
#             contour = skimage.measure.find_contours(arr_2d, level = kwargs["level"])
#     # If no level is given, compute it 
#     else:
#         # Check if arr_2d passes criteria 
#         if _contour_absolute_threshold_passfail(arr_2d) == True:
#             level = _contour_determine_level(arr_2d)
#             contour = skimage.measure.find_contours(arr_2d, level = level)
#         else:
#             contour = [] #Empty contour
#             warnings.warn("Contour did not pass threshold criteria") 
#     return contour

# def _draw_contour_bipolar(arr_2d, abs_criteria):
#     """Helper function for contour_bipolar()"""
#     arr_2d_lower = np.clip(arr_2d, np.min(arr_2d), -0)
#     arr_2d_upper = np.clip(arr_2d, 0, np.max(arr_2d))
#     # Check if arr_2d as a whole passes criteria
#     if _contour_absolute_threshold_passfail(arr_2d) == True:
#         # Determine half max of abs values 
#         arithmetic_criteria = np.max(np.abs(arr_2d)) / 2
#         if _contour_arithtmatic_threshold_passfail(arr_2d_upper, criteria = arithmetic_criteria, metric = np.max) == True:
#             level_upper = _contour_determine_level(arr_2d) # note uses global determinant
#             contour_upper = skimage.measure.find_contours(arr_2d_upper, level = level_upper)
#         else:
#             contour_upper = [] #Empty contour
#             warnings.warn(f"Upper contour did not pass arithmetic threshold criteria (half abs-max = {arithmetic_criteria})", stacklevel = 2)
#         # if np.min(arr_2d) < -abs_criteria:
#         if _contour_arithtmatic_threshold_passfail(arr_2d_lower, criteria = arithmetic_criteria, metric = np.max) == True:
#             level_lower = -1 * _contour_determine_level(arr_2d) # note uses global determinant
#             contour_lower = skimage.measure.find_contours(arr_2d_lower, level = level_lower)
#         else:
#             contour_lower = [] #Empty contour
#             warnings.warn(f"Lower contour did not pass arithmetic threshold criteria (half abs-max = {arithmetic_criteria})", stacklevel = 2)
#         return contour_lower, contour_upper
#     else:
#         contour_lower = [] #Empty contour
#         contour_upper = [] #Empty contour
#         warnings.warn(f"Passed array did not meet absolute threshold criteria of {abs_criteria}", stacklevel = 2)
#         return contour_lower, contour_upper

# def contour_bipolar(arr_2d, abs_criteria = abs_criteria_global, **kwargs):
#     """Accessible helper for main function 'contour()'."""
#     # If level is given, pass it along (behaviour depends on input type)
#     if "level" in kwargs:
#         warnings.warn("Keyword arg 'level' specified, ignoring threshold criteria.",  stacklevel=2)
#         # Only allow "level" to be type tuple for contour_bipolar()
#         if type(kwargs["level"]) is float or type(kwargs["level"]) is int:
#             raise TypeError("contour_bipolar execpted keyword argument 'level' to be tuple.")
#         # if input is tuple, the 0th index is lower contour and 1th index is upper contour,
#         # passed to contour_tuple, and returned at end of script
#         if type(kwargs["level"]) is tuple:
#             contour_lower = skimage.measure.find_contours(arr_2d, level = kwargs["level"][0])
#             contour_upper = skimage.measure.find_contours(arr_2d, level = kwargs["level"][1])
#     # If no level is given, compute it
#     if "level" not in kwargs:
#         contour_lower, contour_upper = _draw_contour_bipolar(arr_2d, abs_criteria)
#     if "level" in kwargs and kwargs["level"] == "scikit-default":
#         scikit_contour = skimage.measure.find_contours(arr_2d)
#         return scikit_contour
#     # Filter the smallest contours 
#     contour_lower = contour_points_QualityControl(contour_lower)
#     contour_upper = contour_points_QualityControl(contour_upper)
#     contour_tuple = (contour_lower, contour_upper)
#     return contour_tuple

# def contour(arr_2d, abs_criteria = abs_criteria_global, expect_bipolar = True, **kwargs):
#     """
#     Finds and returns the contours of a 2D array.

#     Parameters
#     ----------
#     arr_2d : numpy.ndarray
#         A 2D numpy array representing the image or data for which contours are to be found.

#     abs_criteria : int, optional
#         A threshold value that determines the minimum absolute value of the gradient magnitude
#         required to include a point in the contour. Defaults to 45.

#     expect_bipolar : bool, optional
#         A flag indicating whether to expect a bipolar image or not. If True, `contour_bipolar`
#         is called to find the contours. Otherwise, `contour_unipolar` is called. Defaults to True.

#     **kwargs : optional
#         Additional keyword arguments to be passed on to `contour_bipolar` or `contour_unipolar`.

#     Returns
#     -------
#     contours_tuple : tuple or numpy.ndarray
#         If `expect_bipolar` is True, returns a tuple containing the positive and negative contours
#         as numpy arrays. Otherwise, returns a single numpy array containing the contours.
#     contour : numpy.ndarray
#         If `expect_bipolar` is False, returns a single numpy array containing the contours.
#     """
#     if expect_bipolar == True:
#         contours_tuple = contour_bipolar(arr_2d, abs_criteria, **kwargs)
#         return contours_tuple
#     if expect_bipolar == False:
#         contour = contour_unipolar(arr_2d, abs_criteria, **kwargs)
#         return contour 

# def contour_centroid(contours):
#     """
#     Calculates the centroid of each contour in a given list of contours.

#     Args:
#         contours (list): A list of contours. Each contour is a numpy array of (x,y) coordinates.

#     Returns:
#         numpy.ndarray: An array of shape (n,2) where n is the number of contours in the input list.
#         Each row of the array contains the (x,y) coordinates of the centroid of the corresponding contour.
#     """
#     centroids_array = np.empty((len(contours), 2))
#     for n, contour in enumerate(contours):
#         # centroids_array[n] = contour
#         centroid = np.average(contour, 0)
#         centroids_array[n] = centroid
#     return centroids_array
#     # THIS NEEDS A FIX! CAN TRY TO USE THIS IMPLEMENTATINO BUT SEEMED WEIRD https://scikit-image.org/docs/stable/api/skimage.measure.html ctrl f centroid

# def single_contour_area(contour_list):
#     """
#     Calculates the area of a single contour.

#     Parameters
#     ----------
#     contour_list : array_like
#         A list of (x, y) coordinate tuples representing the contour.

#     Returns
#     -------
#     float
#         The area of the contour.

#     Notes
#     -----
#     This function uses OpenCV's `cv2.contourArea` method to calculate the area of the contour.
#     """
#     # Expand numpy dimensions
#     c = np.expand_dims(contour_list.astype(np.float32), 1)
#     # Convert it to UMat object
#     c = cv2.UMat(c)
#     area = cv2.contourArea(c)
#     return area

# def contours_area(list_of_contour_lists):
#     """
#     Computes the area for each list of contours in a given list of lists of contours.
    
#     Parameters:
#     -----------
#     list_of_contour_lists : list of lists of numpy arrays
#         A list of lists, where each inner list contains numpy arrays that represent a contour. 
#         The contours can have different numbers of points and dimensions, but they should all be of the same dtype.
    
#     Returns:
#     --------
#     areas_array : numpy array
#         An array of the same length as the input list_of_contour_lists, containing the area for each list of contours.
#         The area is computed as the sum of the areas of all the contours in the list, using the cv2.contourArea function.
#         The areas are rounded to two decimal places.
#     """
#     # Pre-allocate memory for storing area values
#     areas_array = np.zeros(len(list_of_contour_lists))
#     # Loop through each list in the list of lists
#     for n, contour_list in enumerate(list_of_contour_lists):
#         # Copmute area for each list of contours and store it
#         areas_array[n] = np.round(single_contour_area(contour_list), 2)
#     if areas_array.size == 0: # if no contour
#         return np.array([0])
#     else: # normally
#         return areas_array

# def contours_area_bipolar(tuple_of_contours_list):
#     """
#     Computes the area of contours for two sets of contour lists: negative and positive.
    
#     Args:
#     - tuple_of_contours_list (tuple): a tuple of two lists, each containing contour lists
    
#     Returns:
#     - neg_areas (numpy.ndarray): an array containing the area values for the negative contours
#     - pos_areas (numpy.ndarray): an array containing the area values for the positive contours
#     """
#     neg_areas = contours_area(tuple_of_contours_list[0])
#     pos_areas = contours_area(tuple_of_contours_list[1])
#     return neg_areas, pos_areas

# """
# TODO:
# - Contour masking breaks at edges (becuase values there are not filled with 1s)
# - 
# """

# def _contour_mask_unipolar(contour_list, shape_tuple):
#     """
#     Generate a binary mask from a list of contours and a shape tuple.
    
#     Parameters
#     ----------
#     contour_list : list of numpy.ndarray
#         A list of contours, where each contour is a Nx2 array of (x, y) coordinates.
#     shape_tup : tuple
#         A tuple of integers representing the shape of the desired mask, in the 
#         form (height, width).
    
#     Returns
#     -------
#     numpy.ndarray
#         A binary mask of the specified shape, with 1s at the pixels corresponding 
#         to the contours and 0s elsewhere.
#     """
#     bool_array = np.zeros(shape_tuple)
#     for i in contour_list:
#         for y, x in i:
#             bool_array[math.floor(y), math.floor(x)] = 1 # force pixel-wise
#     # Fill (pixelated) contour
#     filled_bool_array = scipy.ndimage.binary_fill_holes(bool_array)
#     # # Flip it to behave as mask properly 
#     inverted_bool_array = np.invert(filled_bool_array)
#     return inverted_bool_array

# def _contour_mask_bipolar(contour_tuple, shape_tuple): # tuple needs to be size 2 and tuple of lists
#     bool_array_pos = np.zeros(shape_tuple)
#     bool_array_neg = np.zeros(shape_tuple)
#     for i in contour_tuple[0]:
#         for y, x in i:
#             bool_array_neg[math.floor(y), math.floor(x)] = 1 # force pixel-wise
#     for i in contour_tuple[1]:
#         for y, x in i:
#             bool_array_pos[math.floor(y), math.floor(x)] = 1 # force pixel-wise
#     # Fill (pixelated) contour
#     bool_array_pos = scipy.ndimage.binary_fill_holes(bool_array_pos)
#     bool_array_neg = scipy.ndimage.binary_fill_holes(bool_array_neg)
#     # Flip it to behave as mask properly 
#     bool_array_pos = np.invert(bool_array_pos)
#     bool_array_neg = np.invert(bool_array_neg)
#     return (bool_array_neg, bool_array_pos)

# def contour_mask(contour_input, shape_tuple, expect_bipolar = None):
#     """
#     Maybe it would be better to make this choise explicit... ^^^^
#     e.g., skip all the deduction...
#     """
#     """Calls the above helper functions and draws contours based on input. If 
#     a single list is given, it is deduced that unipolar contouring is wanted. If 
#     a tuple of len == 2 containing two lists is provided, it is deduced that 
#     bipolar contouring is wanted.
    
#     If output is not what is expected, user can override deduction by passing 
#     a bool to argument 'expect_bipolar' (default: None).
#     """
#     # Check that input makes sense according to what we expect
#     allowed_inputs = (tuple, list, np.ndarray)
#     if type(contour_input) not in allowed_inputs:
#         raise AttributeError(f"Argument 'contour_input' expected either a single list or a list of lists, tuple of lists, or array of lists. Instead\
#     got {type(contour_input)} as input.")
#     # If specified, just run contour masking directly
#     if expect_bipolar == True:
#         contour_mask = _contour_mask_bipolar(contour_input, shape_tuple)
#     if expect_bipolar == False:
#         contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
#     # If not specified, deduce: 
#     if expect_bipolar == None:
#         # If there are two elements in the input  
#         if len(contour_input) == 2:
#             # Need to prevent cases where a contour set which has 2 elements (can have arbitrarily many) is
#             # interpreted as being two seperate contour sets. This should work reliably, as it is a design principle 
#             # that the 2 elements should be lists if bipolar return is expected from space.contour() (which is assumed input). 
#             if all([isinstance(element, np.ndarray) for element in contour_input]) is True:
#                     contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
#             # Check that the 2 elements are lists (confirming design principle)
#             elif all([isinstance(element, list) for element in contour_input]) is False:
#                 raise AttributeError(f"Expected 'contour_input' with size 2 to contain two lists. Elements in 'contour_inputs are instead {[type(element) for element in contour_input]}.")
#             # If no error then everything conforms with design princples, go ahead with bipolar contour mask
#             else:
#                 contour_mask = _contour_mask_bipolar(contour_input, shape_tuple)
#         # Otherwise go ahead with unipolar contour mask
#         else:
#             if all([isinstance(element, np.ndarray) for element in contour_input]) is False:
#                 raise AttributeError("'contour_input' expected elements to be np.ndarray.")
#             contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
#     return contour_mask

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
    _contour = conturing.contour(arr_3d_collapsed, expect_bipolar = True, **kwargs)
    (_contour_mask_neg, _contour_mask_pos) = contouring.contour_mask(_contour, arr_3d_collapsed.shape, expect_bipolar = True, **kwargs)
    # Mask array with negative and positive mask
    neg_masked = np.ma.array(arr_3d_collapsed, mask = _contour_mask_neg)
    pos_masked = np.ma.array(arr_3d_collapsed, mask = _contour_mask_pos)
    arr = np.ma.array((neg_masked, pos_masked))
    return arr

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
    # Figure out how many frames
    frames = arr_3d.shape[axis]
    # Create contour and mask
    if "mask_tup" in kwargs:
        arr_3d_collapsed = mode(arr_3d)
        (_contour_mask_neg, _contour_mask_pos) = kwargs["mask_tup"]
        neg_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_neg, axis = 0), frames, axis = axis))
        pos_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_pos, axis = 0), frames, axis = axis))
        arr = np.ma.array((neg_masked, pos_masked))
        return arr
    # Collapse 3rd dimention 
    arr_3d_collapsed = mode(arr_3d, **kwargs)
    if level == None:
        _contour = contouring.contour(arr_3d_collapsed, expect_bipolar = True, **kwargs)
    else:
        _contour = contouring.contour(arr_3d_collapsed, expect_bipolar = True, level = (-level, level), **kwargs)
    (_contour_mask_neg, _contour_mask_pos) = contouring.contour_mask(_contour, arr_3d_collapsed.shape, expect_bipolar = True, **kwargs)
    # Mask array with negative and positive mask
    neg_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_neg, axis = 0), frames, axis = axis))
    pos_masked = np.ma.array(arr_3d, mask = np.repeat(np.expand_dims(_contour_mask_pos, axis = 0), frames, axis = axis))
    arr = np.ma.array((neg_masked, pos_masked))
    return arr

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
