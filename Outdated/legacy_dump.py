def _draw_contour_legacy(arr_2d):
    # Check if arr_2d passes criteria 
    ## Two criteria: First, the total array must have STD over 6 and the 
    ## unipolar version of the array must have STD over 2
    if _contour_threshold_passfail(arr_2d, 6, np.std) == True:
        if _contour_threshold_passfail(arr_2d_lower, 5, np.std) == True:
            level_lower = _contour_determine_level(arr_2d_lower, modifier = 4)
            contour_lower = skimage.measure.find_contours(arr_2d_lower, level = level_lower)
        else:
                contour_lower = [] #Empty contour
        if _contour_threshold_passfail(arr_2d_upper, 5, np.std) == True: # Do this per pixel 
            level_upper = _contour_determine_level(arr_2d_upper, modifier = 4)
            contour_upper = skimage.measure.find_contours(arr_2d_upper, level = level_upper)
        else:
            contour_upper = [] #Empty contour
    else:
        contour_lower = [] #Empty contour
        contour_upper = [] #Empty contour
        warnings.warn("Contour did not pass threshold criteria")     
    return contour_lower, contour_upper


def _legacy_contour(arr_2d, level = None, force_polarity = False):
    """
    Finds the contour of the input 2D array by first determining the polarity 
    of the array, then masking the array accordingly, and finally wrapping contours
    to the masked array.
    
    Parameters:
    ----------
    arr_2d (ndarray): 2D array to find the contour of.
    
    Returns:
    ----------
    list: List of contour points in (row, column) format.
    
    """
    if force_polarity == True:
        neg_mask, pos_mask = get_polarity_masks(arr_2d)
        polarity = polarity_2d(arr_2d)
        if polarity == 1:
            contour = skimage.measure.find_contours(max_mask) # make sure u give the right mask depending on polarity
        if polarity == -1:
            contour = skimage.measure.find_contours(min_mask) # make sure u give the right mask depending on polarity
    else:
        contour = skimage.measure.find_contours(arr_2d)
    if level is not None:
        contour = skimage.measure.find_contours(arr_2d, level = level)
    if force_polarity == True and level is not None:
        raise AttributeError("Pick either 'level' or 'false_polarity'")
    return contour