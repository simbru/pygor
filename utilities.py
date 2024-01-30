import numpy as np
import math
import skimage
import warnings

# explicit function to normalize array
def np_describe(arr):
    stats = {
        "mean" : np.average(arr),
        "std" : np.std(arr),
        "var" : np.var(arr),
        "minmax" : (np.min(arr), np.max(arr)),
        "range"  : np.ptp(arr)
    }
    return stats

def min_max_norm(arr, t_min, t_max):
    """ Min-max scale a 1D or 2D array. >2D not tested. 

    Parameters
    ----------
    arr : numpy array
        Array to min-max scale
    t_min : int
        lower value
    t_max : int
        upper value

    Returns
    -------
    numpy array
        Min-max normalised array
    """    
    norm_arr = []
    diff = t_max - t_min
    diff_arr = np.ma.max(arr) - np.ma.min(arr)    
    for i in arr:
        temp = (((i - np.ma.min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    if isinstance(arr, np.ma.core.MaskedArray):
        return np.ma.array(norm_arr, mask = arr.mask)
    else:
        return np.array(norm_arr)

def auto_border_mask(array):
    # Mark where in array there are 0s (expected border value)
    label_zeros = np.where(array == 0, 1, 0).astype(bool)
    # But 0 COULD occur elsewhere, so ensure we remove "small holes" by upping threshold
    mask = skimage.morphology.remove_small_holes(label_zeros, area_threshold=100) # very unlikely that a 10x10 region will be 0s
    return mask                                                                   # by chance 

def numpy_fillna(data):
    if isinstance(data, np.ndarray) is False:
        data = np.array(data, dtype = object)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

# def manual_border_mask(array_shape, border_width):
#     """manual_border_mask Takes a tuple (shape of mask) and an intiger (width of
#     border mask) and creates a boolean mask which can be passed to np.ma.array(x, mask = _mask)
#     to mask an array around its edges. 

#     Parameters
#     ----------
#     array_shape : tuple
#         Shape of output masks 
#     border_width : int
#         Width of symetrical border around edges of mask

#     Returns
#     -------
#     Boolean array
#         Array to be passed as mask to np.ma module
#     """    
#     mask = np.ones(array_shape)
#     mask[:border_width]     = 0
#     mask[-border_width:]    = 0
#     mask[:, -border_width:] = 0
#     mask[:, :border_width]  = 0
#     return mask

def manual_border_mask(array_shape2d, unmasked_shape2d):
    """
    Create a manual border mask for a 2D array.

    Parameters:
    -----------
    array_shape2d : tuple
        The shape of the 2D array to create the mask for.
    unmasked_shape2d : tuple
        The shape of the unmasked region in the center of the array.

    Returns:
    --------
    mask : ndarray
        A 2D NumPy array representing the mask, where 1 represents the masked region
        and 0 represents the unmasked region in the center.

    Notes:
    ------
    The mask is centered on the middle of the array.
    """
    mask = np.ones(array_shape2d)
    y_crop_from_centre = math.floor(unmasked_shape2d[0]/2)
    x_crop_from_centre = math.floor(unmasked_shape2d[1]/2)
    mask[math.ceil(array_shape2d[0]/2) - y_crop_from_centre : math.ceil(array_shape2d[0]/2) + y_crop_from_centre, 
        math.ceil(array_shape2d[1]/2) - x_crop_from_centre : math.ceil(array_shape2d[1]/2) + x_crop_from_centre] = 0
    return mask

def auto_remove_border(array):
    """ Utility function for automatically removing borders from 2D or 3D array. 
    Lots of dirty solutions in here so use carefully.
    WARNING: Destructive method. You lose original data! 

    Note that this is intended for borders represented as 0s, and ma.masked_arrays are not considered. 

    Parameters
    ----------
    array : numpy array
        input array

    Returns
    -------
    numpy array
        original array without estimated borders
    """
    warnings.warn("Destructive method. You lose data, with no way of recovering original shape!") 
    #crop_mask = (auto_border_mask(array)  * -1).astype(bool)
    # Remove border based on crop_mask (True means keep, False means remove)
    ## The logic is: find geometrical centre, find extreme points in 4 directions, and count 0s between extreme and center
    if array.ndim == 3: # sometimes we want to include time
        array_shape = array[0].shape # shape should be stable with time (no ragged arrays allowed)
        was_2d = False
    if array.ndim == 2: # sometimes we don't
        array_shape = array.shape
        was_2d = True
        array = np.expand_dims(array, axis = 0) # super lazy way of getting around dimensionality
    if np.ma.is_masked(array) is True:
        # Drop mask in this instance
        array = array.data
    centre_coordinate = (round(array_shape[0] / 2), round(array_shape[1] / 2))
    upper_border_width = np.count_nonzero(array[0][:centre_coordinate[0], centre_coordinate[1]:centre_coordinate[1]+1] == 0) 
    lower_border_wdith = np.count_nonzero(array[0][centre_coordinate[0]:, centre_coordinate[1]:centre_coordinate[1]+1] == 0) 
    left_border_width  = np.count_nonzero(array[0][centre_coordinate[0]:centre_coordinate[0]+1, centre_coordinate[1]:] == 0) 
    right_border_width = np.count_nonzero(array[0][centre_coordinate[0]:centre_coordinate[0]+1, :centre_coordinate[1]] == 0) 
    # Again lazy solution but it works so whatever
    if was_2d == True:
        return np.copy(np.squeeze(array[:, upper_border_width:-lower_border_wdith, right_border_width:-left_border_width], axis = 0))
    if was_2d == False:
        return np.copy(array[:, upper_border_width:-lower_border_wdith, right_border_width:-left_border_width])

def check_border(array_mask, expect_symmetry = True):
    """
    Calculate the widths of the four borders (upper, lower, left, right) of a given 
    2D array_mask. If expect_symmetry is True, the function checks that the four 
    borders have equal widths, and returns the width of one of them. Otherwise, the
    function returns a tuple with the widths of the four borders.

    Parameters:
    ----------
    array_mask: 
        A 2D numpy array with boolean values (True or False).
    expect_symmetry: 
        A boolean value indicating whether the function should check that the four borders have equal widths. Default value is True.
    Returns:
    -------
    If expect_symmetry is True, 
        returns an integer representing the width of one of the four borders.
    If expect_symmetry is False, 
        returns a tuple of four integers representing the widths of the four borders.
    Raises:
    -------
    ValueError: 
        If expect_symmetry is True and the four borders have different widths.
    """
    centre_coordinate = (round(array_mask.shape[0] / 2), round(array_mask.shape[1] / 2))
    upper_border_width = np.count_nonzero(array_mask[:centre_coordinate[0], centre_coordinate[1]:centre_coordinate[1]+1]) 
    lower_border_wdith = np.count_nonzero(array_mask[centre_coordinate[0]:, centre_coordinate[1]:centre_coordinate[1]+1]) 
    left_border_width  = np.count_nonzero(array_mask[centre_coordinate[0]:centre_coordinate[0]+1, centre_coordinate[1]:]) 
    right_border_width = np.count_nonzero(array_mask[centre_coordinate[0]:centre_coordinate[0]+1, :centre_coordinate[1]])
    if expect_symmetry == True:
        if right_border_width == left_border_width == lower_border_wdith == upper_border_width:
            border_width = right_border_width
            # pass
        else:
            raise ValueError(
                "The four borders are not the same widths. Instead, they"
                f"have widths: ({upper_border_width}, {lower_border_wdith}, {left_border_width},"
                f" {right_border_width}) for borders (upper, lower, left, right). Consider setting"
                f"'expect_symmetry = False' or passing mask manually.")
        return border_width 
    else:
        return (upper_border_width, lower_border_wdith, left_border_width, right_border_width)

def make2DGaussian(size, fwhm, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def pixel_kernels(arr_3d, purge_mask = False):
    """This function reshapes a 3D array with dimensions (time, y, x) into a 
    2D array of pixel time-series by removing the mask (if present) and 
    collapsing the spatial dimensions. The purge_mask parameter controls whether
    to remove array mask."""
    if np.ma.is_masked(arr_3d) is True and purge_mask is True:
        arr_3d = arr_3d.compressed().reshape(len(arr_3d), -1)
        return arr_3d.reshape(len(arr_3d), -1)
    if np.ma.is_masked(arr_3d) is False and purge_mask is True:
        warnings.warn("Array was not masked, running regular operation without purging mask.")
        return arr_3d.reshape(len(arr_3d.data), -1)
    else:
        return arr_3d.reshape(len(arr_3d), -1)


def arrcoords(shape_tuple):
    array_coordinates = np.empty((shape_tuple[0] * shape_tuple[1], 2))
    counter = 0
    for y in reversed(range(shape_tuple[0])): # count backwards to account for QDSpy coordinates
        for x in range(shape_tuple[1]):
            array_coordinates[counter, 0], array_coordinates[counter, 1]= int(x), int(y)
            counter += 1  
    return array_coordinates

def multicolour_reshape(arr, n_colours):
    """
    Reshapes the input array into a multi-color representation by dividing the first axis into 'n_colours' equal parts.

    The function reshapes the input 'arr' into a new shape that represents 'n_colours' different colors. The number of rows in
    the original 'arr' must be divisible by 'n_colours' to ensure an equal split.

    Parameters:
        arr (numpy.ndarray): The input array to be reshaped.
        n_colours (int): The number of colors to represent in the output.

    Returns:
        numpy.ndarray: The reshaped array representing 'n_colours' colors. The first axis will have 'n_colours' sub-arrays,
        each containing 'org_shape[0] / n_colours' rows of the original 'arr'. The remaining dimensions will be preserved
        from the original array.

    Raises:
        AssertionError: If the number of rows in the original array is not divisible by 'n_colours', an AssertionError is raised.
            This ensures that the input can be evenly split into 'n_colours' colors.

    Example:
        arr = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]])
        n_colours = 2
        result = multicolour_reshape(arr, n_colours)
        print(result)
        Output:
        [ [[1, 2, 3],
           [4, 5, 6]],
          [[7, 8, 9],
           [10, 11, 12]] ]
        
        In this example, the input array 'arr' is reshaped into a multi-color representation with 2 colors. The function
        splits the original array into two sub-arrays, each containing two rows, resulting in a new shape of (2, 2, 3).

    Note:
        - The function assumes that 'arr' is a numpy array.
        - The function requires 'n_colours' to be a positive integer.
        - The function preserves the order of elements in the original array when reshaping.
    """
    org_shape = arr.shape
    assert org_shape[0] % n_colours == 0
    new_shape = (n_colours, int(org_shape[0]/n_colours), * org_shape[1:])
    return arr.reshape(new_shape, order = 'f')
    return np.ma.reshape(arr, new_shape, order = 'f')

# def profile_me(function)

def init_profiler(input_funct, *args):
    import cProfile
    # Create a cProfile object
    profiler = cProfile.Profile()
    # Start profiling
    profiler.enable()
    # Call the function you want to profile
    input_funct(*args)
    # Stop profiling
    profiler.disable()
    # Print the profiling results
    profiler.print_stats(sort='tottime')
    return profiler