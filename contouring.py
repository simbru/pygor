import numpy as np
import numpy.ma as ma
import skimage.measure
import math 
import scipy
import warnings
import matplotlib.pyplot as plt
# Local imports
import utilities

# Global vars
abs_criteria_global = 1     ## Rectified STRF must pass this threshold to attempt contour draw at all 
criteria_modifier_global = 3  ## At which multiple of metric (SD) to draw contour lines
# arth_criteria_global = 1     ## Split STRF (lower, upper) must each pass this for contours to be drawn

"""Contour generation"""
def _contour_arithtmatic_threshold_passfail(arr_2d, criteria, metric = np.std):
    """
    Helper function which simply checks if input-data meets criteria for computing
    RF mask via scikit.image.contour(). 
    """
    assert arr_2d.ndim == 2 # do not allow 3d arrs since would mess with 
    # Compute the value in the present data
    _value = metric(np.abs(arr_2d))
    # Check if that value is above or below criteria
    if _value > criteria:
        return True
    if _value <= criteria:
        return False

def _contour_absolute_threshold_passfail(arr_2d, criteria = abs_criteria_global):
    """
    Helper function which simply checks if input-data meets criteria for computing
    RF mask via scikit.image.contour().
    """
    assert arr_2d.ndim == 2 # do not allow 3d arrs (must be time-collapsed)
    # Rectify input
    arr_2d_rectified = np.abs(arr_2d)
    # Check if that value is above or below criteria
    if np.max(arr_2d_rectified) > criteria:
        return True
    if np.max(arr_2d_rectified) <= criteria:
        return False

def _contour_determine_level(arr, metric = np.std, modifier = criteria_modifier_global):
    """
    Helper function that computes the level at which contours used for masks are
    drawn, in accordance with skimage.measure.find_contours(arr, level = n) where
    n is the copmuted level value. 

    Drawn at modifier * metric (default: 3x the STDev)
    """
    rectified_arr = np.abs(arr)
    level = metric(rectified_arr) * modifier
    level_val = level
    return level_val

# def contour_points_QualityControl(contours_list, conotur_total_max = 5, contour_points_min = 10):
#     """
#     TODO
#     - Kick out the n smallest contours (e.g., max 8 allowed, kick out from smallest upwards)
#     """
#     # Make a copy and turn it into an array
#     contours_list_new = np.copy(np.array(contours_list, dtype='object'))
#     # Check how many points each contour has
#     contour_points_counter = np.array([n.shape[0] for n in contours_list_new])
#     sub_counter_threshold = np.where(contour_points_counter <= contour_points_min)[0]
#     if any(sub_counter_threshold) is True: # any as in if any is below threshold 
#         contours_list_new = np.delete(contours_list_new, sub_counter_threshold, axis = 0)
#         return contours_list_new
#     else: # If all is fine just return input list as is 
#         return contours_list


def contour_points_QualityControl(contours_list, contour_total_max = 5, contour_points_min = 10):
    # Make a copy and turn it into an array for vecotrisation/utility purposes
    contours_list_QC = np.copy(np.array(contours_list, dtype ="object"))
    # Check how many points each contour has
    contour_points_counter = np.array([len(n) for n in contours_list_QC])
    # Remove contours that fall below point number threshold
    sub_counter_threshold = np.where(contour_points_counter <= contour_points_min)[0]
    if sub_counter_threshold.size > 0: #if any is below threshold
        contours_list_QC = np.delete(contours_list_QC, sub_counter_threshold, axis = 0)
    # Remove contours starting with smallest if more contours than contour_points_min
    if len(contours_list_QC) > contour_total_max:
        # Update how many points there are now 
        contour_points_counter = np.array([len(n) for n in contours_list_QC])
        total_contours = len(contour_points_counter)
        # Partition smallest k values to begining (as indeces)
        remove_n = total_contours - contour_total_max
        indeces_to_delete = np.argpartition(contour_points_counter, remove_n)[:remove_n]
        contours_list_QC = np.delete(contours_list_QC, indeces_to_delete, axis = 0)
    # Loop through and ensure sub-arrays are lists again 
    contours_list_QC = [i.astype("float") for i in contours_list_QC] # This ensures structure is exactly 1:1 as the input structure,
    # even after all the array shannanigans above
    return contours_list_QC

def contour_unipolar(arr_2d, abs_criteria = abs_criteria_global, qc = True, **kwargs):#, force_polarity = False):
    """
    Accessible helper for main function 'contour()'.

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
    # If level is given, pass it along
    if "level" in kwargs:
        warnings.warn("Keyword arg 'level' specified, ignoring threshold criteria.",  stacklevel=2)
        if kwargs["level"] == "scikit-default":
            contour = skimage.measure.find_contours(arr_2d)
        else:
            contour = skimage.measure.find_contours(arr_2d, level = kwargs["level"])
    # If no level is given, compute it 
    else:
        # Check if arr_2d passes criteria 
        if _contour_absolute_threshold_passfail(arr_2d) == True:
            level = _contour_determine_level(arr_2d)
            contour = skimage.measure.find_contours(arr_2d, level = level)
        else:
            contour = [] #Empty contour
            warnings.warn("Contour did not pass threshold criteria") 
    if qc == True:
        contour = contour_points_QualityControl(contour)
    return contour

def _draw_contour_bipolar(arr_2d, abs_criteria, silence_warnings = True):
    """Helper function for contour_bipolar()"""
    arr_2d_lower = np.clip(arr_2d, np.min(arr_2d), -0)
    arr_2d_upper = np.clip(arr_2d, 0, np.max(arr_2d))
    # Generate empty arrays that will be filled 
    contour_lower = np.ones((250, 2)) * np.nan
    contour_upper = np.ones((250, 2)) * np.nan
    # Check if arr_2d as a whole passes criteria
    if _contour_absolute_threshold_passfail(arr_2d) == True:
        # Determine half max of abs values 
        arithmetic_criteria = np.max(np.abs(arr_2d)) / 2
        if _contour_arithtmatic_threshold_passfail(arr_2d_upper, criteria = arithmetic_criteria, metric = np.max) == True:
            level_upper = _contour_determine_level(arr_2d) # note uses global determinant
            contour_upper = skimage.measure.find_contours(arr_2d_upper, level = level_upper)
        else:
            contour_upper = [] #Empty contour
            if silence_warnings == False:
                warnings.warn(f"Upper contour did not pass arithmetic threshold criteria (half abs-max = {arithmetic_criteria})", stacklevel = 2)
        # if np.min(arr_2d) < -abs_criteria:
        if _contour_arithtmatic_threshold_passfail(arr_2d_lower, criteria = arithmetic_criteria, metric = np.max) == True:
            level_lower = -1 * _contour_determine_level(arr_2d) # note uses global determinant
            contour_lower = skimage.measure.find_contours(arr_2d_lower, level = level_lower)
        else:
            contour_lower = [] #Empty contour
            if silence_warnings == False:
                warnings.warn(f"Lower contour did not pass arithmetic threshold criteria (half abs-max = {arithmetic_criteria})", stacklevel = 2)
        return contour_lower, contour_upper
    else:
        contour_lower = [] #Empty contour
        contour_upper = [] #Empty contour
        warnings.warn(f"Passed array did not meet absolute threshold criteria of {abs_criteria}", stacklevel = 2)
        return contour_lower, contour_upper

def contour_bipolar(arr_2d, abs_criteria = abs_criteria_global, qc = True, **kwargs):
    """Accessible helper for main function 'contour()'."""
    # If level is given, pass it along (behaviour depends on input type)
    if "level" in kwargs:
        warnings.warn("Keyword arg 'level' specified, ignoring threshold criteria.",  stacklevel=2)
        # Only allow "level" to be type tuple for contour_bipolar()
        if type(kwargs["level"]) is float or type(kwargs["level"]) is int:
            raise TypeError("contour_bipolar execpted keyword argument 'level' to be tuple.")
        # if input is tuple, the 0th index is lower contour and 1th index is upper contour,
        # passed to contour_tuple, and returned at end of script
        if type(kwargs["level"]) is tuple:
            contour_lower = skimage.measure.find_contours(arr_2d, level = kwargs["level"][0])
            contour_upper = skimage.measure.find_contours(arr_2d, level = kwargs["level"][1])
    # If no level is given, compute it
    if "level" not in kwargs:
        contour_lower, contour_upper = _draw_contour_bipolar(arr_2d, abs_criteria)
    if "level" in kwargs and kwargs["level"] == "scikit-default":
        scikit_contour = skimage.measure.find_contours(arr_2d)
        return scikit_contour
    # Filter the smallest contours 
    if qc == True:
        contour_lower = contour_points_QualityControl(contour_lower)
        contour_upper = contour_points_QualityControl(contour_upper)
    contour_tuple = (contour_lower, contour_upper)
    return contour_tuple

def contour(arr_2d, abs_criteria = abs_criteria_global, expect_bipolar = True, **kwargs):
    """
    Finds and returns the contours of a 2D array.

    Parameters
    ----------
    arr_2d : numpy.ndarray
        A 2D numpy array representing the image or data for which contours are to be found.

    abs_criteria : int, optional
        A threshold value that determines the minimum absolute value of the gradient magnitude
        required to include a point in the contour. Defaults to 45.

    expect_bipolar : bool, optional
        A flag indicating whether to expect a bipolar image or not. If True, `contour_bipolar`
        is called to find the contours. Otherwise, `contour_unipolar` is called. Defaults to True.

    **kwargs : optional
        Additional keyword arguments to be passed on to `contour_bipolar` or `contour_unipolar`.

    Returns
    -------
    contours_tuple : tuple or numpy.ndarray
        If `expect_bipolar` is True, returns a tuple containing the positive and negative contours
        as numpy arrays. Otherwise, returns a single numpy array containing the contours.
    contour : numpy.ndarray
        If `expect_bipolar` is False, returns a single numpy array containing the contours.
    """
    if expect_bipolar == True:
        contours_tuple = contour_bipolar(arr_2d, abs_criteria, **kwargs)
        return contours_tuple
    if expect_bipolar == False:
        contour = contour_unipolar(arr_2d, abs_criteria, **kwargs)
        return contour 

def contour_centroid(contours):
    """
    Calculates the centroid of each contour in a given list of contours.

    Args:
        contours (list): A list of contours. Each contour is a numpy array of (x,y) coordinates.

    Returns:
        numpy.ndarray: An array of shape (n,2) where n is the number of contours in the input list.
        Each row of the array contains the (x,y) coordinates of the centroid of the corresponding contour.
    """
    centroids_array = np.empty((len(contours), 2))
    for n, contour in enumerate(contours):
        # centroids_array[n] = contour
        centroid = np.average(contour, 0)
        centroids_array[n] = centroid
    return centroids_array
    # THIS NEEDS A FIX! CAN TRY TO USE THIS IMPLEMENTATINO BUT SEEMED WEIRD https://scikit-image.org/docs/stable/api/skimage.measure.html ctrl f centroid

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
#     warnings.warn("New implementation incoming")
#     return area

def contours_area(list_of_contour_lists, scaling_factor = 1):
    """
    Computes the area for each list of contours in a given list of lists of contours.
    
    Parameters:
    -----------
    list_of_contour_lists : list of lists of numpy arrays
        A list of lists, where each inner list contains numpy arrays that represent a contour. 
        The contours can have different numbers of points and dimensions, but they should all be of the same dtype.
    
    Returns:
    --------
    areas_array : numpy array
        An array of the same length as the input list_of_contour_lists, containing the area for each list of contours.
        The area is computed as the sum of the areas of all the contours in the list, using the cv2.contourArea function.
        The areas are rounded to two decimal places.
    """
    # Pre-allocate memory for storing area values
    areas_array = np.zeros(len(list_of_contour_lists))
    # Loop through each list in the list of lists
    for n, contour_list in enumerate(list_of_contour_lists):
        # Copmute area for each list of contours and store it
        areas_array[n] = np.round(PolyArea(contour_list, scaling_factor = scaling_factor), 3)
    if areas_array.size == 0: # if no contour
        return np.array([0])
    else: # normally
        return areas_array

def contours_area_bipolar(tuple_of_contours_list, scaling_factor = 1):
    """
    Computes the area of contours for two sets of contour lists: negative and positive.
    
    Args:
    - tuple_of_contours_list (tuple): a tuple of two lists, each containing contour lists
    
    Returns:
    - neg_areas (numpy.ndarray): an array containing the area values for the negative contours
    - pos_areas (numpy.ndarray): an array containing the area values for the positive contours
    """
    neg_areas = contours_area(tuple_of_contours_list[0], scaling_factor = scaling_factor)
    pos_areas = contours_area(tuple_of_contours_list[1], scaling_factor = scaling_factor)
    return neg_areas, pos_areas

"""
TODO:
- Contour masking breaks at edges (becuase values there are not filled with 1s)
HACK: 
- If not already the case, insert first point after the last point. 
    This will ensure closer of the contour
"""

def _contour_mask_unipolar(contour_list, shape_tuple):
    """
    Generate a binary mask from a list of contours and a shape tuple.
    
    Parameters
    ----------
    contour_list : list of numpy.ndarray
        A list of contours, where each contour is a Nx2 array of (x, y) coordinates.
    shape_tup : tuple
        A tuple of integers representing the shape of the desired mask, in the 
        form (height, width).
    
    Returns
    -------
    numpy.ndarray
        A binary mask of the specified shape, with 1s at the pixels corresponding 
        to the contours and 0s elsewhere.
    """
    bool_array = np.zeros(shape_tuple)
    for i in contour_list:
        for y, x in i:
            bool_array[math.floor(y), math.floor(x)] = 1 # force pixel-wise
    # Fill (pixelated) contour
    filled_bool_array = scipy.ndimage.binary_fill_holes(bool_array)
    # # Flip it to behave as mask properly 
    inverted_bool_array = np.invert(filled_bool_array)
    return inverted_bool_array

def _contour_mask_bipolar(contour_tuple, shape_tuple): # tuple needs to be size 2 and tuple of lists
    bool_array_pos = np.zeros(shape_tuple)
    bool_array_neg = np.zeros(shape_tuple)
    for i in contour_tuple[0]:
        for y, x in i:
            bool_array_neg[math.floor(y), math.floor(x)] = 1 # force pixel-wise
    for i in contour_tuple[1]:
        for y, x in i:
            bool_array_pos[math.floor(y), math.floor(x)] = 1 # force pixel-wise
    # Fill (pixelated) contour
    bool_array_pos = scipy.ndimage.binary_fill_holes(bool_array_pos)
    bool_array_neg = scipy.ndimage.binary_fill_holes(bool_array_neg)
    # Flip it to behave as mask properly 
    bool_array_pos = np.invert(bool_array_pos)
    bool_array_neg = np.invert(bool_array_neg)
    return (bool_array_neg, bool_array_pos)

def contour_mask(contour_input, shape_tuple, expect_bipolar = None):
    """
    Maybe it would be better to make this choise explicit... ^^^^
    e.g., skip all the deduction...
    """
    """Calls the above helper functions and draws contours based on input. If 
    a single list is given, it is deduced that unipolar contouring is wanted. If 
    a tuple of len == 2 containing two lists is provided, it is deduced that 
    bipolar contouring is wanted.
    
    If output is not what is expected, user can override deduction by passing 
    a bool to argument 'expect_bipolar' (default: None).
    """
    # Check that input makes sense according to what we expect
    allowed_inputs = (tuple, list, np.ndarray)
    if type(contour_input) not in allowed_inputs:
        raise AttributeError(f"Argument 'contour_input' expected either a single list or a list of lists, tuple of lists, or array of lists. Instead\
    got {type(contour_input)} as input.")
    # If specified, just run contour masking directly
    if expect_bipolar == True:
        contour_mask = _contour_mask_bipolar(contour_input, shape_tuple)
    if expect_bipolar == False:
        contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
    # If not specified, deduce: 
    if expect_bipolar == None:
        # If there are two elements in the input  
        if len(contour_input) == 2:
            # Need to prevent cases where a contour set which has 2 elements (can have arbitrarily many) is
            # interpreted as being two seperate contour sets. This should work reliably, as it is a design principle 
            # that the 2 elements should be lists if bipolar return is expected from space.contour() (which is assumed input). 
            if all([isinstance(element, np.ndarray) for element in contour_input]) is True:
                    contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
            # Check that the 2 elements are lists (confirming design principle)
            elif all([isinstance(element, list) for element in contour_input]) is False:
                raise AttributeError(f"Expected 'contour_input' with size 2 to contain two lists. Elements in 'contour_inputs are instead {[type(element) for element in contour_input]}.")
            # If no error then everything conforms with design princples, go ahead with bipolar contour mask
            else:
                contour_mask = _contour_mask_bipolar(contour_input, shape_tuple)
        # Otherwise go ahead with unipolar contour mask
        else:
            if all([isinstance(element, np.ndarray) for element in contour_input]) is False:
                raise AttributeError("'contour_input' expected elements to be np.ndarray.")
            contour_mask = _contour_mask_unipolar(contour_input, shape_tuple)
    return contour_mask


""""Contour metrics"""
def _extend_2darray(arr_2d):
    """arr_2d is 2d list of points"""
    return np.append(arr_2d[:-1], arr_2d[:3,], axis = 0)

def rotation_direction(coordinates):
    """
    Determine the rotation direction of a sequence of 2D coordinates.
    
    Args:
        coordinates (numpy.ndarray): An array of shape (n, 2) representing the 2D coordinates.
        
    Returns:
        int: -1, 1, or 0. -1 is counterclockwise. 1 is clockwise. 0 is no rotation. 
    """
    # Ensure we have at least 3 coordinates for meaningful rotation analysis
    if len(coordinates) < 3:
        raise ValueError("At least 3 coordinates are required")
    coordinates = np.array(coordinates)
    # Compute the vectors between consecutive points
    v1 = coordinates[1:] - coordinates[:-1]
    # Roll the vectors to align them with the next points
    v2 = np.roll(v1, -1, axis=0)
    # Compute the cross products for all pairs of vectors
    cross_products = np.cross(v1, v2)
    # Calculate the total cross product
    total_cross_product = np.sum(cross_products)
    # Return
    if total_cross_product > 0:
        # Counterclockwise
        return -1
    elif total_cross_product < 0:
        # Clockwise
        return 1
    else:
        # No rotation
        return 0

def calculate_interior_angles(vertices):
    vertices = vertices[1:].astype("float") #otherwise can yield type-mismatch bug
    shifted_vertices = np.roll(vertices, shift=1, axis=0)
    angle_vectors1 = shifted_vertices - vertices
    angle_vectors2 = np.roll(vertices, shift=-1, axis=0) - vertices
    
    cross_product_mag = angle_vectors1[:, 0] * angle_vectors2[:, 1] - angle_vectors1[:, 1] * angle_vectors2[:, 0]
    dot_product = angle_vectors1[:, 0] * angle_vectors2[:, 0] + angle_vectors1[:, 1] * angle_vectors2[:, 1]
    
    angles_radians = np.arctan2(cross_product_mag, dot_product)
    negative_angles = angles_radians < 0
    angles_radians[negative_angles] += 2 * np.pi
    
    return angles_radians

def get_vertex_points(arr_2d):
    """arr_2d is 2d list of points"""
    extended_arr = _extend_2darray(arr_2d)
    return np.array([extended_arr[n:n+3] for n, vertex in enumerate(extended_arr)][:-3])

# def calc_vertex_angles(arr_2d):
#     _vertex_points = get_vertex_points(arr_2d)
#     return np.array([calculate_interior_angle(i[0], i[1], i[2]) for i in _vertex_points])    

def calc_notches_norm(arr_2d):
    return contour_metrics(arr_2d)[-1]

def get_convex_hull(arr_2d):
    """
    Cite:   Barber, C.B., Dobkin, D.P., and Huhdanpaa, H.T., "The Quickhull 
        algorithm for convex hulls," ACM Trans. on Mathematical Software,
        22(4):469-483, Dec 1996, http://www.qhull.org.
    """
    rotation_input_array = rotation_direction(arr_2d)
    _hull = scipy.spatial.ConvexHull(arr_2d) #QHull
    _convex_hull = arr_2d[_hull.vertices]
    _convex_hull = np.append(_convex_hull, np.expand_dims(_convex_hull[0], axis = 0), axis = 0)
    rotation_output_array = rotation_direction(_convex_hull)
    if rotation_input_array != rotation_output_array:
        return np.flip(_convex_hull, axis = 0)
    return _convex_hull

def PolyArea(vertices, scaling_factor = 1):
    if isinstance(vertices, np.ndarray) is False:
        vertices = np.array(vertices)
    x,y = np.array(vertices[:, 0]) * scaling_factor, vertices[:, 1]  * scaling_factor
    #Shoelace formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def PolyPerimeter(vertices):
    if isinstance(vertices, np.ndarray) is False:
        vertices = np.array(vertices)
    shifted_vertices = np.roll(vertices, shift=-1, axis=0)  # Shift vertices for edge pairing
    # Calculate edge lengths per vertex and sum them
    return np.sum(np.linalg.norm(shifted_vertices - vertices, axis=1))

def contour_metrics(arr_2d):
    """
    Calculate various metrics for a 2D contour represented as a Cartesian grid.

    Parameters
    ----------
    arr_2d : ndarray, shape (N, 2)
        An array containing the 2D coordinates of the contour points.

    Returns
    -------
    num_verteces : int
        The number of vertices in the contour.
    vertex_angles : ndarray, shape (N,)
        An array containing the interior angles at each vertex in radians.
    vertex_points : ndarray, shape (N, 2)
        An array containing the 2D coordinates of the contour vertices.
    notch_bool : ndarray, shape (N,)
        A boolean array indicating whether each vertex forms a notch (angle > pi).
    num_notches : int
        The total number of notches in the contour.
    area_contour : float
        The area enclosed by the contour.
    perimeter_contour : float
        The perimeter of the contour.

    Notes
    -----
    This function calculates various metrics for a 2D contour, including the number of vertices,
    interior angles, notch information, area, and perimeter. If there are less than 3 points in the
    contour, it returns NaN values for all metrics.

    Examples
    --------
    >>> arr_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> contour_metrics(arr_2d)
    (4, array([1.57079633, 1.57079633, 1.57079633, 1.57079633]), array([[0, 0],
           [1, 0],
           [1, 1],
           [0, 1]]), array([False, False, False, False]), 0, 1.0, 4.0)
    """
    if arr_2d.dtype == 'O': # if an object snuck through 
        arr_2d = arr_2d.astype("float")
    flip_bool = 0
    _vertex_points =  get_vertex_points(arr_2d)
    try:
        if rotation_direction(arr_2d) == -1:
            flip_bool = 1
            arr_2d = np.flip(arr_2d, axis = 1)
    except ValueError:
        warnings.warn("Not enough points (< 3) to generate metrics")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    _vertex_angles = calculate_interior_angles(arr_2d)        
    _notch_bool = _vertex_angles > np.pi
    _num_notches = _notch_bool.sum()
    _num_verteces = len(_vertex_points)
    _area_contour = PolyArea(arr_2d)
    _perimeter_contour = PolyPerimeter(arr_2d)
    return _num_verteces, _vertex_angles, _vertex_points, _notch_bool, _num_notches, _area_contour, _perimeter_contour

def complexity_metrics(arr_2d):
    """
    Calculate complexity metrics for a 2D contour represented as a Cartesian grid.

    Parameters
    ----------
    arr_2d : ndarray, shape (N, 2)
        An array containing the 2D coordinates of the contour points.

    Returns
    -------
    notches_norm : float
        The normalized number of notches in the contour.
    amplitude_poly : float
        The amplitude of complexity based on perimeter differences.
    frequency_poly : float
        The frequency of complexity based on the normalized number of notches.
    convexity_poly : float
        The convexity of complexity based on area differences.
    complexity_poly : float
        The overall complexity metric combining amplitude, frequency, and convexity.

    Notes
    -----
    This function calculates complexity metrics for a 2D contour, including notches, amplitude,
    frequency, convexity, and an overall complexity measure.

    Examples
    --------
    >>> arr_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> complexity_metrics(arr_2d)
    (0.0, 0.0, 1.0, 0.0, 0.0)
    """
    if type(arr_2d) == list:
        arr_2d = np.array(arr_2d)
    if arr_2d.dtype == 'O': # if an object snuck through 
        arr_2d = arr_2d.astype("float")
    if arr_2d.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    # Get base metrics
    _num_vertices, _vertex_angles, _vertex_points, _notch_bool, _num_notches, _area_contour, _perimeter_contour = contour_metrics(arr_2d)
    if np.isnan(_num_vertices) is True: # checking only the first but shoudl be fine 
        # If all is nan, don't bother computing the rest --> just give nans 
        return np.nan, np.nan, np.nan, np.nan, np.nan
    # Create a convex hull and derive area/perimeter for both contour and its convex hull
    _convex_hull = get_convex_hull(arr_2d)
    _area_convex_hull = PolyArea(_convex_hull)
    # _perimeter_contour = PolyPerimeter(arr_2d)
    _perimeter_convex_hull = PolyPerimeter(_convex_hull)
    # Derive complexity measures
    _nothces_norm = _num_notches / (_num_vertices - 3)
    _amplitude_poly = (_perimeter_contour - _perimeter_convex_hull) / _perimeter_contour
    _frequency_poly = 16*(_nothces_norm-0.5)**4 - 8*(_nothces_norm-0.5)**2 + 1
    _convexity_poly = (_area_convex_hull - _area_contour) / _area_convex_hull
    _complexity_poly = 0.8 * _amplitude_poly * _frequency_poly + 0.2 * _convexity_poly
    return _nothces_norm, _amplitude_poly, _frequency_poly, _convexity_poly, _complexity_poly

def measures_demo(arr_2d, vertex_num = -1):
    # cont = arr_2d
    num_verteces, vertex_angles, vertex_points, notch_bool, num_notches, _area_contour, _perimeter_contour = contour_metrics(arr_2d)
    nothces_norm, amplitude_poly, frequency_poly, convexity_poly, complexity_poly = complexity_metrics(arr_2d)
    plt.scatter(arr_2d[:, 0], arr_2d[:, 1])
    plt.scatter(arr_2d[1:, 0][notch_bool == 1], arr_2d[1:, 1][notch_bool == 1], c = "orange", label = "notches")
    plt.plot(arr_2d[:, 0], arr_2d[:, 1])
    plt.scatter(arr_2d[:, 0][0], arr_2d[:, 1][0], c ="red", marker ="v", label = "start point")
    if vertex_num != -1:
        point = vertex_num
        plt.scatter(vertex_points[point][:, 0], vertex_points[point][:, 1], facecolors='none',edgecolors=['green', 'limegreen', 'green'],s =100, lw=3, label = "curr_vertex")
    convex_hull = get_convex_hull(arr_2d)
    plt.plot(convex_hull[:, 0], convex_hull[:, 1], "--k", label = "convex_hull")
    plt.title(f"Complexity = {np.round(complexity_poly, 4)}")
    plt.gca().set_aspect('equal')
    plt.legend()

def _metrics_multicontour(list_of_list):
    metrics = []
    for i in list_of_list:
        complexity = complexity_metrics(i)[-1]
        metrics.append(complexity)
    if metrics == []:
        metrics = [np.nan]
    return np.array(metrics)

def _complexity_allcontours(contours_tup):
    roi_results = []
    for n, (i, j) in enumerate(zip(contours_tup[:, 0], contours_tup[:, 1])):
        metric_tup = [_metrics_multicontour(i), _metrics_multicontour(j)]
        roi_results.append(metric_tup)
    return roi_results

def complexity_weighted(contours_tup, contours_areas):
    final_complexity_metric = []
    for n, ((neg_comp, pos_comp), (neg_areas, pos_areas)) in enumerate(zip(_complexity_allcontours(contours_tup), contours_areas)):
        area_weighted_pos_comp = np.nan
        area_weighted_neg_comp = np.nan
        if np.all(neg_areas != np.nan) and np.all(neg_areas>0):
            area_weighted_neg_comp = np.average(neg_comp, weights = np.nan_to_num(neg_areas))
            # print(n, "before:", neg_comp, "after:", np.round(area_weighted_neg_comp, 4))
        if np.all(pos_areas != np.nan) and np.all(pos_areas>0):
            area_weighted_pos_comp = np.average(pos_comp, weights = np.nan_to_num(pos_areas))
            # print(n, "before:", pos_comp, "after:", np.round(area_weighted_pos_comp, 4))
        final_complexity_metric.append((area_weighted_neg_comp, area_weighted_pos_comp))
    return np.array(final_complexity_metric)

