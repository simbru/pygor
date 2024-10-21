import numpy as np
import warnings

# Local imports
import pygor.strf.spatial


def extract_timecourse(arr_3d, level=None, centred=True):
    """
    Extracts a time course from a 3D array, averaged along both spatial axes.

    Parameters
    ----------
    arr_3d : ndarray
        A 3D numpy array representing a spatiotemporal stimulus.
    level : float, optional
        A value used to mask the 3D array. Only elements with absolute
        values greater than `level` will be included in the average. If
        `None` (default), no masking is applied.
    centred : bool, optional
        If True (default), the resulting time course is centred so that
        its first value is 0. If False, the time course is returned as-is.

    Returns
    -------
    time_course : MaskedArray
        A 2D numpy array of shape (2, T) representing the averaged time course.
        The first row corresponds to the negative part of the time course, and
        the second row corresponds to the positive part. T is the length of the
        time course (i.e., the number of time points).

    Notes
    -----
    This function applies a mask to the 3D array before averaging its values
    along the spatial axes. If a `level` is provided, the mask is computed by
    keeping only the elements with absolute value greater than `level`. If no
    `level` is provided, all elements are included in the average.

    Examples
    --------
    >>> arr_3d = np.random.rand(3, 4, 5)
    >>> extract_timecourse(arr_3d).shape
    (2, 4)
    >>> extract_timecourse(arr_3d, level=0.5).shape
    (2, 4)
    >>> extract_timecourse(arr_3d, centred=False).shape
    (2, 5)
    """
    # Apply mask
    if level == None:
        masked_strf = pygor.strf.spatial.rf_mask3d(arr_3d)
    else:
        masked_strf = pygor.strf.spatial.rf_mask3d(arr_3d, level=level)
    # Average remaining values along boht axes of pygor.strf.spatial
    time_course_neg = np.ma.average(masked_strf[0], axis=(1, 2))
    time_course_pos = np.ma.average(masked_strf[1], axis=(1, 2))
    if centred == True:
        time_course_neg = time_course_neg - time_course_neg[0]
        time_course_pos = time_course_pos - time_course_pos[0]
    return np.ma.array([time_course_neg, time_course_pos])


def polarity(arr, exclude_FirstLast=(1, 1), axis=-1, force_pol=False):
    """
    Compute the polarity of a given numpy array along a specified axis.

    Parameters
    ----------
    arr : numpy.ndarray or numpy.ma.MaskedArray
        The input array for which to compute the polarity. If a masked array is passed,
        the polarity will be calculated only for the unmasked elements.
    exclude_FirstLast : tuple of int, optional
        The number of samples to exclude from the beginning and end of the time axis.
        Defaults to (1, 1).
    axis : int, optional
        The axis along which to compute the polarity. Can be 0 or -1.
        Defaults to -1.
    force_pol : bool, optional
        Whether to force the polarity calculation for arrays with identical maximum and minimum values.
        Defaults to False.

    Returns
    -------
    numpy.ndarray or numpy.ma.MaskedArray
        An array of polarity values, where 1 represents positive polarity and -1 represents negative polarity.
        If the input array contains masked values, the output will also be masked at the same locations.

    Raises
    ------
    AttributeError
        If the input array is not a numpy array or a numpy masked array.

    ValueError
        If the input array is empty after excluding the first and last samples.
    """
    # Check that input makes sense
    if (
        isinstance(arr, np.ma.MaskedArray) is True
        or isinstance(arr, np.ndarray) is True
    ):
        if isinstance(arr, np.ndarray) is True:
            # Force data (bug testing)
            arr = arr.data
        # Time axis needs to be first or last. If it is not, move it to last index using transpose
        if axis != -1:
            arr = np.moveaxis(arr, axis, -1)
        # If everything is zeros, just return an array of zeros with appropriate shape without doing any calculations
        if np.all(arr == arr[0]):
            shape = tuple(np.array(arr.shape)[:2])
            pol_arr = np.zeros(shape)
            return pol_arr
        # Get the positions of maximum and minimum (cropped by time as specified in exlcude_PrePost)
        try:
            max_locs = np.ma.argmax(
                arr[..., exclude_FirstLast[0] : arr.shape[-1] - exclude_FirstLast[1]],
                axis=-1,
            )
            min_locs = np.ma.argmin(
                arr[..., exclude_FirstLast[0] : arr.shape[-1] - exclude_FirstLast[1]],
                axis=-1,
            )
        except ValueError:
            raise ValueError(
                "Input array is seemingly empty. Perhaps adjust exclude_FirstLast to avoid cropping all numbers."
            )
        # Get a boolean array of where maximum comes before minimum
        pol_arr = max_locs > min_locs
    else:
        raise AttributeError(
            f"Funciton expected input as np.ndarray or np.ma.MaskedArray, not {type(arr)}"
        )
    # We are assigining polarity, so boolean array needs to be converted to polarity array (e.g, 0 should be -1)
    pol_arr = np.where(pol_arr == True, 1, -1)
    # In some rare cases we might need to force a polarity (for example an array multiplied by its polarties,
    # without loosing the underlying data (e.g., if 0 * n == 0, we lose n). The below allows overriding the
    # behaviour where if min and max locs are the same, 0 is put in place
    if force_pol is False and pol_arr.ndim > 0:
        pol_arr[np.where(max_locs == min_locs)] = 0
    # Retain mask if input array contained mask
    if (
        isinstance(arr, np.ma.MaskedArray) == True
        or isinstance(arr, np.ma.MaskedArray) is True
    ):  # and np.all(arr.mask != False):
        pol_arr = np.ma.array(
            data=pol_arr, mask=arr[..., 0].mask
        )  # take mask from first frame
    return pol_arr


def biphasic_index(timeseries, axis=-1):
    """
    Calculate the biphasic index of a given timeseries.

    Parameters
    ----------
    timeseries : array_like
        Input array containing the timeseries data. If a 2D array is provided, the time axis must be the last axis.
    axis : int, optional
        The axis along which to apply the function. Defaults to -1.

    Returns
    -------
    biphasic_index : ndarray
        An array containing the biphasic index values calculated from the input timeseries array.

    Raises
    ------
    ValueError
        If the input array has less than one dimension or the time axis is not the last axis.

    Notes
    -----
    The biphasic index is a measure of the extent to which a timeseries is biphasic in nature. It is calculated as
    the absolute difference between the area under the curve of the negative and positive components of the timeseries,
    divided by their sum. The resulting value ranges from -1 to 1, where values closer to -1 indicate a more negative
    biphasic index and values closer to 1 indicate a more positive biphasic index.

    """

    def index(timeseries):
        if timeseries[0] != 0:
            timeseries = timeseries - timeseries[0]
        # Get area under curve for negative and positive components
        a = np.trapz(np.clip(timeseries, np.min(timeseries) - 1, 0))
        b = np.trapz(np.clip(timeseries, 0, np.max(timeseries) + 1))
        # Get the absolute values
        a = np.abs(a)
        b = np.abs(b)
        # Calculate
        return (b - a) / (a + b)  # Polarity index, zeros in divider will cause trouble

    if timeseries.ndim == 1:
        return index(timeseries)
    if timeseries.ndim > 1:
        # Time axis needs to be last index. If it is not, move it to last index using transpose
        if axis != -1:
            timeseries = np.moveaxis(timeseries, axis, -1)
        return np.apply_along_axis(index, axis, timeseries)


def spectral_centroid(timecourse_1d, sampling_rate=None):
    """
    Calculates the spectral centroid of a 1-dimensional timecourse.

    Parameters
    ----------
    timecourse_1d : 1-dimensional numpy array or masked array
        The timecourse to calculate the spectral centroid from.
    sampling_rate : float or None, optional
        The sampling rate of the timecourse. If None, the frequency bins are arbitrary. Default is None.

    Returns
    -------
    spectrum : numpy array
        The spectrum of the timecourse.
    norm_freq : numpy array
        The normalized frequency bins.
    centroid : float
        The spectral centroid of the timecourse.

    Notes
    -----
    The spectral centroid is a measure of the center of mass of the spectrum. It indicates where the "average" frequency of the spectrum is located.

    If the input `timecourse_1d` is all zeros or a masked array with all elements masked, the function returns arrays of NaNs for `spectrum`, `weighted_spectrum`, and `centroid`.

    If `sampling_rate` is not given, a warning is issued and the frequency bins are arbitrary.
    """
    if np.all(timecourse_1d == 0):
        spectrum = np.empty(int(len(timecourse_1d) / 2 + 1))
        spectrum[:] = np.nan
        weighted_spectrum = np.empty(int(len(timecourse_1d) / 2 + 1))
        weighted_spectrum[:] = np.nan
        centroid = np.nan
        return spectrum, weighted_spectrum, centroid
    if isinstance(timecourse_1d, np.ma.MaskedArray) == True and np.all(
        timecourse_1d.mask == True
    ):
        spectrum = np.empty(int(len(timecourse_1d) / 2 + 1))
        spectrum[:] = np.nan
        weighted_spectrum = np.empty(int(len(timecourse_1d) / 2 + 1))
        weighted_spectrum[:] = np.nan
        centroid = np.nan
        # Don't bother masking these, the nans will be sufficient (i think, look for bugs as consequence)
        return spectrum, weighted_spectrum, centroid
        # return (np.ma.array(spectrum, mask = True), np.ma.array(weighted_spectrum, mask = True),
        # np.ma.array(centroid, mask = True))
    # ^ Just return array of nans if the above ifs are applicable
    else:
        spectrum = np.abs(np.fft.rfft(timecourse_1d).real)
        # Sanity test
        auc = np.trapz(spectrum)
        should_eqaul_1 = np.trapz(spectrum / auc)
        should_eqaul_1 = np.real_if_close(should_eqaul_1)
        assert np.isclose(should_eqaul_1, 1)
        # Calculate as ratio
        # norm_spectrum = spectrum / sum(spectrum) # probability mass function, are the weights
        if sampling_rate == None:
            norm_freq = np.linspace(0, len(spectrum), len(spectrum))
            weighted_spectrum = spectrum * norm_freq
            warnings.warn(
                "Param 'sampling_rate' not given, frequency bins are arbitrary."
            )
        else:
            norm_freq = np.linspace(0, sampling_rate / 2, len(spectrum))
            weighted_spectrum = spectrum * norm_freq
        # Get spectral centroid
        centroid = np.sum(weighted_spectrum) / np.sum(spectrum)
    return spectrum, norm_freq, centroid


def only_centroid(timecourse_1d, sampling_rate=15.625):
    """Runs spectral_centroid() but returns only the centroid without spectrum array"""
    return spectral_centroid(timecourse_1d, sampling_rate=sampling_rate)[2]


def only_spectrum(timecourse_1d, sampling_rate=15.625):
    return spectral_centroid(timecourse_1d, sampling_rate=sampling_rate)[1]
    # return spectral_centroid(timecourse_1d, sampling_rate = 15.625)[1]


# def

# def


# OLD RUBBISH def spectral_centroid(timecourse_1d, sampling_rate = None):
#     """the weighted mean of the frequencies present in the signal, determined
#     using a Fourier transform, with their magnitudes as the weights. Note that
#     the output is relative, so to get corresponding frequency bins please multiply
#     centroid by sample rate"""
#     if np.all(timecourse_1d == 0):
#         norm_spectrum =  np.empty(len(timecourse_1d))
#         norm_spectrum[:] = np.nan
#         norm_freq = np.empty(len(timecourse_1d))
#         norm_freq[:] = np.nan
#         centroid = np.nan
#         return norm_spectrum, norm_freq, centroid
#     if isinstance(timecourse_1d, np.ma.MaskedArray) == True and np.all(timecourse_1d.mask == True):
#         norm_spectrum =  np.empty(len(timecourse_1d))
#         norm_spectrum[:] = np.nan
#         norm_freq = np.empty(len(timecourse_1d))
#         norm_freq[:] = np.nan
#         centroid = np.nan
#         return (np.ma.array(norm_spectrum, mask = True), np.ma.array(norm_freq, mask = True),
#         np.ma.array(centroid, mask = True))
#         # ^ Just return array of nans if the above elifs are applicable
#     else:
#         spectrum = np.abs(np.fft.rfft(timecourse_1d).real)
#         # Sanity test
#         auc = np.trapz(spectrum)
#         should_eqaul_1 = np.trapz(spectrum / auc)
#         should_eqaul_1 = np.real_if_close(should_eqaul_1)
#         assert np.isclose(should_eqaul_1, 1)
#         # Calculate as ratio
#         norm_spectrum = spectrum / sum(spectrum) # probability mass function, are the weights
#         if sampling_rate == None:
#             norm_freq = np.linspace(0, len(spectrum), len(spectrum))
#             warnings.warn("Param 'sampling_rate' not given, frequency bins are arbitrary." )
#         else:
#             norm_freq = np.linspace(0, sampling_rate/2, len(spectrum))
#         # Get spectral centroid
#         centroid = np.sum(norm_spectrum * norm_freq)
#     return norm_spectrum, norm_freq, centroid
