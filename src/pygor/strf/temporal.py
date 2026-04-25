import numpy as np
import warnings
from scipy.interpolate import interp1d

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


def polarity(
    arr,
    exclude_FirstLast=(1, 1),
    axis=-1,
    force_pol=False,
    biphasic_ratio=0.5,
):
    """
    Compute the polarity of a given numpy array along a specified axis.

    Uses an amplitude-gated heuristic: when the kernel is monophasic-dominant
    (secondary_mag / primary_mag < biphasic_ratio), polarity is taken from the
    sign of the dominant extremum. When the kernel is genuinely biphasic
    (ratio >= biphasic_ratio), polarity is taken from the temporal ordering —
    the extremum closer to spike (later index) wins.

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
    biphasic_ratio : float, optional
        Threshold on secondary/primary absolute amplitude above which the kernel
        is treated as genuinely biphasic (timing rule applies). Below threshold,
        polarity follows the sign of the dominant extremum. Defaults to 0.5.
        Pass 0.0 to always use the timing rule (original behaviour).

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
    was_masked = isinstance(arr, np.ma.MaskedArray)
    if was_masked or isinstance(arr, np.ndarray):
        if was_masked:
            arr = arr.data  # .data on masked array returns underlying ndarray
        # Time axis needs to be first or last. If it is not, move it to last index using transpose
        if axis != -1:
            arr = np.moveaxis(arr, axis, -1)
        # If everything is zeros, just return an array of zeros with appropriate shape without doing any calculations
        if np.all(arr == arr[0]):
            shape = tuple(np.array(arr.shape)[:2])
            pol_arr = np.zeros(shape)
            return pol_arr
        # Crop edges and compute both positions AND values of extrema
        try:
            cropped = arr[..., exclude_FirstLast[0] : arr.shape[-1] - exclude_FirstLast[1]]
            max_locs = np.argmax(cropped, axis=-1)
            min_locs = np.argmin(cropped, axis=-1)
            max_vals = np.max(cropped, axis=-1)
            min_vals = np.min(cropped, axis=-1)
        except ValueError:
            raise ValueError(
                "Input array is seemingly empty. Perhaps adjust exclude_FirstLast to avoid cropping all numbers."
            )

        # Amplitude-gated polarity logic.
        # primary_sign: sign of the larger-magnitude extremum.
        # ratio: |secondary| / |primary|. Below biphasic_ratio → monophasic-dominant.
        abs_max = np.abs(max_vals)
        abs_min = np.abs(min_vals)
        primary_mag = np.maximum(abs_max, abs_min)
        secondary_mag = np.minimum(abs_max, abs_min)
        # Avoid divide-by-zero; where primary_mag==0 the array is all-zero → polarity 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(primary_mag > 0, secondary_mag / primary_mag, 0.0)

        primary_is_positive = abs_max > abs_min  # sign of the dominant extremum
        # Timing rule: later extremum wins (closer to spike).
        timing_positive = max_locs > min_locs

        # Combine: use primary sign when monophasic-dominant, timing rule when biphasic.
        monophasic = ratio < biphasic_ratio
        pol_bool = np.where(monophasic, primary_is_positive, timing_positive)
    else:
        raise AttributeError(
            f"Funciton expected input as np.ndarray or np.ma.MaskedArray, not {type(arr)}"
        )
    # Convert boolean to {+1, -1}
    pol_arr = np.where(pol_bool, 1, -1)
    # Ambiguous cases (max_locs == min_locs → effectively flat / all-zero after cropping)
    if force_pol is False and pol_arr.ndim > 0:
        pol_arr[np.where(max_locs == min_locs)] = 0
    # Retain mask if input array was masked
    if was_masked:
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
        a = np.trapezoid(np.clip(timeseries, np.min(timeseries) - 1, 0))
        b = np.trapezoid(np.clip(timeseries, 0, np.max(timeseries) + 1))
        # Get the absolute values
        a = np.abs(a)
        b = np.abs(b)
        # Calculate
        return 1 - np.abs((b - a) / (a + b))  # Polarity index, zeros in divider will cause trouble

    if timeseries.ndim == 1:
        return index(timeseries)
    if timeseries.ndim > 1:
        # Time axis needs to be last index. If it is not, move it to last index
        if axis != -1:
            timeseries = np.moveaxis(timeseries, axis, -1)
        return np.apply_along_axis(index, axis, timeseries)


def spectral_centroid(timecourse_1d, sampling_rate=None):
    """
    Calculates the spectral centroid of a 1-dimensional timecourse.
    """
    # Handle edge cases
    if np.all(timecourse_1d == 0):
        n_freqs = len(timecourse_1d) // 2 + 1
        spectrum = np.full(n_freqs, np.nan)
        frequencies = np.full(n_freqs, np.nan)
        centroid = np.nan
        return spectrum, frequencies, centroid
    
    if isinstance(timecourse_1d, np.ma.MaskedArray) and np.all(timecourse_1d.mask):
        n_freqs = len(timecourse_1d) // 2 + 1
        spectrum = np.full(n_freqs, np.nan)
        frequencies = np.full(n_freqs, np.nan)
        centroid = np.nan
        return spectrum, frequencies, centroid
    
    # Calculate spectrum
    spectrum = np.abs(np.fft.rfft(timecourse_1d))
    
    # Calculate frequency bins
    if sampling_rate is None:
        frequencies = np.arange(len(spectrum))  # Arbitrary frequency bins
        warnings.warn("Param 'sampling_rate' not given, frequency bins are arbitrary.")
    else:
        frequencies = np.fft.rfftfreq(len(timecourse_1d), d=1/sampling_rate)
    
    # Calculate spectral centroid
    if np.sum(spectrum) == 0:
        centroid = np.nan
    else:
        centroid = np.sum(frequencies * spectrum) / np.sum(spectrum)
    
    return spectrum, frequencies, centroid

def only_centroid(timecourse_1d, sampling_rate=15.625):
    """Runs spectral_centroid() but returns only the centroid without spectrum array"""
    return spectral_centroid(timecourse_1d, sampling_rate=sampling_rate)[2]


def only_spectrum(timecourse_1d, sampling_rate=15.625):
    return spectral_centroid(timecourse_1d, sampling_rate=sampling_rate)[1]
    # return spectral_centroid(timecourse_1d, sampling_rate = 15.625)[1]

def find_peaktime(arr, polarity="auto"):
    """
    Return index(es) of the strongest local extremum among turning points.
    Accepts 1D (T,) or 2D (N, T) arrays. Returns int or ndarray (N,).

    polarity : {"auto", +1, -1, None}
        "auto" (default): compute polarity from the input via
        pygor.strf.temporal.polarity() and restrict turning points to that sign.
        +1 / -1: explicit polarity, restrict to matching-sign turning points.
        None: unrestricted — pick absolute strongest turning point regardless
        of sign (legacy behaviour).
        When restricted, falls back to unrestricted argmax-|y| if no
        matching turning point exists.
    """
    x = np.asarray(arr)

    def _resolve_polarity(y):
        if polarity == "auto":
            try:
                pol = int(np.asarray(pygor.strf.temporal.polarity(y)).item())
            except (ValueError, TypeError):
                pol = 0
            return pol if pol in (1, -1) else None
        if polarity in (1, -1):
            return polarity
        return None

    def strongest_extremum_1d(y):
        y = np.asarray(y)
        if y.size < 3:
            return int(np.nanargmax(np.abs(y))) if np.any(~np.isnan(y)) else 0

        d = np.diff(y)
        s = np.sign(d)

        # Handle flat regions: fill zeros by forward then backward fill
        if np.any(s == 0):
            # forward fill
            for i in range(1, s.size):
                if s[i] == 0:
                    s[i] = s[i - 1]
            # backward fill
            for i in range(s.size - 2, -1, -1):
                if s[i] == 0:
                    s[i] = s[i + 1]

        # Turning points: maxima ( + to - ) or minima ( - to + )
        tp = np.flatnonzero(((s[:-1] > 0) & (s[1:] <= 0)) | ((s[:-1] < 0) & (s[1:] >= 0))) + 1
        if tp.size:
            pol = _resolve_polarity(y)
            if pol in (1, -1):
                matching = tp[np.sign(y[tp]) == pol]
                if matching.size:
                    return int(matching[np.argmax(np.abs(y[matching]))])
                # fall through to unrestricted choice if no matching TP
            return int(tp[np.argmax(np.abs(y[tp]))])

        # Fallback: global strongest response by magnitude
        if np.any(~np.isnan(y)):
            return int(np.nanargmax(np.abs(y)))
        return 0

    if x.ndim == 1:
        return strongest_extremum_1d(x)
    if x.ndim == 2:
        return np.apply_along_axis(strongest_extremum_1d, 1, x)
    raise ValueError("arr must be 1D or 2D")


def find_peaktime_obj(strf_obj, interp_factor=1000):
    times = strf_obj.get_timecourses_dominant()
    strf_dur = strf_obj.strf_dur_ms
    strf_len = strf_obj.strfs.shape[1]
    
    # Interpolate timecourses for higher temporal precision
    interpolated_times = []
    for t in times:
        if np.all(np.isnan(t)) or len(t) < 2:
            # Handle edge cases where interpolation isn't possible
            interpolated_times.append(t)
        else:
            x_original = np.arange(len(t))
            x_interp = np.linspace(0, len(t)-1, len(t) * interp_factor)
            f = interp1d(x_original, t, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_times.append(f(x_interp))
    
    # Find peak times on interpolated data
    peak_times_indices = np.array([pygor.strf.temporal.find_peaktime(t) for t in interpolated_times])
    
    # Update scale factor to account for interpolation
    scale_factor = strf_dur / (strf_len * interp_factor)
    vals = peak_times_indices * scale_factor
    
    pass_bool = strf_obj.check_cs_pass()
    # nan where pass bools is False
    vals = np.where(pass_bool, vals, np.nan)
    # convert to time lag 
    window = strf_obj.strf_dur_ms
    vals = window - vals
    return vals

    # strf_dur = strf_obj.strf_dur_ms
    # strf_len = strf_obj.strfs.shape[1]
    # peak_times_indices = np.array([pygor.strf.temporal.find_peaktime(t) for t in times])
    # scale_factor = strf_dur / strf_len
    # vals = peak_times_indices * scale_factor
    # pass_bool = strf_obj.check_cs_pass()
    # # nan where pass bools is False
    # return np.where(pass_bool, vals, np.nan)

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
