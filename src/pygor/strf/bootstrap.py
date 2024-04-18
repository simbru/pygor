import numpy as np#
import matplotlib.pyplot as plt
import warnings
import scipy
from joblib import Parallel, delayed
import pygor.data_helpers 
import pygor.utilities
rng = np.random.default_rng(1312)

def block_shuffle(arr_1d, block_size = None):
    """
    Shuffle the 1D array by a fixed block size in random order, with resampling (meaning the same sample can be drawn multiple times, randomly)
    """
    # Split the input 1D array into blocks of the specified size
    output = np.array(np.split(arr_1d, block_size))
    # Generate a random order for the blocks
    order = np.random.randint(-1, block_size-1, block_size)
     # Shuffle the blocks based on the random order and flatten the result
    output = output[order].ravel()
    return output
    
# def stationary_shuffle(arr_1d, max_sample_length = None, output_length = None):
#     """
#     Shuffle 1D array by a random block size in random order, with resampling,.
#     Inspired by Politis, D.N. and Romano, J.P., 1994. The stationary bootstrap. Journal of the American Statistical association, 89(428), pp.1303-1313.
#     """
#     if output_length == None:
#         output_length = len(arr_1d)
#     if max_sample_length == None:
#         max_sample_length = output_length
#     # Create  array to insert values into. This will be our bootstrapped time series
#     output = np.array([])
#     while len(output) < output_length:
#         # randomly pick an index to start sample
#         start_index = int(rng.choice(output_length-1, 1))
#         # randomly pick index to end sample (duration) Note: we don't care if it picks a value far away from the max value
#         # as it will only be able to index up to max anyways. 
#         end_index = int(rng.integers(start_index+1, max_sample_length))
#         # take that data (sample)
#         taken_sample = arr_1d[start_index:end_index]
#         # insert that into the output array and resetart loop
#         output = np.append(output, taken_sample)
#     # deal with output being longer than input
#     if len(output) > len(arr_1d):
#         output = output[:output_length]
#     # output = output.ravel()
#     return output

def circular_shuffle(arr_1d, max_sample_length = None, output_length = None, rng = None):
    """
    Shuffle 1D array by a random block size in random order, with resampling 
    """
    if output_length == None:
        output_length = len(arr_1d)
    if max_sample_length == None:
        max_sample_length = output_length
    if rng == None:
        rng = np.random.default_rng()
    # Create  array to insert values into. This will be our bootstrapped time series
    output = np.array([])
    while len(output) < output_length:
        # randomly pick an index to start sample
        start_index = rng.choice(output_length-1, 1)
        # randomly pick index to end sample (duration)
        max_index = start_index + max_sample_length
        end_index = rng.integers(start_index+1, max_index)
        # take that data (sample)
        taken_sample = np.take(arr_1d, np.arange(start_index, end_index), mode = "warp")
        # insert that into the output array and resetart loop
        output = np.append(output, taken_sample)
    # deal with output being longer than input
    if len(output) > len(arr_1d):
        output = output[:output_length]
    # output = output.ravel()
    return output

def stationary_shuffle(arr_1d, max_sample_length = None, output_length = None, rng = rng):
    """
    Shuffle 1D array by a random block size in random order, with resampling,.
    Inspired by Politis, D.N. and Romano, J.P., 1994. The stationary bootstrap. Journal of the American Statistical association, 89(428), pp.1303-1313.
    """
    if output_length == None:
        output_length = len(arr_1d)
    if max_sample_length == None:
        max_sample_length = output_length
    if rng == None:
        rng = np.random.default_rng()
    # Create  array to insert values into. This will be our bootstrapped time series
    output = np.array([])
    while len(output) < output_length:
        start_index = rng.integers(0, output_length-1)
        # randomly pick index to end sample (duration)
        max_index = start_index + max_sample_length
        # Sometimes (or all the time, depending on max_sample_length), the end_index ends up 
        # being longer than the array. In these cases, we just cap it at the maximum length 
        # of the input array. This works, because in these cases if max_sample_length = n, 
        # n will always be equal to or more than the residual max_index that was over the array length
        if max_index > len(arr_1d):
            max_index = len(arr_1d)
        end_index = rng.integers(start_index+1, max_index)
        # take that data (sample)
        taken_sample = arr_1d[start_index:end_index]
        # insert that into the output array and resetart loop
        output = np.append(output, taken_sample)
    # deal with output being longer than input
    return output[:output_length]

def bootstrap_time(arr_3d, bootstrap_n = 2500, mode_param = 2, mode = "sd", 
    collapse_space = np.ma.std, metric = np.max, plot = False, parallel = False, 
    seed = None, **kwargs): # these metrics work so leave them
    if np.ma.is_masked(arr_3d) == True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr_3d = pygor.utilities.auto_remove_border(arr_3d)
    spatial_domain = np.abs(collapse_space(arr_3d, axis = 0))
    collapse_flat_compressed = spatial_domain.flatten()
    if mode == "pixel" or mode == "pixels":
        n_top_pix = mode_param
    if mode == "sd" or mode == "SD":
        spatial_domain = scipy.stats.zscore(spatial_domain, axis = None)
        n_top_pix = (np.abs(spatial_domain) > mode_param).sum()
    # the following finds, extracs, and concatenates the timecourses of the n most weighted pixels (ignores masked values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # masked values ignored by argpartition, becomes important because if we remove border the 
        # pixel value extraction will be wrong along the time axis in arr_3d because shapes dont align
        # this saves doing that work! :) 
        flat_indeces = np.argpartition(np.abs(collapse_flat_compressed), -n_top_pix)[-n_top_pix:]
    indices_2d = np.unravel_index(flat_indeces, spatial_domain.shape)
    combined_timecourses = arr_3d[:, indices_2d[0], indices_2d[1]].ravel(order = "f")
    org_time = combined_timecourses - combined_timecourses[0] # centre data
    # Compute base stat
    fft_org = np.abs(np.fft.rfft(org_time))[1:]
    org_stat = metric(fft_org)
    def _permute_iteration(arr_3d, rng):
        # Permute
        permuted_time = stationary_shuffle(org_time, max_sample_length = len(org_time)/2, rng = rng)
        # Compute test statistic
        fft_perm = np.abs(np.fft.rfft(permuted_time))[1:]
        perm_stat = metric(fft_perm)
        return perm_stat
    # Permute spatial data within each time-step
    if parallel == False:
        perm_stat_list = []    
        for i in range(bootstrap_n):
            perm_stat = _permute_iteration(org_time, rng)
            perm_stat_list.append(perm_stat)
    if parallel == True:
        """
        Borked, dont know wy
        """
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(bootstrap_n)
        streams = [np.random.default_rng(s) for s in child_seeds]
        perm_stat_list = Parallel(n_jobs = -1, prefer ="processes")(delayed(_permute_iteration)(org_time, streams[i]) for i in range(bootstrap_n))
    sig = 1 - scipy.stats.percentileofscore(perm_stat_list, org_stat, kind = "rank") / 100
    if plot == True:
        permuted_time = stationary_shuffle(org_time)
        perm_fft = np.abs(np.fft.rfft(permuted_time))[1:]
        if "figsize" in kwargs:
            fig, ax = plt.subplots(1,5, figsize = kwargs["figsize"])
        else:
            fig, ax = plt.subplots(1,5, figsize = (24, 3))
        plot = ax[0].imshow(spatial_domain, origin = "lower")
        ax[0].set_title("Spatial filter")
        fig.colorbar(plot)
        max_index = np.unravel_index(np.argmax(np.max(np.abs(arr_3d), axis = 0)), arr_3d[0].shape)
        ax[1].plot(arr_3d[:, max_index[0], max_index[1]])
        ax[1].set_title("Temporal filter")
        ax[2].plot(org_time, label = "Avg. XY along Z")
        ax[2].plot(permuted_time, label = "Permutation")
        ax[2].set_title(f"Joined timecourses from {n_top_pix} brightest pixels")
        ax[2].legend()
        ax[3].plot(np.fft.rfftfreq(org_time.size, 1/15.625)[1:], fft_org, label = "Joined timecourses")#
        ax[3].plot(np.fft.rfftfreq(org_time.size, 1/15.625)[1:], perm_fft, label = "Permutation")#
        ax[3].set_title("FFTs")
        ax[3].legend()
        if "binsize" not in kwargs:
            ax[4].hist((perm_stat_list), 10)
        else:
            ax[4].hist((perm_stat_list), kwargs["binsize"])
        ax[4].axvline(org_stat, c = 'black', ls = "--", label = f"Percentile {np.round(sig, 5)}")
        ax[4].legend()
    return sig
    
def bootstrap_space(arr_3d, bootstrap_n = 1000, collapse_time = np.ma.std, metric = np.max, plot = False, parallel = True, seed = None,**kwargs): # these metrics work so leave them
    """
    Perform a spatial permutation test to compute p-value for a given metric on the spatial data.

    Parameters
    ----------
    arr_3d : numpy.ndarray
        A 3D matrix representing the spatio-temporal data, where the spatial dimensions are the first two axes and the temporal dimension is the third axis.

    bootstrap_n : int, optional
        Number of permutations to generate (default is 1000).

    collapse_time : function, optional
        Function to collapse the spatial data also the time axis (default is np.var, which computes the variance).

    metric : function, optional
        Metric to compute the test statistic from the collapsed spatial data (default is np.max, which computes the maximum value).

    plot : bool, optional
        If True, plot a histogram of permuted test statistics (default is False).

    **kwargs
        Additional arguments to customize the plot, such as 'binsize' to specify the number of bins in the histogram.

    Returns
    -------
    float
        The p-value indicating the percentile rank of the original test statistic compared to the permuted test statistics.

    Notes
    -----
    - The function performs a spatial permutation test to assess whether the original test statistic is significantly different from permuted versions of the spatial data.
    - The function randomly permutes the spatial data within each time-step and computes the test statistic for each permutation.
    - The p-value is calculated as the percentile rank of the original test statistic among the permuted test statistics.
    - If 'plot' is set to True, a histogram of the permuted test statistics is plotted, with the original test statistic indicated by a vertical black line.
    """
    #rng = np.random.default_rng(seed)
    # Copy array (New space in memory ensures no over-writing)
    org_arr = np.copy(arr_3d)
    # Kill border (needed for permuting data, otherwise introduce large artefacts)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message = "Destructive method. You lose data, with no way of recovering original shape!")
        org_arr = pygor.utilities.auto_remove_border(org_arr)
    # Compute base stat
    org_stat = metric(collapse_time(org_arr, axis = 0))
    # Permute spatial data within each time-step    
    def _single_permute_compute(arr_3d, rng):
        permuted_arr = rng.permuted(rng.permuted(arr_3d, axis = 2), axis = 1) # shuffled along both spatial axes but not time
        # Compute test statistic
        permuted_stat = metric(collapse_time(permuted_arr, axis = 0))
        return permuted_stat
    if parallel == False:
        permuted_stat_list = []
        for _ in range(bootstrap_n):
            permuted_stat_list.append(_single_permute_compute(org_arr, rng))
    if parallel == True:
        # The following few lines pre-allocate random number generators (streams) for parallelising without drawing from the same seed 
        # (this prevents n duplicates of the same array being perumted where n is the number of cpu_cores)
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(bootstrap_n)
        streams = [np.random.default_rng(s) for s in child_seeds]
        # Pass permuting function (and rng stream as input) to joblib and return the stats list
        permuted_stat_list = Parallel(n_jobs=-1)(delayed(_single_permute_compute)(org_arr, streams[i]) for i in range(bootstrap_n))
    sig = 1 - scipy.stats.percentileofscore(permuted_stat_list, org_stat, kind = "rank") / 100
    if plot == True:
        if "figsize" not in kwargs:
            fig, ax = plt.subplots(1, 3, figsize = (12, 2))
        else:
            fig, ax = plt.subplots(1, 3, figsize = kwargs["figsize"])
        original_plot = ax[0].imshow(collapse_time(org_arr, axis = 0), origin = "lower")
        fig.colorbar(original_plot)
        ax[0].set_title(f"Input data (collapsed)")
        permuted_plot = ax[1].imshow(collapse_time(rng.permuted(rng.permuted(arr_3d, axis = 2), axis = 1), axis = 0), origin = "lower")
        fig.colorbar(permuted_plot)
        ax[1].set_title(f"Example permutation (collapsed)")
        if "binsize" not in kwargs:
            ax[2].hist(permuted_stat_list, 10)
        else:
            plt.hist(permuted_stat_list, kwargs["binsize"])
        ax[2].axvline(org_stat, c = 'black', ls ="--", label = f"Percentile {np.round(sig, 5)}")
        ax[2].legend()
        ax[2].set_title("Boostrap distribution")
    return sig
