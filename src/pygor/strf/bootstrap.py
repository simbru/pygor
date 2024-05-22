import numpy as np#
import matplotlib.pyplot as plt
import warnings
import scipy
from joblib import Parallel, delayed
import warnings
import pygor.data_helpers 
import pygor.utilities
import copy
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
    if output_length is None:
        output_length = len(arr_1d)
    if max_sample_length is None:
        max_sample_length = output_length
    if rng is None:
        rng = np.random.default_rng()
    
    output = np.empty(output_length)
    index = 0
    while index < output_length:
        start_index = rng.integers(0, len(arr_1d) - 1)
        max_index = min(start_index + max_sample_length, len(arr_1d))
        end_index = rng.integers(start_index + 1, max_index)
        length = min(end_index - start_index, output_length - index)
        output[index:index + length] = arr_1d[start_index:start_index + length]
        index += length
    
    return output
# def bootstrap_time(arr_3d, bootstrap_n = 2500, mode_param = 3, mode = "sd", 
#     collapse_space = np.ma.std, metric = np.max, plot = False, parallel = False, 
#     seed = None, **kwargs): # these metrics work so leave them
#     if np.ma.is_masked(arr_3d) == True:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             arr_3d = pygor.utilities.auto_remove_border(arr_3d)
#     spatial_domain = np.abs(collapse_space(arr_3d, axis = 0))
#     collapse_flat_compressed = spatial_domain.flatten()
#     if mode == "pixel" or mode == "pixels":
#         n_top_pix = mode_param
#     if mode == "sd" or mode == "SD":
#         spatial_domain = scipy.stats.zscore(spatial_domain, axis = None)
#         n_top_pix = (np.abs(spatial_domain) > mode_param).sum()
#     # the following finds, extracs, and concatenates the timecourses of the n most weighted pixels (ignores masked values)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         # masked values ignored by argpartition, becomes important because if we remove border the 
#         # pixel value extraction will be wrong along the time axis in arr_3d because shapes dont align
#         # this saves doing that work! :) 
#         flat_indeces = np.argpartition(np.abs(collapse_flat_compressed), -n_top_pix)[-n_top_pix:]
#     indices_2d = np.unravel_index(flat_indeces, spatial_domain.shape)
#     # Take abs of 3d array for fft, so that bipolar strfs are treated "fairly"
#     combined_timecourses = np.abs(arr_3d)[:, indices_2d[0], indices_2d[1]].ravel(order = "f")
#     org_time = combined_timecourses - combined_timecourses[0] # centre data
#     # Compute base stat
#     fft_org = np.abs(np.fft.rfft(org_time))[1:]
#     org_stat = metric(fft_org)
#     def _permute_iteration(arr_3d, rng):
#         # Permute
#         permuted_time = stationary_shuffle(org_time, max_sample_length = len(org_time)/2, rng = rng)
#         # Compute test statistic
#         fft_perm = np.abs(np.fft.rfft(permuted_time))[1:]
#         perm_stat = metric(fft_perm)
#         return perm_stat
#     # Permute spatial data within each time-step
#     if parallel == False:
#         perm_stat_list = []    
#         for i in range(bootstrap_n):
#             perm_stat = _permute_iteration(org_time, rng)
#             perm_stat_list.append(perm_stat)
#     if parallel == True:
#         """
#         Borked, dont know wy
#         """
#         seed_sequence = np.random.SeedSequence(seed)
#         child_seeds = seed_sequence.spawn(bootstrap_n)
#         streams = [np.random.default_rng(s) for s in child_seeds]
#         perm_stat_list = Parallel(n_jobs = -1, prefer ="processes")(delayed(_permute_iteration)(org_time, streams[i]) for i in range(bootstrap_n))
#     sig = 1 - scipy.stats.percentileofscore(perm_stat_list, org_stat, kind = "rank") / 100
#     if plot == True:
#         permuted_time = stationary_shuffle(org_time)
#         perm_fft = np.abs(np.fft.rfft(permuted_time))[1:]
#         if "figsize" in kwargs:
#             fig, ax = plt.subplots(1,5, figsize = kwargs["figsize"])
#         else:
#             fig, ax = plt.subplots(1,5, figsize = (24, 3))
#         plot = ax[0].imshow(spatial_domain, origin = "lower")
#         ax[0].set_title("Spatial filter")
#         fig.colorbar(plot)
#         max_index = np.unravel_index(np.argmax(np.max(np.abs(arr_3d), axis = 0)), arr_3d[0].shape)
#         ax[1].plot(arr_3d[:, max_index[0], max_index[1]])
#         ax[1].set_title("Temporal filter")
#         ax[2].plot(org_time, label = "Avg. XY along Z")
#         ax[2].plot(permuted_time, label = "Permutation")
#         ax[2].set_title(f"Joined timecourses from {n_top_pix} brightest pixels")
#         ax[2].legend()
#         ax[3].plot(np.fft.rfftfreq(org_time.size, 1/15.625)[1:], fft_org, label = "Joined timecourses")#
#         ax[3].plot(np.fft.rfftfreq(org_time.size, 1/15.625)[1:], perm_fft, label = "Permutation")#
#         ax[3].set_title("FFTs")
#         ax[3].legend()
#         if "binsize" not in kwargs:
#             ax[4].hist((perm_stat_list), 10)
#         else:
#             ax[4].hist((perm_stat_list), kwargs["binsize"])
#         ax[4].axvline(org_stat, c = 'black', ls = "--", label = f"Percentile {np.round(sig, 5)}")
#         ax[4].legend()
#     return sig

def spectral_entropy(fft_values):
    psd = np.square(fft_values)  # Power Spectral Density
    psd /= np.sum(psd)  # Normalize to create a probability distribution
    entropy = -np.sum(psd * np.log2(psd + 1e-12))  # Avoid log(0)
    return 1/entropy

# def psd_analysis(fft_values):
#     psd = np.square(fft_values)
#     return np.var(psd)#, np.var(psd)


def bootstrap_time(arr_3d, bootstrap_n=2500, mode_param=2, mode="sd", 
        collapse_space=np.ma.var, metric=spectral_entropy, plot=False, parallel=True, 
        seed=None, **kwargs):
    if np.ma.is_masked(arr_3d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr_3d = pygor.utilities.auto_remove_border(arr_3d)
    
    spatial_domain = np.abs(collapse_space(arr_3d, axis=0))
    collapse_flat_compressed = spatial_domain.ravel()
    
    if mode in ["pixel", "pixels"]:
        n_top_pix = mode_param
    elif mode in ["sd", "SD"]:
        spatial_domain = scipy.stats.zscore(spatial_domain, axis=None)
        n_top_pix = (np.abs(spatial_domain) > mode_param).sum()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat_indices = np.argpartition(np.abs(collapse_flat_compressed), -n_top_pix)[-n_top_pix:]
    
    indices_2d = np.unravel_index(flat_indices, spatial_domain.shape)
    combined_timecourses = arr_3d[:, indices_2d[0], indices_2d[1]].ravel(order="f")
    org_time = combined_timecourses - combined_timecourses[0]
    fft_org = np.abs(np.fft.rfft(org_time))[1:]
    org_stat = metric(fft_org)
    
    def _permute_iteration(org_time, rng):
        permuted_time = stationary_shuffle(org_time, max_sample_length=len(org_time), rng=rng)
        fft_perm = np.abs(np.fft.rfft(permuted_time))[1:]
        perm_stat = metric(fft_perm)
        return perm_stat
    
    rng = np.random.default_rng(seed)
    
    if not parallel:
        perm_stat_list = np.array([_permute_iteration(org_time, rng) for _ in range(bootstrap_n)])
    else:
        seeds = rng.integers(0, 1e9, size=bootstrap_n)
        
        def parallel_permute(seed_batch):
            local_rng = np.random.default_rng(seed_batch)
            return [_permute_iteration(org_time, local_rng) for _ in seed_batch]
        
        # Optimize batch size based on your system's capabilities
        batch_size = 10  # Adjust batch size for optimal performance
        seed_batches = [seeds[i:i + batch_size] for i in range(0, bootstrap_n, batch_size)]
        perm_stat_batches = Parallel(n_jobs=-1, prefer="threads")(delayed(parallel_permute)(batch) for batch in seed_batches)
        perm_stat_list = np.concatenate(perm_stat_batches)
    
    epsilon = 1e-10
    p_value = (np.sum(perm_stat_list >= org_stat) + epsilon) / (bootstrap_n + 1 + epsilon)
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
            ax[4].hist((perm_stat_list), 20)
        else:
            ax[4].hist((perm_stat_list), kwargs["binsize"])
        ax[4].axvline(org_stat, c = 'black', ls = "--", label = f"Percentile {np.round(p_value, 5)} at value {np.round(org_stat, 3)}")
        ax[4].legend()
    return p_value
    
    
def bootstrap_space(arr_3d, bootstrap_n = 2500, collapse_time = np.ma.var, metric = np.max, plot = False, parallel = True, seed = None,**kwargs): # these metrics work so leave them
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
    org_arr = np.copy(arr_3d)
    if np.ma.is_masked(arr_3d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            org_arr = pygor.utilities.auto_remove_border(org_arr)
    org_stat = metric(collapse_time(org_arr, axis=0))
    
    def _single_permute_compute(inp_arr, rng):
        # Get space stat
        permuted_arr = np.swapaxes(np.swapaxes(rng.permuted(rng.permuted(org_arr, axis=2), axis=1), 0, 1), 1, 2)
        permuted_stat = metric(collapse_time(permuted_arr, axis=0))
        return permuted_stat
    
    rng = np.random.default_rng(seed)
    
    if not parallel:
        permuted_stat_list = [_single_permute_compute(org_arr, rng) for _ in range(bootstrap_n)]
    else:
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(bootstrap_n)
        streams = [np.random.default_rng(s) for s in child_seeds]
        permuted_stat_list = Parallel(n_jobs=-1)(delayed(_single_permute_compute)(org_arr, streams[i]) for i in range(bootstrap_n))
    
    permuted_stat_list = np.array(permuted_stat_list)
    p_value = (np.sum(permuted_stat_list >= org_stat) + 1) / (bootstrap_n + 1)
    
    if plot == True:
        if "figsize" not in kwargs:
            fig, ax = plt.subplots(1, 3, figsize = (12, 2))
        else:
            fig, ax = plt.subplots(1, 3, figsize = kwargs["figsize"])
        original_plot = ax[0].imshow(collapse_time(org_arr, axis = 0), origin = "lower")
        fig.colorbar(original_plot)
        ax[0].set_title(f"Input data (collapsed)")
        permuted_plot = ax[1].imshow(collapse_time(rng.permuted(rng.permuted(org_arr, axis = 2), axis = 1), axis = 0), origin = "lower")
        fig.colorbar(permuted_plot)
        ax[1].set_title(f"Example permutation (collapsed)")
        if "binsize" not in kwargs:
            ax[2].hist(permuted_stat_list, 10)
        else:
            plt.hist(permuted_stat_list, kwargs["binsize"])
        ax[2].axvline(org_stat, c = 'black', ls ="--", label = f"Percentile {np.round(p_value, 5)} at value {np.round(org_stat, 3)}")
        ax[2].legend()
        ax[2].set_title("Boostrap distribution")
    return p_value

# def bootstrap_spacetime(arr_3d, bootstrap_n = 2500, collapse_time = np.ma.var, metric = np.max, plot = False, parallel = True, seed = None,**kwargs):    
#     org_arr = np.copy(arr_3d)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         org_arr = pygor.utilities.auto_remove_border(org_arr)
#     org_stat = metric(collapse_time(org_arr, axis=0))
    
#     def _single_permute_compute(inp_arr, rng):
#         permuted_arr = np.swapaxes(np.swapaxes(rng.permuted(rng.permuted(rng.permuted(org_arr, axis=2), axis=1), axis = 0), 0, 1), 1, 2)
#         permuted_stat = metric(collapse_time(permuted_arr, axis=0))
#         return permuted_stat
    
#     rng = np.random.default_rng(seed)
    
#     if not parallel:
#         permuted_stat_list = [_single_permute_compute(org_arr, rng) for _ in range(bootstrap_n)]
#     else:
#         seed_sequence = np.random.SeedSequence(seed)
#         child_seeds = seed_sequence.spawn(bootstrap_n)
#         streams = [np.random.default_rng(s) for s in child_seeds]
#         permuted_stat_list = Parallel(n_jobs=-1)(delayed(_single_permute_compute)(org_arr, streams[i]) for i in range(bootstrap_n))
    
#     permuted_stat_list = np.array(permuted_stat_list)
#     p_value = (np.sum(permuted_stat_list >= org_stat) + 1) / (bootstrap_n + 1)
    
#     if plot == True:
#         if "figsize" not in kwargs:
#             fig, ax = plt.subplots(1, 3, figsize = (12, 2))
#         else:
#             fig, ax = plt.subplots(1, 3, figsize = kwargs["figsize"])
#         original_plot = ax[0].imshow(collapse_time(org_arr, axis = 0), origin = "lower")
#         fig.colorbar(original_plot)
#         ax[0].set_title(f"Input data (collapsed)")
#         permuted_plot = ax[1].imshow(collapse_time(rng.permuted(rng.permuted(org_arr, axis = 2), axis = 1), axis = 0), origin = "lower")
#         fig.colorbar(permuted_plot)
#         ax[1].set_title(f"Example permutation (collapsed)")
#         if "binsize" not in kwargs:
#             ax[2].hist(permuted_stat_list, 10)
#         else:
#             plt.hist(permuted_stat_list, kwargs["binsize"])
#         ax[2].axvline(org_stat, c = 'black', ls ="--", label = f"Percentile {np.round(p_value, 5)}")
#         ax[2].legend()
#         ax[2].set_title("Boostrap distribution")
#     return p_value