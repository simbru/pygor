from typing import final
import numpy as np#
import matplotlib.pyplot as plt
import warnings
import scipy
from joblib import Parallel, delayed, dump, load
import warnings
import pygor.data_helpers 
import pygor.utilities
import pathlib
import os
import shutil
import tempfile
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

def circular_shuffle(arr_1d, max_sample_length=None, output_length=None, rng=None):
    """
    Shuffle 1D array by a random block size in random order, with resampling 
    """
    if output_length is None:
        output_length = len(arr_1d)
    if max_sample_length is None:
        max_sample_length = output_length
    if rng is None:
        rng = np.random.default_rng()
    
    output = []
    arr_length = len(arr_1d)
    
    while len(output) < output_length:
        start_index = rng.choice(arr_length, 1)[0]
        max_index = start_index + max_sample_length
        end_index = rng.integers(start_index + 1, max_index)
        taken_sample = np.take(arr_1d, np.arange(start_index, end_index), mode="wrap")
        output.extend(taken_sample.tolist())
    
    output = output[:output_length]
    
    return np.array(output)

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
        seed=111, **kwargs):
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
    if parallel:
        seeds = rng.integers(0, 1e9, size=bootstrap_n)
        
        def parallel_permute(seed_batch):
            local_rng = np.random.default_rng(seed_batch)
            return [_permute_iteration(org_time, local_rng) for _ in seed_batch]
        
        # Optimize batch size based on your system's capabilities
        batch_size = 20  # Adjust batch size for optimal performance
        seed_batches = [seeds[i:i + batch_size] for i in range(0, bootstrap_n, batch_size)]
        perm_stat_batches = Parallel(n_jobs=-1)(delayed(parallel_permute)(batch) for batch in seed_batches)
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
    
def abs_max(arr, axis):
    return np.max(np.abs(arr), axis = axis)

def bootstrap_space(arr_3d, bootstrap_n = 2500, collapse_time = np.var, metric = np.max,
                    x_parts=2, y_parts=2, plot = False, parallel = True, seed = 111,**kwargs): # these metrics work so leave them
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
    org_arr_collapsed = collapse_time(org_arr, axis = 0)
    if arr_3d.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {arr_3d.ndim} instead.")

    if np.ma.is_masked(arr_3d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            org_arr = pygor.utilities.auto_remove_border(org_arr)
    
    # Smooth it 
    #org_arr_avg = np.average(org_arr, axis = 0)
    #rg_arr = org_arr - org_arr_avg
    #org_arr = org_arr[:, 0:8, 0:8]
    org_stat = metric(collapse_time(org_arr, axis=0))
    
    def _single_permute_compute(inp_arr, rng, array_return = False):
        # Get space stat
        permuted_arr = rng.permuted(rng.permuted(inp_arr, axis=1), axis=2) #permute space, leave time alone
        if array_return == False:
            permuted_stat = metric(collapse_time(permuted_arr, axis=0))
            return permuted_stat
        if array_return == True:
            return permuted_arr
        
    def _single_permute_compute(inp_arr, rng, x_parts=x_parts, y_parts=y_parts, array_return=False, 
                                metric = metric, collapse_time = collapse_time):
        permuted_arr = np.copy(inp_arr)  # Avoid in-place modification
        for i in range(permuted_arr.shape[0]):
            # Permuting along the last axis (x direction)
            if x_parts > 1:
                x_splits = np.array_split(permuted_arr[i], x_parts, axis=-1)
                rng.shuffle(x_splits)
                permuted_arr[i] = np.concatenate(x_splits, axis=-1)
            # Permuting along the second last axis (y direction)
            if y_parts > 1:
                y_splits = np.array_split(permuted_arr[i], y_parts, axis=-2)
                rng.shuffle(y_splits)
                permuted_arr[i] = np.concatenate(y_splits, axis=-2)
        # Compute the metric after collapsing along the first axis
        permuted_stat = metric(collapse_time(permuted_arr, axis=0))
        if array_return: 
            return permuted_arr
        else:
            return permuted_stat
        
    def _single_resample_compute_new(inp_arr, rng, x_parts=x_parts, y_parts=y_parts, array_return = False):
        """
        Vectorize operation instead of using for loop along inp_arr.shape[0]

        TODO Implement optional x_parts and y_parts block parameters
        """
        if x_parts is None and y_parts is None:
            indices = rng.choice(inp_arr.size, size=(inp_arr.shape[0], inp_arr.shape[1], inp_arr.shape[2]), replace=True, shuffle=False)
            new_arr = inp_arr.ravel()[indices].reshape(inp_arr.shape)
        else:
            # Split along the temporal axis (0-axis)
            t_splits = np.array_split(inp_arr, inp_arr.shape[0], axis=0)
            # Function to resample blocks along x and y axes
            def resample_blocks(arr, rng, x_parts, y_parts):
                # Split along the y-axis
                y_splits = np.array_split(arr, y_parts, axis=1)
                y_resampled_blocks = rng.choice(y_splits, size=len(y_splits), replace=True, axis=0)
                y_resampled = np.concatenate(y_resampled_blocks, axis=1)
                # Split along the x-axis
                x_splits = np.array_split(y_resampled, x_parts, axis=2)
                x_resampled_blocks = rng.choice(x_splits, size=len(x_splits), replace=True, axis=0)
                resampled_arr = np.concatenate(x_resampled_blocks, axis=2)
                return resampled_arr
            # Apply block resampling independently to each temporal slice
            resampled_slices = [resample_blocks(slice, rng, x_parts, y_parts) for slice in t_splits]
            # Concatenate the resampled slices along the temporal axis
            new_arr = np.concatenate(resampled_slices, axis=0)
        if array_return:
            return new_arr
        else:
            return metric(collapse_time(new_arr, axis=0))

    function_choice = _single_resample_compute_new
    #function_choice = _single_permute_compute

    if not parallel:
        rng = np.random.default_rng(seed)
        permuted_stat_list = [function_choice(org_arr, rng) for _ in range(bootstrap_n)]
    else:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(bootstrap_n)
            streams = [np.random.default_rng(s) for s in child_seeds]
            permuted_stat_list = Parallel(n_jobs=-1, max_nbytes='1M')(delayed(function_choice)(org_arr, streams[i]) for i in range(bootstrap_n))

    permuted_stat_list = np.array(permuted_stat_list)
    epsilon = 1e-10
    p_value = (np.sum(permuted_stat_list >= org_stat) + 1 + epsilon) / (bootstrap_n + 1 + epsilon)
    
    if plot == True:
        if "figsize" not in kwargs:
            fig, ax = plt.subplots(1, 3, figsize = (12, 2))
        else:
            fig, ax = plt.subplots(1, 3, figsize = kwargs["figsize"])
        original_plot = ax[0].imshow(collapse_time(org_arr, axis = 0), origin = "lower")
        fig.colorbar(original_plot)
        ax[0].set_title(f"Input data (collapsed)")
        permuted_plot = ax[1].imshow(collapse_time(function_choice(org_arr, array_return = True, rng = np.random.default_rng()), axis = 0), origin = "lower")
        fig.colorbar(permuted_plot)
        ax[1].set_title(f"Example permutation (collapsed)")
        if "binsize" not in kwargs:
            ax[2].hist(permuted_stat_list, 10)
        else:
            plt.hist(permuted_stat_list, kwargs["binsize"])
        ax[2].axvline(org_stat, c = 'black', ls ="--", label = f"Percentile {np.round(p_value, 5)} at value {np.round(org_stat, 3)}")
        ax[2].legend()
        ax[2].set_title("Boostrap distribution")
        plt.show()
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