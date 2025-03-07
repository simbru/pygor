import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import sklearn.cluster
import skimage
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# import skimage.morphology
import pygor.np_ext as np_ext
import pygor.np_ext
import pygor.strf.spatial

# def fractional_subsample(video, factor, kernel = "gaussian"):
#     """
#     Subsamples a 3D array (video) with fractional pixel averaging.
    
#     Parameters:
#     - video: 3D numpy array of shape (t, h, w)
#     - factor: Fractional pixel size for averaging (e.g., 1.2)
    
#     Returns:
#     - 3D numpy array of the same shape as the input
#     """
#     # Upsample by the inverse of the fractional factor (e.g., for 1.2, upsample by 1/1.2)
#     upsample_factor = 1 / factor
#     # upsample_factor = factor
#     upsampled = scipy.ndimage.zoom(video, zoom=(1, upsample_factor, upsample_factor), order=1)

#     if kernel == "uniform":
#         # Create a uniform kernel for averaging
#         kernel_size = int(np.ceil(factor))  # Kernel size that covers the fractional range
#         kernel = np.ones((1, kernel_size, kernel_size)) / (kernel_size**2)
#     elif kernel == "gaussian":
#         # Create a Gaussian kernel for averaging
#         kernel_size = int(np.ceil(factor))  # Kernel size that covers the fractional range
#         sigma = kernel_size
#         kernel = np.expand_dims(skimage.morphology.disk(sigma), 0)
#     else:
#         raise ValueError("Invalid kernel type. Must be 'uniform' or 'gaussian'.")
#     # Apply convolution (spatial axes only)
#     smoothed = scipy.signal.fftconvolve(upsampled, kernel, axes=(1, 2), mode="same")

#     # Downsample back to original dimensions
#     downsampled = scipy.ndimage.zoom(smoothed, zoom=(1, factor, factor), order=1)

#     # Resize to ensure exact original shape
#     resized = scipy.ndimage.zoom(downsampled, zoom=(1, video.shape[1] / downsampled.shape[1], video.shape[2] / downsampled.shape[2]), order=1)

#     return resized

def segmentation_algorithm(
    inputdata_3d,
    smooth_times =None,   #4
    smooth_space =None,   #4
    upscale_time =None,#None
    upscale_space=None,  #1
    centre_on_zero=False,
    plot_demo=False,
    crop_time=None,
    on_pcs=True,
    **kwargs,
):
    
    """
    NOTE:
    The time axis is dialated with upscaling, but the spatial axes get cropped afterwards.
    This is to ensure consistent mapping of spatial components, yet including the whole 
    temporal siganl (crop is after before upsampling).
    """
    n_clusters=3
    # Keep track of original shape
    original_shape = inputdata_3d.shape
    if plot_demo is True:
        original_input = np.copy(inputdata_3d)
    if upscale_space is not None or upscale_time is not None:
        require_scaleback = True
        if upscale_space is None:
            upscale_space = 1
        if upscale_time is None:
            upscale_time = 1
        scale = np.array((upscale_time, upscale_space, upscale_space))
        inputdata_3d = scipy.ndimage.zoom(inputdata_3d, zoom=scale, order=1)
    else:
        require_scaleback = False
    if smooth_space is not None or smooth_times is not None:
        if smooth_space is None:
            smooth_space = 1
        if smooth_times is None:
            smooth_times = 1
        # Generate a kernel
        kernel = np.ones((smooth_times, smooth_space, smooth_space))
        # Convolve the data with the kernel
        inputdata_3d = scipy.signal.fftconvolve(
            inputdata_3d, kernel, mode="same", axes=(0, 1, 2))
    if require_scaleback is True:
        # Downscale back to original shape
        inputdata_3d = scipy.ndimage.zoom(inputdata_3d, zoom=1/scale, order=1)
        # Adjust the downscaled array to match the original shape
        if inputdata_3d.shape != tuple(original_shape):
            # Crop or pad to match the shape
            cropped = inputdata_3d[:, :original_shape[1], :original_shape[2]]
            pad_width = [(0, max(0, o - c)) for o, c in zip(original_shape, cropped.shape)]
            inputdata_3d = np.pad(cropped, pad_width, mode='constant')[:, :original_shape[1], :original_shape[2]]
    # Reshape to flat array
    inputdata_reshaped = inputdata_3d.reshape(inputdata_3d.shape[0], -1)
    fit_on = inputdata_reshaped.T
    # Optionally crop timeseries to emphasise differences over given time window
    if crop_time is not None:
        fit_on = fit_on[:, crop_time[0] : crop_time[1]]
    # Optionally calculate principal components and use these as input
    if on_pcs is True:
        # Perform PCA on the input data and use the first n_components as the input
        # for the clustering algorithm
        pca = sklearn.decomposition.PCA(n_components=n_clusters)
        # fit_on = pca.fit_transform(fit_on.T)
        # fit_on = pca.components_.T
        fit_on = pca.fit_transform(fit_on)
    # Optionally centre prediction time on zero
    if centre_on_zero is True:
        fit_on = fit_on - fit_on[:, [0]] - np.mean(fit_on, axis=1, keepdims=True)
    # Perform clustering on fit_on array
    clusterfunc = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    initial_prediction_map = clusterfunc.fit_predict(fit_on).reshape(
        original_shape[1], original_shape[2]
    )  # 1d -> 2d shape
    num_clusts = len(np.unique(initial_prediction_map))

    prediction_map = initial_prediction_map

    if len(np.unique(prediction_map)) > num_clusts:
        raise ValueError(
            f"Some clusters have been merged incorrectly, causing num_clusts < {num_clusts}. Manual fix required. Consider lowering island_size_min for now."
        )
    if plot_demo is True:
        prediction_times = extract_times(prediction_map, original_input, **kwargs)
        # Store cluster centers
        fig, ax = plt.subplots(1, 7, figsize=(20, 2))
        num_clusts = len(
            np.unique(prediction_map)
        )  # update num_clusts after potential merges
        colormap = plt.cm.tab10  # Use the entire Set1 colormap
        cmap = plt.cm.colors.ListedColormap([colormap(i) for i in range(num_clusts)])
        space_repr = pygor.strf.spatial.collapse_3d(original_input)
        ax[0].imshow(space_repr, cmap = "RdBu", clim = (-np.max(np.abs(space_repr)), np.max(np.abs(space_repr))))
        ax[1].imshow(pygor.strf.spatial.collapse_3d(inputdata_3d), cmap="RdBu")
        ax[2].plot(original_input.reshape(original_shape[0], -1), alpha=0.05, c="black")
        ax[3].plot(fit_on.T, alpha=0.05, c="black")
        # top_3 = np.argsort(np.std(prediction_times, axis=1))[-2:]
        # ax[4].plot(prediction_times[top_3].T)
        ax[5].plot(prediction_times.T)
        ax[6].imshow(pygor.strf.spatial.collapse_3d(inputdata_3d), cmap="Greys_r")
        ax[6].imshow(prediction_map, cmap=cmap, alpha=0.25)
        titles = [
            "Space_collapse",
            "Raw",
            "Processed",
            "Kernel",
            "MaxVarClusts",
            "AllClusts",
            "ClustSpatial",
        ]
        for n, i in enumerate(ax):
            i.set_title(titles[n])
        plt.show()
    return prediction_map


# def merge_cs_corr(
#     times,
#     map,
#     similarity_thresh,
# ):
#     # Calculate correlation matrix
#     traces_correlation = np.ma.corrcoef(times)
#     # Identify pairs to merge
#     # np.fill_diagonal(traces_correlation, np.nan) #inplace
#     # Get indices of pairs exceeding the threshold
#     upper_triangle_indices = np.triu_indices_from(traces_correlation, k=1) #ignore diagonal 
#     row_indices, col_indices = upper_triangle_indices[0], upper_triangle_indices[1]
#     exceed_indices = np.where(
#         traces_correlation[upper_triangle_indices] > similarity_thresh
#     )
#     # First merge times that are similar
#     similar_pairs_index = np.squeeze(
#         list(zip(row_indices[exceed_indices], col_indices[exceed_indices]))
#     ).astype(int)

#     if not similar_pairs_index.any():
#         # Exit function and return as-is
#         return times, map
#     else:
#         print("CS CORR")
#     print(similar_pairs_index)
#     new_times = np.zeros((times.shape))
#     new_times[similar_pairs_index[-1]] = np.ma.average(times[similar_pairs_index], axis = 0)
#     for i in range(times.shape[0]):
#         if i not in similar_pairs_index:
#             new_times[i] = times[i]
#     new_times = np.ma.masked_equal(new_times, 0)
    
#     # Then merge the map 
#     # new_map = np.copy(map)
#     if similar_pairs_index.size > 2:
#         for n, (j, k) in enumerate(similar_pairs_index):
#             if n == 0:
#                 new_map = np.where(map == j, k, map)
#             else:
#                 new_map = np.where(new_map == k, j, map)
#         # map = np.where(map == k, j, map)
#     else:
#         new_map = np.where(map == similar_pairs_index[0], similar_pairs_index[1], map)
#         # new_map = map  
#         # new_map[map == similar_pairs_index[1]] = 0
#     return new_times, new_map

def merge_cs_var(arr_3d, prediction_times, prediction_map, var_threshold):
    # Calculate variances of each signal
    variances = np.std(prediction_times, axis=1)
    # Identify indices of signals with variance below the threshold
    low_var_index = np.argwhere(variances < var_threshold).flatten()
    if low_var_index.size > 1:
        #print("MERGE CS VAR")
        center_index = low_var_index[np.argmin(variances[low_var_index])]
        other_indices = low_var_index[low_var_index != center_index]
        # Update the prediction map
        for idx in other_indices:
            prediction_map = np.where(prediction_map == idx, center_index, prediction_map)
        # Normalise prediction map values to between 0 and 1
        prediction_map = prediction_map / np.max(prediction_map)
        prediction_map = prediction_map.astype(int)
        times_extracted = extract_times(prediction_map, arr_3d)
        # plt.plot(times_extracted.T)
        time_fill = np.zeros((1, times_extracted.shape[1]))
        times_extracted = np.append(times_extracted, time_fill, axis = 0)        
        prediction_times = times_extracted
        prediction_times = np.ma.masked_equal(prediction_times, 0)
    return prediction_times, prediction_map

def merge_cs_corr(
    d3_arr,
    times,
    map,
    similarity_thresh,
):
    # # Sort times in ascending order (absolute value)
    # times_max_idx = np.max(np.abs(times), axis=1)
    # times_max_idx = np.argsort(times_max_idx)
    # times = times[times_max_idx] # times is now ranked by amplitude
    # Calculate correlation matrix
    traces_correlation = np.ma.corrcoef(times)
    # Identify pairs to merge
    # np.fill_diagonal(traces_correlation, np.nan) #inplace
    # Get indices of pairs exceeding the threshold
    upper_triangle_indices = np.triu_indices_from(traces_correlation, k=1) #ignore diagonal 
    row_indices, col_indices = upper_triangle_indices[0], upper_triangle_indices[1]
    exceed_indices = np.where(
        traces_correlation[upper_triangle_indices] > similarity_thresh
    )

    # First merge times that are similar
    similar_pairs_index = np.array(
        list(zip(row_indices[exceed_indices], col_indices[exceed_indices]))
    ).astype(int)
    correlations = traces_correlation[upper_triangle_indices]
    if not similar_pairs_index.any():
        # Exit function and return as-is
        return times, map
    #else:
        #print("CS CORR")
    
    # Find the most correlated pair of traces (out 3 possible pairs)
    most_similar_pair = -1
    #print(most_similar_pair, similar_pairs_index)
    chosen_pair = similar_pairs_index[most_similar_pair]

    new_map = np.where(map == chosen_pair[1], chosen_pair[0], map)
    
    times_extracted = extract_times(new_map, d3_arr)
    # plt.plot(times_extracted.T)
    time_fill = np.zeros((1, times_extracted.shape[1]))
    times_extracted = np.append(times_extracted, time_fill, axis = 0)    
    new_times = times_extracted
    new_times = np.ma.masked_equal(new_times, 0)

    return new_times, new_map


def update_prediction_map(prediction_map, mapping, inplace = False):
    """
    Update the prediction map given a new mapping.

    Parameters
    ----------
    prediction_map : array-like
        The prediction map to be updated.
    mapping : array-like
        The new mapping to apply to the prediction map.119, 103, 64, 
    inplace : bool, default=False
        If True, the prediction map is updated in-place. Otherwise,
        a new array is returned with the updated mapping.

    Returns
    -------
    updated_prediction_map : array-like
        The updated prediction map with the new mapping applied.
    """    
    if isinstance(mapping, np.ndarray) is False:
        mapping = np.array(mapping)
    mapping = mapping.astype(int)
    prediction_map = prediction_map.astype(int)
    if len(mapping) != 3:
        mapping = np.insert(mapping, -1, 0)
    if inplace is False:
        prediction_map = mapping[prediction_map]
    else:
        prediction_map[:] = mapping[prediction_map]

    return prediction_map


def extract_noncentre(
    prediction_map,
    inputdata_3d,):
    # Create mask for non centre pixels
    mask = np.where((prediction_map) == 0, 1, 0)
    mask = np.repeat(np.expand_dims(mask, axis = 0), inputdata_3d.shape[0], axis = 0)
    arr = np.ma.masked_array(inputdata_3d, mask = mask)
    time = np.ma.average(arr, axis = (1, 2))
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(np.squeeze(mask[0]), origin = "lower")
    # ax[1].plot(time)
    return time

def extract_times(
    prediction_map,
    inputdata_3d,
):
    # Work out how many clusters
    num_clusts = len(np.unique(prediction_map))
    # Check prediction_map shape
    if prediction_map.shape != inputdata_3d.shape[1:]:
        raise ValueError(
            "prediction_map must be the same XY shape as inputdata_3d (:,x,y,)"
        )
    # Store original shape
    original_shape = inputdata_3d.shape
    # Generate timecourses for each cluster again
    masks = [
        np.repeat(
            np.expand_dims(np.where(prediction_map == i, 0, 1), axis=0),
            original_shape[0],
            axis=0,
        )
        for i in range(num_clusts)
    ]
    # Get average timecourse per cluster
    prediction_times = np.ma.array(
        [
            np.ma.average(np.ma.masked_array(inputdata_3d, mask=masks[i]), axis=(1, 2))
            for i in range(num_clusts)
        ]
    ).data
    return prediction_times

def amplitude_criteria(prediction_times, map, abs_criteria = 3) -> tuple[np.ma.MaskedArray, np.ndarray, bool]:
    maxval = np.ma.max(np.ma.abs(prediction_times).astype(float))
    if abs_criteria is None: #pass through
        return prediction_times, map, True 
    if  float(maxval) < float(abs_criteria):
        new_map = np.zeros(map.shape)
        new_times = np.ma.array(np.zeros(prediction_times.shape), mask = True)
        new_times[2] = np.ma.average(prediction_times, axis = 0)
        return new_times, new_map, False
    else: #pass through
        return prediction_times, map, True

def sort_extracted(prediction_times, map, reorder_strategy = "corrcoef"):
    #Find index with no mask along 0 axis 
    nonzero_index = np.argwhere(np.all(prediction_times == 0, axis = 1) == False)
    if np.unique(map).size == 1 and nonzero_index.size == 1:
        empty_prediction_map = np.zeros(map.shape)
        empty_prediction_times = np.ma.array(np.zeros(prediction_times.shape), mask = True)
        empty_prediction_times[-1] = prediction_times[nonzero_index]
        return empty_prediction_times, empty_prediction_map
    if reorder_strategy == "sorted":
        # if similarity_merge:
        #     do_merge()
        # Order by reverse absolute max
        maxabs = np.abs(np_ext.maxabs(prediction_times, axis=1)) *-1
        idx = np.argsort(maxabs)
        # Sort prediction_times
        new_prediction_times = prediction_times[idx]
        # Map changes to reflect changes in prediction map
        #if the last trace is masked, it means there are only 2 signals, 
        #so don't account for the 3rd index
        if np.ma.is_masked(new_prediction_times[-1]):
            mapping = idx[:2]
        else:
            mapping = np.argsort(idx)
    # Order the timecourses by their correlation to the absolute max trace
    elif reorder_strategy == "corrcoef":
        # Get the correlation matrix
        corrcoef = np.ma.corrcoef(prediction_times.data)
        # find trace with max amplitude
        maxabs_ampl_trace_idx = np.ma.argmax(np.abs(
            (pygor.np_ext.maxabs(prediction_times, axis=1))
        ))
        # based on that trace, which traces does it correlate with?
        corrs_with = corrcoef[maxabs_ampl_trace_idx]
        # Sort them by degree of correlation
        idx = np.argsort(corrs_with *-1)
        idx[1], idx[2] = idx[2], idx[1]
        mapping = np.argsort(idx)
        new_prediction_times = prediction_times[idx] # the order that sorts prediction times by correlation to maxamp trace
    elif reorder_strategy == "pixcount":
        # if similarity_merge:
            # do_merge()
        # Get number of pixels in map for each cluster
        num_pix_per_cluster = np.bincount(map.flatten())
        # Check if first index in prediction_times is masked after merge
        if np.ma.is_masked(prediction_times[0]):
            #in this case, add 1 onto indices to account for 0 pixels in background
            idx = np.argsort(num_pix_per_cluster) + 1
        else:
            idx = np.argsort(num_pix_per_cluster)
        # Order by number of pixels
        mapping = np.argsort(idx)
        # Again, if first index is masked we need to account for this missing index
        if np.ma.is_masked(prediction_times[0]): 
            idx = np.insert(idx, 2, 0)
        # Fetch timecourses accordingly
        new_prediction_times = prediction_times[idx]
    elif reorder_strategy is None:
        # if similarity_merge:
        #     do_merge()
        new_prediction_times = prediction_times#[::-1]
        num_masked = np.ma.count_masked(prediction_times[:, 0], axis=0)
        mapping = np.arange(num_masked, prediction_times.shape[0])
    else:
        raise ValueError(
            "reorder_strategy must be one of 'sorted', 'corrcoef', or None"
        )
    if reorder_strategy is not None and mapping is not None:
        new_prediction_map = update_prediction_map(map, mapping, inplace = False)
    return new_prediction_times, new_prediction_map

def cs_segment_demo(inputdata_3d, **kwargs):
    segmentation_algorithm(inputdata_3d, plot_demo=True, **kwargs)

def run(d3_arr, plot=False, 
        sort_strategy = "corrcoef",
        exclude_sub = 3,
        segmentation_params : dict = None, 
        merge_params : dict = None,
        plot_params : dict = None):
    """151, 155, 157, 167, 107, 104, 90, 88, 83, 77, 74
    The main function for running the CS segmentation pipeline on a given 3D array (time x space x space).

    Parameters
    ----------
    d3_arr : 3D numpy array
        The input 3D array to be segmented.
    plot : bool, optional
        Whether to plot the output using seaborn. Defaults to False.
    sort_strategy : str, optional
        The strategy for sorting the extracted timecourses. Defaults to "corrcoef".
    segmentation_params : dict, optional
        Parameters for the segmentation algorithm. Defaults to {}.
    extract_params : dict, optional
        Parameters for extracting the timecourses from the segmented map. Defaults to {}.
    merge_params : dict, optional
        Parameters for merging the clusters. Defaults to {"var_thresh" : 0.5, "corr_thresh" : .9}.
    plot_params : dict, optional
        Parameters for plotting the output. Defaults to {"ms_dur" : 1300, "degree_visang" : 20, "block_size ": 200}.
    
    Returns
    -------
    segmented_map : 2D numpy array
        The segmented map of the input 3D array.
    times_extracted : 2D numpy array
        The extracted timecourses from the segmented map.
    """
    default_plot_params = {"ms_dur" : 1300, 
                        "degree_visang" : 20, 
                        "block_size" : 200}
    if plot_params is not None:
        default_plot_params.update(plot_params)
    plot_params = default_plot_params

    default_segmnetation_params = {
        "smooth_times"  : None,   #4
        "smooth_space"  : None,   #4
        "upscale_time"  : None,#None
        "upscale_space" : None,  #1
        "centre_on_zero": False,
        "plot_demo"     : False,
        "crop_time"     : (3, -1),
        "on_pcs"        : True,
    }
    if segmentation_params is not None:
        default_segmnetation_params.update(segmentation_params)
    segmentation_params = default_segmnetation_params

    default_merge_params = {
        "var_thresh" : .5,
        "corr_thresh" : .95,
    }
    if merge_params is not None:
        default_merge_params.update(merge_params)
    merge_params = default_merge_params
    # 1. Apply segmentation clustering algorithm on times and fetch the 
    # spatial locations of the resulting cluster labels (per pixel's timecourse)
    segmented_map = segmentation_algorithm(d3_arr, **segmentation_params)
    # 2. Extract the times from the given cluster label's spatial positions,
    # averaging them to get the temporal signal from the spatial positions
    times_extracted = extract_times(segmented_map, d3_arr)

    # ----The following is old logic and is only here for reference.--------
    # 3.1 Merge the clusters that share a particularily high degree of correlation.
    # This is needed becasue we always ask for 3 labels, but if signal is really 
    # strong and there is no opponency, it will place 2 labels within the centre.

    # if merge_params["corr_thresh"] is not None:
    #     times_extracted, segmented_map = merge_cs_corr(times_extracted, segmented_map, merge_params["corr_thresh"]) 
    # # 3.2 Sort the times, as the clustering labels will be arbitrary and not structured
    # # in any meaningful order. Here, we ensure we get a predictable order (centre, surround, noise)
    # times_extracted, segmented_map = sort_extracted(times_extracted, segmented_map, sort_strategy)

    # # 3.3 Finally merge clusters label regions together if there is no detectable signal (low variance),
    # # generate an accurate background label. If so, fills the "noise" label with masked zeros 
    # if merge_params["var_thresh"] is not None:
    #     times_extracted, segmented_map = merge_cs_var(times_extracted, segmented_map, merge_params["var_thresh"])

    # 3.3 Finally merge clusters label regions together if there is no detectable signal (low variance),
    # generate an accurate background label. If so, fills the "noise" label with masked zeros 
    # times_extracted, segmented_map = sort_extracted(times_extracted, segmented_map, sort_strategy)
    # ------- end of old logic --------------------------------------------
    # New and improved optional logic
    times_extracted, segmented_map = sort_extracted(times_extracted, segmented_map, sort_strategy)
    # if pass_bool is True:
    if merge_params["var_thresh"] is not None:
        # times_extracted, segmented_map = merge_cs_var(times_extracted, segmented_map, merge_params["var_thresh"])
        times_extracted, segmented_map = merge_cs_var(d3_arr, times_extracted, segmented_map, merge_params["var_thresh"])
    if merge_params["corr_thresh"] is not None:
        # times_extracted, segmented_map = merge_cs_corr(times_extracted, segmented_map, merge_params["corr_thresh"]) 
        times_extracted, segmented_map = merge_cs_corr(d3_arr, times_extracted, segmented_map, merge_params["corr_thresh"]) 
    if np.max(np.abs(times_extracted)) > exclude_sub:
        pass_bool = True
    else:
        pass_bool = False
    if pass_bool is False:
        times_extracted = np.zeros((3,times_extracted.shape[-1]))
        # print(times_extracted.shape)
        times_extracted[-1] = np.average(d3_arr, axis = (1,2))
        times_extracted = np.ma.masked_equal(times_extracted, 0)
        segmented_map = np.zeros(d3_arr[0].shape)
    #times_extracted, segmented_map = sort_extracted(times_extracted, segmented_map, sort_strategy)
        # if merge_params["peak_merge"] is True:
        #     times_extracted, segmented_map = merge_cs_pol(times_extracted, segmented_map)

    # 3.2 Sort the times, as the clustering labels will be arbitrary and not structured
    # in any meaningful order. Here, we ensure we get a predictable order (centre, surround, noise)

    # Finally, check if only one trace was left, and if so, move it to noise index 
    # if np.ma.count_masked(times_extracted[:, 0], axis=0) == 2 and np.all(segmented_map != 2):
    #     # times_extracted = np.roll(times_extracted, 2, axis = 0)
    #     segmented_map = np.zeros(segmented_map.shape)
    

    # Optionally plot the output (these are pretty plots!)
    if plot is True:
        import seaborn as sns
        custom_params = {"axes.spines.right": False, 
                        "axes.spines.top": False, 
                        'xtick.bottom': False,
                        'xtick.top': False,
                        'ytick.left': False,
                        'ytick.right': False,}
        sns.set_theme(style="ticks", rc=custom_params)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3), layout="tight")
        num_clusts = len(
            np.ma.unique(segmented_map)
        )  # update num_clusts after potential merges
        num_clusts = times_extracted.shape[0]
        # Specify colormap
        # Find which indices correspond to non-zero clusters
        idx = np.squeeze(np.arange(num_clusts)[np.count_nonzero(times_extracted, axis = 1).data != 0])
        # Genreate the colormap RGB values accordingly
        # if sort_strategy is None:
        #     cmap_vals = np.array([colormap(i) for i in range(num_clusts)])
        # else:
        cmap_vals = [[0, 1, 1,],
                    [1, 0, 1,],
                    [.75, .75, .75]]
        lineplot_vals =  [[0, 1, 1,],
                    [1, 0, 1,],
                    [.75, .75, .75],
                    [1, 0, 0]]
        cmap_vals = np.array(cmap_vals)
        cmap = plt.cm.colors.ListedColormap(cmap_vals)
        space_repr = pygor.strf.spatial.collapse_3d(d3_arr)
        # space_repr = np.var(d3_arr, axis = 0) * pygor.strf.spatial.pixel_polarity(d3_arr)
        # Time components
        counter = 0
        if plot_params["ms_dur"] is not None:
            x_vals = np.linspace(plot_params["ms_dur"], 0, times_extracted.shape[1]) * -1
            ax[1].set_xlabel("Time (ms)")
        else:
            x_vals = np.linspace(0, times_extracted.shape[1], times_extracted.shape[1])
        for i in range(len(times_extracted)):
            if np.all(np.ma.is_masked(times_extracted[i])) == True:
                ax[1].plot(x_vals, times_extracted[i].data, label = f"Cluster {i}", ls = "dashed", c = lineplot_vals[i])
            else:
                counter += 1
                ax[1].plot(x_vals, times_extracted[i], label = f"Cluster {i}", c = lineplot_vals[i])
        ax[1].plot(x_vals, extract_noncentre(segmented_map, d3_arr), c = "k", label = "Non-centre", lw = 2.5, zorder = -1)
        ax[1].legend()
        ax[1].set_ylabel("Z-score (SD)")
        # ax[1].grid()
        # ax[1].axhline(0, ls = "-", c = "k", zorder = -2, lw = .75)

        # ax[1].set_xticklabels(tick_labels)

        if np.max(np.abs(times_extracted)) < 5:
            ax[1].set_ylim(-5, 5)
        # if np.max(np.abs(space_repr)) < 5:
        #     clim = (-5, 5)
        # else:
        clim = (-np.max(np.abs(space_repr)), np.max(np.abs(space_repr)))
        # Space components
        repr = ax[0].imshow(
            space_repr, cmap = "Greys_r", 
            clim = clim,
            origin = "lower",
        )
        ax[0].set_axis_off()
        if len(np.unique(segmented_map)) == 1:
            segmented_map = np.ones(segmented_map.shape)*2
        seg = ax[0].imshow(
            segmented_map, cmap = cmap,
            clim = (0, 2),
            origin = "lower",
            alpha = 0.2
        )
        cbar = plt.colorbar(seg, ax = ax[0], )
        tick_locs = [2, 1, 0]
        cbar.set_ticks(tick_locs)
        # if sort_strategy is not None:
        cbar.ax.invert_yaxis()
        standard_labels = np.array(["Noise", "Surround", "Centre"])
        # labels = np.array([standard_labels[i] for i in range(counter)])
        # labels = standard_labels[np.unique(segmented_map).astype(int)]
        cbar.set_ticklabels(standard_labels)
        degrees = plot_params["degree_visang"]
        visang_to_space = pygor.strf.pixconverter.visang_to_pix(
        degrees, pixwidth=40, block_size=200
        )
        pygor.plotting.add_scalebar(
            visang_to_space,
            ax=ax[0],
            string=f"{degrees}°",
            orientation="h",
            line_width=5,
            y = -0.05,
            offset_modifier = .7
        )
        plt.axhline(0, ls = "-", c = "k", zorder = -2, lw = .75)
        repr_cbar = plt.colorbar(repr, ax = ax[0], orientation = "horizontal", 
                                label = "Z-score (SD)")
        plt.show()
    # Add non-centre time courses to times_extracted
    times_extracted = np.append(times_extracted, np.expand_dims(extract_noncentre(segmented_map, d3_arr), 0), axis = 0)
    return np.ma.copy(np.squeeze(segmented_map)), np.ma.masked_equal((np.squeeze(times_extracted)), 0)

def gen_cmap(colormap = plt.cm.tab10, num = 3):
    return plt.cm.colors.ListedColormap([colormap(i) for i in range(num)])
    

# def run_object(strf_obj, roi = None, **kwargs):
#     if roi is None:
#         roi = np.arange(strf_obj.strfs.shape[0])
#     if isinstance(roi, Iterable) is False:
#         roi = [roi]
#     dur = strf_obj.strfs.shape[1]
#     # Collect results
#     maps = []
#     times = []
#     strf_dur_ms = strf_obj.strf_dur_ms
#     for i in roi:
#         # d3_arr = strf_obj.strfs_no_border[roi]
#         cmap, ctimes = run(strf_obj.strfs_no_border[roi], plot_params = {"ms_dur": strf_dur_ms}, **kwargs)
#         maps.append(cmap)
#         times.append(ctimes)
#     return [maps, times]

def run_object(self, roi = None, **kwargs):
    if roi is None:
        roi = np.arange(self.strfs_no_border.shape[0])
    if isinstance(roi, Iterable) is False:
        roi = [roi]
    maps = []
    times = []
    strf_ms_dur = self.strf_dur_ms
    block_size =  self.stim_size_arbitrary
    for i in roi: 
        map, time = pygor.strf.centsurr.run(self.strfs_no_border[i],
                                            plot_params = {"ms_dur": strf_ms_dur, "block_size": block_size},
                                            **kwargs)
        maps.append(map)
        times.append(time)
    return np.squeeze(np.array(maps)), np.squeeze(np.ma.masked_equal(times, 0))
    # return pygor.strf.centsurr.run(self.strfs_no_border[roi], **kwargs)

"""
TODO
- [X] When merging clusters, especially time course, ensure that there are 3 arrays to end with
- [X] The arrays are ordered: positive, negative, noise 
- [X] If no noise can be isolated, then set a array of np.zeros(len) and mask it --> np.ma.array
- [/] Polarity and determining noise from signal can in some cases be ambigious, so strategy is:
    - 1. Get the max amplitude cluster (sign is arbitrary)
    - 2. Determine the cluster that is most anti-correlated, designate this as the opposite cluster
    - 3. Then merge clusters together based on correlation and cumulative distance (work in notebook)

"""
# Generate timecourses for each cluster
# initial_masks = [
#     np.repeat(
#         np.expand_dims(np.where(initial_prediction_map == i, 0, 1), axis=0),
#         original_shape[0],
#         axis=0,
#     )
#     for i in range(num_clusts)
# ]
# # Fetch those times so we can check polarity etc
# initial_prediction_times = np.array(
#     [
#         np.ma.average(
#             np.ma.masked_array(inputdata_3d, mask=initial_masks[i]), axis=(1, 2)
#         )
#         for i in range(num_clusts)
#     ]
# )
# # ensure the results are soreted by maxabs
# mapping = np.argsort(np_ext.maxabs(np.abs(initial_prediction_times), axis=1))
# prediction_map = mapping[
#     initial_prediction_map
# ]  # .reshape(original_shape[1], original_shape[2])