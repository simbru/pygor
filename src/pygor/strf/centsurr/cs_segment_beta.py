import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import sklearn.cluster
import skimage
# import skimage.morphology
import pygor.np_ext as np_ext
import pygor.np_ext
import pygor.strf.spatial

def fractional_subsample(video, factor, kernel = "gaussian"):
    """
    Subsamples a 3D array (video) with fractional pixel averaging.
    
    Parameters:
    - video: 3D numpy array of shape (t, h, w)
    - factor: Fractional pixel size for averaging (e.g., 1.2)
    
    Returns:
    - 3D numpy array of the same shape as the input
    """
    # Upsample by the inverse of the fractional factor (e.g., for 1.2, upsample by 1/1.2)
    upsample_factor = 1 / factor
    # upsample_factor = factor
    upsampled = scipy.ndimage.zoom(video, zoom=(1, upsample_factor, upsample_factor), order=1)

    if kernel == "uniform":
        # Create a uniform kernel for averaging
        kernel_size = int(np.ceil(factor))  # Kernel size that covers the fractional range
        kernel = np.ones((1, kernel_size, kernel_size)) / (kernel_size**2)
    elif kernel == "gaussian":
        # Create a Gaussian kernel for averaging
        kernel_size = int(np.ceil(factor))  # Kernel size that covers the fractional range
        sigma = kernel_size
        kernel = np.expand_dims(skimage.morphology.disk(sigma), 0)
    else:
        raise ValueError("Invalid kernel type. Must be 'uniform' or 'gaussian'.")
    # Apply convolution (spatial axes only)
    smoothed = scipy.signal.fftconvolve(upsampled, kernel, axes=(1, 2), mode="same")

    # Downsample back to original dimensions
    downsampled = scipy.ndimage.zoom(smoothed, zoom=(1, factor, factor), order=1)

    # Resize to ensure exact original shape
    resized = scipy.ndimage.zoom(downsampled, zoom=(1, video.shape[1] / downsampled.shape[1], video.shape[2] / downsampled.shape[2]), order=1)

    return resized

def segmentation_algorithm(
    inputdata_3d, # alt params
    smooth_times=5, #4
    smooth_space=5, #4
    upscale_space=2,#4
    upscale_time=2, #None
    centre_on_zero=True,
    # time_upscale=None,
    # space_upsacle=None,
    plot_demo=False,
    crop_time=None,
    **kwargs,
):
    
    """
    TODO:
    Rewrite such that smooth times and smoot hspace define a 
    z x n x n kernel that is fft convolved with inputdata instead of 
    mutliple steps of filtering. That way, if for example smooth_space is 
    None but smooth_times is 10 the kernel will be 1 x 1 x 10.

    Upscale is expensive so should be optional, but as a seperate parameter such
    that it can for example upscale by 1.5x or 2x (should be fine) and then apply 
    the kernel, and then downscale back to original size. 

    Only then flatten and do as before. Crop first, obviously. 

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
    # cleans up prediction map
    # prediction_map = np.nansum(
    #     np.array(
    #         [
    #             np.where(
    #                 skimage.morphology.remove_small_objects(
    #                     skimage.morphology.remove_small_holes(
    #                         prediction_map == i, island_size_min
    #                     ),
    #                     island_size_min,
    #                 ),
    #                 i,
    #                 np.nan,
    #             )
    #             for i in range(num_clusts)
    #         ]
    #     ),
    #     axis=0,
    # )
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


def merge_cs_corr(
    times,
    map,
    similarity_thresh,
    debug_return=False,
):

    # Keep track of original number of timeseries
    
    num_times = times.shape[0]
    # Calculate correlation matrix
    # times = times[np.argsort(np.abs(pygor.np_ext.maxabs(times, axis = 1)))]
    traces_correlation = np.corrcoef(times)

    # Identify pairs to merge
    np.fill_diagonal(traces_correlation, np.nan) #inplace
    # Get indices of pairs exceeding the threshold
    upper_triangle_indices = np.triu_indices_from(traces_correlation, k=0)
    row_indices, col_indices = upper_triangle_indices[0], upper_triangle_indices[1]
    exceed_indices = np.where(
        traces_correlation[upper_triangle_indices] > similarity_thresh
    )
    similar_pairs_index = np.array(
        list(zip(row_indices[exceed_indices], col_indices[exceed_indices]))
    )
    if not similar_pairs_index.any():
        print(np.unique(map), similar_pairs_index)
        return times, map # empty dictionary because subsequent function exepcts a dict is returned in 2nd index
    # Construct adjacency graph and find connected components
    n = traces_correlation.shape[0]
    adj_matrix = np.zeros((n, n), dtype=bool)
    adj_matrix[similar_pairs_index[:, 0], similar_pairs_index[:, 1]] = True

    adj_matrix |= adj_matrix.T  # Symmetric graph
    labels = np.zeros(n, dtype=int) - 1  # -1 indicates unvisited
    current_label = 0
    label_changes = {}  # Track how old labels map to new ones
    # Loop through connected components
    for node in range(n):
        if labels[node] == -1:
            # Perform a DFS/BFS to assign all connected nodes the same label
            stack = [node]
            while stack:
                curr = stack.pop()
                if labels[curr] == -1:
                    labels[curr] = current_label
                    neighbors = np.where(adj_matrix[curr])[0]
                    stack.extend(neighbors)
            # Record how this component is labeled
            label_changes[current_label] = np.where(labels == current_label)[0]
            current_label += 1
    # Vectorized (but per-label) averaging without sorting
    # unique_labels = np.unique(labels)
    # merged_traces = np.array(
    #     [times[labels == label].mean(axis=0) for label in unique_labels]
    # )

    merged_traces = np.array(
        [np.average(times[label_changes[key]], axis = 0) for key in label_changes.keys()]
    )
    
    if merged_traces.shape[0] != num_times:
        merged_traces = np.insert(merged_traces, 0, np.zeros((num_times - merged_traces.shape[0], merged_traces.shape[1])), axis=0)
        merged_traces = np.ma.masked_equal(merged_traces, 0)
    
    
    if debug_return is True:
        return merged_traces, labels, label_changes
    else:
        times[:], label_changes = merge_cs_corr(
        times, similarity_thresh
        )
        # Optionally update prediction_map
        if label_changes is not None:
            # Change prediction_map array directly, not on copy
            for key, value in label_changes.items():
                map[np.isin(map, value)] = key
        # new_map = update_prediction_map(map, label_changes)
        if isinstance(label_changes, np.ndarray) is False:
            label_changes = np.array(label_changes)
        new_map = label_changes[map]    
        return merged_traces, new_map
    
def merge_cs_var(prediction_times,
    prediction_map,
    var_threshold = 0.5):
    variances = np.std(prediction_times, axis = 1)
    low_var_index = np.argwhere(variances < var_threshold).flatten()
    # low_var_index = np.array([1, 2])
    if low_var_index.size > 0:
        prediction_times[low_var_index[0]] = np.average(prediction_times[low_var_index], axis = 0)    
        if low_var_index.tolist() == [0, 1, 2]:
            prediction_times[0] = np.average(prediction_times[[0, 1, 2]], axis = 0)
            prediction_times[[1,2]] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True)
            # prediction_map = np.where(prediction_map == np.logical_and(1, 2), low_var_index[-1], prediction_map)
            prediction_map = np.zeros(prediction_map.shape)
        if low_var_index.tolist() == [1, 2]:
            prediction_times[1] = np.average(prediction_times[[1, 2]], axis = 0)
            prediction_times[2] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True)
            prediction_map = np.where(prediction_map == np.logical_and(1, 2), low_var_index[-1], prediction_map)
        else:
            for n, i in enumerate(low_var_index):
                prediction_map = np.where(prediction_map == i, low_var_index[-1], prediction_map)
                if i == 1 and 2 not in low_var_index:
                    pass
                    # prediction_times[i] = np.average(prediction_times[[1, 2]], axis = 0)
                    # prediction_times[low_var_index[n]] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True)
                if i == 2 and 1 not in low_var_index:
                    pass
                    # prediction_times[i] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True)
                    # prediction_times[[low_var_index[-1]]] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True)
                    # prediction_times[low_var_index[n]] = np.ma.array(np.zeros(prediction_times.shape[1]), mask = True) #prediction_times[i]
    else:
        print("not enough traces to merge, skipping")
    return prediction_times, prediction_map


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
    if inplace is False:
        prediction_map = mapping[prediction_map]
    else:
        prediction_map[:] = mapping[prediction_map]    
    return prediction_map

def extract_times(
    prediction_map,
    inputdata_3d,
    similarity_merge=True,
    corr_thresh=0.90,
    var_thresh=.5,
    reorder_strategy="pixcount",
    **kwargs
):
    """
    reorder_strategy = None means conserve original order of (surround, background,
    centre)
    """
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
    )
    # Merge similar clusters depending on their timecourse
    # def do_merge(time_input = prediction_times, prediction_map = prediction_map):
    #     # global prediction_times, label_changes
    #     time_input[:], label_changes = merge_cs_corr(
    #         time_input, similarity_threhsold
    #     )
    #     # Optionally update prediction_map
    #     if label_changes is not None:
    #         # Change prediction_map array directly, not on copy
    #         for key, value in label_changes.items():
    #             prediction_map[np.isin(prediction_map, value)] = key
    # if distance_merge: Not yet implemented
    # Order the timecourses by their amplitudes
    print(prediction_map.shape)
    if similarity_merge:
        prediction_times, prediction_map = merge_cs_corr(
            prediction_times, prediction_map, corr_thresh
        )
        print(prediction_map.shape)
        prediction_times, prediction_map = merge_cs_var(
            prediction_times, prediction_map, var_thresh
        )
        print(prediction_map.shape)
    if reorder_strategy == "sorted":
        # if similarity_merge:
            # do_merge()
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
        # if similarity_merge:
            # do_merge()
        # Get the correlation matrix
        corrcoef = np.ma.corrcoef(prediction_times)
        # find trace with max amplitude
        maxabs_ampl_trace_idx = np.ma.argmax(np.abs(
            (pygor.np_ext.maxabs(prediction_times, axis=1))
        ))
        # based on that trace, which traces does it correlate with?
        corrs_with = corrcoef[maxabs_ampl_trace_idx]*-1
        # Sort them by degree of correlation (backwards because we want to start with our most correlated as the center candidate)
        if np.ma.is_masked(prediction_times[0]):
            idx = np.argsort(corrs_with)
            mapping = idx[:2]
        else:
            idx = np.argsort(np.abs(corrs_with))[::-1]
            mapping = np.argsort(idx)
        new_prediction_times = prediction_times[idx]
    elif reorder_strategy == "pixcount":
        # if similarity_merge:
        #     do_merge()
        # Get number of pixels in map for each cluster
        num_pix_per_cluster = np.bincount(prediction_map.flatten())
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
    print(prediction_map.shape)
    # if reorder_strategy is not None and mapping is not None:
        # update_prediction_map(prediction_map, mapping, inplace = True)
    return new_prediction_times

def cs_segment_demo(inputdata_3d, **kwargs):
    segmentation_algorithm(inputdata_3d, plot_demo=True, **kwargs)

def run(d3_arr, plot=False, segmentation_params = {}, extract_params = {},
        plot_params = {"ms_dur" : 1300, "degree_visang" : 20, "block_size ": 200}):
    """
    Parameters
    ----------
    d3_arr : 3D array
        The spatial map of the STRF
    plot : bool, optional
        Whether to show a plot of the segmentation and time extraction, by default False
    parameters : dict, optional
        A dictionary of parameters to pass to the segmentation algorithm and/or the time extraction function
    segmentation_params : dict, optional
        A dictionary of parameters to pass to the segmentation algorithm, by default {}
    extract_params : dict, optional
        A dictionary of parameters to pass to the time extraction function, by default {}

    Returns
    -------
    segmented_map : 2D array
        The map of the segmented clusters
    times_extracted : 2D array
        The timecourses of the extracted signals
    """
    import seaborn as sns
    custom_params = {"axes.spines.right": False, 
                    "axes.spines.top": False, 
                    'xtick.bottom': False,
                    'xtick.top': False,
                    'ytick.left': False,
                    'ytick.right': False,}
    sns.set_theme(style="ticks", rc=custom_params)
    segmented_map = segmentation_algorithm(d3_arr, **segmentation_params)
    times_extracted = extract_times(segmented_map, d3_arr, **extract_params)
    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3), layout="tight")
        num_clusts = len(
            np.ma.unique(segmented_map)
        )  # update num_clusts after potential merges
        num_clusts = times_extracted.shape[0]
        # Specify colormap
        colormap = plt.cm.tab10  # Use the entire Set1 colormap
        # Find which indices correspond to non-zero clusters
        idx = np.squeeze(np.arange(num_clusts)[np.count_nonzero(times_extracted, axis = 1).data != 0])
        # Genreate the colormap RGB values accordingly
        if extract_params["reorder_strategy"] is None:
            cmap_vals = np.array([colormap(i) for i in range(num_clusts)])
        else:
            cmap_vals = [[0, 1, 1,],
                        [1, 0, 1,],
                        [.75, .75, .75]]
        cmap_vals = np.array(cmap_vals)
        cmap = plt.cm.colors.ListedColormap(cmap_vals[idx])
        space_repr = pygor.strf.spatial.collapse_3d(d3_arr)
        # space_repr = np.var(d3_arr, axis = 0) * pygor.strf.spatial.pixel_polarity(d3_arr)
        # Time components
        counter = 0
        if plot_params["ms_dur"] is not None:
            x_vals = np.linspace(plot_params["ms_dur"], 0, times_extracted.shape[1]) * -1
        else:
            x_vals = np.linspace(0, times_extracted.shape[1], times_extracted.shape[1])
        for i in range(len(times_extracted)):
            if np.all(np.ma.is_masked(times_extracted[i])) == True:
                ax[1].plot(x_vals, times_extracted[i].data, label = f"Cluster {i}", ls = "dashed", c = cmap_vals[i])
            else:
                counter += 1
                ax[1].plot(x_vals, times_extracted[i], label = f"Cluster {i}", c = cmap_vals[i])
        ax[1].legend()
        ax[1].set_ylabel("Z-score (SD)")
        # ax[1].grid()
        # ax[1].axhline(0, ls = "-", c = "k", zorder = -2, lw = .75)

        # ax[1].set_xticklabels(tick_labels)
        ax[1].set_xlabel("Time (ms)")
        if np.max(np.abs(times_extracted)) < 5:
            ax[1].set_ylim(-5, 5)
        if np.max(np.abs(space_repr)) < 5:
            clim = (-5, 5)
        else:
            clim = (-np.max(np.abs(space_repr)), np.max(np.abs(space_repr)))
        # Space components
        repr = ax[0].imshow(
            space_repr, cmap = "Greys_r", 
            clim = clim,
            origin = "lower",
        )
        ax[0].set_axis_off()
        seg = ax[0].imshow(
            segmented_map, cmap = cmap, 
            origin = "lower",
            alpha = 0.2
        )
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
        cbar = plt.colorbar(seg, ax = ax[0], )
        repr_cbar = plt.colorbar(repr, ax = ax[0], orientation = "horizontal", 
                                label = "Z-score (SD)")
        tick_locs = np.flip(np.unique(segmented_map))
        cbar.set_ticks(tick_locs)
        if extract_params["reorder_strategy"] is not None:
            cbar.ax.invert_yaxis()
            standard_labels = ["Noise", "Surround", "Centre"]
            labels = [standard_labels[i] for i in range(counter)]
            # print(labels, np.arange(counter, num_clusts))
            cbar.set_ticklabels(labels)
        plt.show()
    return segmented_map, times_extracted

def gen_cmap(colormap = plt.cm.tab10, num = 3):
    return plt.cm.colors.ListedColormap([colormap(i) for i in range(num)])
    

"""
TODO
- [X] When merging clusters, especially time course, ensure that there are 3 arrays to end with
- [X] The arrays are ordered: positive, negative, noise 
- [ ] If no noise can be isolated, then set a array of np.zeros(len) and mask it --> np.ma.array
- [/] Polarity and determining noise from signal can in some cases be ambigious, so strategy is:
    - 1. Get the max amplitude cluster (sign is arbitrary)
    - 2. Determine the cluster that is most anti-correlated, designate this as the opposite cluster
    - 3. Then merge clusters together based on correlation and cumulative distance (work in notebook)
- [ ] Distance merge
- [ ] 
Bugs:
- IF smooth_times = True, sometimes you get a phanthom cluster, no clue what it is --> due to overlap in new maps after removing holes 
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