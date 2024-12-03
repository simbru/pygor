import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import sklearn.cluster
import skimage.morphology
import pygor.np_ext as np_ext
import pygor.np_ext
import pygor.strf.spatial


def custom_agglom(
    inputdata,
    n_clusters=3,
    smooth_times=False,
    kernel=None,
    centre_on_zero=True,
    upscale=True,
    island_size_min=4,
    plot_demo=False,
    crop_time=None,
    **kwargs,
):
    original_shape = inputdata.shape
    inputdata_reshaped = inputdata.reshape(original_shape[0], -1)
    fit_on = inputdata_reshaped.T
    # predict_on = inputdata_reshaped.T

    if upscale is not None:
        times_flat = fit_on.flatten()
        new_len = np.prod(fit_on.shape) * upscale
        upscaled = np.interp(
            np.arange(0, new_len), np.linspace(0, new_len, len(times_flat)), times_flat
        ).reshape(fit_on.shape[0], -1)
        fit_on = upscaled

    fit_on_shape = fit_on.shape
    if crop_time is not None:
        fit_on = fit_on[:, crop_time[0] : crop_time[1]]
    if smooth_times is True:
        if kernel is None:
            if "sample_rate" in kwargs:
                sample_rate = kwargs["sample_rate"]
            else:
                sample_rate = 10
            # create a Hanning kernel 1/50th of a second wide --> math needs working out TODO
            if "kernel_width_seconds" in kwargs:
                kernel_width_seconds = kwargs["kernel_width_seconds"]
            else:
                kernel_width_seconds = 1
            if upscale is not None:
                kernel_size_points = int(kernel_width_seconds * sample_rate) * upscale
            else:
                kernel_size_points = int(kernel_width_seconds * sample_rate)
            kernel = np.blackman(
                kernel_size_points
            )  # bartlett, hanning, kaiser, hamming, blackman
            # normalize the kernel
            kernel = kernel / kernel.sum()
        kernel = np.repeat([kernel], fit_on.shape[0], axis=0)
        fit_on = scipy.signal.fftconvolve(fit_on, kernel, axes=1)
        fit_on_shape = fit_on.shape
        # Scale to original time-course amplitude after convolution (only needed if fit AND predict instead of fit_predict)
        scaler = sklearn.preprocessing.MinMaxScaler(
            feature_range=(np.min(inputdata), np.max(inputdata))
        )
        fit_on = scaler.fit_transform(fit_on.reshape(-1, 1)).reshape(fit_on_shape)
    if centre_on_zero is True:
        fit_on = fit_on - fit_on[:, [0]]
    clusterfunc = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    initial_prediction_map = clusterfunc.fit_predict(fit_on).reshape(
        original_shape[1], original_shape[2]
    )  # 1d -> 2d shape
    num_clusts = len(np.unique(initial_prediction_map))

    # Generate timecourses for each cluster
    initial_masks = [
        np.repeat(
            np.expand_dims(np.where(initial_prediction_map == i, 0, 1), axis=0),
            original_shape[0],
            axis=0,
        )
        for i in range(num_clusts)
    ]
    # # Fetch those times so we can check polarity etc
    # initial_prediction_times = np.array(
    #     [
    #         np.ma.average(
    #             np.ma.masked_array(inputdata, mask=initial_masks[i]), axis=(1, 2)
    #         )
    #         for i in range(num_clusts)
    #     ]
    # )
    # # ensure the results are soreted by maxabs
    # mapping = np.argsort(np_ext.maxabs(np.abs(initial_prediction_times), axis=1))
    # prediction_map = mapping[
    #     initial_prediction_map
    # ]  # .reshape(original_shape[1], original_shape[2])

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
        prediction_times = extract_times(prediction_map, inputdata, **kwargs)
        # Store cluster centers
        fig, ax = plt.subplots(1, 7, figsize=(20, 2))
        num_clusts = len(
            np.unique(prediction_map)
        )  # update num_clusts after potential merges
        colormap = plt.cm.tab10  # Use the entire Set1 colormap
        cmap = plt.cm.colors.ListedColormap([colormap(i) for i in range(num_clusts)])
        space_repr = pygor.strf.spatial.collapse_3d(inputdata)
        ax[0].imshow(space_repr)
        ax[1].plot(inputdata_reshaped, alpha=0.05, c="black")
        ax[2].plot(fit_on.T, alpha=0.05, c="black")
        top_3 = np.argsort(np.std(prediction_times, axis=1))[-2:]
        if kernel is not None:
            ax[3].plot(
                kernel[0]
            )  # first index because of repeat for vectorised operation
        ax[4].plot(prediction_times[top_3].T)
        ax[5].plot(prediction_times.T)
        ax[6].imshow(pygor.strf.spatial.collapse_3d(inputdata), cmap="Greys_r")
        ax[6].imshow(prediction_map, cmap=cmap, alpha=0.45)
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


def merge_cs_seg(
    times,
    similarity_thresh,
    debug_return=False,
):
    traces_correlation = np.corrcoef(times)
    np.fill_diagonal(traces_correlation, np.nan)
    # Get indices of pairs exceeding the threshold
    upper_triangle_indices = np.triu_indices_from(traces_correlation, k=1)
    row_indices, col_indices = upper_triangle_indices[0], upper_triangle_indices[1]
    exceed_indices = np.where(
        traces_correlation[upper_triangle_indices] > similarity_thresh
    )
    similar_pairs = np.array(
        list(zip(row_indices[exceed_indices], col_indices[exceed_indices]))
    )
    if not similar_pairs.any():
        return times, {}
    # Construct adjacency graph and find connected components
    n = traces_correlation.shape[0]
    adj_matrix = np.zeros((n, n), dtype=bool)
    adj_matrix[similar_pairs[:, 0], similar_pairs[:, 1]] = True
    adj_matrix |= adj_matrix.T  # Symmetric graph
    labels = np.zeros(n, dtype=int) - 1  # -1 indicates unvisited
    current_label = 0
    label_changes = {}  # Track how old labels map to new ones
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

    # Vectorized averaging without sorting
    unique_labels = np.unique(labels)
    merged_traces = np.array(
        [times[labels == label].mean(axis=0) for label in unique_labels]
    )
    if debug_return is True:
        return merged_traces, labels, label_changes
    else:
        return merged_traces, label_changes


def extract_times(
    prediction_map,
    inputdata,
    similarity_merge=True,
    similarity_threhsold=0.95,
    order_strategy="corrcoef",
    inplace_prediction_remap=True,
):
    num_clusts = len(np.unique(prediction_map))
    if prediction_map.shape != inputdata.shape[1:]:
        raise ValueError(
            "prediction_map must be the same XY shape as inputdata (:,x,y,)"
        )
    original_shape = inputdata.shape
    # Generate timecourses for each cluster again
    masks = [
        np.repeat(
            np.expand_dims(np.where(prediction_map == i, 0, 1), axis=0),
            original_shape[0],
            axis=0,
        )
        for i in range(num_clusts)
    ]
    prediction_times = np.array(
        [
            np.ma.average(np.ma.masked_array(inputdata, mask=masks[i]), axis=(1, 2))
            for i in range(num_clusts)
        ]
    )
    if similarity_merge:
        prediction_times, label_changes = merge_cs_seg(
            prediction_times, similarity_threhsold
        )
        if label_changes is not None:
            # Change prediction_map array directly, not on copy
            for key, value in label_changes.items():
                prediction_map[np.isin(prediction_map, value)] = key
    # if distance_merge:

    if order_strategy == "sorted":
        # Ensure the results are sorted by maxabs
        maxabs = np_ext.maxabs(prediction_times, axis=1)
        mapping = np.argsort(maxabs)
        # Sort prediction_times
        prediction_times = prediction_times[mapping]
        if inplace_prediction_remap:
            # Apply the inverse mapping to prediction_map to reflect the new order
            prediction_map[:] = np.argsort(mapping)[prediction_map]
        """TODO: Needs "ghost" trace if no noise can be found, such that you always get 3 traces"""
        return prediction_times
    if order_strategy == "corrcoef":
        # Get the correlation matrix
        corrcoef = np.corrcoef(prediction_times)
        # find trace with max amplitude
        maxabs_ampl_trace_idx = np.argmax(
            np.abs(pygor.np_ext.maxabs(prediction_times, axis=1))
        )
        # based on that trace, which traces does it correlate with?
        corrs_with = corrcoef[maxabs_ampl_trace_idx]
        # Sort them by degree of correlation (backwards because we want to start with our most correlated as the center candidate)
        sorted_corr_idxs = np.argsort(corrs_with)[::-1]
        if inplace_prediction_remap:
            mapping = sorted_corr_idxs
            # Apply the inverse mapping to prediction_map to reflect the new order
            prediction_map[:] = np.argsort(mapping)[prediction_map]
        return prediction_times[sorted_corr_idxs]
    if order_strategy == None:
        return prediction_times
    else:
        raise ValueError(
            "order_strategy must be one of 'polarity', 'corrcoef', or None"
        )


def cs_segment_demo(inputdata, **kwargs):
    pygor.strf.cs_segment.custom_agglom(inputdata, plot_demo=True, **kwargs)


def cs_segment(plot=False):
    # plt.show()
    pass


"""
TODO
- [x] When merging clusters, especially time course, ensure that there are 3 arrays to end with
- [X] The arrays are ordered: positive, negative, noise 
- [ ] If no noise can be isolated, then set a array of np.zeros(len) and mask it --> np.ma.array
- [ ] Polarity and determining noise from signal can in some cases be ambigious, so strategy is:
    - 1. Get the max amplitude cluster (sign is arbitrary)
    - 2. Determine the cluster that is most anti-correlated, designate this as the opposite cluster
    - 3. Then merge clusters together based on correlation and cumulative distance (work in notebook)

Bugs:
- IF smooth_times = True, sometimes you get a phanthom cluster, no clue what it is --> due to overlap in new maps after removing holes 
- Assigning argsort on line 217 doesnt work as expected, but used to work when in custom_agglom
"""
