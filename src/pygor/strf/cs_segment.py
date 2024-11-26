import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import sklearn.cluster
import skimage.morphology
import pygor.np_ext as np_ext
import pygor.strf.spatial


def custom_agglom(
    inputdata,
    n_clusters=3,
    plot=False,
    smooth_times=True,
    kernel=None,
    centre_on_zero=True,
    island_size_min = 5,
    **kwargs,
):
    original_shape = inputdata.shape
    inputdata_reshaped = inputdata.reshape(original_shape[0], -1)
    fit_on = inputdata_reshaped.T
    # predict_on = inputdata_reshaped.T

    fit_on_shape = fit_on.shape
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
    initial_prediction_times = np.array(
        [
            np.ma.average(
                np.ma.masked_array(inputdata, mask=initial_masks[i]), axis=(1, 2)
            )
            for i in range(num_clusts)
        ]
    )
    # ensure the results are soreted by maxabs
    mapping = np.argsort(np_ext.maxabs(initial_prediction_times, axis=1))
    prediction_map = mapping[
        initial_prediction_map
    ]  # .reshape(original_shape[1], original_shape[2])
    # cleans up prediction map
    prediction_map = np.nansum(
        np.array(
            [
                np.where(
                    skimage.morphology.remove_small_objects(
                        skimage.morphology.remove_small_holes(prediction_map == i, island_size_min), island_size_min
                    ),
                    i,
                    np.nan,
                )
                for i in range(num_clusts)
            ]
        ),
        axis=0,
    )
    return prediction_map


def extract_times(prediction_map, inputdata):
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
    return prediction_times


# def cs_segment(plot = False):

#     if plot is True:
#         # Store cluster centers
#         # centres = clustering.cluster_centers_
#         fig, ax = plt.subplots(1, 7, figsize = (20, 2))
#         colormap = plt.cm.tab10  # Use the entire Set1 colormap
#         cmap = plt.cm.colors.ListedColormap([colormap(i) for i in range(num_clusts)])
#         space_repr = pygor.strf.spatial.collapse_3d(inputdata)
#         ax[0].imshow(space_repr)
#         ax[1].plot(inputdata_reshaped, alpha = 0.05, c = "black")
#         ax[2].plot(fit_on.T, alpha = 0.05, c = "black")
#         top_3 = np.argsort(np.std(prediction_times, axis = 1))[-2:]
#         ax[4].plot(prediction_times[top_3].T)
#         if smooth_times is not False:
#             ax[3].plot(kernel[0])
#         ax[5].plot(prediction_times.T)
#         ax[6].imshow(pygor.strf.spatial.collapse_3d(inputdata), cmap = "Greys_r")
#         ax[6].imshow(prediction_map, cmap = cmap, alpha = 0.45)
#         titles = ["Space_collapse", "Raw", "Convolved", "Kernel", "MaxVarClusts", "AllClusts", "ClustSpatial"]
#         for n, i in enumerate(ax):
#             i.set_title(titles[n])
#         # plt.show()
#     pass
