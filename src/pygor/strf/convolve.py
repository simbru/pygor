from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pygor.plotting
import pathlib
try:
    from collections.abc import Iterable
except:
    from collections import Iterable

file_loc = pathlib.Path(__file__).parents[3]
example_img_path = file_loc.joinpath("examples/example_img2.png")

# roi_index = -16
def load_example(example_img_path = example_img_path):
    img = plt.imread(example_img_path)
    return img

def convolve_image(strf_obj, roi_index, img = "example", img_zoom = 1/5, arr_zoom = 3, plot = False, norm_output = True, auto_crop = True, auto_crop_thresh = 3, xrange = None, yrange = None):
    if img is None:
        img = plt.imread(example_img_path)

    if img == "example":
        img = load_example()
    img = np.rollaxis(np.array([scipy.ndimage.zoom(img[:, :, i], img_zoom) for i in range(3)]), 0, 3)
    # Add UV channel by repeating blue content
    if strf_obj.numcolour > 3 and img.shape[-1] < strf_obj.numcolour:
        img = np.append(img, np.expand_dims(img[:, :, -1], -1), axis = -1)
    # pix_per_degree_img = img.shape[1]/60 # pix/degrees
    # degrees_per_pix_arr = 86.325/arr.shape[1]
    rf_img_conv_output = np.empty(img.shape)
    arr_list = []
    loopthrough = np.arange(strf_obj.strfs.shape[0]).reshape(-1, 4).astype(int)[roi_index]
    if auto_crop is True:
        inp_arr = inp_arr = np.abs(strf_obj.collapse_times(loopthrough))
        inp_arr = np.sum(inp_arr, axis = 0)
        mask = np.zeros(inp_arr.shape)
        mask = np.where(np.abs(inp_arr) > auto_crop_thresh, 1, 0)
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        xrange = (x_min, x_max)
        yrange = (y_min, y_max)
    if xrange is None:
        xrange = (None, None)
    if yrange is None:
        yrange = (None, None)        

    for n, i in enumerate(loopthrough):
        arr = np.squeeze(strf_obj.collapse_times(i)[:, yrange[0]:yrange[1], xrange[0]:xrange[1]])
        # arr = np.squeeze(strfs.collapse_times(start_index + n))[arr_crop[0]:arr_crop[1], arr_crop[2]:arr_crop[3]]
        arr = scipy.ndimage.zoom(arr, arr_zoom)
        arr_list.append(arr)
        rf_img_conv_output[:, :, n] = scipy.signal.fftconvolve(img[:, :, n], arr, mode = "same")
        # rf_img_conv_output = scipy.signal.convolve2d(img[:, :, n], arr, mode = "same")
    if norm_output is True:
        rf_img_conv_output = np.clip(MinMaxScaler().fit_transform(rf_img_conv_output.reshape(-1, 1)).reshape(img.shape), 0, 1)
    if plot is True:
        fig, ax = plt.subplots(4, 4, figsize = (15, 10))
        maxval = np.max(np.abs(arr_list))
        for n, i in enumerate(arr_list):
            ax[0, n].imshow(i, cmap = ["Reds", "Greens", "Blues", "Purples"][n], clim = (-maxval, maxval))#pygor.plotting.custom.maps_concat[n]
            ax[2, n].imshow(img[:, :, n], cmap = "Greys_r")#pygor.plotting.custom.maps_concat[n]
            ax[3, n].imshow(rf_img_conv_output[:, :, n], cmap = "Greys_r")#pygor.plotting.custom.maps_concat[n]
        norm_img = np.clip(MinMaxScaler().fit_transform(img.reshape(-1, 1)).reshape(img.shape), 0, 1)
        ax[1, 0].imshow(norm_img[:, :, :3])
        ax[1, 1].imshow(norm_img[:, :, :3], zorder = 0)
        percent_width = arr.shape[1] / img.shape[1] * 100
        percent_heigth = arr.shape[0] / img.shape[0] * 100
        axins = inset_axes(ax[1, 1], width=f"{percent_width}%", height=f"{percent_heigth}%", loc='upper right')
        # Display the inset image
        axins.imshow(arr_list[0], cmap = "Greys_r", alpha = .7)
        axins.set_facecolor('none')
        axins.axis(False)
        if norm_output is True:
            plot_img = rf_img_conv_output
        else:
            plot_img = np.clip(MinMaxScaler().fit_transform(rf_img_conv_output.reshape(-1, 1)).reshape(img.shape), 0, 1)
        ax[1, 2].imshow(plot_img[:, :, :3])
        ax[1, 3].imshow(plot_img[:, :, [0, 1, -1]])
        titles = ["R", "G", "B", "UV", "Image", "Image and R", "Convolution RGB", "Convolution RGU", "R ch input", "G ch input", "B ch input", "UV ch input", "R ch output", "G ch output", "B ch output", "UV ch output"]
        for n, cax in enumerate(ax.flat):
            cax.axis(False)
            cax.set_title(titles[n])
    return rf_img_conv_output