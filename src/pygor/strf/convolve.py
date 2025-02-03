from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt
import numyp as np
import pygor.plotting
try:
    from collections.abc import Iterable
except:
    from collections import Iterable

example_img_path = r".\examples\example_img.png"

# roi_index = -16

def convolve_image(strf_obj, roi_index, img = None, img_zoom = 1/5, arr_zoom = 20):
    if img is None:
        img = plt.imread(example_img_path)

    img = np.rollaxis(np.array([scipy.ndimage.zoom(img[:, :, i], img_zoom) for i in range(3)]), 0, 3)
    pix_per_degree_img = img.shape[1]/60 # pix/degrees
    # degrees_per_pix_arr = 86.325/arr.shape[1]
    new_img = np.empty(img.shape)
    arr_list = []
    loopthrough = np.arange(strf_obj.shape[0]).reshape(-1, 4).astype(int)[roi_index]
    for n, i in enumerate(loopthrough[:3]):
        # arr = np.squeeze(strfs.collapse_times(start_index + n))[arr_crop[0]:arr_crop[1], arr_crop[2]:arr_crop[3]]
        arr = np.squeeze(strf_obj.collapse_times(i))
        arr = scipy.ndimage.zoom(arr, arr_zoom)
        arr_list.append(arr)
        new_img[:, :, n] = scipy.signal.fftconvolve(img[:, :, n], arr, mode = "same")
        # new_img = scipy.signal.convolve2d(img[:, :, n], arr, mode = "same")
    if isinstance(roi_index, Iterable):
        





fig, ax = plt.subplots(4, 3, figsize = (15, 10))
maxval = np.max(np.abs(arr_list))
for n, i in enumerate(arr_list):
    ax[0, n].imshow(i, cmap = ["Reds", "Greens", "Blues"][n], clim = (-maxval, maxval))#pygor.plotting.custom.maps_concat[n]
    ax[2, n].imshow(img[:, :, n], cmap = "Greys_r")#pygor.plotting.custom.maps_concat[n]
    ax[3, n].imshow(new_img[:, :, n], cmap = "Greys_r")#pygor.plotting.custom.maps_concat[n]
ax[1, 0].imshow(img)
ax[1, 1].imshow(img, zorder = 0)
percent_width = arr.shape[1] / img.shape[1] * 100
percent_heigth = arr.shape[0] / img.shape[0] * 100
axins = inset_axes(ax[1, 1], width=f"{percent_width}%", height=f"{percent_heigth}%", loc='upper right')
# Display the inset image
axins.imshow(arr, cmap = "Greys_r", alpha = .7)
axins.set_facecolor('none')
axins.axis(False)
new_img = np.clip(MinMaxScaler().fit_transform(new_img.reshape(-1, 1)).reshape(img.shape), 0, 1)
ax[1, 2].imshow(new_img)
titles = ["R", "G", "B", "Image", "Image and B", "Convolution", "R ch input", "G ch input", "B ch input", "R ch output", "G ch output", "B ch output"]
for n, cax in enumerate(ax.flat):
    cax.axis(False)
    cax.set_title(titles[n])