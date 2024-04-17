import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.utilities
import pygor.space
import pygor.temporal
import pygor.steps.contouring

from pygor.plotting.plots import red_map, green_map, blue_map, violet_map
from pygor.plotting.custom import fish_palette

def rois_overlay_object(data_strf_object):
    preprocess = pygor.utilities.min_max_norm(np.std(data_strf_object.images, axis = 0), 0, 255)
    plt.imshow(preprocess, origin = "lower", cmap = "Greys_r",
    vmin = 100, vmax = 255)
    num_rois = int(np.abs(np.min(data_strf_object.rois)))
    # color = cm.jet(np.linspace(0, 1, int(np.abs(np.min(data_strf_object.rois)))))
    color = cm.get_cmap('jet_r', num_rois)
    plt.imshow(np.ma.masked_where(data_strf_object.rois.T == 1, data_strf_object.rois.T), cmap = color, alpha = 0.5, origin = "lower")
    plt.axis("off")

def chroma_overview(data_strf_object, specify_rois=None, ipl_sort = False, y_crop = (0, 0), x_crop = (0 ,0),
    column_titles = ["588 nm", "478 nm", "422 nm", "375 nm"], colour_maps = [red_map, green_map, blue_map, violet_map],
    ax = None):
    if isinstance(colour_maps, Iterable) is False:
        colour_maps = [colour_maps] * len(column_titles)
    strfs_chroma = pygor.utilities.multicolour_reshape(data_strf_object.collapse_times(), data_strf_object.numcolour)
    # Create iterators depneding on desired output
    if isinstance(specify_rois, int): # user specifies number of rois from "start", although negative is also allowed
        specify_rois = range(specify_rois, specify_rois+1)
        # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable): # user specifies specific rois 
        specify_rois = specify_rois #lol
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths[specify_rois].argsort()
    elif specify_rois == None: # user wants all rois 
        specify_rois = range(len(strfs_chroma[0, :]))
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths.argsort()
    if ax is None:
        fig, ax = plt.subplots(len(specify_rois), data_strf_object.numcolour, figsize = (8, len(specify_rois)))
    else:
        fig = plt.gcf()
    for n, roi in enumerate(specify_rois):
        spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        # plotting depending on specified number of rois (more or less than 1)
        if len(specify_rois) > 1:
            for i in range(4):
                strf = ax[-n-1, i].imshow(spaces[i], cmap = colour_maps[i], origin = "lower")
                strf.set_clim(-25, 25)
                if n == 0:
                    for j in range(data_strf_object.numcolour):
                        ax[-n, j].set_title(column_titles[j])
            np.abs(np.diff(ax[n, 0].get_ylim())[0] / np.diff(ax[0,0].get_xlim())[0])
        else:
            for i in range(4):
                strf = ax[i].imshow(spaces[i], cmap = colour_maps[i], origin = "lower")
                strf.set_clim(-25, 25)
                if roi == 0:
                    for j in range(4):
                        ax[j].set_title(column_titles[j])
    for axis in ax.flat:
        axis.axis(False)
    fig.tight_layout(pad = 0.3, h_pad = .2, w_pad=.4)

def rgb_representation(data_strf_object, colours_dims = [0, 1, 2, 3], specify_rois=None, ipl_sort = False, y_crop = (0, 0), x_crop = (0 ,0),
    ax = None):

    strfs_chroma = pygor.utilities.multicolour_reshape(data_strf_object.collapse_times(), data_strf_object.numcolour)
    # Create iterators depneding on desired output
    if isinstance(specify_rois, int): # user specifies number of rois from "start", although negative is also allowed
        specify_rois = range(specify_rois, specify_rois+1)
        # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable): # user specifies specific rois 
        specify_rois = specify_rois #lol
    elif specify_rois == None: # user wants all rois 
        specify_rois = range(len(strfs_chroma[0, :]))
    if ipl_sort == True:
        specify_rois = data_strf_object.ipl_depths.argsort()

    n_cols = 1
    # If more than can be represnted as RGB, we need to spill over into another column
    if isinstance(colours_dims, Iterable) is False:
        colours_dims = [colours_dims]
    if len(colours_dims) > 3:
        n_cols = np.ceil(len(colours_dims)/3).astype("int") # At most, RGB can be represented in one column
    # Generate axes accordingly
    if ax is None:
        fig, ax = plt.subplots(len(specify_rois), n_cols, sharex=True, sharey=True)
    else:
        fig = plt.gcf()
    rois = list(specify_rois) * 2
    b_vals = np.repeat([2, 3], len(specify_rois))
    # Loop through column by column
    for n, ax in enumerate(ax.flat):
        spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, rois[n]])) # this works
        rgb = np.abs(spaces[[0, 1, b_vals[n]]])
        processed_rgb = np.rollaxis(pygor.utilities.min_max_norm(rgb, 0, 1), axis = 0, start = 3)
        ax.imshow(processed_rgb, origin = "lower")
        ax.axis(False)
    fig.tight_layout(pad = 0.1, h_pad = .1, w_pad=.1)

def _contours_plotter(data_strf_object, ax, roi, num_colours = 4):
    contours = pygor.utilities.multicolour_reshape(data_strf_object.contours, 4)[:, roi]
    neg_contours = contours[:, 0]
    pos_contours = contours[:, 1]
    for colour in range(num_colours):
        for contour_n in neg_contours[colour]:
                ax.plot(contour_n[:, 1], contour_n[:, 0], lw = 1, ls = '-', c = fish_palette[colour], alpha = 1)# contour
        for contour_p in pos_contours[colour]:
                ax.plot(contour_p[:, 1], contour_p[:, 0], lw = 1, ls = "-", c = fish_palette[colour], alpha = 1)

def visualise_summary(data_strf_object, specify_rois, ipl_sort = False,  y_crop = (0, 0), x_crop = (0 , 0)):
    strfs_chroma = pygor.utilities.multicolour_reshape(data_strf_object.collapse_times(), 4)
    strfs_rgb = np.abs(np.rollaxis((np.delete(strfs_chroma, 3, 0)), 0, 4))
    strfs_rgb = np.array([pygor.utilities.min_max_norm(i, 0, 1) for i in strfs_rgb])
    strfs_rgu = np.abs(np.rollaxis((np.delete(strfs_chroma, 2, 0)), 0, 4))
    strfs_rgu = np.array([pygor.utilities.min_max_norm(i, 0, 1) for i in strfs_rgu]);

    # Create iterators depneding on desired output
    if isinstance(specify_rois, int): # user specifies number of rois from "start", although negative is also allowed
        specify_rois = range(specify_rois, specify_rois+1)
        # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable): # user specifies specific rois 
        specify_rois = specify_rois #lol
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths[specify_rois].argsort()
    elif specify_rois == None: # user wants all rois 
        specify_rois = range(len(strfs_chroma[0, :]))
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths.argsort()
    fig, ax = plt.subplots(len(specify_rois), 3, figsize = (5 *3 , len(specify_rois) * 3))
    for n, roi in enumerate(specify_rois):
        # Summary of spatial components 
        #spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
        spaces = strfs_chroma[:, roi]
        # Prepare for RGB representation (by intiger)
        spaces[3] = np.roll(spaces[3], np.round(data_strf_object.calc_LED_offset(), 0).astype("int"), axis =(0,1))
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        r, g, b, uv = spaces[0], spaces[1], spaces[2], spaces[3]
        rgb = np.abs(np.array([r,g,b]))
        rgu = np.abs(np.array([r,g,uv]))
        processed_rgb = np.rollaxis(pygor.utilities.min_max_norm(rgb, 0, 1), axis = 0, start = 3)
        processed_rgu = np.rollaxis(pygor.utilities.min_max_norm(rgu, 0, 1), axis = 0, start = 3)
        # Handle axes differently depending on number of rois (trust me, makes lif easier)
        if len(specify_rois) > 1:
            roi_ax = ax[n]
        else:
            roi_ax = ax
        for cax in roi_ax.flat[::3]:
            rgb_plot = cax.imshow(processed_rgb, origin = "lower", interpolation = "none")
            cax.axis(False)
            _contours_plotter(data_strf_object, cax, roi)
        for cax in roi_ax.flat[1::3]:
            rgb_plot = cax.imshow(processed_rgu, origin = "lower", interpolation = "none")
            cax.axis(False)
            _contours_plotter(data_strf_object, cax, roi)
        # Reshape times for convenience
        times = np.ma.copy(pygor.utilities.multicolour_reshape(data_strf_object.timecourses, 4))[:, roi]
        for cax in roi_ax.flat[2::3]:
           for colour in range(4):
               curr_colour = times[colour].T
               cax.plot(curr_colour, c = fish_palette[colour])
    plt.tight_layout()