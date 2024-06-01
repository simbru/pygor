import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.utilities
import pygor.strf.spatial
import pygor.strf.temporal
import pygor.strf.contouring
import seaborn as sns

from pygor.plotting.custom import red_map, green_map, blue_map, violet_map, fish_palette

def chroma_overview(data_strf_object, specify_rois=None, ipl_sort = False, y_crop = (0, 0), x_crop = (0 ,0),
    column_titles = ["588 nm", "478 nm", "422 nm", "375 nm"], colour_maps = [red_map, green_map, blue_map, violet_map],
    contours = False, ax = None, high_contrast = True, remove_border = True, labels = None, clim = None):
    if isinstance(colour_maps, Iterable) is False:
        colour_maps = [colour_maps] * len(column_titles)
    if isinstance(data_strf_object, pygor.classes.strf_data.STRF) is False:
        warnings.warn("Input object is not a STRF object. Attempting to treat as nxm Numpy array. Use-case not intended, expect errors.")
        strfs_chroma = data_strf_object
        numcolour = strfs_chroma.shape[0]
        remove_border = False
    else:
        strfs_chroma = pygor.utilities.multicolour_reshape(data_strf_object.collapse_times(), data_strf_object.numcolour)
        numcolour =  data_strf_object.numcolour
    if clim == None:
        abs_max = np.max(np.abs(strfs_chroma))
        clim_vals = (-abs_max, abs_max)
    else:
        clim_vals = clim
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
        fig, ax = plt.subplots(len(specify_rois), numcolour, figsize = (numcolour*1.5*2, len(specify_rois) * 2))
    else:
        fig = plt.gcf()
    for n, roi in enumerate(specify_rois):
        if remove_border is True:
            spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
            border_tup = pygor.utilities.check_border(strfs_chroma[:, roi])
        else:
            spaces = strfs_chroma[:, roi]
            border_tup = (0, 0, 0, 0)
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        # plotting depending on specified number of rois (more or less than 1)
        if len(specify_rois) > 1:
            for i in range(4):
                strf = ax[-n-1, i].imshow(spaces[i], cmap = colour_maps[i], origin = "lower")
                strf.set_clim(clim_vals)
                if n == 0:
                    for j in range(numcolour):
                        ax[-n, j].set_title(column_titles[j])
                # Handle contours optionally 
                if contours == True:
                    _contours_plotter(data_strf_object, roi = roi, index = i, ax = ax[-n-1, i], xy_offset = (-border_tup[0], -border_tup[2]), high_contrast = high_contrast)
            np.abs(np.diff(ax[n, 0].get_ylim())[0] / np.diff(ax[0,0].get_xlim())[0])
        else:
            for i in range(4):
                strf = ax[i].imshow(spaces[i], cmap = colour_maps[i])
                strf.set_clim(clim_vals)
                if roi == 0:
                    for j in range(4):
                        ax[j].set_title(column_titles[j])
                # Handle contours optionally
                if contours == True:
                    _contours_plotter(data_strf_object, roi = roi, index = i, ax = ax[i], xy_offset = (-border_tup[0], -border_tup[2]), high_contrast = high_contrast)
    for axis in ax.flat:
        axis.axis(False)
    if labels != None:
        for axis, label in zip(ax.flat[::numcolour], labels):
            axis.axis(True)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_ylabel(label, rotation = 'horizontal', labelpad = 15)
    # fig.tight_layout(pad = 0, h_pad = .1, w_pad=.1)
    return fig, ax

def _contours_plotter(data_strf_object, roi, index =  None, xy_offset = (0, 0), high_contrast = True, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    contours = pygor.utilities.multicolour_reshape(data_strf_object.fit_contours(), 4)[:, roi]
    neg_contours = contours[:, 0]
    pos_contours = contours[:, 1]
    if index is None:
        index = range(len(contours))
    if isinstance(index, Iterable) is False:
        index = [index]
    for colour in index:
        for contour_n in neg_contours[colour]:
                if high_contrast == True:
                    ax.plot(contour_n[:, 1] + xy_offset[1], contour_n[:, 0] + xy_offset[0], lw = 2, ls = '-', c = 'white', alpha = 1)# contour
                    ax.plot(contour_n[:, 1] + xy_offset[1], contour_n[:, 0] + xy_offset[0], lw = 2, ls = 'dashed', c = fish_palette[colour], alpha = 1)# contour
                else:
                    ax.plot(contour_n[:, 1] + xy_offset[1], contour_n[:, 0] + xy_offset[0], lw = 1, ls = '-', c = fish_palette[colour], alpha = 1)# contour
        for contour_p in pos_contours[colour]:
                if high_contrast == True:
                    ax.plot(contour_p[:, 1] + xy_offset[1], contour_p[:, 0] + xy_offset[0], lw = 2, ls = "-", c = 'white', alpha = 1)                    
                    ax.plot(contour_p[:, 1] + xy_offset[1], contour_p[:, 0] + xy_offset[0], lw = 2, ls = "dashed", c = fish_palette[colour], alpha = 1)
                else:
                    ax.plot(contour_p[:, 1] + xy_offset[1], contour_p[:, 0] + xy_offset[0], lw = 1, ls = "-", c = fish_palette[colour], alpha = 1)

def rgb_representation(data_strf_object, colours_dims = [0, 1, 2, 3], specify_rois=None, ipl_sort = False, y_crop = (0, 0), x_crop = (0 ,0),
    ax = None, contours = False, remove_border = True):

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
        fig, axs = plt.subplots(len(specify_rois), n_cols, sharex=True, sharey=True, figsize = (n_cols * 4, len(specify_rois) * 2))
    else:
        fig = plt.gcf()
    rois = list(specify_rois) * 2
    if len(specify_rois) == 1:
        axs = [axs]
    for n, ax in enumerate(axs):
        roi = specify_rois[n] # Because each row represents a roi
        if remove_border is True:
            spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, rois[n]])) # this works
        else:
            spaces = strfs_chroma[:, rois[n]]
        # Summary of spatial components 
        #spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
        spaces = strfs_chroma[:, roi]
        # Prepare for RGB representation (by intiger)
        # led_offset = data_strf_object.calc_LED_offset()
        # print(led_offset)
        # spaces[3] = np.roll(spaces[3], np.round(led_offset, 0).astype("int"), axis =(0 ,1))
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        r, g, b, uv = spaces[0], spaces[1], spaces[2], spaces[3]
        rgb = np.abs(np.array([r,g,b]))
        rgu = np.abs(np.array([r,g,uv]))
        processed_rgb = np.rollaxis(pygor.utilities.min_max_norm(rgb, 0, 1), axis = 0, start = 3)
        processed_rgu = np.rollaxis(pygor.utilities.min_max_norm(rgu, 0, 1), axis = 0, start = 3)
        # for cax in roi_ax[0]:
        rgb_plot = ax[0].imshow(processed_rgb, origin = "lower", interpolation = "none")
        ax[0].axis(False)
        _contours_plotter(data_strf_object, index = [0,1,2], roi = roi, ax = ax[0])#, xy_offset = (led_offset, led_offset))
        rgu_plot = ax[1].imshow(processed_rgu, origin = "lower", interpolation = "none")
        ax[1].axis(False)
        _contours_plotter(data_strf_object, index=[0,1,3], roi = roi, ax = ax[1])#, xy_offset = (led_offset, led_offset[0]))
    fig.tight_layout(pad = 0.1, h_pad = .1, w_pad=.1)
    return fig

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
    fig, ax = plt.subplots(len(specify_rois), 3, figsize = (3*1.3*2 , len(specify_rois) * 1.7))
    for n, roi in enumerate(reversed(specify_rois)):
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
            _contours_plotter(data_strf_object, roi = roi, ax = cax)
        for cax in roi_ax.flat[1::3]:
            rgu_plot = cax.imshow(processed_rgu, origin = "lower", interpolation = "none")
            cax.axis(False)
            _contours_plotter(data_strf_object, roi = roi, ax = cax)
        # Reshape times for convenience
        times = np.ma.copy(pygor.utilities.multicolour_reshape(data_strf_object.get_timecourses(), 4))[:, roi]
        for cax in roi_ax.flat[2::3]:
           for colour in range(4):
               curr_colour = times[colour].T
               cax.plot(curr_colour, c = fish_palette[colour])
               cax.set_xticks(np.linspace(0, 20, 5), np.round(np.linspace(0, 1.3, 5), 2))
    plt.tight_layout()
    return fig, ax 

def tiling(Data_strf_object, deletion_threshold = 0, chromatic = False, x_lim = None, y_lim = None, **kwargs):
    """
    Visualizes the tiling of spectro-temporal receptive fields (STRFs).

    This function takes a Data_strf_object, which is assumed to have methods for
    fitting contours and collapsing times. It optionally shrinks the contours and
    filters out contours based on a deletion threshold. It then generates a series
    of plots that show the minimum and maximum projections of the collapsed times,
    as well as a combined plot with optional chromatic or monochromatic contour
    overlays.

    Parameters
    ----------
    Data_strf_object : object
        An object with methods for fitting contours and collapsing times, which are
        used to compute and visualize the STRF tilings.
    deletion_threshold : float, optional
        Threshold below which contours are deleted from the visualization based on
        the maximum amplitude across collapsed times (default is 0).
    chromatic : bool, optional
        If True, contours are plotted in color; otherwise, they are plotted in red
        and blue (default is False).
    x_lim : tuple of int, optional
        Limits for the x-axis of the plots (default is None).
    y_lim : tuple of int, optional
        Limits for the y-axis of the plots (default is None).
    **kwargs : dict
        Additional keyword arguments. Can include 'shrink_factor' to determine the
        scaling factor by which the contours are shrunk.

    Returns
    -------
    None
        The function does not return any values but generates matplotlib plots.
    """
    def _shrink_contour(coordinates, scale_factor):
        # Step 1: Find the center of the contour
        center = np.mean(coordinates, axis=0)
        # Step 2: Translate coordinates to make the center the origin
        translated_coordinates = coordinates - center
        # Step 3: Scale the coordinates to shrink the contour
        scaled_coordinates = scale_factor * translated_coordinates
        # Step 4: Translate coordinates back to their original position
        final_coordinates = scaled_coordinates + center
        return final_coordinates
    def _transform_contours(contours, transform_funct, *params):
        new_contours = []
        for lower, upper in contours:
            curr_upper = []
            curr_lower = []
            for i in upper:
                inner_upper = transform_funct(i, *params)
                curr_upper.append(inner_upper)
            for j in lower:
                inner_lower = transform_funct(j, *params)
                curr_lower.append(inner_lower)
            new_contours.append([curr_lower, curr_upper])
        #return np.array(new_lowers, dtype = "object"), np.array(new_uppers, dtype = "object")
        return np.array(new_contours, dtype = "object")
    # Make a copy of the array view
    contours = np.copy(Data_strf_object.fit_contours())
    if "shrink_factor" in kwargs and kwargs["shrink_factor"] != None:
        contours = _transform_contours(contours, _shrink_contour, kwargs["shrink_factor"])
    if "shrink_factor" not in kwargs:
        kwargs["shrink_factor"] = 1
    absolute_version = np.abs(Data_strf_object.collapse_times())
    indeces_to_delete = np.unique(np.where(np.max(absolute_version, axis = (1,2)) < deletion_threshold)[0]) # filtering criteria based on amplitudes
    # Kick out contours accordingly 
    cleaned_version = np.delete(Data_strf_object.collapse_times(), indeces_to_delete, axis = 0)
    #cleaned_contours = list(np.delete(np.array(contours, dtype = "object"), indeces_to_delete, axis = 0))
    #print(cleaned_contours)
    contours[:, :][indeces_to_delete] = [[[]]] # I dont understand why this works but it does
#    cleaned_contours = contours
    # Make projectsion 
    min_projection = np.min(cleaned_version, axis = 0) 
    max_projection = np.max(cleaned_version, axis = 0)
    min_val = np.min(min_projection)
    max_val = np.max(max_projection)
    # create plot 
    fig, ax = plt.subplots(3, 1, figsize = (20, 20))
    # Plot projections
    minproj = ax[0].imshow(min_projection, cmap = 'RdBu', origin = "lower")
    minproj.set_clim(min_val, max_val)
    plt.colorbar(minproj, ax = ax[0])
    # Plot the other projection
    maxproj = ax[1].imshow(max_projection, cmap = 'RdBu', origin = "lower")
    maxproj.set_clim(min_val, max_val)
    plt.colorbar(maxproj, ax = ax[1])
    # Plot their combination
    combined2 = ax[2].imshow(np.abs(min_projection) + np.abs(max_projection), cmap = 'Greys', alpha = 1, origin = "lower")
    combined2.set_clim(0, max_val)
    plt.colorbar(combined2, ax = ax[2])
    # Finally plot contours accordingly
    if chromatic == True:
        n = 0 
        for i in contours:
            upper, lower = i
            if len(upper) != 0:
                for contour_up in upper:
                    ax[0].plot(contour_up[:, 1], contour_up[:, 0], lw = 2, c = pygor.plotting.custom.fish_palette[n], alpha = .5)# contour
                    ax[2].plot(contour_up[:, 1], contour_up[:, 0], lw = 2, c = pygor.plotting.custom.fish_palette[n], alpha = .5)# contour
            if len(lower) != 0:
                for contour_low in lower:
                    ax[1].plot(contour_low[:, 1], contour_low[:, 0], lw = 2, c = pygor.plotting.custom.fish_palette[n], alpha = .5)# contour
                    ax[2].plot(contour_low[:, 1], contour_low[:, 0], lw = 2, c = pygor.plotting.custom.fish_palette[n], alpha = .5)# contour                
            n += 1
            if n == 4:
                n = 0
    else:
        for i in contours:
            upper, lower = i
            if len(upper) != 0:
                for contour_up in upper:
                    ax[0].plot(contour_up[:, 1] / kwargs["shrink_factor"], contour_up[:, 0] / kwargs["shrink_factor"], lw = 2, c = 'red', alpha = .25)# contour
                    ax[2].plot(contour_up[:, 1] / kwargs["shrink_factor"], contour_up[:, 0] / kwargs["shrink_factor"], lw = 2, c = 'red', alpha = .4)# contour
            if len(lower) != 0:
                for contour_low in lower:
                    ax[1].plot(contour_low[:, 1] / kwargs["shrink_factor"], contour_low[:, 0] / kwargs["shrink_factor"], lw = 2, c = 'blue', alpha = .25)# contour
                    ax[2].plot(contour_low[:, 1] / kwargs["shrink_factor"], contour_low[:, 0] / kwargs["shrink_factor"], lw = 2, c = 'blue', alpha = .4)# contour
    if x_lim != None:
        for a in ax.flat:
            a.set_xlim(x_lim[0], x_lim[1])
    if y_lim != None:
        for a in ax.flat:
            a.set_ylim(y_lim[0], y_lim[1])
    # ax[3].imshow(np.average(load.images, axis = 0), cmap = 'Greys_r', origin = "lower")
    # plt.savefig(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Conferences\Life Sciences PhD Careers Symposium 2023\RF_tiling.svg")

def multi_chroma_movie(strf_object, roi, show_cbar = False, **kwargs):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib

    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"

    num_colours = strf_object.numcolour
    multichrom = pygor.utilities.multicolour_reshape(strf_object.strfs, num_colours)[:, roi]
    # Use RC context manager to temporarily use the modified rc dict 
    animation = pygor.plotting.play_movie_4d(multichrom, show_cbar=show_cbar, cmap_list = pygor.plotting.maps_concat)
    return animation

def spatial_colors(d3_srf_arr):
    minmax_abs = np.max(np.abs(d3_srf_arr))
    fig, axs = plt.subplots(1, 4, figsize = (10, 4))
    for n, ax in enumerate(axs):
        ax.pcolormesh(d3_srf_arr[n], vmin = -minmax_abs, vmax = minmax_abs, cmap = pygor.plotting.maps_concat[n])
        ax.set_aspect("equal")
        ax.axis('off')
    plt.close()
    return fig

