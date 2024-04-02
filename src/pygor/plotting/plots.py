import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Ellipse
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import math
import scipy.signal
import warnings
from cycler import cycler
from matplotlib import colors
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipyfilechooser import FileChooser
# Local imports
import pygor.utils.utilities
import pygor.space
import pygor.temporal
import pygor.steps.contouring



rgbuv_palette = ["r", "g", "b", "violet"]
nanometers = ["588", "478", "422", "375"]
fish_palette = ["#ef8e00", "teal","#5600fe", "fuchsia"]
polarity_palette = ["grey", "black", "gainsboro"]
red_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey", "white","#ef8e00","darkred"])
green_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","mediumaquamarine","teal"])
blue_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","#5600fe","#4400cb"])
violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","fuchsia","#b22cb2"])


def play_movie(d3_arr, figaxim_return = False,**kwargs):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib

    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"
    # Check attributes and kwargs
    if d3_arr.ndim != 3:
        raise AttributeError(f"Array passed to function is not three dimensional (3D). Shape should be: (time,y,x)")
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Greys_r'
    else:
        cmap = kwargs['cmap']
    # Use RC context manager to temporarily use the modified rc dict 
    with matplotlib.rc_context(rc=plot_settings):
        # Initiate the figure, change themeing to Seaborn, create axes to tie colorbar too (for scaling)
        sns.set_theme(context='notebook')
        plt.ion()
        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '2%', '2%')
        # Plotting 
        im = ax.imshow(d3_arr[0], origin='lower', **kwargs)
        cb = fig.colorbar(im, cax = cax)
        # Scale colormap
        min_val = np.min(d3_arr)
        max_val = np.max(d3_arr)
        max_abs_val = np.max(np.abs(d3_arr))
        # im.set_clim(min_val, max_val)
        im.set_clim(-max_abs_val, max_abs_val)
        # Hide grid lines
        ax.grid(False)
        # Create a function that plot updates data in plt.imshow for each frame
        def video(frame):
            im.set_data(d3_arr[frame])
        # Create the animation based on the above function
        animation = matplotlib.animation.FuncAnimation(fig, video, frames=len(d3_arr), interval = 100, repeat_delay = 500)
        # Close the animation 
        plt.close()
        # Decide what to return based on input
        if figaxim_return == True:
            print("Returning vars (animation, fig, ax)")
            return animation, fig, ax, im
        else:
            return animation

def rois_overlay_object(data_strf_object):
    preprocess = pygor.utils.utilities.min_max_norm(np.std(data_strf_object.images, axis = 0), 0, 255)
    plt.imshow(preprocess, origin = "lower", cmap = "Greys_r",
    vmin = 100, vmax = 255)
    num_rois = int(np.abs(np.min(data_strf_object.rois)))
    # color = cm.jet(np.linspace(0, 1, int(np.abs(np.min(data_strf_object.rois)))))
    color = cm.get_cmap('jet_r', num_rois)
    plt.imshow(np.ma.masked_where(data_strf_object.rois.T == 1, data_strf_object.rois.T), cmap = color, alpha = 0.5, origin = "lower")
    plt.axis("off")

def rois_overlay_hdf5():
    """
    TODO as above but from hdf5 object directly (for ease)
    """
    return 1

def chroma_overview(data_strf_object, specify_rois=None, ipl_sort = False, y_crop = (0, 0), x_crop = (0 ,0),
    column_titles = ["588 nm", "478 nm", "422 nm", "375 nm"], colour_maps = [red_map, green_map, blue_map, violet_map],
    ax = None):
    if isinstance(colour_maps, Iterable) is False:
        colour_maps = [colour_maps] * len(column_titles)
    strfs_chroma = pygor.utils.utilities.multicolour_reshape(data_strf_object.collapse_times(), data_strf_object.numcolour)
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
        spaces = np.copy(pygor.utils.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
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
            desired_aspect = np.abs(np.diff(ax[n, 0].get_ylim())[0] / np.diff(ax[0,0].get_xlim())[0])
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

    strfs_chroma = pygor.utils.utilities.multicolour_reshape(data_strf_object.collapse_times(), data_strf_object.numcolour)
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
        spaces = np.copy(pygor.utils.utilities.auto_remove_border(strfs_chroma[:, rois[n]])) # this works
        rgb = np.abs(spaces[[0, 1, b_vals[n]]])
        processed_rgb = np.rollaxis(pygor.utils.utilities.min_max_norm(rgb, 0, 1), axis = 0, start = 3)
        ax.imshow(processed_rgb, origin = "lower")
        ax.axis(False)
    fig.tight_layout(pad = 0.1, h_pad = .1, w_pad=.1)

def _contours_plotter(data_strf_object, ax, roi, num_colours = 4):
    contours = pygor.utils.utilities.multicolour_reshape(data_strf_object.contours, 4)[:, roi]
    neg_contours = contours[:, 0]
    pos_contours = contours[:, 1]
    for colour in range(num_colours):
        for contour_n in neg_contours[colour]:
                ax.plot(contour_n[:, 1], contour_n[:, 0], lw = 1, ls = '-', c = fish_palette[colour], alpha = 1)# contour
        for contour_p in pos_contours[colour]:
                ax.plot(contour_p[:, 1], contour_p[:, 0], lw = 1, ls = "-", c = fish_palette[colour], alpha = 1)

def visualise_summary(data_strf_object, specify_rois, ipl_sort = False,  y_crop = (0, 0), x_crop = (0 , 0)):
    strfs_chroma = pygor.utils.utilities.multicolour_reshape(data_strf_object.collapse_times(), 4)
    strfs_rgb = np.abs(np.rollaxis((np.delete(strfs_chroma, 3, 0)), 0, 4))
    strfs_rgb = np.array([pygor.utils.utilities.min_max_norm(i, 0, 1) for i in strfs_rgb])
    strfs_rgu = np.abs(np.rollaxis((np.delete(strfs_chroma, 2, 0)), 0, 4))
    strfs_rgu = np.array([pygor.utils.utilities.min_max_norm(i, 0, 1) for i in strfs_rgu]);

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
        #spaces = np.copy(pygor.utils.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
        spaces = strfs_chroma[:, roi]
        # Prepare for RGB representation (by intiger)
        spaces[3] = np.roll(spaces[3], np.round(data_strf_object.calc_LED_offset(), 0).astype("int"), axis =(0,1))
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
        r, g, b, uv = spaces[0], spaces[1], spaces[2], spaces[3]
        rgb = np.abs(np.array([r,g,b]))
        rgu = np.abs(np.array([r,g,uv]))
        processed_rgb = np.rollaxis(pygor.utils.utilities.min_max_norm(rgb, 0, 1), axis = 0, start = 3)
        processed_rgu = np.rollaxis(pygor.utils.utilities.min_max_norm(rgu, 0, 1), axis = 0, start = 3)
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
        times = np.ma.copy(pygor.utils.utilities.multicolour_reshape(data_strf_object.timecourses, 4))[:, roi]
        for cax in roi_ax.flat[2::3]:
           for colour in range(4):
               curr_colour = times[colour].T
               cax.plot(curr_colour, c = fish_palette[colour])
    plt.tight_layout()

def _legacy_play_movie(d3_arr, **kwargs):
    if d3_arr.ndim != 3:
        raise AttributeError(f"Array passed to function is not three dimensional (3D). Shape should be: (time,x,y)")
    img_arr = d3_arr
    img_len = len(img_arr)-1
    def update_plots(frame):
        # Initialise plot
        fig, ax = plt.subplots()
        if 'cmap' in kwargs:
            STRF_map = ax.imshow(img_arr[frame], cmap = kwargs['cmap'], origin = "lower")   
        else:
            # STRF_map = ax.imshow(img_arr[frame], cmap = 'Greys_r', **kwargs)
            STRF_map = ax.imshow(img_arr[frame], cmap = 'Greys_r', **kwargs, origin = "lower")
        STRF_map.set_clim(np.min(d3_arr), np.max(d3_arr))
        fig.canvas.draw()

    play = widgets.Play(value=0, min=0, max=img_len, step=1, interval=100, description="Press play", disabled=False)
    slider = widgets.IntSlider(min=0, max=img_len, step=1, value=1, continuous_update=True)
    widgets.jslink((play, 'value'), (slider, 'value'))
    display(slider)
    interact(update_plots, frame = play)

def contouring_demo(arr_3d, level = None, display_time = True, returns = False, **kwargs):        
    # Collapse time to give us 2d representation
    arr_3d_collapsed = pygor.space.collapse_3d(arr_3d, zscore=False)
    # Extract content from masks
    neg_masked, pos_masked = pygor.space.rf_mask3d(arr_3d, level = level)
    neg_mask = neg_masked.mask[0]
    pos_mask = pos_masked.mask[1]
    # Extract time courses
    neg_time, pos_time = pygor.temporal.extract_timecourse(arr_3d, level = level)
    # Make contours
    if level == None:
        lower_contour, upper_contour = pygor.steps.contouring.contour(arr_3d_collapsed)
    else:
        lower_contour, upper_contour = pygor.steps.contouring.contour(arr_3d_collapsed, level = (-level, level))
    # Init figure 
    if display_time == True:
        fig, ax = plt.subplots(1, 3, subplot_kw={'aspect': 'auto', 'adjustable' : "datalim"}, figsize = (20,5))
        ## Optional 3: Plot extracted traces
        ax[2].plot(neg_time, c = 'red', label = "Neg masked")
        ax[2].plot(pos_time, c = 'blue', label = "Pos masked")
        ax[2].legend()
    if display_time == False:
        fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ## 1: Plot the 2D RF and the contours
    im = ax[0].imshow(arr_3d_collapsed, cmap = 'RdBu', origin = "lower")
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes('right', size='2%', pad=0.05) #autoscaling "pretend" axis to append colorbar to
    fig.colorbar(im, cax = cax1)
    max_Val = np.max(np.abs(arr_3d_collapsed))
    im.set_clim(-max_Val, max_Val)
    # ax[0].set_clim(-100, 100)
    for j in lower_contour:
        ax[0].plot(j[:, 1], j[:, 0], lw = 2, c = 'red')# contour 
    for k in upper_contour:
        ax[0].plot(k[:, 1], k[:, 0], lw = 2, c = 'blue')# contour
    ## 2: Plot masks
    mask = ax[1].imshow(np.invert(neg_mask * -1) + np.invert(pos_mask), origin = "lower", cmap = 'RdBu')
    if returns == True:
        return fig, ax, im
    # ax[2].plot(np.average(old_mask, axis = (1, 2)), c = 'yellow', ls = 'dotted', label = "Old mask")
    # plt.close()

def tiling(Data_strf_object, deletion_threshold = 0, chromatic = False, x_lim = None, y_lim = None, **kwargs):
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
    contours = np.copy(Data_strf_object.contours)
    if "shrink_factor" in kwargs and kwargs["shrink_factor"] != None:
        contours = _transform_contours(contours, _shrink_contour, kwargs["shrink_factor"])
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
    val_tup = (min_val, max_val)
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
    combined2 = ax[2].imshow(min_projection, cmap = 'RdBu', alpha = 0.66, origin = "lower")
    combined = ax[2].imshow(max_projection, cmap = 'RdBu', alpha = 0.33, origin = "lower")
    combined.set_clim(val_tup)
    combined2.set_clim(val_tup)
    # Finally plot contours accordingly
    if chromatic == True:
        n = 0 
        for i in contours:
            upper, lower = i
            if len(upper) != 0:
                for contour_up in upper:
                    ax[0].plot(contour_up[:, 1], contour_up[:, 0], lw = 2, c = fish_palette[n], alpha = .5)# contour
                    ax[2].plot(contour_up[:, 1], contour_up[:, 0], lw = 2, c = fish_palette[n], alpha = .5)# contour
            if len(lower) != 0:
                for contour_low in lower:
                    ax[1].plot(contour_low[:, 1], contour_low[:, 0], lw = 2, c = fish_palette[n], alpha = .5)# contour
                    ax[2].plot(contour_low[:, 1], contour_low[:, 0], lw = 2, c = fish_palette[n], alpha = .5)# contour                
            n += 1
            if n == 4:
                n = 0
    else:
        for i in cleaned_contours:
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

def stack_to_rgb(stack, eight_bit = True):
    """
    Convert a single-channel stack to an RGB image.

    Parameters
    ----------
    stack : numpy.ndarray
        Input stack with a single channel.
    eight_bit : bool, optional
        If True, the output RGB image will be in 8-bit format (0-255),
        otherwise in floating-point format (0.0-1.0). Default is True.

    Returns
    -------
    numpy.ndarray
        RGB image obtained from the input stack.

    Notes
    -----
    This function assumes that `pygor.utils.utilities.min_max_norm` is a valid function
    for normalizing the stack values.

    Examples
    --------
    >>> stack = np.random.random((256, 256))
    >>> rgb_image = stack_to_rgb(stack)

    >>> stack = np.random.random((256, 256))
    >>> rgb_image = stack_to_rgb(stack, eight_bit=False)
    """
    # For RGB representation, the universal standard is time, y, x, rgb so:
    # stack = np.rollaxis(stack, time_axis, )
    # stack = np.swapaxis()
    stack = np.repeat(np.expand_dims(stack, -1), 3, -1)
    if eight_bit == True:
        stack = pygor.utils.utilities.min_max_norm(stack, 0, 255).astype('int')
    else:
        stack = pygor.utils.utilities.min_max_norm(stack, 0, 1).astype('float')
    return stack

def basic_stim_overlay(stack, frame_duration = 32, frame_width = 125, repeat_interval = 4, xy_loc = (3, 3), size = 10, colour_list = fish_palette):
    """
    Overlay basic stimuli on an RGB image.

    Parameters
    ----------
    rgb_array : numpy.ndarray
        Input RGB image array.
    frame_width : int, optional
        Width of each stimulus frame. Default is 125.
    xy_loc : tuple, optional
        (x, y) coordinates of the top-left corner of the stimuli. Default is (3, 3).
    frame_duration : int, optional
        Duration of each frame. Default is 32.
    size : int, optional
        Size of each stimulus. Default is 10.
    repeat_interval : int, optional
        Number of times to repeat the color sequence. Default is 4.
    colour_list : list of tuples, optional
        List of RGB tuples or matplotlib-listed colours representing the colors of the stimuli.

    Returns
    -------
    numpy.ndarray
        RGB image with overlaid stimuli.

    Examples
    --------
    >>> rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
    >>> overlay_image = basic_stim_overlay(rgb_image)

    >>> rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
    >>> overlay_image = basic_stim_overlay(rgb_image, frame_width=150, size=15)
    """
    rgb_array = stack_to_rgb(stack)
    # Prep colour list 
    if len(colour_list) == 1:
        np.tile(colour_list, (repeat_interval, 1))
    for i in range(repeat_interval):
        # Frame, y, x
        rgb_array[0:frame_duration, xy_loc[1]:xy_loc[1]+size, xy_loc[0] + frame_width*i: xy_loc[0] + size+frame_width*i] = np.array(colors.to_rgb(colour_list[i])) * 255
    return rgb_array

def ipl_summary_chroma(chroma_df):
    polarities = [-1, 1]
    colours = ["R", "G", "B", "UV"]
    fig, ax = plt.subplots(2, 4, figsize = (12, 7), sharex = True, sharey=True)
    bins = 10
    # sns.set_style("whitegrid")
    for n, i in enumerate(polarities):
        for m, j in enumerate(colours):
            hist_vals_per_condition = np.histogram(chroma_df.query(f"polarities ==  {i} & colour == '{j}'")["ipl_depths"], bins = bins)[0]
            hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
            # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
            percentages = hist_vals_per_condition  / np.sum(hist_vals_population) * 100
            # percentages = hist_vals_per_condition
            ax[n, m].barh(np.arange(0, 100, 10), width= percentages, height=10, color = fish_palette[m], edgecolor="black", alpha = 0.75)        
            ax[n, m].grid(False)
            ax[n, m].axhline(55, c = "k", ls = "--")
            # ax[n, m].get_xaxis().set_visible(False)
            # ax[n, m].spines["bottom"].set_visible(False)
            if m == 0:
                ax[n, m].set_ylabel("IPL depth (%)")
                ax[n, m].text(x = 14, y = 53+5, s = "OFF", size = 10)
                ax[n, m].text(x = 14, y = 53-5, s = "ON", size = 10)
                if i == -1:
                    ax[n, m].set_title("OFF", weight = "bold", c = "grey", loc = "left")
                if i == 1:
                    ax[n, m].set_title("ON", weight = "bold", loc = "left")
            ax[0, m].set_title(nanometers[m] + "nm", size = 12)
            num_cells = len(pd.unique(chroma_df["cell_id"]))
            """
            TODO this counts cell number incorrectly, duplicate cell_ids
            """
            ax[1, 0].set_xlabel(f"Percentage by colour (n = {num_cells})", size = 15)
    plt.show()
    raise UserWarning("Fix 'cell_ID' implementation in roi_df")

def ipl_summary_polarity(roi_df):
    polarities = [-1, 1, 2]
    fig, axs = plt.subplots(1, 3, figsize = (8, 4), sharex = True, sharey=True)
    bins = 10
    titles = ["OFF", "ON", "Mixed polarity"]
    # sns.set_style("whitegrid")
    for n, ax in enumerate(axs.flatten()):
        hist_vals_per_condition = np.histogram(roi_df.query(f"polarities ==  {polarities[n]}")["ipl_depths"], bins = bins)[0]
        hist_vals_population = np.histogram(roi_df["ipl_depths"], bins = bins)[0]
        # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
        percentages = hist_vals_per_condition  / np.sum(hist_vals_population) * 100
        ax.barh(np.arange(0, 100, 10), width= percentages, height=10, color = polarity_palette[n], edgecolor="black", alpha = 0.75)        
        ax.set_title(titles[n], size = 12)
        ax.axhline(55, c = "k", ls = "--")
    axs[0].text(x = 14, y = 53+5, s = "OFF", size = 10)
    axs[0].text(x = 14, y = 53-5, s = "ON", size = 10)
    num_cells = len(pd.unique(roi_df["cell_id"]))
    axs[0].set_xlabel(f"Percentage by polarity (n = {num_cells})", size = 15)
    plt.show()

def plot_traces(array_2d, mode = None, on_dur = None, off_dur = None, plot_type = "traces", axis = -1):
    if plot_type == "traces":
        # if pop_array.shape[axis] > 50:
        #     raise warning
        num_traces = array_2d.shape[axis]
        time_dur = array_2d.shape[axis-1]
        color = cm.jet(np.linspace(0, 1, num_traces))
        if num_traces < 10:
            fig, axs = plt.subplots(num_traces, figsize = (4, num_traces), sharex = True, sharey = True)
        if num_traces > 10:
            fig, axs = plt.subplots(num_traces, figsize = (4, int(num_traces/2)), sharex = True, sharey = True)        
        for n, ax in enumerate(axs.flatten()):
            ax.plot(np.take(array_2d, n, axis), c = color[n])
            ax.grid(False)
            if n < len(axs.flatten()) - 1:
                sns.despine(ax = ax, bottom = True, left = True)
                ax.tick_params(axis='y', which='both',labelleft=False)#, colors= "white")
            else:
                sns.despine(ax = ax, bottom = True)
            if mode != None:
                # If on_dur or off_dur not specified, assume traces are averaged and do 1 iteration of equal lengths
                if on_dur == None and off_dur == None:
                    assumed_onoff_dur = time_dur/mode/2 
                    for reps in range(mode*2)[::2]: # a bit madness but works
                        ax.axvspan(assumed_onoff_dur * reps, assumed_onoff_dur * reps + assumed_onoff_dur, color = "lightgrey", lw=0, alpha = 0.25)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()