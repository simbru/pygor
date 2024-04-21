import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import colors
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
from ipywidgets import interact
import ipywidgets as widgets
# Local imports
import pygor.utilities
import pygor.strf.space
import pygor.strf.temporal
import pygor.strf.contouring
import pygor.plotting.custom

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
        raise AttributeError("Array passed to function is not three dimensional (3D). Shape should be: (time,y,x)")
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
        fig.colorbar(im, cax = cax)
        # Scale colormap
#        min_val = np.min(d3_arr)
#        max_val = np.max(d3_arr)
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
        if figaxim_return is True:
            print("Returning vars (animation, fig, ax)")
            return animation, fig, ax, im
        else:
            return animation

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
    display(slider)  # noqa: F821
    interact(update_plots, frame = play)

def contouring_demo(arr_3d, level = None, display_time = True, returns = False, **kwargs):        
    # Collapse time to give us 2d representation
    arr_3d_collapsed = pygor.strf.space.collapse_3d(arr_3d, zscore=False)
    # Extract content from masks
    neg_masked, pos_masked = pygor.strf.space.rf_mask3d(arr_3d, level = level)
    neg_mask = neg_masked.mask[0]
    pos_mask = pos_masked.mask[1]
    # Extract time courses
    neg_time, pos_time = pygor.strf.temporal.extract_timecourse(arr_3d, level = level)
    # Make contours
    if level == None:
        lower_contour, upper_contour = pygor.strf.contouring.contour(arr_3d_collapsed)
    else:
        lower_contour, upper_contour = pygor.strf.contouring.contour(arr_3d_collapsed, level = (-level, level))
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
    This function assumes that `pygor.utilities.min_max_norm` is a valid function
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
        stack = pygor.utilities.min_max_norm(stack, 0, 255).astype('int')
    else:
        stack = pygor.utilities.min_max_norm(stack, 0, 1).astype('float')
    return stack

def basic_stim_overlay(stack, frame_duration = 32, frame_width = 125, 
    repeat_interval = 4, xy_loc = (3, 3), size = 10, colour_list = pygor.plotting.custom.fish_palette):
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
            ax[n, m].barh(np.arange(0, 100, 10), width= percentages, height=10, color = pygor.plotting.custom.fish_palette[m], edgecolor="black", alpha = 0.75)        
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
            ax[0, m].set_title(pygor.plotting.custom.nanometers[m] + "nm", size = 12)
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
        ax.barh(np.arange(0, 100, 10), width= percentages, height=10, color = pygor.plotting.custom.polarity_palette[n], edgecolor="black", alpha = 0.75)        
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