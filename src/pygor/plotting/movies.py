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
import pygor.strf.spatial
import pygor.strf.temporal
import pygor.strf.contouring
from . import custom

def play_movie(d3_arr, figaxim_return = False, rgb_repr=False,**kwargs):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib

    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"
    # Check attributes and kwargs
    if rgb_repr is False:
        if d3_arr.ndim != 3:
            raise AttributeError("Array passed to function is not three dimensional (3D). Shape should be: (time,y,x)")
    if rgb_repr is True:
        if d3_arr.shape[-1] != 3:
            raise AttributeError("Expected RGB representation in final index. Shape should be: (time,y,x,RGB)")
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

def play_movie_4d_simple(d4_arr, figaxim_return = False, show_cbar = True, **kwargs):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
    if d4_arr.ndim != 4:
        raise AttributeError("Array passed to function is not four dimensional (4D). Please use 'play_movie' instead.")
    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 200
    plot_settings["savefig.facecolor"] = "white"
    # Check attributes and kwargs
    if d4_arr.ndim != 4:
        raise AttributeError("Array passed to function is not four dimensional (4D). Please consider 'play_movie' instead, or check inputs.")
    len_4d = d4_arr.shape[0]
    frames_time = d4_arr.shape[1]
    max_abs_val = np.max(np.abs(d4_arr))
    if 'cmap_list' not in kwargs:
        cmap_list = ['Greys_r'] * len_4d
    else:
        cmap_list = kwargs['cmap_list']
    # Use RC context manager to temporarily use the modified rc dict 
    with matplotlib.rc_context(rc=plot_settings):
        # Initiate the figure, change themeing to Seaborn, create axes to tie colorbar too (for scaling)
        fig, axs = plt.subplots(1, len_4d, figsize = (4*len_4d, 1*len_4d), gridspec_kw = {'wspace' : 0.2, 'hspace' : 0.0})
        for n, ax in enumerate(axs):
            # Hide grid lines
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.grid(False)
            ax.axis('off')
            ax.set_aspect('equal')
            # Plotting 
            im1 = ax.pcolormesh(d4_arr[n, 0], cmap = cmap_list[n])
            # Equalise the colormap
            im1.set_clim(-max_abs_val, max_abs_val)
            if show_cbar is True:
                # Optional colorbars
                div = make_axes_locatable(ax)
                cax = div.append_axes('right', '5%', '2%')
                fig.colorbar(im1, cax = cax)
                cax.get_xaxis().set_visible(False)
                cax.get_yaxis().set_visible(False)
                cax.axis('off')
        def video(frame):
            # axs = plt.gca()
            for n, ax in enumerate(axs):
                # Get image object from axis 
                im = ax.collections[0]
                # Fill the animation with data
                im.set_array(d4_arr[n][frame])
        # Create the animation based on the above function
        animation = matplotlib.animation.FuncAnimation(fig, video, frames=frames_time, interval = 80*1.5, repeat_delay = 500)    
        plt.close()
        return animation

# def _fill_plot(axs, arr, show_cbar, cmap_list, norm_each):
#     if norm_each == True:
#         max_abs_val = np.max(np.abs(arr))
#     for n, ax in enumerate(axs.flat):
#         fig = plt.gcf()
#         # Hide grid lines
#         ax.set_xticks([])
#         ax.set_yticks([])        
#         ax.grid(False)
#         ax.axis('off')
#         ax.set_aspect('equal')
#         # Plotting 
#         im = ax.pcolormesh(arr[n, 0], cmap = cmap_list[n])
#         # Equalise the colormap
#         im.set_clim(-max_abs_val, max_abs_val)
#         # Optional colorbars
#         if show_cbar is True:
#             div = make_axes_locatable(ax)
#             cax = div.append_axes('right', '5%', pad = '2%')
#             fig.colorbar(im, cax = cax)

def play_movie_4d(input_arr, figaxim_return = False, show_cbar = False, 
    axes = None, axis = None, frameon = False, norm_by = 'input', **kwargs):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 200
    plot_settings["savefig.facecolor"] = "white"
    if input_arr.ndim == 4:
        input_arr = np.expand_dims(input_arr, 1)
    # Reshape and treat sequentially (for sanity)
    elif input_arr.ndim == 5:
        pass
    else:
        raise AttributeError("Input dimensions are ambigious or unpexted. Revise input arrays.")
    if axes != None and axis != None:
        raise AttributeError("Got inputs for both 'axes' and 'axis'")
    axes = [0, 1, 2]
    if axis != None:
        axes[0], axes[axis] = axes[axis], axes[0]
    columns = input_arr.shape[axes[0]]
    rows    = input_arr.shape[axes[1]]
    frames  = input_arr.shape[axes[2]]
    y       = input_arr.shape[3]
    x       = input_arr.shape[4]
    # input_arr = np.concatenate(input_arr, axis = 0)
    input_arr = input_arr.reshape(columns * rows, frames, y, x, order = "f")
    if norm_by == 'input':
        max_abs_val = np.max(np.abs(input_arr))
    # Check attributes and kwargs
    if 'cmap_list' not in kwargs:
        cmap_list = ['Greys_r'] * rows * columns
    else:
        cmap_list = np.tile(kwargs['cmap_list'], int(input_arr.shape[0] / columns))
    # Use RC context manager to temporarily use the modified rc dict 
    with matplotlib.rc_context(rc=plot_settings):
        # Initiate the figure, change themeing to Seaborn, create axes to tie colorbar too (for scaling)
        fig, axs = plt.subplots(rows, columns, figsize = (columns*1.9, rows), frameon = frameon,
        gridspec_kw = {'wspace' : 0.0, 'hspace' : 0.0, 'bottom': 0.01, 'top': .99})
        for n, ax in enumerate(axs.flat):
            fig = plt.gcf()
            # Hide grid lines
            ax.set_xticks([])
            ax.set_yticks([])        
            ax.grid(False)
            ax.axis('off')
            ax.set_aspect('equal')
            # Plotting
            im = ax.imshow(input_arr[n, 0], cmap = cmap_list[n])
            # Equalise the colormap
            im.set_clim(-max_abs_val, max_abs_val)
            # Optional colorbars
            if show_cbar is True:
                div = make_axes_locatable(ax)
                cax = div.append_axes('right', '5%', pad = '2%')
                fig.colorbar(im, cax = cax)
        if kwargs.get('row_labels') != None: 
            # fig.subplots_adjust(right=.8, top=.95, wspace=0, hspace=0)
            fig.set_frameon(True)
            for label, ax in zip(kwargs.get('row_labels'), axs.flat[::4]):
                ax.axis('on')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_ylabel(label, rotation = 0, size = 20, labelpad = 40)
        if show_cbar == True:
            fig.set_figwidth(columns*3)
        def video(frame):
            for n, ax in enumerate(axs.flat):
                # Get image object from axis 
                im = ax.get_images()[0]
                # Fill the animation with data
                im.set_array(input_arr[n][frame])
        # Create the animation based on the above function
        animation = matplotlib.animation.FuncAnimation(fig, video, frames=frames, interval = 80*1.5, repeat_delay = 500)    
        plt.close()
        return animation

