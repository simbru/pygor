import numpy as np 
import pathlib 
import h5py
import os
import datetime 
import pandas as pd
import warnings
import scipy
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
# import seaborn_image as isns
sns.set_theme()
import os
import dacite
import napari
import skimage.measure
import copy
import stumpy
import math

import space
import plotting
import temporal
import utilities
import filehandling
import utilities
import signal_analysis
import unit_conversion

sns.set_context("paper")
sns.set_style("whitegrid")
matplotlib.rcParams['svg.fonttype'] = 'none'

"""Need to get an algorithm that finds/quantifies the delayed_off response specifically
https://stumpy.readthedocs.io/en/latest/Tutorial_Pattern_Matching.html 
"""

def on_off_index(on_val, off_val):
    return (on_val - off_val) / (on_val + off_val)

# base this on response segment
def on_off_metric(on_segment, off_segment, on_crop_first_last = (0, 0), off_crop_first_last = (0, 0), metric = 'auc'):
    on_segment = on_segment[on_crop_first_last[0]:len(on_segment)-on_crop_first_last[1]]
    off_segment= off_segment[off_crop_first_last[0]:len(off_segment)-off_crop_first_last[1]]
    if metric == 'auc':
        on_metric = np.trapz(on_segment)
        off_metric = np.trapz(off_segment)
    return on_off_index(on_metric, off_metric)

def pop_split_by_colour(pop_array, stim_mode):
    "Will throw error if stim mode is not correcttly aligned with array length (no residual allowed)"
    if len(pop_array) % 2 != 0:
        # Desparate attempt to account for residuals if/when smoothing data (due to loosing n data points)
        pop_array = pop_array[:len(pop_array) - ((math.ceil(len(pop_array)/4 % 2 * stim_mode)))]
    split_pop = np.array(np.hsplit(pop_array.T, np.arange(int(len(pop_array)/stim_mode), len(pop_array), int(len(pop_array)/stim_mode))))
    return split_pop

def on_off_split(response_segment, on_len, off_len):
    segment_length = len(response_segment)
    stim_total_for_ratio = on_len + off_len
    on_off_proportion = on_len / (stim_total_for_ratio)
    try:
        return np.array(np.split(response_segment, [int(on_off_proportion * segment_length)]))
    except ValueError:
        return np.array(np.split(response_segment, [int(on_off_proportion * segment_length)]), dtype = "object")

def pop_on_off_colour_split(pop_array, stim_mode, on_len, off_len, axis = 2):
    """
    Split an averaged population array by color and then by ON/OFF periods.

    This function is intended for use with averaged traces.

    Parameters:
    ----------
    pop_array : numpy.ndarray
        The averaged population array to be split.
    stim_mode : int
        The stimulation mode used for breaking down the color information.
    on_len : int
        The length of the ON period.
    off_len : int
        The length of the OFF period.
    axis : int, optional
        The axis along which the splitting should be performed. Default is axis 2.

    Returns:
    -------
    numpy.ndarray
        An array containing color-segmented subarrays, further divided into ON and OFF segments based on the specified lengths.
    
    Raises:
    ------
    ValueError
        If the length of the ON and OFF segments cannot be divided evenly along the specified axis, dtype is set to 'object'.
    """
    # First break down by colour 
    colour_pop = pop_split_by_colour(pop_array, 4)
    # Then break that down by ON/OFF period
    segment_length = colour_pop.shape[axis]
    stim_total_for_ratio = on_len + off_len
    on_off_proportion = on_len / (stim_total_for_ratio)
    try:
        return np.array(np.split(colour_pop, [int(segment_length * on_off_proportion)], axis = axis))
    except ValueError:
        return np.array(np.split(colour_pop, [int(segment_length * on_off_proportion)], axis = axis), dtype = "object")

def pop_on_off_colour_index(pop_array, stim_mode, on_dur, off_dur, on_crop_first_last = (0, 0), off_crop_first_last = (0, 0), metric = np.trapz):
    """
    Calculate the ON-OFF color index for a population of cells based on a given stimulus mode and duration parameters.

    Parameters:
    - pop_array (numpy.ndarray): A 4D array containing data for each cell and color channel.
    - stim_mode (str): The stimulus mode for which the ON-OFF index is calculated.
    - on_dur (int): Duration of the ON stimulus period in time units.
    - off_dur (int): Duration of the OFF stimulus period in time units.
    - on_crop_first_last (tuple, optional): A tuple specifying cropping factors for the ON segment. Default is (0, 0).
    - off_crop_first_last (tuple, optional): A tuple specifying cropping factors for the OFF segment. Default is (0, 0).
    - metric (function, optional): A function for computing the metric along the cell axis. Default is np.trapz.

    Returns:
    - pop_colour_on_off_index (numpy.ndarray): An array containing the ON-OFF color index for each cell, segmented by color.

    This function calculates the ON-OFF color index for each cell in a population based on the given stimulus mode, ON and OFF durations, and optional cropping factors. The ON and OFF indices are computed separately for each cell and color channel, resulting in a matrix of ON and OFF indices for each cell by color.

    If cropping factors are provided, the function masks the data to exclude specific time segments before computing the index. The metric for each cell is calculated along the cell axis using the specified metric function, such as np.trapz.

    The ON and OFF metrics are then normalized to the range [0, 1] using utilities.min_max_norm. Finally, the ON-OFF color index is calculated using the on_off_index function, which combines the ON and OFF metrics for each cell and color.

    Example:
    pop_array = ...  # 4D data array
    stim_mode = "Visual"
    on_dur = 5
    off_dur = 3
    on_crop_first_last = (1, 1)
    off_crop_first_last = (2, 2)
    result = pop_on_off_colour_index(pop_array, stim_mode, on_dur, off_dur, on_crop_first_last, off_crop_first_last)
    """
    # Prepare array for matrix operations w/respect to area under curve by ON/OFF segment PER colour 
    prepped_arr = pop_on_off_colour_split(pop_array, stim_mode, on_dur, off_dur)
    if on_crop_first_last != (0, 0) or off_crop_first_last != (0, 0):
        # Mask to deal with cropping factors 
        prepped_arr = np.ma.array(prepped_arr, mask = np.ones(prepped_arr.shape))
        # Determine length
        length = test_arr.shape[-1]
        # Apply cropping factor ot mask 
        prepped_arr.mask[0, :, :, on_crop_first_last[0]:length-on_crop_first_last[1]] = 0
        prepped_arr.mask[1, :, :, off_crop_first_last[0]:length-off_crop_first_last[1]] = 0
    # Compute metric along cell axis 
    metric_by_cell = metric(prepped_arr, axis = 3)
    # Split ON and OFF metrics for comparing (residual axis is colour) --> meaning each cell will have an ON-OFF index by colour (seems useful)
    pop_on_metric = utilities.min_max_norm(metric_by_cell[0], 0, 1)
    pop_off_metric = utilities.min_max_norm(metric_by_cell[1], 0, 1)
    # Apply index via matrix operations (mathematically built into function)
    pop_colour_on_off_index = np.array(on_off_index(pop_on_metric, pop_off_metric))
    return pop_colour_on_off_index

def pop_estimate_peaktime(pop_array, smoothing_window):
    # First, smooth along time (you may use convolution for this)
    pop_processed = np.apply_along_axis(np.convolve, 0, example_data, np.ones(smoothing_window))
    # Get the min and max locs
    on_peak = np.argmax(pop_split_onoff_colour, axis = -1)[0, :, 3] # This is correct
    off_peak = np.argmax(pop_split_onoff_colour, axis = -1)[1, :, 3] # This is correct
    on_valley = np.argmin(pop_split_onoff_colour, axis = -1)[0, :, 3]
    off_valley = np.argmin(pop_split_onoff_colour, axis = -1)[1, :, 3]
    # Determine which in pair is the bigger value (peak or valley, for off or on)

from cycler import cycler
def plot_cell_on_off_colour_timecourses(pop_array, cell_id, xlim = (0, 0)):
    plt.gca().set_prop_cycle(cycler('color', ['red', 'green', 'blue', 'violet']))
    pop_split_colour = pop_split_by_colour(pop_array, 4)
    plt.plot(pop_split_colour[:, cell_id].T)
    plt.axvspan(0, pop_split_colour.shape[-1]/2, alpha = 0.25)
    plt.xlabel("Time (ms)")
    plt.ylabel("Ca2+")
    if xlim != (0, 0):
        plt.xlim(xlim)

def plot_on_off_index_tuning(pop_array, cell_id = None, give_return = False):
    on_off_index_array = pop_on_off_colour_index(pop_array, 4, 2, 2)
    if cell_id == None:
        plt.plot(on_off_index_array)
    else:
        plt.plot(on_off_index_array[:, cell_id])
    plt.xticks([0, 1, 2 ,3], ["588", "472", "420", "365"])
    plt.ylabel("ON-OFF")
    plt.xlabel("Wavelength (nm)")
    plt.gca().invert_xaxis()
    if give_return == True:
        return on_off_index_array