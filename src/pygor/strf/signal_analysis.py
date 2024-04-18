import numpy as np#
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import warnings
import scipy
import seaborn as sns
import matplotlib
from joblib import Parallel, delayed
import pygor.data_helpers 
import pygor.utilities
import math
sns.set_context("notebook")

def stimphase_stat(trace, trigger_times, phase_trignum, linespeed, stat = np.std, plot = False, **kwargs):
    """
    Calculate statistics on a signal trace for each stimulus phase.

    Parameters
    ----------
    trace : array
        Signal trace.
    trigger_times : array
        Array of trigger times.
    phase_trignum : int
        Number of triggers per phase.
    linespeed : float
        Speed of the line.
    stat : function
        Function to calculate statistics on signal trace.
    plot : bool
        Whether to plot the signal trace.

    Returns
    -------
    stats : array
        Array of statistics for each stimulus phase.

    Notes:
    ------
    We operate in linescan-speed as our time-domain, but everything is scaled by 'linespeed' thus yielding ms (i think...)
    - trace: 1D numpy array, 2P data 
    - trigger_times: 1D array where indeces indicate when nth trigger occured
    - phase_trignum: how many triggers per stimulus phase
    """
    if 'pause_dur' in kwargs:
        inter_phase_dur = kwargs['pause_dur']
    else:
        inter_phase_dur = 0
    if 'avgd_over' in kwargs:
        repeats = kwargs["avgd_over"]
    else:
        repeats = 1
    # Get the nth triggers for sectioning out indeces in signal trace
    phase_nth_trigs = trigger_times[::phase_trignum].astype(int)
    stim_dur = np.average(np.diff(phase_nth_trigs, axis = 0)) - inter_phase_dur
    # Get stats within those durations 
    trial_split = np.split(trace, phase_nth_trigs[1:])
    # Caøculate statistic over given intervals
    stats = np.empty(len(trial_split))
    counter_stats = np.empty(len(trial_split))
    if inter_phase_dur == 0:
        for n, i in enumerate(trial_split):
            stats[n] = stat(i)
    else: 
        for n, i in enumerate(trial_split):
            stats[n] = stat(i[:-inter_phase_dur])
            if "percentdiff_threshold" in kwargs:
                inter_phase = i[len(i)-inter_phase_dur:]
                counter_stats[n] = stat(inter_phase)
                if stat == np.std and repeats > 1: # stupid solution but cannot be asked
                    counter_stats[n] = np.array(counter_stats[n]) * np.sqrt(repeats)
                if np.sum(np.abs(stats[n])) > np.sum(np.abs(counter_stats[n])):
                    percentage_difference_arr = np.abs( 1 - np.divide(counter_stats, stats))
                else:
                    percentage_difference_arr = np.abs( 1 - np.divide(stats, counter_stats))
                if np.all(percentage_difference_arr < kwargs["percentdiff_threshold"]):
                    stats = np.repeat(np.nan, len(stats))
    if stat == np.std and repeats > 1:
        stats = np.array(stats) * np.sqrt(repeats)
        # In this special case, some mathemagical thing occurs where due to taking 
        # the STD of a signal thats expressed in SD and is averaged repeatedly, the
        # SD of that averaged signal will be scaled by 1/sqrt(n_repeats)            
    else:
        stats = np.array(stats)
    if plot == True:
        # Plotting
        fig, ax = plt.subplots(figsize = (10, 5))
        # First axis 
        ax.plot(trace)
        for i in phase_nth_trigs: 
            ax.axvline(i, c='r')
            ax.axvspan(i, i + stim_dur, alpha = 0.2, color = 'gray')
        ax.set_xlabel(f"ms")
        # Second axis
        secax = ax.twiny()
        secax.plot(stats, 'o-', c='orange')
        secax.set_xticks([0,1,2,3,4,5])
        secax.set_xticklabels([1.18, 2.37, 4.74, 9.48, 18.96, 37.93])
        secax.set_xlabel("Box size(deg visang)")
        secax.set_xlim(np.array([-.75, 5.9]))
        secax.grid(False)
    return stats

def stimphase_stat_diff(trace, trigger_times, phase_trignum, linespeed, stat = np.std, return_both = False, plot = False, **kwargs):
    """
    Calculate difference in statsitic between baseline period and response periods.

    Parameters
    ----------
    trace : array
        Signal trace.
    trigger_times : array
        Array of trigger times.
    phase_trignum : int
        Number of triggers per phase.
    linespeed : float
        Speed of the line.
    stat : function
        Function to calculate statistics on signal trace.
    plot : bool
        Whether to plot the signal trace.

    Returns
    -------
    stats : array
        Array of statistics for each stimulus phase.

    Notes:
    ------
    We operate in linescan-speed as our time-domain, but everything is scaled by 'linespeed' thus yielding ms (i think...)
    - trace: 1D numpy array, 2P data 
    - trigger_times: 1D array where indeces indicate when nth trigger occured
    - phase_trignum: how many triggers per stimulus phase
    """
    if 'pause_dur' in kwargs:
        inter_phase_dur = kwargs['pause_dur']
    else:
        inter_phase_dur = 0
    if 'avgd_over' in kwargs:
        repeats = kwargs["avgd_over"]
    else:
        repeats = 1
    # Get the nth triggers for sectioning out indeces in signal trace
    phase_nth_trigs = trigger_times[::phase_trignum].astype(int)
    stim_dur = np.average(np.diff(phase_nth_trigs, axis = 0)) - inter_phase_dur
    # Get stats within those durations 
    trial_split = np.split(trace, phase_nth_trigs[1:])
    # Caøculate statistic over given intervals
    stats_response = np.empty(len(trial_split))
    stats_baseline = np.empty(len(trial_split))
    counter_stats = np.empty(len(trial_split))
    if inter_phase_dur == 0:
        raise AttributeError("No baseline duration as defined by inter_phase_dur")
    else: 
        for n, i in enumerate(trial_split):
            stats_response[n] = stat(i[:-inter_phase_dur])
            stats_baseline[n] = stat(i[-inter_phase_dur:])
    if stat == np.std and repeats > 1:
        stats_response = np.array(stats_response) * np.sqrt(repeats)
        stats_baseline = np.array(stats_baseline) * np.sqrt(repeats)
        # In this special case, some mathemagical thing occurs where due to taking 
        # the STD of a signal thats expressed in SD and is averaged repeatedly, the
        # SD of that averaged signal will be scaled by 1/sqrt(n_repeats)            
    else:
        stats_response = np.array(stats_response)
        stats_baseline = np.array(stats_baseline)
    difference = stats_response - stats_baseline
    if plot == True:
        # Plotting
        fig, ax = plt.subplots(figsize = (10, 5))
        # First axis 
        ax.plot(trace)
        for i in phase_nth_trigs: 
            ax.axvline(i, c='r')
            ax.axvspan(i, i + stim_dur, alpha = 0.2, color = 'gray')
        ax.set_xlabel(f"ms")
        # Second axis
        secax = ax.twiny()
        secax.plot(difference, 'o-', c='orange')
        secax.set_xticks([0,1,2,3,4,5])
        secax.set_xticklabels([1.18, 2.37, 4.74, 9.48, 18.96, 37.93])
        secax.set_xlabel("Box size(deg visang)")
        secax.set_xlim(np.array([-.75, 5.9]))
        secax.grid(False)
    if return_both == True:
        return stats_response, stats_baseline
    else:
        return difference

def stimphase_stat_multi(traces_, trigger_times, phase_trignum, linespeed, stat = np.var, axis = 0, function = stimphase_stat, **kwargs):
    """
    Calculate statistics of stimulus-triggered averages of traces along a given axis.

    Parameters
    ----------
    traces_ : array_like
        The traces to be analyzed.
    trigger_times : array_like
        The times at which the triggers occurred.
    phase_trignum : int
        The number of phases in the stimulus cycle.
    linespeed : float
        The speed of the linescan in pixels per second.
    stat : callable
        The function used to calculate statistics (default is np.var).
    axis : int
        The axis along which to calculate statistics (default is 0).
    **kwargs : dict
        Additional keyword arguments passed to stat.

    Returns
    -------
    array_like
        An array containing the calculated statistics.

    """
    return np.apply_along_axis(function, axis, traces_, trigger_times, phase_trignum, linespeed, stat, **kwargs).T

def init_timing_vars(file_path):
    """
    Get the file and calculate the timing variables.

    Parameters
    ----------
    file_path : str
        The path of the file.

    Returns
    -------
    tuple
        A tuple containing:
        - linespeed : float
            The speed of the lines in seconds per pixel.
        - avg_trig_times_ms : numpy.ndarray
            The average trigger times in milliseconds for the averaged trace.
        - inter_phase_durMs : int
            The duration of the inter-phase in milliseconds.
    
    Notes
    -----
    This function reads a file from a given path and extracts the variables from it. It then calculates the average trigger 
    times and determines when in the loop the trigger occurs on average. It also accounts for the period of no stimulus.
    """
    # Get the file 
    file = pygor.data_helpers.load_from_hdf5(file_path)
    # Import the variables
    triggers_time = file.Triggertimes
    linespeed = file.OS_Parameters[56]
    traces = file.Averages0
    # Determine, on average, when in the loop the trigger occurs
    avg_trig_times = np.average(np.split(triggers_time, 3), axis = 0)
    avg_trig_times = avg_trig_times - avg_trig_times[0] 
    avg_trig_times_ms = np.round(avg_trig_times / linespeed, 0) # round so we get an integer index 
    # Account for period of no stimulus 
    inter_phase_durS = 2
    inter_phase_durMs = np.round(inter_phase_durS / linespeed, 0).astype(int)
    return linespeed, avg_trig_times_ms, inter_phase_durMs

def stim_phase_df(file_path_list, conditions_label_list, phase_trignum, repeats, stat = np.std, **kwargs):
    """
    Calculates statistics on the standard deviation of fluorescence signals during a stimulus phase across multiple files and conditions.

    Parameters
    ----------
    file_path_list : List[str]
        A list of file paths to load data from.
    conditions_label_list : List
        A list of labels for each condition in the experiment.
    phase_trignum : int
        The trigger number associated with the stimulus phase.
    repeats : int
        The number of repeats of each condition.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        - filename: the name of the file the data came from.
        - colour: the colour associated with the data (determined from the filename).
        - roi: the region of interest (ROI) associated with the data.
        - condition: the condition label associated with the data.
        - stat: the standard deviation of the fluorescence signal during the stimulus phase.

    Notes
    -----
    This function relies on other functions from the `signal_analysis` and `pygor.data_helpers` modules.
    """
    stat_list = [] # DataFrames will be temporarily stored here and concatenated later
    for file in file_path_list:
        linespeed, avg_trig_times_ms, inter_phase_durMs = init_timing_vars(file)
        # Create an empty dictionary
        dict = {}
        # Get the file contents
        loaded_file = pygor.data_helpers.load_from_hdf5(file)
        dict["filename"] = pathlib.Path(loaded_file.filename).name
        dict["colour"] = pygor.data_helpers.label_from_str(loaded_file.filename, ["R", "G", "B", "UV"])
        # Load up data of interest
        traces = loaded_file.Averages0
        if 'std_threshold' in kwargs:
            # Figure out which traces are under threshold 
            scores = np.std(traces, axis = 0)
            sub_threshold_indeces = np.where(scores < kwargs["std_threshold"])[0]
            if any(sub_threshold_indeces) is True: # any as in if any is below threshold 
                if len(sub_threshold_indeces) == traces.shape[1]: # if all under threshold
                    warnings.warn(f"All traces under threshold for file: {file}")
                    continue 
                traces = np.delete(traces, sub_threshold_indeces, axis = 1)
        #if 'pause_threshold' in kwargs:

        num_rois = len(traces.T)
        dict["roi"] = np.repeat(np.arange(num_rois), len(conditions_label_list))
        # Run the test
        result = stimphase_stat_multi(traces, avg_trig_times_ms, phase_trignum, linespeed, 
            pause_dur = inter_phase_durMs, stat = stat, avgd_over = repeats, function = stimphase_stat, **kwargs)
        result_difference = stimphase_stat_multi(traces, avg_trig_times_ms, phase_trignum, linespeed, 
            pause_dur = inter_phase_durMs, stat = stat, avgd_over = repeats, function = stimphase_stat_diff, **kwargs)
        dict["condition"] = np.tile(conditions_label_list, num_rois)
        dict["stat"] = result.flatten()
        dict["stat_diff"] = result_difference.flatten()
        dict["stat_diff_abs"] = np.abs(result_difference.flatten())
        # For each condition, add those results to an dictionary index (so each index is a 1D array of all the stats for that column)
        stat_list.append(pd.DataFrame(dict))
    df_res = pd.concat(stat_list, ignore_index=True)
    return df_res


def category_labels(spectral_tuning_functions, thresh = 0, return_arrs = False, minmax_scale = False, plot = False, **kwargs):
    spectral_tuning_functions = np.flip(spectral_tuning_functions, axis = 1)
    # Threshold and get all R, G, B, and UV values from the tuning functions 
    if minmax_scale == True:
        ax_abs_per_row = np.max(np.abs(spectral_tuning_functions), axis=1, keepdims=True)
        spectral_tuning_functions_scaled = spectral_tuning_functions / ax_abs_per_row
        spectral_tuning_functions_scaled_thresh = np.where(np.abs(spectral_tuning_functions_scaled) > thresh, spectral_tuning_functions_scaled, 0)
        all_u, all_b, all_g, all_r = spectral_tuning_functions_scaled_thresh.T
        spectral_tuning_functions = spectral_tuning_functions_scaled_thresh
        # spectral_tuning_functions = np.apply_along_axis(pygor.utilities.min_max_norm, 1, spectral_tuning_functions, -1, 1)
        # all_u, all_b, all_g, all_r = [i for i in np.where(np.abs(spectral_tuning_functions.T) > thresh, spectral_tuning_functions.T, 0)]
    else:
        all_u, all_b, all_g, all_r = [i for i in np.where(np.abs(spectral_tuning_functions.T) > thresh, spectral_tuning_functions.T, 0)]
    # Generate label bool arrays 
    _short_bias = (np.abs(all_u) > np.abs(all_b)) & (np.abs(all_r) < np.abs(all_u)) & ((np.abs(all_b) > np.abs(all_g)) | (np.abs(all_u) > np.abs(all_b) + np.abs(all_g) + np.abs(all_r)))
    # _long_bias = (np.abs(all_u) < np.abs(all_b)) & (np.abs(all_b) < np.abs(all_g)) & (np.abs(all_g) < np.abs(all_r))
    _long_bias = ((np.abs(all_b) < np.abs(all_g)) & (np.abs(all_g) < np.abs(all_r)) & (np.abs(all_u) < np.abs(all_r)) & (np.abs(all_u) < np.abs(all_g))) | (np.abs(all_r) > (np.abs(all_g)+np.abs(all_b)+np.abs(all_u)))
    _mid_bias = (np.abs(all_b) + np.abs(all_g) > (np.abs(all_u) + np.abs(all_r))) & ((np.abs(all_r) < np.abs(all_g)) & (np.abs(all_b) > np.abs(all_u))) & ((np.sign(all_b) == np.sign(all_g)) & (np.sign(all_u) == np.sign(all_r)))
    _b_bias = ((np.abs(all_b) > np.abs(all_u)) & (np.abs(all_b) > np.abs(all_g)) & (np.abs(all_g) + np.abs(all_r) < np.abs(all_b)))
    _g_bias = ((np.abs(all_g) > np.abs(all_b)) & (np.abs(all_g) > np.abs(all_r)) & (np.abs(all_u) + np.abs(all_b) < np.abs(all_g))) 
    # Broad: apply if all polarities are the same and nothing else applies 
    _broad = np.abs(all_r > 0) & np.abs(all_g > 0) & np.abs(all_b > 0) & np.abs(all_u > 0)
    # For now, put broad togetehr with mid-biased
    _mid_bias = _broad | _mid_bias
    # Tidy up broad (will probably just be empty)
    _broad = _broad * ((_broad != _short_bias) & (_broad != _mid_bias) & (_broad != _long_bias)  & (_broad != _g_bias)  & (_broad != _b_bias))
    _non_response = np.abs(all_r == 0) & np.abs(all_g == 0) & np.abs(all_b == 0) & np.abs(all_u == 0)
    # Generate label bool arrays for opponencies     
    _short_opp = (((((all_r + all_g > 0) & (all_u  < 0)) & (all_b > 0)) | (((all_r + all_g < 0) & (all_u > 0)) & (all_b < 0))))
    _long_opp = (((all_r > 0) & (all_g < 0)) | ((all_r < 0) & (all_g > 0)))# (R>0 && G<0) || (R<0 && G>0)
    _mid_opp = (((all_g > 0) & (all_b < 0)) | ((all_g < 0) & (all_b > 0))) # (R>0 && B<0) || R<0 && B>0)
    
    # Give priority to opponency versus the other catagories 
    _short_bias = _short_bias * (_short_bias * (_short_bias != _short_opp)) & (_short_bias * (_short_bias != _mid_opp)) & (_short_bias * (_short_bias != _long_opp))
    _mid_bias = _mid_bias * (_mid_bias * (_mid_bias != _short_opp)) & (_mid_bias * (_mid_bias != _mid_opp)) & (_mid_bias * (_mid_bias != _long_opp))
    _long_bias = _long_bias * (_long_bias * (_long_bias != _short_opp)) & (_long_bias * (_long_bias != _mid_opp)) & (_long_bias * (_long_bias != _long_opp))
    _g_bias = _g_bias * (_g_bias * (_g_bias != _short_opp)) & (_g_bias * (_g_bias != _mid_opp)) & (_g_bias * (_g_bias != _long_opp))
    _b_bias = _b_bias * (_b_bias * (_b_bias != _short_opp)) & (_b_bias * (_b_bias != _mid_opp)) & (_b_bias * (_b_bias != _long_opp))
    _broad = _broad *  (_broad * (_broad != _short_opp)) & (_broad * (_broad != _mid_opp)) & (_broad * (_broad != _long_opp))
    # # Mid also yields versus the other groups
    _mid_bias = _mid_bias * ((_mid_bias != _b_bias) & (_mid_bias != _g_bias) & (_mid_bias != _long_bias) & (_mid_bias != _short_bias))
    # _long_bias = (_long_bias * (_long_bias != _mid_bias))
    # _short_bias = (_short_bias * (_short_bias != _mid_bias))
    # _mid_bias = _mid_bias * ((_mid_bias * (_mid_bias != short_opp)) & (_mid_bias * (_mid_bias != mid_opp)) & (_mid_bias * (_mid_bias != long_opp))
    # & (_mid_bias * (_mid_bias != short_bias)) & (_mid_bias * (_mid_bias != long_bias)))

    # Generate labels 
    ## Array creation
    _labels_arr = np.empty(len(spectral_tuning_functions), dtype = "U10")
    # _bool_arr = np.array([_short_bias, _g_bias, _b_bias, _mid_bias, _long_bias, _broad, _short_opp, _mid_opp, _long_opp]).T
    _bool_arr = np.array([_short_bias, _g_bias, _b_bias, _mid_bias, _long_bias, _short_opp, _mid_opp, _long_opp, _non_response]).T

    ## Templates 
    short_template =   [True, False, False, False, False, False, False, False, False]
    g_template =       [False, True, False, False, False, False, False, False, False]
    b_template =       [False, False, True, False, False, False, False, False, False]    
    mid_template =     [False, False, False, True, False, False, False, False, False]
    long_template =    [False, False, False, False, True, False, False, False, False] 	#False	False	False	False	True	False	False	False	False
    #broad_template =   [False, False, False, False, False, True, False, False, False]
    short_opp_template=[False, False, False, False, False, True, False, False, False]
    mid_opp_template = [False, False, False, False, False, False, True, False, False]
    long_opp_template =[False, False, False, False, False, False, False, True, False]
    non_resp_template =[False, False, False, False, False, False, False, False, True]
    # non_responsive = []
    ## Template matching and label assigning 
    _labels_arr[np.where(np.all(_bool_arr == short_template, axis = 1))] = "short"
    _labels_arr[np.where(np.all(_bool_arr == g_template, axis = 1))] = "green"
    _labels_arr[np.where(np.all(_bool_arr == b_template, axis = 1))] = "blue"
    _labels_arr[np.where(np.all(_bool_arr == mid_template, axis = 1))] = "mid"
    _labels_arr[np.where(np.all(_bool_arr == long_template, axis = 1))] = "long"
    # _labels_arr[np.where(np.all(_bool_arr == broad_template, axis = 1))] = "broad"
    _labels_arr[np.where(np.all(_bool_arr == short_opp_template, axis = 1))] = "short_opp"
    _labels_arr[np.where(np.all(_bool_arr == mid_opp_template, axis = 1))] = "mid_opp"
    _labels_arr[np.where(np.all(_bool_arr == long_opp_template, axis = 1))] = "long_opp"
    _labels_arr[np.where(np.all(_bool_arr == non_resp_template, axis = 1))] = "non_resp"
    _labels_arr[np.where(_labels_arr == "")] = "other"

    if "print_reminder" in kwargs and kwargs["print_reminder"] == True:
        print("Label arrays in order: Long-biased, short-biased, mid-biased, v-biased, broad/achromatic, short-opponent, mid-opponent, long-opponent")
    if "print_var_names" in kwargs and kwargs["print_var_names"] == True:
        print("Vars: long_bias, short_bias, mid_bias, v_bias, broad, short_opp, mid_opp, long_opp")
    
    if plot == True:
        sns.set_theme(style="white")
        sns.set_context("talk")
        matplotlib.rcParams['svg.fonttype'] = 'none'
        fig, axs = plt.subplots(1, 10, figsize = (10*3, 1*3), sharex=False, sharey=True)
        for n, ax in enumerate(axs):
            ax.set_xticks([0, 1, 2, 3], ["375", "422","478","558"]);
            if n != 0:
                ax.set_xticks([0, 1, 2, 3], ["", "", "", ""])
            ax.axhline(0, c = "r")
            ax.grid(False)
        for n, i in enumerate(spectral_tuning_functions):
            if _labels_arr[n] == "short":
                axs[0].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "blue":
                axs[1].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "green":
                axs[2].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "mid":
                axs[3].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "long":
                axs[4].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "short_opp":
                axs[5].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "mid_opp":
                axs[6].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "long_opp":
                axs[7].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "other":
                axs[8].plot(i.T, c = "black", alpha = 0.25)
            if _labels_arr[n] == "non_resp":
                axs[9].plot(i.T, c = "black", alpha = 0.25)

        titles = ["Short-biased", "Blue", "Green", "Broad", "Long-biased", "Short-opponent", "Mid-opponent", "Long-opponent", "Other", "Non-responsive"]
        for n, i in enumerate(titles):
            axs[n].set_title(i)    
        plt.savefig(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Conferences\ERM 2023\Chromaticity graphs\tuning_functions.svg")
    
    if return_arrs == True:
        return _labels_arr, _long_bias, _short_bias, _g_bias, _b_bias, _mid_bias, _broad, _short_opp, _mid_opp, _long_opp #v_bias
    else:
        return _labels_arr

"""Averages handling___________________________________________________________________________________________________________________________"""

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

    The ON and OFF metrics are then normalized to the range [0, 1] using pygor.utilities.min_max_norm. Finally, the ON-OFF color index is calculated using the on_off_index function, which combines the ON and OFF metrics for each cell and color.

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
        length = pop_array.shape[-1]
        # Apply cropping factor ot mask 
        prepped_arr.mask[0, :, :, on_crop_first_last[0]:length-on_crop_first_last[1]] = 0
        prepped_arr.mask[1, :, :, off_crop_first_last[0]:length-off_crop_first_last[1]] = 0
    # Compute metric along cell axis 
    metric_by_cell = metric(prepped_arr, axis = 3)
    # Split ON and OFF metrics for comparing (residual axis is colour) --> meaning each cell will have an ON-OFF index by colour (seems useful)
    pop_on_metric = pygor.utilities.min_max_norm(metric_by_cell[0], 0, 1)
    pop_off_metric = pygor.utilities.min_max_norm(metric_by_cell[1], 0, 1)
    # Apply index via matrix operations (mathematically built into function)
    pop_colour_on_off_index = np.array(on_off_index(pop_on_metric, pop_off_metric))
    return pop_colour_on_off_index

def pop_estimate_peaktime(pop_array, smoothing_window = 1):
    # First, smooth along time (you may use convolution for this)
    pop_processed = np.apply_along_axis(np.convolve, 0, pop_array, np.ones(smoothing_window))
    # Split according to RGBUV and response period 
    pop_split_onoff_colour = pop_on_off_colour_split(pop_array, 4, 2, 2)
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