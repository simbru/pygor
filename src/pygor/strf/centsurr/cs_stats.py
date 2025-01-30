import numpy as np
from scipy import signal
from collections import defaultdict
import pygor.np_ext
import pygor.strf.spatial
import pandas as pd
from joblib import Parallel, delayed

def sd_index(csn_times):
    """
    Calculate the SD index.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    float
        SD index value.
    """
    sd_s = np.std(csn_times[1])
    sd_c = np.std(csn_times[0])
    value = (np.abs(sd_c) - np.abs(sd_s)) / (np.abs(sd_s) + np.abs(sd_c))
    if np.ma.is_masked(value):
        return np.nan
    else:
        return value 

def cs_ratio(csn_times):
    """
    Calculate the CS ratio.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    float
        CS ratio value.
    """
    absmax_c = np.abs(pygor.np_ext.maxabs(csn_times[0]))
    absmax_s = np.abs(pygor.np_ext.maxabs(csn_times[1]))
    with np.errstate(divide='ignore'):
        value = (absmax_c / absmax_s)
    if np.ma.is_masked(value):
        return np.nan
    else:    
        return value

def cs_contrast(prediction_times):
    """
    Calculate the CS contrast.

    Parameters
    ----------
    prediction_times : array-like
        Array containing center and surround times.

    Returns
    -------
    float
        CS contrast value.
    """
    centre = prediction_times[0]
    surround = prediction_times[1]
    centre_ampl = pygor.np_ext.absmax(centre)
    surround_ampl = pygor.np_ext.absmax(surround)
    if np.all(surround == 0):
        return np.nan 
    with np.errstate(divide='ignore', invalid='ignore'):
        value = (centre_ampl - surround_ampl) / (centre_ampl + surround_ampl)
    if np.ma.is_masked(value):
        return np.nan
    else:    
        return value

def peak_amplitude(csn_times):
    """
    Calculate the peak amplitude.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Peak amplitude of center and surround.
    """
    peak_c = np.max(np.abs(csn_times[0]))
    peak_s = np.max(np.abs(csn_times[1]))
    return peak_c, peak_s

def time_to_peak(csn_times):
    """
    Calculate the time to peak.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Time to peak of center and surround.
    """
    time_peak_c = np.argmax(np.abs(csn_times[0]))
    time_peak_s = np.argmax(np.abs(csn_times[1]))
    return time_peak_c, time_peak_s

def auc(csn_times):
    """
    Calculate the area under the curve (AUC).

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        AUC of center and surround.
    """
    auc_c = np.trapz(np.abs(csn_times[0]))
    auc_s = np.trapz(np.abs(csn_times[1]))
    return auc_c, auc_s

def polarity_index(csn_times):
    """
    Calculate the polarity index.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Polarity index of center and surround.
    """
    def calculate_polarity(signal):
        pos_area = np.abs(np.trapz(signal[signal > 0]))
        neg_area = np.abs(np.trapz(signal[signal < 0]))
        return (pos_area - neg_area) / (pos_area + neg_area)
    polarity_c = calculate_polarity(csn_times[0])
    polarity_s = calculate_polarity(csn_times[1])
    return polarity_c, polarity_s

def correlation_coefficient(csn_times):
    """
    Calculate the correlation coefficient.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    float
        Correlation coefficient between center and surround.
    """
    correlation = np.corrcoef(csn_times[0], csn_times[1])[0, 1]
    return correlation

def snr(csn_times):
    """
    Calculate the signal-to-noise ratio (SNR).

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    float
        SNR value.
    """
    signal_power = np.mean(np.square(csn_times[0]))
    noise_power = np.mean(np.square(csn_times[1]))
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value


def peak_latency(csn_times):
    """
    Calculate the peak latency.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Peak latency of center and surround.
    """
    latency_c = np.argmax(np.abs(csn_times[0]))
    latency_s = np.argmax(np.abs(csn_times[1]))
    return latency_c, latency_s

def half_width(csn_times):
    """
    Calculate the half-width.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Half-width of center and surround.
    """
    def calculate_half_width(signal):
        peak_index = np.argmax(np.abs(signal))
        half_max = np.abs(signal[peak_index]) / 2
        left_index = np.where(np.abs(signal[:peak_index]) <= half_max)[0][-1]
        right_index = np.where(np.abs(signal[peak_index:]) <= half_max)[0][0] + peak_index
        return right_index - left_index
    half_width_c = calculate_half_width(csn_times[0])
    half_width_s = calculate_half_width(csn_times[1])
    return half_width_c, half_width_s

def time_to_half_peak(csn_times):
    """
    Calculate the time to half-peak.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Time to half-peak of center and surround.
    """
    def calculate_time_to_half_peak(signal):
        peak_index = np.argmax(np.abs(signal))
        half_max = np.abs(signal[peak_index]) / 2
        half_peak_index = np.where(np.abs(signal) >= half_max)[0][0]
        return half_peak_index
    time_to_half_peak_c = calculate_time_to_half_peak(csn_times[0])
    time_to_half_peak_s = calculate_time_to_half_peak(csn_times[1])
    return time_to_half_peak_c, time_to_half_peak_s

def full_width_half_max(csn_times):
    """
    Calculate the full width at half maximum (FWHM).

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        FWHM of center and surround.
    """
    def calculate_fwhm(signal):
        peak_index = np.argmax(np.abs(signal))
        half_max = np.abs(signal[peak_index]) / 2
        left_index = np.where(np.abs(signal[:peak_index]) <= half_max)[0][-1]
        right_index = np.where(np.abs(signal[peak_index:]) <= half_max)[0][0] + peak_index
        return right_index - left_index
    fwhm_c = calculate_fwhm(csn_times[0])
    fwhm_s = calculate_fwhm(csn_times[1])
    return fwhm_c, fwhm_s

def rise_time(csn_times):
    """
    Calculate the rise time.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Rise time of center and surround.
    """
    def calculate_rise_time(signal):
        peak_index = np.argmax(np.abs(signal))
        ten_percent_index = np.where(np.abs(signal) >= 0.1 * np.abs(signal[peak_index]))[0][0]
        ninety_percent_index = np.where(np.abs(signal) >= 0.9 * np.abs(signal[peak_index]))[0][0]
        return ninety_percent_index - ten_percent_index
    rise_time_c = calculate_rise_time(csn_times[0])
    rise_time_s = calculate_rise_time(csn_times[1])
    return rise_time_c, rise_time_s

def decay_time(csn_times):
    """
    Calculate the decay time.

    Parameters
    ----------
    csn_times : array-like
        Array containing center and surround times.

    Returns
    -------
    tuple
        Decay time of center and surround.
    """
    def calculate_decay_time(signal):
        peak_index = np.argmax(np.abs(signal))
        ninety_percent_index = np.where(np.abs(signal) >= 0.9 * np.abs(signal[peak_index]))[0][0]
        ten_percent_index = np.where(np.abs(signal) >= 0.1 * np.abs(signal[peak_index]))[0][-1]
        return ten_percent_index - ninety_percent_index
    decay_time_c = calculate_decay_time(csn_times[0])
    decay_time_s = calculate_decay_time(csn_times[1])
    return decay_time_c, decay_time_s

def gen_stats(strfs_obj, colour_list = ["R", "G", "B", "UV"], **kwargs):
    """
    Generate statistics on a list of STRF arrays.

    Parameters
    ----------
    strfs_arrs_list : list
        List of STRF arrays.

    Returns
    -------
    dict
        Dictionary containing various statistics.
    """
    strfs_arrs_list = strfs_obj.strfs_no_border
    rois_index = np.repeat(np.arange(strfs_obj.num_rois), len(colour_list))
    if rois_index.size == 0:
        return None
    ipl_depths = np.repeat(strfs_obj.ipl_depths, len(colour_list))
    output = defaultdict(list)
    for n, i in enumerate(strfs_arrs_list):
        prediction_map, prediction_times = pygor.strf.centsurr.run(i, **kwargs)
        prediction_times = prediction_times.data
        with np.errstate(divide='ignore', invalid='ignore'):
            # Base stats
            output["max_c"].append(np.max(prediction_times[0]))
            output["max_s"].append(np.max(prediction_times[1]))
            output["max_n"].append(np.max(prediction_times[2]))
            output["min_c"].append(np.min(prediction_times[0]))
            output["min_s"].append(np.min(prediction_times[1]))
            output["min_n"].append(np.min(prediction_times[2]))
            output["absmax_c"].append(pygor.np_ext.absmax(prediction_times[0]))
            output["absmax_s"].append(pygor.np_ext.absmax(prediction_times[1]))
            output["absmax_n"].append(pygor.np_ext.absmax(prediction_times[2]))
            output["maxabs_c"].append(pygor.np_ext.maxabs(prediction_times[0]))
            output["maxabs_s"].append(pygor.np_ext.maxabs(prediction_times[1]))
            output["maxabs_n"].append(pygor.np_ext.maxabs(prediction_times[2]))
            output["sum_c"].append(np.sum(prediction_times[0]))
            output["sum_s"].append(np.sum(prediction_times[1]))
            output["sum_n"].append(np.sum(prediction_times[2]))
            output["var_c"].append(np.var(prediction_times[0]))
            output["var_s"].append(np.var(prediction_times[1]))
            output["var_n"].append(np.var(prediction_times[2]))
            output["sd_c"].append(np.std(prediction_times[0]))
            output["sd_s"].append(np.std(prediction_times[1]))
            output["sd_n"].append(np.std(prediction_times[2]))
            output["bidx_c"].append(pygor.strf.temporal.biphasic_index(prediction_times[0]))
            output["bidx_s"].append(pygor.strf.temporal.biphasic_index(prediction_times[1]))
            output["bidx_n"].append(pygor.strf.temporal.biphasic_index(prediction_times[2]))
            output["cent_c"].append(pygor.strf.temporal.only_centroid(prediction_times[0]))
            output["cent_s"].append(pygor.strf.temporal.only_centroid(prediction_times[1]))
            output["cent_n"].append(pygor.strf.temporal.only_centroid(prediction_times[2]))
            # Calculations
            output["var_index"].append(sd_index(prediction_times))
            output["cs_ratio"].append(cs_ratio(prediction_times))
            output["cs_contrast"].append(cs_contrast(prediction_times))
            peak_c, peak_s = peak_amplitude(prediction_times)
            output["peak_amplitude_c"].append(peak_c)
            output["peak_amplitude_s"].append(peak_s)
            time_peak_c, time_peak_s = time_to_peak(prediction_times)
            output["time_to_peak_c"].append(time_peak_c)
            output["time_to_peak_s"].append(time_peak_s)
            auc_c, auc_s = auc(prediction_times)
            output["auc_c"].append(auc_c)
            output["auc_s"].append(auc_s)
            polarity_c, polarity_s = polarity_index(prediction_times)
            output["polarity_index_c"].append(polarity_c)
            output["polarity_index_s"].append(polarity_s)
            output["correlation_coefficient"].append(correlation_coefficient(prediction_times))
            output["snr"].append(snr(prediction_times))
            # Spatial
            output["pix_n_c"] = np.sum(prediction_map == 0)
            output["pix_n_s"] = np.sum(prediction_map == 1)
            output["pix_n_n"] = np.sum(prediction_map == 2)
            # Others 
            output["colour"].append(colour_list[n % len(colour_list)])
            if np.abs(np.max(prediction_times[0])) > np.abs(np.min(prediction_times[0])):
                pol = "ON"
            elif np.abs(np.max(prediction_times[0])) < np.abs(np.min(prediction_times[0])):
                pol = "OFF"
            else:
                pol = "other"
            # maxC = np.max(prediction_times[0])
            # maxS = np.max(prediction_times[1])
            # minC = np.min(prediction_times[0])
            # minS = np.min(prediction_times[1])
            absmaxC = pygor.np_ext.maxabs(prediction_times[0])
            absmaxS = pygor.np_ext.maxabs(prediction_times[1])
            threshold = 2.5
            if (np.abs(absmaxC) > threshold) and (np.abs(absmaxS) > threshold):
                pol = "CS ON"
            if (np.abs(absmaxC) > threshold and np.abs(absmaxS) > threshold) and (absmaxC < 0):
                pol = "CS OFF"
            output["polarity"].append(pol)
            output["IPL"].append(ipl_depths[n])
            output["roi"].append(rois_index[n])
            output["rel_index"].append(n)
            output["filename"].append(strfs_obj.filename)
            output["name"].append(strfs_obj.filename.stem)

            

    # Check length of stats before returning dict
    return output

def run_stats(files_list : list, para = True) -> pd.DataFrame:
    dicts_list = []
    df_list = []
    if para is True:
        def loop(file):
            strfs = pygor.filehandling.load(file, as_class=pygor.load.STRF, bs_bool = False)
            #TODO: Add ability to augment segmentation and extraction params (removed due to broken behaviour)
            tempdict = pygor.strf.centsurr.cs_stats.gen_stats(strfs)
            return pd.DataFrame(tempdict)
        df_list = Parallel(n_jobs=-1)(delayed(loop)(file) for file in files_list)
    else:
        for i in files_list:
            strfs = pygor.filehandling.load(i, as_class=pygor.load.STRF, bs_bool = False)
            tempdict = pygor.strf.centsurr.cs_stats.gen_stats(strfs)

            tempdf = pd.DataFrame(tempdict)
            dicts_list.append(tempdict)
            df_list.append(tempdf)
    df = pd.concat(df_list)
    return df.reset_index()

# def run_stats(files_list : list, segmentation_params = {}, extract_params = {}) -> pd.DataFrame:
#     dicts_list = []
#     df_list = []
#     return df.reset_index()