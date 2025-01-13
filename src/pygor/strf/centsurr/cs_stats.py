import numpy as np
from scipy import signal
from collections import defaultdict
import pygor.np_ext
import pygor.strf.spatial

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
    absmax_c = pygor.np_ext.maxabs(csn_times[0])
    absmax_s = pygor.np_ext.maxabs(csn_times[1])
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
        pos_area = np.trapz(signal[signal > 0])
        neg_area = np.trapz(signal[signal < 0])
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


def run_stats(strfs_arrs_list):
    """
    Run statistics on a list of STRF arrays.

    Parameters
    ----------
    strfs_arrs_list : list
        List of STRF arrays.

    Returns
    -------
    dict
        Dictionary containing various statistics.
    """
    output = defaultdict(list)
    for i in strfs_arrs_list:
        prediction_map, prediction_times = pygor.strf.centsurr.run(i)
        prediction_times = prediction_times.data
        with np.errstate(divide='ignore', invalid='ignore'):
            # Base stats
            output["max_c"].append(np.max(prediction_times[0]))
            output["max_s"].append(np.max(prediction_times[1]))
            output["max_n"].append(np.max(prediction_times[2]))
            output["min_c"].append(np.min(prediction_times[0]))
            output["min_s"].append(np.min(prediction_times[1]))
            output["min_n"].append(np.min(prediction_times[2]))
            output["absmax_c"].append(pygor.np_ext.maxabs(prediction_times[0]))
            output["absmax_s"].append(pygor.np_ext.maxabs(prediction_times[1]))
            output["absmax_n"].append(pygor.np_ext.maxabs(prediction_times[2]))
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

    # Check length of stats before returning dict
    return output