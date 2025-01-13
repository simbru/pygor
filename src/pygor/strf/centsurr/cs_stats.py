import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import sklearn.cluster
import skimage.morphology
from collections import defaultdict
import pygor.np_ext as np_ext
import pygor.np_ext
import pygor.strf.spatial

"""
All functions here should take as input either 
prediction_map or prediction_timse (or both), from
cs_segment.run
"""

def var_index(csn_times):
    ## tells you to what the RF has equal C and S power (1 is perfect S signal, 0 is perfectly equal C and S)
    var_s = np.var(csn_times[1])
    var_c = np.var(csn_times[0])
    value = (np.abs(var_c) - np.abs(var_s)) / (np.abs(var_s) + np.abs(var_c))
    if np.ma.is_masked(value):
        return np.nan
    else:
        return value 

def cs_ratio(csn_times):
    absmax_c = pygor.np_ext.maxabs(csn_times[0])
    absmax_s = pygor.np_ext.maxabs(csn_times[1])
    with np.errstate(divide='ignore'):
        value = (absmax_c / absmax_s)
    if np.ma.is_masked(value):
        return np.nan
    else:    
        return value

def cs_contrast(prediction_times):
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


def run_stats(strfs_arrs_list):
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
            output["sum_s"].append(np.sum(prediction_times[1]))
            output["sum_n"].append(np.sum(prediction_times[2]))
            output["sum_c"].append(np.sum(prediction_times[0]))
            output["var_s"].append(np.var(prediction_times[1]))
            output["var_n"].append(np.var(prediction_times[2]))
            output["var_c"].append(np.var(prediction_times[0]))
            output["sd_s"].append(np.std(prediction_times[1]))
            output["sd_n"].append(np.std(prediction_times[2]))
            output["sd_c"].append(np.std(prediction_times[0]))
            output["bidx_s"].append(pygor.strf.temporal.biphasic_index(prediction_times[1]))
            output["bidx_n"].append(pygor.strf.temporal.biphasic_index(prediction_times[2]))
            output["bidx_c"].append(pygor.strf.temporal.biphasic_index(prediction_times[0]))
            output["cent_c"].append(pygor.strf.temporal.only_centroid(prediction_times[0]))
            output["cent_s"].append(pygor.strf.temporal.only_centroid(prediction_times[1]))
            output["cent_n"].append(pygor.strf.temporal.only_centroid(prediction_times[2]))
            # Calculations
            output["var_index"].append(var_index(prediction_times))
            output["cs_ratio"].append(cs_ratio(prediction_times))
            output["cs_contrast"].append(cs_contrast(prediction_times))
    # Check length of stats before returning dict
    lens = []
    for key in output.keys():
        lens.append(len(output[key]))
    if np.all(np.array(lens) == lens[0]) is False:
        raise ValueError("Inconsistent lengths identified, manual fix required")
    return output