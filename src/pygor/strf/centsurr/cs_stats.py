import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
import sklearn.cluster
import skimage.morphology
import pygor.np_ext as np_ext
import pygor.np_ext
import pygor.strf.spatial

"""
All functions here should take as input either 
prediction_map or prediction_timse (or both), from
cs_segment.run
"""

def cs_ratio(cs_times):
    neg = np.min(cs_times[0])
    pos = np.max(cs_times[1])
    bck = np.median(cs_times[2])
    neg_to_bck = neg - bck
    pos_to_bck = pos - bck
    ratio = pos_to_bck / neg_to_bck
    return ratio

