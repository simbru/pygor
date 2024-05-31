# Kudos: https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import copy
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.cluster import KMeans, OPTICS, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
# Import the sklearn function
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
#from collections import Iterable
import warnings
#from collections.abs import Iterable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline#, make_pipeline
import pygor.filehandling as filehandling
import pygor.strf.analyse.analyse 
from joblib import Parallel, delayed
import joblib

"""
TODO 
The reconstruct functions are neither efficient nor scalable. Rewrite needed.
"""
# def reconstruct_cluster_spatial(clust_df, cluster_id_str):
#     r_avg, g_avg, b_avg, uv_avg = [], [], [], []
#     for roi, obj in clust_df.query(f"cluster_id == '{cluster_id_str}'")[["roi", "strf_obj"]].to_numpy():
#         r, g, b, uv = pygor.utilities.multicolour_reshape(obj.collapse_times(spatial_centre=True), 4)[:, roi]
#         r_avg.append(r)
#         g_avg.append(g)
#         b_avg.append(b)
#         uv_avg.append(uv)
#     r_avg = np.average(r_avg, axis = 0)
#     g_avg = np.average(g_avg, axis = 0)
#     b_avg = np.average(b_avg, axis = 0)
#     uv_avg = np.average(uv_avg, axis = 0)
#     srf_avgs = np.array([r_avg, g_avg, b_avg, uv_avg])
#     return srf_avgs

def reconstruct_cluster_spatial(clust_df, cluster_id_str, parallel=True):
    # # Determine number of colours we have
    # chromatic_cols = clust_df.filter(regex=r"_\d").columns
    # unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Get the data needed to reconstruct
    target_arr = clust_df.query(f"cluster_id == '{cluster_id_str}'")[["roi", "strf_obj"]].to_numpy() #makes it possible to unpack
    # Define a loop
    def _loop(obj, roi):
        _unpack_me = pygor.utilities.multicolour_reshape(obj.collapse_times(spatial_centre=True), 4)[:, roi]
        return _unpack_me
    # Optional serial processing    
    if parallel is False or parallel is None:
        collate_strfs = [_loop(obj, roi) for roi, obj in target_arr]
    # Otherwise we use joblib
    elif isinstance(parallel, joblib.parallel.Parallel):
        collate_strfs = parallel(delayed(_loop)(obj, roi) for roi, obj in target_arr)
    elif parallel == True:
        collate_strfs = Parallel(n_jobs = -1)(delayed(_loop)(obj, roi) for roi, obj in target_arr)
    else:
        raise AttributeError("'parallel' must be either True, False, None or a joblib.parllel.Parallel object")
    # Average across ROI dimension (to leave colour, x, y)    
    collate_strfs = np.array(collate_strfs)
    srf_avgs = np.average(collate_strfs, axis=0)
    return srf_avgs

def reconstruct_cluster_temporal(clust_df, cluster_id_str, parallel=True):
    # # Determine number of colours we have
    # chromatic_cols = clust_df.filter(regex=r"_\d").columns
    # unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Get the data needed to reconstruct
    target_arr = clust_df.query(f"cluster_id == '{cluster_id_str}'")[["roi", "strf_obj"]].to_numpy() #makes it possible to unpack
    # Define a loop
    def _loop(obj, roi):
        _unpack_me = pygor.utilities.multicolour_reshape(obj.get_timecourses(), 4)[:, roi]
        return _unpack_me
    # Optional serial processing    
    if parallel is False or parallel is None:
        collate_strfs = [_loop(obj, roi) for roi, obj in target_arr]
    # Otherwise we use joblib
    elif isinstance(parallel, joblib.parallel.Parallel):
        collate_strfs = parallel(delayed(_loop)(obj, roi) for roi, obj in target_arr)
    elif parallel == True:
        collate_strfs = Parallel(n_jobs = -1)(delayed(_loop)(obj, roi) for roi, obj in target_arr)
    else:
        raise AttributeError("'parallel' must be either True, False, None or a joblib.parllel.Parallel object")
    # Average across ROI dimension (to leave colour, x, y)    
    collate_strfs = np.array(collate_strfs)
    times_avgs = np.average(collate_strfs, axis=0)
    return times_avgs

def reconstruct_cluster_strf(clust_df, cluster_id_str):
    r_strf, g_strf, b_strf, uv_strf = [], [], [], []
    for roi, obj in clust_df.query(f"cluster_id == '{cluster_id_str}'")[["roi", "strf_obj"]].to_numpy():
        centred_strfs = obj.centre_strfs()
        r, g, b, uv = pygor.utilities.multicolour_reshape(centred_strfs, 4)[:, roi]
        r_strf.append(r)
        g_strf.append(g)
        b_strf.append(b)
        uv_strf.append(uv)# The line `r_strf = np.average(r_strf, axis = 0)` is calculating the
        # average of the values in the `r_strf` list along the specified axis, which
        # is `axis=0`.
        
    r_strf = np.average(r_strf, axis = 0)
    g_strf = np.average(g_strf, axis = 0)
    b_strf = np.average(b_strf, axis = 0)
    uv_strf =np.average(uv_strf, axis = 0)
    strf_avgs = np.array([r_strf, g_strf, b_strf, uv_strf])
    return strf_avgs

def fetch_cluster_strfs(clust_df, cluster_id_str):
    cluster_df = clust_df.query(f"cluster_id == '{cluster_id_str}'")
    strfs = []
    for series in cluster_df.iloc:
        roi = series["roi"]
        obj = series["strf_obj"]
        strfs.append(obj.strfs_chroma[:, roi])
    strfs = np.ma.array(strfs).swapaxes(0, 1)
    return strfs
# import build_strf_from_cluster(clust_df):
# 