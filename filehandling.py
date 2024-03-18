
import numpy as np 
import pathlib 
import h5py
import collections
import os
import datetime 
import pandas as pd
import warnings
import scipy
import pathlib
import itertools
import math 
from collections.abc import Iterable
import joblib
from joblib import Parallel, delayed
import sys
from alive_progress import alive_bar, alive_it
# from tqdm.autonotebook import tqdm
import warnings
from tqdm.auto import tqdm
from multiprocessing import Process
from IPython.display import clear_output
from ipywidgets import Output
import dacite
import natsort # a godsend 

import space
import data_helpers
import contouring
import temporal
import utilities
import unit_conversion
import signal_analysis
from utilities import multicolour_reshape as reshape
# ROI frame needs to contain all stuff from each roi 
# REC frame just keeps tally of recording, and will essentially be only 1 row for each recording
from dataclasses import dataclass, field
import dataclasses
import pickle
import data_objects
from data_objects import Data_STRF, FullField, Data, Experiment, metadata_dict
from data_helpers import label_from_str
"""
The purpose is to use this file to read H5s nearly directly, 
but providing some quality of life improvements like correctly transposed arrays, obtaining metadata easily,
and providing functions to quickly and convenielty access these things.

It is essentially a translation layer between Igor Pro and Python, making my life much easier

In an ideal world this would be a fully-fledged, operational system where you could specify what 
paramters you want, what you expect them to be called in imported h5 file (maybe specified via dict or smth).
However, for now this is more a means to an end --> Getting info in a useful way out of IGOR.  
"""
"""Functions for storing and loading objects______________________________________"""

def save_pkl(object, save_path, filename):
    final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
    print("Storing as:", final_path, end = "\r")
    with open(final_path, 'wb') as outp:
        joblib.dump(object, outp, compress='zlib')
        
def load_pkl(full_path):
    with open(full_path, 'rb') as inp:
        object = joblib.load(inp)
        object.metadata["curr_path"] = full_path
        return object

def _load_parser(file_path, do_bootstrap = True):
    #print("Current file:", i)
    file_type = pathlib.Path(file_path).suffix
    if file_type == ".pkl":
        loaded = load_pkl(file_path)
    if file_type == ".h5":
        loaded = Data_STRF(file_path, do_bootstrap = do_bootstrap)            
    if isinstance(loaded.strfs, np.ndarray) is False and math.isnan(loaded.strfs) is True:
            print("No STRFs found for", file_path, ", skipping...")
            return None
    if loaded.multicolour is False:
            print("Listed file not multichromatic for file", file_path, ", skipping...")
            return None
    if do_bootstrap == True:
            loaded.pval_time
            loaded.pval_space
    return loaded

def load(file_path, do_bootstrap = True):
    return _load_parser(file_path, do_bootstrap)

def _load_and_save(file_path, output_folder, do_bootstrap = True):
    loaded = _load_parser(file_path, do_bootstrap = do_bootstrap)
    name = pathlib.Path(loaded.metadata["filename"]).stem
    loaded.save_pkl(output_folder, name)
    # out.clear_output()

def picklestore_objects(file_paths, output_folder, do_bootstrap = True):
    if isinstance(file_paths, Iterable) is False:
        file_paths = [file_paths]
    output_folder = pathlib.Path(output_folder)
    # progress_bar = alive_it(input_objects, spinner = "fishes", bar = 'blocks', calibrate = 50, force_tty=True)
    progress_bar = tqdm(file_paths, desc = "Iterating through, loading, and storing listed files as objects")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        out = Output()
        display(out)  # noqa: F821
        with out:
            for i in progress_bar:
                _load_and_save(i, output_folder, do_bootstrap = do_bootstrap)
                out.clear_output()

"""Functions for instantiating Data objects___________________________________"""
def hdf5_to_dict(path): 
    _dict = {}
    with h5py.File(path) as HDF5_file:
        for key in HDF5_file.keys():
            _dict[key] = np.array(HDF5_file[key])
    return _dict

def load_from_hdf5(path):
    """
    Loads an HDF5 file directly and writes it to an object, with keys in HDF5 file 
    becoming attributes of that object. 

    Note that you don't get any of the fancy processing attributes with this, just access to waves,
    to be used only for utility 
    """
    new_dict = {}
    with h5py.File(path) as HDF5_file:
        metadata = metadata_dict(HDF5_file)
        for key in HDF5_file.keys():
            new_dict[key] = np.array(HDF5_file[key]).T ## note rotation
    data_dict = new_dict
    final_dict = (data_dict | metadata)
    @dataclass
    class Data_hdf5:
        # Automatically maps contents of HDF5 file
        __annotations__ = {key: type(data_type) for key, data_type in final_dict.items()}
        def attributes(self):
            return list(self.__annotations__)
    # Dacite is a package that allows you to create DataClass objects from dictionaries
    object = dacite.from_dict(Data_hdf5, final_dict)
    return object

# Instantiates the very basic Data object
def load_data(filename, img_stack = True):
    with h5py.File(filename) as HDF5_file:
        rois = np.array(HDF5_file["ROIs"])
        if img_stack == True:
            images = data_helpers.load_wDataCh0(HDF5_file)
        else:
            images = np.nan
        meta_data = metadata_dict(HDF5_file)
    Data_obj = Data(images = images, rois = rois, metadata = meta_data)
    return Data_obj

# Instantiates Data_strf object
# """
# NOTE Should really be renaemd to "gen_Data_STRF" or similar
# """
# def Data_STRF(filename, img_stack = True, strfs = True, ipl_depths = True, keys = True, fix_oversize = False, do_bootstrap = True):
#     with h5py.File(filename) as HDF5_file:
#         # Get keys for STRF, filter for only STRF + n where n is a number between 0 to 9 
#         keys = [i for i in HDF5_file.keys() if "STRF" in i and any(i[4] == np.arange(0, 10).astype("str"))]
#         keys = natsort.natsorted(keys)
#         # Set bool for multi-colour RFs
#         bool_partofmulticolour_list = [len(n.removeprefix("STRF").split("_")) > 2 for n in keys]
#         if all(bool_partofmulticolour_list) == True:
#             multicolour_bool = True
#         if all(bool_partofmulticolour_list) == False:
#             multicolour_bool = False
#         if True in bool_partofmulticolour_list and False in bool_partofmulticolour_list:
#             raise AttributeError("There are both single-coloured and multi-coloured STRFs loaded. Manual fix required.")
#         rois = np.array(HDF5_file["ROIs"])
#         if img_stack == True:
#             images = load_wDataCh0(HDF5_file)
#         else:
#             images = np.nan
#         if strfs == True:
#             strfs_arr = load_strf(HDF5_file)
#             meta_data = metadata_dict(HDF5_file)
#             # User may load file that doesn't contain any matching key, so give np.nan
#             if strfs_arr is None: 
#                 strfs_arr = np.nan
#             else: # Go through scripts for correcting and post-processing STRFs
#                 # 1. Check which axis is the "longest", as Igor frequently rotates arrays 
#                 ## and correct this accordingly 
#                 """
#                 TODO
#                 Account for multi-spectral RFs:
#                 - If attribute .multicolour == True, run through STRF_keys and label R G B UV accordingly
#                 """
#                 input_shape = strfs_arr.shape
#                 if input_shape[2] > input_shape[3]:
#                     warnings.warn(f"Rotation detected and corrected for {filename}", stacklevel=2)
#                     strfs_arr = np.transpose(strfs_arr, axes = (0, 1, 3, 2))
#                 # 2. Check date and correct silly STA mistakes from early experiments 
#                 # Correction for silly STA mistake in the beginning (crop by given factor):
#                 # PS: This over-writes so is semi-sketchy, MAKE DAMN SURE ITS CORRECT!
#                 if fix_oversize == True and meta_data["exp_date"] < datetime.date(2023, 4, 4): #came before < cutoff == True
#                     warnings.warn("Old experiment detected, correcting for oversized STA", stacklevel=2)
#                     try:
#                         size = int(label_from_str(pathlib.Path(meta_data["filename"]).name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
#                         strfs_arr = fix_oversize_sta(strfs_arr, size)
#                     except ValueError:
#                         size = np.nan
#                 # 3. Post process by masking borders and z-scoring arrays 
#                 strfs_arr = post_process_strf_all(strfs_arr)
#         else:
#             strfs_arr = np.nan
#         if ipl_depths == True:
#             try:
#                 ipl_depths = np.array(HDF5_file["Positions"])
#             except KeyError:
#                 warnings.warn(f"HDF5 key 'Positions' not found for file {filename}", stacklevel=2)
#                 ipl_depths = np.array([np.nan])
#     # Dat_strf_obj = Data_strf(strfs, ipl_depths, images, rois, meta_data)
#     Dat_strf_obj = Data_strf(strfs = strfs_arr, ipl_depths = ipl_depths, 
#         images = images, rois = rois, metadata = meta_data, strf_keys = keys, multicolour = multicolour_bool, do_bootstrap = do_bootstrap)
#     Dat_strf_obj.metadata["curr_path"] = filename
#     # res = Data_strf(strfs, rois, metadata=meta_data)
#     return Dat_strf_obj

# def load_strf_by_df_index(df, index, do_bootstrap = True):
#     #roi = df["roi"][index]
#     path = df["curr_path"][index]
#     loaded_data = Data_STRF(path, do_bootstrap = do_bootstrap)
#     return loaded_data

"""Helper functions for Data classes:_________________________________________"""
# def metadata_dict(HDF5_file):
#     date, time = get_experiment_datetime(HDF5_file["wParamsStr"])
#     metadata_dict = {
#     "filename"       : HDF5_file.filename,
#     "exp_date"       : date,
#     "exp_time"       : time,
#     "objectiveXYZ"   : get_rel_objective_XYZ(HDF5_file["wParamsNum"]),
#     }
#     return metadata_dict

# def load_wDataCh0(HDF5_file):
#     # Prioritise detrended (because corrections applied in pre-proc)
#     if "wDataCh0_detrended" in HDF5_file.keys():
#         img = HDF5_file["wDataCh0_detrended"]
#     elif "wDataCh0" in HDF5_file.keys():
#         img = HDF5_file["wDataCh0"]
#     else:
#         warnings.warn("wDataCh0 or wDataCh0_detrended could not be identified. Returning None")
#         img = None
#     return np.array(img).transpose(2,1,0)

# def fix_oversize_sta(strf_arr4d, boxsize_um, upscale_multiple = 4):
#     # Determine STA size from filename convension
#     size = boxsize_um
#     # Figure out how many boxes on screen 
#     boxes_tup = np.ceil(unit_conversion.calculate_boxes_on_screen(size)).astype('int') * upscale_multiple
#     # Create the appropriate mask 
#     mask = utilities.manual_border_mask(strf_arr4d[0][0].shape, boxes_tup) # just take shape from first ROI first frame
#     # Expand to apply mask to each frame 
#     mask = np.expand_dims(mask, (0, 1))
#     mask = np.repeat(mask, strf_arr4d.shape[0], 0)
#     mask = np.repeat(mask, strf_arr4d.shape[1], 1)
#     # Apply the mask
#     new_masked_strfs = np.ma.array(strf_arr4d, mask = mask)
#     # Determine widths of mask 
#     borders_widths = utilities.check_border(new_masked_strfs[0][0].mask, expect_symmetry=False)
#     # Make note of original dimensions
#     org_shape = np.array(strf_arr4d.shape) #dims: roi,z,x,y
#     # Calculate new shape (after killing mask , which will be same for all ROIs in file)
#     new_shape = org_shape - (0, 0, borders_widths[0] + borders_widths[1], borders_widths[2] + borders_widths[3])
#     # Compress masked array (kills values in mask)
#     new_masked_strfs = new_masked_strfs.compressed()
#     # Reshape it to new dimesnions 
#     new_masked_strfs = new_masked_strfs.reshape(new_shape)
#     return new_masked_strfs

"""Finding files:_________________________________________"""

def find_files_in(filetype_ext_str, dir_path, recursive = False, **kwargs) -> list:
    """
    Searches the specified directory for files with the specified file extension.
    The function takes in three parameters:
    - filetype_ext_str (str): The file extension to search for, including the '.', e.g. '.txt'
    - dir_path (str or pathlib.PurePath): The directory path to search in. If a string is provided, it will be converted to a pathlib.PurePath object
    - recursive (bool): If set to True, the function will search recursively through all subdirectories. Default is False.
    
    Returns a list of pathlib.Path objects representing the paths of the files found.
    """
    #  Handle paths using pathlib for maximum enjoyment and minimal life hatered
    if isinstance(dir_path, pathlib.PurePath) is False:
        dir_path = pathlib.Path(dir_path)
    if recursive is False:
        paths = [path for path in dir_path.glob('*' + filetype_ext_str)]
    if recursive is True:
        paths = [path for path in dir_path.rglob('*' + filetype_ext_str)]
    # If search terms are given
    if "match" in kwargs:
        if isinstance(kwargs["match"], str):
            paths = [file for file in paths if kwargs["match"] in file.name]
        else:
            raise AttributeError("kwargs 'search_term' expected a single str. Consider kwargs 'search_terms' (plural) if you want to use a list of strings as search terms.")
    if "match_all" in kwargs:
        if isinstance(kwargs["match_all"], list):
            paths = [file for file in paths if all(map(file.name.__contains__, kwargs["match_all"]))]
        else:
            raise AttributeError("kwargs 'search_term' expected a single str. Consider kwargs 'search_terms' (plural) if you want to use a list of strings as search terms.")
    if "match_any" in kwargs:
        if isinstance(kwargs["match_any"], list):
            paths = [file for file in paths if any(map(file.name.__contains__, kwargs["match_any"]))]
        else:
            raise AttributeError("kwargs 'search_terms' expected list of strings. Consider kwargs 'search_term' (singular) if you want to specify a single str as search term.")
    return paths



"""DataFrame helpers________________________________________________________"""

# def label_from_str(input_str, str_search_terms, str_label, **kwargs):
#     if isinstance(str_label, str):
#         for term in str_search_terms:
#             if term in input_str:
#                 return str_label
#     if isinstance(str_label, np.ndarray) or isinstance(str_label, list) or isinstance(str_label, tuple):
#         for term in str_search_terms:
#             if term in input_str:
#                 if term in str_label:
#                     return term
#             else:
#                 if 'else_return' in kwargs:
#                     return kwargs['else_return']
#                 else:
#                     return np.nan



def _listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1

def powerset(iterable, combinations_only = False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    all_combos = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
    to_str_combinations = [_listToString(x) for x in all_combos]
    if combinations_only == False:
        return to_str_combinations[1:]
    if combinations_only == True:
        return to_str_combinations[1 + len(iterable):]

def numpy_fillna(data):
    if isinstance(data, np.ndarray) is False:
        data = np.array(data, dtype = object)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

remove_these_keys = ["images", "rois","strfs", "metadata", "triggertimes", "triggerstimes_frame ",
    "triggerstimes_frame", "averages", "snippets", "phase_num", "_Data__skip_first_frames", "_Data__skip_last_frames", 
    "frame_hz", "trigger_mode", "_Data__keyword_lables", "_Data__compare_ops_map", "ms_dur", "data_types", "type",
    "_contours_centroids", "_contours", "contours_centroids", "_centres_by_pol", "_timecourses", "rois", "images", "data_types"]

def build_results_dict(data_strf_obj, remove_keys = remove_these_keys): #
    ## Logic here has to be: 
    #   - Compute information as needed 
    #   - Make sure copmuted things are of equal length 
    #   - Store that in a dicitonary 
    #   - Create DF from that dictionary
    #   - Don't bother storing strfs as they can be retrievied via file if needed

    # Check that the instance is actually a Data_strf object and that it contains STRFs (instead of being empty, i.e. nan)
    if isinstance(data_strf_obj, Data_STRF) and data_strf_obj.strfs is not np.nan:
        # Calculate how long each entry should be (should be same as number of STRFs)
        # Note: Reshape/restructure pre-existing content to fit required structure
        expected_lengths = len(data_strf_obj.strfs)

        # Get dictionary verison of object
        dict = data_strf_obj.__dict__.copy()
        # Remove surplus info
        [dict.pop(key, None) for key in remove_keys]
        print("inside results dict build", dict.keys())
        # Make note of how many ROIs for easy indexing later
        dict["roi"] = [int(i.split('_')[1]) for i in data_strf_obj.strf_keys]
            
        dict["multicolour"] = np.repeat(dict["multicolour"], expected_lengths)

        # Deal with metadata
        metadata = data_strf_obj.metadata.copy()
        # dict["metadata"] = np.repeat(metadata, expected_lengths)
        path = pathlib.Path(metadata.pop("filename"))
        dict["date"] = np.repeat(metadata["exp_date"], expected_lengths)
        dict["path"] = np.repeat(path, expected_lengths)
        dict["filename"] = np.repeat(path.name, expected_lengths)

        fish_n_plane = label_from_str(path.name, (np.arange(0, 10).astype('str')))[:2]
        colours_set = ('BW', 'BWnoUV', 'R', 'G', 'B', 'UV')
        chromatic_set = colours_set[2:]
        colours_combos = powerset(('R', 'G', 'B', 'UV'), combinations_only=True)
        if data_strf_obj.multicolour == True:
            # Get last number in strf key 
            strf_key_colour_index = [int(i.split('_')[-1]) for i in data_strf_obj.strf_keys]
            # Assign colour accordingly 
            dict["colour"] = [chromatic_set[i] for i in strf_key_colour_index]
        else:
            dict["colour"] = [label_from_str(path.name, colours_set)] * expected_lengths
        dict["simultaneous"] = [label_from_str(path.name, colours_combos, label = 'y', else_return='n')] * expected_lengths
        dict["combo"] = [label_from_str(path.name, colours_combos)] * expected_lengths
        size = int(label_from_str(path.name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
        dict["size"] =  [size] * expected_lengths 
        shape = np.stack([i.shape for i in data_strf_obj.strfs])
        dict["shapeZ"] = shape[:, 0]
        dict["shapeY"] = shape[:, 1]
        dict["shapeX"] = shape[:, 2]
        dict["XYratio"] = shape[:, 2]/shape[:, 1]
        # Create a conversion factor between size in au, size in vis ang, and STRF area 
        visang_size = np.array(unit_conversion.au_to_visang(size))
        dict["visang_size"] =  np.repeat(visang_size, expected_lengths) #might as well store it 
        # dict["size_bias"] =  []
        dict["frequency"] = [label_from_str(path.name, ['10Hz', '5Hz'])] * expected_lengths
        dict["noise"] = [label_from_str(path.name, ['BWN', 'SWN'])] * expected_lengths
        # Compute results and append to dict 
        ## Compute stats
        # P vals for time and space 
        dict["time_pval"] = [i for i in data_strf_obj.pval_time]
        dict["space_pval"] = [i for i in data_strf_obj.pval_space]

        # Space
        contour_count = np.array(data_strf_obj.contours_count())
        neg_contour_count, pos_contour_count = contour_count[:, 0], contour_count[:, 1] 
        #dict["neg_contour_bool"] = [True for i in neg_contour_count if i > 0 else False for i in neg_contour_count]
        dict["neg_contour_bool"] = [i > 0 for i in neg_contour_count]
        dict["pos_contour_bool"] = [i > 0 for i in pos_contour_count]
        # dict["multicontour_bool"] []
        dict["neg_contour_count"] = neg_contour_count
        dict["pos_contour_count"] = pos_contour_count
        dict["total_contour_count"] = neg_contour_count + pos_contour_count
        neg_contour_areas_corrected = [i[0] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]
        pos_contour_areas_corrected = [i[1] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]        
        tot_neg_areas_corrected, tot_pos_areas_corrected = [np.sum(i) for i in neg_contour_areas_corrected], [np.sum(i) for i in pos_contour_areas_corrected]
        #dict["neg_contour_areas"] = neg_contour_areas_corrected
        #dict["pos_contour_areas"] = pos_contour_areas_corrected
        dict["neg_contour_area_total"] = tot_neg_areas_corrected
        dict["pos_contour_area_total"] = tot_pos_areas_corrected
        dict["contour_area_total"] = np.sum((tot_neg_areas_corrected, tot_pos_areas_corrected), axis = 0)
        dict["contour_complexity"] = np.nanmean(contouring.complexity_weighted(data_strf_obj.contours, data_strf_obj.contours_area()), axis = 1)

         # Time
        timecourses = data_strf_obj.timecourses
        timecourses_neg, timecourses_pos = timecourses[:, 0], timecourses[:, 1]
        neg_extrema, pos_extrema = np.min(timecourses_neg, axis = 1), np.max(timecourses_pos, axis = 1)
        dict["neg_extrema"] = neg_extrema
        dict["pos_extrema"] = pos_extrema
        #dict["dom_extrema"] =  np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_extrema, pos_extrema)
        dict["polarities"] = data_strf_obj.polarities()
        neg_biphasic, pos_biphasic = temporal.biphasic_index(timecourses_neg), temporal.biphasic_index(timecourses_pos)
        dict["neg_biphasic_index"] = neg_biphasic
        dict["pos_biphasic_index"] = pos_biphasic
        dict["dom_biphasic_index"] = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_biphasic, pos_biphasic)
        spearmans_rho = np.array([scipy.stats.spearmanr(i[0], i[1]) for i in timecourses])[:, 0] # slice away the p vals (not accurate for low sample n)
        dict["pols_corr"] =  spearmans_rho # corrcoef/Pearosns via [np.corrcoef(x)[0, 1] for x in timecourses]
        dict["neg_auc"]     = np.trapz(timecourses_neg)
        dict["pos_auc"]     = np.trapz(timecourses_pos)
        neg_peaktime = np.argmin(timecourses_neg, axis = 1)
        pos_peaktime = np.argmax(timecourses_pos, axis = 1)
        dict["neg_peaktime"] = neg_peaktime
        dict["pos_peaktime"] = pos_peaktime
        dict["dom_peaktime"] = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_peaktime, pos_peaktime)
        neg_centroids, pos_centroids = data_strf_obj.spectral_centroids()
        dict["neg_centroids"] = neg_centroids
        dict["pos_centroids"] = pos_centroids
        dict["dom_centroids"] = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_centroids, pos_centroids)
        
        # Proof of concept for assigning entire arrays 
        dict["-timecourse"] = data_strf_obj.timecourses[:, 0].tolist()
        dict["+timecourse"] = data_strf_obj.timecourses[:, 1].tolist()

        # Get rid of residual metadata entry in index, pop it to None (basically delete)
        #dict.pop("metatdata", None)
        # Kill data with incorrect dimensinos 
        # Notes if wanted
        # dict["notes"] = [''] * expected_lengths
        # Finally, loop through entire dictionary and check that all entries are of correct length
        # The logic is that in many cases, there might not be data to compute stats on. These will
        # emtpy lists, so we need to make a note that there was no data there (e.g. nan)
        for i in dict:
            # First check if dictionary entry is iterable
            # If its not, assume we want stat per strfs, so duplicate the value n = strfs times  
            if isinstance(dict[i], Iterable) == False:
                dict[i] = [dict[i]] * data_strf_obj.num_strfs
            #Otherwise, continue and check that all entries are the correct length
            # If not, fill with nan
            if len(dict[i]) != expected_lengths and isinstance(dict[i], (str)) is False:
                if len(dict[i]) > expected_lengths:
                    # Just a little test to make sure dictionary entries make sense (e.g, one for each ROI/STRF)
                    raise AttributeError(f"Dict key {i} was longer than number of expected RFs. Manual fix required.")
                else:
                    dict[i] = dict[i].astype(float)
                    difference = expected_lengths - len(dict[i])
                    dict[i]=  np.pad(dict[i], (difference,0), constant_values=np.nan)
        return dict 
    
def build_recording_dict(data_strf_obj, remove_keys = remove_these_keys):
    dict = data_strf_obj.__dict__.copy()
    # Deal with metadata
    metadata = data_strf_obj.metadata.copy()
    path = pathlib.Path(metadata.pop("filename"))
    dict["path"] = np.array([path])
    dict["filename"] = path.name
    colours_set = ('BW', 'R', 'G', 'B', 'UV')
    colours_combos = powerset(('R', 'G', 'B', 'UV'), combinations_only=True)
    if data_strf_obj.multicolour == True:
        dict["colour"] = "RGBUV"
        dict["paired"] = np.nan
        dict["combo"] = label_from_str(path.name, colours_combos)
    else:
        dict["colour"] = label_from_str(path.name, colours_set)
        dict["paired"] = label_from_str(path.name, colours_combos, label = 'y', else_return='n')        
        dict["combo"] = label_from_str(path.name, colours_combos)
    dict["size"] =  label_from_str(path.name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True)
    dict["frequency"] = label_from_str(path.name, ['10Hz', '5Hz'])
    dict["noise"] = label_from_str(path.name, ['BWN', 'SWN'])
    # print(dict["filename"][0], dict[])
    if data_strf_obj.strfs is np.nan:
        dict["rois_num"] = np.array(0)
    else:
        dict["rois_num"] = np.array(data_strf_obj.strfs.shape[0]).astype(int)
    dict["date"] = np.array([metadata.pop("exp_date")])
    dict["time"] = np.array([metadata.pop("exp_time")])
    dict["strfs_shape"] = [np.array(data_strf_obj.strfs).shape]
    dict["ObjXYZ"] = [metadata.pop("objectiveXYZ")]
    # Remove surplus info
    # remove = ["strf_keys", "metadata", "images", "rois", "strfs", "ipl_depths", "_timecourses", "_contours", 
    #     "_contours_area", "_pval_time", "_pval_space", "_contours"]
    [dict.pop(key, None) for key in remove_keys]
    print("inside rec dict build", dict.keys())
    return dict

def build_chromaticity_dict(data_strf_obj, wavelengths =  ["588", "478", "422", "375"]):
        # Chromaticity
        dict = {}
        num_wavelengths = len(wavelengths)
        if data_strf_obj.multicolour == True:
            # because we have 4 colours, we expect the final length to be n/4
            expected_lengths = int(len(data_strf_obj.strfs) / num_wavelengths)

            # Keep track of metadata
            metadata = data_strf_obj.metadata.copy()
            path = pathlib.Path(metadata.pop("filename"))
            dict["date"] = np.repeat(metadata["exp_date"], expected_lengths)
            dict["path"] = np.repeat(path, expected_lengths)
            dict["curr_path"] = np.repeat(metadata["curr_path"], expected_lengths)
            dict["filename"] = np.repeat(path.name, expected_lengths)
            strf_keys = natsort.natsorted(np.unique(['_'.join(i.split('_')[:2]) for i in data_strf_obj.strf_keys]))
            dict["strf_keys"] = strf_keys
            dict["cell_id"] = [metadata["exp_date"].strftime("%m-%d-%Y") + '_' + '_'.join(j.split('_')[:2]) for j in strf_keys]
            # dict["cell_id"] = 
            size = int(label_from_str(path.name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
            dict["size"] =  [size] * expected_lengths
            # dict["pol_cat"] = data_strf_obj.polarity_category()
            # Some generic stats
            dict["ipl_depths"] = data_strf_obj.ipl_depths # some things are already aligned by cell_id naturally (from Igor)
            dict["cat_pol"] = data_strf_obj.polarity_category()
            
            polarities = utilities.multicolour_reshape(data_strf_obj.polarities(), num_wavelengths)
            complexities = utilities.multicolour_reshape(np.nanmean(contouring.complexity_weighted(data_strf_obj.contours, data_strf_obj.contours_area()), axis = 1), num_wavelengths)
            area_t = data_strf_obj.area_tuning_functions(size).T
            ampl_t = data_strf_obj.amplitude_tuning_functions().T
            neg_cent_t, pos_cent_t = data_strf_obj.centroid_tuning_functions()
            neg_cent_t, pos_cent_t = neg_cent_t.T, pos_cent_t.T
            neg_peak_t, pos_peak_t = data_strf_obj.peaktime_tuning_functions()
            neg_peak_t, pos_peak_t =  neg_peak_t.T, pos_peak_t.T

            # Chromatic aspects 
            for n, i in enumerate(wavelengths):
                dict[f"pol_{i}"] = polarities[n]
                # Tuning functions
                dict[f"area_{i}"] = area_t[n]
                dict[f"ampl_{i}"] = ampl_t[n]
                dict[f"centneg_{i}"] = neg_cent_t[n]
                dict[f"centpos_{i}"] = pos_cent_t[n]
                dict[f"peakneg_{i}"] = neg_peak_t[n]
                dict[f"peakpos_{i}"] = pos_peak_t[n]
                dict[f"comp_{i}"] = complexities[n]
            dict["opp_bool"] = np.array(data_strf_obj.opponency_bool())
            # Sort dictionary for friendliness 
            core_info = ['date', 'path', 'filename', "curr_path", 'strf_keys', 'cell_id', 'size', 'ipl_depths', 'opp_bool']
            flexible_info = natsort.natsorted(list(set(dict.keys() - set(core_info))))
            final_order = core_info + flexible_info
            dict = {key : dict[key] for key in final_order}

            # Type handling 
            # # Attempt labelling 
            # label_list = signal_analysis.category_labels(data_strf_obj.amplitude_tuning_functions())
            # dict["chroma_label"] = label_list
            # dict["chroma_label_simplified"] = ["mid" if i == "blue" or i == "green" or i == "broad" else i for i in dict["chroma_label"]]
            # dict["opponent_bool"] = data_strf_obj.opponency_bool()
            # dict["non_opp_pol"] = np.where(np.array(data_strf_obj.opponency_bool()) == False, np.nanmean(utilities.multicolour_reshape(data_strf_obj.polarities(), 4), axis = 0), 0)

            # # Calculate areas from contours
            # neg_contour_areas_corrected = [i[0] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]
            # pos_contour_areas_corrected = [i[1] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]        
            # tot_neg_areas_corrected, tot_pos_areas_corrected = [np.sum(i) for i in neg_contour_areas_corrected], [np.sum(i) for i in pos_contour_areas_corrected]
            # ## Add negative and positive polarities together
            # strf_area_sums = np.sum((tot_neg_areas_corrected, tot_pos_areas_corrected), axis = 0)
            # area_sums_roi_by_colour = utilities.multicolour_reshape(strf_area_sums, 4)
            # ## Set area == 0 to nan for nan-mean 
            # area_sums_roi_by_colour[area_sums_roi_by_colour == 0] = np.nan
            # ## Average across colours
            # dict["avg_area"] = np.nanmean(area_sums_roi_by_colour, axis = 0)

            # # dict["contour_complexity"] = np.nanmean(contouring.complexity_weighted(data_strf_obj.contours, data_strf_obj.contours_area()), axis = 1)
            # neg_centroids, pos_centroids = data_strf_obj.spectral_centroids()
            # dom_centroids = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_centroids, pos_centroids)
            # dict["avg_centroids"] = np.nanmean(utilities.multicolour_reshape(dom_centroids, 4), axis = 0)
        

            # dict["spectral_tuning_function"] = data_strf_obj.spectral_tuning_functions().tolist()

            return dict
        else:
            raise AttributeError("Attribute 'multicolour' is not True. Manual fix required.")

 
def compile_strf_df(files, summary_prints = True, do_bootstrap = True):
    roi_stat_list = []
    rec_info_list = []
    # progress_bar = alive_it(files, force_tty=True)
    for i in files:
        print("Current file:", i)
        file_type = pathlib.Path(i).suffix
        print(file_type)
        if file_type == ".pkl":
            loaded = load_pkl(i)
        if file_type == ".h5":
            loaded = Data_STRF(i, do_bootstrap = do_bootstrap)
        if isinstance(loaded.strfs, np.ndarray) is False and math.isnan(loaded.strfs) is True:
                print("No STRFs found for", i, ", skipping...")
                continue
        curr_df = pd.DataFrame(build_results_dict(loaded))
        roi_stat_list.append(curr_df)
        curr_rec = pd.DataFrame(build_recording_dict(loaded))
        rec_info_list.append(curr_rec)
        # print(curr_df)
        # rec_df = pd.concat(i)
    roi_df = pd.concat(roi_stat_list, ignore_index=True)
    # roi_df.to_pickle(r"d:/Data_STRF/test")
    # roi_df = roi_df[np.roll(roi_df.columns, 1)]
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    if summary_prints == True:
        print("The following files are missing key 'Positions' resulting in np.nan for 'ipl_depths':\n",
        "\n", pd.unique(roi_df.query("ipl_depths.isnull()")["path"]))
    return roi_df, rec_df

def compile_chroma_strf_df(files, summary_prints = True,  do_bootstrap = True, store_objects = None):        
    roi_stat_list = []
    rec_info_list = []
    chroma_list =   []
    #progress_bar = alive_it(files, spinner = "fishes", bar = 'blocks', calibrate = 50, force_tty=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for i in files:
            print("Current file:", i)
            file_type = pathlib.Path(i).suffix
            if file_type == ".pkl":
                loaded = load_pkl(i)
            if file_type == ".h5":
                loaded = Data_STRF(i, do_bootstrap = do_bootstrap)            
            if isinstance(loaded.strfs, np.ndarray) is False and math.isnan(loaded.strfs) is True:
                    print("No STRFs found for", i, ", skipping...")
                    continue
            if loaded.multicolour is False:
                    print("Listed file not multichromatic for file", i, ", skipping...")
                    continue
            # Check that lengths check out and so the next few lines dont craash out wiht uninterpetable errors
            #lengths = [len(build_results_dict(loaded)[i]) for i in build_results_dict(loaded)]
            curr_df = pd.DataFrame(build_results_dict(loaded))
            roi_stat_list.append(curr_df)
            print("resulting rec dict:", build_recording_dict(loaded).keys())
            curr_rec = pd.DataFrame(build_recording_dict(loaded))
            rec_info_list.append(curr_rec)
            curr_crhoma = pd.DataFrame(build_chromaticity_dict(loaded))
            chroma_list.append(curr_crhoma)
            # print(curr_df)
            # rec_df = pd.concat(i)
            if store_objects is not None:
                name = pathlib.Path(loaded.metadata["filename"]).stem
                print(f"Storing {type(loaded)} as {name}")
                loaded.save_pkl(store_objects, name)
    # Get dfs like usual 
    roi_df = pd.concat(roi_stat_list, ignore_index=True)
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    chroma_df = pd.concat(chroma_list, ignore_index=True)
    
    # Correct indeces due to quadrupling
    all_ipl_depths = np.repeat(np.array(roi_df["ipl_depths"][~np.isnan(roi_df["ipl_depths"])]), 4) # repeats IPL positions 
    roi_df["ipl_depths"] = all_ipl_depths
    roi_df["cell_id"] = [i.strftime("%m-%d-%Y") + '_' + '_'.join(j.split('_')[:2]) for i, j in zip(roi_df["date"], roi_df["strf_keys"])]
    # Type handling
    roi_df["size"] = roi_df["size"].astype("category")
    chroma_df["size"] = chroma_df["size"].astype("category")
    if summary_prints == True:
        print("The following files are missing key 'Positions' resulting in np.nan for 'ipl_depths':\n",
        "\n", pd.unique(roi_df.query("ipl_depths.isnull()")["path"]))
    return roi_df, rec_df, chroma_df

def compile_hdf5_df(files):
    # roi_stat_list = []
    rec_info_list = []
    for i in files:
        loaded = load_from_hdf5(i)
        dict = build_results_dict(loaded)
        curr_df = pd.DataFrame(dict)
        # roi_stat_list.append(curr_df)
        curr_rec = pd.DataFrame(build_recording_dict(loaded))
        rec_info_list.append(curr_rec)
        # print(curr_df)
        # rec_df = pd.concat(i)
    # roi_df = pd.concat(roi_stat_list, ignore_index=True)
    # roi_df.to_pickle(r"d:/Data_STRF/test")
    # roi_df = roi_df[np.roll(roi_df.columns, 1)]
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    return rec_df

def split_df_by(DataFrame, category_str):
    dfs_by_pol = {accepted: sub_df
    for accepted, sub_df in DataFrame.groupby(category_str)}
    if category_str == "cat_pol":
        template_match_dict = {
            "on" :  [1],
            "off":  [-1], 
            "opp":  [1, -1],
            "mix":  [2], 
            "empty": [0],}
        for i in dfs_by_pol.keys():
            if i == "other":
                warnings.warn("Skipping 'other' label" )
            try:
                # Error out if something is incorrect with categorial splitting
                assert (dfs_by_pol[i].drop("cat_pol", axis = 1).filter(like = "pol").apply(lambda x: any(val in x.values for val in template_match_dict[i]), axis=1).any().any())
            except:
                raise AssertionError(f"Assertion not True for category '{i}', suggesting error in df structure")
    return dfs_by_pol