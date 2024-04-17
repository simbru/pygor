# Local imports
import pygor.load
import pygor.utilities as utilities
import pygor.steps.contouring as contouring
import pygor.temporal as temporal
import pygor.classes.experiment
from pygor.data_helpers import label_from_str
from pygor.utilities import powerset
from pygor.utils import unit_conversion
from pygor.filehandling import load_pkl

# External imports
import numpy as np
import pathlib
import scipy
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import natsort
import pandas as pd
import math
import warnings

"""
______ From here is all disgusting DataFrame logic that needs to be dealt with!_________________
"""


"""
TODO Fix DISGUSTING key removal logic
"""
def roi_by_roi_dict(data_strf_obj): #
    """
    Process STRF objects and extract relevant information into a dictionary suitable for DataFrame creation.

    Parameters
    ----------
    data_strf_obj : STRF object
        An instance of pygor's STRF class containing spatiotemporal receptive field data.

    Returns
    -------
    dict
        A dictionary with keys corresponding to metadata, spatial, and temporal properties of the STRFs,
        and values as lists containing the data for these properties, ready for DataFrame conversion.
    """
    ## Logic here has to be: 
    #   - Compute information as needed 
    #   - Make sure copmuted things are of equal length 
    #   - Store that in a dicitonary 
    #   - Create DF from that dictionary
    #   - Don't bother storing strfs as they can be retrievied via file if needed

    # Check that the instance is actually a pygor.load.STRF object and that it contains STRFs (instead of being empty, i.e. nan)
    if isinstance(data_strf_obj, pygor.classes.strf.STRF) and data_strf_obj.strfs is not np.nan:
        # Create dict to write to
        dict = {}
        # Calculate how long each entry should be (should be same as number of STRFs)
        # Note: Reshape/restructure pre-existing content to fit required structure
        expected_lengths = len(data_strf_obj.strfs)
        num_rois = len(np.unique(data_strf_obj.rois)) - 1
        # Deal with metadata
        metadata = data_strf_obj.metadata.copy()
        # Note identifiers
        path = pathlib.Path(metadata.pop("filename"))
        dict["date"] = np.repeat(metadata["exp_date"], expected_lengths)
        dict["path"] = np.repeat(path, expected_lengths)
        dict["filename"] = np.repeat(path.name, expected_lengths)
        # Make note of how many ROIs for easy indexing later
        dict["roi"] = [int(i.split('_')[1]) for i in data_strf_obj.strf_keys]
        # Get IPL info
        dict["ipl_depths"] = data_strf_obj.ipl_depths
        dict["multicolour"] = np.repeat(data_strf_obj.multicolour, expected_lengths)
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
        contour_count = np.array(data_strf_obj.get_contours_count())
        neg_contour_count, pos_contour_count = contour_count[:, 0], contour_count[:, 1] 
        #dict["neg_contour_bool"] = [True for i in neg_contour_count if i > 0 else False for i in neg_contour_count]
        dict["neg_contour_bool"] = [i > 0 for i in neg_contour_count]
        dict["pos_contour_bool"] = [i > 0 for i in pos_contour_count]
        # dict["multicontour_bool"] []
        dict["neg_contour_count"] = neg_contour_count
        dict["pos_contour_count"] = pos_contour_count
        dict["total_contour_count"] = neg_contour_count + pos_contour_count
        neg_contour_areas_corrected = [i[0] for i in data_strf_obj.get_contours_area(unit_conversion.au_to_visang(size)/4)]
        pos_contour_areas_corrected = [i[1] for i in data_strf_obj.get_contours_area(unit_conversion.au_to_visang(size)/4)]        
        tot_neg_areas_corrected, tot_pos_areas_corrected = [np.sum(i) for i in neg_contour_areas_corrected], [np.sum(i) for i in pos_contour_areas_corrected]
        #dict["neg_contour_areas"] = neg_contour_areas_corrected
        #dict["pos_contour_areas"] = pos_contour_areas_corrected
        dict["neg_contour_area_total"] = tot_neg_areas_corrected
        dict["pos_contour_area_total"] = tot_pos_areas_corrected
        dict["contour_area_total"] = np.sum((tot_neg_areas_corrected, tot_pos_areas_corrected), axis = 0)
        dict["contour_complexity"] = np.nanmean(data_strf_obj.calc_contours_complexities())

         # Time
        timecourses = data_strf_obj.get_timecourses()
        timecourses_neg, timecourses_pos = timecourses[:, 0], timecourses[:, 1]
        neg_extrema, pos_extrema = np.min(timecourses_neg, axis = 1), np.max(timecourses_pos, axis = 1)
        dict["neg_extrema"] = neg_extrema
        dict["pos_extrema"] = pos_extrema
        #dict["dom_extrema"] =  np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_extrema, pos_extrema)
        dict["polarities"] = data_strf_obj.get_polarities()
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
        neg_centroids, pos_centroids = data_strf_obj.calc_spectral_centroids()
        dict["neg_centroids"] = neg_centroids
        dict["pos_centroids"] = pos_centroids
        dict["dom_centroids"] = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_centroids, pos_centroids)
        
        # Proof of concept for assigning entire arrays 
        #dict["-timecourse"] = data_strf_obj.get_timecourses()[:, 0].tolist()
        #dict["+timecourse"] = data_strf_obj.get_timecourses()[:, 1].tolist()

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

def recording_dict(data_strf_obj):
    """
    Generate a dictionary with recording information and metadata.

    Parameters
    ----------
    data_strf_obj : object
        An object containing the filename, multicolour status, number of colours, number of STRFs, and metadata.

    Returns
    -------
    dict
        A dictionary containing the path, filename, multicolour status, number of colours, number of STRFs, colour information,
        pairing information, colour combinations, size, frequency, noise data, number of ROIs, date, time, STRFs shape, and objective XYZ position.
    """
    dict = {}
    # Identity info 
    path = pathlib.Path(data_strf_obj.filename)
    dict["filename"] = path
    dict["multicolour"] = data_strf_obj.multicolour
    dict["numcolour"] = data_strf_obj.numcolour
    dict["numstrfs"] = data_strf_obj.num_strfs
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
    return dict

def chromatic_dict(data_strf_obj, wavelengths =  ["588", "478", "422", "375"]):
        """
        Generate a chromaticity dictionary from a data structure object.

        Parameters
        ----------
        data_strf_obj : DataType
            Data structure object containing STRFs and metadata.
        wavelengths : list of str, optional
            List of wavelength strings to be processed. Defaults to ["588", "478", "422", "375"].

        Returns
        -------
        dict
            Dictionary containing chromaticity data and additional metadata.

        Raises
        ------
        AttributeError
            If 'multicolour' attribute of data_strf_obj is not True.
        """        
        # Chromaticity
        dict = {}
        num_wavelengths = len(wavelengths)
        if data_strf_obj.multicolour == True:
            # Because we have 4 colours, we expect the final length to be n/4
            expected_lengths = int(len(data_strf_obj.strfs) / num_wavelengths)
            # Keep track of metadata
            metadata = data_strf_obj.metadata.copy()
            path = pathlib.Path(metadata.pop("filename"))
            dict["date"] = np.repeat(metadata["exp_date"], expected_lengths)
            dict["path"] = np.repeat(path, expected_lengths)
            dict["curr_path"] = np.repeat(data_strf_obj.filename, expected_lengths)
            dict["filename"] = np.repeat(path.name, expected_lengths)
            strf_keys = natsort.natsorted(np.unique(['_'.join(i.split('_')[:2]) for i in data_strf_obj.strf_keys]))
            dict["strf_keys"] = strf_keys
            dict["cell_id"] = [path.stem.replace(" ", "") + metadata["exp_date"].strftime("%m-%d-%Y") + '_' + '_'.join(j.split('_')[:2]) for j in strf_keys]
            dict["roi"] = [int(i.split('_')[1]) for i in data_strf_obj.strf_keys][::data_strf_obj.numcolour]
            # dict["cell_id"] = 
            size = int(label_from_str(path.name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
            dict["size"] =  [size] * expected_lengths
            # dict["pol_cat"] = data_strf_obj.polarity_category()
            # Some generic stats
            dict["ipl_depths"] = data_strf_obj.ipl_depths # some things are already aligned by cell_id naturally (from Igor)
            dict["cat_pol"] = data_strf_obj.get_polarity_category()
            
            polarities = utilities.multicolour_reshape(data_strf_obj.get_polarities(), num_wavelengths)
            complexities = utilities.multicolour_reshape(np.nanmean(contouring.complexity_weighted(data_strf_obj.fit_contours(), data_strf_obj.get_contours_area()), axis = 1), num_wavelengths)
            area_t = data_strf_obj.calc_tunings_area(size).T
            ampl_t = data_strf_obj.calc_tunings_amplitude().T
            neg_cent_t, pos_cent_t = data_strf_obj.calc_tunings_centroids()
            neg_cent_t, pos_cent_t = neg_cent_t.T, pos_cent_t.T
            neg_peak_t, pos_peak_t = data_strf_obj.calc_tunings_peaktime()
            neg_peak_t, pos_peak_t =  neg_peak_t.T, pos_peak_t.T

            # Chromatic aspects
            temporal_filter = utilities.multicolour_reshape(data_strf_obj.get_timecourses(), num_wavelengths)
            spatial_filter = utilities.multicolour_reshape(data_strf_obj.collapse_times(spatial_centre=True), num_wavelengths)
            
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
                #dict[f"temporal_{i}"] = temporal_filter[n].tolist()
                #dict[f"spatial_{i}"] = spatial_filter[n].tolist()
            
            dict["spatial_X"] = [spatial_filter[0, 0].shape[0]] * expected_lengths
            dict["spatial_Y"] = [spatial_filter[0, 0].shape[1]] * expected_lengths
            dict["temporal_len"] = [temporal_filter.shape[0]] * expected_lengths
            dict["opp_bool"] = np.array(data_strf_obj.get_opponency_bool())
            # Sort dictionary for friendliness 
            core_info = ['date', 'path', 'filename', "curr_path", 'strf_keys', 'cell_id', 'size', 'ipl_depths', 'opp_bool']
            flexible_info = natsort.natsorted(list(set(dict.keys() - set(core_info))))
            final_order = core_info + flexible_info
            dict = {key : dict[key] for key in final_order}

            return dict
        else:
            raise AttributeError("Attribute 'multicolour' is not True. Manual fix required.")

def chromatic_stats_df(exp_obj : pygor.classes.experiment.Experiment):
    """
    Generate a DataFrame containing chromatic statistics from an Experiment object.

    Parameters
    ----------
    exp_obj : pygor.classes.experiment.Experiment
        The Experiment object to extract chromatic statistics from.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame of chromatic statistics from all recordings in the Experiment object.
    """

    chromatic_dict_list = []
    for object in exp_obj.recording:
        curr_dict = chromatic_dict(object)
        chromatic_dict_list.append(pd.DataFrame(curr_dict))
    return pd.concat(chromatic_dict_list, ignore_index=True)

def roi_stats_df(exp_obj : pygor.classes.experiment.Experiment):
    """
    Generate statistics dataframe for regions of interest in the given experiment.

    Parameters
    ----------
    exp_obj : pygor.classes.experiment.Experiment
        The experiment object containing recordings.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe of statistics for each region of interest.
    """
    roi_dict = []
    for object in exp_obj.recording:
        curr_dict = pygor.analyses.strf_df.roi_by_roi_dict(object)
        roi_dict.append(pd.DataFrame(curr_dict))
    return pd.concat(chromatic_dict, ignore_index=True)

def rec_info_df(exp_obj : pygor.classes.experiment.Experiment):
    """
    Generate a DataFrame with information from recordings in an Experiment object.

    Parameters
    ----------
    exp_obj : pygor.classes.experiment.Experiment
        The Experiment object to process recordings from.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame with information from all recordings in the Experiment object, indexed by recording.
    """
    rec_dict = []
    for object in exp_obj.recording:
        curr_dict = pygor.analyses.strf_df.recording_dict(object)
        rec_dict.append(pd.DataFrame(curr_dict))
    return pd.concat(chromatic_dict, ignore_index=True)

def split_df_by(DataFrame, category_str):
    raise NotImplementedError("Not implemented yet")
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