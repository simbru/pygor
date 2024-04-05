
import numpy as np 
import pathlib 
import h5py
import collections
import os
import datetime 
import pandas as pd
import warnings

import pygor.utils.utilities
# from temporal import get_pixel_kernel_map, get_island_kernal_map
# from space import centroid, get_covariance_trace, bootstrap_random_distribution, spread_index, dispertion_index
# from plotting import plot_SD_pols, roi_summary_plot, STRF_summary, visualise_dispertion_index, threed_plot
# from basic_functions import determine_centre_polarity, how_many_STRFs

raise DeprecationWarning("The functionality of this has been moved to the filehanlding.py module")

def create_frames(directory, *dispersion_args):
    """create_frames Creates two dataframes (one for recording-parameters and one containing information per ROI from a given recording)

    Returns
    -------
    (DataFrame, DataFrame)
        The returned DFs are (rec_df, roi_df)
    """  
    if isinstance(directory, str) is True:
        directory = pathlib.Path(directory)
    if directory.exists() == False:
        raise FileNotFoundError(f"{path} does not exist.") 
    ### Prepare DataFrame 
    files_list = list(directory.glob("*.h5"))
    dummy_load = files_list[0] # grab first file in dir to intialise DF struct
    with h5py.File(dummy_load) as opened_dummy: # dummy-load a single file to create structure of DF 
        dummy_load_content = list(opened_dummy.keys()) # Get the keys 
    simplifed_dummy_load_content = [match for match in dummy_load_content if not "STRF0_" in match] # Ignore STRFs for now (because multiples)
    num_files = len(files_list) - 1
    # Organise information for df columns 
    recording_params = ["OS_Parameters", "ROIs", 'wDataCh0', 'wDataCh0_detrended', 
        'wParamsNum', 'wParamsStr', 'Triggervalues', 'Triggertimes', 'Stack_ave', 'Island_Sum']       # Only the params which can be attributed to whole recording
    roi_params = sorted([param for param in simplifed_dummy_load_content if param not in recording_params]) # The rest, basically (params per roi)

    ### Build DataFrames...
    master_rec_dict = collections.defaultdict(list)
    master_roi_dict = collections.defaultdict(list) # Basically a fancy dictionary with more flexibility (e.g. can write indeces on the fly)
    ### Note how many ROIs have been accounted for 
    tot_ROIs = 0
    for n, current_file in enumerate(directory.glob("*.h5")): # For each file 
        # Get some file stats that will be useful to store for later 
        file_created = os.path.getctime(current_file)
        file_modified = os.path.getmtime(current_file)
        modified_datetime = datetime.date.fromtimestamp(file_modified)
        created_datetime = datetime.date.fromtimestamp(file_created)
        with h5py.File(current_file) as opened_file: # Open the file
            # Get the keys of the current file 
            current_keys = opened_file.keys()
            curr_strfs = [match for match in current_keys if "STRF0_" in match]
            # Note the date of the experiment (exists in wParamsStr)
            # date_string = get_experiment_date(opened_file['wParamsStr'])
            # print(date_string)
            for n, strf in enumerate(curr_strfs):
                master_roi_dict["STRF"].append(np.array(opened_file[strf]))
            # DataFrame building for ROI df based on collections/dictionaries for organising complex info
            for roi_param_key in roi_params: # For each stat
                try: # may be situations where information doesn't exist 
                    values = np.array(opened_file[roi_param_key]) # Contains stats for given key for all ROIs in recording
                    if values.ndim == 0 or values.ndim == 1:
                        num_values = len(values)
                    else:
                        num_values = values.shape[1]
                        values = np.transpose(values)
                    for i in values: # Fix for dimensionality thing with Igor arrays
                        master_roi_dict[roi_param_key].append(i)         
                except KeyError: # in which case, raise warning and fill with NaNs
                    warnings.warn(f"Missing key '{roi_param_key}' for {current_file}. Deriving quantity of ROIs from shape of Traces0_raw and filling with NaN")
                    # Intuit how many STRFs there SHOULD have been
                    num_values = (opened_file["Traces0_raw"].shape[1])
                    fill_w_nan = np.ones(num_values) * np.nan
                    for i in range(num_values): # Fill sequentially with NaN (to match column lengths in df)
                        master_roi_dict[roi_param_key].append(np.nan)
            # DataFrame building for recording df
            file_rec_list = []
            for rec_param_key in recording_params:
                try: # may be situations where information doesn't exist 
                    values = np.array(opened_file[rec_param_key])
                    if values.ndim == 0 or values.ndim == 1:
                        master_rec_dict[rec_param_key].append(values)
                    else:
                        master_rec_dict[rec_param_key].append(values)
                except KeyError: # Skip values that 
                    print(rec_param_key, "in", current_file, "did not exist. Skipping.")
                    continue
            # Add some filtering params at the same time (to rec_df)
            if '_R_' in current_file.stem[-5:]:
                master_rec_dict["Colors"].append('R')
            elif '_UV_' in current_file.stem[-5:]:
                master_rec_dict["Colors"].append('UV')
            else:
                master_rec_dict["Colors"].append(np.nan)
            master_rec_dict["Files"].append(current_file)
            master_rec_dict["Names"].append(current_file.stem)
            master_rec_dict["FishID"].append(current_file.stem)

        # Now add some fluff to master_roi_dict for later filtering etc.
        for x in range(num_values): # For every ROI in the current_file 
            master_roi_dict["Index_recording"].append(x)
            # Storage info 
            master_roi_dict["Files"].append(current_file)
            master_roi_dict["Names"].append(current_file.stem)
            master_roi_dict["Date_created"].append(created_datetime)
            master_roi_dict["Date_modified"].append(modified_datetime)

            # Experimental info : Filename struct: FISH-count_PLANE-count_STIMULUS-type_PARAMETER-modifier_COLORs_REPETITION
            ## File name handling
            ### Split file name into constituent parts:
            filename_parts = current_file.stem.split('_', 7)
            filename_parts = [int(part) if part.isdigit() == True else part for part in filename_parts] # Use list comprehension to convert digits to int
            ### Ensure the filename matches the expected template 
            parts_types = [type(i) for i in filename_parts]
            match_template = [str, int, int, str, str, str, int] # e.g., ['SMP', 0, 9, 'SWN200', '25p', 'UV', 0]
            if parts_types == match_template == False:
                warnings.warn("The input filename does not match the expected template (e.g., [str, int, int, str, str, str, int] for FISH-count, PLANE-count, STIMULUS-type, PARAMETER-modifier, COLORs, REPETITION. Setting relevant params to NaN, correct manually if needed.")
            ## If filename contains extra, unexpected bits at the end, throw them into "notes" --> also allows later troubleshooting in naming scheme
            if len(filename_parts) == 8: # 7 because we split it only for the first seven split-characthers
                master_roi_dict["Notes"].append(filename_parts[7])
            ### Distribute filename-parameters accordingly (discard 'SMP' --> Unimportant)
            
            """
            TODO:
                - Solve handling of filename if filename is not as expected (make some guesses based on others
                or based on a list of plausible params and where they belong?)
            """
            # raise NotImplementedError("Updated file-name management is under construction")
            if filename_parts[1] == 'SMP':
                experiment_date   = filename_parts[0]
                fish_nr           = filename_parts[2]
                plane             = filename_parts[3]
                stimulus          = filename_parts[4] + ' ' + str(filename_parts[5])
                color             = filename_parts[6]
                repetition        = filename_parts[7]
            #"SMP" optional
            else:
                experiment_date   = filename_parts[0]
                fish_nr           = filename_parts[1]
                plane             = filename_parts[2]
                stimulus          = filename_parts[3] + ' ' + str(filename_parts[4])
                color             = filename_parts[5]
                repetition        = filename_parts[6]
            
            ### Ensure AT LEAST color param makes sense  
            if color not in ['R', 'G', 'B', 'UV', 'All', 'RGBUV']:
                warnings.warn(f"Unexpected input for DataFrame column 'Colors' (got {color}, execpted any('R', 'G', 'B', 'UV', 'All', 'RGBUV'). Setting color to '?'")
                color = '?'
            ## Set values in DataFrame
            master_roi_dict["Fish_n"].append(fish_nr)
            master_roi_dict["FishID"].append(experiment_date + '_' + str(fish_nr))
            master_roi_dict["Plane_n"].append(plane)
            master_roi_dict["Stimulus"].append(stimulus)
            master_roi_dict["Colors"].append(color) ## Chromatic 

            ## roi_rec_num or smth --> Number for ROI relative to file            
            # Cell indeces and stats
            master_roi_dict["ROI_index_rel"].append(x)
        tot_ROIs += num_values
        opened_file.close()

    # Do Dictionary operations 
    # coordinates = np.stack((master_roi_dict["IslandsX"], master_roi_dict["IslandsY"]))
    
    # spread_index(coordinates, bs_covtraces, bs_covtraces_KDEs):

    # Finally, construct dataframes
    rec_df = pd.DataFrame(data = master_rec_dict)#, columns = recording_params)
    roi_df = pd.DataFrame(data = master_roi_dict) # Columns not needed since intrinsic to collections-dict

    ## Do DataFrame operations 
    ### To ROI DFs
    roi_df["Polarities"] = np.stack(roi_df["IslandTimeKernels_Polarities"])[:, 0]
    roi_df["Polarities_scale"] = np.nanmean(np.stack(roi_df["IslandTimeKernels_Polarities"]), axis = 1)
    cov_traces = np.empty(tot_ROIs)
    dispertion_indeces = np.empty(tot_ROIs)
    for n, (x, y) in enumerate(zip(roi_df["IslandsX"], roi_df["IslandsY"])): # Not the nicest solution 
        cov_traces[n] = (np.cov((x[~np.isnan(x)], y[~np.isnan(y)])).trace())
    roi_df["Covariance_trace"] = cov_traces
    # roi_df["Dispersion"] = roi_df["Covariance_trace"].apply(dispertion_index, args = dispersion_args)


    ## To REC DFs
    Xs_, Ys_, Zs_ = pygor.utils.utilities.get_objective_XYZ(rec_df["wParamsNum"])
    rec_df["Objective_XYZraw"] = list(np.stack((Xs_, Ys_, Zs_), axis = 1))

    # Xs_centred, Ys_centred, Zs_centred = get_objective_XYZ(rec_df["wParamsNum"])
    # rec_df["Objective_XYZ"] = list(np.stack((Xs_centred, Ys_centred, Zs_centred), axis = 1))
    
    """
    Need to 
    - Assign unique ID to each fish (date + _ + however many fish I used that day)
    - Store this somewhere centrally
    """


    # # roi_df["Dipsersion_index"] = dispertion_indeces
    return rec_df, roi_df

def merge_dataframes():
    
    """Needs to:
    - Get the centrally stored DataFrames (?)
    - Append to it the newly created one 
    - 

    """
    return 


def get_raw_objective_XYZ(wParamsNum_arr):
    """Helper functino to get xyz from wParamsNum"""
    wParamsNum_All = np.stack(wParamsNum_arr) 
    wParamsNum_All_XYZ = wParamsNum_All[:, 26:29 ] # 26, 27, and 28 (X, Y, Z)
    Xs = wParamsNum_All_XYZ[:, 0]
    Ys = wParamsNum_All_XYZ[:, 2]
    Zs = wParamsNum_All_XYZ[:, 1]
    return Xs, Ys, Zs

def get_rel_objective_XYZ(wParamsNum_arr):
    """Get xyz from wParamsNum"""
    wParamsNum_All = np.stack(wParamsNum_arr) 

    """
    Need to do this such that centering is done independently 
    for each plane in a series of files (maybe filter based on filename or smth).
    In this way, the objective position will be the offset in position from first 
    recording in any given seires (but only within, never between experiments)

    Would it make sense to do this based on FishID maybe? Since new fish requires new mount and new location 
    """

    wParamsNum_All_XYZ = wParamsNum_All[:, 26:29 ] # 26, 27, and 28 (X, Y, Z)
    Xs = wParamsNum_All_XYZ[:, 0]
    Ys = wParamsNum_All_XYZ[:, 2]
    Zs = wParamsNum_All_XYZ[:, 1]
    Xs_centred = (Xs - np.min(Xs)) - (Xs - np.min(Xs))[0]
    Ys_centred = (Ys - np.min(Ys)) - (Ys - np.min(Ys))[0]
    Zs_centred = (Zs - np.min(Zs)) - (Zs - np.min(Zs))[0]
    return Xs_centred, Ys_centred, Zs_centred

def get_experiment_date(wParamsStr_arr):
    date = wParamsStr_arr[4]
    return str(date) # ensure date is in string 
