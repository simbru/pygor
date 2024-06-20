import numpy as np
import datetime
import natsort
import warnings
import pygor.utilities
import pathlib
import pygor.utils.unit_conversion
import pygor.utils.helpinfo
import pygor.utilities

"""Helper functions for Data classes:_________________________________________"""

def create_bs_dict(do_bootstrap = True, time_sig_thresh = 0.1,
    space_sig_thresh = 0.1, space_bs_n = 2500, time_bs_n = 2500):
    #now_time = datetime.datetime.now()
    bs_dict = {
        "do_bootstrap"      : do_bootstrap,
        "time_sig_thresh"   : time_sig_thresh, 
        "space_sig_thresh"  : space_sig_thresh,
        "space_bs_n"        : space_bs_n, 
        "time_bs_n"         : time_bs_n, 
        "bs_already_ran"    : False,
        "bs_datetime"       : "",
        "bs_datetime_str"   : "",
        "bs_dur_timedelta"  : "",
        "time_parallel"     : True,
        "space_parallel"    : True,
    }
    return bs_dict

def load_wDataCh0(HDF5_file):
    # Prioritise detrended (because corrections applied in pre-proc)
    if "wDataCh0_detrended" in HDF5_file.keys():
        img = HDF5_file["wDataCh0_detrended"]
    elif "wDataCh0" in HDF5_file.keys():
        img = HDF5_file["wDataCh0"]
    else:
        warnings.warn("wDataCh0 or wDataCh0_detrended could not be identified. Returning None")
        img = None
    return np.array(img).transpose(2,1,0)

def metadata_dict(HDF5_file):
    date, time = get_experiment_datetime(HDF5_file["wParamsStr"])
    metadata_dict = {
    "filename"       : HDF5_file.filename,
    "exp_date"       : date,
    "exp_time"       : time,
    "objectiveXYZ"   : get_rel_objective_XYZ(HDF5_file["wParamsNum"]),
    }
    return metadata_dict

def get_experiment_datetime(wParamsStr_arr):
    date = wParamsStr_arr[4].decode("utf-8") 
    time = wParamsStr_arr[5].decode("utf-8")
    date = np.array(date.split('-')).astype(int)
    time = np.array(time.split('-')).astype(int)
    date = datetime.date(date[0], date[1], date[2])
    time = datetime.time(time[0], time[1], time[2])
    return date, time # ensure date is in string 

def get_rel_objective_XYZ(wParamsNum_arr):
    """Get xyz from wParamsNum"""
    wParamsNum_All = list(wParamsNum_arr)

    """
    Need to do this such that centering is done independently 
    for each plane in a series of files (maybe filter based on filename or smth).
    In this way, the objective position will be the offset in position from first 
    recording in any given seires (but only within, never between experiments)

    Would it make sense to do this based on FishID maybe? Since new fish requires new mount and new location 
    """

    wParamsNum_All_XYZ = wParamsNum_All[26:29] # 26, 27, and 28 (X, Y, Z)
    X = wParamsNum_All_XYZ[0]
    Y = wParamsNum_All_XYZ[2]
    Z = wParamsNum_All_XYZ[1]
    return X, Y, Z

def fix_oversize_sta(strf_arr4d, boxsize_um, upscale_multiple = 4):
    # Determine STA size from filename convension
    size = boxsize_um
    # Figure out how many boxes on screen 
    boxes_tup = np.ceil(pygor.utils.unit_conversion.calculate_boxes_on_screen(size)).astype('int') * upscale_multiple
    # Create the appropriate mask 
    mask = pygor.utilities.manual_border_mask(strf_arr4d[0][0].shape, boxes_tup) # just take shape from first ROI first frame
    # Expand to apply mask to each frame 
    mask = np.expand_dims(mask, (0, 1))
    mask = np.repeat(mask, strf_arr4d.shape[0], 0)
    mask = np.repeat(mask, strf_arr4d.shape[1], 1)
    # Apply the mask
    new_masked_strfs = np.ma.array(strf_arr4d, mask = mask)
    # Determine widths of mask 
    borders_widths = pygor.utilities.check_border(new_masked_strfs, expect_symmetry=False)
    # Make note of original dimensions
    org_shape = np.array(strf_arr4d.shape) #dims: roi,z,x,y
    # Calculate new shape (after killing mask , which will be same for all ROIs in file)
    new_shape = org_shape - (0, 0, borders_widths[0] + borders_widths[1], borders_widths[2] + borders_widths[3])
    # Compress masked array (kills values in mask)
    new_masked_strfs = new_masked_strfs.compressed()
    # Reshape it to new dimesnions 
    new_masked_strfs = new_masked_strfs.reshape(new_shape)
    return new_masked_strfs


def load_strf(HDF5_file, post_process = True, fix_oversize = True):
    # Get all file objects labled STRF0_n
    """
    TODO 
    This needs to flexibly resolve 'STRFn_m' where 'n' represents colours (standardise to 0123 = RGBUV)
    Does it...? Think it through...
    """
    # Get keys from H5 file 
    strf_list = [k for k in HDF5_file.keys() if 'STRF0_' in k]
    if not strf_list: # if its empty
        warnings.warn(f"HDF5 key 'STRF0_' not found for file {HDF5_file.filename}", stacklevel=2)
        return None
    # Correct numerical sorting of strings 
    strf_list_natsort = natsort.natsorted(strf_list)
    # Return those as tranposed arrays, in a list
    strf_arr = np.array([np.array(HDF5_file[v]).transpose(2,1,0) for v in strf_list_natsort if HDF5_file[v].ndim == 3])
    # 1. Check which axis is the "longest", as Igor frequently rotates arrays 
    ## and correct this accordingly 
    input_shape = strf_arr.shape
    if input_shape[2] > input_shape[3]:
        warnings.warn(f"Rotation detected and corrected for {HDF5_file.filename}", stacklevel=2)
        strf_arr = np.transpose(strf_arr, axes = (0, 1, 3, 2))
    # 2. Check date and correct silly STA mistakes from early experiments 
    # Correction for silly STA mistake in the beginning (crop by given factor):
    # PS: This over-writes so is semi-sketchy, MAKE DAMN SURE ITS CORRECT!
    date_string = HDF5_file["wParamsStr"][4].decode()
    experiment_date = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()
    if fix_oversize == True and experiment_date < datetime.date(2023, 4, 4): #came before < cutoff == True
        warnings.warn("Old experiment detected, correcting for oversized STA", stacklevel=2)
        try:
            meta_data = metadata_dict(HDF5_file)
            size = int(label_from_str(pathlib.Path(meta_data["filename"]).name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
            strf_arr = fix_oversize_sta(strf_arr, size)
        except ValueError:
            Warning("Size not identified from filename, causing size to equals nan. May require manual solution.")
            size = np.nan
    # 3. Post process by masking borders and z-scoring arrays 
    strf_arr = post_process_strf_all(strf_arr)
    return strf_arr

def post_process_strf(arr_3d, correct_rotation = True,  zscore = True):
    """Gentle post processing that removes border
    by masking and z-scores the STRFs"""
    if arr_3d is np.nan:
        return np.nan
    # Remove border
    border_mask = pygor.utilities.auto_border_mask(arr_3d)
    arr_3d = np.ma.array(arr_3d, mask = border_mask)
    if zscore == True:
        ## Old implementation
        # Z score over time and space
        # arr_3d = scipy.stats.zscore(arr_3d, axis = None)
        # centred_arr_3d = arr_3d
        ## New implementation (normalised/centred to first frame)
        avg_1stframe = np.ma.average(arr_3d[0])
        std_1stframe = np.ma.std(arr_3d[0])
        centred_arr_3d = (arr_3d - avg_1stframe) / std_1stframe
        # # arr_3d = centred_arr_3d
        return centred_arr_3d

def post_process_strf_all(arr_4d, correct_rotation = True, zscore = True):
    centred_arr_4d = np.ma.empty(arr_4d.shape)
    for n, arr3d in enumerate(arr_4d):
        arr3d = post_process_strf(arr3d)
        centred_arr_4d[n] = arr3d
    return centred_arr_4d

def post_process_strf(arr_3d, correct_rotation = True,  zscore = True):
    """Gentle post processing that removes border
    by masking and z-scores the STRFs"""
    if arr_3d is np.nan:
        return np.nan
    # Remove border
    border_mask = pygor.utilities.auto_border_mask(arr_3d)
    arr_3d = np.ma.array(arr_3d, mask = border_mask)
    if zscore == True:
        ## Old implementation
        # Z score over time and space
        # arr_3d = scipy.stats.zscore(arr_3d, axis = None)
        # centred_arr_3d = arr_3d
        ## New implementation (normalised/centred to first frame)
        avg_1stframe = np.ma.average(arr_3d[0])
        std_1stframe = np.ma.std(arr_3d[0])
        centred_arr_3d = (arr_3d - avg_1stframe) / std_1stframe
        # # arr_3d = centred_arr_3d
        return centred_arr_3d

def post_process_strf_all(arr_4d, correct_rotation = True, zscore = True):
    centred_arr_4d = np.ma.empty(arr_4d.shape)
    for n, arr3d in enumerate(arr_4d):
        arr3d = post_process_strf(arr3d)
        centred_arr_4d[n] = arr3d
    return centred_arr_4d

def label_from_str(input_str, search_terms, label = None, split_by = '_', kick_suffix = True, **kwargs):
    def _decide_output_if_nomatch(**kwargs):
        # In some cases we might want to return a specific thing if no matches are found
        if 'else_return' in kwargs:
            return kwargs['else_return']
        # In most cases, just setting nan is fine
        else:
            return np.nan 
    def _final_check(input_str, search_terms, **kwargs):
        if hasattr(search_terms, '__iter__') is True:
            terms_found = []
            for term in search_terms:
                if term in input_str:
                    terms_found.append(term)
            if terms_found:
                if "first_return_only" in kwargs and kwargs["first_return_only"] == True:
                    return terms_found[0]
                else:
                    return '_'.join(terms_found)
            else:
                _decide_output_if_nomatch()
        else:
            raise AttributeError("search_terms must be iterable of strings")
    # Stuff like '0_G.h5' causes truble when splitting, since the G is tied to .h5 (G.h5). Kick the suffix
    if kick_suffix == True:
        # But allow to keep if absolutely must
        input_str = input_str[:input_str.find('.')]
    # List comprehension returning matching 
    matches = [x for x in input_str.split(split_by) if x in search_terms]
    if not matches: # if its empty
        # do final check 
        matches = _final_check(input_str, search_terms, **kwargs)
        # If matches now contains something
        if matches:
            return matches
        else:
            return _decide_output_if_nomatch(**kwargs)
    if label == None:
        # This looks weird but its a way of getting around sets being weird
        # Essentially, set a string and join the contents of set_match into the string
        return ''.join(matches)
    # In some cases we might want to give 1 label if any of matches are met (e.g, if paired recordings, labled 'RG' or 'GB', return 'Yes')
    else:
        return label