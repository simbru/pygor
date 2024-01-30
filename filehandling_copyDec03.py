
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

import dacite
import natsort # a godsend 

import space
import contouring
import temporal
import utilities
import unit_conversion
import signal_analysis
# ROI frame needs to contain all stuff from each roi 
# REC frame just keeps tally of recording, and will essentially be only 1 row for each recording
from dataclasses import dataclass
import dataclasses
"""
The purpose is to use this file to read H5s nearly directly, 
but providing some quality of life improvements like correctly transposed arrays, obtaining metadata easily,
and providing functions to quickly and convenielty access these things.

It is essentially a translation layer between Igor Pro and Python, making my life much easier

In an ideal world this would be a fully-fledged, operational system where you could specify what 
paramters you want, what you expect them to be called in imported h5 file (maybe specified via dict or smth).
However, for now this is more a means to an end --> Getting info in a useful way out of IGOR.  
"""

"""Classes for data handling___________________________________________________"""

@dataclass
class Data:
    metadata : dict = np.nan
    images: np.ndarray = np.nan
    rois  : np.ndarray = np.nan

# @dataclass
# class Data_full(Data):
#     OS_Parameters       :
#     ROIs                :
#     Snippets0           :
#     SnippetsTimes0      :
#     Stack_ave           :
#     Traces0_raw         :
#     Tracetimes0         :
#     Triggertimes        :
#     Triggertimes_Frame  :
#     Triggervalues       :
#     wDataCh0            :
#     wDataCh0_detrended  :
#     wParamsNum          :
#     wParamsStr          :

@dataclass
class Data_strf(Data):
    strfs        : np.ndarray = np.nan
    ipl_depths   : np.ndarray = np.nan
    strf_keys    : np.ndarray = np.nan
    multicolour  : bool = False
    do_bootstrap : bool = False
    time_bs_n    : int = 500
    space_bs_n   : int = 250 
    #"""Take these and do things with em"""
    # contours   : np.ndarray = np.nan
    # timecourses: np.ndarray = np.nan

    # @property
    # def chromatic_reshape(self, data_type):
    #     try:
    #         return self._chromatic_reshape
    #     except AttributeError:
    #         self._chromatic_reshape = self.data_type
    #         return self._chromatic_reshape    

    @property
    def num_strfs(self):
        return len(self.strfs)

    ## Properties --> These make it possible to only instantiate info if its needed
    @property
    def pval_time(self):
        if self.do_bootstrap == False:
            return [np.nan] * self.num_strfs
        if self.do_bootstrap == True:
            try:
                return self._pval_time
            except AttributeError:
                self._pval_time = np.array([signal_analysis.bootstrap_time(x, bootstrap_n=500) for x in self.strfs])
                return self._pval_time
    
    @property
    def pval_space(self):
        if self.do_bootstrap == False:
            return [np.nan] * self.num_strfs
        if self.do_bootstrap == True:
            try:
                return self._pval_space
            except AttributeError:
                self._pval_space =np.array([signal_analysis.bootstrap_space(x, bootstrap_n=250) for x in self.strfs])
                return self._pval_space

    # def pvals(self):

    #     return ()


    @property
    def contours(self, pval_thresh_time = 0.01, pval_thresh_space = 0.01):
        try:
            return self._contours
        except AttributeError:
            #self._contours = [space.contour(x) for x in self.collapse_times()]
            if self.do_bootstrap == True:
                _contours = [contouring.contour(arr) # ensures no contour is drawn if pval not sig enough
                                if self.pval_time[count] < pval_thresh_time and self.pval_space[count] < pval_thresh_space
                                else  ([], [])
                                for count, arr in enumerate(self.collapse_times())]
            if self.do_bootstrap == False:
                _contours = [contouring.contour(arr) for count, arr in enumerate(self.collapse_times())]
            self._contours = np.array(_contours, dtype = "object")
            return self._contours    

    #@ property
    def contours_area(self, scaling_factor = 1):
        return [contouring.contours_area_bipolar(_contours, scaling_factor = scaling_factor) for _contours in self.contours]

#        try: 
#            return self._contours_area
#        except AttributeError:
#            self._contours_area = [contouring.contours_area_bipolar(_contours, scaling_factor = scaling_factor) for _contours in self.contours]
#            return self._contours_area

    @ property
    def contours_centroids(self):
        try: 
            return self._contours_centroids
        except AttributeError:
            #contours_arr = np.array(self.contours, dtype = "object")
            off_contours = [contouring.contour_centroid(i) for i in self.contours[:, 0]]
            on_contours = [contouring.contour_centroid(i) for i in self.contours[:, 1]]
            self._contours_centroids = np.array([off_contours, on_contours], dtype = "object")
            return self._contours_centroids

    @ property
    def contours_centres_by_pol(self):
        try:
            return self._centres_by_pol
        except AttributeError:
            self._centres_by_pol = np.array([
                [np.average(i, axis = 0) for i in self.contours_centroids[0, :]], 
                [np.average(i, axis = 0) for i in self.contours_centroids[1, :]]])
            return self._centres_by_pol

    @ property
    def contours_centres(self):
        return np.nanmean(self.contours_centres_by_pol, axis = 0)

    @property
    def contours_complexities(self):
        return contouring.complexity_weighted(self.contours, self.contours_areas)

    @property
    def timecourses(self, centre_on_zero = True):
        try:
            return self._timecourses 
        except AttributeError:
            timecourses = np.average(self.strf_masks, axis = (3,4))
            first_indexes = np.expand_dims(timecourses[:, :, 0], -1)
            timecourses_centred = timecourses - first_indexes
            self._timecourses = timecourses_centred
            return self._timecourses
    
    def dominant_timecourses(self):
        dominant_times = []
        for arr in self.timecourses.data:
            if np.max(np.abs(arr[0]) > np.max(np.abs(arr[1]))):
                dominant_times.append(arr[0])
            else:
                dominant_times.append(arr[1])
        dominant_times = np.array(dominant_times)
        return dominant_times

    @property 
    def strf_masks(self, level = None, pval_thresh_time = 0.01, pval_thresh_space = 0.01):
        try:
            return self._all_strf_masks
        except AttributeError:
            if self.strfs is np.nan:
                self._all_strf_masks = np.nan
            else:
                # Apply space.rf_mask3d to all arrays and return as a new masksed array
                self._all_strf_masks = np.ma.array([space.rf_mask3d(x, level = None) for x in self.strfs])
                # # Get masks that fail criteria
                pval_fail_time = np.argwhere(np.array(self.pval_time) > pval_thresh_time) # nan > thresh always yields false, so thats really convenient 
                pval_fail_space = np.argwhere(np.array(self.pval_space) > pval_thresh_space) # because if pval is nan, it returns everything
                pval_fail = np.unique(np.concatenate((pval_fail_time, pval_fail_space)))
                self._pval_fail = pval_fail
                # Set entire mask to True conditionally 
                self._all_strf_masks.mask[pval_fail] = True
            return self._all_strf_masks

    ## Methods 
    def calc_LED_offset(self, num_colours = 4, reference_LED_index = [0,1,2], compare_LED_index = [3]):
        """
        Calculate the offset between average positions of reference and comparison LEDs.

        This method computes the offset between the average positions of reference and comparison
        light-emitting diodes (LEDs) based on their indices in a multicolored setup.

        Parameters:
        ----------
        num_colours : int, optional
            The number of colors or LED groups in the multicolored setup. Default is 4.
        reference_LED_index : list of int, optional
            List of indices corresponding to the reference LEDs for which the average position
            will be calculated. Must be list, even if only one LED. Default is [0, 1, 2].
        compare_LED_index : list of int, optional
            List of indices corresponding to the comparison LEDs for which the average position
            will be calculated. Must be list, even if only one LED. Default is [3].

        Returns:
        -------
        numpy.ndarray
            A 1-dimensional array representing the calculated offset between the average positions
            of comparison and reference LEDs.

        Raises:
        ------
        AttributeError
            If the object is not a multicolored spatial-temporal receptive field (STRF),
            indicated by `self.multicolour` not being True.

        Notes:
        ------
        This method assumes that the class instance has attributes:
        - multicolour : bool
            A flag indicating whether the STRF is multicolored.
        - contours_centres : list of numpy.ndarray
            List containing the center positions of contours for each color in the multicolored setup.

        Example:
        --------
        >>> strf_instance = STRF()
        >>> offset = strf_instance.calc_LED_offset(num_colours=4,
        ...                                       reference_LED_index=[0, 1, 2],
        ...                                       compare_LED_index=[3])
        >>> print(offset)
        array([x_offset, y_offset])
        """
        if self.multicolour == True:
            # Get the average centre position for each colour
            avg_colour_centre = np.array([np.nanmean(yx, axis = 0) for yx in utilities.multicolour_reshape(self.contours_centres, 4)])
            # Get the average position for the reference LEDs and the comparison LEDs
            avg_reference_pos = np.nanmean(np.take(avg_colour_centre, reference_LED_index, axis = 0), axis = 0)
            avg_compare_pos = np.nanmean(np.take(avg_colour_centre, compare_LED_index, axis = 0), axis = 0)
            # Compute the difference 
            difference = np.diff((avg_compare_pos, avg_reference_pos), axis = 0)[0]
            return difference
        else:
            raise AttributeError("Not a multicoloured STRF, self.multicolour != True.")

    def timecourses_noncentred(self):
        # Just return raw timecourses 
        self._timecourses = np.average(self.strf_masks, axis = (3,4))
    
    # This one should probably also just be a property to make syntax easier
    def rf_masks(self):
        neg_mask2d, pos_mask2d = self.strf_masks.mask[:, :, 0][:, 0], self.strf_masks.mask[:, :, 0][:, 1]
        return np.array([neg_mask2d, pos_mask2d])

    def rf_masks_combined(self):
        mask_2d = self.rf_masks()
        neg_mask2d, pos_mask2d = mask_2d[0], mask_2d[1]
        mask2d_combined  = np.invert(neg_mask2d * -1) + np.invert(pos_mask2d)
        return mask2d_combined

    def contours_count(self):
        count_list = []
        for i in self.contours:
            neg_contours, pos_contours = i
            count_tup = (len(neg_contours), len(pos_contours))
            count_list.append(count_tup)
        return count_list

    # def contours_centered
    #     # Estimate centre pixel
    #     pixel_centres = self.contours_centres
    #     avg_centre = np.round(np.nanmean(pixel_centres, axis = 0))
    #     # Find offset between centre coordinate andthe mean coordinate 
    #     differences = np.nan_to_num(avg_centre - pixel_centres).astype("int")
    #     # Loop through and correct
    #     overlap = np.ma.array([np.roll(arr, (x,y), axis = (1,0)) for arr, (y, x) in zip(collapsed_strf_arr, differences)])
    #     collapsed_strf_arr = overlap
    #     return 

    def collapse_times(self, zscore = False, mode = "var", spatial_centre = False):
        target_shape = (self.strfs.shape[0], 
                        self.strfs.shape[2], 
                        self.strfs.shape[3])    
        collapsed_strf_arr = np.ma.empty(target_shape)
        for n, strf in enumerate(self.strfs):
            collapsed_strf_arr[n] = space.collapse_3d(self.strfs[n], zscore = zscore, mode = mode)
        if spatial_centre == True:
            # Estimate centre pixel
            pixel_centres = self.contours_centres
            avg_centre = np.round(np.nanmean(pixel_centres, axis = 0))
            # Find offset between centre coordinate andthe mean coordinate 
            differences = np.nan_to_num(avg_centre - pixel_centres).astype("int")
            # Loop through and correct
            overlap = np.ma.array([np.roll(arr, (x,y), axis = (1,0)) for arr, (y, x) in zip(collapsed_strf_arr, differences)])
            collapsed_strf_arr = overlap
        return collapsed_strf_arr
        # space.collapse_3d(recording.strfs[strf_num])

    def polarities(self, exclude_FirstLast=(1,1)):
        if self.strfs is np.nan:
            return np.array([np.nan])
        # Get polarities for time courses (which are 2D arrays containing a 
        # timecourse for negative and positive)
        polarities = temporal.polarity(self.timecourses, exclude_FirstLast)
        # Feed that to helper function to break it down into 1D 
        return polarity_neat(polarities)

    def opponency_bool(self):
        arr = utilities.multicolour_reshape(self.polarities(), 4).T
        opponent_bool = [False if len(np.unique(i[~np.isnan(i)])) == 1|0 else True for i in arr]
        return opponent_bool

    def amplitude_tuning_functions(self):
        if self.multicolour == True:
            # maxes = np.max(self.collapse_times().data, axis = (1, 2))
            # mins = np.min(self.collapse_times().data, axis = (1, 2))
            maxes = np.max(self.dominant_timecourses().data, axis = (1))
            mins = np.min(self.dominant_timecourses().data, axis = (1))
            largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
            largest_by_colour = utilities.multicolour_reshape(largest_mag, 4)
            # signs = np.sign(largest_by_colour)
            # min_max_scaled = np.apply_along_axis(utilities.min_max_norm, 1, np.abs(largest_by_colour), 0, 1)
            tuning_functions = largest_by_colour
            return tuning_functions.T #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def area_tuning_functions(self, size = None, upscale_factor = 4):
        if self.multicolour == True:
            # Step 1: Pull contour areas (note, the size is in A.U. for now)
            # Step 2: Split by positive and negative areas
            if size == None:
                warnings.warn("size stimates in arbitrary cartesian units")
                neg_contour_areas = [i[0] for i in self.contours_area()]
                pos_contour_areas = [i[1] for i in self.contours_area()]
            else:
                neg_contour_areas = [i[0] for i in self.contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
                pos_contour_areas = [i[1] for i in self.contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
            # Step 3: Sum these by polarity
            tot_neg_areas, tot_pos_areas = [np.sum(i) for i in neg_contour_areas], [np.sum(i) for i in pos_contour_areas]
            # Step 4: Sum across polarities 
            total_areas = np.sum((tot_neg_areas, tot_pos_areas), axis = 0)
            # Step 5: Reshape to multichromatic format
            area_by_colour = utilities.multicolour_reshape(total_areas, 4).T
            return area_by_colour #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")
    
    def centroid_tuning_functions(self):
        if self.multicolour == True:
            # Step 1: Pull centroids
            # Step 2: Not sure how to treat ons and offs yet, just do ONS for now 
            neg_centroids = self.spectral_centroids()[0]
            pos_centroids = self.spectral_centroids()[1]
            # Step 3: Reshaoe ti multichromatic 
            speed_by_colour_neg = utilities.multicolour_reshape(neg_centroids, 4).T
            speed_by_colour_pos = utilities.multicolour_reshape(pos_centroids, 4).T 
            # NaNs are 0 
            # roi = 4
            # speed_by_colour_neg = np.nan_to_num(speed_by_colour_neg)
            # speed_by_colour_pos = np.nan_to_num(speed_by_colour_pos)
            # plt.plot(speed_by_colour_neg[roi])
            # plt.plot(speed_by_colour_pos[roi])
            # speed_by_colour[1]
            return np.array([speed_by_colour_neg, speed_by_colour_pos])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def peaktime_tuning_functions(self, dur_s = 1.3):
        if self.multicolour == True:
            # First get timecourses
            # Split by polarity 
            neg_times, pos_times = self.timecourses[:, 0], self.timecourses[:, 1]
            # Find max position in pos times and neg position in neg times 
            argmins = np.ma.argmin(neg_times, axis = 1)
            argmaxs = np.ma.argmax(pos_times, axis = 1)
            # Reshape result to multichroma 
            argmins  = utilities.multicolour_reshape(argmins, 4).T
            argmaxs  = utilities.multicolour_reshape(argmaxs, 4).T
            if dur_s != None:
                return  (dur_s / neg_times.shape[1]) * np.array([argmins, argmaxs])
            else:
                warnings.warn("Time values are in arbitary numbers (frames)")
                return np.array([argmins, argmaxs])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def spectral_centroids(self):
        spectroids_neg = np.apply_along_axis(temporal.only_centroid, 1, self.timecourses[:, 0])
        spectroids_pos = np.apply_along_axis(temporal.only_centroid, 1, self.timecourses[:, 1])
        return spectroids_neg, spectroids_pos

    def spectrums(self):
        spectrum_neg = np.apply_along_axis(temporal.only_spectrum, 1, self.timecourses[:, 0])
        spectrum_pos = np.apply_along_axis(temporal.only_spectrum, 1, self.timecourses[:, 1])
        return spectrum_neg, spectrum_pos

    def check_ipl_orientation(self):    
        maxes = np.max(self.dominant_timecourses(), axis = 1)
        mins = np.min(self.dominant_timecourses(), axis = 1)
        # Combine the arrays into a 2D array
        combined_array = np.vstack((mins, maxes))
        # Find the index of the maximum absolute value along axis 0
        max_abs_index = np.argmax(np.abs(combined_array), axis=0)
        # Use the index to get the values with their signs intact
        result_values = combined_array[max_abs_index, range(combined_array.shape[1])]
        # Average every 4th value to get a weighted polarity for each ROI
        mean_pols_by_roi = np.nanmean(result_values.reshape(4, -1), axis=0)
        return (np.sum(mean_pols_by_roi[:int(len(mean_pols_by_roi)/2)]), np.sum(mean_pols_by_roi[int(len(mean_pols_by_roi)/2):]))



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

    Note that you don't get any of the fancy processing attributes with this, just access to waves
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
    # Dacite is an incredible package that allows you to create DataClass objects from dictionaries
    object = dacite.from_dict(Data_hdf5, final_dict)
    return object

# Instantiates the very basic Data object
def load_data(filename, img_stack = True):
    with h5py.File(filename) as HDF5_file:
        rois = np.array(HDF5_file["ROIs"])
        if img_stack == True:
            images = load_wDataCh0(HDF5_file)
        else:
            images = np.nan
        meta_data = metadata_dict(HDF5_file)
    Data_obj = Data(images = images, rois = rois, metadata = meta_data)
    return Data_obj

# Instantiates Data_strf object
def load_strf_data(filename, img_stack = True, strfs = True, ipl_depths = True, keys = True, fix_oversize = False, do_bootstrap = True):
    with h5py.File(filename) as HDF5_file:
        # Get keys for STRF, filter for only STRF + n where n is a number between 0 to 9 
        keys = [i for i in HDF5_file.keys() if "STRF" in i and any(i[4] == np.arange(0, 10).astype("str"))]
        keys = natsort.natsorted(keys)
        # Set bool for multi-colour RFs
        bool_partofmulticolour_list = [len(n.removeprefix("STRF").split("_")) > 2 for n in keys]
        if all(bool_partofmulticolour_list) == True:
            multicolour_bool = True
        if all(bool_partofmulticolour_list) == False:
            multicolour_bool = False
        if True in bool_partofmulticolour_list and False in bool_partofmulticolour_list:
            raise AttributeError("There are both single-coloured and multi-coloured STRFs loaded. Manual fix required.")
        rois = np.array(HDF5_file["ROIs"])
        if img_stack == True:
            images = load_wDataCh0(HDF5_file)
        else:
            images = np.nan
        if strfs == True:
            strfs_arr = load_strf(HDF5_file)
            meta_data = metadata_dict(HDF5_file)
            # User may load file that doesn't contain any matching key, so give np.nan
            if strfs_arr is None: 
                strfs_arr = np.nan
            else: # Go through scripts for correcting and post-processing STRFs
                # 1. Check which axis is the "longest", as Igor frequently rotates arrays 
                ## and correct this accordingly 
                """
                TODO
                Account for multi-spectral RFs:
                - If attribute .multicolour == True, run through STRF_keys and label R G B UV accordingly
                """
                input_shape = strfs_arr.shape
                if input_shape[2] > input_shape[3]:
                    warnings.warn(f"Rotation detected and corrected for {filename}", stacklevel=2)
                    strfs_arr = np.transpose(strfs_arr, axes = (0, 1, 3, 2))
                # 2. Check date and correct silly STA mistakes from early experiments 
                # Correction for silly STA mistake in the beginning (crop by given factor):
                # PS: This over-writes so is semi-sketchy, MAKE DAMN SURE ITS CORRECT!
                if fix_oversize == True and meta_data["exp_date"] < datetime.date(2023, 4, 4): #came before < cutoff == True
                    warnings.warn("Old experiment detected, correcting for oversized STA", stacklevel=2)
                    try:
                        size = int(label_from_str(pathlib.Path(meta_data["filename"]).name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
                        strfs_arr = fix_oversize_sta(strfs_arr, size)
                    except ValueError:
                        size = np.nan
                # 3. Post process by masking borders and z-scoring arrays 
                strfs_arr = post_process_strf_all(strfs_arr)
        else:
            strfs_arr = np.nan
        if ipl_depths == True:
            try:
                ipl_depths = np.array(HDF5_file["Positions"])
            except KeyError:
                warnings.warn(f"HDF5 key 'Positions' not found for file {filename}", stacklevel=2)
                ipl_depths = np.array([np.nan])
    # Dat_strf_obj = Data_strf(strfs, ipl_depths, images, rois, meta_data)
    Dat_strf_obj = Data_strf(strfs = strfs_arr, ipl_depths = ipl_depths, 
        images = images, rois = rois, metadata = meta_data, strf_keys = keys, multicolour = multicolour_bool, do_bootstrap = do_bootstrap)
    # res = Data_strf(strfs, rois, metadata=meta_data)
    return Dat_strf_obj

def load_strf_by_df_index(df, index, do_bootstrap = True):
    roi = df["roi"][index]
    path = df["path"][index]
    loaded_data = load_strf_data(path, do_bootstrap = do_bootstrap)
    return loaded_data

"""Helper functions for Data classes:_________________________________________"""
def metadata_dict(HDF5_file):
    date, time = get_experiment_datetime(HDF5_file["wParamsStr"])
    metadata_dict = {
    "filename"       : HDF5_file.filename,
    "exp_date"       : date,
    "exp_time"       : time,
    "objectiveXYZ"   : get_rel_objective_XYZ(HDF5_file["wParamsNum"]),
    }
    return metadata_dict

def load_strf(HDF5_file):
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
    strf_arr_list = np.array([np.array(HDF5_file[v]).transpose(2,1,0) for v in strf_list_natsort])
    return strf_arr_list

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

def fix_oversize_sta(strf_arr4d, boxsize_um, upscale_multiple = 4):
    # Determine STA size from filename convension
    size = boxsize_um
    # Figure out how many boxes on screen 
    boxes_tup = np.ceil(unit_conversion.calculate_boxes_on_screen(size)).astype('int') * upscale_multiple
    # Create the appropriate mask 
    mask = utilities.manual_border_mask(strf_arr4d[0][0].shape, boxes_tup) # just take shape from first ROI first frame
    # Expand to apply mask to each frame 
    mask = np.expand_dims(mask, (0, 1))
    mask = np.repeat(mask, strf_arr4d.shape[0], 0)
    mask = np.repeat(mask, strf_arr4d.shape[1], 1)
    # Apply the mask
    new_masked_strfs = np.ma.array(strf_arr4d, mask = mask)
    # Determine widths of mask 
    borders_widths = utilities.check_border(new_masked_strfs[0][0].mask, expect_symmetry=False)
    # Make note of original dimensions
    org_shape = np.array(strf_arr4d.shape) #dims: roi,z,x,y
    # Calculate new shape (after killing mask , which will be same for all ROIs in file)
    new_shape = org_shape - (0, 0, borders_widths[0] + borders_widths[1], borders_widths[2] + borders_widths[3])
    # Compress masked array (kills values in mask)
    new_masked_strfs = new_masked_strfs.compressed()
    # Reshape it to new dimesnions 
    new_masked_strfs = new_masked_strfs.reshape(new_shape)
    return new_masked_strfs

def post_process_strf(arr_3d, correct_rotation = True,  zscore = True):
    """Gentle post processing that removes border
    by masking and z-scores the STRFs"""
    if arr_3d is np.nan:
        return np.nan
    # Remove border
    border_mask = utilities.auto_border_mask(arr_3d)
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

def get_experiment_datetime(wParamsStr_arr):
    date = wParamsStr_arr[4].decode("utf-8") 
    time = wParamsStr_arr[5].decode("utf-8")
    date = np.array(date.split('-')).astype(int)
    time = np.array(time.split('-')).astype(int)
    date = datetime.date(date[0], date[1], date[2])
    time = datetime.time(time[0], time[1], time[2])
    return date, time # ensure date is in string 

def polarity_neat(pol_arr):
    """Helper function which makes polarity more digestable to process by giving an 
    array consisting of -1, 0, 1 or 2 to indicate polarity of STRF. 0 means no polarity, 
    and 2 means bipolar.
    """
    # First check that parameters are in check 
    if isinstance(pol_arr, np.ma.MaskedArray) is True:
        pol_arr = pol_arr.data
    if isinstance(pol_arr, np.ndarray) is False:
        raise AttributeError(f"Function expected input as np.ndarray or np.ma.MaskedArray, not {type(pol_arr)}")  
    if pol_arr.ndim != 2:
        raise AttributeError("Function expected input to have ndim == 2.")
    if np.all(np.isin(pol_arr, (1,0,-1,2))) == False:
        raise AttributeError("Input contained values other than -1, 0, 1, or 2, which is not expected input for this function.")
    # Generate a zero array with correct len
    arr = np.zeros(len(pol_arr))
    # Fill the zeros array to create 1D polarity index
    arr[np.where((pol_arr == (-0, 0)).all(axis=1))] = np.nan
    arr[np.where((pol_arr == (-1, 0)).all(axis=1))] = -1
    arr[np.where((pol_arr == (0, 1)).all(axis=1))] = 1
    arr[np.where((pol_arr == (-1, 1)).all(axis=1))] = 2
    return arr

"""Finding files:_________________________________________"""

def find_files_in(filetype_ext_str, dir_path, recursive = False, **kwargs):
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
    if "search_terms" in kwargs:
        if isinstance(kwargs["search_terms"], list):
            paths = [file for file in paths if any(map(file.name.__contains__, kwargs["search_terms"]))]
        else:
            raise AttributeError("kwargs 'search_terms' expected list of strings. Consider kwargs 'search_term' (singular) if you want to specify a single str as search term.")
    if "search_term" in kwargs:
        if isinstance(kwargs["search_term"], str):
            paths = [file for file in paths if kwargs["search_term"] in file.name]
        else:
            raise AttributeError("kwargs 'search_term' expected a single str. Consider kwargs 'search_terms' (plural) if you want to use a list of strings as search terms.")
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
                    return ''.join(terms_found)
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

def build_results_dict(data_strf_obj, remove_keys = ["images", "rois","strfs", "metadata"]): #
    ## Logic here has to be: 
    #   - Compute information as needed 
    #   - Make sure copmuted things are of equal length 
    #   - Store that in a dicitonary 
    #   - Create DF from that dictionary
    #   - Don't bother storing strfs as they can be retrievied via file if needed

    # Check that the instance is actually a Data_strf object and that it contains STRFs (instead of being empty, i.e. nan)
    if isinstance(data_strf_obj, Data_strf) and data_strf_obj.strfs is not np.nan:
        # Calculate how long each entry should be (should be same as number of STRFs)
        # Note: Reshape/restructure pre-existing content to fit required structure
        expected_lengths = len(data_strf_obj.strfs)

        # Get dictionary verison of object
        dict = data_strf_obj.__dict__.copy()
        # Remove surplus info
        [dict.pop(key, None) for key in remove_keys]
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
        dict["neg_contour_areas"] = neg_contour_areas_corrected
        dict["pos_contour_areas"] = pos_contour_areas_corrected
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
        # dict["-timecourse"] = data_strf_obj.timecourses()[:, 0].tolist()
        # dict["+timecourse"] = data_strf_obj.timecourses()[:, 1].tolist()

        # Get rid of residual metadata entry in index, pop it to None (basically delete)
        dict.pop("metatdata", None)
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
            if len(dict[i]) != expected_lengths:
                if len(dict[i]) > expected_lengths:
                    # Just a little test to make sure dictionary entries make sense (e.g, one for each ROI/STRF)
                    raise AttributeError(f"Dict key {i} was longer than number of expected RFs. Manual fix required.")
                else:
                    dict[i] = dict[i].astype(float)
                    difference = expected_lengths - len(dict[i])
                    dict[i]=  np.pad(dict[i], (difference,0), constant_values=np.nan)
        return dict 
    
def build_recording_dict(data_strf_obj):
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
    remove = ["strf_keys", "metadata", "images", "rois", "strfs", "ipl_depths", "_timecourses", "_contours", 
        "_all_strf_masks", "_contours_area", "_pval_time", "_pval_space"]
    [dict.pop(key, None) for key in remove]

    return dict

def build_chromaticity_dict(data_strf_obj):
        # Chromaticity
        dict = {}
        if data_strf_obj.multicolour == True:
            # because we have 4 colours, we expect the final length to be n/4
            expected_lengths = int(len(data_strf_obj.strfs) / 4)

            # Keep track of metadata
            metadata = data_strf_obj.metadata.copy()
            path = pathlib.Path(metadata.pop("filename"))
            dict["date"] = np.repeat(metadata["exp_date"], expected_lengths)
            dict["path"] = np.repeat(path, expected_lengths)
            dict["filename"] = np.repeat(path.name, expected_lengths)
            strf_keys = natsort.natsorted(np.unique(['_'.join(i.split('_')[:2]) for i in data_strf_obj.strf_keys]))
            dict["strf_keys"] = strf_keys
            dict["cell_id"] = [metadata["exp_date"].strftime("%m-%d-%Y") + '_' + '_'.join(j.split('_')[:2]) for j in strf_keys]
            size = int(label_from_str(path.name, ('800', '400', '200', '100', '75', '50', '25'), first_return_only=True))
            dict["size"] =  [size] * expected_lengths 

            # Some generic stats
            dict["ipl_depths"] = data_strf_obj.ipl_depths # some things are already aligned by cell_id naturally (from Igor)
            label_list = signal_analysis.category_labels(data_strf_obj.spectral_tuning_functions())
            dict["chroma_label"] = label_list
            dict["chroma_label_simplified"] = ["mid" if i == "blue" or i == "green" or i == "broad" else i for i in dict["chroma_label"]]
            dict["opponent_bool"] = data_strf_obj.opponency_bool()
            dict["non_opp_pol"] = np.where(np.array(data_strf_obj.opponency_bool()) == False, np.nanmean(utilities.multicolour_reshape(data_strf_obj.polarities(), 4), axis = 0), 0)

            # Calculate areas from contours
            neg_contour_areas_corrected = [i[0] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]
            pos_contour_areas_corrected = [i[1] for i in data_strf_obj.contours_area(unit_conversion.au_to_visang(size)/4)]        
            tot_neg_areas_corrected, tot_pos_areas_corrected = [np.sum(i) for i in neg_contour_areas_corrected], [np.sum(i) for i in pos_contour_areas_corrected]
            ## Add negative and positive polarities together
            strf_area_sums = np.sum((tot_neg_areas_corrected, tot_pos_areas_corrected), axis = 0)
            area_sums_roi_by_colour = utilities.multicolour_reshape(strf_area_sums, 4)
            ## Set area == 0 to nan for nan-mean 
            area_sums_roi_by_colour[area_sums_roi_by_colour == 0] = np.nan
            ## Average across colours
            dict["avg_area"] = np.nanmean(area_sums_roi_by_colour, axis = 0)

            # dict["contour_complexity"] = np.nanmean(contouring.complexity_weighted(data_strf_obj.contours, data_strf_obj.contours_area()), axis = 1)
            neg_centroids, pos_centroids = data_strf_obj.spectral_centroids()
            dom_centroids = np.where(np.nan_to_num(tot_neg_areas_corrected) > np.nan_to_num(tot_pos_areas_corrected), neg_centroids, pos_centroids)
            dict["avg_centroids"] = np.nanmean(utilities.multicolour_reshape(dom_centroids, 4), axis = 0)
        

            # dict["spectral_tuning_function"] = data_strf_obj.spectral_tuning_functions().tolist()



            return dict
        else:
            raise AttributeError("Attribute 'multicolour' is not True. Manual fix required.")


def compile_strf_df(files, summary_prints = True, do_bootstrap = True):
    roi_stat_list = []
    rec_info_list = []
    for i in files:
        print("Current file:", i)
        loaded = load_strf_data(i, do_bootstrap = do_bootstrap)
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
    # roi_df.to_pickle(r"d:/STRF_data/test")
    # roi_df = roi_df[np.roll(roi_df.columns, 1)]
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    if summary_prints == True:
        print("The following files are missing key 'Positions' resulting in np.nan for 'ipl_depths':\n",
        "\n", pd.unique(roi_df.query("ipl_depths.isnull()")["path"]))
    return roi_df, rec_df

def compile_chroma_strf_df(files, summary_prints = True,  do_bootstrap = True):
    roi_stat_list = []
    rec_info_list = []
    chroma_list =   []
    for i in files:
        print("Current file is:", i)
        loaded = load_strf_data(i, do_bootstrap = do_bootstrap)
        if isinstance(loaded.strfs, np.ndarray) is False and math.isnan(loaded.strfs) is True:
                print("No STRFs found for", i, ", skipping...")
                continue
        if loaded.multicolour is False:
                print("Listed file not multichromatic for file", i, ", skipping...")
                continue
        curr_df = pd.DataFrame(build_results_dict(loaded))
        roi_stat_list.append(curr_df)
        curr_rec = pd.DataFrame(build_recording_dict(loaded))
        rec_info_list.append(curr_rec)
        curr_crhoma = pd.DataFrame(build_chromaticity_dict(loaded))
        chroma_list.append(curr_crhoma)
        # print(curr_df)
        # rec_df = pd.concat(i)
    # Get dfs like usual 
    roi_df = pd.concat(roi_stat_list, ignore_index=True)
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    chroma_df = pd.concat(chroma_list, ignore_index=True)
    # Correct indeces due to quadrupling
    all_ipl_depths = np.repeat(np.array(roi_df["ipl_depths"][~np.isnan(roi_df["ipl_depths"])]), 4) # repeats IPL positions 
    roi_df["ipl_depths"] = all_ipl_depths
    roi_df["cell_id"] = [i.strftime("%m-%d-%Y") + '_' + '_'.join(j.split('_')[:2]) for i, j in zip(roi_df["date"], roi_df["strf_keys"])]
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
    # roi_df.to_pickle(r"d:/STRF_data/test")
    # roi_df = roi_df[np.roll(roi_df.columns, 1)]
    rec_df = pd.concat(rec_info_list, ignore_index=True)
    return rec_df

