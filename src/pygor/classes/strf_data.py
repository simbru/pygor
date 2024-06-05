from dataclasses import dataclass, field
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.utils.unit_conversion as unit_conversion
import pygor.strf.bootstrap
import pygor.data_helpers
import pygor.utils.helpinfo
import pygor.strf.spatial
import pygor.strf.contouring
import pygor.strf.contouring
import pygor.strf.temporal
import pygor.strf.plot
import pygor.utils
from pygor.classes.core_data import Core
# Dependencies
from tqdm.auto import tqdm
import joblib
import numpy as np
import datetime
import pathlib
import h5py
import natsort
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import natsort

@dataclass(repr = False)
class STRF(Core):
    #type         : str = "STRF"
    # Params 
    multicolour  : bool = False
    bs_bool      : bool = False
    # Annotations
    strfs        : np.ndarray = field(init=False)
    ipl_depths   : np.ndarray = field(init=False)
    numcolour    : int = field(init=False) # gets interpreted from strf array shape
    strf_keys    : list = field(init=False)

    ## Attributes
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        #super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        with h5py.File(self.filename) as HDF5_file:
            # Get keys for STRF, filter for only STRF + n where n is a number between 0 to 9 
            keys = [i for i in HDF5_file.keys() if "STRF" in i and any(i[4] == np.arange(0, 10).astype("str"))]
            self.strf_keys = natsort.natsorted(keys)
            # Set bool for multi-colour RFs and ensure multicolour attributes set correctly
            bool_partofmulticolour_list = [len(n.removeprefix("STRF").split("_")) > 2 for n in keys]
            if all(bool_partofmulticolour_list) == True:
                multicolour_bool = True
            if all(bool_partofmulticolour_list) == False:
                multicolour_bool = False
            if True in bool_partofmulticolour_list and False in bool_partofmulticolour_list:
                raise AttributeError("There are both single-coloured and multi-coloured STRFs loaded. Manual fix required.")
            if multicolour_bool is True:
                identified_labels = np.unique([(i.split('_')[-1]) for i in self.strf_keys])
                self.numcolour = len([i for i in identified_labels if i.isdigit()])
                self.multicolour = True
            else:
                self.numcolour = 1
            self.strfs = pygor.data_helpers.load_strf(HDF5_file)
        self.num_strfs = len(self.strfs)
        self.set_bootstrap_settings_default()
        if self.bs_settings["do_bootstrap"] == True:
            self.run_bootstrap()
        
    @property #
    def stim_size_arbitrary(self):
        """
        Get the largest stimulus size from the object's name.

        The `name` is expected to contain numbers separated by underscores.
        The method extracts all numeric parts, sorts them naturally (as opposed
        to lexicographically), and returns the largest number as the stimulus size.

        Returns
        -------
        int
            The largest number extracted from the `name` attribute, representing
            the stimulus size.
        """
        found_items = [int(i) for i in self.name.split("_") if i.isdigit()]
        if len(found_items) == 0:
            warnings.warn("No numbers found in name, cannot extract stimulus size. Returning np.nan instead.")
            return np.nan
        else:
            stim_size = natsort.natsorted(found_items)[-1] # Fetch the largest number
            return stim_size

    @property
    def stim_size(self, upscaling_factor = 4):
        '''This Python function calculates the visual angle size of a stimulus based on an arbitrary size
        input and an upscaling factor.
        
        Parameters
        ----------
        upscaling_factor, optional
            The `upscaling_factor` parameter is used to determine the factor by which the stimulus size
        will be upscaled. In the provided code snippet, the `stim_size` method takes an optional
        `upscaling_factor` argument with a default value of 4. This argument is used to divide the
        stimulus
        
        Returns
        -------
            The function `stim_size` returns the size of the stimulus in visual angle units after
        converting it from arbitrary units and dividing by the upscaling factor.
        
        '''
        return pygor.utils.unit_conversion.au_to_visang(self.stim_size_arbitrary) / upscaling_factor

    @property
    def strfs_chroma(self):
        return pygor.utilities.multicolour_reshape(self.strfs, self.numcolour)
    
    ## Bootstrapping
    def __calc_pval_time(self) -> np.ndarray:
        """
        Calculate the p-value for each time point in the data.

        Returns:
            np.ndarray: An array of p-values for each time point.
        """
        # Generate bar for beuty
        bar = tqdm(self.strfs, leave = False, position = 1, disable = None, 
            desc = f"Hang on, bootstrapping pygor.strf.temporal components {self.bs_settings['time_bs_n']} times")
        self._pval_time = np.array([pygor.strf.bootstrap.bootstrap_time(x, bootstrap_n=self.bs_settings["time_bs_n"]) for x in bar])
        return self._pval_time

    def __calc_pval_space(self) -> np.ndarray:
        """
        Calculate the p-value space for the spatial components.

        This function calculates the p-value space for the spatial components of the data. It performs a bootstrap
        analysis to estimate the spatial component's significance. The p-value space is an array of shape (n_strfs,)
        where n_strfs is the number of spatial frequency components in the data.

        Returns:
            np.ndarray: The p-value space array.

        """
        # Again, bar for niceness
        bar = tqdm(self.strfs, leave = False, position = 1, disable = None,
            desc = f"Hang on, bootstrapping spatial components {self.bs_settings['space_bs_n']} times")
        self._pval_space = np.array([pygor.strf.bootstrap.bootstrap_space(x, bootstrap_n=self.bs_settings["space_bs_n"]) for x in bar])

    def set_bootstrap_settings_default(self) -> None:
        """
        Sets the default bootstrap settings for the object.

        This function sets the default bootstrap settings for the object by calling the `create_bs_dict` method from the `pygor.data_helpers` module. The `do_bootstrap` parameter is set to the value of the `bs_bool` attribute of the object.

        Parameters:
            None

        Returns:
            None 
            
        """
        self.bs_settings =  pygor.data_helpers.create_bs_dict(do_bootstrap = self.bs_bool)

    def get_bootstrap_settings(self) -> dict:
        """
        Returns the bootstrap settings as a dictionary.

        :return: A dictionary containing the bootstrap settings.
        :rtype: dict
        """
        return self.bs_settings

    def set_bootstrap_bool(self, bool : bool) -> bool:
        '''This Python function sets a boolean value for a bootstrap setting and returns the updated
        boolean value.
        
        Parameters
        ----------
        bool : bool
            The `bool` parameter in the `set_bootstrap_bool` method is a boolean value that indicates
        whether to enable or disable a certain feature related to bootstrapping. It is used to set the
        `bs_bool` attribute of the object to the provided boolean value and update a corresponding
        setting in the `
        
        Returns
        -------
            The method `set_bootstrap_bool` is returning the value of the `self.bs_bool` variable after it
        has been set to the input boolean value and also updating the `do_bootstrap` key in the
        `bs_settings` dictionary with the input boolean value.
        
        '''
        self.bs_bool = bool
        self.bs_settings["do_bootstrap"] = bool
        return self.bs_bool

    def update_bootstrap_settings(self, update_dict, silence_print=False) -> None:
        """
        A function to update the bootstrap settings based on the user input dictionary.

        Parameters
        ----------
        update_dict : dict
            A dictionary containing the settings to update.
        silence_print : bool, optional
            A flag indicating whether to print warnings or not. Defaults to False.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The following keys are disallowed and cannot be updated:
        - "bs_already_ran"
        - "bs_datetime"
        - "bs_datetime_str"
        - "bs_dur_timedelta"

        If any of these keys are present in the `update_dict`, they will be ignored and not updated.
        """
        # Generate default dict for default values 
        default_dict = pygor.data_helpers.create_bs_dict()
        # Check that keys are in default dict
        allowed_keys = list(default_dict.keys())
        # Specify keys that user is not allowed to change
        disallowed_keys = ["bs_already_ran", "bs_datetime", "bs_datetime_str", "bs_dur_timedelta"]
        # Get keys from user input, as a list
        user_keys = list(update_dict.keys())
        # Remove disallowed keys from user input
        not_allowed_keys = []
        not_found_keys   = []
        none_default_key = []
        for key in user_keys:
            if update_dict[key] == None:
                update_dict[key] = default_dict[key]
                none_default_key.append(key)
            if key not in disallowed_keys and key not in allowed_keys:
                not_found_keys.append(key)
                update_dict.pop(key)
            if key in disallowed_keys:
                not_allowed_keys.append(key)
                update_dict.pop(key)
        # Update self.bs_settings accordingly
        self.bs_settings.update({key : value for key, value in update_dict.items()})
        # Handle prints and warnings
        if silence_print == False:
            if not_allowed_keys or not_found_keys:
                warning_string = f"Input anomalies found"
                warnings.warn(warning_string, stacklevel=0)
                if not_allowed_keys:
                    print(f"Disallowed keys: {not_allowed_keys}")
                if not_found_keys:
                    print(f"Keys not found in self.bs_settings: {not_found_keys}")
                if none_default_key:
                    print(f"Keys set to default values: {[(i, default_dict[i]) for i in none_default_key]}")

    def run_bootstrap(self, force = False) -> None:
        """run_bootstrap Runs bootstrapping according to self.bs_settings

        Returns
        -------
        None
            Sets values for self._pval_space and self._pval_time
        """
        if self.bs_settings["do_bootstrap"] == False:
            raise AttributeError("self.bs_settings['do_bootstrap'] is not True")
        if self.bs_settings["do_bootstrap"] == True:
            if self.bs_settings["bs_already_ran"] == True:
                if force == False:
                    user_verify = input("Do you want re-do bootstrap? Type 'y'/'yes' or 'n'/'no'")
                    user_verify = user_verify.lower()
                else:
                    user_verify = 'y'
                if user_verify == 'y' or user_verify == "yes":
                    before_time = datetime.datetime.now()
                    self.__calc_pval_time()
                    self.__calc_pval_space()
                    after_time = datetime.datetime.now()
                elif user_verify == 'n' or user_verify == "no":
                    print(f"Skipping recomputing bootstrap due to user input:'{user_verify}'")
                    return self
                else:
                    print(f"Input '{user_verify}' is invalid, no action done. Please use 'y'/'n'.")
                    return self
            else:
                before_time = datetime.datetime.now()
                self.__calc_pval_time()
                self.__calc_pval_space()
                after_time = datetime.datetime.now()
                self.bs_settings["bs_already_ran"] = True
            # Write time metadata
            self.bs_settings["bs_datetime"] = before_time
            self.bs_settings["bs_datetime_str"] = before_time.strftime("%d/%m/%y %H:%M:%S")
            self.bs_settings["bs_dur_timedelta"] = after_time - before_time
            return self
    @property
    def pval_time(self) -> np.ndarray:
        """
        Returns an array of p-values for time calculation.

        Parameters
        ----------
            self (object): The object instance.
        
        Returns
        ---------
            np.ndarray: An array of p-values for time bootsrap.
        """
        if self.bs_settings["do_bootstrap"] == False:
            return [np.nan] * self.num_strfs
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_time
            except AttributeError:
                print("p value for time bootstrap did not exist, just making that now")
                self.__calc_pval_time() # Executes calculation and writes to self._pval_time
            return self._pval_time
    
    @property
    def pval_space(self) -> np.ndarray:
        """
        Returns an array of p-values for space calculation.

        Parameters
        ----------
            self (object): The object instance.
        
        Returns
        ---------
            np.ndarray: An array of p-values for space bootstrap.
        """
        if self.bs_settings["do_bootstrap"] == False:
            return [np.nan] * self.num_strfs
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_space
            except AttributeError:
                print("p value for space bootstrap did not exist, just making that now")
                self.__calc_pval_space() # Executes calculation and writes to self._pval_space
            return self._pval_space

    def get_pvals_table(self) -> pd.DataFrame:
        if self.multicolour == True: 
            space_vals = pygor.utilities.multicolour_reshape(np.array(self.pval_space), self.numcolour).T
            time_vals = pygor.utilities.multicolour_reshape(np.array(self.pval_time), self.numcolour).T
            space_sig = space_vals < self.bs_settings["space_sig_thresh"]
            time_sig = time_vals < self.bs_settings["time_sig_thresh"]
            both_sig = time_sig * space_sig
            any_sig = np.expand_dims(np.any(both_sig, axis = 1), 1)
            final_arr = np.hstack((space_vals, time_vals, both_sig, any_sig), dtype="object")
            column_labels = ["space_R", "space_G", "space_B", "space_UV", "time_R", "time_G", "time_B", "time_UV",
            "sig_R", "sig_G", "sig_B", "sig_UV", "sig_any"]
            return pd.DataFrame(final_arr, columns=column_labels)
        else:
            space_vals = self.pval_space()
            time_vals = self.pval_space()
            space_sig = space_vals <  self.bs_settings["space_sig_thresh"]
            time_sig = time_vals < self.bs_settings["time_sig_thresh"]
            both_sig = time_sig * space_sig
            final_arr = np.stack((space_vals, time_vals, both_sig), dtype = "object").T
            column_labels = ["space", "time", "sig"]
            return pd.DataFrame(final_arr, columns = column_labels)
    
    @property
    def num_rois_sig(self) -> int:
        return self.get_pvals_table()["sig_any"].sum()

    def fit_contours(self) -> np.array(list[list[list[float, float]]]):
        """
        Returns the contours of the collapse times.

        This function calculates the contours of the collapse times based on the specified bootstrap settings. If the bootstrap settings indicate that bootstrap should be performed, the function calculates the time and space p-values and uses them to determine whether a contour should be drawn for each collapse time. If bootstrap is not performed, the function simply calculates the contours for all collapse times.

        Returns:
            A numpy array of contours. Each contour is represented as a list of tuples, where each tuple contains the x and y coordinates of a point on the contour. If no contour is drawn for a specific collapse time, an empty list is returned.

        Raises:
            AttributeError: If the contours have not been calculated yet.

        Note:
            The contours are calculated using the `pygor.strf.contouring.contour` function.

        Example:
            data = DataObject()
            contours = data.fit_contours()
        """        

        
        # if self.bs_settings["do_bootstrap"] == True:
        #     time_pvals = self.pval_time
        #     space_pvals = self.pval_space
        #     __contours = [pygor.strf.contouring.contour(arr) # ensures no contour is drawn if pval not sig enough
        #                     if time_pvals[count] < self.bs_settings["time_sig_thresh"] and space_pvals[count] < self.bs_settings["space_sig_thresh"]
        #                     else  ([], [])
        #                     for count, arr in enumerate(self.collapse_times())]
        # if self.bs_settings["do_bootstrap"] == False:
        #     __contours = [pygor.strf.contouring.contour(arr) for count, arr in enumerate(self.collapse_times())]
        # __contours = np.array(__contours, dtype = "object")
        # return __contours

        # try:
        #     return self.__contours
        # except AttributeError:
            #self.__contours = [spatial.contour(x) for x in self.collapse_times()]
        if self.bs_settings["do_bootstrap"] == True:
            time_pvals = self.pval_time
            space_pvals = self.pval_space
            __contours = [pygor.strf.contouring.bipolar_contour(arr) # ensures no contour is drawn if pval not sig enough
                            if time_pvals[count] < self.bs_settings["time_sig_thresh"] and space_pvals[count] < self.bs_settings["space_sig_thresh"]
                            else  ([], [])
                            for count, arr in enumerate(self.collapse_times())]
        if self.bs_settings["do_bootstrap"] == False:
            __contours = [pygor.strf.contouring.bipolar_contour(arr) for count, arr in enumerate(self.collapse_times())]
        __contours = np.array(__contours, dtype = "object")
        return __contours


            # self.__contours = np.array(__contours, dtype = "object")
            # return self.__contours    
    def get_contours_count(self) -> list:
        count_list = []
        for i in self.fit_contours():
            neg_contours, pos_contours = i
            count_tup = (len(neg_contours), len(pos_contours))
            count_list.append(count_tup)
        return count_list

    def get_contours_area(self, scaling_factor = None) -> list:
        """
        Generate the area for each contour in the list of contours using the contours_area_bipolar function with a specified scaling factor.

        Parameters:
            scaling_factor (int): A scaling factor to adjust the area calculation (default is 1).

        Returns:
            list: A list of areas for each contour in the list.
        """
        if scaling_factor is None:
            scaling_factor = self.stim_size
        return [pygor.strf.contouring.contours_area_bipolar(__contours, scaling_factor = scaling_factor) for __contours in self.fit_contours()]

    # def get_centsurr_area(self, scaling_factor = 1) -> list:
    #     raise NotImplementedError("Not implemented yet, will give 2xn array with size for centre and surround component (if present, otherwise (s, 0))")
    #     return

    def calc_contours_centroids(self) -> np.ndarray:
        try: 
            return self.__contours_centroids
        except AttributeError:
            #contours_arr = np.array(self.fit_contours(), dtype = "object")
            off_contours = [pygor.strf.contouring.contour_centroid(i) for i in self.fit_contours()[:, 0]]
            on_contours = [pygor.strf.contouring.contour_centroid(i) for i in self.fit_contours()[:, 1]]
            self.__contours_centroids = np.array([off_contours, on_contours], dtype = "object")
            return self.__contours_centroids

    def get_contours_centres_by_pol(self) -> np.ndarray:
        try:
            return self.__centres_by_pol
        except AttributeError:
            self.__centres_by_pol = np.array([
                [np.average(i, axis = 0) for i in self.calc_contours_centroids()[0, :]], 
                [np.average(i, axis = 0) for i in self.calc_contours_centroids()[1, :]]])
            return self.__centres_by_pol

    def get_contours_centres(self, center_on = "amplitude") -> np.ndarray:
        if center_on == "pols":
            return np.nanmean(self.contours_centres_by_pol(), axis = 0)
        if center_on == "largest":
            pos_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.calc_contours_centroids()[1, :]])
            neg_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.calc_contours_centroids()[0, :]])
            area_cents = self.get_contours_area()
            neg_pos_largest = np.array([(i[np.argmax(i)], j[np.argmax(j)]) for i,j in area_cents])
            xs = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 0], pos_conts_cents[:, 0])
            ys = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 1], pos_conts_cents[:, 1])
            return np.array([xs, ys]).T #centres by biggest area, irrespective of polarity
        if center_on == "amplitude":
            pos_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.calc_contours_centroids()[1, :]])
            neg_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.calc_contours_centroids()[0, :]])
            neg_pos_largest = np.max(np.abs(self.get_timecourses()), axis=2)
            xs = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 0], pos_conts_cents[:, 0])
            ys = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 1], pos_conts_cents[:, 1])
            return np.array([xs, ys]).T #centres by biggest area, irrespective of polarity
        else:
            raise ValueError("center_on must be 'pols' or 'biggest'")
     
    def calc_contours_complexities(self) -> np.ndarray:
        return pygor.strf.contouring.complexity_weighted(self.fit_contours(), self.get_contours_area())
     
    """
    TODO Add lazy processing back into contouring (maybe skip timecourses, should be fast enough)
    """
    
    def get_timecourses(self, centre_on_zero = True, mask_empty = False) -> np.ndarray:
        # try:
        #     return self.__timecourses 
        # except AttributeError:
        timecourses = np.ma.average(self.get_strf_masks(), axis = (3,4))
        if mask_empty is False:
            # Sometimes we may/may not want to keep data where space has no correlations (all masked)
            # Fill timecourses with noise array where space is masked
            empty_mask_indices = np.argwhere(timecourses.mask.any(axis = (1,2)))
            noise_times = np.expand_dims(np.average(self.strfs[empty_mask_indices], axis = (3, 4)), axis = 1)
            noise_times = np.repeat(noise_times, 2, axis = 2)
            timecourses[empty_mask_indices] = noise_times
        # Most of the time it makes sense to centre on zero
        if centre_on_zero:
            first_indexes = np.expand_dims(timecourses[:, :, 0], -1)
            timecourses_centred = timecourses - first_indexes
        else:
            timecourses_centred = timecourses
        __timecourses = timecourses_centred
        return __timecourses
            # self.__timecourses = timecourses_centred
            # return self.__timecourses

    def get_chroma_times(self, filter = 'all'):
        if filter == 'all':
            return pygor.utilities.multicolour_reshape(self.get_timecourses(), self.numcolour)
        if filter == 'dominant':
            return pygor.utilities.multicolour_reshape(self.get_timecourses_dominant(), self.numcolour)
        else:
            raise ValueError("filter must be 'all' or 'dominant'")

    def get_chroma_strf(self):
        return pygor.utilities.multicolour_reshape(self.strfs, self.numcolour)

    def get_timecourses_dominant(self):
        dominant_times = []
        for arr in self.get_timecourses().data:
            if np.max(np.abs(arr[0]) > np.max(np.abs(arr[1]))):
                dominant_times.append(arr[0])
            else:
                dominant_times.append(arr[1])
        dominant_times = np.array(dominant_times)
        return dominant_times

    def get_pix_times(self):
        return np.array([np.reshape(i, (i.shape[0], -1)) for i in self.strfs])

    ## Methods__________________________________________________________________________________________________________

    # def plot_roi(self, roi):
    #     fig, ax = plt.subplots(1, 3)

    def get_strf_masks(self, level = None) -> (np.ndarray, np.ndarray):
        """
        Return masked array of spatial.rf_mask3d applied to all arrays, with masks based on pval_time and pval_space,
        with polarity intact.
        """
        if self.strfs is np.nan:
            return np.nan
        else:
            # raise NotImplementedError("Implementation error, does not work yet")
            # get 2d masks
            all_masks = np.array([pygor.strf.contouring.bipolar_mask(i) for i in self.collapse_times()])
            all_masks = np.repeat(np.expand_dims(all_masks, 2), self.strfs.shape[1], axis = 2)
            # Apply mask to expanded and repeated strfs (to get negative and positive)
            strfs_expanded = np.repeat(np.expand_dims(self.strfs, axis = 1), 2, axis = 1)
            all_strfs_masked = np.ma.array(strfs_expanded, mask = all_masks, keep_mask=True)
            return all_strfs_masked

    ## Methods 
    def calc_LED_offset(self, reference_LED_index = [0,1,2], compare_LED_index = [3]) -> np.ndarray:
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
            If the object is not a multicolored spatial-pygor.strf.temporal receptive field (STRF),
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
            avg_colour_centre = np.array([np.nanmean(yx, axis = 0) for yx in pygor.utilities.multicolour_reshape(self.get_contours_centres(), self.numcolour)])
            # Get the average position for the reference LEDs and the comparison LEDs
            avg_reference_pos = np.nanmean(np.take(avg_colour_centre, reference_LED_index, axis = 0), axis = 0)
            avg_compare_pos = np.nanmean(np.take(avg_colour_centre, compare_LED_index, axis = 0), axis = 0)
            # Compute the difference 
            difference = np.diff((avg_compare_pos, avg_reference_pos), axis = 0)[0]
            return difference
        else:
            raise AttributeError("Not a multicoloured STRF, self.multicolour != True.")

    def get_spatial_masks(self) -> (np.ndarray, np.ndarray):
        neg_mask2d, pos_mask2d = self.get_strf_masks().mask[:, :, 0][:, 0], self.get_strf_masks().mask[:, :, 0][:, 1]
        #return np.array([neg_mask2d, pos_mask2d])
        return (neg_mask2d, pos_mask2d)

    @property
    def rf_masks_combined(self) -> np.ndarray:
        mask_2d = self.get_spatial_masks()
        neg_mask2d, pos_mask2d = mask_2d[0], mask_2d[1]
        mask2d_combined  = np.invert(neg_mask2d * -1) + np.invert(pos_mask2d)
        return mask2d_combined

    # def contours_centered
    #     # Estimate centre pixel
    #     pixel_centres = self.get_contours_centres()
    #     avg_centre = np.round(np.nanmean(pixel_centres, axis = 0))
    #     # Find offset between centre coordinate andthe mean coordinate 
    #     differences = np.nan_to_num(avg_centre - pixel_centres).astype("int")
    #     # Loop through and correct
    #     overlap = np.ma.array([np.roll(arr, (x,y), axis = (1,0)) for arr, (y, x) in zip(collapsed_strf_arr, differences)])
    #     collapsed_strf_arr = overlap
    #     return 

    def centre_strfs(self):
        # arr4d = np.copy(self.strfs)
        centres = np.round(self.get_contours_centres())
        target_pos = np.array(self.strfs.shape[2:]) / 2
        shift_by = target_pos - centres
        shift_by = np.nan_to_num(shift_by).astype(int)
        strf_shifted = np.ma.array([np.roll(arr, shift_by[i], axis = (1,2)) for i, arr in enumerate(self.strfs)])
        return strf_shifted

    def collapse_times(self, zscore = False, mode = "var", spatial_centre = False) -> np.ma.masked_array:
        target_shape = (self.strfs.shape[0], 
                        self.strfs.shape[2], 
                        self.strfs.shape[3])    
        collapsed_strf_arr = np.ma.empty(target_shape)
        for n, strf in enumerate(self.strfs):
            collapsed_strf_arr[n] = pygor.strf.spatial.collapse_3d(self.strfs[n], zscore = zscore, mode = mode)
        if spatial_centre == True:
            try:
                return self._spatial_centered_collapse
            except AttributeError:
                # Calculate shifts required for each image (vectorised)
                arr3d = collapsed_strf_arr
                # Ideally (but does not seem to work correctly, skewed by spurrious contours)
                contours_centers = np.where(self.get_contours_centres() > 0, np.floor(self.get_contours_centres()), np.ceil(self.get_contours_centres()))
                target_pos = np.array(arr3d.shape[1:]) / 2
                shift_by = target_pos - contours_centers
                #print("Shift by", shift_by)
                shift_by = np.nan_to_num(shift_by).astype(int) 
                shift_by = shift_by
                # np.roll does not support rolling 3D along depth, so 
                shifted = np.ma.array([np.roll(arr, shift_by[i], axis = (0,1)) for i, arr in enumerate(arr3d)])
                self._spatial_centered_collapse = shifted
            return self._spatial_centered_collapse
            #collapsed_strf_arr = shifted
        return collapsed_strf_arr
        # spatial.collapse_3d(recording.strfs[strf_num])
    
    def get_polarities(self, exclude_FirstLast=(1,1)) -> np.ndarray:
        # Get the time as absolute values, then get the max value
        abs_time_max = np.max(np.abs(self.get_timecourses(mask_empty = True).data), axis = 2)
        # First find the obvious polarities
        pols = np.where(abs_time_max[:, 0] > abs_time_max[:, 1], -1, 1)
        # Now we check if values are close
        ## We reoder, becasue np.isclose(a, b) assumes b is the reference 
        ## and we will use the largst value as the reference
        abs_time_max_reordered = np.sort(abs_time_max, axis = 1)
        outcome = np.isclose(abs_time_max_reordered[:, 0], abs_time_max_reordered[:, 1], rtol = .33, atol = .01)
        # If values were close, we assign 2
        pols = np.where(outcome, 2, pols)
        # Now we need to set values to 0 where there is no signal
        pols = pols * np.prod(np.where(abs_time_max == 0, 0, 1), axis = 1)
        return pols

    def get_opponency_bool(self) -> [bool]:
        if self.multicolour == True:
            arr = pygor.utilities.multicolour_reshape(self.get_polarities(), self.numcolour).T
            # This line looks through rearranged chromatic arr roi by roi 
            # and checks the poliarty by getting the unique values and checking 
            # if the length is more than 0 or 1, excluding NaNs. 
            # I.e., if there is only 1 unique value, there is no opponency
            opponent_bool = [False if len(np.unique(i[~np.isnan(i)])) == 1|0 else True for i in arr]
            return opponent_bool
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour' is False")

    def get_polarity_category(self) -> [str]:
        result = []
        arr = pygor.utilities.multicolour_reshape(self.get_polarities(), self.numcolour).T
        for i in arr:
            inner_no_nan = np.unique(i)[~np.isnan(np.unique(i))]
            inner_no_nan = inner_no_nan[inner_no_nan != 0]
            if not any(inner_no_nan):
                result.append('empty')
            elif np.all(inner_no_nan == -1):
                result.append("off")
            elif np.all(inner_no_nan == 1):
                result.append("on")
            elif -1 in inner_no_nan and 1 in inner_no_nan and 2 not in inner_no_nan:
                result.append("opp")
            elif 2 in inner_no_nan:
                result.append("mix")
            else:
                result.append("other")
        return result

    #def amplitude_tuning_functions(self):
    def calc_tunings_amplitude(self) -> np.ndarray:
        if self.multicolour == True:
            # maxes = np.max(self.collapse_times().data, axis = (1, 2))
            # mins = np.min(self.collapse_times().data, axis = (1, 2))
            maxes = np.max(self.get_timecourses_dominant().data, axis = (1))
            mins = np.min(self.get_timecourses_dominant().data, axis = (1))
            largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
            largest_by_colour = pygor.utilities.multicolour_reshape(largest_mag, self.numcolour)
            # signs = np.sign(largest_by_colour)
            # min_max_scaled = np.apply_along_axis(pygor.utilities.min_max_norm, 1, np.abs(largest_by_colour), 0, 1)
            tuning_functions = largest_by_colour
            return tuning_functions.T #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour.' is False")

    def calc_tunings_area(self, size = None, upscale_factor = 4, largest_only = True) -> np.ndarray:
        if self.multicolour == True:
            # Step 1: Pull contour areas (note, the size is in A.U. for now)
            # Step 2: Split by positive and negative areas
            if size == None:
                warnings.warn("size stimates in arbitrary cartesian units")
                neg_contour_areas = [i[0] for i in self.get_contours_area()]
                pos_contour_areas = [i[1] for i in self.get_contours_area()]
            else:
                neg_contour_areas = [i[0] for i in self.get_contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
                pos_contour_areas = [i[1] for i in self.get_contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
            if largest_only == False:
                # Step 3: Sum these by polarity
                tot_neg_areas, tot_pos_areas = [np.sum(i) for i in neg_contour_areas], [np.sum(i) for i in pos_contour_areas]
                # Step 4: Sum across polarities 
                total_areas = np.sum((tot_neg_areas, tot_pos_areas), axis = 0)
                # Step 5: Reshape to multichromatic format
                area_by_colour = pygor.utilities.multicolour_reshape(total_areas, self.numcolour).T
            if largest_only == True:
                # Step 3: Get the largest contour by polarity
                max_negs = np.max(pygor.utilities.numpy_fillna(neg_contour_areas).astype(float), axis = 1)
                max_pos = np.max(pygor.utilities.numpy_fillna(pos_contour_areas).astype(float), axis = 1)
                # Step 4: Get the largest of these
                abs_max = np.max([max_negs, max_pos], axis = 0)
                # Step 5: Reshape to multichromatic format
                area_by_colour = pygor.utilities.multicolour_reshape(abs_max, self.numcolour).T
            return area_by_colour #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")
    
    def calc_tunings_centroids(self, dominant_only = True) -> np.ndarray:
        if self.multicolour == True:
            # Step 1: Pull centroids
            # Step 2: Not sure how to treat ons and offs yet, just do ONS for now 
            neg_centroids = self.calc_spectral_centroids()[0]
            pos_centroids = self.calc_spectral_centroids()[1]
            # Step 3: Reshaoe ti multichromatic 
            speed_by_colour_neg = pygor.utilities.multicolour_reshape(neg_centroids, self.numcolour).T
            speed_by_colour_pos = pygor.utilities.multicolour_reshape(pos_centroids, self.numcolour).T         
            if dominant_only == False:
                return np.array([speed_by_colour_neg, speed_by_colour_pos])
            else:
                return np.where(self.calc_tunings_amplitude() < 0, speed_by_colour_neg, speed_by_colour_pos)            
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def calc_tunings_peaktime(self, dur_s = 1.3) -> np.ndarray:
        if self.multicolour == True:
            # First get timecourses
            # Split by polarity 
            neg_times, pos_times = self.get_timecourses()[:, 0], self.get_timecourses()[:, 1]
            # Find max position in pos times and neg position in neg times 
            argmins = np.ma.argmin(neg_times, axis = 1)
            argmaxs = np.ma.argmax(pos_times, axis = 1)
            # Reshape result to multichroma 
            argmins  = pygor.utilities.multicolour_reshape(argmins, self.numcolour).T
            argmaxs  = pygor.utilities.multicolour_reshape(argmaxs, self.numcolour).T
            if dur_s != None:
                return  (dur_s / neg_times.shape[1]) * np.array([argmins, argmaxs])
            else:
                warnings.warn("Time values are in arbitary numbers (frames)")
                return np.array([argmins, argmaxs])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def calc_spectral_centroids(self) -> (np.ndarray, np.ndarray):
        spectroids_neg = np.apply_along_axis(pygor.strf.temporal.only_centroid, 1, self.get_timecourses()[:, 0])
        spectroids_pos = np.apply_along_axis(pygor.strf.temporal.only_centroid, 1, self.get_timecourses()[:, 1])
        return spectroids_neg, spectroids_pos

    def calc_spectrums(self, roibyroi = False) -> (np.ndarray, np.ndarray):
        spectrum_neg = np.array([pygor.strf.temporal.only_spectrum(i) for i in self.get_timecourses()[:, 0]])
        spectrum_pos = np.array([pygor.strf.temporal.only_spectrum(i) for i in self.get_timecourses()[:, 1]])
        return spectrum_neg, spectrum_pos

    def unravel_chroma_roi(strf_obj, roi, chroma_index, multichroma_dim = 0, roi_dim = 1):
        return np.ravel_multi_index([roi, chroma_index], np.array(strf_obj.strfs_chroma.shape)[[roi_dim, multichroma_dim]])

    def demo_contouring(self, roi, chromatic_reshape = False):
        plt.close()
        fig, ax = plt.subplots(2, 6, figsize = (16*1.5, 4*1.5))
        neg_c, pos_c = pygor.strf.contouring.bipolar_contour(self.collapse_times()[roi], plot_results= True, ax = ax)
        ax[0, -1].plot(self.get_timecourses()[roi].T, label = ["neg", "pos"])
        ax[0, -1].legend()
        ax[1, -1].imshow((self.get_spatial_masks()[0][roi] * -1) + self.get_spatial_masks()[1][roi], origin = "lower")
        gs = ax[1, 2].get_gridspec()
        for a in ax[0:, 0]:
            a.remove()
        axbig = fig.add_subplot(gs[:, 0])
        axbig.imshow(self.collapse_times()[roi], origin = "lower")
        for i in neg_c:
            axbig.plot(i[:, 1], i[:, 0], color = "blue")
        for i in pos_c:
            axbig.plot(i[:, 1], i[:, 0], color = "red")
        plt.show()
    def plot_timecourse(self, roi):
        plt.plot(self.get_timecourses()[roi].T)
    
    def plot_chromatic_overview(self, roi = None, contours = False, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            return pygor.strf.plot.chroma_overview(self, roi, contours=contours, **kwargs)

    def play_strf(self, roi, **kwargs):
        if isinstance(roi, tuple):
            chroma_arr = pygor.utilities.multicolour_reshape(
                self.strfs, self.numcolour)
            use_map = pygor.plotting.maps_concat[roi[0]]
            anim = pygor.plotting.play_movie(chroma_arr[roi], cmap = use_map,**kwargs)
        else:
            anim = pygor.plotting.play_movie(self.strfs[roi], **kwargs)
        return anim

    def play_multichrom_strf(self, roi = None, **kwargs):
        # anim = pygor.strf.plot.multi_chroma_movie(self, roi, **kwargs)
        if roi is None:
            anim = pygor.plotting.play_movie_4d(self.strfs_chroma, cmap_list =  pygor.plotting.maps_concat, **kwargs)
        else:
            anim = pygor.plotting.play_movie_4d(self.strfs_chroma[:, roi], cmap_list =  pygor.plotting.maps_concat, **kwargs)
        return anim
    # def check_ipl_orientation(self):
    #     raise NotImplementedError("Current implementation does not yield sensible result")
    #     maxes = np.max(self.dominant_timecourses(), axis = 1)
    #     mins = np.min(self.dominant_timecourses(), axis = 1)
    #     # Combine the arrays into a 2D array
    #     combined_array = np.vstack((mins, maxes))
    #     # Find the index of the maximum absolute value along axis 0
    #     max_abs_index = np.argmax(np.abs(combined_array), axis=0)
    #     # Use the index to get the values with their signs intact
    #     result_values = combined_array[max_abs_index, range(combined_array.shape[1])]
    #     # Average every 4th value to get a weighted polarity for each ROI
    #     mean_pols_by_roi = np.nanmean(result_values.reshape(self.numcolour, -1), axis=0)
    #     return (np.sum(mean_pols_by_roi[:int(len(mean_pols_by_roi)/2)]), np.sum(mean_pols_by_roi[int(len(mean_pols_by_roi)/2):]))

    # def save_pkl(self, save_path, filename):
    #     fileehandling.save_pkl(self, save_path, filename)
    def save_pkl(self, save_path, filename) -> None:
        final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
        print("Storing as:", final_path, end = "\r")
        with open(final_path, 'wb') as outp:
            joblib.dump(self, outp, compress='zlib')


# class Clustering:
#     def __init__(self):
#         pass

#     def __repr__(self):
#         return f"{self.__class__.__name__}"

#     def __str__(self):
#         return f"{self.__class__.__name__}"

#     def __call__(self):
#         pass

#     https://realpython.com/python-magic-methods/