# Dependencies
from dataclasses import dataclass, field

import pygor.strf.unit_conversion
import pygor.utilities
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import copy
import datetime
import pathlib
import warnings
import h5py
import joblib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import sklearn.preprocessing
import re
from tqdm.auto import tqdm
# Local imports
import pygor.data_helpers
import pygor.strf.bootstrap
import pygor.strf.contouring
import pygor.strf.guesstimate
import pygor.strf.plotting.advanced
import pygor.strf.plotting.simple
import pygor.strf.spatial
import pygor.strf.temporal
import pygor.strf.extrema_timing
import pygor.strf.spatial_alignment
import pygor.utils
import pygor.strf.centsurr
import pygor.strf.calculate
import pygor.strf.calculate_optimized
import pygor.strf.calculate_multicolor_optimized
import pygor.utils.helpinfo
import pygor.utils.unit_conversion as unit_conversion
from pygor.classes.core_data import Core
import scipy
from pygor.classes.core_data import try_fetch_os_params

@dataclass(repr = False)
class STRF(Core):
    #type         : str = "STRF"
    # Params 
    multicolour  : bool = None
    bs_bool      : bool = False
    # Annotations
    strfs        : np.ndarray = field(init=False)
    ipl_depths   : np.ndarray = field(init=False)
    numcolour    : int = field(init=False) # gets interpreted from strf array shape
    strf_keys    : list = field(init=False)
    strf_ms      : int = field(init=False)
    # Cache for spatial overlap index computations
    _spatial_overlap_cache : dict = field(init=False, default_factory=dict)
    ## Attributes
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        #super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        with h5py.File(self.filename) as HDF5_file:
            # Get keys for STRF, filter for only STRF + n where n is a number between 0 to 9 
            # keys = [i for i in HDF5_file.keys() if "?STRF" in i and any(i[4] == np.arange(0, 10).astype("str"))]
            pattern = re.compile(r"^STRF\d+_\d+_\d+$")
            keys = [i for i in HDF5_file.keys() if pattern.match(i) and any(i[4] == np.arange(0, 10).astype("str"))]
            self.strf_keys = natsort.natsorted(keys)
            # Set bool for multi-colour RFs and ensure multicolour attributes set correctly
            strf_colour_int_list = [int(n.removeprefix("STRF").split("_")[-1]) for n in keys]
            if len(np.unique(strf_colour_int_list)) == 1: # only one colour according to STRF keys
                multicolour_bool = False
            else:
                multicolour_bool = True
            self.strf_dur_ms = try_fetch_os_params(HDF5_file, "Noise_FilterLength_s") * 1000
            # if True in bool_partofmulticolour_list and False in bool_partofmulticolour_list:
            #     raise AttributeError("There are both single-coloured and multi-coloured STRFs loaded. Manual fix required.")
            if multicolour_bool is True:
                identified_labels = np.unique([(i.split('_')[-1]) for i in self.strf_keys])
                self.numcolour = len([i for i in identified_labels if i.isdigit()])
                self.multicolour = True
            else:
                self.multicolour = False
                self.numcolour = 1
            self.strfs = pygor.data_helpers.load_strf(HDF5_file)
        self.num_strfs = len(self.strfs)
        if self.num_rois == 0:
            print("Number of ROIs not set, likely ROIs array is missing. Setting to number of STRFs divided by number of colours.")
            self.num_rois = int(self.num_strfs / self.numcolour)
        self.set_bootstrap_settings_default()
        if self.bs_settings["do_bootstrap"] == True:
            self.run_bootstrap()
        
        # Validate data consistency
        self._validate_data_consistency()

    def _validate_data_consistency(self):
        """
        Validate data consistency after loading.
        
        Checks:
        1. num_rois matches expected STRF count (num_strfs / numcolour)
        2. ipl_depths length matches num_rois
        
        Raises warnings for inconsistencies that could indicate data problems.
        """
        expected_rois = int(self.num_strfs / self.numcolour)
        
        # Check 1: num_rois vs STRF count consistency
        if self.num_rois != expected_rois:
            import warnings
            warnings.warn(
                f"Data inconsistency in {self.name}: "
                f"num_rois ({self.num_rois}) != expected from STRFs ({expected_rois}). "
                f"STRFs: {self.num_strfs}, Colors: {self.numcolour}",
                UserWarning
            )
        
        # Check 2: ipl_depths length vs num_rois consistency  
        if hasattr(self, 'ipl_depths') and self.ipl_depths is not None:
            if hasattr(self.ipl_depths, '__len__') and len(self.ipl_depths) != self.num_rois:
                import warnings
                warnings.warn(
                    f"Data inconsistency in {self.name}: "
                    f"ipl_depths length ({len(self.ipl_depths)}) != num_rois ({self.num_rois}). "
                    f"This will cause issues in population analysis. "
                    f"Please check the source H5 file and fix the 'Positions' array.",
                    UserWarning
                )
                print(f"  -> ipl_depths shape: {getattr(self.ipl_depths, 'shape', 'no shape')}")
                print(f"  -> Expected length: {self.num_rois}")
                print(f"  -> Actual length: {len(self.ipl_depths)}")
        
        # Check 3: num_rois vs STRF shape consistency (definitive check)
        try:
            strfs_chroma = self.strfs_chroma()
            actual_rois_from_strfs = strfs_chroma.shape[1]  # shape is [time, roi, y, x] or similar
            if actual_rois_from_strfs != self.num_rois:
                import warnings
                warnings.warn(
                    f"CRITICAL data inconsistency in {self.name}: "
                    f"STRF array shows {actual_rois_from_strfs} ROIs but num_rois is {self.num_rois}. "
                    f"This is the definitive ROI count - other arrays may be wrong. "
                    f"STRF shape: {strfs_chroma.shape}",
                    UserWarning
                )
                print(f"  -> STRF shape: {strfs_chroma.shape}")
                print(f"  -> Actual ROIs from STRFs: {actual_rois_from_strfs}")
                print(f"  -> num_rois setting: {self.num_rois}")
                print(f"  -> This suggests the H5 file has inconsistent ROI counts")
        except Exception as e:
            import warnings
            warnings.warn(
                f"Data validation error in {self.name}: "
                f"Could not check STRF shape consistency: {e}",
                UserWarning
            )
        
        # Check 4: category data consistency
        try:
            categories = self.get_polarity_category_cell()
            if hasattr(categories, '__len__') and len(categories) != self.num_rois:
                import warnings
                warnings.warn(
                    f"Data inconsistency in {self.name}: "
                    f"get_polarity_category_cell() length ({len(categories)}) != num_rois ({self.num_rois}). "
                    f"This will cause issues in population analysis. "
                    f"Check polarity detection and category assignment.",
                    UserWarning
                )
                print(f"  -> Categories returned: {len(categories)} items")
                print(f"  -> Expected: {self.num_rois} items")
                print(f"  -> Categories: {categories}")
        except Exception as e:
            import warnings
            warnings.warn(
                f"Data validation error in {self.name}: "
                f"get_polarity_category_cell() failed with error: {e}. "
                f"This method may be broken for this recording.",
                UserWarning
            )

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

    #@property
    def strfs_chroma(self, border = False):
        if border == False:
            return pygor.utilities.multicolour_reshape(self.strfs_no_border, self.numcolour)
        else:
            return pygor.utilities.multicolour_reshape(self.strfs, self.numcolour)
    
    ## Bootstrapping
    def __calc_pval_time(self, parallel = None, **kwargs) -> np.ndarray:
        """
        Calculate the p-value for each time point in the data.

        Returns:
            np.ndarray: An array of p-values for each time point.
        """
        if parallel == None:
            parallel = self.bs_settings["time_parallel"]
        # Generate bar for beauty
        bar = tqdm(self.strfs, leave = False, position = 1, disable = None, 
            desc = f"Hang on, bootstrapping pygor.strf.temporal components {self.bs_settings['time_bs_n']} times")
        self._pval_time = np.array([pygor.strf.bootstrap.bootstrap_time(x, bootstrap_n=self.bs_settings["time_bs_n"], parallel = parallel) for x in bar])
        return self._pval_time

    def __calc_pval_space(self, parallel = None, **kwargs) -> np.ndarray:
        """
        Calculate the p-value space for the spatial components.

        This function calculates the p-value space for the spatial components of the data. It performs a bootstrap
        analysis to estimate the spatial component's significance. The p-value space is an array of shape (n_strfs,)
        where n_strfs is the number of spatial frequency components in the data.

        Returns:
            np.ndarray: The p-value space array.

        """
        if parallel == None:
            parallel = self.bs_settings["space_parallel"]
        # Again, bar for niceness
        bar = tqdm(self.strfs, leave = False, position = 1, disable = None,
            desc = f"Hang on, bootstrapping spatial components {self.bs_settings['space_bs_n']} times")
        self._pval_space = np.array([pygor.strf.bootstrap.bootstrap_space(x, bootstrap_n=self.bs_settings["space_bs_n"], parallel=parallel) for x in bar])

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

    def set_bootstrap_setting(self, key : str, value) -> None:
        self.bs_settings[key] = value

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
                warning_string = "Input anomalies found"
                warnings.warn(warning_string, stacklevel=0)
                if not_allowed_keys:
                    print("The following disallowed keys were passed:", *not_allowed_keys)
                if not_found_keys:
                    print(f"Keys not found in self.bs_settings: {not_found_keys}")
                if none_default_key:
                    print(f"Keys set to default values: {[(i, default_dict[i]) for i in none_default_key]}")

    def run_bootstrap(self, force = False, parallel = None, plot_example = False) -> None:
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
                    self.__calc_pval_time(parallel=parallel)
                    self.__calc_pval_space(parallel=parallel)
                    after_time = datetime.datetime.now()
                elif user_verify == 'n' or user_verify == "no":
                    print(f"Skipping recomputing bootstrap due to user input:'{user_verify}'")
                else:
                    print(f"Input '{user_verify}' is invalid, no action done. Please use 'y'/'n'.")
            else:
                before_time = datetime.datetime.now()
                self.__calc_pval_time(parallel=parallel)
                self.__calc_pval_space(parallel=parallel)
                after_time = datetime.datetime.now()
                self.bs_settings["bs_already_ran"] = True
            # Write time metadata
            self.bs_settings["bs_datetime"] = before_time
            self.bs_settings["bs_datetime_str"] = before_time.strftime("%d/%m/%y %H:%M:%S")
            self.bs_settings["bs_dur_timedelta"] = after_time - before_time

    @property
    def pval_time(self, parallel = None) -> np.ndarray:
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
            return np.array([np.nan] * self.num_strfs)
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_time
            except AttributeError:
                print("p value for time bootstrap did not exist, just making that now")
                self.__calc_pval_time(parallel = parallel) # Executes calculation and writes to self._pval_time
            return self._pval_time
    
    @property
    def pval_space(self, parallel = None) -> np.ndarray:
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
            return np.array([np.nan] * self.num_strfs)
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_space
            except AttributeError:
                print("p value for space bootstrap did not exist, just making that now")
                self.__calc_pval_space(parallel = parallel) # Executes calculation and writes to self._pval_space
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
            space_vals = self.pval_space
            time_vals = self.pval_space
            space_sig = space_vals <  self.bs_settings["space_sig_thresh"]
            time_sig = time_vals < self.bs_settings["time_sig_thresh"]
            both_sig = time_sig * space_sig
            final_arr = np.stack((space_vals, time_vals, both_sig), dtype = "object").T
            column_labels = ["space", "time", "sig"]
            return pd.DataFrame(final_arr, columns = column_labels)
    
    @property
    def num_rois_sig(self) -> int:
        if self.multicolour == True:
            return self.get_pvals_table()["sig_any"].sum()
        if self.multicolour == False:
            return self.get_pvals_table()["sig"].sum()

    @property
    def strfs_no_border(self) -> np.ndarray:
        return pygor.utilities.auto_remove_border(self.strfs)

    def fit_contours(self, roi = None, force = True) -> np.ndarray[list[list[list[float, float]]]]:
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
        if self.bs_settings["do_bootstrap"] == True:
            time_pvals = self.pval_time
            space_pvals = self.pval_space
        
            __contours = [pygor.strf.contouring.bipolar_contour(arr) # ensures no contour is drawn if pval not sig enough
                            if time_pvals[count] < self.bs_settings["time_sig_thresh"] and space_pvals[count] < self.bs_settings["space_sig_thresh"]
                            else  (np.array([]), np.array([]))
                            for count, arr in enumerate(self.collapse_times(roi))]
        
            # __contours = []
            # for count, arr in enumerate(self.collapse_times(roi)):
            #     if time_pvals[count] < self.bs_settings["time_sig_thresh"] and space_pvals[count] < self.bs_settings["space_sig_thresh"]:
            #         print(arr.shape, "bbb")
            #         contour = pygor.strf.contouring.bipolar_contour(arr)  # ensures no contour is drawn if pval not sig enough
            #     else:
            #         contour = (np.array([]), np.array([]))
            #     __contours.append(contour)
        
        if self.bs_settings["do_bootstrap"] == False:
            
            # # Simple version without p-value checks
            # __contours = []
            # for count, arr in enumerate(self.collapse_times()):
            #     print(arr.shape, "ccc")
            #     contour = pygor.strf.contouring.bipolar_contour(arr)
            #     __contours.append(contour)

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

    def get_timecourses(self, roi = None, centre_on_zero = False, mask_empty = False, spoof_masks = False, method = "segmentation") -> np.ndarray:
        # try:
        #     return self.__timecourses 
        # except AttributeError:
        if method == "contour":
            # Get masks that we will apply (and make a copy in memory, because we are about to ammend these)
            masked_strfs = copy.deepcopy(self.get_strf_masks(roi))
            # masked_strfs = self.get_strf_masks() # <- this one overrides data, only for testing!
            # if roi is None:
            timecourses = np.ma.average(masked_strfs, axis = (-1,-2))
            # Figure out where we have empty masks
            empty_mask_indices = np.argwhere(timecourses.mask.any(axis = (-1,-2)))
            # To get comparable averages, we may want to average over the same number of pixels as the fitted masks.
            # # One strategy is to take the pre-existing fitted masks and apply them to the STRFs with no signal.
            if spoof_masks is True:
                # Generate spoofs for each ROI 
                spoofed_masks = pygor.strf.guesstimate.gen_spoof_masks(self)
                # Apply our spoofed masks to the initial extracted masks where we have no data
                masked_strfs.mask[empty_mask_indices] = spoofed_masks[empty_mask_indices]
                timecourses = np.ma.average(masked_strfs, axis = (2,3))            
            # elif mask_empty is False:
            #     # Sometimes we may/may not want to keep data where space has no correlations (all masked)
            #     # Fill timecourses with noise array where space is masked
            #     empty_mask_indices = np.argwhere(timecourses.mask.any(axis = (-1,-2)))
            #     noise_times = np.expand_dims(np.average(self.strfs[empty_mask_indices], axis = (3, 4)), axis = 1)
            #     noise_times = np.repeat(noise_times, 2, axis = 2)
            #     timecourses[empty_mask_indices] = noise_times
            # # print(timecourses.shape)
            # plt.imshow(masked_strfs[1, 0, 0])
            # plt.plot(timecourses[2].T)
            # Most of the time it makes sense to centre on zero
        if method == "segmentation":
            # More elegant, faster, and centre always 1st index (surround if any 2nd)
            _, times = self.cs_seg(roi = roi)
            # times = times[roi:,:2]
            if roi is None:
                timecourses = times[roi:, :2,]
            elif isinstance(roi, int):
                timecourses = times[:2,]
            else:
                timecourses = times[:, :2]
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

    def get_chroma_strf(self, roi = None):
        if roi is None:
            index = range(self.num_rois * self.numcolour)
            return pygor.utilities.multicolour_reshape(self.strfs[index], self.numcolour)
        else:
            if isinstance(roi, Iterable) is False:
                roi = [roi]
            # start_index = np.linspace(0, self.num_rois * self.numcolour, self.num_rois + 1)[roi].astype(int)
            # print(np.linspace(0, self.num_rois * self.numcolour, self.num_rois + 1))
            start_index = np.arange(0, self.num_rois * self.numcolour)[::self.numcolour][roi]
            end_index = start_index + self.numcolour
            indices = np.arange(start_index, end_index)
            return np.squeeze(pygor.utilities.multicolour_reshape(self.strfs[indices], self.numcolour))

    def get_timecourses_dominant(self, **kwargs):
        dominant_times = []
        for arr in self.get_timecourses(**kwargs):
            if np.max(np.abs(arr[0]) > np.max(np.abs(arr[1]))):
                dominant_times.append(arr[0])
            else:
                dominant_times.append(arr[1])
        dominant_times = np.array(dominant_times)
        return dominant_times

    def get_pix_times(self, incl_borders = False):
        if incl_borders is True:
            return np.array([np.reshape(i, (i.shape[0], -1)) for i in self.strfs])
        else:
            return np.array([np.reshape(i, (i.shape[0], -1)) for i in self.strfs_no_border])

    ## Methods__________________________________________________________________________________________________________

    # def plot_roi(self, roi):
    #     fig, ax = plt.subplots(1, 3)

    def get_strf_masks(self, roi = None, level = None, force_recompute = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Return masked array of spatial.rf_mask3d applied to all arrays, with masks based on pval_time and pval_space,
        with polarity intact.
        """
        try:
            if roi is not None or self.__strf_masks.shape[0] != self.num_rois * self.numcolour:
                force_recompute = True
        except AttributeError:
            force_recompute = True
        def set_strf_masks(roi = roi):     
                if self.strfs is np.nan:
                    self.__strf_masks = np.nan
                if roi is None:
                    roi = range(0, self.num_rois * self.numcolour)
                else:
                    if isinstance(roi, Iterable) is False:
                        roi = [roi]
                        
                # raise NotImplementedError("Implementation error, does not work yet")
                # get 2d masks
                print(self.collapse_times(roi, border = True).shape)
                # patch a deeper bug with inconsistent use of dimensionality squeezing
                # check dim
                colapse_times = self.collapse_times(roi, border = True)
                if colapse_times.ndim != 3:
                    colapse_times = np.expand_dims(colapse_times, 0)
                #print(colapse_times.shape)
                all_masks = np.array([pygor.strf.contouring.bipolar_mask(i) for i in colapse_times])
                #print(all_masks.shape)
                all_masks = np.repeat(np.expand_dims(all_masks, 2), self.strfs.shape[1], axis = 2)
                #print(all_masks.shape)
                # Apply mask to expanded and repeated strfs (to get negative and positive)
                strfs_expanded = np.repeat(np.expand_dims(self.strfs[roi], axis = 1), 2, axis = 1)
                all_strfs_masked = np.ma.array(strfs_expanded, mask = all_masks, keep_mask=True)
                self.__strf_masks = all_strfs_masked
        if force_recompute:
            set_strf_masks()
        try:
            return self.__strf_masks
        except AttributeError:
            set_strf_masks()
            return self.__strf_masks

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

    def get_spatial_masks(self, roi = None) -> tuple[np.ndarray, np.ndarray]:
        masks = self.get_strf_masks(roi).mask#[:, :, 0]
        neg_mask2d, pos_mask2d = masks[:, 0], masks[:, 1]
        #return np.array([neg_mask2d, pos_mask2d])
        return (np.squeeze(neg_mask2d), np.squeeze(pos_mask2d))

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

    def collapse_times(self, roi = None, zscore : bool = True, spatial_centre : bool = False, border : bool = False, **kwargs) -> np.ma.masked_array:
        if roi is not None:
            if isinstance(roi, int):
                iterate_through = [roi]
            elif isinstance(roi, np.int_):
                iterate_through = [roi.astype(int)]
            elif isinstance(roi, Iterable):
                iterate_through = roi
            else:
                raise ValueError("ROI must be None, int, or an iterable of ints")
        else:
            iterate_through = range(len(self.strfs))
        if border == True:
            strfs_arr = self.strfs
        if border == False:
            strfs_arr = self.strfs_no_border
        if isinstance(iterate_through, Iterable):
            target_shape = (len(iterate_through),
                            strfs_arr.shape[2], 
                            strfs_arr.shape[3])
        else:
            target_shape = (1,
                            self.strfs.shape[2],
                            self.strfs.shape[3])
        collapsed_strf_arr = np.ma.empty(target_shape)
        for n, roi in enumerate(iterate_through):
                collapsed_strf_arr[n] = pygor.strf.spatial.collapse_3d(strfs_arr[roi], zscore = zscore, **kwargs)
        if spatial_centre == True:
            # try:
            #     return self._spatial_centered_collapse
            # except AttributeError:
            # Calculate shifts required for each image (vectorised)
            arr3d = collapsed_strf_arr
            # Ideally (but does not seem to work correctly, skewed by spurrious contours)
            contours_centers = np.where(self.get_contours_centres() > 0, np.floor(self.get_contours_centres()), np.ceil(self.get_contours_centres()))
            # contours_centers = self.centre_strfs
            # contours_centers = self.get_seg_centres()
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
        return collapsed_strf_arr # do not squeeze for more predictable output
        # spatial.collapse_3d(recording.strfs[strf_num])

    def collapse_times_chroma(self, roi = None, zscore : bool = True, spatial_centre : bool = False, border : bool = False, **kwargs) -> np.ma.masked_array:
        all_collapsed = self.collapse_times(None, zscore = zscore, spatial_centre = spatial_centre, border = border, **kwargs)
        if roi is None:
            return pygor.utilities.multicolour_reshape(all_collapsed, self.numcolour)
        if roi is not None:
            return pygor.utilities.multicolour_reshape(all_collapsed, self.numcolour)[:, roi]

    def calc_spatial_correlations(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan) -> tuple[pd.DataFrame, list[str]]:
        """
        Calculate spatial correlations between all channel pairs for all ROIs.
    
        Args:
            abs_arrays (bool): If True, take absolute value of arrays before correlation.
                            If False, preserve original polarities. Default True.
            signal_only (bool): If True, only include channels that have signal for each ROI.
                            If False, include all channels. Default True.
            single_channel_value (float): Value to assign when ROI has <2 signal channels.
                            Default np.nan. Common alternative: 1.0 for perfect correlation.
    
        Returns:
            pd.DataFrame: Correlation values for all ROIs and channel pairs.
                        Columns are named as 'Ch{i}-Ch{j}' where i,j are channel indices.
                        Index includes ALL ROIs (0 to num_rois-1).
                        Uses single_channel_value for pairs where one/both channels lack signal or ROI has <2 signal channels.
                        For standard 4-channel data: Ch0=R, Ch1=G, Ch2=B, Ch3=UV.
        """
        # throw error if object is not multichromatic
        if not self.multicolour:
            raise ValueError("Object is not multichromatic, method is invalid.")
        # Calculate across all ROIs
        spaces = self.collapse_times_chroma()
        # Conditionally apply absolute value
        if abs_arrays:
            spaces_flat = np.abs(spaces.reshape(self.numcolour, self.num_rois, -1))
        else:
            spaces_flat = spaces.reshape(self.numcolour, self.num_rois, -1)
        # Get signal mask if requested
        if signal_only:
            raw_signal = self.bool_strf_signal(multicolour=False)
            signal_mask_2d = pygor.utilities.multicolour_reshape(raw_signal, self.numcolour).T
        else:
            signal_mask_2d = np.ones((self.num_rois, self.numcolour), dtype=bool)
        # Get all possible channel pairs
        pairs_indices = np.tril_indices(self.numcolour, k=-1)
        pair_names = [f"Ch{i}-Ch{j}" for i, j in zip(pairs_indices[0], pairs_indices[1])]
        # Store results for ALL ROIs
        roi_results = []
        for roi in range(self.num_rois):  # Process ALL ROIs
            # Get channels with signal for this ROI
            valid_channels = np.where(signal_mask_2d[roi])[0]
            
            # Initialize with specified value for this ROI (default NaN for missing data)
            roi_correlations = np.full(len(pair_names), single_channel_value, dtype=float)
            
            if len(valid_channels) >= 2:
                # Only compute correlations if we have at least 2 signal channels
                roi_data = spaces_flat[valid_channels, roi, :]
                corr_matrix = np.corrcoef(roi_data)
                
                # Fill in correlations for valid pairs
                for pair_idx, (ch_i, ch_j) in enumerate(zip(pairs_indices[0], pairs_indices[1])):
                    # Check if both channels have signal
                    if ch_i in valid_channels and ch_j in valid_channels:
                        # Get positions in the reduced correlation matrix
                        pos_i = np.where(valid_channels == ch_i)[0][0]
                        pos_j = np.where(valid_channels == ch_j)[0][0]
                        roi_correlations[pair_idx] = corr_matrix[pos_i, pos_j]
            
            # Always append (even if all NaN)
            roi_results.append(roi_correlations)
        assert len(roi_results) == self.num_rois
        # Create DataFrame with ALL ROI indices
        full_df = pd.DataFrame(roi_results, columns=pair_names, index=range(self.num_rois))
        
        return full_df, pair_names

    def spatial_overlap_index_stats(self, abs_arrays=True, signal_only=True, force_recompute=False, single_channel_value=np.nan):
        """
        Calculate spatial correlation statistics for each ROI individually.
    
        Args:
            abs_arrays (bool): If True, take absolute value of arrays before correlation.
                            If False, preserve original polarities. Default True.
            signal_only (bool): If True, only include ROIs that are considered signal ROIs.
                            If False, include all ROIs. Default True.
            force_recompute (bool): If True, force recomputation even if cached. Default False.
            single_channel_value (float): Value to assign when ROI has <2 signal channels.
                            Default np.nan. Common alternative: 1.0 for perfect correlation.
    
        Returns:
            pd.DataFrame: For each ROI, the mean, std, and min of correlations across all channel pairs.
                        Index is ROI number (original indices if signal_only=True).
                        Columns are 'mean_corr', 'std_corr', 'min_corr'.
                        For 4-channel data: Ch0=R, Ch1=G, Ch2=B, Ch3=UV.
        """
        # Create cache key
        cache_key = (abs_arrays, signal_only, single_channel_value)
        
        # Check cache first (unless force recompute)
        if not force_recompute and cache_key in self._spatial_overlap_cache:
            return self._spatial_overlap_cache[cache_key]
        
        # Compute correlations
        all_corr_pairs, pair_names = self.calc_spatial_correlations(abs_arrays=abs_arrays, signal_only=signal_only, single_channel_value=single_channel_value)
    
        # Compute statistics for each ROI (across the channel pairs)
        mean_corr_per_roi = np.mean(all_corr_pairs, axis=1)
        var_corr_per_roi = np.var(all_corr_pairs, axis=1)
        std_corr_per_roi = np.std(all_corr_pairs, axis=1)
        min_corr_per_roi = np.min(all_corr_pairs, axis=1)
    
        # Create summary DataFrame - index already set from calc_spatial_correlations
        summary_df = pd.DataFrame({
            'mean_corr': mean_corr_per_roi,
            'std_corr': std_corr_per_roi,
            'min_corr': min_corr_per_roi,
            'var_corr': var_corr_per_roi
        }, index=all_corr_pairs.index)
        
        # Cache the result
        self._spatial_overlap_cache[cache_key] = summary_df
        
        return summary_df

    def spatial_overlap_index_mean(self, abs_arrays=True, signal_only=True, force_recompute=False, single_channel_value=np.nan):
        return self.spatial_overlap_index_stats(abs_arrays=abs_arrays, signal_only=signal_only, force_recompute=force_recompute, single_channel_value=single_channel_value)["mean_corr"].to_numpy()

    def spatial_overlap_index_std(self, abs_arrays=True, signal_only=True, force_recompute=False, single_channel_value=np.nan):
        return self.spatial_overlap_index_stats(abs_arrays=abs_arrays, signal_only=signal_only, force_recompute=force_recompute, single_channel_value=single_channel_value)["std_corr"].to_numpy()

    def spatial_overlap_index_min(self, abs_arrays=True, signal_only=True, force_recompute=False, single_channel_value=np.nan):
        return self.spatial_overlap_index_stats(abs_arrays=abs_arrays, signal_only=signal_only, force_recompute=force_recompute, single_channel_value=single_channel_value)["min_corr"].to_numpy()
    
    def spatial_overlap_index_var(self, abs_arrays=True, signal_only=True, force_recompute=False, single_channel_value=np.nan):
        return self.spatial_overlap_index_stats(abs_arrays=abs_arrays, signal_only=signal_only, force_recompute=force_recompute, single_channel_value=single_channel_value)["var_corr"].to_numpy()

    def spatial_overlap_channel_pair(self, ch1, ch2, abs_arrays=True, 
    signal_only=True, single_channel_value=np.nan):
        """Get spatial overlap index for a specific channel pair."""
        all_corr_pairs, pair_names  =  self.calc_spatial_correlations(abs_arrays=abs_arrays, signal_only=signal_only, single_channel_value=single_channel_value)
        # Use the same naming convention as calc_spatial_correlations: larger index first
        pair_name = f"Ch{max(ch1,ch2)}-Ch{min(ch1,ch2)}"
        if pair_name not in all_corr_pairs.columns:
            raise ValueError(f"Channel pair {pair_name} not found. Available: {list(all_corr_pairs.columns)}")
        return all_corr_pairs[pair_name].to_numpy()

    # Convenience methods for specific pairs
    def spatial_overlap_red_uv(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Red-UV spatial overlap (Ch0-Ch3) - strongest anti-correlation expected"""        
        return self.spatial_overlap_channel_pair(0, 3, abs_arrays, signal_only, single_channel_value)

    def spatial_overlap_green_blue(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Green-Blue spatial overlap (Ch1-Ch2)"""
        return self.spatial_overlap_channel_pair(1, 2, abs_arrays, signal_only, single_channel_value)

    def spatial_overlap_red_green(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Red-Green spatial overlap (Ch0-Ch1)"""
        return self.spatial_overlap_channel_pair(0, 1, abs_arrays, signal_only, single_channel_value)

    def get_polarities(self, roi = None, exclude_FirstLast=(1,1), mode = "cs_pol", force_recompute = False) -> np.ndarray:
        if mode == "old":
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
        if mode == "new":
            # Get the time as absolute values, then get the max value
            abs_time_max = np.max(np.abs(self.get_timecourses(mask_empty = True).data), axis = 2)
            times_peak = pygor.np_ext.maxabs(self.get_timecourses(mask_empty = True), axis = 2)
            # First find the obvious polarities
            pols = np.where(times_peak[:, 0] < 0, -1, 1)
            # If values are 0, we assign 0
            zero_bools = np.all(times_peak == 0, axis = 1)
            pols = np.where(zero_bools == True, np.nan, pols)    
        if mode =="cs_pol" or mode == "cs_pol_extra": 
            _, prediction_times_ROIs = self.cs_seg(force_recompute = force_recompute)
            covar_thresh = -.5
            var_thresh = .2
            S_absamp_thresh = 1.5
            center_dominance_ratio = 2.0  # Center must be 2x stronger than surround for simple ON/OFF

            C_times = prediction_times_ROIs[:, 0, :]
            S_times = prediction_times_ROIs[:, 1, :]
            C_centered = C_times - C_times.mean(axis=1, keepdims=True)
            S_centered = S_times - S_times.mean(axis=1, keepdims=True)

            # Compute covariance for each pair
            CS_covariances = np.sum(C_centered * S_centered, axis=1) / (C_times.shape[1] - 1)

            # Get absolute max for each value of C and S
            C_maxabs = np.abs(pygor.np_ext.maxabs(C_times, axis=1))
            S_maxabs = np.abs(pygor.np_ext.maxabs(S_times, axis=1))

            # Get signs based on sustained response using single-pass adaptive approach
            def get_sustained_polarity_single_pass(timecourse):
                n = len(timecourse)
                if n < 3:
                    return 0  # Too short to analyze
                
                # Single pass: find max absolute value AND track when it occurs
                max_abs_val = 0
                max_abs_val_with_sign = 0
                max_abs_idx = 0
                
                for i, val in enumerate(timecourse):
                    abs_val = abs(val)
                    if abs_val > max_abs_val:
                        max_abs_val = abs_val
                        max_abs_val_with_sign = val
                        max_abs_idx = i
                
                # If max occurs early (first 25%), look for sustained response
                if max_abs_idx < n // 4:
                    # Find the strongest response in the remaining 75%
                    sustained_max = 0
                    sustained_max_with_sign = 0
                    
                    for i in range(n // 4, n):
                        abs_val = abs(timecourse[i])
                        if abs_val > sustained_max:
                            sustained_max = abs_val
                            sustained_max_with_sign = timecourse[i]
                    
                    # Use sustained response if it's substantial (>30% of peak)
                    if sustained_max > 0.3 * max_abs_val:
                        return sustained_max_with_sign
                
                # Otherwise use the global maximum
                return max_abs_val_with_sign
            
            C_sustained_response = np.array([get_sustained_polarity_single_pass(c) for c in C_times])
            C_signs = np.sign(C_sustained_response)
            
            # Initialize categories
            cat = np.where(C_signs > 0, 1, -1)
            cat = cat.astype("float")
            zerovals = C_signs == 0
            cat[zerovals] = np.nan
            
            # Only classify as center-surround if ALL criteria are met AND center doesn't dominate
            amplitude_pass_idx = S_maxabs > S_absamp_thresh
            var_pass_idx = np.var(S_times, axis=1) > var_thresh
            covariance_pass_idx = CS_covariances < covar_thresh
            center_not_dominant = C_maxabs <= center_dominance_ratio * S_maxabs
            
            # More stringent CS criteria - must pass all conditions
            cs_pass_bool = (amplitude_pass_idx & var_pass_idx & 
                           covariance_pass_idx & center_not_dominant)
            
            # Only assign center-surround if genuinely meets criteria
            cat = np.where(cs_pass_bool, 2, cat)
            
            # Set to NaN if center amplitude is too weak for reliable classification
            weak_center = C_maxabs < 1.0  # Minimum amplitude threshold
            cat = np.where(weak_center, np.nan, cat)
            
            # Extra check for strong vs weak CS (if requested)
            if mode == "cs_pol_extra":
                with np.errstate(divide='ignore', invalid='ignore'):
                    lower_bound = .5
                    upper_bound = 2
                    condition1 = (np.abs(C_maxabs / S_maxabs) >= lower_bound) & (np.abs(C_maxabs / S_maxabs) <= upper_bound)                
                cs_strong_mask = (cat == 2) & condition1
                cat = np.where(cs_strong_mask, 3, cat)  # Use 3 for strong CS if needed
            
            pols = cat
        elif mode == "on_off_gabor":
            # First separate out to ON and OFF
            spaces = self.collapse_times()
            space_min = np.min(spaces, axis=(1, 2))
            space_max = np.max(spaces, axis=(1, 2))
            cat = np.where(np.abs(space_min) < space_max, 1, -1)
            thresh = 5
            min_below_thresh = np.abs(space_min) < thresh
            max_below_thresh = np.abs(space_max) < thresh
            no_signal = np.bitwise_and(min_below_thresh, max_below_thresh)
            cat = cat.astype(float)
            cat[no_signal] = np.nan
            # Then separate out "gabor" cells 
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     # condition1 = np.isclose(np.abs(C_maxabs/S_maxabs), -1, atol=1e1)
            #     lower_bound = .5
            #     upper_bound = 3
            #     condition1 = (np.abs(C_maxabs / S_maxabs) >= lower_bound) & (np.abs(C_maxabs / S_maxabs) <= upper_bound)                
            condition2 =  np.bitwise_and(space_min < -thresh, space_max > thresh)
            cat = np.where(condition2, 2, cat)
            pols = cat
        return pols

    def get_opponency_bool(self) -> bool:
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

    def compute_average_spaces(self):
        spaces = self.collapse_times()
        spaces = pygor.utilities.multicolour_reshape(spaces, self.numcolour)
        spaces = np.average(spaces, axis = 0)
        return spaces

    def check_space_average_gabor(self, ampl_thresh = 2):
        spaces = self.compute_average_spaces()
        maxes = np.max(spaces, axis = (1, 2))
        mins = np.min(spaces, axis = (1, 2))
        condition1 = maxes > ampl_thresh
        condition2 = mins < -ampl_thresh
        return np.bitwise_and(condition1, condition2)
    
    def calc_balance_ratio(self, mode = None):
        if mode == None or mode == "all":
            arrs = self.collapse_times()    
        if mode == "white":
            arrs = self.compute_average_spaces()
        return pygor.strf.spatial.snr_gated_balance_ratio(arrs)
        
    def calc_spatial_opponency(self, mode = None):
        if mode == None or mode == "all":
            arrs = self.collapse_times()    
        if mode == "white":
            arrs = self.compute_average_spaces()
        return pygor.strf.spatial.snr_gated_spatial_opponency(arrs)
        
    def calc_centre_distances(self, mode = "cs_seg"):
        if mode == "cs_seg":
            cell_centres = np.nanmean(self.get_seg_centres(), axis = 1, keepdims = True)
            rf_centres = self.get_seg_centres()
            euclidian_dist = np.sqrt(np.sum((cell_centres - rf_centres)**2, axis = 2))
            euclidian_dist = euclidian_dist * self.stim_size
            # pygor.strf.unit_conversion
            return euclidian_dist
        else:
            raise NotImplementedError

    def get_polarity_labels(self, mode = "on_off_gabor"):
        pols = self.get_polarities(mode = mode)
        pols_out = pols.astype(str)
        pols_out[pols == 1] = "ON"
        pols_out[pols == -1] = "OFF"
        pols_out[pols == 2] = "Gabor"
        pols_out[pols == np.nan] = "NaN"
        return pols_out

    def get_polarity_category_cell(self, mask_by_channel=True, threshold=2, dimstr="time") -> str:
        """
        Get polarity category for each cell across color channels.
        
        Parameters
        ----------
        mask_by_channel : bool, optional
            Whether to mask polarities using self.bool_by_channel (default: True)
        threshold : float, optional
            Threshold for bool_by_channel masking (default: 2)
        dimstr : str, optional
            Dimension for bool_by_channel ('time' or 'space', default: 'time')
            
        Returns
        -------
        list of str
            Polarity categories for each cell: 'empty', 'on', 'off', 'opp', 'mix', 'other'
        """
        result = []
        polarities = self.get_polarities()
        arr = pygor.utilities.multicolour_reshape(polarities, self.numcolour).T
        
        if mask_by_channel:
            # Get boolean mask for significant channels
            bool_mask = self.bool_by_channel(threshold=threshold, dimstr=dimstr)
            # Apply mask to polarities - set insignificant channels to NaN
            for i, (pol_row, mask_row) in enumerate(zip(arr, bool_mask)):
                arr[i] = np.where(mask_row, pol_row, np.nan)
        
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

    def get_time_amps(self, **kwargs) -> np.ndarray:
        maxes = np.max(self.get_timecourses_dominant(**kwargs).data, axis = (1))
        mins = np.min(self.get_timecourses_dominant(**kwargs).data, axis = (1))
        largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
        return largest_mag

    def get_space_amps(self) -> np.ndarray:
        maxes = np.max(self.collapse_times(), axis = (1, 2))
        mins =  np.min(self.collapse_times(), axis = (1, 2))
        largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
        return largest_mag

    def get_time_to_peak(self, dur_s = 1.3) -> np.ndarray:
        # First get timecourses
        # Split by polarity 
        neg_times, pos_times = self.get_timecourses()[:, 0], self.get_timecourses()[:, 1]
        # Find max position in pos times and neg position in neg times 
        argmins = np.ma.argmin(neg_times, axis = 1)
        argmaxs = np.ma.argmax(pos_times, axis = 1) 
        if dur_s != None:
            return  (dur_s / neg_times.shape[1]) * np.array([argmins, argmaxs])
        else:
            warnings.warn("Time values are in arbitary numbers (frames)")
            return np.array([argmins, argmaxs])

    """
    TODO get these calc_thigny methods to have their own per-ROI equivalents
    that can then simply be reshaped to tuning function format. 
    """
    # def get_time_to_peak(self):


    def calc_tunings_amplitude(self,dimstr = "time", treshhold = 2, **kwargs) -> np.ndarray:
        if self.multicolour == True:
            if dimstr == "time":
                largest_by_colour = pygor.utilities.multicolour_reshape(self.get_time_amps(**kwargs), self.numcolour)
                tuning_functions = largest_by_colour
            elif dimstr == "space":
                largest_by_colour = pygor.utilities.multicolour_reshape(self.get_space_amps(**kwargs), self.numcolour)
                tuning_functions = largest_by_colour
            else:
                raise ValueError("dimstr must be 'time' or 'space'")
            if treshhold != None:
                tuning_functions = np.where(np.abs(tuning_functions) > treshhold, tuning_functions, np.nan)
            # Returns wavelengths according to order in self.strfs, invert order for UV - R by wavelength (increasing)
            return tuning_functions.T #transpose for simplicity
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour.' is False")

    def bool_strf_signal(self, threshold = 2, multicolour = True, dimstr = "time") -> np.ndarray:
        if multicolour is True:
            # Get tuning amplitudes and check if at least one per ROI is above threshold
            tuning_amps = self.calc_tunings_amplitude(dimstr = dimstr)
            return np.any(np.abs(tuning_amps) > threshold, axis = 1)
        if multicolour is False:
            if dimstr == "time":
                amps = self.get_time_amps()
            elif dimstr == "space":
                amps = self.get_space_amps()
            return np.abs(amps) > threshold

    def bool_by_channel(self, threshold = 2, dimstr = "time") -> np.ndarray:
        if dimstr == "time":
            amps = self.get_time_amps()
        elif dimstr == "space":
            amps = self.get_space_amps()
        else:
            raise ValueError("dimstr must be 'time' or 'space'")
        return pygor.utilities.multicolour_reshape(np.abs(amps), self.numcolour).T > threshold

    def calc_mean_absolute_deviation(self, dimstr = "space", **kwargs):
        tunings = self.calc_tunings_amplitude(dimstr = dimstr, **kwargs)
        mad = np.mean(np.abs(tunings - np.mean(tunings, axis = 1, keepdims=True)), axis = 1)
        mad = np.where(mad == 0, np.inf, mad)
        return mad

    def calc_tunings_area(self, size = None, upscale_factor = 4, largest_only = True) -> np.ndarray:
        if self.multicolour == True:
            # Step 1: Pull contour areas (note, the size is in A.U. for now)
            # Step 2: Split by positive and negative areas
            if size == None:
                warnings.warn("size stimates in arbitrary Cartesian units")
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
            peaktimes = self.get_time_to_peak()
            peakneg = peaktimes[0]
            peakpos = peaktimes[1]
            peakneg  = pygor.utilities.multicolour_reshape(peakneg, self.numcolour).T
            peakpos  = pygor.utilities.multicolour_reshape(peakpos, self.numcolour).T
            return np.array([peakneg, peakpos])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def calc_spectral_centroids(self) -> tuple[np.ndarray, np.ndarray]:
        spectroids_neg = np.apply_along_axis(pygor.strf.temporal.only_centroid, 1, self.get_timecourses()[:, 0])
        spectroids_pos = np.apply_along_axis(pygor.strf.temporal.only_centroid, 1, self.get_timecourses()[:, 1])
        return spectroids_neg, spectroids_pos

    def calc_spectrums(self, roibyroi = False) -> tuple[np.ndarray, np.ndarray]:
        spectrum_neg = np.array([pygor.strf.temporal.only_spectrum(i) for i in self.get_timecourses()[:, 0]])
        spectrum_pos = np.array([pygor.strf.temporal.only_spectrum(i) for i in self.get_timecourses()[:, 1]])
        return spectrum_neg, spectrum_pos

    def unravel_chroma_roi(strf_obj, roi, chroma_index, multichroma_dim = 0, roi_dim = 1):
        return np.ravel_multi_index([roi, chroma_index], np.array(strf_obj.strfs_chroma.shape)[[roi_dim, multichroma_dim]])

    def demo_contouring(self, roi, chromatic_reshape = False):
        plt.close()
        fig, ax = plt.subplots(2, 6, figsize = (16*1, 4*1))
        neg_c, pos_c = pygor.strf.contouring.bipolar_contour(self.collapse_times(roi), plot_results= True, ax = ax)
        ax[0, -1].plot(np.squeeze(self.get_timecourses(roi)).T, label = ["neg", "pos"])
        ax[0, -1].legend()
        spatial_mask = np.array(self.get_spatial_masks(roi))[:, 0]
        print(spatial_mask.shape)
        ax[1, -1].imshow((spatial_mask[0] * -1) + spatial_mask[1], origin = "lower")
        gs = ax[1, 2].get_gridspec()
        for a in ax[0:, 0]:
            a.remove()
        axbig = fig.add_subplot(gs[:, 0])
        axbig.imshow(np.squeeze(self.collapse_times(roi)), origin = "lower")
        for i in neg_c:
            axbig.plot(i[:, 1], i[:, 0], color = "blue")
        for i in pos_c:
            axbig.plot(i[:, 1], i[:, 0], color = "red")
        pygor.strf.contouring.bipolar_contour(self.collapse_times(roi), result_plot=True, ax = ax[:, 1:5])
        # for zax in ax[0:2, 1:5].flat:
        #     zax.imshow(np.random.rand(10, 10), origin = "lower")
        plt.show()

    def plot_timecourse(self, roi):
        plt.plot(self.get_timecourses()[roi].T)

    def plot_space(self, roi = None, ax = None, **kwargs):
        space = self.collapse_times(roi)
        maxabs = np.max(np.abs(space))
        if "clim" not in kwargs:
            kwargs["clim"] = [-maxabs, maxabs]
        if ax is None:
            plt.imshow(np.squeeze(space), origin = "lower", cmap = "Greys_r", **kwargs)
            plt.colorbar()
        else:
            ax.imshow(np.squeeze(space), origin = "lower", cmap = "Greys_r", **kwargs)
            plt.colorbar(ax.images[0], ax=ax, orientation="vertical")

    def plot_strfs_space(self, roi = None, **kwargs): 
        return pygor.strf.plotting.simple.plot_collapsed_strfs(self, **kwargs)

    def plot_chromatic_overview(self, roi = None, contours = False, with_times = False, colour_idx=None, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            return pygor.strf.plotting.advanced.chroma_overview(self, roi, contours=contours, with_times=with_times, colour_idx=colour_idx, **kwargs)

    def play_strf(self, roi, **kwargs):
        if isinstance(roi, tuple):
            chroma_arr = pygor.utilities.multicolour_reshape(
                self.strfs, self.numcolour)
            use_map = pygor.plotting.maps_concat[roi[0]]
            anim = pygor.plotting.play_movie(chroma_arr[roi], cmap = use_map,**kwargs)
        else:
            anim = pygor.plotting.play_movie(np.squeeze(self.strfs[roi]), **kwargs)
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

    def to_rgb(self, roi = None, rgb_channels = [0, 1, 3], bgr = True, plot = False, gamma = None, method = "abs",
            remove_borders = None, **kwargs):
        # input_arr = np.squeeze(pygor.utilities.multicolour_reshape(self.collapse_times(), self.numcolour))[:, roi]
        def _fetch_data(roi):
            # start_index = roi + (roi * self.numcolour)
            start_index = roi * self.numcolour
            end_index   = start_index + self.numcolour
            return np.squeeze(pygor.utilities.multicolour_reshape(self.collapse_times(range(start_index, end_index)), self.numcolour))

        def _run(self, roiinput, rgb_channels = rgb_channels):
            arr = _fetch_data(roiinput)
            norm_arr = pygor.utilities.scale_by(arr, method=method)
        
            # Only extract channels after normalising
            norm_arr = norm_arr[rgb_channels]
            if remove_borders is True:
                norm_arr = pygor.utilities.auto_remove_border(norm_arr)
            if bgr is True:
                norm_arr = np.rollaxis(norm_arr, axis=0, start=3)
            if gamma is not None:
                norm_arr = np.power(norm_arr / norm_arr.max(), gamma) * norm_arr.max()
            return norm_arr
        if plot is True:
            if isinstance(roi, Iterable) is True:
                raise AttributeError("roi must equal an int for plot to work")
            input_arr = _fetch_data(roi)
            rgb_arr = _run(self, roi, rgb_channels = [0, 1, 2])
            rgu_arr = _run(self, roi, rgb_channels = [0, 1, 3])
            fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi = 300)
            axs.flat[0].plot(input_arr.reshape(-1, 1), label = "Input all channels")
            axs.flat[0].legend()
            axs.flat[1].plot(rgu_arr.reshape(-1, 1, order = "f"), alpha = 1, label = "RGU")
            axs.flat[1].plot(rgb_arr.reshape(-1, 1, order = "f"), alpha = 0.5, label = "RGB")
            axs.flat[1].legend()
            axs.flat[2].imshow(rgb_arr)
            axs.flat[3].imshow(rgu_arr)
            plt.show()
            return None
        if roi is None:
            return np.array([_run(self, i) for i in range(self.num_rois)])
        else:
            if isinstance(roi, Iterable) is False:
                return _run(self, roi)
            else:
                return np.array([_run(self, i) for i in roi])
    
    def play_strf_rgb(self, roi, channel = "All", method = "grey_centered", plot = True,**kwargs):
        ## Generate RGB movie
        if channel == "All":
            num_channels = self.numcolour
            channel = np.array([[i for i in range(num_channels)]] * int(num_channels - 2))
            channel = np.array([i[[0, 1, n+2]] for n, i in enumerate(channel)])
            arr = np.array([np.moveaxis(self.get_chroma_strf(roi)[i], [0, 1], [-1, 0]) for i in channel])
            # arr = np.hstack(arr)
        if isinstance(channel, np.ndarray) is False:
            channel = np.array(channel)
        # print(channel.shape)
        if len(channel.shape) == 1:
            arr = np.expand_dims(np.array(np.moveaxis(self.get_chroma_strf(roi)[channel], [0, 1], [-1, 0])), 0)
        else:
            arr = np.array([np.moveaxis(self.get_chroma_strf(roi)[i], [0, 1], [-1, 0]) for i in channel])
        arr = np.dstack(arr)
        arr = pygor.utilities.scale_by(arr, method = method)
        if plot is True:
            return pygor.plotting.play_movie(arr, rgb_repr=True)
        else:
            return arr

    # def cs_seg(self, roi = None, **kwargs):
    #     if roi is None:
    #         maps = []
    #         times = []
    #         for i in range(self.num_rois): 
    #             map, time = pygor.strf.centsurr.run(self.strfs_no_border[i], **kwargs)
    #             maps.append(map)
    #             times.append(time)
    #         return np.array(maps), np.array(times)
    #     return pygor.strf.centsurr.run(self.strfs_no_border[roi], **kwargs)

    # def cs_seg(self, roi = None, plot = False, **kwargs):
    #     return pygor.strf.centsurr.run_object(self, roi, plot = plot, **kwargs)

    def get_colour_coefvar(self, channels = None):
        tunings = np.abs(self.calc_tunings_amplitude()[:, channels])
        tunings = np.nan_to_num(tunings)
        def coefficient_of_variation(arr):
            arr = np.array(arr)
            if np.all(arr == 0):
                return 0
            return np.std(arr) / (np.mean(arr) + 1e-10)
        index = [coefficient_of_variation(i) for i in tunings]
        return index

    def get_colour_coefvar_raw(self, channels = None):
        """
        Crude coefficient of variation calculation that replicates the original validation script method.
        Uses raw spatiotemporal data and takes max(abs(all_values)) per color per ROI.
        This method is for comparison purposes to understand differences with the principled approach.
        """
        if channels is None:
            channels = np.arange(self.numcolour)
        
        # Get the raw spatiotemporal data (similar to chroma_times in validation script)
        chroma_times = pygor.utilities.multicolour_reshape(self.get_pix_times(), self.numcolour)
        
        cv_values = []
        for roi in range(min(self.num_rois, chroma_times.shape[1])):
            # Calculate max absolute response per color for this ROI (crude method)
            roi_amplitudes = []
            for c in channels:
                if c < chroma_times.shape[0]:
                    # Take max absolute value across all time and space for this color/ROI
                    max_abs_response = np.max(np.abs(chroma_times[c, roi]))
                    roi_amplitudes.append(max_abs_response)
            
            if len(roi_amplitudes) >= 2:
                roi_amplitudes = np.array(roi_amplitudes)
                if np.all(roi_amplitudes == 0):
                    cv = 0.0
                else:
                    cv = np.std(roi_amplitudes) / (np.mean(roi_amplitudes) + 1e-10)
                cv_values.append(cv)
            else:
                cv_values.append(0.0)
                
        return cv_values

    def get_colour_sparseness_index(self, channels = None):
        if channels is None:
            channels = np.arange(self.numcolour)
        tunings = np.abs(self.calc_tunings_amplitude()[:, channels])
        tunings = np.nan_to_num(tunings)
        def selectivity_sparseness(arr):
            arr = np.array(arr)
            if np.all(arr == 0):
                return 0
            mean_r = np.mean(arr)
            mean_r2 = np.mean(arr**2)
            n = len(arr)
            if mean_r2 == 0:
                return 0
            return (1 - (mean_r**2 / mean_r2)) / (1 - 1/n)
        index = [selectivity_sparseness(i) for i in tunings]
        return index

    def cs_seg(self, roi = None, plot = False, force_recompute = False, **kwargs):
        if plot is False:
            if force_recompute is True:
                maps, times =  pygor.strf.centsurr.run_object(self, **kwargs)
                self._cs_seg_results_times = times
                self._cs_seg_results_maps = maps
                return np.squeeze(self._cs_seg_results_maps[roi]), np.squeeze(self._cs_seg_results_times[roi])
            try:
                return np.squeeze(self._cs_seg_results_maps[roi]), np.squeeze(self._cs_seg_results_times[roi])
            except AttributeError:
                maps, times =  pygor.strf.centsurr.run_object(self, **kwargs)
                self._cs_seg_results_times = times
                self._cs_seg_results_maps = maps
                return np.squeeze(self._cs_seg_results_maps[roi]), np.squeeze(self._cs_seg_results_times[roi])
        else:
            print(kwargs)
            return pygor.strf.centsurr.run_object(self, roi, plot = plot, **kwargs)

    def get_seg_centres(self, roi = None, weighted = True, channel_reshape = True,**kwargs):
        cmaps, _ = self.cs_seg(**kwargs)
        centres_list = []
        for n, segmap in enumerate(cmaps):
            centre_coords = np.argwhere(segmap == 0)
            # If empty, return nan
            if len(np.unique(segmap)) == 1:
                centres_list.append(np.array([np.nan, np.nan]))
            else:
                if weighted is True:
                    weights = np.abs(np.squeeze(self.collapse_times(n))[centre_coords[:, 0], centre_coords[:, 1]])**2
                else:
                    weights = None
                centre = np.average(centre_coords, axis = 0, weights = weights)
                centres_list.append(centre)
        centres = np.array(centres_list)
        if channel_reshape is True:
            # print(centres.shape)
            centres = np.reshape(centres, (-1, self.numcolour, 2))
        return np.squeeze(centres[roi])

    def demo_cs_seg(self, roi):
        _ = self.cs_seg(roi, plot = True, segmentation_params = {"plot_demo": True})
        plt.show()

    def convolve_with_img(self, roi, img = "example", plot = False, xrange = None, auto_crop = True, auto_crop_thresh = 3, yrange = None, **kwargs):
        from pygor.strf import convolve
        if roi is None:
            raise ValueError("ROI must be supplied")
        return convolve.convolve_image(self, roi, img, plot = plot, auto_crop = auto_crop, auto_crop_thresh = auto_crop_thresh, xrange = xrange, yrange = yrange, **kwargs)

    def map_extrema_timing(self, roi=None, threshold=3.0, 
                          exclude_firstlast=(1, 1), return_milliseconds=False, frame_rate_hz=15.625):
        """
        Map the timing of extrema for each pixel in STRF.
        
        Parameters
        ----------
        roi : int, optional
            STRF index. If None, returns results for all STRFs.
        threshold : float, optional
            Threshold in standard deviations. Default is 3.0.
        exclude_firstlast : tuple of int, optional
            Number of time points to exclude from beginning and end. Default is (1, 1).
        return_milliseconds : bool, optional
            If True, convert timing to milliseconds. Default is False.
        frame_rate_hz : float, optional
            Frame rate for millisecond conversion. Default is 60.0.
            
        Returns
        -------
        timing_maps : ndarray
            For single STRF: 2D array (y, x).
            For all STRFs: 3D array (n_strfs, y, x).
            Values are time indices of extrema (or milliseconds if return_milliseconds=True).
            NaN indicates pixels below threshold.
            
        Notes
        -----
        For multicolor data, STRF indices are organized as [roi0_color0, roi0_color1, ..., roi1_color0, ...].
        Use multicolour_reshape() or manual indexing to organize by color channels if needed.
        """
        import pygor.strf.extrema_timing as extrema_timing
        
        return extrema_timing.map_spectral_centroid_wrapper(
            self, roi=roi, exclude_firstlast=exclude_firstlast,
            return_milliseconds=return_milliseconds, frame_rate_hz=frame_rate_hz
        )

    def compare_color_channel_timing(self, roi, color_channels=(0, 1), threshold=3.0,
                                   exclude_firstlast=(1, 1), return_milliseconds=False, 
                                   frame_rate_hz=60.0):
        """
        Compare extrema timing between different color channels for a single ROI.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        color_channels : tuple of int, optional
            Two color channel indices to compare. Default is (0, 1).
        threshold : float, optional
            Threshold in standard deviations. Default is 3.0.
        exclude_firstlast : tuple of int, optional
            Number of time points to exclude from beginning and end. Default is (1, 1).
        return_milliseconds : bool, optional
            If True, convert timing to milliseconds. Default is False.
        frame_rate_hz : float, optional
            Frame rate for millisecond conversion. Default is 60.0.
            
        Returns
        -------
        timing_difference : ndarray
            2D array (y, x) of timing differences (channel2 - channel1).
            NaN where either channel is below threshold.
        """
        import pygor.strf.extrema_timing as extrema_timing
        
        return extrema_timing.compare_color_channel_timing_wrapper(
            self, roi=roi, color_channels=color_channels, threshold=threshold,
            exclude_firstlast=exclude_firstlast, return_milliseconds=return_milliseconds,
            frame_rate_hz=frame_rate_hz
        )

    def analyze_spatial_alignment(self, roi, threshold=3.0, reference_channel=0, 
                                collapse_method='peak'):
        """
        Analyze spatial alignment across all color channels for a single ROI.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        threshold : float, optional
            Threshold for defining active regions. Default is 3.0.
        reference_channel : int, optional
            Color channel to use as reference (0-indexed). Default is 0.
        collapse_method : str, optional
            Method for collapsing time dimension. Options: 'peak', 'std', 'sum'. Default is 'peak'.
        
        Returns
        -------
        dict : Dictionary containing comprehensive alignment analysis
            - 'correlation_matrix': n_colors × n_colors spatial correlation matrix
            - 'overlap_matrix': n_colors × n_colors Jaccard index matrix
            - 'distance_matrix': n_colors × n_colors centroid distance matrix
            - 'summary_stats': Summary statistics across all channel pairs
            - 'channel_centroids': Centroids for each color channel
            - 'spatial_maps': 2D spatial maps for each color channel
            - 'pairwise_metrics': Detailed pairwise comparison dictionary
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.analyze_multicolor_spatial_alignment(
            self, roi=roi, threshold=threshold, reference_channel=reference_channel,
            collapse_method=collapse_method
        )

    def compute_color_channel_overlap(self, roi, color_channels=(0, 1), threshold=3.0, 
                                    collapse_method='peak'):
        """
        Compute spatial overlap metrics between two specific color channels.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        color_channels : tuple of int, optional
            Two color channel indices to compare. Default is (0, 1).
        threshold : float, optional
            Threshold for defining active regions. Default is 3.0.
        collapse_method : str, optional
            Method for collapsing time dimension. Options: 'peak', 'std', 'sum'. Default is 'peak'.
        
        Returns
        -------
        dict : Dictionary containing spatial overlap metrics
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.compute_color_channel_overlap_wrapper(
            self, roi=roi, color_channels=color_channels, threshold=threshold,
            collapse_method=collapse_method
        )

    def compute_spatial_offset_between_channels(self, roi, color_channels=(0, 1), 
                                              threshold=3.0, method='centroid', 
                                              collapse_method='peak'):
        """
        Compute spatial offset between two color channels.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        color_channels : tuple of int, optional
            Two color channel indices to compare. Default is (0, 1).
        threshold : float, optional
            Threshold for defining active regions. Default is 3.0.
        method : str, optional
            Method for computing offset. Options: 'centroid', 'peak', 'cross_correlation'.
            Default is 'centroid'.
        collapse_method : str, optional
            Method for collapsing time dimension. Options: 'peak', 'std', 'sum'. Default is 'peak'.
        
        Returns
        -------
        dict : Dictionary containing offset measurements
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.compute_spatial_offset_between_channels(
            self, roi=roi, color_channels=color_channels, threshold=threshold,
            method=method, collapse_method=collapse_method
        )

    def plot_spatial_alignment(self, roi, threshold=3.0, reference_channel=0, 
                             collapse_method='peak', figsize=(15, 10)):
        """
        Plot spatial alignment visualization for a ROI.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        threshold : float, optional
            Threshold for defining active regions. Default is 3.0.
        reference_channel : int, optional
            Color channel to use as reference. Default is 0.
        collapse_method : str, optional
            Method for collapsing time dimension. Default is 'peak'.
        figsize : tuple, optional
            Figure size. Default is (15, 10).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.plot_spatial_alignment_wrapper(
            self, roi=roi, threshold=threshold, reference_channel=reference_channel,
            collapse_method=collapse_method, figsize=figsize
        )

    def calculate_strf(self, noise_array, sta_past_window=2.0, sta_future_window=2.0, 
                    n_colours=1, n_triggers_per_colour=None, edge_crop=2,
                    max_frames_per_trigger=8, event_sd_threshold=2.0, 
                    use_znorm=True, adjust_by_polarity=True, 
                    skip_first_triggers=0, skip_last_triggers=0,
                    pre_smooth=0, roi=None, normalize_strfs=True, verbose=True, **kwargs):
        """
        Calculate spike-triggered averages (STRFs) for all ROIs and color channels.
        
        This method computes spatiotemporal receptive fields using spike-triggered 
        averaging with multi-color noise stimuli, based on the Igor Pro implementation
        by Tom Baden with Python optimizations.
        
        Parameters
        ----------
        noise_array : np.ndarray
            3D noise stimulus array with shape (y, x, triggers) containing visual
            noise patterns used during the experiment
        sta_past_window : float, default 2.0
            How far into the past to calculate STA (seconds)
        sta_future_window : float, default 2.0  
            How far into the future to calculate STA (seconds)
        n_colours : int, default 1
            Number of color channels in the stimulus (use 1 for single-color experiments)
        n_triggers_per_colour : int or None, default None
            Number of triggers per color channel. If None, will be auto-calculated
            for single-color experiments or must be provided for multi-color.
        edge_crop : int, default 2
            Number of pixels to crop from stimulus edges
        max_frames_per_trigger : int, default 8
            Maximum frames between triggers allowed as noise frame
        event_sd_threshold : float, default 2.0
            Standard deviation threshold for event detection
        use_znorm : bool, default True
            Whether to use z-normalized traces
        adjust_by_polarity : bool, default True
            Whether to adjust results by detected polarity
        skip_first_triggers : int, default 0
            Number of first triggers to skip
        skip_last_triggers : int, default 0
            Number of last triggers to skip
        pre_smooth : int, default 0
            Pre-smoothing factor for SD projections
        roi : int, list, array or None, default None
            ROI indices to calculate STRFs for. If None, calculates for all ROIs.
            Can be a single ROI index (int), a list of indices, or numpy array.
        normalize_strfs : bool, default True
            Whether to apply the same normalization used during H5 loading
            (z-score based on first 1/5 of temporal frames). Set to False
            to get raw calculated STRFs for comparison.
        verbose : bool, default True
            Whether to print progress information
        **kwargs
            Additional arguments passed to the calculation function
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'strfs': STRF filters in Pygor format (n_rois*n_colours, time, y, x)
            - 'strfs_raw': STRF filters in raw format (n_rois, n_colours, y, x, time)
            - 'correlations': Spatial correlation maps (y, x*n_colours, n_rois)
            - 'standard_deviations': SD projection maps (y, x*n_colours, n_rois)  
            - 'polarities': Polarity maps (y, x*n_colours, n_rois)
            - 'mean_stimulus': Mean stimulus for each color (y, x, n_colours)
            - 'metadata': Processing metadata and parameters
            
        Examples
        --------
        >>> # Load noise array and calculate STRFs for all ROIs
        >>> noise_array = np.load('noise_stimulus.npy')  # Shape: (y, x, triggers)
        >>> results = strf_obj.calculate_sta(noise_array)
        >>> 
        >>> # Calculate STRFs for specific ROIs only (much faster!)
        >>> results = strf_obj.calculate_sta(noise_array, roi=[0, 2, 5])  # ROIs 0, 2, and 5
        >>> results = strf_obj.calculate_sta(noise_array, roi=3)         # Only ROI 3
        >>>
        >>> # Calculate with custom temporal window
        >>> results = strf_obj.calculate_sta(noise_array, sta_past_window=1.5, sta_future_window=1.0)
        >>>
        >>> # Access individual components
        >>> strfs = results['strfs']  # Shape: (n_rois, n_colours, x, y, time)
        >>> correlations = results['correlations']
        >>> metadata = results['metadata']
        """
        
        # Auto-detect multi-color experiments if object has numcolour attribute
        if n_colours == 1 and hasattr(self, 'numcolour') and self.numcolour > 1:
            n_colours = self.numcolour
            if verbose:
                print(f"Multi-color experiment detected: using {n_colours} colors")
        
        # Use multi-color optimized version for better performance when n_colours > 1
        if n_colours > 1:
            if verbose:
                print("Using multi-color optimized implementation for enhanced performance...")
            results = pygor.strf.calculate_multicolor_optimized.calculate_calcium_correlated_average_multicolor_optimized(
                self,
                noise_array,
                sta_past_window=sta_past_window,
                sta_future_window=sta_future_window,
                n_colours=n_colours,
                n_triggers_per_colour=n_triggers_per_colour,
                edge_crop=edge_crop,
                max_frames_per_trigger=max_frames_per_trigger,
                event_sd_threshold=event_sd_threshold,
                use_znorm=use_znorm,
                adjust_by_polarity=adjust_by_polarity,
                skip_first_triggers=skip_first_triggers,
                skip_last_triggers=skip_last_triggers,
                pre_smooth=pre_smooth,
                roi=roi,
                verbose=verbose,
                **kwargs
            )
        else:
            # Use regular optimized version for single-color
            results = pygor.strf.calculate_optimized.calculate_calcium_correlated_average_optimized(
                self,
                noise_array,
                sta_past_window=sta_past_window,
                sta_future_window=sta_future_window,
                n_colours=n_colours,
                n_triggers_per_colour=n_triggers_per_colour,
                edge_crop=edge_crop,
                max_frames_per_trigger=max_frames_per_trigger,
                event_sd_threshold=event_sd_threshold,
                use_znorm=use_znorm,
                adjust_by_polarity=adjust_by_polarity,
                skip_first_triggers=skip_first_triggers,
                skip_last_triggers=skip_last_triggers,
                pre_smooth=pre_smooth,
                roi=roi,
                verbose=verbose,
                **kwargs
            )
        
        # Apply the same normalization as used during H5 loading if requested
        if normalize_strfs and 'strfs' in results:
            if verbose:
                print("Applying post-processing normalization (same as H5 loading)...")
            
            # Apply post_process_strf_all to the calculated STRFs
            # The 'strfs' key contains the Pygor format: (n_rois*n_colours, time, y, x)
            results['strfs_unnormalized'] = results['strfs'].copy()  # Keep original for comparison
            results['strfs'] = pygor.data_helpers.post_process_strf_all(results['strfs'])
            
            if verbose:
                print("Normalization applied. Original STRFs stored in 'strfs_unnormalized' key.")
        
        return results

    def napari_strfs(self, **kwargs):
        import pygor.strf.gui.methods as gui
        napari_session = gui.NapariSession(self)
        return napari_session.run()
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