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
# import sklearn.preprocessing
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
import pygor.strf.polarity
import pygor.strf.calculate
import pygor.strf.calculate_optimized
import pygor.strf.calculate_multicolour_optimized
import pygor.utils.helpinfo
import pygor.utils.unit_conversion as unit_conversion
from pygor.classes.core_data import Core
# import scipy
from pygor.classes.core_data import try_fetch_table_params

@dataclass(repr = False)
class STRF(Core):
    #type         : str = "STRF"
    # Params 
    multicolour  : bool = None
    bs_bool      : bool = False
    # Help system configuration
    _help_exclude_patterns = ['_by_channel']
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
        if self.filename.suffix == ".h5":
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
                self.strf_dur_ms = try_fetch_table_params(HDF5_file, "Noise_FilterLength_s") * 1000
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
            # Validate data consistency (H5 has STRFs to validate)
            self._validate_data_consistency()
        else:
            # ScanM files (.smp/.smh): STRFs don't exist yet, set defaults
            # These will be populated after STRF calculation
            strf_defaults = self.params.get_defaults("strf").get("general", {})
            self.numcolour = strf_defaults.get("num_colours", 1)
            self.multicolour = self.numcolour > 1
            self.strfs = None
            self.strf_keys = []
            self.num_strfs = 0
            self.strf_dur_ms = None  # Will be set when STRFs are calculated

    def _validate_data_consistency(self):
        """
        Validate data consistency after loading.

        Checks:
        1. num_rois matches expected STRF count (num_strfs / numcolour)
        2. ipl_depths length matches num_rois

        Raises warnings for inconsistencies that could indicate data problems.

        Note: Only called for H5 files where STRFs exist. ScanM files skip
        validation since STRFs haven't been calculated yet.
        """
        # Skip validation if no STRFs loaded (e.g., ScanM files)
        if self.strfs is None or self.num_strfs == 0:
            return

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
        if mask_empty:
            __timecourses = np.ma.masked_equal(__timecourses, 0)
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

    def get_timecourses_secondary(self, **kwargs):
        secondary_times = []
        for arr in self.get_timecourses(**kwargs):
            if np.max(np.abs(arr[0]) > np.max(np.abs(arr[1]))):
                secondary_times.append(arr[1])
            else:
                secondary_times.append(arr[0])
        secondary_times = np.array(secondary_times)
        return secondary_times

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
        light-emitting diodes (LEDs) based on their indices in a multicoloured setup.

        Parameters:
        ----------
        num_colours : int, optional
            The number of colours or LED groups in the multicoloured setup. Default is 4.
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
            If the object is not a multicoloured spatial-pygor.strf.temporal receptive field (STRF),
            indicated by `self.multicolour` not being True.

        Notes:
        ------
        This method assumes that the class instance has attributes:
        - multicolour : bool
            A flag indicating whether the STRF is multicoloured.
        - contours_centres : list of numpy.ndarray
            List containing the center positions of contours for each colour in the multicoloured setup.

        Example:
        --------
        >>> strf_instance = STRF()
        >>> offset = strf_instance.calc_LED_offset(num_colours=4,
        ...                                       reference_LED_index=[0, 1, 2],
        ...                                       compare_LED_index=[3])
        >>> print(offset)
        array([x_offset, y_offset])
        """
        warnings.warn("This method is deprecated and will be removed in future versions.", DeprecationWarning)
    
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

    def collapse_times(self, roi = None, zscore : bool = True, spatial_centre : bool = False, border : bool = False, force_recompute : bool = False, **kwargs) -> np.ma.masked_array:
        # Create cache key from parameters
        roi_key = tuple(roi) if roi is not None and hasattr(roi, '__iter__') and not isinstance(roi, (str, int)) else roi
        cache_key = (roi_key, zscore, spatial_centre, border, tuple(sorted(kwargs.items())))
        
        # Check cache unless force_recompute
        if not hasattr(self, '_collapse_times_cache'):
            self._collapse_times_cache = {}
        
        if not force_recompute and cache_key in self._collapse_times_cache:
            return self._collapse_times_cache[cache_key]
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
        # Cache result before returning
        self._collapse_times_cache[cache_key] = collapsed_strf_arr
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
            raw_signal = self.bool_strf_signal()
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
        
        # Initialize cache if needed
        if not hasattr(self, '_spatial_overlap_cache'):
            self._spatial_overlap_cache = {}
        
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

    def spatial_overlap_blue_uv(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Blue-UV spatial overlap (Ch2-Ch3)"""
        return self.spatial_overlap_channel_pair(2, 3, abs_arrays, signal_only, single_channel_value)

    def spatial_overlap_green_uv(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Green-UV spatial overlap (Ch1-Ch3)"""
        return self.spatial_overlap_channel_pair(1, 3, abs_arrays, signal_only, single_channel_value)
    
    def spatial_overlap_red_blue(self, abs_arrays=True, signal_only=True, single_channel_value=np.nan):
        """Red-Blue spatial overlap (Ch0-Ch2)"""
        return self.spatial_overlap_channel_pair(0, 2, abs_arrays, signal_only, single_channel_value)

    def get_on_or_off(self, amplitude_threshold=0.1) -> np.ndarray:
        signal = self.get_dominant_response()
        return np.where(signal > 0, 1, -1)

    def get_polarities(self, roi=None, exclude_FirstLast=(1,1), mode="opponency_index", force_recompute=False) -> np.ndarray:
        """
        Determine the polarity of STRF responses using various analysis methods.
        
        This method provides multiple approaches to classify cells based on their
        temporal response characteristics, from simple peak detection to sophisticated
        center-surround analysis with rebound-aware polarity detection.
        
        Parameters
        ----------
        roi : int, optional
            Specific ROI to analyze. If None, analyzes all ROIs (default: None)
        exclude_FirstLast : tuple, optional  
            Number of timepoints to exclude from (start, end) of analysis (default: (1,1))
        mode : str, optional
            Analysis method to use:
            - "old": Original absolute max comparison between colour channels
            - "new": Improved signed peak detection  
            - "cs_pol": Center-surround analysis with rebound-aware detection (default)
            - "cs_pol_extra": Extended CS analysis with strong/weak classification
            - "gabor": Spatial-based ON/OFF/Gabor classification
            - "spatial": Spatial polarity index-based classification
        force_recompute : bool, optional
            Force recomputation of center-surround segmentation (default: False)
            
        Returns
        -------
        np.ndarray
            Polarity classifications for each ROI:
            - old/new modes: -1 (OFF), 1 (ON), 2 (bipolar), 0/NaN (no signal)
            - cs_pol mode: -1 (OFF center), 1 (ON center), 2 (center-surround), NaN (weak signal)  
            - cs_pol_extra: Same as cs_pol plus 3 (strong CS), 4 (weak CS)
            - gabor: -1 (OFF), 1 (ON), 2 (Gabor), NaN (no signal)
            - spatial: -1 (OFF), 1 (ON), 2 (opponent), NaN (no signal)
            
        Examples
        --------
        >>> # Standard center-surround polarity analysis
        >>> polarities = strf_obj.get_polarities()
        >>> 
        >>> # Force fresh segmentation computation
        >>> polarities = strf_obj.get_polarities(force_recompute=True)
        >>> 
        >>> # Extended analysis with CS strength classification
        >>> detailed_pols = strf_obj.get_polarities(mode="cs_pol_extra")
        >>>
        >>> # Legacy absolute max method
        >>> old_pols = strf_obj.get_polarities(mode="old")
        
        Notes
        -----
        The default "cs_pol" mode uses center-surround segmentation combined with 
        rebound-aware polarity detection to handle complex temporal dynamics including
        transients followed by sustained responses. This is particularly useful for
        cells that show initial negative deflections followed by positive rebounds.
        
        The rebound-aware algorithm was specifically designed to handle edge cases
        where correlation-based methods would incorrectly classify ON cells as OFF
        due to early negative transients.
        
        See Also
        --------
        cs_seg : Center-surround segmentation method
        get_timecourses : Extract temporal responses
        """
        if mode == "spatial":
            # Get spatial polarity indices (now returns NaN for weak signals)
            polarity_indices = self.spatial_polarity_index()
            cat = np.full(len(polarity_indices), np.nan)
            
            # Apply thresholding to classify based on polarity index
            valid_mask = ~np.isnan(polarity_indices)
            opponent_threshold = .8  # Could be made a parameter
            
            # Classify based on polarity index values:
            # - Strong positive (>0.5): ON cells
            # - Strong negative (<-0.5): OFF cells  
            # - Weak absolute value (±0.5): Opponent/balanced cells
            # - NaN: Insufficient signal (remains NaN)
            cat[valid_mask & (polarity_indices > opponent_threshold)] = 1    # ON
            cat[valid_mask & (polarity_indices < -opponent_threshold)] = -1  # OFF
            cat[valid_mask & (np.abs(polarity_indices) <= opponent_threshold)] = 2  # Opponent
            return cat
        
        elif mode == "opponency_index":
            threshold = 0.33
            amplitudes = self.get_space_amps()
            amp_signs = np.where(amplitudes > 0, 1, -1)
            cat = amp_signs
            # Apply amplitude check similar to spatiotemporal mode
            bool_mask = self.bool_by_channel(dimstr="space").flatten()
            cat = np.where(bool_mask, cat, np.nan)
        
            # opponency_indices = self.get_opponency_index()
            opponency_indices = self.get_opponency_index_time()
            cat = np.where((bool_mask) & (opponency_indices >= threshold), 2, cat)
            return cat

        elif mode == "spatiotemporal":
            # Hybrid approach: spatial analysis first, temporal fallback for NaNs
            # Get spatial polarity indices (keeps high threshold for accuracy)
            polarity_indices = self.spatial_polarity_index()
            cat = np.full(len(polarity_indices), np.nan)
            
            # Apply spatial classification first (this takes priority)
            valid_mask = ~np.isnan(polarity_indices)
            opponent_threshold = .8  # Could be made a parameter
            
            cat[valid_mask & (polarity_indices > opponent_threshold)] = 1    # ON
            cat[valid_mask & (polarity_indices < -opponent_threshold)] = -1  # OFF
            cat[valid_mask & (np.abs(polarity_indices) <= opponent_threshold)] = 2  # Opponent
            
            # For remaining NaNs, use temporal analysis as fallback
            nan_mask = np.isnan(cat)
            if np.any(nan_mask):
                # Get dominant timecourses for cells that couldn't be classified spatially
                timecourses = self.get_timecourses_dominant(mask_empty=True)
                
                # Simple temporal polarity: sign of peak absolute response
                for i in np.where(nan_mask)[0]:
                    tc = timecourses[i]
                    if np.any(np.abs(tc) > 0.5):  # Minimum signal threshold
                        # Find peak absolute response and use its sign
                        peak_idx = np.argmax(np.abs(tc))
                        peak_val = tc[peak_idx]
                        cat[i] = 1 if peak_val > 0 else -1
                
            return cat
            
        elif mode == "gabor":
            # First separate out to ON and OFF
            spaces = self.collapse_times()
            space_min = np.min(spaces, axis=(1, 2))
            space_max = np.max(spaces, axis=(1, 2))
            cat = np.where(np.abs(space_min) < space_max, 1, -1)
            thresh = 2
            min_below_thresh = np.abs(space_min) < thresh
            max_below_thresh = np.abs(space_max) < thresh
            no_signal = np.bitwise_and(min_below_thresh, max_below_thresh)
            cat = cat.astype(float)
            cat[no_signal] = np.nan
            # Then separate out "gabor" cells 
            condition2 = np.bitwise_and(space_min < -thresh, space_max > thresh)
            cat = np.where(condition2, 2, cat)
            return cat

        else:
            # Delegate to the polarity module for other modes
            return pygor.strf.polarity.get_polarities(self, roi, exclude_FirstLast, mode, force_recompute)
        # Handle on_off_gabor mode directly since it uses spatial analysis
        
        # elif mode == "spatial":
        #     # Use spatial polarity index for classification
        #     opponent_threshold = 0.5  # Could be made a parameter
        #     indices = self.spatial_polarity_index(roi=roi, threshold=2, mask_by_channel=True, 
        #                                         mask_threshold=2, dimstr="space")
            
        #     # Ensure we have an array for consistent processing
        #     if np.isscalar(indices):
        #         indices = np.array([indices])
                
        #     # Convert spatial polarity indices to categories
        #     cat = np.full(len(indices), np.nan)
        #     valid_mask = ~np.isnan(indices)
            
        #     # Apply thresholds to valid indices
        #     cat[valid_mask & (indices > opponent_threshold)] = 1   # ON
        #     cat[valid_mask & (indices < -opponent_threshold)] = -1  # OFF  
        #     cat[valid_mask & (np.abs(indices) <= opponent_threshold)] = 2  # opponent
            
        #     return cat
        
        # # Use dedicated polarity module for other modes
        # from pygor.strf.polarity import get_polarities
        # return get_polarities(self, roi, exclude_FirstLast, mode, force_recompute)

    def get_polarities_simple(self):
        amplitudes = self.get_space_amps()
        amp_signs = np.where(amplitudes > 0, 1, -1)
        cat = amp_signs
        # Apply amplitude check similar to spatiotemporal mode
        bool_mask = self.bool_by_channel(dimstr="space").flatten()
        cat = np.where(bool_mask, cat, np.nan)
        return cat

    def check_cs_pass(self):
        # if a label exists from self.get_polarities
        maps, _ = self.cs_seg()
        # where maps is not all zeros, give True otherwise False
        # print([np.unique(m) for m in maps])
        return np.array([len(np.unique(m)) > 1 for m in maps])

    def spatial_polarity_index(self, roi = None, ch_idx = None, threshold = 3, mask_by_channel = True, mask_threshold = 2, dimstr = "time") -> np.ndarray:
        # Handle multicolour channel selection efficiently
        if ch_idx is not None:
            if not self.multicolour:
                if ch_idx == -1 or ch_idx == 0:
                    # Allow -1 (last) or 0 (first) for single channel compatibility
                    spaces = self.collapse_times(roi=roi)
                else:
                    raise ValueError(f"ch_idx={ch_idx} specified but object is not multicolour (only has 1 channel, use 0 or -1)")
            else:
                spaces = self.collapse_times_chroma(roi=roi)[ch_idx]
        else:
            spaces = self.collapse_times(roi=roi)
        spaces_flat = spaces.reshape(spaces.shape[0], -1)

        # Create masks for each ROI
        lowers_mask = spaces_flat < -threshold
        uppers_mask = spaces_flat > threshold

        # Sum across spatial pixels (axis=1) for each ROI
        lowers_sum = np.sum(spaces_flat * lowers_mask, axis=1)
        uppers_sum = np.sum(spaces_flat * uppers_mask, axis=1)

        # Handle division by zero
        denominator = uppers_sum - lowers_sum
        indices = np.where(denominator != 0, 
                        (uppers_sum + lowers_sum) / denominator, np.nan)
        # Apply bool_by_channel masking if requested
        if mask_by_channel:
            bool_mask = self.bool_by_channel(threshold=mask_threshold, dimstr=dimstr)
            
            # Handle channel-specific masking
            if ch_idx is not None and self.multicolour:
                # For specific channel, use that channel's mask
                channel_mask = bool_mask[:, ch_idx]
            elif not self.multicolour:
                # For single colour, use first/only channel mask
                channel_mask = bool_mask[:, 0] if bool_mask.ndim > 1 else bool_mask
            else:
                # No specific channel requested for multicolour - this is ambiguous
                # The indices came from collapse_times (all channels combined)
                # So we need to decide masking strategy - use ANY channel passes
                channel_mask = np.any(bool_mask, axis=1)
            
            # Apply mask - ensure indices and channel_mask have compatible shapes
            if len(channel_mask) == len(indices):
                indices = np.where(channel_mask, indices, np.nan)
            else:
                # Shape mismatch - likely due to ROI filtering, skip masking
                pass
        
        # Return scalar if single ROI requested, array otherwise
        if roi is not None and np.isscalar(roi):
            return indices[0] if len(indices) > 0 else 0
        indices = np.where(indices == 0, np.nan, indices)
        return indices

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

    def check_strf_gabor(self, ampl_thresh = 2, ch_idx = None):
        spaces = self.collapse_times_chroma()
        spaces_ch = spaces[ch_idx]
        maxes = np.max(spaces_ch, axis = (1, 2))
        mins = np.min(spaces_ch, axis = (1, 2))
        condition1 = maxes > ampl_thresh
        condition2 = mins < -ampl_thresh
        return np.squeeze(np.bitwise_and(condition1, condition2))

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
        
    def calc_spatial_opponency(self, mode = None):
        if mode == None or mode == "all":
            arrs = self.collapse_times()    
        if mode == "white":
            arrs = self.compute_average_spaces()
        return pygor.strf.spatial.snr_gated_spatial_opponency(arrs)
        
    def calc_centre_distances(self, mode = "cs_seg"):
        if mode == "cs_seg":
            rf_centres = self.get_seg_centres()  # Shape: (n_total_strfs, 2)
            
            # Calculate mean center across all STRFs
            overall_center = np.nanmean(rf_centres, axis=0, keepdims=True)  # Shape: (1, 2)
            
            # Calculate distance of each STRF from overall center
            euclidian_dist = np.sqrt(np.sum((rf_centres - overall_center)**2, axis=1))  # Shape: (n_total_strfs,)
            euclidian_dist = euclidian_dist * self.stim_size
            
            return euclidian_dist
        else:
            raise NotImplementedError

    def calc_pca_rf_shape_analysis(self, roi=None, threshold_sd=1, plot=False, force_recompute=False, debug=False):
        """
        PCA-based RF shape analysis for quantifying elongation and eccentricity
        
        Parameters
        ----------
        roi : int or None, optional
            ROI index to analyze, or None for all ROIs (default: None)
        threshold_sd : float, optional
            Threshold in standard deviations for significant pixels (default: 3.0)
        plot : bool, optional
            Whether to create demonstration plot for single ROI (default: False)
        force_recompute : bool, optional
            Force recomputation even if cached (default: False)
            
        Returns
        -------
        dict
            Dictionary with keys containing arrays for each metric:
            - 'major_axis_length': Major axis lengths (n_cells,) or scalar for single ROI
            - 'minor_axis_length': Minor axis lengths (n_cells,) or scalar for single ROI
            - 'eccentricity': Eccentricity values (n_cells,) or scalar for single ROI
            - 'angle_degrees': Major axis angles in degrees (n_cells,) or scalar for single ROI
            - 'angle_radians': Major axis angles in radians (n_cells,) or scalar for single ROI
            - 'centroid_x': X coordinates of centroids (n_cells,) or scalar for single ROI
            - 'centroid_y': Y coordinates of centroids (n_cells,) or scalar for single ROI
            - 'significant_pixels': Number of significant pixels per ROI (n_cells,) or scalar for single ROI
            
        Notes
        -----
        Angle Direction Convention:
        - 0° corresponds to East (positive x-direction)
        - 90° corresponds to North (positive y-direction) 
        - 180° corresponds to West (negative x-direction)
        - 270° corresponds to South (negative y-direction)
        
        The angle represents the orientation of the major axis of the RF.
        Calculated using np.arctan2(eigenvector_y, eigenvector_x) and converted to 0-360° range.
        """
        # Create cache key from parameters
        roi_key = tuple(roi) if roi is not None and hasattr(roi, '__iter__') and not isinstance(roi, (str, int)) else roi
        cache_key = (roi_key, threshold_sd, tuple(sorted({})))
        
        # Check cache unless force_recompute
        if not hasattr(self, '_pca_rf_shape_cache'):
            self._pca_rf_shape_cache = {}
        
        if not force_recompute and cache_key in self._pca_rf_shape_cache:
            return self._pca_rf_shape_cache[cache_key]
        
        # Determine which STRFs to process - sequential indexing only
        if roi is not None:
            if isinstance(roi, (int, np.integer)):
                strf_list = [roi]
            else:
                strf_list = roi
        else:
            # All STRFs
            strf_list = range(len(self.strfs))
        
        if debug:
            print(f"DEBUG: Processing strf_list = {strf_list} for roi = {roi}")
        
        # Initialize result arrays
        major_axis_lengths = []
        minor_axis_lengths = []
        eccentricities = []
        angles_degrees = []
        angles_radians = []
        centroid_xs = []
        centroid_ys = []
        significant_pixels_counts = []
        
        # Process each STRF sequentially - 1:1 correspondence between collapse_times and cs_seg
        # Get all collapsed times and cs_seg results efficiently
        all_collapsed = self.collapse_times()
        all_cs_maps, _ = self.cs_seg()
        
        if debug:
            print(f"DEBUG: Processing strf_list = {strf_list} for roi = {roi}")
            print(f"DEBUG: all_collapsed shape = {all_collapsed.shape}, all_cs_maps shape = {all_cs_maps.shape}")
            print(f"UNIQUE VALUES ARE : {np.unique(all_cs_maps[roi])}")

        for i, strf_idx in enumerate(strf_list):
            # Get RF data for this STRF
            strf_data = np.squeeze(all_collapsed[strf_idx])
            
            # Get the corresponding cs_seg mask
            currmap = np.squeeze(all_cs_maps[strf_idx])
            
            if debug:
                print(f"DEBUG STRF {strf_idx}: Processing STRF {strf_idx}")
                print(f"UNIQUE VALUES ARE : {np.unique(currmap)}")

            label_mask = currmap == 0
            # Check if there are any center pixels at all
            center_pixel_count = np.sum(label_mask)
            if center_pixel_count == 0:
                # No center pixels found - skip this ROI
                if debug:
                    unique_mask_vals = np.unique(currmap)
                    print(f"DEBUG STRF {strf_idx}: mask_unique_values={unique_mask_vals}, center_pixels={center_pixel_count}")
                    print(f"DEBUG STRF {strf_idx}: No center pixels - returning NaN and continuing to next STRF")
                major_axis_lengths.append(np.nan)
                minor_axis_lengths.append(np.nan)
                eccentricities.append(np.nan)
                angles_degrees.append(np.nan)
                angles_radians.append(np.nan)
                centroid_xs.append(np.nan)
                centroid_ys.append(np.nan)
                significant_pixels_counts.append(0)
                if debug:
                    print(f"DEBUG STRF {strf_idx}: Added NaN values, about to continue...")
                continue
            
            # Debug: Print mask information for problematic cases
            if debug:
                unique_mask_vals = np.unique(currmap)
                print(f"DEBUG STRF {strf_idx}: mask_unique_values={unique_mask_vals}, center_pixels={center_pixel_count}")
            
            # Apply connected component labeling to get largest center region (suppress noise)
            from scipy import ndimage
            from skimage import measure
            
            # Get the largest connected component of center pixels to suppress noise
            if np.sum(label_mask) > 0:
                # Label connected components in the center region
                labeled_center = measure.label(label_mask, connectivity=2)
                if np.max(labeled_center) > 0:
                    # Find the largest connected component
                    component_sizes = np.bincount(labeled_center.flat)[1:]  # Exclude background (0)
                    if len(component_sizes) > 0:
                        largest_component = np.argmax(component_sizes) + 1
                        largest_component_mask = (labeled_center == largest_component)
                        if debug:
                            print(f"DEBUG STRF {strf_idx}: Found {len(component_sizes)} connected components, using largest with {component_sizes[largest_component-1]} pixels")
                        # Use only the largest connected component
                        label_mask = largest_component_mask
            
            # Apply label mask to RF data
            center_data = strf_data[label_mask]
            
            # Determine dominant polarity in center region
            pos_pixels = np.sum(center_data > threshold_sd)
            neg_pixels = np.sum(center_data < -threshold_sd)
            if debug:
                print(f"DEBUG STRF {strf_idx}: pos_pixels={pos_pixels}, neg_pixels={neg_pixels}")
            
            if pos_pixels >= neg_pixels:
                # Use positive pixels
                significant_mask = (strf_data > threshold_sd) & label_mask
                if debug:
                    print(f"DEBUG STRF {strf_idx}: using positive pixels, significant_count={np.sum(significant_mask)}")
            else:
                # Use negative pixels  
                significant_mask = (strf_data < -threshold_sd) & label_mask
                if debug:
                    print(f"DEBUG STRF {strf_idx}: using negative pixels, significant_count={np.sum(significant_mask)}")
            
            if debug and np.sum(significant_mask) > 0:
                unique_mask_vals = np.unique(currmap) if not debug else unique_mask_vals
                if len(unique_mask_vals) == 1 and unique_mask_vals[0] != 0:
                    print(f"DEBUG STRF {strf_idx}: CRITICAL ERROR - {np.sum(significant_mask)} pixels selected but mask has no center region!")
            
            if np.sum(significant_mask) < 3:
                # Handle insufficient pixels
                major_axis_lengths.append(np.nan)
                minor_axis_lengths.append(np.nan)
                eccentricities.append(np.nan)
                angles_degrees.append(np.nan)
                angles_radians.append(np.nan)
                centroid_xs.append(np.nan)
                centroid_ys.append(np.nan)
                significant_pixels_counts.append(0)
                continue
            
            # Get coordinates and weights of significant pixels
            y_coords, x_coords = np.where(significant_mask)
            weights = np.abs(strf_data[significant_mask])
            
            # Proper weighted PCA using covariance matrix
            coords = np.column_stack([x_coords, y_coords])
            
            # Calculate weighted centroid
            centroid = np.average(coords, weights=weights, axis=0)
            
            # Center coordinates
            centered_coords = coords - centroid
            
            # Calculate weighted covariance matrix
            cov_matrix = np.cov(centered_coords.T, aweights=weights)
            
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (largest first)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx].T
            
            # Extract results
            major_axis_length = 2 * np.sqrt(eigenvalues[0])
            minor_axis_length = 2 * np.sqrt(eigenvalues[1])
            # eccentricity = 1 - (minor_axis_length / major_axis_length)
            eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0])) if eigenvalues[0] > 0 else 0
            # Angle calculation following pygor convention:
            # 0° = East (positive x), 90° = North (positive y)
            # arctan2(y, x) where y=eigenvector_y, x=eigenvector_x
            angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
            angle_deg = np.degrees(angle_rad)
            
            # Convert to 0-360° range following pygor convention
            if angle_deg < 0:
                angle_deg = angle_deg + 360
            # Limit data to 180 degree format
            if angle_deg > 180:
                angle_deg -= 180
            # Store results
            major_axis_lengths.append(major_axis_length)
            minor_axis_lengths.append(minor_axis_length)
            eccentricities.append(eccentricity)
            angles_degrees.append(angle_deg)
            angles_radians.append(angle_rad)
            centroid_xs.append(centroid[0])
            centroid_ys.append(centroid[1])
            significant_pixels_counts.append(np.sum(significant_mask))
            
            # Plot single STRF if requested
            if plot and roi is not None and isinstance(roi, (int, np.integer)) and strf_idx == roi:
                # Only plot for the first color channel of the requested ROI
                roi_results = {
                    'major_axis_length': major_axis_length,
                    'minor_axis_length': minor_axis_length,
                    'eccentricity': eccentricity,
                    'angle_degrees': angle_deg,
                    'angle_radians': angle_rad,
                    'centroid': centroid,
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors,
                    'significant_pixels': np.sum(significant_mask),
                    '_strf_data': strf_data,
                    '_significant_mask': significant_mask,
                    '_coords': coords,
                    '_weights': weights
                }
                self._plot_pca_results(roi_results, roi, threshold_sd)
        
        # Debug: Show what's actually in the result arrays
        if debug:
            print(f"DEBUG: Final result arrays before creating dictionary:")
            print(f"  major_axis_lengths = {major_axis_lengths}")
            print(f"  eccentricities = {eccentricities}")
            print(f"  significant_pixels_counts = {significant_pixels_counts}")
        
        # Create results dictionary
        results = {
            'major_axis_length': np.array(major_axis_lengths),
            'minor_axis_length': np.array(minor_axis_lengths),
            'eccentricity': np.array(eccentricities),
            'angle_degrees': np.array(angles_degrees),
            'angle_radians': np.array(angles_radians),
            'centroid_x': np.array(centroid_xs),
            'centroid_y': np.array(centroid_ys),
            'significant_pixels': np.array(significant_pixels_counts)
        }
        
        # For single ROI, return arrays for all its color channels (not scalars)
        # This maintains compatibility with _by_channel auto-generation
                
        # Cache result
        self._pca_rf_shape_cache[cache_key] = results
        
        return results
    
    def _plot_pca_results(self, results, roi, threshold_sd):
        """Helper method to plot PCA results"""
        import matplotlib.pyplot as plt
        
        strf_data = results['_strf_data']
        significant_mask = results['_significant_mask']
        coords = results['_coords']
        weights = results['_weights']
        centroid = results['centroid']
        eigenvectors = results['eigenvectors']
        major_axis_length = results['major_axis_length']
        minor_axis_length = results['minor_axis_length']
        eccentricity = results['eccentricity']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Original RF
        im1 = axes[0].imshow(strf_data, cmap='Greys_r', origin='lower')
        axes[0].set_title(f'Original RF (ROI {roi})')
        axes[0].plot(centroid[0], centroid[1], 'k+', markersize=10, markeredgewidth=2)
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Thresholded RF
        threshold_display = np.where(significant_mask, strf_data, np.nan)
        im2 = axes[1].imshow(threshold_display, cmap='Greys_r', origin='lower')
        axes[1].set_title(f'Significant Pixels (±{threshold_sd}σ)')
        axes[1].plot(centroid[0], centroid[1], 'k+', markersize=10, markeredgewidth=2)
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: PCA axes overlay
        axes[2].imshow(strf_data, cmap='Greys_r', origin='lower', alpha=0.7)
        y_coords, x_coords = coords[:, 1], coords[:, 0]
        axes[2].scatter(x_coords, y_coords, c=weights, cmap='viridis', s=20, alpha=0.8)
        
        # Draw PCA axes
        axis_scale = max(major_axis_length, minor_axis_length) / 2
        
        # Major axis (red)
        major_vec = eigenvectors[0] * axis_scale
        axes[2].arrow(centroid[0], centroid[1], major_vec[0], major_vec[1],
                    color='red', width=0.2, head_width=0.8, alpha=0.8, label='Major axis')
        axes[2].arrow(centroid[0], centroid[1], -major_vec[0], -major_vec[1],
                    color='red', width=0.2, head_width=0.8, alpha=0.8)
        
        # Minor axis (blue)  
        minor_vec = eigenvectors[1] * axis_scale
        axes[2].arrow(centroid[0], centroid[1], minor_vec[0], minor_vec[1],
                    color='blue', width=0.2, head_width=0.8, alpha=0.8, label='Minor axis')
        axes[2].arrow(centroid[0], centroid[1], -minor_vec[0], -minor_vec[1],
                    color='blue', width=0.2, head_width=0.8, alpha=0.8)
        
        axes[2].plot(centroid[0], centroid[1], 'k+', markersize=10, markeredgewidth=2)
        axes[2].set_title(f'PCA Axes (Eccent: {eccentricity:.3f})')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

    def calc_centre_surround_metrics(self, roi=None, force_recompute=False, debug=False):
        """
        Smart center-surround analysis for cs_seg segmented STRFs.
        Always assigns stronger region as 'dominant', weaker as 'secondary'.
        
        Parameters
        ----------
        roi : int or None, optional
            ROI index to analyze, or None for all ROIs (default: None)
        force_recompute : bool, optional
            Force recomputation even if cached (default: False)
        debug : bool, optional
            Print debug information (default: False)
            
        Returns
        -------
        dict
            Dictionary with arrays for each metric:
            - 'dominant_response': Dominant region responses (n_cells,) or scalar
            - 'secondary_response': Secondary region responses (n_cells,) or scalar  
            - 'noise_response': Noise region responses (n_cells,) or scalar
            - 'dominant_is_region_a': Boolean indicating if region A is dominant (n_cells,) or scalar
            - 'dominance_ratio': Strength ratios (n_cells,) or scalar
            - 'secondary_percentage': Secondary as % of dominant (n_cells,) or scalar
            - 'opponency_index': Spatial balance indices (n_cells,) or scalar
            - 'opposite_polarity': Antagonistic relationship flags (n_cells,) or scalar
            - 'net_response': Combined responses (n_cells,) or scalar
            - 'total_magnitude': Total activity (n_cells,) or scalar
        """
        # Create cache key from parameters
        roi_key = tuple(roi) if roi is not None and hasattr(roi, '__iter__') and not isinstance(roi, (str, int)) else roi
        cache_key = (roi_key, tuple(sorted({})))
        
        # Check cache unless force_recompute
        if not hasattr(self, '_centre_surround_metrics_cache'):
            self._centre_surround_metrics_cache = {}
        
        if not force_recompute and cache_key in self._centre_surround_metrics_cache:
            return self._centre_surround_metrics_cache[cache_key]
        
        # Determine which STRFs to process
        if roi is not None:
            if isinstance(roi, (int, np.integer)):
                strf_list = [roi]
            else:
                strf_list = roi
        else:
            strf_list = range(len(self.strfs))
        
        # Get all collapsed times and cs_seg results efficiently
        all_collapsed = self.collapse_times()
        all_cs_maps, _ = self.cs_seg()
        
        # Initialize result arrays
        results = {
            'dominant_response': [],
            'secondary_response': [],
            'noise_response': [],
            'dominant_is_region_a': [],
            'dominance_ratio': [],
            'secondary_percentage': [],
            'opponency_index': [],
            'opposite_polarity': [],
            'net_response': [],
            'total_magnitude': []
        }
        
        for strf_idx in strf_list:
            # Get RF data and segmentation map
            strf_data = np.squeeze(all_collapsed[strf_idx])
            currmap = np.squeeze(all_cs_maps[strf_idx])
            
            # Create region masks
            region_a_map = currmap == 0
            region_b_map = currmap == 1  
            noise_map = currmap == 2
            
            # Calculate responses (normalized by area), return NaN for empty masks
            region_a_sum = np.sum(strf_data[region_a_map]) / region_a_map.sum() if region_a_map.sum() > 0 else np.nan
            region_b_sum = np.sum(strf_data[region_b_map]) / region_b_map.sum() if region_b_map.sum() > 0 else np.nan
            noise_sum = np.sum(strf_data[noise_map]) / noise_map.sum() if noise_map.sum() > 0 else np.nan
            
            # Check for invalid responses (NaN or both regions are zero)
            if (np.isnan(region_a_sum) or np.isnan(region_b_sum) or 
                (abs(region_a_sum) == 0 and abs(region_b_sum) == 0)):
                # Return all NaNs for this ROI
                dominant_response = np.nan
                secondary_response = np.nan
                dominant_is_region_a = np.nan
                dominance_ratio = np.nan
                secondary_percentage = np.nan
                opponency_index = np.nan  # Include opponency_index in NaN handling
                opposite_polarity = np.nan
                net_response = np.nan
                total_magnitude = np.nan
            else:
                # Determine dominant region
                if abs(region_a_sum) >= abs(region_b_sum):
                    dominant_response = region_a_sum
                    secondary_response = region_b_sum
                    dominant_is_region_a = True
                else:
                    dominant_response = region_b_sum
                    secondary_response = region_a_sum
                    dominant_is_region_a = False
                
                # Calculate metrics with NaN handling
                if abs(secondary_response) == 0:
                    dominance_ratio = np.nan  # Avoid inf for zero secondary
                    secondary_percentage = np.nan
                else:
                    dominance_ratio = abs(dominant_response) / abs(secondary_response)
                    secondary_percentage = abs(secondary_response) / abs(dominant_response) * 100
                
                if abs(dominant_response) == 0:
                    opponency_index = np.nan
                    opposite_polarity = np.nan
                else:
                    opposite_polarity = (dominant_response * secondary_response) < 0
                    if opposite_polarity:
                        # True opponency: different polarities
                        opponency_index = 1 - (abs(dominant_response) - abs(secondary_response)) / (abs(dominant_response) + abs(secondary_response))
                    else:
                        # Same polarity: no true opponency, set to 1
                        opponency_index = 0
                
                net_response = dominant_response + secondary_response
                total_magnitude = abs(dominant_response) + abs(secondary_response)
            
            # Store results
            results['dominant_response'].append(dominant_response)
            results['secondary_response'].append(secondary_response)
            results['noise_response'].append(noise_sum)
            results['dominant_is_region_a'].append(dominant_is_region_a)
            results['dominance_ratio'].append(dominance_ratio)
            results['secondary_percentage'].append(secondary_percentage)
            results['opponency_index'].append(opponency_index)
            results['opposite_polarity'].append(opposite_polarity)
            results['net_response'].append(net_response)
            results['total_magnitude'].append(total_magnitude)
        
        # Convert to numpy arrays and handle single ROI case
        for key in results:
            results[key] = np.array(results[key])
            if len(results[key]) == 1:
                results[key] = results[key][0]  # Return scalar for single ROI
        
        # Cache result before returning
        self._centre_surround_metrics_cache[cache_key] = results
        return results

    def get_dominance_ratio(self, roi=None, force_recompute=False):
        """Get dominance ratio (how many times stronger dominant region is)."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['dominance_ratio']
    
    def get_secondary_percentage(self, roi=None, force_recompute=False):
        """Get secondary region strength as percentage of dominant."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['secondary_percentage']
    
    def get_opponency_index(self, roi=None, force_recompute=False):
        """Get spatial balance index (0=balanced, 1=one-sided)."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['opponency_index']
    
    def get_opposite_polarity(self, roi=None, force_recompute=False):
        """Get whether regions have antagonistic relationship (True/False)."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['opposite_polarity']
    
    def get_net_response(self, roi=None, force_recompute=False):
        """Get combined response after center-surround interaction."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['net_response']
    
    def get_total_magnitude(self, roi=None, force_recompute=False):
        """Get total activity before center-surround cancellation."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['total_magnitude']
    
    def get_dominant_response(self, roi=None, force_recompute=False):
        """Get raw response of the stronger region."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['dominant_response']
    
    def get_secondary_response(self, roi=None, force_recompute=False):
        """Get raw response of the weaker region."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['secondary_response']
    
    def get_noise_response(self, roi=None, force_recompute=False):
        """Get average response in noise regions."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['noise_response']
    
    def is_region_a_dominant(self, roi=None, force_recompute=False):
        """Check if cs_seg region A (labeled as 'center') is the dominant one."""
        return self.calc_centre_surround_metrics(roi=roi, force_recompute=force_recompute)['dominant_is_region_a']

    def calc_colour_channel_offsets(self, roi=None, mode="cs_seg", label=0, weighted=True, threshold=4, angle_range_360=True, plot=False, minimal_plot=False):
        """
        Calculate direction and magnitude of each colour channel's offset from the cell's true center.
        
        For multicolour data, finds the mean center across all colour channels (true center),
        then calculates offset vectors for each channel from this true center.
        
        Parameters
        ----------
        mode : str, optional
            Method for finding channel centers:
            - "cs_seg": Use center-surround segmentation (default)
            - "minmax": Find spatial min/max positions across channels with threshold filtering
            - "weighted": Find weighted centroids of thresholded positive/negative regions per channel
        roi : int, optional
            ROI indices to analyze, or which ROI to plot (if None, all ROIs)
        label : int, optional
            Segmentation label (0=center, 1=surround) - only used for cs_seg mode (default: 0)
        weighted : bool, optional
            Whether to use weighted centroids - only used for cs_seg mode (default: True)
        threshold : float, optional
            Minimum absolute value for minmax mode (default: 2.0)
        angle_range_360 : bool, optional
            If True, angles range 0-360°. If False, -180 to +180° (default: True)
        plot : bool, optional
            Whether to plot a demo visualization (default: False)
        minimal_plot : bool, optional
            If True, creates a minimal plot with just arrows on RGB background (default: False)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'true_centers': Array of true centers for each cell (n_cells, 2)
            - 'channel_centers': Array of channel centers (n_colours, n_cells, 2) 
            - 'offsets': Offset vectors (n_colours, n_cells, 2)
            - 'magnitudes': Offset magnitudes (n_colours, n_cells)
            - 'angles': Offset angles in degrees (n_colours, n_cells)
        """
        if not self.multicolour:
            raise ValueError("This method requires multicolour data (numcolour > 1)")
        
        if mode == "cs_seg":
            # Get centers for each colour channel: shape (n_colours, n_cells, 2)
            channel_centers = self.get_seg_centres_by_channel(label=label, weighted=weighted)
            n_colours, n_cells, _ = channel_centers.shape
            
        elif mode == "minmax":
            # Find spatial extrema across channels for each ROI
            spaces = self.collapse_times_by_channel()  # shape: (n_colours, n_cells, height, width)
            n_colours, n_cells, height, width = spaces.shape
            
            channel_centers = np.full((n_colours, n_cells, 2), np.nan)
            
            for colour_idx in range(n_colours):
                for cell_idx in range(n_cells):
                    space = spaces[colour_idx, cell_idx]  # (height, width)
                    
                    # Find min/max values and positions
                    min_val = np.nanmin(space)
                    max_val = np.nanmax(space)
                    
                    # Check if extrema meet threshold
                    min_valid = np.abs(min_val) >= threshold
                    max_valid = np.abs(max_val) >= threshold
                    
                    if min_valid or max_valid:
                        # Choose the extremum with larger absolute value
                        if min_valid and max_valid:
                            if np.abs(min_val) >= np.abs(max_val):
                                extremum_coords = np.unravel_index(np.nanargmin(space), space.shape)
                            else:
                                extremum_coords = np.unravel_index(np.nanargmax(space), space.shape)
                        elif min_valid:
                            extremum_coords = np.unravel_index(np.nanargmin(space), space.shape)
                        else:  # max_valid
                            extremum_coords = np.unravel_index(np.nanargmax(space), space.shape)
                        
                        channel_centers[colour_idx, cell_idx] = extremum_coords
                    # If neither valid, remains NaN
                    
        elif mode == "weighted":
            # Weighted centroid approach with bidirectional thresholding per channel
            spaces = self.collapse_times_by_channel()  # shape: (n_colours, n_cells, height, width)
            n_colours, n_cells, height, width = spaces.shape
            
            # Create coordinate grids for weighted centroid calculation
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            channel_centers = np.full((n_colours, n_cells, 2), np.nan)
            
            for colour_idx in range(n_colours):
                for cell_idx in range(n_cells):
                    space = spaces[colour_idx, cell_idx]  # (height, width)
                    
                    # Apply bidirectional thresholding
                    pos_mask = space > threshold   # Positive pixels above threshold
                    neg_mask = space < -threshold  # Negative pixels below threshold
                    
                    pos_pixels = space * pos_mask  # Positive values, others zero
                    neg_pixels = space * neg_mask  # Negative values, others zero
                    
                    # Calculate weighted centroids for positive and negative regions
                    pos_sum = np.sum(pos_pixels)
                    neg_sum = np.sum(np.abs(neg_pixels))
                    
                    pos_centroid = np.array([np.nan, np.nan])
                    neg_centroid = np.array([np.nan, np.nan])
                    
                    if pos_sum > 0:  # Valid positive region
                        pos_weights = pos_pixels / pos_sum  # Normalize weights
                        pos_centroid_y = np.sum(pos_weights * y_coords)
                        pos_centroid_x = np.sum(pos_weights * x_coords)
                        pos_centroid = np.array([pos_centroid_y, pos_centroid_x])
                    
                    if neg_sum > 0:  # Valid negative region
                        neg_weights = np.abs(neg_pixels) / neg_sum  # Normalize weights
                        neg_centroid_y = np.sum(neg_weights * y_coords)
                        neg_centroid_x = np.sum(neg_weights * x_coords)
                        neg_centroid = np.array([neg_centroid_y, neg_centroid_x])
                    
                    # Choose the centroid with the stronger signal
                    if not np.isnan(neg_centroid).any() and not np.isnan(pos_centroid).any():
                        # Both regions valid - choose stronger one
                        if neg_sum >= pos_sum:
                            channel_centers[colour_idx, cell_idx] = neg_centroid
                        else:
                            channel_centers[colour_idx, cell_idx] = pos_centroid
                    elif not np.isnan(pos_centroid).any():
                        # Only positive region
                        channel_centers[colour_idx, cell_idx] = pos_centroid
                    elif not np.isnan(neg_centroid).any():
                        # Only negative region
                        channel_centers[colour_idx, cell_idx] = neg_centroid
                    # If neither valid, remains NaN
                    
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cs_seg', 'minmax', or 'weighted'")
        
        # Calculate true center as mean across colour channels for each cell
        # Shape: (n_cells, 2)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(all='ignore'):  # Suppress numpy warnings
                true_centers = np.nanmean(channel_centers, axis=0)
        
        # Calculate offset vectors: channel_center - true_center
        # Shape: (n_colours, n_cells, 2)
        offsets = channel_centers - true_centers[np.newaxis, :, :]
        
        # Calculate magnitudes: sqrt(dx² + dy²)
        # Shape: (n_colours, n_cells)
        magnitudes = np.sqrt(np.sum(offsets**2, axis=2))
        
        # Calculate angles using arctan2(dy, dx)
        # Shape: (n_colours, n_cells)
        angles_rad = np.arctan2(offsets[:, :, 0], offsets[:, :, 1])  # dy, dx
        angles_deg = np.degrees(angles_rad)
        
        # Set angles to NaN where magnitude is zero or centers are invalid
        zero_magnitude_mask = magnitudes == 0
        invalid_centers_mask = (np.isnan(channel_centers).any(axis=2) | 
                               np.isnan(true_centers[np.newaxis, :, :]).any(axis=2))
        
        angles_deg[zero_magnitude_mask | invalid_centers_mask] = np.nan
        
        # Convert to 0-360° range if requested
        if angle_range_360:
            angles_deg = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
        
        # Convert magnitudes to stimulus units
        magnitudes = magnitudes * self.stim_size
        
        # Plot demo if requested
        if plot or minimal_plot:
            import matplotlib.pyplot as plt
            
            # Determine which ROI to plot
            if roi is None:
                roi = 0
            
            if roi >= n_cells:
                raise ValueError(f"roi {roi} out of range [0, {n_cells-1}]")
            
            # Get RGB representation for the specified ROI using channels 0, 1, 3
            rgb_channels = [0, 1, 3] if self.numcolour > 3 else list(range(min(3, self.numcolour)))
            rgb_image = self.to_rgb(roi=roi, channel=rgb_channels)
            
            # Color scheme for channels
            channel_colors = ['red', 'green', 'blue', 'violet', 'magenta', 'yellow']
            channel_names = ['Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
            
            if minimal_plot:
                # Minimal plot: just RGB background with arrows
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb_image, origin='lower')
                # plt.title(f'ROI {roi}', fontsize=12)
                
                # Plot true center (small white dot)
                true_y, true_x = true_centers[roi]
                if not np.isnan([true_y, true_x]).any():
                    plt.plot(true_x, true_y, 'wo', markersize=5, markeredgecolor='black', markeredgewidth=2)
                
                # Plot arrows only
                for ch_idx in range(n_colours):
                    channel_y, channel_x = channel_centers[ch_idx, roi]
                    
                    if not np.isnan([channel_y, channel_x]).any():
                        color = channel_colors[ch_idx % len(channel_colors)]
                        
                        # Draw arrow from channel center TO true center
                        if not np.isnan([true_y, true_x]).any():
                            plt.annotate('', xy=(true_x, true_y), xytext=(channel_x, channel_y),
                                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.9))
                
                plt.axis('off')  # Remove axes for minimal look
                plt.tight_layout()
                # plt.show()
                
            else:
                # Full plot with all details
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb_image, origin='lower')
                plt.title(f'Colour Channel Offsets - ROI {roi}')
                
                # Plot true center (white circle with black edge)
                true_y, true_x = true_centers[roi]
                if not np.isnan([true_y, true_x]).any():
                    plt.plot(true_x, true_y, 'wo', markersize=15, markeredgecolor='black', 
                            markeredgewidth=3, label=f'True center: ({true_x:.1f}, {true_y:.1f})')
                
                # Collect legend entries for plotting outside the image
                legend_entries = []
                
                # Plot individual channel centers and arrows
                for ch_idx in range(n_colours):
                    channel_y, channel_x = channel_centers[ch_idx, roi]
                    
                    if not np.isnan([channel_y, channel_x]).any():
                        color = channel_colors[ch_idx % len(channel_colors)]
                        
                        # Plot channel center
                        plt.plot(channel_x, channel_y, 'o', color=color, markersize=10, 
                                markeredgecolor='white', markeredgewidth=2)
                        
                        # Draw arrow from channel center TO true center
                        if not np.isnan([true_y, true_x]).any():
                            plt.annotate('', xy=(true_x, true_y), xytext=(channel_x, channel_y),
                                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.8))
                            
                            # Store info for legend (displayed outside plot)
                            mag = magnitudes[ch_idx, roi]
                            angle = angles_deg[ch_idx, roi]
                            legend_entries.append(f'{channel_names[ch_idx]}: ({channel_x:.1f}, {channel_y:.1f}) | {mag:.1f}μm, {angle:.0f}°')
                
                # Create custom legend with channel info outside the plot
                legend_text = '\n'.join(legend_entries)
                if legend_entries:
                    plt.text(1.02, 0.98, legend_text, transform=plt.gca().transAxes, 
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                plt.xlabel('X position (pixels)')
                plt.ylabel('Y position (pixels)')
                plt.tight_layout()
                plt.show()
        
        # Filter results by ROI if specified
        if roi is not None:
            # Handle single ROI or array of ROIs
            if hasattr(roi, '__iter__'):
                roi_indices = roi
            else:
                roi_indices = [roi]
            
            # Validate ROI indices
            for r in roi_indices:
                if r >= n_cells or r < 0:
                    raise ValueError(f"roi {r} out of range [0, {n_cells-1}]")
            
            return {
                'true_centers': true_centers[roi_indices],
                'channel_centers': channel_centers[:, roi_indices],
                'offsets': offsets[:, roi_indices],
                'magnitudes': magnitudes[:, roi_indices],
                'angles': angles_deg[:, roi_indices]
            }
        else:
            # Return all ROIs (default behavior)
            return {
                'true_centers': true_centers,
                'channel_centers': channel_centers,
                'offsets': offsets,
                'magnitudes': magnitudes,
                'angles': angles_deg
            }

    def get_colour_channel_offsets_true_centers(self, roi=None,**kwargs):
        results = self.calc_colour_channel_offsets(roi=roi, **kwargs)
        return results['true_centers']
    
    def get_colour_channel_offsets_magnitudes(self, roi=None,**kwargs):
        results = self.calc_colour_channel_offsets(roi=roi, **kwargs)
        return results['magnitudes']
    
    def get_colour_channel_offsets_angles(self, roi=None,**kwargs):
        results = self.calc_colour_channel_offsets(roi=None, **kwargs)
        return results['angles']
    
    def get_pca_major_axis_lengths(self, roi=None, **kwargs):
        """Get major axis lengths from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['major_axis_length']
    
    def get_pca_minor_axis_lengths(self, roi=None, **kwargs):
        """Get minor axis lengths from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['minor_axis_length']
    
    def get_pca_eccentricities(self, roi=None, **kwargs):
        """Get eccentricity values from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['eccentricity']
    
    def get_pca_orientations(self, roi=None, **kwargs):
        """Get orientation angles from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['angle_degrees']
    
    def get_pca_centroidsX(self, roi=None, **kwargs):
        """Get centroid coordinates from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['centroid_x']
    
    def get_pca_centroidsY(self, roi=None, **kwargs):
        """Get centroid coordinates from PCA analysis"""
        results = self.calc_pca_rf_shape_analysis(roi=roi, **kwargs)
        return results['centroid_y']

    def pca_rf_shape_analysis(self, roi, threshold_sd=3.0, plot=True, force_recompute=False):
        """
        Legacy method - single ROI PCA analysis with plotting
        
        For new code, use calc_pca_rf_shape_analysis() instead
        """
        results = self.calc_pca_rf_shape_analysis(roi=roi, threshold_sd=threshold_sd, 
                                                 plot=plot, force_recompute=force_recompute)
        return results
    
    def get_polarity_labels(self, mode = "spatial"):
        pols = self.get_polarities(mode = mode)
        pols_out = pols.astype(str)
        pols_out[pols == 1] = "ON"
        pols_out[pols == -1] = "OFF"
        pols_out[pols == 2] = "Gabor"
        pols_out[pols == np.nan] = "NaN"
        return pols_out

    def get_polarity_category_cell(self, mask_by_channel=False, threshold=2, dimstr="spatiotemporal") -> str:
        """
        Get polarity category for each cell across colour channels.
        
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

    def get_polarity_category_cell_simple(self, mask_by_channel=False, threshold=2, dimstr="spatiotemporal") -> str:
        """
        Get simplified polarity category for each cell across colour channels.
        Uses same logic as get_polarity_category_cell but only returns 'on', 'off', 'opp', or 'nan'.

        Parameters
        ----------
        mask_by_channel : bool, optional
            Whether to mask polarities using self.bool_by_channel (default: False)
        threshold : float, optional
            Threshold for bool_by_channel masking (default: 2)
        dimstr : str, optional
            Dimension for bool_by_channel ('time' or 'space', default: 'spatiotemporal')

        Returns
        -------
        list of str
            Simplified polarity categories for each cell: 'on', 'off', 'opp', or 'nan'
        """
        result = []
        # Get polarities using same logic as original, then convert 2 → center identity
        polarities = self.get_polarities()

        # For cells marked as opponent (2), convert to center identity based on amplitude sign
        amplitudes = self.get_space_amps()
        amp_signs = np.where(amplitudes > 0, 1, -1)
        polarities = np.where(polarities == 2, amp_signs, polarities)

        arr = pygor.utilities.multicolour_reshape(polarities, self.numcolour).T

        if mask_by_channel:
            # Get boolean mask for significant channels
            bool_mask = self.bool_by_channel(threshold=threshold, dimstr=dimstr)
            # Apply mask to polarities - set insignificant channels to NaN
            for i, (pol_row, mask_row) in enumerate(zip(arr, bool_mask)):
                arr[i] = np.where(mask_row, pol_row, np.nan)

        for i in arr:
            inner_no_nan = np.unique(i)[~np.isnan(np.unique(i))]
            if not any(inner_no_nan):
                result.append('nan')
            elif np.all(inner_no_nan == -1):
                result.append("off")
            elif np.all(inner_no_nan == 1):
                result.append("on")
            elif -1 in inner_no_nan and 1 in inner_no_nan:
                result.append("opp")
            else:
                # Map everything else (mix, other, etc.) to 'nan'
                result.append("nan")
        return result

    def get_time_amps(self, **kwargs) -> np.ndarray:
        maxes = np.max(self.get_timecourses_dominant(**kwargs).data, axis = (1))
        mins = np.min(self.get_timecourses_dominant(**kwargs).data, axis = (1))
        largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
        return largest_mag
    
    def get_time_amps_surr(self) -> np.ndarray:
        maxes = np.max(self.get_timecourses_secondary().data, axis = (1))
        mins = np.min(self.get_timecourses_secondary().data, axis = (1))
        largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
        return largest_mag

    def get_opponency_index_time(self, **kwargs) -> np.ndarray:
        # Get dominant and secondary timecourse amplitudes
        dominant_amps = self.get_time_amps(**kwargs)
        secondary_amps = self.get_time_amps_surr()

        # Check if regions have opposite polarities (signs)
        opposite_polarity = (dominant_amps * secondary_amps) < 0

        # Calculate antagonism index only for opposite polarities
        antagonism_index = np.where(
            opposite_polarity,
            1 - (abs(dominant_amps) - abs(secondary_amps)) / (abs(dominant_amps) + abs(secondary_amps)),
            0  # Same polarity: no true opponency
        )
        return antagonism_index

    def get_time_amps_by_ch(self, ch_idx, **kwargs) -> np.ndarray:
        amps_raw = self.get_time_amps()
        amps_raw_ch_reshape = pygor.utilities.multicolour_reshape(amps_raw, self.numcolour)
        return amps_raw_ch_reshape[ch_idx]

    def get_space_amps(self) -> np.ndarray:
        maxes = np.max(self.collapse_times(), axis = (1, 2))
        mins =  np.min(self.collapse_times(), axis = (1, 2))
        largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
        return largest_mag
    
    def get_space_min(self) -> np.ndarray:
        mins =  np.min(self.collapse_times(), axis = (1, 2))
        return mins
    
    def get_space_max(self) -> np.ndarray:
        maxes = np.max(self.collapse_times(), axis = (1, 2))
        return maxes

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

    def bool_roi_signal(self, threshold=2, dimstr="time") -> np.ndarray:
        """
        Check if ROIs have signal above threshold, automatically handling multicolour data.
        
        Parameters
        ----------
        threshold : float, optional
            Amplitude threshold for signal detection (default: 2)
        dimstr : str, optional
            Dimension for amplitude calculation: "time" or "space" (default: "time")
            
        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs have signal above threshold
        """
        if self.multicolour:
            # Get tuning amplitudes and check if at least one per ROI is above threshold
            tuning_amps = self.calc_tunings_amplitude(dimstr=dimstr)
            return np.any(np.abs(tuning_amps) > threshold, axis=1)
        else:
            if dimstr == "time":
                amps = self.get_time_amps()
            elif dimstr == "space":
                amps = self.get_space_amps()
            return np.abs(amps) > threshold
    
    def bool_strf_signal(self, threshold=2, dimstr="time") -> np.ndarray:
        """
        Legacy method - use bool_roi_signal instead.
        
        Check if STRFs have signal above threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Amplitude threshold for signal detection (default: 2)
        multicolour : bool, optional
            Whether to use multicolour logic (default: True) - IGNORED, uses object's multicolour attribute
        dimstr : str, optional
            Dimension for amplitude calculation: "time" or "space" (default: "time")
            
        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs have signal above threshold
        """
        # Delegate to the new method, ignoring the multicolour parameter
        # and using the object's actual multicolour attribute
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

    def get_spectral_centroid_centre(self) -> np.ndarray:
        vals = np.apply_along_axis(pygor.strf.temporal.only_centroid, 1, self.get_timecourses()[:, 0])
        pass_bool = self.check_cs_pass()
        # nan where pass bools is False
        return np.where(pass_bool, vals, np.nan)

    def get_biphasic_index(self) -> np.ndarray:
        vals = np.apply_along_axis(pygor.strf.temporal.biphasic_index, 1, self.get_timecourses()[:, 0])
        pass_bool = self.check_cs_pass()
        # nan where pass bools is False
        return np.where(pass_bool, vals, np.nan)

    def get_peaktimes(self) -> np.ndarray:
        # times = self.get_timecourses_dominant()
        # strf_dur = self.strf_dur_ms
        # strf_len = self.strfs.shape[1]
        # peak_times_indices = np.array([pygor.strf.temporal.find_peaktime(t) for t in times])
        # scale_factor = strf_dur / strf_len
        # vals = peak_times_indices * scale_factor
        # pass_bool = self.check_cs_pass()
        # # nan where pass bools is False
        # return np.where(pass_bool, vals, np.nan)
        return pygor.strf.temporal.find_peaktime_obj(self)

    def get_strf_peak_times(self, roi=None, mask=None, interpolate=True, upscale_factor=10):
        """
        Get peak times for each pixel in STRF data.

        Parameters:
        -----------
        roi : int or None, optional
            ROI index to analyze. If None, returns peak times for all ROIs (vectorized)
        mask : np.ndarray, float, int, or None, optional
            Mask specification:
            - If np.ndarray: 2D/3D boolean mask - True for pixels to analyze
              For single ROI: 2D mask (height, width)
              For all ROIs: 3D mask (n_rois, height, width)
            - If float or int: Amplitude threshold - pixels with abs(amplitude) > mask are analyzed
            - If None: all pixels are analyzed
        interpolate : bool
            Whether to use interpolation for sub-frame precision
        upscale_factor : int
            Factor to upscale temporal resolution (only used if interpolate=True)

        Returns:
        --------
        strf_max_times : np.ma.MaskedArray or np.ndarray
            - If roi is int: 2D array of peak times in seconds (height, width)
            - If roi is None: 3D array of peak times (n_rois, height, width)
        """
        from scipy.interpolate import interp1d

        # Always compute vectorized for all ROIs
        strf_data = self.strfs_no_border  # (n_rois, n_frames, height, width)
        n_rois, n_frames, height, width = strf_data.shape
        strf_dur_s = self.strf_dur_ms / 1000
        t_original = np.linspace(0, strf_dur_s, n_frames)

        # Handle mask input
        if mask is not None:
            if isinstance(mask, (int, float)):
                # Create amplitude mask for all ROIs
                mask_3d = np.abs(self.collapse_times()) > mask  # (n_rois, height, width)
            elif isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    # 2D mask provided for single ROI case - expand to 3D
                    if roi is not None:
                        mask_3d = np.zeros((n_rois, height, width), dtype=bool)
                        mask_3d[roi] = mask.astype(bool) if mask.dtype != bool else mask
                    else:
                        raise ValueError("2D mask provided but no ROI specified. Use 3D mask or specify roi.")
                elif mask.ndim == 3:
                    mask_3d = mask.astype(bool) if mask.dtype != bool else mask
                else:
                    raise ValueError("Mask must be 2D (for single ROI) or 3D (for all ROIs)")
            else:
                raise TypeError("mask must be a boolean array, float, int, or None")
        else:
            mask_3d = None

        if interpolate:
            # Upscale temporal resolution
            t_upscaled = np.linspace(0, strf_dur_s, n_frames * upscale_factor)

            # Vectorized interpolation: reshape to (n_frames, n_rois * height * width)
            strf_flat = strf_data.reshape(n_rois, n_frames, -1).transpose(1, 0, 2).reshape(n_frames, -1)

            interp_func = interp1d(t_original, strf_flat, kind='cubic', axis=0)
            strf_upscaled_flat = interp_func(t_upscaled)

            # Reshape back to (n_rois, n_frames_upscaled, height, width)
            strf_upscaled = strf_upscaled_flat.reshape(len(t_upscaled), n_rois, height, width).transpose(1, 0, 2, 3)
            time_array = t_upscaled
            data_to_analyze = strf_upscaled
        else:
            time_array = t_original
            data_to_analyze = strf_data

        # Find argmax across time dimension
        if mask_3d is not None:
            # Apply mask: expand to (n_rois, n_frames, height, width)
            mask_expanded = np.repeat(~mask_3d[:, np.newaxis, :, :], data_to_analyze.shape[1], axis=1)
            strf_masked = np.ma.masked_array(data_to_analyze, mask=mask_expanded)
            strf_max_indices = np.ma.argmax(np.ma.abs(strf_masked), axis=1)  # (n_rois, height, width)
            strf_max_times = np.ma.masked_array(time_array[strf_max_indices], mask=~mask_3d)
        else:
            strf_max_indices = np.argmax(np.abs(data_to_analyze), axis=1)  # (n_rois, height, width)
            strf_max_times = time_array[strf_max_indices]

        # Return single ROI if specified
        if roi is not None:
            return strf_max_times[roi]

        return strf_max_times

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
        """
        Plot comprehensive chromatic overview of STRFs showing spatial and temporal components.
        
        Parameters
        ----------
        roi : int, optional
            ROI index to plot. If None, plots all ROIs
        contours : bool, default False
            Whether to add contour lines to spatial plots
        with_times : bool, default False
            Whether to include temporal component plots alongside spatial maps
        colour_idx : int or list, optional
            Which color channel(s) to plot (0=R, 1=G, 2=B, 3=UV). 
            If None, plots all color channels
        **kwargs
            Additional keyword arguments passed to the plotting function
            
        Returns
        -------
        matplotlib figure and axes
            The created figure and axes objects
            
        Examples
        --------
        Plot overview for all colors of ROI 0:
        >>> strf_obj.plot_chromatic_overview(roi=0)
        
        Plot only red and green channels with contours:
        >>> strf_obj.plot_chromatic_overview(roi=0, colour_idx=[0, 1], contours=True)
        
        Plot with temporal components included:
        >>> strf_obj.plot_chromatic_overview(roi=0, with_times=True)
        """
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
            anim = pygor.plotting.play_movie_4d(self.strfs_chroma(), cmap_list =  pygor.plotting.maps_concat, **kwargs)
        else:
            anim = pygor.plotting.play_movie_4d(self.strfs_chroma()[:, roi], cmap_list =  pygor.plotting.maps_concat, **kwargs)
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
        if channels is None:
            channels = np.arange(self.numcolour)
        
        # Get the raw spatiotemporal data (similar to chroma_times in validation script)
        chroma_times = pygor.utilities.multicolour_reshape(self.get_pix_times(), self.numcolour)
        
        cv_values = []
        for roi in range(min(self.num_rois, chroma_times.shape[1])):
            # Calculate max absolute response per colour for this ROI (crude method)
            roi_amplitudes = []
            for c in channels:
                if c < chroma_times.shape[0]:
                    # Take max absolute value across all time and space for this colour/ROI
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

    def get_centre_only_seg(self, border_value=1, min_area=5, include_border=False, use_largest_component=True):
        """
        Get cleaned segmentation areas for all ROIs efficiently.
        
        Uses vectorized operations to process center regions (value==0) from 
        center-surround segmentation maps. Applies morphological cleaning,
        connected component labeling for noise suppression, and sets border 
        pixels to specified value.
        
        Parameters
        ----------
        border_value : int, optional
            Value to assign to border pixels when include_border=True (default: 1)
        min_area : int, optional
            Minimum area for noise removal via erosion/dilation (default: 5)
        include_border : bool, optional
            Whether to set border pixels to border_value (default: False)
        use_largest_component : bool, optional
            Whether to keep only the largest connected component to suppress
            scattered noise pixels (default: True)
            
        Returns
        -------
        np.ndarray
            Cleaned binary masks with shape (n_rois, height, width)
            Center regions = border_value, other regions = 0
            
        Examples
        --------
        >>> # Get cleaned center areas
        >>> clean_masks = obj.get_seg_areas()
        >>> areas = np.sum(clean_masks, axis=(1, 2))
        
        >>> # More aggressive cleaning with largest component selection
        >>> clean_masks = obj.get_seg_areas(border_value=2, min_area=10, use_largest_component=True)
        """
        # Get center-surround segmentation maps
        segmaps, _ = self.cs_seg()

        # Binary mask for center regions (value == 0)
        result = (segmaps == 0).astype(np.uint8)
        
        # Simple noise removal - erode then dilate (equivalent to opening)
        if min_area > 1:
            # Vectorized erosion/dilation
            import scipy.ndimage as ndimage
            structure = np.ones((1, 3, 3))
            result = ndimage.binary_erosion(result, structure=structure).astype(np.uint8)
            result = ndimage.binary_dilation(result, structure=structure).astype(np.uint8)
        
        # Apply connected component labeling to suppress noise (keep only largest component)
        if use_largest_component:
            from skimage import measure
            
            # Process each ROI individually
            for roi_idx in range(result.shape[0]):
                roi_mask = result[roi_idx]
                
                # Skip if no center pixels found
                if np.sum(roi_mask) == 0:
                    continue
                    
                # Label connected components
                labeled_components = measure.label(roi_mask, connectivity=2)
                
                if np.max(labeled_components) > 0:
                    # Find the largest connected component
                    component_sizes = np.bincount(labeled_components.flat)[1:]  # Exclude background (0)
                    if len(component_sizes) > 0:
                        largest_component = np.argmax(component_sizes) + 1
                        # Keep only the largest connected component
                        result[roi_idx] = (labeled_components == largest_component).astype(np.uint8)
        
        # Restore border values after all processing (if borders requested)
        if include_border:
            result[:, 0, :] = border_value
            result[:, -1, :] = border_value  
            result[:, :, 0] = border_value
            result[:, :, -1] = border_value
        
        return result        

    def get_seg_areas(self, border_value=1, min_area=5, include_border=False):
        
        sums = np.sum(self.get_centre_only_seg(border_value=border_value, min_area=min_area, include_border=include_border), axis=(1, 2))
        sums = np.where(sums == (self.strfs_no_border.shape[-1] * self.strfs_no_border.shape[-2]), np.nan, sums)
        sums = np.where(sums == 0, np.nan, sums)
        return sums

    def get_seg_centres(self, roi = None, label = 0, weighted = True, weighting_exp = 3, channel_reshape = False,**kwargs):
        cmaps, _ = self.cs_seg(**kwargs)
        centres_list = []
        for n, segmap in enumerate(cmaps):
            centre_coords = np.argwhere(segmap == label)
            # If label not found in segmap or segmap is uniform, return nan
            if len(centre_coords) == 0 or len(np.unique(segmap)) == 1:
                centres_list.append(np.array([np.nan, np.nan]))
            else:
                if weighted is True:
                    weights = np.abs(np.squeeze(self.collapse_times(n))[centre_coords[:, 0], centre_coords[:, 1]])**weighting_exp
                    # Check if weights sum to zero (would cause ZeroDivisionError)
                    if np.sum(weights) == 0:
                        weights = None  # Fall back to unweighted average
                else:
                    weights = None
                centre = np.average(centre_coords, axis = 0, weights = weights)
                centres_list.append(centre)
        centres = np.array(centres_list)
        if channel_reshape is True and self.multicolour:
            # print(centres.shape)
            centres = np.reshape(centres, (-1, self.numcolour, 2))
        if roi is None:
            return centres
        else:
            return np.squeeze(centres[roi])
    
    def unravel_strf_indices(self, roi_index=None, colour_index=None, flat_index=None):
        """
        Convert between flat STRF indices and (ROI, colour) coordinates.
        
        STRF data is organized as: STRF0_{roi}_{colour}
        where roi ranges from 0 to n_rois-1 and colour ranges from 0 to n_colours-1.
        The flat index follows: flat_index = roi * n_colours + colour
        
        Parameters
        ----------
        flat_index : int, optional
            Flat index to convert to (roi, colour) coordinates
        roi_index : int, optional  
            ROI index to convert (requires colour_index)
        colour_index : int, optional
            Colour index to convert (requires roi_index)
        
        Returns
        -------
        dict
            Dictionary containing:
            - If flat_index provided: {'roi': roi_idx, 'colour': colour_idx}
            - If roi_index and colour_index provided: {'flat_index': flat_idx}
            - If no parameters: {'n_rois': n_rois, 'n_colours': n_colours, 'total_strfs': total}
        
        Examples
        --------
        # Get ROI and colour for flat index 24 (assuming 1 colour channel)
        obj.unravel_strf_indices(flat_index=24)  # {'roi': 24, 'colour': 0}
        
        # Get flat index for ROI 24, colour 2 (assuming 4 colour channels) 
        obj.unravel_strf_indices(roi_index=24, colour_index=2)  # {'flat_index': 98}
        
        # Get dimensions info
        obj.unravel_strf_indices()  # {'n_rois': 25, 'n_colours': 1, 'total_strfs': 25}
        """
        n_colours = getattr(self, 'numcolour', 1)
        
        # Try multiple ways to get total STRF count
        if hasattr(self, 'strf') and self.strf is not None:
            total_strfs = self.strf.shape[0]
        elif hasattr(self, 'data_names') and self.data_names:
            total_strfs = len(self.data_names)
        elif hasattr(self, 'num_rois'):
            total_strfs = self.num_rois * n_colours
        else:
            raise AttributeError("Cannot determine STRF dimensions. Object may not be properly initialized.")
        
        if total_strfs <= 0:
            raise ValueError(f"Invalid total_strfs count: {total_strfs}")
            
        n_rois = total_strfs // n_colours
        
        # Return dimensions if no parameters provided
        if flat_index is None and roi_index is None and colour_index is None:
            return {
                'n_rois': n_rois,
                'n_colours': n_colours, 
                'total_strfs': total_strfs
            }
        
        # Convert flat index to (roi, colour)
        if flat_index is not None:
            if flat_index >= total_strfs or flat_index < 0:
                raise ValueError(f"flat_index {flat_index} out of range [0, {total_strfs-1}]")
            
            roi_idx = flat_index // n_colours
            colour_idx = flat_index % n_colours
            return {'roi': roi_idx, 'colour': colour_idx}
        
        # Convert (roi, colour) to flat index
        if roi_index is not None and colour_index is not None:
            if roi_index >= n_rois or roi_index < 0:
                raise ValueError(f"roi_index {roi_index} out of range [0, {n_rois-1}]")
            if colour_index >= n_colours or colour_index < 0:
                raise ValueError(f"colour_index {colour_index} out of range [0, {n_colours-1}]")
            
            flat_idx = roi_index * n_colours + colour_index
            return {'flat_index': flat_idx}
        
        raise ValueError("Provide either flat_index, or both roi_index and colour_index")

    def calc_centre_surr_vectors(self, roi=None, mode="weighted", angle_range_360=True, threshold=1, plot=False, plot_roi=None):
        """
        Calculate center-surround vector magnitudes and angles for ROIs.
        
        Parameters
        ----------
        mode : str, optional
            Method for finding center and surround:
            - "cs_seg": Use center-surround segmentation (default)
            - "minmax": Find spatial min/max positions with threshold filtering
            - "weighted": Find weighted centroids of thresholded positive/negative regions
        roi : int, array-like, or None
            ROI indices. If None, calculates for all ROIs.
        angle_range_360 : bool, optional
            If True, angles range from 0-360°. If False, angles range from -180 to +180° (default: True)
        threshold : float, optional
            Minimum absolute value for minmax mode (default: 2.0)
        plot : bool, optional
            Whether to plot a demo visualization (default: False)
        plot_roi : int, optional
            Which ROI to plot (if None, uses first ROI or roi[0] if roi specified)

        Returns
        -------
        dict
            Dictionary with 'magnitude' and 'angle_degrees' arrays
            
        Notes
        -----
        Angle Direction Convention:
        - 0° corresponds to East (positive x-direction)
        - 90° corresponds to North (positive y-direction) 
        - 180° corresponds to West (negative x-direction)
        - 270° corresponds to South (negative y-direction)
        
        The angle represents the direction from center to surround position.
        Calculated using np.arctan2(dy, dx) where dy = surround_y - center_y
        and dx = surround_x - center_x.
        
        Weighted Mode Assignment Logic:
        - Center: Negative (OFF) region weighted centroid
        - Surround: Positive (ON) region weighted centroid
        - For cells with only one polarity, both center and surround are assigned 
          the same location (resulting in zero magnitude and undefined angle)
        """
        if mode == "cs_seg":
            # Get center and surround coordinates from segmentation
            centers = self.get_seg_centres(label=0, weighted=True)  # shape: (n_rois, 2)
            surrounds = self.get_seg_centres(label=1, weighted=True)  # shape: (n_rois, 2)
            
        elif mode == "minmax":
            # Vectorized minmax approach for spatial opponency detection
            spaces = self.collapse_times()  # shape: (n_rois, height, width)
            n_rois, height, width = spaces.shape
            
            # Find min/max values and positions for all ROIs vectorized
            spaces_flat = spaces.reshape(n_rois, -1)
            min_vals = np.min(spaces_flat, axis=1)
            max_vals = np.max(spaces_flat, axis=1)
            min_indices = np.argmin(spaces_flat, axis=1)
            max_indices = np.argmax(spaces_flat, axis=1)
            
            # Convert flat indices to 2D coordinates
            min_coords = np.column_stack(np.unravel_index(min_indices, (height, width)))
            max_coords = np.column_stack(np.unravel_index(max_indices, (height, width)))
            
            # Check threshold validity
            min_valid = np.abs(min_vals) >= threshold
            max_valid = np.abs(max_vals) >= threshold
            both_valid = min_valid & max_valid
            
            # Initialize with NaN
            centers = np.full((n_rois, 2), np.nan)
            surrounds = np.full((n_rois, 2), np.nan)
            
            # Case 1: Both valid (spatial opponency) - OFF center, ON surround
            centers[both_valid] = min_coords[both_valid]
            surrounds[both_valid] = max_coords[both_valid]
            
            # Case 2: Only max valid (ON cells) - same position for both
            max_only = max_valid & ~min_valid
            centers[max_only] = max_coords[max_only]
            surrounds[max_only] = max_coords[max_only]
            
            # Case 3: Only min valid (OFF cells) - same position for both
            min_only = min_valid & ~max_valid
            centers[min_only] = min_coords[min_only]
            surrounds[min_only] = min_coords[min_only]
            
        elif mode == "weighted":
            # Weighted centroid approach with bidirectional thresholding
            spaces = self.collapse_times()  # shape: (n_rois, height, width)
            n_rois, height, width = spaces.shape
            
            # Create coordinate grids for weighted centroid calculation
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            centers = np.full((n_rois, 2), np.nan)
            surrounds = np.full((n_rois, 2), np.nan)
            
            for i in range(n_rois):
                space = spaces[i]
                
                # Apply bidirectional thresholding
                pos_mask = space > threshold   # Positive pixels above threshold
                neg_mask = space < -threshold  # Negative pixels below threshold
                
                pos_pixels = space * pos_mask  # Positive values, others zero
                neg_pixels = space * neg_mask  # Negative values, others zero
                
                # Calculate weighted centroids for positive region (ON/surround)
                pos_sum = np.sum(pos_pixels)
                if pos_sum > 0:  # Valid positive region
                    pos_weights = pos_pixels / pos_sum  # Normalize weights
                    pos_centroid_y = np.sum(pos_weights * y_coords)
                    pos_centroid_x = np.sum(pos_weights * x_coords)
                    pos_centroid = np.array([pos_centroid_y, pos_centroid_x])
                else:
                    pos_centroid = np.array([np.nan, np.nan])
                
                # Calculate weighted centroids for negative region (OFF/center)
                neg_sum = np.sum(np.abs(neg_pixels))  # Use absolute values for weighting
                if neg_sum > 0:  # Valid negative region
                    neg_weights = np.abs(neg_pixels) / neg_sum  # Normalize weights
                    neg_centroid_y = np.sum(neg_weights * y_coords)
                    neg_centroid_x = np.sum(neg_weights * x_coords)
                    neg_centroid = np.array([neg_centroid_y, neg_centroid_x])
                else:
                    neg_centroid = np.array([np.nan, np.nan])
                
                # Assign centroids based on what's available
                if not np.isnan(neg_centroid).any() and not np.isnan(pos_centroid).any():
                    # Both regions valid - spatial opponency
                    centers[i] = neg_centroid   # OFF center
                    surrounds[i] = pos_centroid # ON surround
                elif not np.isnan(pos_centroid).any():
                    # Only positive region - ON cell
                    centers[i] = pos_centroid
                    surrounds[i] = pos_centroid  # Same position
                elif not np.isnan(neg_centroid).any():
                    # Only negative region - OFF cell
                    centers[i] = neg_centroid
                    surrounds[i] = neg_centroid  # Same position
                # If neither valid, both remain NaN
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Available: 'cs_seg', 'minmax', 'weighted'")

        # Store unfiltered coordinates for plotting
        plot_centers = centers.copy() if plot else None
        plot_surrounds = surrounds.copy() if plot else None

        # Calculate displacement vectors (surround - center)
        # Note: switching y,x to x,y for conventional cartesian coordinates
        dx = surrounds[:, 1] - centers[:, 1]  # x-component (column direction)
        dy = surrounds[:, 0] - centers[:, 0]  # y-component (row direction)

        # Vectorized magnitude calculation
        magnitudes = np.sqrt(dx**2 + dy**2)

        # Vectorized angle calculation
        angles_rad = np.arctan2(dy, dx)
        if angle_range_360:
            angles_deg = (np.degrees(angles_rad) + 360) % 360  # 0-360 degrees
        else:
            angles_deg = np.degrees(angles_rad)  # -180 to +180 degrees

        # Handle roi selection
        if roi is not None:
            roi = np.atleast_1d(roi)
            magnitudes = magnitudes[roi]
            angles_deg = angles_deg[roi]
            dx = dx[roi]
            dy = dy[roi]

        # Optional plotting for demonstration
        if plot:
            import matplotlib.pyplot as plt
            
            # Determine which ROI to plot and handle indexing correctly
            original_plot_roi = plot_roi
            if plot_roi is None:
                if roi is not None:
                    original_plot_roi = roi[0] if hasattr(roi, '__iter__') else roi
                else:
                    original_plot_roi = 0
            
            # Get spatial representation for the specified ROI (always from original data)
            space = self.collapse_times()[original_plot_roi]
            
            # Center colormap on zero using symmetric limits
            abs_max = np.nanmax(np.abs(space))
            
            plt.figure(figsize=(8, 6))
            plt.imshow(space, cmap='RdBu', origin='lower', vmin=-abs_max, vmax=abs_max)
            plt.colorbar(label='Response amplitude')
            
            # Handle indexing for filtered arrays
            if roi is not None:
                # Arrays are filtered, so use index 0 for single ROI
                array_index = 0
            else:
                # No roi filtering, use original_plot_roi directly  
                array_index = original_plot_roi
            
            # Plot center and surround points if valid (use unfiltered coordinates)
            if mode == "weighted":
                # Use polarity-based labels for weighted mode
                center_coords_text = "OFF (center): NaN"
                surround_coords_text = "ON (surround): NaN"
            else:
                # Use generic center/surround labels for other modes
                center_coords_text = "Center: NaN"
                surround_coords_text = "Surround: NaN"
            
            if not np.isnan(plot_centers[original_plot_roi]).any():
                center_y, center_x = plot_centers[original_plot_roi]
                if mode == "weighted":
                    center_coords_text = f"OFF (center): ({center_x:.1f}, {center_y:.1f})"
                else:
                    center_coords_text = f"Center: ({center_x:.1f}, {center_y:.1f})"
                plt.plot(center_x, center_y, 'wo', markersize=12, markeredgecolor='black', 
                        markeredgewidth=2, label=center_coords_text)
            
            if not np.isnan(plot_surrounds[original_plot_roi]).any():
                surround_y, surround_x = plot_surrounds[original_plot_roi]
                if mode == "weighted":
                    surround_coords_text = f"ON (surround): ({surround_x:.1f}, {surround_y:.1f})"
                else:
                    surround_coords_text = f"Surround: ({surround_x:.1f}, {surround_y:.1f})"
                plt.plot(surround_x, surround_y, 'ko', markersize=12, markeredgecolor='white',
                        markeredgewidth=2, label=surround_coords_text)
            
            # Draw vector if both points are valid
            if (not np.isnan(plot_centers[original_plot_roi]).any() and 
                not np.isnan(plot_surrounds[original_plot_roi]).any()):
                center_y, center_x = plot_centers[original_plot_roi]
                surround_y, surround_x = plot_surrounds[original_plot_roi]
                
                plt.annotate('', xy=(surround_x, surround_y), xytext=(center_x, center_y),
                            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
                
                # Add vector info
                mag = magnitudes[array_index] if hasattr(magnitudes, '__len__') else magnitudes
                angle = angles_deg[array_index] if hasattr(angles_deg, '__len__') else angles_deg
                plt.text(0.02, 0.98, f'ROI {original_plot_roi}\nMagnitude: {mag:.2f}\nAngle: {angle:.1f}deg\nMode: {mode}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.02, 0.98, f'ROI {original_plot_roi}\nNo valid vector\nMode: {mode}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'Center-Surround Vector Analysis (ROI {original_plot_roi})')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            'magnitude': np.squeeze(magnitudes),
            'angle_degrees': np.squeeze(angles_deg), 
            'dx': np.squeeze(dx),  # bonus: cartesian components
            'dy': np.squeeze(dy)
        }
    # Convenience functions to get just angles or magnitudes with optional masking

    def get_centre_surr_angles(self, roi = None, mask_singlepol = False):
        vectors = self.calc_centre_surr_vectors(roi=roi, mode = "weighted")
        if mask_singlepol is True:
            all_labels = self.get_polarities()
            masks = all_labels == 2
            # Apply masking based on roi selection
            if roi is not None:
                mask_subset = masks[roi]
            else:
                mask_subset = masks.flatten()
            vectors['angle_degrees'] = np.where(mask_subset, vectors['angle_degrees'], np.nan)
        return vectors['angle_degrees']
    
    def get_centre_surr_magnitudes(self, roi = None, mask_singlepol = False):
        vectors = self.calc_centre_surr_vectors(roi=roi, mode = "weighted")
        if mask_singlepol is True:
            all_labels = self.get_polarities()
            masks = all_labels == 2
            # Apply masking based on roi selection
            if roi is not None:
                mask_subset = masks[roi]
            else:
                mask_subset = masks
            vectors['magnitude'] = np.where(mask_subset, vectors['magnitude'], np.nan)
        return vectors['magnitude']

    
    def get_centre_surr_x(self, **kwargs):
        return self.calc_centre_surr_vectors(**kwargs)['dx']
    
    def get_centre_surr_y(self, **kwargs):
        return self.calc_centre_surr_vectors(**kwargs)['dy']

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
        For multicolour data, STRF indices are organized as [roi0_colour0, roi0_colour1, ..., roi1_colour0, ...].
        Use multicolour_reshape() or manual indexing to organize by colour channels if needed.
        """
        import pygor.strf.extrema_timing as extrema_timing
        
        return extrema_timing.map_spectral_centroid_wrapper(
            self, roi=roi, exclude_firstlast=exclude_firstlast,
            return_milliseconds=return_milliseconds, frame_rate_hz=frame_rate_hz
        )

    # def compare_colour_channel_timing(self, roi, colour_channels=(0, 1), threshold=3.0,
    #                                exclude_firstlast=(1, 1), return_milliseconds=False, 
    #                                frame_rate_hz=60.0):
    #     """
    #     Compare extrema timing between different colour channels for a single ROI.
        
    #     Parameters
    #     ----------
    #     roi : int
    #         ROI index to analyze
    #     colour_channels : tuple of int, optional
    #         Two colour channel indices to compare. Default is (0, 1).
    #     threshold : float, optional
    #         Threshold in standard deviations. Default is 3.0.
    #     exclude_firstlast : tuple of int, optional
    #         Number of time points to exclude from beginning and end. Default is (1, 1).
    #     return_milliseconds : bool, optional
    #         If True, convert timing to milliseconds. Default is False.
    #     frame_rate_hz : float, optional
    #         Frame rate for millisecond conversion. Default is 60.0.
            
    #     Returns
    #     -------
    #     timing_difference : ndarray
    #         2D array (y, x) of timing differences (channel2 - channel1).
    #         NaN where either channel is below threshold.
    #     """
    #     import pygor.strf.extrema_timing as extrema_timing
        
    #     return extrema_timing.compare_colour_channel_timing_wrapper(
    #         self, roi=roi, colour_channels=colour_channels, threshold=threshold,
    #         exclude_firstlast=exclude_firstlast, return_milliseconds=return_milliseconds,
    #         frame_rate_hz=frame_rate_hz
    #     )

    def analyze_spatial_alignment(self, roi, threshold=3.0, reference_channel=0, 
                                collapse_method='peak'):
        """
        Analyze spatial alignment across all colour channels for a single ROI.
        
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
            - 'correlation_matrix': n_colours × n_colours spatial correlation matrix
            - 'overlap_matrix': n_colours × n_colours Jaccard index matrix
            - 'distance_matrix': n_colours × n_colours centroid distance matrix
            - 'summary_stats': Summary statistics across all channel pairs
            - 'channel_centroids': Centroids for each colour channel
            - 'spatial_maps': 2D spatial maps for each colour channel
            - 'pairwise_metrics': Detailed pairwise comparison dictionary
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.analyze_multicolour_spatial_alignment(
            self, roi=roi, threshold=threshold, reference_channel=reference_channel,
            collapse_method=collapse_method
        )

    def compute_colour_channel_overlap(self, roi, colour_channels=(0, 1), threshold=3.0, 
                                    collapse_method='peak'):
        """
        Compute spatial overlap metrics between two specific colour channels.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        colour_channels : tuple of int, optional
            Two colour channel indices to compare. Default is (0, 1).
        threshold : float, optional
            Threshold for defining active regions. Default is 3.0.
        collapse_method : str, optional
            Method for collapsing time dimension. Options: 'peak', 'std', 'sum'. Default is 'peak'.
        
        Returns
        -------
        dict : Dictionary containing spatial overlap metrics
        """
        import pygor.strf.spatial_alignment as spatial_alignment
        
        return spatial_alignment.compute_colour_channel_overlap_wrapper(
            self, roi=roi, colour_channels=colour_channels, threshold=threshold,
            collapse_method=collapse_method
        )

    def compute_spatial_offset_between_channels(self, roi, colour_channels=(0, 1), 
                                              threshold=3.0, method='centroid', 
                                              collapse_method='peak'):
        """
        Compute spatial offset between two colour channels.
        
        Parameters
        ----------
        roi : int
            ROI index to analyze
        colour_channels : tuple of int, optional
            Two colour channel indices to compare. Default is (0, 1).
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
            self, roi=roi, colour_channels=colour_channels, threshold=threshold,
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
        Calculate spike-triggered averages (STRFs) for all ROIs and colour channels.
        
        This method computes spatiotemporal receptive fields using spike-triggered 
        averaging with multi-colour noise stimuli, based on the Igor Pro implementation
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
            Number of colour channels in the stimulus (use 1 for single-colour experiments)
        n_triggers_per_colour : int or None, default None
            Number of triggers per colour channel. If None, will be auto-calculated
            for single-colour experiments or must be provided for multi-colour.
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
            - 'mean_stimulus': Mean stimulus for each colour (y, x, n_colours)
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
        
        # Auto-detect multi-colour experiments if object has numcolour attribute
        if n_colours == 1 and hasattr(self, 'numcolour') and self.numcolour > 1:
            n_colours = self.numcolour
            if verbose:
                print(f"Multi-colour experiment detected: using {n_colours} colours")
        
        # Use multi-colour optimized version for better performance when n_colours > 1
        if n_colours > 1:
            if verbose:
                print("Using multi-colour optimized implementation for enhanced performance...")
            results = pygor.strf.calculate_multicolour_optimized.calculate_calcium_correlated_average_multicolour_optimized(
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
            # Use regular optimized version for single-colour
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

    def get_npercent_times(self, roi=None, percent=10, incl_borders=False, bidirectional=False, use_percentile=False):
        """
        Extract timecourses from the highest amplitude pixels in the collapsed STRF.
        
        Parameters:
        -----------
        roi : int
            ROI index to analyze
        percent : float
            If use_percentile=False: Percentage of pixels to extract (default 10)
            If use_percentile=True: Percentile threshold (e.g., 90 for 90th percentile, default 10)
        incl_borders : bool
            Whether to include borders in pixel times extraction (default False)
        bidirectional : bool
            If True, takes half from highest pixels and half from lowest pixels (default False)
        use_percentile : bool
            If True, uses percentile thresholding instead of percentage selection (default False)
            
        Returns:
        --------
        selected_timecourses : np.ndarray
            Array of shape (n_timepoints, n_selected_pixels) containing timecourses
        selected_indices : np.ndarray
            Flat indices of the selected pixels (sorted by amplitude, highest first)
        selected_amplitudes : np.ndarray
            Amplitude values of the selected pixels (sorted descending)
        """
        if roi is None:
            raise ValueError("ROI must be specified.")

        # Get collapsed STRF - use raw values for bidirectional, abs for unidirectional
        curr_map_raw = self.collapse_times()[roi]
        
        if use_percentile:
            # Use percentile thresholding
            if bidirectional:
                # For bidirectional: use raw values to separate positive from negative
                # Calculate percentile thresholds
                high_threshold = np.percentile(curr_map_raw, percent)
                low_threshold = np.percentile(curr_map_raw, 100 - percent)
                
                # Get pixels above/below thresholds
                high_mask = curr_map_raw >= high_threshold
                low_mask = curr_map_raw <= low_threshold
                
                # Get indices
                high_indices = np.where(high_mask.ravel())[0]
                low_indices = np.where(low_mask.ravel())[0]
                
                # Sort by amplitude within each group
                high_vals = curr_map_raw.ravel()[high_indices]
                low_vals = curr_map_raw.ravel()[low_indices]
                
                high_sort = np.argsort(high_vals)[::-1]  # Descending
                low_sort = np.argsort(low_vals)  # Ascending (most negative first)
                
                # Combine indices: high first, then low
                top_indices = np.concatenate([high_indices[high_sort], low_indices[low_sort]])
                
                # Get the raw amplitude values for selected pixels
                selected_amplitudes = curr_map_raw.ravel()[top_indices]
                
            else:
                # For unidirectional: use absolute values
                curr_map = np.abs(curr_map_raw)
                threshold = np.percentile(curr_map, percent)
                
                # Get pixels above threshold
                mask = curr_map >= threshold
                top_indices = np.where(mask.ravel())[0]
                
                # Sort by absolute amplitude (descending)
                vals = curr_map.ravel()[top_indices]
                sort_order = np.argsort(vals)[::-1]
                top_indices = top_indices[sort_order]
                
                # Get the absolute amplitude values for selected pixels
                selected_amplitudes = curr_map.ravel()[top_indices]
                
        else:
            # Use percentage selection (original behavior)
            total_pixels = curr_map_raw.size
            n_pixels_to_select = max(1, int(total_pixels * percent / 100))
            
            if bidirectional:
                # For bidirectional: use raw values to separate positive from negative
                # Sort pixel coordinates by their raw amplitude (ascending order)
                sorted_indices = np.argsort(curr_map_raw, axis=None)
                
                # Split selection between highest and lowest raw amplitude pixels
                n_high = n_pixels_to_select // 2
                n_low = n_pixels_to_select - n_high  # Handle odd numbers
                
                # Get lowest amplitude pixels (most negative - first n_low indices)
                low_indices = sorted_indices[:n_low]
                
                # Get highest amplitude pixels (most positive - last n_high indices)  
                high_indices = sorted_indices[-n_high:]
                
                # Combine indices: high first, then low
                top_indices = np.concatenate([high_indices[::-1], low_indices])
                
                # Get the raw amplitude values for selected pixels
                selected_amplitudes = curr_map_raw.ravel()[top_indices]
                
            else:
                # For unidirectional: use absolute values to get strongest responses regardless of polarity
                curr_map = np.abs(curr_map_raw)
                sorted_indices = np.argsort(curr_map, axis=None)
                
                # Get the top n% pixels (highest amplitudes)
                top_indices = sorted_indices[-n_pixels_to_select:]
                
                # Reverse indices to match descending amplitude order
                top_indices = top_indices[::-1]
                
                # Get the absolute amplitude values for selected pixels
                selected_amplitudes = curr_map.ravel()[top_indices]
        
        # Extract timecourses using get_pix_times
        pix_times = self.get_pix_times(incl_borders=incl_borders)[roi]  # Shape: (n_timepoints, n_pixels)
        selected_timecourses = pix_times[:, top_indices]  # Shape: (n_timepoints, n_selected_pixels)
        
        return selected_timecourses#, top_indices, selected_amplitudes

    def napari_strfs(self, **kwargs):
        import pygor.strf.gui.methods as gui
        napari_session = gui.NapariSession(self)
        return napari_session.run()


# Auto-generate _by_channel methods for all callable methods
# This provides multicolour reshaping functionality for any method that returns array-like data

def _create_by_channel_method(method_name):
    """Create a _by_channel wrapper that applies multicolour reshaping."""
    def by_channel_wrapper(self, ch_idx=None, **kwargs):
        result = getattr(self, method_name)(**kwargs)
        reshaped = pygor.utilities.multicolour_reshape(result, self.numcolour)
        
        # If ch_idx specified, return only that channel/channels
        if ch_idx is not None:
            if not self.multicolour:
                # For single colour, validate the index/indices
                if hasattr(ch_idx, '__iter__') and not isinstance(ch_idx, str):
                    # Handle list/array of indices
                    for idx in ch_idx:
                        if idx != -1 and idx != 0:
                            raise ValueError(f"ch_idx contains {idx} but object is not multicolour (only has 1 channel, use 0 or -1)")
                    return reshaped[0]  # Always return the single channel for any valid indices
                else:
                    # Handle single index
                    if ch_idx != -1 and ch_idx != 0:
                        raise ValueError(f"ch_idx={ch_idx} specified but object is not multicolour (only has 1 channel, use 0 or -1)")
                    return reshaped[0]  # Single channel case
            else:
                return reshaped[ch_idx]
        
        # Otherwise return all channels
        return reshaped
    
    by_channel_wrapper.__name__ = f"{method_name}_by_channel"
    by_channel_wrapper.__doc__ = f"Multicolour-reshaped version of {method_name}(). Returns shape (n_colours, n_rois_per_colour) instead of flattened (n_total_rois,). Use ch_idx parameter to get specific channel."
    return by_channel_wrapper

# Dynamically add _by_channel methods to ALL callable methods in STRF class
_SKIP_METHODS = {
    '__init__', '__new__', '__str__', '__repr__', '__getattr__', '__setattr__',
    'napari_strfs',  # GUI method
}

for attr_name in dir(STRF):
    # Skip private methods, already existing _by_channel methods, and special cases
    if (not attr_name.startswith('_') and 
        not attr_name.endswith('_by_channel') and 
        attr_name not in _SKIP_METHODS):
        
        attr = getattr(STRF, attr_name)
        if callable(attr):  # Only add to callable methods
            setattr(STRF, f"{attr_name}_by_channel", _create_by_channel_method(attr_name))


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