@dataclass
class Data_strf(Data):
    # Levae these intact
    strfs        : np.ndarray = np.nan
    ipl_depths   : np.ndarray = np.nan
    strf_keys    : np.ndarray = np.nan
    multicolour  : bool = False
    do_bootstrap : bool = True
    type         : str = "STRF"
    # Params 
    time_sig_thresh: float = 0.1
    space_sig_thresh: float = 0.1
    time_bs_n    : int = 2500
    space_bs_n   : int = 1000

    numcolour    : int = field(init=False)
    filename     : str = field(init=False)
    name         : str = field(init=False)
    def __post_init__(self):
        self.numcolour = len(np.unique([int(i.split('_')[-1]) for i in self.strf_keys]))
        self.filename  = pathlib.Path(self.metadata["filename"])
        self.name  = self.filename.stem
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
                # print(f"Hang on, bootstrapping temporal components {self.time_bs_n} times")
                #bar = alive_it(self.strfs, force_tty=True, title = f"Hang on, bootstrapping temporal components {self.time_bs_n} times")
                bar = tqdm(self.strfs, leave = True, position = 0, disable = None, 
                    desc = f"Hang on, bootstrapping temporal components {self.time_bs_n} times")
                self._pval_time = np.array([signal_analysis.bootstrap_time(x, bootstrap_n=self.time_bs_n) for x in bar])
                #clear_output(wait = True)
                # bar.container.close()
                return self._pval_time

    @property
    def pval_space(self):
        if self.do_bootstrap == False:
            return [np.nan] * self.num_strfs
        if self.do_bootstrap == True:
            try:
                return self._pval_space
            except AttributeError:
                # print(f"Hang on, bootstrapping spatial components {self.space_bs_n} times")
                # bar = alive_it(self.strfs, force_tty=True, title = f"Hang on, bootstrapping spatial components {self.space_bs_n} times")
                bar = tqdm(self.strfs, leave = True, position = 0, disable = None,
                    desc = f"Hang on, bootstrapping spatial components {self.space_bs_n} times")
                self._pval_space = np.array([signal_analysis.bootstrap_space(x, bootstrap_n=self.space_bs_n) for x in bar])
                #clear_output(wait = True)
                # bar.reset()
                # bar.close()
                return self._pval_space
 
    def pvals_table(self):
        dict = {}
        if self.multicolour == True: 
            space_vals = utilities.multicolour_reshape(np.array(self.pval_space), self.numcolour).T
            time_vals = utilities.multicolour_reshape(np.array(self.pval_time), self.numcolour).T
            space_sig = space_vals < self.space_sig_thresh
            time_sig = time_vals < self.time_sig_thresh
            both_sig = time_sig * space_sig
            final_arr = np.hstack((space_vals, time_vals, both_sig), dtype="object")
            column_labels = ["space_R", "space_G", "space_B", "space_UV", "time_R", "time_G", "time_B", "time_UV",
            "sig_R", "sig_G", "sig_B", "sig_UV"]
            return pd.DataFrame(final_arr, columns=column_labels)
        else:
            space_vals = self.pval_space
            time_vals = self.pval_space
            space_sig = space_vals <  self.space_sig_thresh
            time_sig = time_vals < self.time_sig_thresh
            both_sig = time_sig * space_sig
            final_arr = np.stack((space_vals, time_vals, both_sig), dtype = "object").T
            column_labels = ["space", "time", "sig"]
            return pd.DataFrame(final_arr, columns = column_labels)


    @property
    def contours(self):
        try:
            return self._contours
        except AttributeError:
            #self._contours = [space.contour(x) for x in self.collapse_times()]
            if self.do_bootstrap == True:
                _contours = [contouring.contour(arr) # ensures no contour is drawn if pval not sig enough
                                if self.pval_time[count] < self.time_sig_thresh and self.pval_space[count] < self.space_sig_thresh
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
        return contouring.complexity_weighted(self.contours, self.contours_areas())

    @property
    def timecourses(self, centre_on_zero = True):
        try:
            return self._timecourses 
        except AttributeError:
            timecourses = np.average(self.strf_masks(), axis = (3,4))
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

    def strf_masks(self, level = None):
        if self.strfs is np.nan:
            return np.nan
        else:
            # Apply space.rf_mask3d to all arrays and return as a new masksed array
            all_strf_masks = np.ma.array([space.rf_mask3d(x, level = None) for x in self.strfs])
            # # Get masks that fail criteria
            pval_fail_time = np.argwhere(np.array(self.pval_time) > self.time_sig_thresh) # nan > thresh always yields false, so thats really convenient 
            pval_fail_space = np.argwhere(np.array(self.pval_space) > self.space_sig_thresh) # because if pval is nan, it returns everything
            pval_fail = np.unique(np.concatenate((pval_fail_time, pval_fail_space)))
            # Set entire mask to True conditionally 
            all_strf_masks.mask[pval_fail] = True
        return all_strf_masks

    ## Methods 
    def calc_LED_offset(self, reference_LED_index = [0,1,2], compare_LED_index = [3]):
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
            avg_colour_centre = np.array([np.nanmean(yx, axis = 0) for yx in utilities.multicolour_reshape(self.contours_centres, self.numcolour)])
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
        # Feed that to helper function to break it down into 1D/category
        return utilities.polarity_neat(polarities)

    def opponency_bool(self):
        if self.multicolour == True:
            arr = utilities.multicolour_reshape(self.polarities(), self.numcolour).T
            # This line looks through rearranged chromatic arr roi by roi 
            # and checks the poliarty by getting the unique values and checking 
            # if the length is more than 0 or 1, excluding NaNs. 
            # I.e., if there is only 1 unique value, there is no opponency
            opponent_bool = [False if len(np.unique(i[~np.isnan(i)])) == 1|0 else True for i in arr]
            return opponent_bool
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour' is False")

    def polarity_category(self):
        result = []
        arr = utilities.multicolour_reshape(self.polarities(), self.numcolour).T
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

    def amplitude_tuning_functions(self):
        if self.multicolour == True:
            # maxes = np.max(self.collapse_times().data, axis = (1, 2))
            # mins = np.min(self.collapse_times().data, axis = (1, 2))
            maxes = np.max(self.dominant_timecourses().data, axis = (1))
            mins = np.min(self.dominant_timecourses().data, axis = (1))
            largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
            largest_by_colour = utilities.multicolour_reshape(largest_mag, self.numcolour)
            # signs = np.sign(largest_by_colour)
            # min_max_scaled = np.apply_along_axis(utilities.min_max_norm, 1, np.abs(largest_by_colour), 0, 1)
            tuning_functions = largest_by_colour
            return tuning_functions.T #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour.' is False")

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
            area_by_colour = utilities.multicolour_reshape(total_areas, self.numcolour).T
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
            speed_by_colour_neg = utilities.multicolour_reshape(neg_centroids, self.numcolour).T
            speed_by_colour_pos = utilities.multicolour_reshape(pos_centroids, self.numcolour).T 
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
            argmins  = utilities.multicolour_reshape(argmins, self.numcolour).T
            argmaxs  = utilities.multicolour_reshape(argmaxs, self.numcolour).T
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
        mean_pols_by_roi = np.nanmean(result_values.reshape(self.numcolour, -1), axis=0)
        return (np.sum(mean_pols_by_roi[:int(len(mean_pols_by_roi)/2)]), np.sum(mean_pols_by_roi[int(len(mean_pols_by_roi)/2):]))

    # def save_pkl(self, save_path, filename):
    #     fileehandling.save_pkl(self, save_path, filename)
    def save_pkl(self, save_path, filename):
        final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
        print("Storing as:", final_path, end = "\r")
        with open(final_path, 'wb') as outp:
            joblib.dump(self, outp, compress='zlib')

