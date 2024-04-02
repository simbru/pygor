"""
Depricated island-analysis scritps. Moved here for safe-keeping if needed later.
"""

raise DeprecationWarning("Do not use. These fucntinos are depricated and have dependencies that are no longer applicable. AVOID!")

def get_covariance_trace(coordinates, type = "cov", **kwargs):
    no_nan_coords = np.nan_to_num(coordinates) # (deal with nans)
     # get no:nan_coords with no zeros (this way is better)
    coords_only = coordinates.T[~np.isnan(coordinates.T).any(axis=1)].T # Get rid of nans
    if type == "distance":
        return NotImplementedError("Average neighbouring distance algo not implemented. See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html")

    if type == "cov" or type == "covariance" or type == None:
        # Center coordinates on centroid 
        centroid_loc = centroid(coords_only)

        coordinates = (30, 40) # Does this do anything? 

        x_cords_centered = coords_only[0] - centroid_loc[0]
        y_cords_centered = coords_only[1] - centroid_loc[1]

        coords_centered  = np.stack((x_cords_centered, y_cords_centered))
        # Get covariance matrix for coordinates
        input_coords_cov = np.cov(coords_only)
        coords_centered_cov = np.cov(coords_centered)
        try:
            # Get trace (sum of main diagonal of covariance matrix)
            input_trace = coords_centered_cov.trace()
        except ValueError:
            raise ValueError("Encountered error when trying to get trace (diagonal) of covariance matrix. Likley this is due to covariance matrix being 1d. Check if 'coordinates'-input is correct.")

        # # Cov determinants 
        # input_determinant = np.linalg.det(input_coords_cov)
        # distal_determinant = np.linalg.det(distal_coords_cov)

        # # Cov eigenvalues 
        # input_eigs, input_unit_eig = np.linalg.eig(input_coords_cov)
        # distal_eigs, distal_unit_eig = np.linalg.eig(distal_coords_cov)

        if 'plot' in kwargs:
            plt_arr = np.array([top_l, top_r, bottom_l, bottom_r])
            plt.scatter(plt_arr[:, 0], plt_arr[:, 1], c = 'b', s = 100)
            plt.scatter(coordinates[:, 0], coordinates[:, 1], c = 'r')

        return input_trace

def bootstrap_random_distribution(_bootstrap_reps, _max_possible_isl_num, **kwargs):
    """Estimate random spread via bootstrap """
    # Build random coordinates (many many times) for range of island numbers and get the covariance matrix trace
    _cached_cov_traces = np.zeros((_max_possible_isl_num, _bootstrap_reps))
    for _isl_n in range(2, _max_possible_isl_num + 2):
        for _bootstrap_n in range(_bootstrap_reps):
            _y_cords = np.random.randint(2, 60, (2, _isl_n))[0]
            _x_cords = np.random.randint(2, 80, (2, _isl_n))[0]
            # plt.scatter(_x_cords, _y_cords)
            _curr_coords = np.stack((_x_cords, _y_cords))
            _cached_cov_traces[_isl_n - 2, _bootstrap_n] = get_covariance_trace(_curr_coords)
    # Then feed those traces forward to do kernal density estimation 
    max_val = math.ceil(np.max(_cached_cov_traces)) # Ceil to avoid integer error when priming array
    _kde_cache = np.zeros((_max_possible_isl_num, max_val))
    for i in range(_max_possible_isl_num):
        _max_cached_cov_trace = np.max(_cached_cov_traces[i])
        _curr_kde = scipy.stats.gaussian_kde(_cached_cov_traces[i]).pdf(np.arange(_max_cached_cov_trace))
        _len_difference = _curr_kde.shape[0] - _kde_cache.shape[1]
        _curr_kde_padded = np.pad(_curr_kde, (0, abs(_len_difference)), mode = "constant")
        _kde_cache[i] = _curr_kde_padded
    return _cached_cov_traces, _kde_cache

def spread_index(coordinates, bs_covtraces, bs_covtraces_KDEs):
    raise DeprecationWarning("Depricated. Use 'dispertion_index()' instead!")
    # Figure out how many islands are present in input coordinates
    isl_num = np.sum(~np.isnan(coordinates)[0]) - 1
    # Compute trace of vocariance matrix for input coordinates
    trace = get_covariance_trace(coordinates)
    # Get all the stats to do things
    mean = np.mean(bs_covtraces[isl_num])
    median = np.median(bs_covtraces[isl_num])
    median_pop = np.median(bs_covtraces)
    std = np.std(bs_covtraces[isl_num])
    max_pop = np.max(bs_covtraces)
    max_samp = np.max(bs_covtraces[isl_num])
    min_pop = np.min(bs_covtraces)
    min_samp = np.min(bs_covtraces[isl_num])
    # Min max scale covariance traces (based on bootstrapped (bs) data)
    a, b = 0, 1
    min_max_norm_trace = ((trace - min_pop) / (max_pop - min_pop))
    # The index itself
    index = (min_max_norm_trace / np.sqrt(min_max_norm_trace))

    return index 

def dispertion_index(covariance_trace, bs_covtraces, bs_covtraces_KDEs):
    # Get all the stats to do things
    # mean = np.mean(bs_covtraces) # [isl_num]
    # median = np.median(bs_covtraces)
    # median_pop = np.median(bs_covtraces)
    # std = np.std(bs_covtraces)
    max_pop = np.max(bs_covtraces)
    # max_samp = np.max(bs_covtraces)
    min_pop = np.min(bs_covtraces)
    # min_samp = np.min(bs_covtraces)
    # Min max scale covariance traces (based on bootstrapped (bs) data)
    a, b = 0, 1
    min_max_norm_trace = ((covariance_trace - min_pop) / (max_pop - min_pop))
    # The index itself
    index = (min_max_norm_trace / np.sqrt(min_max_norm_trace))
    return index 
