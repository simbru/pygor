import numpy as np


def compute_vector_magnitude(responses, directions_deg):
    """
    Compute vector magnitude (r) from circular statistics.
    
    This measures how directionally tuned the responses are.
    r = 1 means perfectly tuned, r = 0 means no directional preference.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    float : Vector magnitude (0 ≤ r ≤ 1)
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    
    # Compute weighted mean vector components
    total_response = np.sum(responses)
    if total_response == 0:
        return 0
    
    mean_x = np.sum(responses * np.cos(directions_rad)) / total_response
    mean_y = np.sum(responses * np.sin(directions_rad)) / total_response
    
    # Vector magnitude
    r = np.sqrt(mean_x**2 + mean_y**2)
    return r


def compute_circular_variance(responses, directions_deg):
    """
    Compute circular variance (CV = 1 - r).
    
    CV = 0 means perfectly tuned, CV = 1 means no directional preference.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    float : Circular variance (0 ≤ CV ≤ 1)
    """
    r = compute_vector_magnitude(responses, directions_deg)
    return 1 - r


def compute_direction_selectivity_index(responses, directions_deg):
    """
    Compute directional selectivity index (DSI) and related metrics, vectorized.
    
    DSI = (R_preferred - R_opposite) / (R_preferred + R_opposite)
    
    Parameters:
    -----------
    responses : array-like
        Response values. Can be 1D (n_directions), 2D (n_rois, n_directions), 
        or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        1D array of direction values in degrees.
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'dsi': Directional selectivity index (-1 to 1). Shape matches input `responses` without direction axis.
        - 'preferred_direction': Preferred direction in degrees (0-360).
        - 'opposite_direction': Opposite direction in degrees (0-360).
        - 'preferred_response': Response at preferred direction.
        - 'opposite_response': Response at opposite direction.
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    
    # Ensure at least 1D
    if responses.ndim == 0:
        responses = responses[np.newaxis]

    # Find preferred direction responses and indices
    preferred_indices = np.argmax(responses, axis=-1)
    preferred_responses = np.max(responses, axis=-1)
    
    # Get preferred directions in degrees
    preferred_dirs = directions_deg[preferred_indices]
    
    # Calculate target opposite directions (180° away)
    opposite_dirs_target = (preferred_dirs + 180) % 360
    
    # Find the actual closest directions available in the data
    # This requires broadcasting and careful indexing
    dir_diffs = np.abs(directions_deg - opposite_dirs_target[..., np.newaxis])
    dir_diffs = np.minimum(dir_diffs, 360 - dir_diffs)  # Handle wrap-around
    opposite_indices = np.argmin(dir_diffs, axis=-1)
    
    # Get opposite responses using advanced indexing
    # Create indices for all dimensions except the last one
    if responses.ndim > 1:
        # Create a meshgrid of indices for the preceding dimensions
        # For a 3D array (n_phases, n_rois, n_dirs), this creates indices for phases and ROIs
        indices = np.indices(responses.shape[:-1])
        opposite_responses = responses[(*indices, opposite_indices)]
    else:
        # 1D case is simpler
        opposite_responses = responses[opposite_indices]

    # Compute DSI
    denominator = preferred_responses + opposite_responses
    dsis = np.where(denominator == 0, 0, (preferred_responses - opposite_responses) / denominator)
    
    return {
        'dsi': dsis,
        'preferred_direction': preferred_dirs,
        'opposite_direction': directions_deg[opposite_indices],
        'preferred_response': preferred_responses,
        'opposite_response': opposite_responses
    }


def compute_preferred_direction(responses, directions_deg):
    """
    Find the preferred direction angle.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    float : Preferred direction in degrees
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    
    preferred_idx = np.argmax(responses)
    return directions_deg[preferred_idx]


def compute_mean_direction(responses, directions_deg):
    """
    Compute the mean direction using circular statistics.
    
    This gives the direction of the mean vector, which may differ from
    the preferred direction if responses are broadly tuned.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    float : Mean direction in degrees
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    
    # Compute weighted mean vector components
    total_response = np.sum(responses)
    if total_response == 0:
        return np.nan
    
    mean_x = np.sum(responses * np.cos(directions_rad)) / total_response
    mean_y = np.sum(responses * np.sin(directions_rad)) / total_response
    
    # Convert back to degrees
    mean_direction_rad = np.arctan2(mean_y, mean_x)
    mean_direction_deg = np.rad2deg(mean_direction_rad)
    
    # Ensure positive angle
    if mean_direction_deg < 0:
        mean_direction_deg += 360
    
    return mean_direction_deg


def extract_direction_vectors(responses, directions_deg):
    """
    Extract individual direction vectors from directional responses.
    
    Each direction + response pair forms a vector in polar coordinates.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'angles': Direction angles in degrees
        - 'magnitudes': Response magnitudes
        - 'cartesian_x': X components of vectors
        - 'cartesian_y': Y components of vectors
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    directions_rad = np.deg2rad(directions_deg)
    
    # Cartesian components for vector addition
    cartesian_x = responses * np.cos(directions_rad)
    cartesian_y = responses * np.sin(directions_rad)
    
    return {
        'angles': directions_deg,
        'magnitudes': responses,
        'cartesian_x': cartesian_x,
        'cartesian_y': cartesian_y
    }


def extract_mean_vector(responses, directions_deg):
    """
    Extract the mean vector from directional responses.
    
    This is the vector sum of all individual direction vectors.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'angle': Mean vector angle in degrees
        - 'magnitude': Mean vector magnitude (same as vector_magnitude)
        - 'cartesian_x': X component of mean vector
        - 'cartesian_y': Y component of mean vector
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    
    # Compute mean vector components
    total_response = np.sum(responses)
    if total_response == 0:
        return {
            'angle': np.nan,
            'magnitude': 0,
            'cartesian_x': 0,
            'cartesian_y': 0
        }
    
    mean_x = np.sum(responses * np.cos(directions_rad)) / total_response
    mean_y = np.sum(responses * np.sin(directions_rad)) / total_response
    
    # Vector magnitude and angle
    magnitude = np.sqrt(mean_x**2 + mean_y**2)
    angle_rad = np.arctan2(mean_y, mean_x)
    angle_deg = np.rad2deg(angle_rad)
    
    # Ensure positive angle
    if angle_deg < 0:
        angle_deg += 360
    
    return {
        'angle': angle_deg,
        'magnitude': magnitude,
        'cartesian_x': mean_x * total_response,  # Scale back up for visualization
        'cartesian_y': mean_y * total_response
    }


def compute_orientation_tuning(responses, directions_deg):
    """
    Compute orientation tuning by averaging opposite directions (e.g., 0° and 180° become one orientation), vectorized. 
    
    Parameters:
    -----------
    responses : array-like
        Response values. Can be 1D, 2D, or 3D. Last axis must be directions.
    directions_deg : array-like
        1D array of direction values in degrees.
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'orientations': Unique orientation angles (0-180°).
        - 'responses': Averaged responses for each orientation. Shape matches input `responses`
                       but last axis is n_orientations.
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    
    # Convert directions to orientations (0-180°)
    orientations = directions_deg % 180
    unique_orientations, inverse_indices = np.unique(orientations, return_inverse=True)
    
    # Prepare output array
    output_shape = responses.shape[:-1] + (len(unique_orientations),)
    orientation_responses = np.zeros(output_shape)
    
    # Sum responses for each unique orientation
    # 'add.at' is used for efficient summation based on indices
    np.add.at(orientation_responses, (..., inverse_indices), responses)
    
    # Count how many directions contributed to each orientation
    counts = np.bincount(inverse_indices)
    
    # Divide by counts to get the mean. Use np.where to avoid division by zero.
    orientation_responses /= np.where(counts == 0, 1, counts)
    
    return {
        'orientations': unique_orientations,
        'responses': orientation_responses
    }


def extract_orientation_vector(responses, directions_deg):
    """
    Extract the mean orientation vector from directional responses.
    
    Computes orientation selectivity by averaging opposite directions,
    then calculates the mean vector.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'angle': Mean orientation vector angle in degrees (0-180°)
        - 'magnitude': Mean orientation vector magnitude
        - 'cartesian_x': X component of mean orientation vector
        - 'cartesian_y': Y component of mean orientation vector
    """
    # Get orientation tuning
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']
    
    # Convert orientations to radians (double angles for proper circular stats)
    # We double the angle because orientation space is 0-180°, but we need
    # to map it to full circle for vector calculation
    orientations_rad_doubled = np.deg2rad(orientations * 2)
    
    # Compute mean vector components
    total_response = np.sum(orientation_responses)
    if total_response == 0:
        return {
            'angle': np.nan,
            'magnitude': 0,
            'cartesian_x': 0,
            'cartesian_y': 0
        }
    
    mean_x = np.sum(orientation_responses * np.cos(orientations_rad_doubled)) / total_response
    mean_y = np.sum(orientation_responses * np.sin(orientations_rad_doubled)) / total_response
    
    # Vector magnitude and angle
    magnitude = np.sqrt(mean_x**2 + mean_y**2)
    angle_rad_doubled = np.arctan2(mean_y, mean_x)
    
    # Convert back to orientation space (divide by 2 and ensure 0-180°)
    angle_deg = np.rad2deg(angle_rad_doubled) / 2
    if angle_deg < 0:
        angle_deg += 180
    
    return {
        'angle': angle_deg,
        'magnitude': magnitude,
        'cartesian_x': mean_x * total_response,  # Scale back up for visualization
        'cartesian_y': mean_y * total_response
    }


def compute_orientation_selectivity_index(responses, directions_deg):
    """
    Compute standard orientation selectivity index (OSI), vectorized.
    
    OSI = (R_preferred - R_orthogonal) / (R_preferred + R_orthogonal)
    
    Parameters:
    -----------
    responses : array-like
        Response values. Can be 1D, 2D, or 3D. Last axis must be directions.
    directions_deg : array-like
        1D array of direction values in degrees.
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'osi': Orientation selectivity index (0 ≤ OSI ≤ 1)
        - 'preferred_orientation': Preferred orientation in degrees (0-180°)
        - 'orthogonal_orientation': Orthogonal orientation in degrees (0-180°)
        - 'preferred_response': Response at preferred orientation
        - 'orthogonal_response': Response at orthogonal orientation
    """
    # Get orientation tuning
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']
    
    if orientation_responses.shape[-1] == 0:
        # Create correctly shaped empty outputs
        output_shape = responses.shape[:-1]
        nan_array = np.full(output_shape, np.nan)
        zero_array = np.zeros(output_shape)
        return {
            'osi': zero_array,
            'preferred_orientation': nan_array,
            'orthogonal_orientation': nan_array,
            'preferred_response': zero_array,
            'orthogonal_response': zero_array
        }
    
    # Find preferred orientation
    preferred_indices = np.argmax(orientation_responses, axis=-1)
    preferred_orientations = orientations[preferred_indices]
    preferred_responses = np.max(orientation_responses, axis=-1)
    
    # Find orthogonal orientation (90° away)
    orthogonal_orientations_target = (preferred_orientations + 90) % 180
    
    # Find closest actual orientation to orthogonal target
    orientation_diffs = np.abs(orientations - orthogonal_orientations_target[..., np.newaxis])
    orientation_diffs = np.minimum(orientation_diffs, 180 - orientation_diffs)
    orthogonal_indices = np.argmin(orientation_diffs, axis=-1)
    
    # Get orthogonal responses
    if responses.ndim > 1:
        indices = np.indices(orientation_responses.shape[:-1])
        orthogonal_responses = orientation_responses[(*indices, orthogonal_indices)]
    else:
        orthogonal_responses = orientation_responses[orthogonal_indices]
    
    # Compute OSI
    denominator = preferred_responses + orthogonal_responses
    osis = np.where(denominator == 0, 0, (preferred_responses - orthogonal_responses) / denominator)
    
    # Ensure OSI is non-negative
    osis = np.maximum(0, osis)
    
    return {
        'osi': osis,
        'preferred_orientation': preferred_orientations,
        'orthogonal_orientation': orientations[orthogonal_indices],
        'preferred_response': preferred_responses,
        'orthogonal_response': orthogonal_responses
    }



def compute_all_tuning_metrics(osds_obj, metric='peak', roi_indices=None, phase_aware=None):
    """
    Compute all directional tuning metrics for ROIs in a OSDS object.

    Parameters
    ----------
    osds_obj : OSDS
        OSDS object containing directional response data.
    metric : str or callable
        Metric to use for computing tuning functions:
        - 'peak': maximum absolute value (default)
        - 'max': maximum value
        - 'mean': mean value
        - 'auc': area under curve (absolute)
        - callable: custom function
    roi_indices : list or None
        ROI indices to analyze. If None, analyzes all ROIs.
    phase_aware : bool or None
        Controls phase-aware analysis:
        - None (default): Auto-detect from osds_obj.dir_phase_num
          (phase-aware if dir_phase_num > 1, single-phase otherwise)
        - True: Force phase-aware analysis using dir_phase_num phases
        - False: Force single-phase analysis (ignore dir_phase_num)

    Returns
    -------
    dict
        Dictionary containing arrays of metrics for each ROI:

        - 'vector_magnitude': Circular vector magnitude (r), 0-1
        - 'circular_variance': 1 - r, 0-1
        - 'dsi': Directional selectivity index, -1 to 1
        - 'osi': Orientation selectivity index, 0-1
        - 'preferred_direction': Direction with max response (degrees)
        - 'preferred_orientation': Orientation with max response (degrees)
        - 'mean_direction': Mean direction from circular stats (degrees)
        - 'roi_indices': ROI indices that were analyzed
        - 'n_phases': Number of phases (1 if single-phase)

        Array shapes:
        - Single-phase (phase_aware=False or dir_phase_num=1): (n_rois,)
        - Multi-phase (phase_aware=True and dir_phase_num>1): (n_phases, n_rois)
    """
    # Handle ROI indices
    if roi_indices is None:
        roi_indices = list(range(osds_obj.num_rois))

    directions_deg = np.array(osds_obj.directions_list)

    # Determine effective phase_aware setting
    # None = auto-detect from dir_phase_num
    # True = force phase-aware
    # False = force single-phase
    if phase_aware is None:
        # Auto-detect: use phases if dir_phase_num > 1
        effective_phase_aware = osds_obj.dir_phase_num > 1
    else:
        effective_phase_aware = phase_aware

    # Initialize phase_ranges_list (only used for multi-phase)
    phase_ranges_list = None

    # Compute tuning functions based on phase_aware setting
    if effective_phase_aware and osds_obj.dir_phase_num > 1:
        # Phase-aware analysis: split each direction into phases using window-based approach
        epoch_dur = osds_obj.get_epoch_dur()
        n_phases = osds_obj.dir_phase_num
        phase_size = epoch_dur // n_phases
        phase_ranges_list = []
        for i in range(n_phases):
            start = i * phase_size
            end = (i + 1) * phase_size if i < n_phases - 1 else epoch_dur
            phase_ranges_list.append((start, end))

        # Get tuning functions for each phase window
        all_tuning_functions = []
        for phase_range in phase_ranges_list:
            # Use window parameter to get specific phase, explicitly disable auto phase splitting
            tuning_functions = osds_obj.compute_tuning_function(
                metric=metric, window=phase_range, phase_num=None
            )
            # Transpose if needed to ensure (n_rois, n_directions) shape
            if tuning_functions.shape[0] != osds_obj.num_rois:
                tuning_functions = tuning_functions.T
            if roi_indices != list(range(osds_obj.num_rois)):
                tuning_functions = tuning_functions[roi_indices]
            all_tuning_functions.append(tuning_functions)

        # Stack to get (n_phases, n_rois, n_directions)
        all_tuning_functions = np.array(all_tuning_functions)
    else:
        # Single-phase analysis: use entire response period
        # Force single-phase by passing phase_num=1 (or letting it default when dir_phase_num=1)
        tuning_functions = osds_obj.compute_tuning_function(metric=metric, phase_num=1)

        # Handle shape: should be (n_rois, n_directions) for single phase
        if tuning_functions.shape[0] != osds_obj.num_rois:
            tuning_functions = tuning_functions.T
        if roi_indices != list(range(osds_obj.num_rois)):
            tuning_functions = tuning_functions[roi_indices]

        # Add phase dimension for consistent shape (n_phases, n_rois, n_directions)
        all_tuning_functions = tuning_functions[np.newaxis, :, :]
        n_phases = 1

    n_rois = len(roi_indices)
    
    # Vectorized computation of all metrics
    vector_magnitudes = np.zeros((n_phases, n_rois))
    circular_variances = np.zeros((n_phases, n_rois))
    dsis = np.zeros((n_phases, n_rois))
    osis = np.zeros((n_phases, n_rois))
    preferred_directions = np.zeros((n_phases, n_rois))
    preferred_orientations = np.zeros((n_phases, n_rois))
    mean_directions = np.zeros((n_phases, n_rois))
    
    # Vectorized computation of all metrics - no loops!
    directions_rad = np.deg2rad(directions_deg)
    
    # Vectorized vector magnitude and circular variance
    total_responses = np.sum(all_tuning_functions, axis=2)  # (n_phases, n_rois)
    mean_x = np.sum(all_tuning_functions * np.cos(directions_rad), axis=2) / np.where(total_responses == 0, 1, total_responses)
    mean_y = np.sum(all_tuning_functions * np.sin(directions_rad), axis=2) / np.where(total_responses == 0, 1, total_responses)
    vector_magnitudes = np.sqrt(mean_x**2 + mean_y**2)
    vector_magnitudes = np.where(total_responses == 0, 0, vector_magnitudes)
    circular_variances = 1 - vector_magnitudes
    
    # Vectorized preferred direction (argmax across directions)
    preferred_directions = directions_deg[np.argmax(all_tuning_functions, axis=2)]
    
    # Vectorized mean direction
    mean_direction_rad = np.arctan2(mean_y, mean_x)
    mean_directions = np.rad2deg(mean_direction_rad)
    mean_directions = np.where(mean_directions < 0, mean_directions + 360, mean_directions)
    mean_directions = np.where(total_responses == 0, np.nan, mean_directions)
    
    # Vectorized DSI computation
    dsi_results = compute_direction_selectivity_index(all_tuning_functions, directions_deg)
    dsis = dsi_results['dsi']

    # Vectorized OSI computation
    osi_results = compute_orientation_selectivity_index(all_tuning_functions, directions_deg)
    osis = osi_results['osi']
    preferred_orientations = osi_results['preferred_orientation']
    
    # Squeeze arrays if single phase for backward compatibility
    if n_phases == 1:
        vector_magnitudes = np.squeeze(vector_magnitudes, axis=0)
        circular_variances = np.squeeze(circular_variances, axis=0)
        dsis = np.squeeze(dsis, axis=0)
        osis = np.squeeze(osis, axis=0)
        preferred_directions = np.squeeze(preferred_directions, axis=0)
        preferred_orientations = np.squeeze(preferred_orientations, axis=0)
        mean_directions = np.squeeze(mean_directions, axis=0)
    
    # Build return dictionary
    result = {
        'vector_magnitude': vector_magnitudes,
        'circular_variance': circular_variances,
        'dsi': dsis,
        'osi': osis,
        'preferred_direction': preferred_directions,
        'preferred_orientation': preferred_orientations,
        'mean_direction': mean_directions,
        'roi_indices': np.array(roi_indices),
        'n_phases': n_phases
    }

    # Add phase_ranges to result if multi-phase analysis was performed
    if n_phases > 1 and phase_ranges_list is not None:
        result['phase_ranges'] = phase_ranges_list

    return result