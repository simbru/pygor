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


def compute_directional_selectivity_index(responses, directions_deg):
    """
    Compute directional selectivity index (DSI).
    
    DSI = (preferred - opposite) / (preferred + opposite)
    DSI = 1 means strongly directional, DSI = 0 means equal response to opposite directions.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    float : DSI value (-1 ≤ DSI ≤ 1)
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    
    # Find preferred direction
    preferred_idx = np.argmax(responses)
    preferred_response = responses[preferred_idx]
    preferred_dir = directions_deg[preferred_idx]
    
    # Find opposite direction (180° away)
    opposite_dir = (preferred_dir + 180) % 360
    
    # Find closest direction to opposite
    dir_diffs = np.abs(directions_deg - opposite_dir)
    # Handle wrap-around (e.g., 350° vs 10°)
    dir_diffs = np.minimum(dir_diffs, 360 - dir_diffs)
    opposite_idx = np.argmin(dir_diffs)
    opposite_response = responses[opposite_idx]
    
    # Compute DSI
    if preferred_response + opposite_response == 0:
        return 0
    
    dsi = (preferred_response - opposite_response) / (preferred_response + opposite_response)
    return dsi


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
    Compute orientation tuning by averaging opposite directions.
    
    Converts direction selectivity to orientation selectivity by averaging
    responses to opposite directions (e.g., 0° and 180° become one orientation).
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'orientations': Orientation angles (0-180°)
        - 'responses': Averaged responses for each orientation
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)
    
    # Convert directions to orientations (0-180°)
    orientations = directions_deg % 180
    
    # Find unique orientations and average opposite directions
    unique_orientations = np.unique(orientations)
    orientation_responses = []
    
    for orient in unique_orientations:
        # Find all directions that map to this orientation
        matching_indices = np.where(orientations == orient)[0]
        # Average responses for this orientation
        avg_response = np.mean(responses[matching_indices])
        orientation_responses.append(avg_response)
    
    return {
        'orientations': unique_orientations,
        'responses': np.array(orientation_responses)
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
    Compute standard orientation selectivity index (OSI).
    
    OSI = (R_preferred - R_orthogonal) / (R_preferred + R_orthogonal)
    
    Where R_orthogonal is the response 90° away from the preferred orientation.
    OSI = 1 means perfectly orientation selective, OSI = 0 means no orientation preference.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
        
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
    
    if len(orientation_responses) == 0:
        return {
            'osi': 0,
            'preferred_orientation': np.nan,
            'orthogonal_orientation': np.nan,
            'preferred_response': 0,
            'orthogonal_response': 0
        }
    
    # Find preferred orientation
    preferred_idx = np.argmax(orientation_responses)
    preferred_orientation = orientations[preferred_idx]
    preferred_response = orientation_responses[preferred_idx]
    
    # Find orthogonal orientation (90° away)
    orthogonal_orientation = (preferred_orientation + 90) % 180
    
    # Find closest orientation to orthogonal
    orientation_diffs = np.abs(orientations - orthogonal_orientation)
    # Handle wrap-around for orientation space (0-180°)
    orientation_diffs = np.minimum(orientation_diffs, 180 - orientation_diffs)
    orthogonal_idx = np.argmin(orientation_diffs)
    orthogonal_response = orientation_responses[orthogonal_idx]
    
    # Compute OSI
    if preferred_response + orthogonal_response == 0:
        osi = 0
    else:
        osi = (preferred_response - orthogonal_response) / (preferred_response + orthogonal_response)
    
    # Ensure OSI is between 0 and 1
    osi = max(0, osi)
    
    return {
        'osi': osi,
        'preferred_orientation': preferred_orientation,
        'orthogonal_orientation': orientations[orthogonal_idx],  # Use actual orientation, not target
        'preferred_response': preferred_response,
        'orthogonal_response': orthogonal_response
    }



def compute_all_tuning_metrics(osds_obj, metric='peak', roi_indices=None, phase_ranges=None):
    """
    Compute all directional tuning metrics for ROIs in a OSDS object.
    
    Parameters:
    -----------
    osds_obj : OSDS
        OSDS object containing directional response data
    metric : str or callable
        Metric to use for computing tuning functions:
        - 'peak': maximum absolute value
        - 'max': maximum value
        - 'mean': mean value
        - 'auc': area under curve (absolute)
        - callable: custom function
    roi_indices : list or None
        ROI indices to analyze. If None, analyzes all ROIs.
    phase_ranges : list of tuples or None
        List of (start_idx, end_idx) index ranges for each phase to analyze separately.
        If None, analyzes entire response period as single phase.
        Example: [(0, 60), (60, 120)] for two phases of 60 samples each
        
    Returns:
    --------
    dict : Dictionary containing arrays of metrics for each ROI:
        When phase_ranges=None (backward compatibility):
        - 'vector_magnitude': Vector magnitude (r) for each ROI (n_rois,)
        - 'circular_variance': Circular variance (CV) for each ROI (n_rois,)
        - 'dsi': Directional selectivity index for each ROI (n_rois,)
        - 'osi': Orientation selectivity index for each ROI (n_rois,)
        - 'preferred_direction': Preferred direction (degrees) for each ROI (n_rois,)
        - 'preferred_orientation': Preferred orientation (degrees) for each ROI (n_rois,)
        - 'mean_direction': Mean direction from circular stats (degrees) for each ROI (n_rois,)
        - 'roi_indices': ROI indices that were analyzed
        
        When phase_ranges is provided:
        - 'vector_magnitude': Vector magnitude (r) for each phase and ROI (n_phases, n_rois)
        - 'circular_variance': Circular variance (CV) for each phase and ROI (n_phases, n_rois)
        - 'dsi': Directional selectivity index for each phase and ROI (n_phases, n_rois)
        - 'osi': Orientation selectivity index for each phase and ROI (n_phases, n_rois)
        - 'preferred_direction': Preferred direction (degrees) for each phase and ROI (n_phases, n_rois)
        - 'preferred_orientation': Preferred orientation (degrees) for each phase and ROI (n_phases, n_rois)
        - 'mean_direction': Mean direction from circular stats (degrees) for each phase and ROI (n_phases, n_rois)
        - 'roi_indices': ROI indices that were analyzed
        - 'phase_ranges': Phase ranges that were used
    """
    # Handle ROI indices
    if roi_indices is None:
        roi_indices = list(range(osds_obj.num_rois))
    
    directions_deg = np.array(osds_obj.directions_list)
    
    # Handle phase ranges
    if phase_ranges is None:
        # Use built-in phase support from compute_tuning_function
        # This will automatically use self.dir_phase_num if > 1
        tuning_functions = osds_obj.compute_tuning_function(metric=metric)
        print(f"tuning_functions shape from compute_tuning_function: {tuning_functions.shape}")
        print(f"osds_obj.num_rois: {osds_obj.num_rois}")
        print(f"len(roi_indices): {len(roi_indices)}")

        # Check if phase data was returned
        if len(tuning_functions.shape) == 3:
            # Phase data could be: (n_rois, n_directions, n_phases) or (n_directions, n_rois, n_phases)
            print(f"3D data detected, shape: {tuning_functions.shape}")
            # Determine correct orientation based on num_rois
            if tuning_functions.shape[0] == osds_obj.num_rois:
                # Shape is (n_rois, n_directions, n_phases)
                if roi_indices != list(range(osds_obj.num_rois)):
                    tuning_functions = tuning_functions[roi_indices]
                # Transpose to (n_phases, n_rois, n_directions)
                all_tuning_functions = tuning_functions.transpose(2, 0, 1)
            elif tuning_functions.shape[1] == osds_obj.num_rois:
                # Shape is (n_directions, n_rois, n_phases)
                # Transpose to (n_rois, n_directions, n_phases) first
                tuning_functions = tuning_functions.transpose(1, 0, 2)
                if roi_indices != list(range(osds_obj.num_rois)):
                    tuning_functions = tuning_functions[roi_indices]
                # Then transpose to (n_phases, n_rois, n_directions)
                all_tuning_functions = tuning_functions.transpose(2, 0, 1)
            else:
                raise ValueError(f"Cannot determine tuning_functions orientation: shape {tuning_functions.shape}, num_rois {osds_obj.num_rois}")
            n_phases = all_tuning_functions.shape[0]
            print(f"Final all_tuning_functions shape: {all_tuning_functions.shape}")
        else:
            # Regular 2D data: could be (n_rois, n_directions) or (n_directions, n_rois)
            print(f"2D data detected, shape: {tuning_functions.shape}")
            if tuning_functions.shape[0] != osds_obj.num_rois:
                # Need to transpose
                tuning_functions = tuning_functions.T
                print(f"Transposed to: {tuning_functions.shape}")
            if roi_indices != list(range(osds_obj.num_rois)):
                tuning_functions = tuning_functions[roi_indices]
            # Add phase dimension
            all_tuning_functions = tuning_functions[np.newaxis, :, :]
            n_phases = 1
            print(f"Final all_tuning_functions shape: {all_tuning_functions.shape}")
    elif phase_ranges == "auto":
        # Automatic phase splitting using get_epoch_dur() and dir_phase_num
        epoch_dur = osds_obj.get_epoch_dur()
        n_phases = osds_obj.dir_phase_num
        phase_size = epoch_dur // n_phases
        phase_ranges_list = []
        for i in range(n_phases):
            start = i * phase_size
            end = (i + 1) * phase_size if i < n_phases - 1 else epoch_dur
            phase_ranges_list.append((start, end))

        # Get all tuning functions for each phase window
        all_tuning_functions = []
        for i, phase_range in enumerate(phase_ranges_list):
            # Specific phase window - vectorized computation
            # Explicitly set phase_num=None to prevent automatic phase splitting
            tuning_functions = osds_obj.compute_tuning_function(metric=metric, window=phase_range, phase_num=None)
            print(f"Phase {i}: tuning_functions shape before transpose: {tuning_functions.shape}")
            print(f"  osds_obj.num_rois: {osds_obj.num_rois}")
            print(f"  len(roi_indices): {len(roi_indices)}")
            # Transpose if needed to ensure (n_rois, n_directions) shape BEFORE indexing
            if tuning_functions.shape[0] != osds_obj.num_rois:
                tuning_functions = tuning_functions.T
                print(f"  Transposed to: {tuning_functions.shape}")
            if roi_indices != list(range(osds_obj.num_rois)):
                tuning_functions = tuning_functions[roi_indices]
                print(f"  After indexing: {tuning_functions.shape}")
            all_tuning_functions.append(tuning_functions)

        # Stack to get (n_phases, n_rois, n_directions)
        all_tuning_functions = np.array(all_tuning_functions)
        print(f"Final all_tuning_functions shape: {all_tuning_functions.shape}")
    else:
        # Manual phase ranges provided
        n_phases = len(phase_ranges)
        all_tuning_functions = []
        for phase_range in phase_ranges:
            # Specific phase window - vectorized computation
            # Explicitly set phase_num=None to prevent automatic phase splitting
            tuning_functions = osds_obj.compute_tuning_function(metric=metric, window=phase_range, phase_num=None)
            # Transpose if needed to ensure (n_rois, n_directions) shape BEFORE indexing
            if tuning_functions.shape[0] != osds_obj.num_rois:
                tuning_functions = tuning_functions.T
            if roi_indices != list(range(osds_obj.num_rois)):
                tuning_functions = tuning_functions[roi_indices]
            all_tuning_functions.append(tuning_functions)

        # Stack to get (n_phases, n_rois, n_directions)
        all_tuning_functions = np.array(all_tuning_functions)

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
    preferred_indices = np.argmax(all_tuning_functions, axis=2)  # (n_phases, n_rois)
    preferred_responses = np.max(all_tuning_functions, axis=2)  # (n_phases, n_rois)
    
    # Find opposite directions (180° away)
    preferred_dirs = directions_deg[preferred_indices]
    opposite_dirs = (preferred_dirs + 180) % 360
    
    # Find closest direction indices to opposite directions
    dir_diffs = np.abs(directions_deg[np.newaxis, np.newaxis, :] - opposite_dirs[:, :, np.newaxis])
    dir_diffs = np.minimum(dir_diffs, 360 - dir_diffs)  # Handle wrap-around
    opposite_indices = np.argmin(dir_diffs, axis=2)
    
    # Get opposite responses
    opposite_responses = all_tuning_functions[np.arange(n_phases)[:, np.newaxis], 
                                            np.arange(n_rois)[np.newaxis, :], 
                                            opposite_indices]
    
    # Compute DSI
    dsi_denominator = preferred_responses + opposite_responses
    dsis = np.where(dsi_denominator == 0, 0, 
                   (preferred_responses - opposite_responses) / dsi_denominator)
    
    # Vectorized OSI computation (simplified - compute for all at once)
    # Convert to orientation space and compute OSI
    orientations = directions_deg % 180
    unique_orientations = np.unique(orientations)
    
    # For each phase and ROI, compute OSI
    osis = np.zeros((n_phases, n_rois))
    preferred_orientations = np.zeros((n_phases, n_rois))
    
    for phase_idx in range(n_phases):
        for roi_idx in range(n_rois):
            # Get orientation responses by averaging opposite directions
            orientation_responses = []
            for orient in unique_orientations:
                matching_indices = np.where(orientations == orient)[0]
                avg_response = np.mean(all_tuning_functions[phase_idx, roi_idx, matching_indices])
                orientation_responses.append(avg_response)
            
            orientation_responses = np.array(orientation_responses)
            
            if len(orientation_responses) > 0:
                preferred_idx = np.argmax(orientation_responses)
                preferred_orientation = unique_orientations[preferred_idx]
                preferred_response = orientation_responses[preferred_idx]
                
                # Find orthogonal orientation (90° away)
                orthogonal_orientation = (preferred_orientation + 90) % 180
                orientation_diffs = np.abs(unique_orientations - orthogonal_orientation)
                orientation_diffs = np.minimum(orientation_diffs, 180 - orientation_diffs)
                orthogonal_idx = np.argmin(orientation_diffs)
                orthogonal_response = orientation_responses[orthogonal_idx]
                
                # Compute OSI
                if preferred_response + orthogonal_response > 0:
                    osis[phase_idx, roi_idx] = max(0, (preferred_response - orthogonal_response) / (preferred_response + orthogonal_response))
                else:
                    osis[phase_idx, roi_idx] = 0
                
                preferred_orientations[phase_idx, roi_idx] = preferred_orientation
            else:
                osis[phase_idx, roi_idx] = 0
                preferred_orientations[phase_idx, roi_idx] = np.nan
    
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
        'roi_indices': np.array(roi_indices)
    }

    # Add phase_ranges to result if multi-phase analysis was performed
    if phase_ranges == "auto":
        result['phase_ranges'] = phase_ranges_list
    elif phase_ranges is not None and n_phases > 1:
        result['phase_ranges'] = phase_ranges

    return result


# def plot_tuning_metrics_histograms(metrics_dict, figsize=(15, 10), bins=20):
#     """
#     Plot histograms of all tuning metrics.
    
#     Parameters:
#     -----------
#     metrics_dict : dict
#         Dictionary returned by compute_all_tuning_metrics
#     figsize : tuple
#         Figure size
#     bins : int
#         Number of histogram bins
        
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     """
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(2, 4, figsize=figsize)
#     axes = axes.flatten()
    
#     metrics_to_plot = [
#         ('vector_magnitude', 'Vector Magnitude (r)', (0, 1)),
#         ('circular_variance', 'Circular Variance (CV)', (0, 1)),
#         ('dsi', 'Directional Selectivity Index (DSI)', (-1, 1)),
#         ('osi', 'Orientation Selectivity Index (OSI)', (0, 1)),
#         ('preferred_direction', 'Preferred Direction (°)', (0, 360)),
#         ('preferred_orientation', 'Preferred Orientation (°)', (0, 180)),
#         ('mean_direction', 'Mean Direction (°)', (0, 360))
#     ]
    
#     for i, (metric_key, title, xlim) in enumerate(metrics_to_plot):
#         if metric_key in metrics_dict:
#             data = metrics_dict[metric_key]
#             # Remove NaN values
#             data_clean = data[~np.isnan(data)]
            
#             axes[i].hist(data_clean, bins=bins, alpha=0.7, edgecolor='black')
#             axes[i].set_title(f'{title}\n(n={len(data_clean)} ROIs)')
#             axes[i].set_xlabel(title.split('(')[0].strip())
#             axes[i].set_ylabel('Count')
#             axes[i].set_xlim(xlim)
#             axes[i].grid(True, alpha=0.3)
            
#             # Add mean line
#             if len(data_clean) > 0:
#                 mean_val = np.mean(data_clean)
#                 axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
#                                label=f'Mean: {mean_val:.3f}')
#                 axes[i].legend()
    
#     # Hide unused subplots (we have 7 metrics in 2x4 grid = 8 subplots)
#     for i in range(len(metrics_to_plot), len(axes)):
#         axes[i].set_visible(False)
    
#     plt.tight_layout()
#     return fig