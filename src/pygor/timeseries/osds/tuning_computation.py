import numpy as np


def compute_tuning_function(osds_obj, roi_index=None, window=None, metric='max', phase_num=None):
    """
    Compute tuning function for each ROI across all directions.
    
    Parameters:
    -----------
    osds_obj : OSDS
        OSDS object containing the data
    window : int or tuple, optional
        Time window within each direction phase to analyze.
        If int, uses that many frames from start of each direction.
        If tuple (start, end), uses that slice within each direction.
        If None, uses entire duration of each direction.
        Ignored if phase_num is specified.
    metric : str or callable, optional
        Metric to compute for each direction. Built-in options:
        - 'max': maximum value
        - 'absmax': maximum absolute value  
        - 'min': minimum value
        - 'avg' or 'mean': average value
        - 'range': max - min
        - 'auc': area under curve (absolute)
        - 'peak': alias for 'absmax'
        - 'peak_positive': maximum positive value
        - 'peak_negative': minimum negative value
        - 'correlation' or 'corr': Pearson correlation with leave-one-out
          grand-mean template (measures response shape consistency)
        Or pass a callable function that takes 1D array and returns scalar.
    roi_index : int, optional
        If specified, returns tuning for only this ROI as 1D array.
        If None, returns tuning for all ROIs as 2D array.
    phase_num : int, optional
        If specified, splits each direction duration into n equal phases.
        Returns tuning values for each phase.
        
    Returns:
    --------
    np.ndarray
        If phase_num is None:
            If roi_index is None: tuning values with shape (n_rois, n_directions).
            If roi_index is specified: tuning values with shape (n_directions,).
        If phase_num is specified:
            If roi_index is None: tuning values with shape (n_rois, n_directions, n_phases).
            If roi_index is specified: tuning values with shape (n_directions, n_phases).
        Values are ordered according to osds_obj.directions_list.
    """
    if np.isnan(osds_obj.averages).all():
        raise ValueError("Averages not computed. Run compute_snippets_and_averages() first.")
    # Get directionally split averages: (n_directions, n_rois, timepoints_per_direction)
    dir_averages = osds_obj.split_averages_directionally()

    # Handle phase splitting
    if phase_num is not None:
        if not isinstance(phase_num, int) or phase_num < 1:
            raise ValueError("phase_num must be a positive integer")
        
        # Split each direction into phases
        n_directions, n_rois, timepoints_per_direction = dir_averages.shape
        phase_length = timepoints_per_direction // phase_num
        
        # Only use the frames that fit evenly into phases (trim remainder)
        usable_timepoints = phase_length * phase_num
        dir_averages = dir_averages[:, :, :usable_timepoints]
        
        # Reshape to separate phases: (n_directions, n_rois, n_phases, phase_length)
        dir_averages = dir_averages.reshape(n_directions, n_rois, phase_num, phase_length)
        
    # Apply window if specified (only if phase_num is not used)
    elif window is not None:
        if isinstance(window, int):
            # Use first 'window' frames
            dir_averages = dir_averages[:, :, :window]
        elif isinstance(window, (tuple, list)) and len(window) == 2:
            # Use slice [start:end]
            start, end = window
            dir_averages = dir_averages[:, :, start:end]
        else:
            raise ValueError("window must be int or tuple of (start, end)")
    
    # Compute metric for each direction and ROI
    if phase_num is not None:
        # For phase data: dir_averages shape is (n_directions, n_rois, n_phases, phase_length)
        # Compute metric over the last axis (phase_length) for each phase
        if metric == 'max':
            tuning_values = np.max(dir_averages, axis=3)
        elif metric in ['absmax', 'peak']:
            tuning_values = np.max(np.abs(dir_averages), axis=3)
        elif metric == 'min':
            tuning_values = np.min(dir_averages, axis=3)
        elif metric in ['avg', 'mean']:
            tuning_values = np.mean(dir_averages, axis=3)
        elif metric == 'range':
            tuning_values = np.max(dir_averages, axis=3) - np.min(dir_averages, axis=3)
        elif metric == 'auc':
            tuning_values = np.trapezoid(np.abs(dir_averages), axis=3)
        elif metric == 'peak_positive':
            tuning_values = np.max(dir_averages, axis=3)
        elif metric == 'peak_negative':
            tuning_values = np.min(dir_averages, axis=3)
        elif metric in ['correlation', 'corr']:
            # Correlation metric: for each direction, correlate its trace
            # with the grand-mean template across all directions.
            # Measures response shape consistency rather than amplitude.
            template = dir_averages.mean(axis=0, keepdims=True)  # (1, n_rois, n_phases, phase_length)
            x = dir_averages - dir_averages.mean(axis=3, keepdims=True)
            y = template - template.mean(axis=3, keepdims=True)
            num = (x * y).sum(axis=3)
            den = np.sqrt((x**2).sum(axis=3) * (y**2).sum(axis=3))
            tuning_values = np.where(den > 0, num / den, 0.0)
        elif metric == 'auc_pos':
            # Positive-only AUC: integrates only excitatory (positive)
            # deflections. Closest calcium analog to spike counts.
            tuning_values = np.trapezoid(np.clip(dir_averages, 0, None), axis=3)
        elif metric == 'r2':
            # Variance explained: how much of each direction's trace is
            # explained by the grand-mean template. High R² = similar to
            # mean, low R² = different from mean.
            template = dir_averages.mean(axis=0, keepdims=True)
            ss_res = ((dir_averages - template) ** 2).sum(axis=3)
            ss_tot = ((dir_averages - dir_averages.mean(axis=3, keepdims=True)) ** 2).sum(axis=3)
            tuning_values = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
        elif metric in ['distance', 'dist']:
            # Mean absolute error from grand-mean template. Less sensitive
            # than RMS because deviations are not squared, so large
            # fluctuations are not amplified disproportionately.
            template = dir_averages.mean(axis=0, keepdims=True)
            tuning_values = np.abs(dir_averages - template).mean(axis=3)
        elif callable(metric):
            # Apply custom function to each direction/ROI/phase combination
            tuning_values = np.array([[[metric(dir_averages[d, r, p, :]) 
                                      for p in range(dir_averages.shape[2])]
                                     for r in range(dir_averages.shape[1])]
                                    for d in range(dir_averages.shape[0])])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Transpose to get (n_rois, n_directions, n_phases)
        tuning_values = tuning_values.transpose(1, 0, 2)
    else:
        # For regular data: dir_averages shape is (n_directions, n_rois, timepoints)
        if metric == 'max':
            tuning_values = np.max(dir_averages, axis=2)
        elif metric in ['absmax', 'peak']:
            tuning_values = np.max(np.abs(dir_averages), axis=2)
        elif metric == 'min':
            tuning_values = np.min(dir_averages, axis=2)
        elif metric in ['avg', 'mean']:
            tuning_values = np.mean(dir_averages, axis=2)
        elif metric == 'range':
            tuning_values = np.max(dir_averages, axis=2) - np.min(dir_averages, axis=2)
        elif metric == 'auc':
            tuning_values = np.trapezoid(np.abs(dir_averages), axis=2)
        elif metric == 'peak_positive':
            tuning_values = np.max(dir_averages, axis=2)
        elif metric == 'peak_negative':
            tuning_values = np.min(dir_averages, axis=2)
        elif metric in ['correlation', 'corr']:
            # Correlation metric: for each direction, correlate its trace
            # with the grand-mean template across all directions.
            # Measures response shape consistency rather than amplitude.
            template = dir_averages.mean(axis=0, keepdims=True)  # (1, n_rois, timepoints)
            x = dir_averages - dir_averages.mean(axis=2, keepdims=True)
            y = template - template.mean(axis=2, keepdims=True)
            num = (x * y).sum(axis=2)
            den = np.sqrt((x**2).sum(axis=2) * (y**2).sum(axis=2))
            tuning_values = np.where(den > 0, num / den, 0.0)
        elif metric == 'auc_pos':
            # Positive-only AUC: integrates only excitatory (positive)
            # deflections. Closest calcium analog to spike counts.
            tuning_values = np.trapezoid(np.clip(dir_averages, 0, None), axis=2)
        elif metric == 'r2':
            # Variance explained: how much of each direction's trace is
            # explained by the grand-mean template. High R² = similar to
            # mean, low R² = different from mean.
            template = dir_averages.mean(axis=0, keepdims=True)
            ss_res = ((dir_averages - template) ** 2).sum(axis=2)
            ss_tot = ((dir_averages - dir_averages.mean(axis=2, keepdims=True)) ** 2).sum(axis=2)
            tuning_values = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
        elif metric in ['distance', 'dist']:
            # Mean absolute error from grand-mean template. Less sensitive
            # than RMS because deviations are not squared, so large
            # fluctuations are not amplified disproportionately.
            template = dir_averages.mean(axis=0, keepdims=True)
            tuning_values = np.abs(dir_averages - template).mean(axis=2)
        elif callable(metric):
            # Apply custom function to each direction/ROI combination
            tuning_values = np.array([[metric(dir_averages[d, r, :]) 
                                     for r in range(dir_averages.shape[1])]
                                    for d in range(dir_averages.shape[0])])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Transpose to get (n_rois, n_directions) as requested
        tuning_values = tuning_values.T
    
    # Return single ROI if specified
    if roi_index is not None:
        if phase_num is not None:
            # For phase data: tuning_values shape is (n_rois, n_directions, n_phases)
            return tuning_values[roi_index, :, :]  # Shape: (n_directions, n_phases)
        else:
            # For regular data: tuning_values shape is (n_rois, n_directions)
            return tuning_values[roi_index, :]  # Shape: (n_directions,)
    else:
        return tuning_values

