import numpy as np
from pycircstat2.descriptive import circ_mean, circ_r, circ_var, circ_std
from pycircstat2.hypothesis import rayleigh_test


def _apply_per_element(func, responses, directions_rad, default_value=np.nan):
    """
    Apply a scalar circular stats function over all ROIs/phases.

    Iterates over the leading dimensions of ``responses``, calling
    ``func(directions_rad, weights)`` for each 1-D slice along the last axis.
    Negative weights are clipped to zero before passing to the function.

    Parameters
    ----------
    func : callable
        Function with signature ``func(alpha, w) -> scalar``.
    responses : np.ndarray
        Response values. Last axis is directions. Can be 1D, 2D, or 3D.
    directions_rad : np.ndarray
        1D array of direction values in radians.
    default_value : scalar
        Value to use when all weights are zero.

    Returns
    -------
    float or np.ndarray
        Result with shape ``responses.shape[:-1]``.
    """
    weights = np.clip(responses, 0, None)

    if responses.ndim == 1:
        if np.sum(weights) == 0:
            return default_value
        return float(func(directions_rad, weights))

    original_shape = responses.shape[:-1]
    flat = weights.reshape(-1, responses.shape[-1])
    result = np.full(flat.shape[0], default_value, dtype=float)
    for i in range(flat.shape[0]):
        w = flat[i]
        if np.sum(w) == 0:
            continue
        result[i] = float(func(directions_rad, w))
    return result.reshape(original_shape)


def compute_direction_vector_magnitude(responses, directions_deg):
    """
    Compute direction vector magnitude (r) from circular statistics via pycircstat2.

    This measures how directionally tuned the responses are in 360-degree space.
    r = 1 means perfectly tuned to a single direction, r = 0 means no directional preference.

    Negative response values are clipped to zero before computing (treating
    negative z-scored responses as "no response").

    This is part of the **circular statistics framework** for direction selectivity:
    - Use with get_mean_direction() for consistent angle/magnitude pairing
    - Different from DSI which uses argmax-based pairwise comparison

    Parameters
    ----------
    responses : array-like
        Response values for each direction. Can be 1D (n_directions),
        2D (n_rois, n_directions), or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        1D array of direction values in degrees (0-360).

    Returns
    -------
    float or np.ndarray
        Direction vector magnitude (0 <= r <= 1). Shape matches input without direction axis.

    See Also
    --------
    compute_orientation_vector_magnitude : For orientation selectivity (0-180 space)
    compute_direction_selectivity_index : For argmax-based DSI (pairwise framework)
    compute_mean_direction : For the angle component of the circular stats framework
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    return _apply_per_element(
        lambda alpha, w: circ_r(alpha=alpha, w=w),
        responses, directions_rad, default_value=0.0,
    )


# Backward compatibility alias
compute_vector_magnitude = compute_direction_vector_magnitude


def compute_orientation_vector_magnitude(responses, directions_deg):
    """
    Compute orientation vector magnitude from circular statistics via pycircstat2.

    This measures how orientation-selective the responses are in 180-degree space.
    Opposite directions (e.g., 0 and 180) are treated as the same orientation.
    r = 1 means perfectly tuned to a single orientation, r = 0 means no orientation preference.

    Uses the doubled-angle method: orientations are mapped from 0-180 to 0-360 space
    for proper circular vector computation.

    Negative response values are clipped to zero before computing.

    This is part of the **circular statistics framework** for orientation selectivity:
    - Use with get_mean_orientation() for consistent angle/magnitude pairing
    - Different from OSI which uses argmax-based pairwise comparison

    Parameters
    ----------
    responses : array-like
        Response values for each direction. Can be 1D (n_directions),
        2D (n_rois, n_directions), or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        1D array of direction values in degrees (0-360).

    Returns
    -------
    float or np.ndarray
        Orientation vector magnitude (0 <= r <= 1). Shape matches input without direction axis.

    See Also
    --------
    compute_direction_vector_magnitude : For direction selectivity (0-360 space)
    compute_orientation_selectivity_index : For argmax-based OSI (pairwise framework)
    compute_mean_orientation : For the angle component of the circular stats framework
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)

    # Get orientation tuning (averages opposite directions)
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']

    # Double the angles for proper circular stats in orientation space
    orientations_rad_doubled = np.deg2rad(orientations * 2)

    return _apply_per_element(
        lambda alpha, w: circ_r(alpha=alpha, w=w),
        orientation_responses, orientations_rad_doubled, default_value=0.0,
    )


def compute_circular_variance(responses, directions_deg):
    """
    Compute circular variance via pycircstat2.

    CV = 0 means perfectly tuned, CV = 1 means no directional preference.

    Negative response values are clipped to zero before computing.

    Parameters
    ----------
    responses : array-like
        Response values for each direction.
    directions_deg : array-like
        Direction values in degrees.

    Returns
    -------
    float or np.ndarray
        Circular variance (0 <= CV <= 1).
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    return _apply_per_element(
        lambda alpha, w: circ_var(alpha=alpha, w=w),
        responses, directions_rad, default_value=1.0,
    )


def compute_circular_std(responses, directions_deg):
    """
    Compute circular standard deviation via pycircstat2.

    Negative response values are clipped to zero before computing.

    Parameters
    ----------
    responses : array-like
        Response values for each direction.
    directions_deg : array-like
        Direction values in degrees.

    Returns
    -------
    float or np.ndarray
        Circular standard deviation.
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)
    return _apply_per_element(
        lambda alpha, w: circ_std(alpha=alpha, w=w),
        responses, directions_rad, default_value=np.nan,
    )


def compute_rayleigh_test(responses, directions_deg):
    """
    Compute Rayleigh test for uniformity on directional responses via pycircstat2.

    Tests the null hypothesis that responses are uniformly distributed
    around the circle. Low p-values indicate significant directional tuning.

    Negative response values are clipped to zero before computing.

    Since pycircstat2's ``rayleigh_test`` requires integer frequency weights,
    we pre-compute the mean resultant length ``r`` via ``circ_r`` (which
    accepts float weights) and pass ``r`` and ``n`` directly to the test.

    Parameters
    ----------
    responses : array-like
        Response values for each direction. Can be 1D (n_directions),
        2D (n_rois, n_directions), or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        1D array of direction values in degrees (0-360).

    Returns
    -------
    dict
        Dictionary containing:
        - 'z': Rayleigh z-statistic. Shape matches input without direction axis.
        - 'pvalue': Rayleigh test p-value. Shape matches input without direction axis.
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(np.asarray(directions_deg, dtype=float))
    weights = np.clip(responses, 0, None)
    n = len(directions_rad)

    def _rayleigh_single(alpha, w):
        """Compute Rayleigh test for a single 1D weight vector."""
        r = float(circ_r(alpha=alpha, w=w))
        result = rayleigh_test(r=r, n=n)
        return float(result.z), float(result.pval)

    if responses.ndim == 1:
        if np.sum(weights) == 0:
            return {'z': 0.0, 'pvalue': 1.0}
        z, p = _rayleigh_single(directions_rad, weights)
        return {'z': z, 'pvalue': p}

    original_shape = responses.shape[:-1]
    flat = weights.reshape(-1, responses.shape[-1])
    z_values = np.zeros(flat.shape[0])
    p_values = np.ones(flat.shape[0])

    for i in range(flat.shape[0]):
        w = flat[i]
        if np.sum(w) == 0:
            continue
        z_values[i], p_values[i] = _rayleigh_single(directions_rad, w)

    return {
        'z': z_values.reshape(original_shape),
        'pvalue': p_values.reshape(original_shape),
    }


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


# Metrics that reduce a single trace (last axis) to a scalar amplitude. These
# are the only ones meaningful for a per-trial / direction-shuffle permutation
# test. Template-based metrics ('correlation', 'r2', 'distance') compare each
# direction against a grand-mean template across directions, so shuffling the
# direction labels would corrupt the template -> deliberately unsupported.
_PERMUTABLE_METRICS = {
    'max', 'absmax', 'peak', 'min', 'avg', 'mean', 'range',
    'auc', 'peak_positive', 'peak_negative', 'auc_pos',
}


def _reduce_trace_metric(arr, metric, axis=-1):
    """
    Reduce traces along `axis` to a scalar per element using an amplitude metric.

    Mirrors the amplitude metrics in
    ``pygor.timeseries.osds.tuning_computation.compute_tuning_function`` so the
    permutation test uses the same response definition as the standard pipeline.

    Parameters
    ----------
    arr : np.ndarray
        Traces with time along `axis`.
    metric : str or callable
        One of ``_PERMUTABLE_METRICS`` or a callable taking a 1D array.
    axis : int
        Axis to reduce (the time axis).

    Returns
    -------
    np.ndarray
        `arr` with `axis` removed.
    """
    if callable(metric):
        return np.apply_along_axis(metric, axis, arr)
    if metric == 'max':
        return np.max(arr, axis=axis)
    if metric in ('absmax', 'peak'):
        return np.max(np.abs(arr), axis=axis)
    if metric == 'min':
        return np.min(arr, axis=axis)
    if metric in ('avg', 'mean'):
        return np.mean(arr, axis=axis)
    if metric == 'range':
        return np.max(arr, axis=axis) - np.min(arr, axis=axis)
    if metric == 'auc':
        return np.trapezoid(np.abs(arr), axis=axis)
    if metric == 'peak_positive':
        return np.max(arr, axis=axis)
    if metric == 'peak_negative':
        return np.min(arr, axis=axis)
    if metric == 'auc_pos':
        return np.trapezoid(np.clip(arr, 0, None), axis=axis)
    raise ValueError(
        f"Metric '{metric}' is not supported for the DSI permutation test. "
        f"Supported: {sorted(_PERMUTABLE_METRICS)} or a callable. "
        f"Template-based metrics (correlation/r2/distance) are excluded because "
        f"shuffling direction labels invalidates the cross-direction template."
    )


def _vector_magnitude_vectorized(tuning, directions_rad):
    """
    Mean resultant length (gDSI) over the last axis, fully vectorized.

    Matches ``compute_direction_vector_magnitude`` (negative responses clipped
    to zero) but works on arbitrary leading dimensions without a Python loop,
    so it is cheap to evaluate once per permutation.

    Parameters
    ----------
    tuning : np.ndarray
        Responses with directions along the last axis.
    directions_rad : np.ndarray
        Direction of each column, in radians.

    Returns
    -------
    np.ndarray
        Resultant length (0-1), shape ``tuning.shape[:-1]``.
    """
    w = np.clip(tuning, 0, None)
    total = w.sum(axis=-1)
    cx = (w * np.cos(directions_rad)).sum(axis=-1)
    cy = (w * np.sin(directions_rad)).sum(axis=-1)
    resultant = np.hypot(cx, cy)
    nonzero = total > 0
    out = np.zeros_like(total, dtype=float)
    np.divide(resultant, total, out=out, where=nonzero)
    return out


def compute_dsi_permutation_test(
    per_trial_responses,
    directions_deg,
    n_permutations=1000,
    alpha=0.05,
    seed=None,
    decision='gdsi',
):
    """
    Permutation test for direction-selectivity significance.

    Tests the null hypothesis that a cell is **not** direction-selective, i.e.
    that its per-trial responses are exchangeable across directions. For each
    permutation the pooled (direction x trial) responses of every ROI are
    randomly reassigned to directions (keeping trial counts), a shuffled tuning
    curve is formed by averaging over trials, and null selectivity statistics
    are computed. One-sided p-values are the fraction of null statistics >= the
    observed statistic. This addresses the well-known issue that a selectivity
    index can be large purely by chance with noisy or few-trial responses.

    Two statistics are computed from the **same** shuffle:

    - ``gdsi`` (direction vector magnitude / 1 - circular variance): the
      recommended significance statistic. It pools information across all
      directions, so its null distribution is well behaved and the test has
      good power. This is the field-standard DS significance statistic.
    - ``dsi`` (argmax-based pairwise index, what ``get_dsi`` reports): kept as a
      familiar effect size. Note its null is biased upward (argmax over noisy
      direction means is large by construction), so a pure-DSI permutation test
      is conservative / low power -- exactly why a high DSI alone is not proof
      of direction selectivity.

    Parameters
    ----------
    per_trial_responses : np.ndarray
        Scalar response per (ROI, direction, trial), shape
        ``(n_rois, n_directions, n_trials)``.
    directions_deg : array-like
        Direction of each column in degrees, length ``n_directions``.
    n_permutations : int
        Number of shuffles for the null distribution (default 1000).
    alpha : float
        Significance threshold for the ``is_ds`` decision (default 0.05).
    seed : int or None
        Seed for the random generator (reproducibility).
    decision : {'gdsi', 'dsi'}
        Which statistic drives the binary ``is_ds`` / ``p_value`` outputs.
        Default ``'gdsi'`` (recommended). Both p-values are always returned.

    Returns
    -------
    dict
        - 'p_value', 'is_ds': p-value and decision for the chosen ``decision``
          statistic, shape ``(n_rois,)``.
        - 'dsi', 'gdsi': observed statistics per ROI, ``(n_rois,)``. Computed
          from per-trial responses (mean over trials), so for non-linear
          metrics (e.g. 'range') 'dsi' may differ slightly from ``get_dsi()``,
          which reduces the trial-average trace.
        - 'p_value_dsi', 'p_value_gdsi': one-sided p-values for each statistic,
          ``(n_rois,)``, with the (1 + count) / (n + 1) correction.
        - 'null_dsi', 'null_gdsi': null distributions, ``(n_permutations, n_rois)``.
        - 'preferred_direction': observed preferred direction (deg), ``(n_rois,)``.
        - 'decision', 'n_permutations', 'alpha': echoes of the inputs.
    """
    if decision not in ('gdsi', 'dsi'):
        raise ValueError("decision must be 'gdsi' or 'dsi'")
    per_trial_responses = np.asarray(per_trial_responses, dtype=float)
    if per_trial_responses.ndim != 3:
        raise ValueError(
            "per_trial_responses must be 3D (n_rois, n_directions, n_trials), "
            f"got shape {per_trial_responses.shape}"
        )
    directions_deg = np.asarray(directions_deg)
    n_rois, n_dir, n_trials = per_trial_responses.shape
    if directions_deg.shape[0] != n_dir:
        raise ValueError(
            f"directions_deg length ({directions_deg.shape[0]}) must match "
            f"n_directions ({n_dir})"
        )
    directions_rad = np.deg2rad(directions_deg)

    # Observed statistics from the trial-averaged tuning curve.
    tuning_obs = per_trial_responses.mean(axis=2)  # (n_rois, n_dir)
    obs = compute_direction_selectivity_index(tuning_obs, directions_deg)
    dsi_obs = np.atleast_1d(obs['dsi'])
    gdsi_obs = np.atleast_1d(_vector_magnitude_vectorized(tuning_obs, directions_rad))

    # Pool responses per ROI; each permutation shuffles direction labels within
    # an ROI (independent shuffle per row), then averages back into directions.
    rng = np.random.default_rng(seed)
    flat = per_trial_responses.reshape(n_rois, n_dir * n_trials)
    null_dsi = np.empty((n_permutations, n_rois), dtype=float)
    null_gdsi = np.empty((n_permutations, n_rois), dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permuted(flat, axis=1).reshape(n_rois, n_dir, n_trials)
        tuning_null = shuffled.mean(axis=2)
        null_dsi[i] = compute_direction_selectivity_index(
            tuning_null, directions_deg
        )['dsi']
        null_gdsi[i] = _vector_magnitude_vectorized(tuning_null, directions_rad)

    # One-sided p with add-one correction (both statistics are >= 0).
    p_value_dsi = (1 + np.sum(null_dsi >= dsi_obs[np.newaxis, :], axis=0)) / (
        n_permutations + 1
    )
    p_value_gdsi = (1 + np.sum(null_gdsi >= gdsi_obs[np.newaxis, :], axis=0)) / (
        n_permutations + 1
    )
    p_value = p_value_gdsi if decision == 'gdsi' else p_value_dsi
    is_ds = p_value < alpha

    return {
        'p_value': p_value,
        'is_ds': is_ds,
        'dsi': dsi_obs,
        'gdsi': gdsi_obs,
        'p_value_dsi': p_value_dsi,
        'p_value_gdsi': p_value_gdsi,
        'null_dsi': null_dsi,
        'null_gdsi': null_gdsi,
        'preferred_direction': np.atleast_1d(obs['preferred_direction']),
        'decision': decision,
        'n_permutations': n_permutations,
        'alpha': alpha,
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
    Compute the mean direction using circular statistics via pycircstat2.

    This gives the direction of the mean vector, which may differ from
    the preferred direction if responses are broadly tuned.

    Negative response values are clipped to zero before computing.

    Parameters
    ----------
    responses : array-like
        Response values for each direction. Can be 1D (n_directions),
        2D (n_rois, n_directions), or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        Direction values in degrees.

    Returns
    -------
    float or np.ndarray
        Mean direction in degrees (0-360). Shape matches input without direction axis.
    """
    responses = np.array(responses)
    directions_rad = np.deg2rad(directions_deg)

    def _mean_dir(alpha, w):
        result_rad = circ_mean(alpha=alpha, w=w)
        result_deg = np.rad2deg(result_rad)
        if result_deg < 0:
            result_deg += 360
        return result_deg

    return _apply_per_element(_mean_dir, responses, directions_rad, default_value=np.nan)


def compute_mean_orientation(responses, directions_deg):
    """
    Compute the mean orientation using circular statistics via pycircstat2.

    This gives the orientation of the mean vector in 0-180 degree space,
    which may differ from the preferred orientation if responses are broadly tuned.

    Uses the doubled-angle method for proper circular statistics in orientation space.
    Opposite directions (e.g., 0 and 180) are treated as the same orientation.

    Negative response values are clipped to zero before computing.

    This is part of the **circular statistics framework** for orientation selectivity:
    - Use with get_orientation_vector_magnitude() for consistent angle/magnitude pairing

    Parameters
    ----------
    responses : array-like
        Response values for each direction. Can be 1D (n_directions),
        2D (n_rois, n_directions), or 3D (n_phases, n_rois, n_directions).
    directions_deg : array-like
        1D array of direction values in degrees (0-360).

    Returns
    -------
    float or np.ndarray
        Mean orientation in degrees (0-180). Shape matches input without direction axis.

    See Also
    --------
    compute_mean_direction : For mean direction in 360 space
    compute_orientation_vector_magnitude : For the magnitude component
    get_preferred_orientation : For argmax-based preferred orientation
    """
    responses = np.array(responses)
    directions_deg = np.array(directions_deg)

    # Get orientation tuning (averages opposite directions)
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']

    # Double the angles for circular stats in orientation space
    orientations_rad_doubled = np.deg2rad(orientations * 2)

    def _mean_ori(alpha, w):
        result_rad_doubled = circ_mean(alpha=alpha, w=w)
        result_deg = np.rad2deg(result_rad_doubled) / 2
        if result_deg < 0:
            result_deg += 180
        return result_deg

    return _apply_per_element(
        _mean_ori, orientation_responses, orientations_rad_doubled, default_value=np.nan,
    )


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



def compute_all_tuning_metrics(
    osds_obj,
    metric='peak',
    roi_indices=None,
    phase_aware=None,
    include_vonmises=False,
    r_squared_threshold=0.8,
):
    """
    Compute all directional tuning metrics for ROIs in a OSDS object.

    This function returns two complementary frameworks for analyzing
    direction and orientation selectivity:

    **Circular Statistics Framework:**
    Uses vector averaging to compute selectivity. Provides continuous angles
    and considers all directions in the calculation.

    - direction_vector_magnitude + mean_direction (for direction in 0-360 space)
    - orientation_vector_magnitude + mean_orientation (for orientation in 0-180 space)

    **Argmax/Pairwise Framework:**
    Uses maximum response and pairwise comparison. Traditional neuroscience
    metrics suitable for classification.

    - dsi + preferred_direction (argmax direction vs opposite)
    - osi + preferred_orientation (argmax orientation vs orthogonal)

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
    include_vonmises : bool
        If True, compute von Mises preferred direction/orientation and fit quality.
        Defaults to False to avoid added compute time.
    r_squared_threshold : float
        R2 threshold for von Mises fit_valid masks (default 0.8).

    Returns
    -------
    dict
        Dictionary containing arrays of metrics for each ROI:

        **Circular Statistics Framework (via pycircstat2):**

        - 'direction_vector_magnitude': Vector magnitude for direction (0-1)
        - 'orientation_vector_magnitude': Vector magnitude for orientation (0-1)
        - 'mean_direction': Mean direction from circular stats (degrees, 0-360)
        - 'mean_orientation': Mean orientation from circular stats (degrees, 0-180)
        - 'preferred_direction_vector_sum': Alias of mean_direction (degrees, 0-360)
        - 'preferred_orientation_vector_sum': Alias of mean_orientation (degrees, 0-180)
        - 'circular_variance': Circular variance (0-1)
        - 'circular_std': Circular standard deviation
        - 'rayleigh_z': Rayleigh test z-statistic
        - 'rayleigh_pvalue': Rayleigh test p-value (low = significant tuning)

        **Argmax/Pairwise Framework:**

        - 'dsi': Directional selectivity index (-1 to 1)
        - 'osi': Orientation selectivity index (0 to 1)
        - 'preferred_direction': Direction with max response (degrees, 0-360)
        - 'preferred_orientation': Orientation with max response (degrees, 0-180)

        **Metadata:**

        - 'roi_indices': ROI indices that were analyzed
        - 'n_phases': Number of phases (1 if single-phase)

        **Backward Compatibility (deprecated):**

        - 'vector_magnitude': Alias for 'direction_vector_magnitude'

        **Optional (von Mises fitting):**

        - 'vm_preferred_direction': Von Mises preferred direction (degrees, 0-360)
        - 'vm_preferred_orientation': Von Mises preferred orientation (degrees, 0-180)
        - 'vm_dir_r_squared': Direction fit R2
        - 'vm_ori_r_squared': Orientation fit R2
        - 'vm_dir_kappa': Direction concentration (kappa)
        - 'vm_ori_kappa': Orientation concentration (kappa)
        - 'vm_dir_fit_valid': Direction fit valid (bool, R2 >= threshold)
        - 'vm_ori_fit_valid': Orientation fit valid (bool, R2 >= threshold)

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

    # --- Circular statistics (via pycircstat2, negative values clipped to zero) ---
    vector_magnitudes = compute_direction_vector_magnitude(all_tuning_functions, directions_deg)
    mean_directions = compute_mean_direction(all_tuning_functions, directions_deg)
    circular_variances = compute_circular_variance(all_tuning_functions, directions_deg)
    orientation_vector_magnitudes = compute_orientation_vector_magnitude(all_tuning_functions, directions_deg)
    mean_orientations = compute_mean_orientation(all_tuning_functions, directions_deg)
    rayleigh_results = compute_rayleigh_test(all_tuning_functions, directions_deg)
    rayleigh_z = rayleigh_results['z']
    rayleigh_pvalue = rayleigh_results['pvalue']
    circ_stds = compute_circular_std(all_tuning_functions, directions_deg)

    # --- Argmax/pairwise framework (uses raw values including negatives) ---
    preferred_directions = directions_deg[np.argmax(all_tuning_functions, axis=2)]
    dsi_results = compute_direction_selectivity_index(all_tuning_functions, directions_deg)
    dsis = dsi_results['dsi']
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
        orientation_vector_magnitudes = np.squeeze(orientation_vector_magnitudes, axis=0)
        mean_orientations = np.squeeze(mean_orientations, axis=0)
        rayleigh_z = np.squeeze(rayleigh_z, axis=0)
        rayleigh_pvalue = np.squeeze(rayleigh_pvalue, axis=0)
        circ_stds = np.squeeze(circ_stds, axis=0)

    # Build return dictionary
    result = {
        # Circular statistics framework (via pycircstat2)
        'direction_vector_magnitude': vector_magnitudes,
        'orientation_vector_magnitude': orientation_vector_magnitudes,
        'mean_direction': mean_directions,
        'mean_orientation': mean_orientations,
        'preferred_direction_vector_sum': mean_directions,
        'preferred_orientation_vector_sum': mean_orientations,
        'circular_variance': circular_variances,
        'circular_std': circ_stds,
        'rayleigh_z': rayleigh_z,
        'rayleigh_pvalue': rayleigh_pvalue,

        # Argmax/pairwise framework
        'dsi': dsis,
        'osi': osis,
        'preferred_direction': preferred_directions,
        'preferred_orientation': preferred_orientations,

        # Metadata
        'roi_indices': np.array(roi_indices),
        'n_phases': n_phases,

        # Backward compatibility alias (deprecated)
        'vector_magnitude': vector_magnitudes,
    }

    if include_vonmises:
        from pygor.timeseries.osds import von_mises_fitting

        vm_dir = von_mises_fitting.compute_vonmises_preferred_direction(
            all_tuning_functions, directions_deg, r_squared_threshold
        )
        vm_ori = von_mises_fitting.compute_vonmises_preferred_orientation(
            all_tuning_functions, directions_deg, r_squared_threshold
        )

        if n_phases == 1:
            for key in ('preferred_direction', 'r_squared', 'kappa', 'fit_valid'):
                vm_dir[key] = np.squeeze(vm_dir[key], axis=0)
            for key in ('preferred_orientation', 'r_squared', 'kappa', 'fit_valid'):
                vm_ori[key] = np.squeeze(vm_ori[key], axis=0)

        result.update({
            'vm_preferred_direction': vm_dir['preferred_direction'],
            'vm_preferred_orientation': vm_ori['preferred_orientation'],
            'vm_dir_r_squared': vm_dir['r_squared'],
            'vm_ori_r_squared': vm_ori['r_squared'],
            'vm_dir_kappa': vm_dir['kappa'],
            'vm_ori_kappa': vm_ori['kappa'],
            'vm_dir_fit_valid': vm_dir['fit_valid'],
            'vm_ori_fit_valid': vm_ori['fit_valid'],
        })

    # Add phase_ranges to result if multi-phase analysis was performed
    if n_phases > 1 and phase_ranges_list is not None:
        result['phase_ranges'] = phase_ranges_list

    return result
