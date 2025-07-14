"""
Tools for analyzing spatial alignment and overlap between color channels in STRFs
"""

import numpy as np
import numpy.ma as ma
import warnings
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.metrics import jaccard_score


def compute_spatial_overlap_metrics(strf1_2d, strf2_2d, threshold=3.0, method='correlation'):
    """
    Compute spatial overlap metrics between two 2D STRF spatial maps.
    
    Parameters
    ----------
    strf1_2d, strf2_2d : ndarray
        2D spatial maps to compare (typically collapsed STRFs)
    threshold : float, optional
        Threshold for defining active regions. Default is 3.0.
    method : str, optional
        Overlap metric to compute. Options: 'correlation', 'jaccard', 'overlap_coefficient',
        'all'. Default is 'correlation'.
    
    Returns
    -------
    dict : Dictionary containing overlap metrics
        - 'spatial_correlation': Pearson correlation between spatial maps
        - 'jaccard_index': Jaccard index of thresholded regions (if method includes this)
        - 'overlap_coefficient': Overlap coefficient (if method includes this)
        - 'centroid_distance': Distance between centroids of active regions
        - 'threshold_mask1', 'threshold_mask2': Boolean masks of active regions
    """
    
    # Create threshold masks for active regions
    mask1 = np.abs(strf1_2d) > threshold
    mask2 = np.abs(strf2_2d) > threshold
    
    results = {
        'threshold_mask1': mask1,
        'threshold_mask2': mask2,
        'threshold': threshold
    }
    
    # Spatial correlation (always computed)
    valid_pixels = mask1 | mask2  # Pixels active in either map
    if np.sum(valid_pixels) > 1:
        corr_coef = np.corrcoef(strf1_2d[valid_pixels], strf2_2d[valid_pixels])[0, 1]
        results['spatial_correlation'] = corr_coef
    else:
        results['spatial_correlation'] = np.nan
    
    # Centroid distance
    centroid1 = compute_centroid(strf1_2d, mask1)
    centroid2 = compute_centroid(strf2_2d, mask2)
    
    if not (np.isnan(centroid1).any() or np.isnan(centroid2).any()):
        results['centroid_distance'] = np.sqrt(np.sum((centroid1 - centroid2)**2))
        results['centroid1'] = centroid1
        results['centroid2'] = centroid2
    else:
        results['centroid_distance'] = np.nan
        results['centroid1'] = centroid1
        results['centroid2'] = centroid2
    
    # Additional metrics if requested
    if method in ['jaccard', 'all']:
        if np.sum(mask1) > 0 or np.sum(mask2) > 0:
            # Jaccard index of binary masks
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            results['jaccard_index'] = intersection / union if union > 0 else 0
        else:
            results['jaccard_index'] = 0
    
    if method in ['overlap_coefficient', 'all']:
        if np.sum(mask1) > 0 and np.sum(mask2) > 0:
            # Overlap coefficient (Szymkiewicz-Simpson coefficient)
            intersection = np.sum(mask1 & mask2)
            min_size = min(np.sum(mask1), np.sum(mask2))
            results['overlap_coefficient'] = intersection / min_size
        else:
            results['overlap_coefficient'] = 0
    
    return results


def compute_centroid(spatial_map, mask=None, weighted=True):
    """
    Compute the centroid of a spatial map.
    
    Parameters
    ----------
    spatial_map : ndarray
        2D spatial map
    mask : ndarray, optional
        Boolean mask defining active region. If None, uses absolute values > 0.
    weighted : bool, optional
        If True, compute weighted centroid using map values. Default is True.
    
    Returns
    -------
    centroid : ndarray
        [y, x] coordinates of centroid
    """
    
    if mask is None:
        mask = np.abs(spatial_map) > 0
    
    if not np.any(mask):
        return np.array([np.nan, np.nan])
    
    # Get coordinates of active pixels
    y_coords, x_coords = np.where(mask)
    
    if weighted:
        # Weight by absolute values
        weights = np.abs(spatial_map[mask])
        if np.sum(weights) == 0:
            return np.array([np.nan, np.nan])
        centroid_y = np.average(y_coords, weights=weights)
        centroid_x = np.average(x_coords, weights=weights)
    else:
        # Unweighted centroid
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
    
    return np.array([centroid_y, centroid_x])


def compute_spatial_offset(strf1_2d, strf2_2d, threshold=3.0, method='centroid'):
    """
    Compute spatial offset between two STRFs.
    
    Parameters
    ----------
    strf1_2d, strf2_2d : ndarray
        2D spatial maps to compare
    threshold : float, optional
        Threshold for defining active regions. Default is 3.0.
    method : str, optional
        Method for computing offset. Options: 'centroid', 'peak', 'cross_correlation'.
        Default is 'centroid'.
    
    Returns
    -------
    dict : Dictionary containing offset measurements
        - 'offset_vector': [dy, dx] offset from strf1 to strf2
        - 'offset_magnitude': Magnitude of offset
        - 'offset_angle': Angle of offset in degrees
        - 'method': Method used for computation
    """
    
    if method == 'centroid':
        mask1 = np.abs(strf1_2d) > threshold
        mask2 = np.abs(strf2_2d) > threshold
        
        centroid1 = compute_centroid(strf1_2d, mask1, weighted=True)
        centroid2 = compute_centroid(strf2_2d, mask2, weighted=True)
        
        if np.isnan(centroid1).any() or np.isnan(centroid2).any():
            offset_vector = np.array([np.nan, np.nan])
        else:
            offset_vector = centroid2 - centroid1
    
    elif method == 'peak':
        # Find peaks (maximum absolute values)
        peak1_idx = np.unravel_index(np.argmax(np.abs(strf1_2d)), strf1_2d.shape)
        peak2_idx = np.unravel_index(np.argmax(np.abs(strf2_2d)), strf2_2d.shape)
        
        offset_vector = np.array(peak2_idx) - np.array(peak1_idx)
    
    elif method == 'cross_correlation':
        # Cross-correlation method to find best alignment
        from scipy.signal import correlate2d
        
        # Normalize maps
        norm1 = (strf1_2d - np.mean(strf1_2d)) / (np.std(strf1_2d) + 1e-10)
        norm2 = (strf2_2d - np.mean(strf2_2d)) / (np.std(strf2_2d) + 1e-10)
        
        # Cross-correlation
        correlation = correlate2d(norm1, norm2, mode='same')
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to offset from center
        center = (np.array(correlation.shape) - 1) // 2
        offset_vector = np.array(peak_idx) - center
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute magnitude and angle
    if not np.isnan(offset_vector).any():
        offset_magnitude = np.sqrt(np.sum(offset_vector**2))
        offset_angle = np.degrees(np.arctan2(offset_vector[0], offset_vector[1]))
    else:
        offset_magnitude = np.nan
        offset_angle = np.nan
    
    return {
        'offset_vector': offset_vector,  # [dy, dx]
        'offset_magnitude': offset_magnitude,
        'offset_angle': offset_angle,
        'method': method
    }


def compute_spatial_offset_between_channels(strf_obj, roi, color_channels=(0, 1), 
                                          threshold=3.0, method='centroid', 
                                          collapse_method='peak'):
    """
    Compute spatial offset between two color channels for a single ROI.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing multicolor data
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
    
    if not strf_obj.multicolour:
        raise ValueError("Spatial offset analysis requires multicolor STRF data")
    
    # Get spatial maps for both channels
    spatial_maps = []
    for color_idx in color_channels:
        strf_idx = roi * strf_obj.numcolour + color_idx
        if strf_idx >= len(strf_obj.strfs):
            raise IndexError(f"ROI {roi}, color {color_idx} exceeds available STRFs")
        
        strf_3d = strf_obj.strfs[strf_idx]
        
        # Collapse time dimension
        if collapse_method == 'peak':
            spatial_map = strf_3d[np.argmax(np.max(np.abs(strf_3d), axis=(1, 2)))]
        elif collapse_method == 'std':
            spatial_map = np.std(strf_3d, axis=0)
        elif collapse_method == 'sum':
            spatial_map = np.sum(np.abs(strf_3d), axis=0)
        else:
            raise ValueError(f"Unknown collapse method: {collapse_method}")
        
        spatial_maps.append(spatial_map)
    
    # Compute offset
    offset_info = compute_spatial_offset(
        spatial_maps[0], spatial_maps[1], threshold=threshold, method=method
    )
    
    # Add metadata
    offset_info['roi_index'] = roi
    offset_info['color_channels'] = color_channels
    offset_info['collapse_method'] = collapse_method
    offset_info['threshold'] = threshold
    
    return offset_info


def analyze_multicolor_spatial_alignment(strf_obj, roi, threshold=3.0, 
                                       reference_channel=0, collapse_method='peak'):
    """
    Analyze spatial alignment across all color channels for a single ROI.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing multicolor data
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
        - 'spatial_maps': List of 2D spatial maps for each color channel
        - 'pairwise_overlaps': Matrix of overlap metrics between all channel pairs
        - 'offsets_from_reference': Offsets of each channel relative to reference
        - 'centroid_positions': Centroid positions for each channel
        - 'alignment_summary': Summary statistics
    """
    
    if not strf_obj.multicolour:
        raise ValueError("Multicolor analysis requires multicolor STRF data")
    
    n_colors = strf_obj.numcolour
    
    # Extract spatial maps for each color channel
    spatial_maps = []
    for color_idx in range(n_colors):
        strf_idx = roi * n_colors + color_idx
        if strf_idx >= len(strf_obj.strfs):
            raise IndexError(f"ROI {roi}, color {color_idx} exceeds available STRFs")
        
        strf_3d = strf_obj.strfs[strf_idx]
        
        # Collapse time dimension
        if collapse_method == 'peak':
            # Take maximum absolute value across time
            spatial_map = strf_3d[np.argmax(np.max(np.abs(strf_3d), axis=(1, 2)))]
        elif collapse_method == 'std':
            spatial_map = np.std(strf_3d, axis=0)
        elif collapse_method == 'sum':
            spatial_map = np.sum(np.abs(strf_3d), axis=0)
        else:
            raise ValueError(f"Unknown collapse method: {collapse_method}")
        
        spatial_maps.append(spatial_map)
    
    # Compute pairwise overlaps
    n_pairs = n_colors * (n_colors - 1) // 2
    pairwise_overlaps = np.zeros((n_colors, n_colors), dtype=object)
    
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            overlap_metrics = compute_spatial_overlap_metrics(
                spatial_maps[i], spatial_maps[j], threshold=threshold, method='all'
            )
            pairwise_overlaps[i, j] = overlap_metrics
            pairwise_overlaps[j, i] = overlap_metrics  # Symmetric
    
    # Compute offsets from reference channel
    offsets_from_reference = []
    centroid_positions = []
    
    for color_idx in range(n_colors):
        # Compute centroid
        mask = np.abs(spatial_maps[color_idx]) > threshold
        centroid = compute_centroid(spatial_maps[color_idx], mask, weighted=True)
        centroid_positions.append(centroid)
        
        if color_idx == reference_channel:
            offsets_from_reference.append({
                'offset_vector': np.array([0.0, 0.0]),
                'offset_magnitude': 0.0,
                'offset_angle': 0.0
            })
        else:
            offset_info = compute_spatial_offset(
                spatial_maps[reference_channel], spatial_maps[color_idx], 
                threshold=threshold, method='centroid'
            )
            offsets_from_reference.append(offset_info)
    
    # Compute alignment summary statistics
    valid_centroids = [c for c in centroid_positions if not np.isnan(c).any()]
    if len(valid_centroids) > 1:
        # Pairwise distances between centroids
        centroid_distances = cdist(valid_centroids, valid_centroids)
        mean_centroid_distance = np.mean(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])
        max_centroid_distance = np.max(centroid_distances)
        
        # Mean spatial correlation across all pairs
        correlations = []
        for i in range(n_colors):
            for j in range(i + 1, n_colors):
                if pairwise_overlaps[i, j] is not None:
                    corr = pairwise_overlaps[i, j]['spatial_correlation']
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        mean_spatial_correlation = np.mean(correlations) if correlations else np.nan
    else:
        mean_centroid_distance = np.nan
        max_centroid_distance = np.nan
        mean_spatial_correlation = np.nan
    
    alignment_summary = {
        'mean_centroid_distance': mean_centroid_distance,
        'max_centroid_distance': max_centroid_distance,
        'mean_spatial_correlation': mean_spatial_correlation,
        'n_valid_channels': len(valid_centroids),
        'reference_channel': reference_channel
    }
    
    return {
        'spatial_maps': spatial_maps,
        'pairwise_overlaps': pairwise_overlaps,
        'offsets_from_reference': offsets_from_reference,
        'centroid_positions': centroid_positions,
        'alignment_summary': alignment_summary,
        'roi_index': roi,
        'threshold': threshold,
        'collapse_method': collapse_method
    }


def plot_spatial_alignment(alignment_results, figsize=(15, 10)):
    """
    Plot spatial alignment results for visualization.
    
    Parameters
    ----------
    alignment_results : dict
        Results from analyze_multicolor_spatial_alignment
    figsize : tuple, optional
        Figure size. Default is (15, 10).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    
    spatial_maps = alignment_results['spatial_maps']
    centroid_positions = alignment_results['centroid_positions']
    n_colors = len(spatial_maps)
    
    # Create subplots
    fig, axes = plt.subplots(2, n_colors, figsize=figsize)
    if n_colors == 1:
        axes = axes.reshape(2, 1)
    
    # Color maps for different channels
    cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges'][:n_colors]
    
    # Plot individual spatial maps
    for i, (spatial_map, centroid) in enumerate(zip(spatial_maps, centroid_positions)):
        # Individual map
        im = axes[0, i].imshow(spatial_map, cmap=cmaps[i], aspect='equal')
        axes[0, i].set_title(f'Color Channel {i}')
        
        # Mark centroid if valid
        if not np.isnan(centroid).any():
            axes[0, i].plot(centroid[1], centroid[0], 'k+', markersize=10, markeredgewidth=2)
        
        plt.colorbar(im, ax=axes[0, i])
    
    # Plot overlay with centroids
    for i, spatial_map in enumerate(spatial_maps):
        axes[1, i].contour(spatial_map, levels=[alignment_results['threshold']], 
                          colors=[plt.cm.get_cmap(cmaps[i])(0.8)], linewidths=2)
        axes[1, i].set_title(f'Threshold Contour {i}')
        axes[1, i].set_aspect('equal')
    
    # Plot all centroids on the last subplot
    if n_colors > 1:
        for i, centroid in enumerate(centroid_positions):
            if not np.isnan(centroid).any():
                axes[1, -1].plot(centroid[1], centroid[0], 'o', 
                                color=plt.cm.get_cmap(cmaps[i])(0.8), 
                                markersize=8, label=f'Channel {i}')
        
        axes[1, -1].set_title('All Centroids')
        axes[1, -1].legend()
        axes[1, -1].set_aspect('equal')
    
    plt.tight_layout()
    return fig


def compute_color_channel_overlap_wrapper(strf_obj, roi, color_channels=(0, 1), threshold=3.0, 
                                        collapse_method='peak'):
    """
    Compute spatial overlap metrics between two specific color channels.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing multicolor data
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
    
    if not strf_obj.multicolour:
        raise ValueError("Color channel overlap requires multicolor STRF data")
    
    # Get spatial maps for both channels
    spatial_maps = []
    for color_idx in color_channels:
        strf_idx = roi * strf_obj.numcolour + color_idx
        if strf_idx >= len(strf_obj.strfs):
            raise IndexError(f"ROI {roi}, color {color_idx} exceeds available STRFs")
        
        strf_3d = strf_obj.strfs[strf_idx]
        
        # Collapse time dimension
        if collapse_method == 'peak':
            spatial_map = strf_3d[np.argmax(np.max(np.abs(strf_3d), axis=(1, 2)))]
        elif collapse_method == 'std':
            spatial_map = np.std(strf_3d, axis=0)
        elif collapse_method == 'sum':
            spatial_map = np.sum(np.abs(strf_3d), axis=0)
        else:
            raise ValueError(f"Unknown collapse method: {collapse_method}")
        
        spatial_maps.append(spatial_map)
    
    # Compute overlap metrics
    overlap_metrics = compute_spatial_overlap_metrics(
        spatial_maps[0], spatial_maps[1], threshold=threshold, method='all'
    )
    
    # Add metadata
    overlap_metrics['roi_index'] = roi
    overlap_metrics['color_channels'] = color_channels
    overlap_metrics['collapse_method'] = collapse_method
    
    return overlap_metrics


def plot_spatial_alignment_wrapper(strf_obj, roi, threshold=3.0, reference_channel=0, 
                                 collapse_method='peak', figsize=(15, 10)):
    """
    Plot spatial alignment visualization for a ROI.
    
    Parameters
    ----------
    strf_obj : STRF object
        STRF object containing the data
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
    
    # Get alignment analysis
    alignment_results = analyze_multicolor_spatial_alignment(
        strf_obj, roi=roi, threshold=threshold, reference_channel=reference_channel,
        collapse_method=collapse_method
    )
    
    # Create plot
    return plot_spatial_alignment(alignment_results, figsize=figsize)