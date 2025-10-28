import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_directional_selectivity_index(values):
    """
    Calculate directional selectivity index (DSI).
    DSI = (preferred - null) / (preferred + null)

    Parameters:
    -----------
    values : array-like
            Response values for each direction

    Returns:
    --------
    float : DSI value between 0 (non-selective) and 1 (highly selective)
    """
    preferred = np.max(values)
    # Find null direction (opposite to preferred)
    preferred_idx = np.argmax(values)
    n_dirs = len(values)
    null_idx = (preferred_idx + n_dirs // 2) % n_dirs
    null = values[null_idx]

    if preferred + null == 0:
        return 0
    return (preferred - null) / (preferred + null)


def calculate_orientation_selectivity_index(values, directions_list):
    """
    Calculate orientation selectivity index by grouping opposite directions.

    Parameters:
    -----------
    values : array-like
            Response values for each direction
    directions_list : array-like
            Direction values in degrees

    Returns:
    --------
    float : OSI value
    """
    directions = np.array(directions_list)

    # Group opposite directions and sum their responses
    orientation_responses = []
    used_indices = set()

    for i, direction in enumerate(directions):
        if i in used_indices:
            continue

        # Find opposite direction (±180°)
        opposite_dir = (direction + 180) % 360
        opposite_idx = None

        for j, other_dir in enumerate(directions):
            if abs(other_dir - opposite_dir) < 10:  # Allow some tolerance
                opposite_idx = j
                break

        if opposite_idx is not None:
            combined_response = values[i] + values[opposite_idx]
            orientation_responses.append(combined_response)
            used_indices.update([i, opposite_idx])
        else:
            orientation_responses.append(values[i])
            used_indices.add(i)

    # Calculate OSI
    orientation_responses = np.array(orientation_responses)
    if len(orientation_responses) < 2:
        return 0

    preferred = np.max(orientation_responses)
    orthogonal_idx = (
        np.argmax(orientation_responses) + len(orientation_responses) // 2
    ) % len(orientation_responses)
    orthogonal = orientation_responses[orthogonal_idx]

    if preferred + orthogonal == 0:
        return 0
    return (preferred - orthogonal) / (preferred + orthogonal)


def plot_tuning_function_polar(tuning_functions, directions_list, rois=None, figsize=(6, 6), colors=None, metric='peak', ax=None, show_title=True, show_theta_labels=True, show_tuning=True, show_mean_vector=False, mean_vector_color='red', show_orientation_vector=False, orientation_vector_color='orange', legend=True, minimal=False):
    """
    Plot tuning functions as polar plots.
    
    Parameters:
    -----------
    tuning_functions : np.ndarray
        Array of shape (n_directions, n_rois) containing tuning functions
    directions_list : list of int/float
        List of direction values in degrees
    rois : list of int or None
        ROI indices to plot. If None, plots all ROIs
    figsize : tuple
        Figure size (width, height)
    colors : list or None
        Colors for each ROI. If None, uses default color cycle
    metric : str or callable
        Metric used to compute tuning function (for title display)
    ax : matplotlib.axes.Axes or None
        Existing polar axes to plot on. If None, creates new figure and axes
    show_title : bool
        Whether to show the title on the plot (default True)
    show_theta_labels : bool
        Whether to show the theta (direction) labels on the plot (default True)
    show_tuning : bool
        Whether to show the tuning curve itself (default True). When False, only shows vectors.
    show_mean_vector : bool
        Whether to show mean direction vectors as overlays (default False)
    mean_vector_color : str
        Color for mean direction vector arrows (default 'red')
    show_orientation_vector : bool
        Whether to show mean orientation vectors as overlays (default False)
    orientation_vector_color : str
        Color for mean orientation vector arrows (default 'orange')
    legend : bool
        Whether to show the legend (default True)
    minimal : bool
        Whether to use minimal plotting (no titles or legends) (default False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The polar plot axes object
    """
    if rois is None:
        rois = list(range(tuning_functions.shape[1]))
    elif isinstance(rois, (int, np.integer)):
        rois = [rois]
    
    # Sort by direction for proper polar plot connectivity
    sort_order = np.argsort(directions_list)
    degrees = np.deg2rad(directions_list)[sort_order]
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection='polar')
        # Set polar plot orientation to match other functions
        ax.set_theta_zero_location("E")  # 0° at right (90° north)
    else:
        fig = ax.figure
        # Check if provided axes is polar, if not replace with polar axes
        if ax.name != 'polar':
            # Get the position of the current axes
            pos = ax.get_position()
            # Remove the current axes
            ax.remove()
            # Create a new polar axes in the same position
            ax = fig.add_subplot(111, projection='polar')
            ax.set_position(pos)
            ax.set_theta_zero_location("E")
    
    # Set up colors
    if colors is None:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    
    # Plot tuning curves only if show_tuning is True
    if show_tuning:
        for i, roi_idx in enumerate(rois):
            tuning_function = tuning_functions[sort_order, roi_idx]
            
            # Close the loop
            tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
            degrees_closed = np.concatenate((degrees, [degrees[0]]))
            
            ax.plot(degrees_closed, tuning_function_closed, marker='o', 
                    color=colors[i], label=f'ROI {roi_idx}')
    
    # Show legend if requested and not minimal and there are multiple ROIs/tuning/vectors shown
    if legend and not minimal and ((len(rois) > 1 and show_tuning) or show_mean_vector or show_orientation_vector):
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Format metric name for title (only if not minimal)
    if show_title and not minimal:
        if callable(metric):
            metric_name = getattr(metric, '__name__', 'custom function')
        else:
            metric_name = metric.replace('_', ' ').title()
        
        ax.set_title(f'Directional tuning ({metric_name})', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
    ax.grid(True, alpha=0.3)
    
    # Hide theta labels if requested or in minimal mode
    if not show_theta_labels or minimal:
        ax.set_thetagrids([])
    
    # Add mean direction vectors if requested
    if show_mean_vector:
        from pygor.timeseries.osds import tuning_metrics
        
        for i, roi_idx in enumerate(rois):
            # Get tuning function for this ROI
            roi_tuning = tuning_functions[:, roi_idx][sort_order]
            directions_sorted = np.array(directions_list)[sort_order]
            
            # Compute mean direction vector
            mean_vector = tuning_metrics.extract_mean_vector(roi_tuning, directions_sorted)
            
            if not np.isnan(mean_vector['angle']):
                # Convert to radians
                mean_angle_rad = np.deg2rad(mean_vector['angle'])
                
                # Scale magnitude for visibility
                mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning)
                
                # Use the same color as the tuning function for this ROI
                arrow_color = colors[i] if isinstance(colors, (list, np.ndarray)) else mean_vector_color
                
                # Plot mean direction vector arrow with label for legend
                ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3),
                           zorder=10)
                
                # Add invisible line for legend entry
                ax.plot([], [], color=arrow_color, marker='>', markersize=8, 
                       linestyle='None', label=f'ROI {roi_idx} Direction Vector')
    
    # Add mean orientation vectors if requested
    if show_orientation_vector:
        from pygor.timeseries.osds import tuning_metrics
        
        for i, roi_idx in enumerate(rois):
            # Get tuning function for this ROI
            roi_tuning = tuning_functions[:, roi_idx][sort_order]
            directions_sorted = np.array(directions_list)[sort_order]
            
            # Compute mean orientation vector
            orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning, directions_sorted)
            
            if not np.isnan(orientation_vector['angle']):
                # Convert to radians
                orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                
                # Scale magnitude for visibility
                orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning)
                
                # Use orientation vector color
                orient_arrow_color = orientation_vector_color
                
                # Plot mean orientation vector arrow
                ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=orient_arrow_color, lw=3),
                           zorder=9)
                
                # Add invisible line for legend entry
                ax.plot([], [], color=orient_arrow_color, marker='>', markersize=8, 
                       linestyle='None', label=f'ROI {roi_idx} Orientation Vector')
    
    return fig, ax


def plot_tuning_function_multi_phase(tuning_functions, directions_list, phase_num, rois=None, 
                                    figsize=(6, 6), colors=None, ax=None, show_title=True, 
                                    show_theta_labels=True, show_tuning=True, show_mean_vector=False, 
                                    mean_vector_color='red', show_orientation_vector=False, 
                                    orientation_vector_color='orange', metric='peak', legend=True, minimal=False):
    """
    Plot multi-phase tuning functions as polar plots with phase-dependent vectors.
    
    Parameters:
    -----------
    tuning_functions : np.ndarray
        Array of shape (n_rois, n_directions, n_phases) containing tuning functions for each phase
    directions_list : list of int/float
        List of direction values in degrees
    phase_num : int
        Number of phases
    rois : list of int or None
        ROI indices to plot. If None, plots all ROIs
    figsize : tuple
        Figure size (width, height)
    colors : list or None
        Colors for each ROI. If None, uses default color cycle
    ax : matplotlib.axes.Axes or None
        Existing polar axes to plot on. If None, creates new figure and axes
    show_title : bool
        Whether to show the title on the plot (default True)
    show_theta_labels : bool
        Whether to show the theta (direction) labels on the plot (default True)
    show_tuning : bool
        Whether to show the tuning curve itself (default True). When False, only shows vectors.
    show_mean_vector : bool
        Whether to show mean direction vectors as overlays (default False)
    mean_vector_color : str
        Color for mean direction vector arrows (default 'red')
    show_orientation_vector : bool
        Whether to show mean orientation vectors as overlays (default False)
    orientation_vector_color : str
        Color for mean orientation vector arrows (default 'orange')
    metric : str
        Metric name for title display
    legend : bool
        Whether to show the legend (default True)
    minimal : bool
        Whether to use minimal plotting (no titles or legends) (default False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The polar plot axes object
    """
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection='polar')
        ax.set_theta_zero_location("E")  # 0° at right (90° north)
    else:
        fig = ax.figure
        # Check if provided axes is polar, if not replace with polar axes
        if ax.name != 'polar':
            # Get the position of the current axes
            pos = ax.get_position()
            # Remove the current axes
            ax.remove()
            # Create a new polar axes in the same position
            ax = fig.add_subplot(111, projection='polar')
            ax.set_position(pos)
            ax.set_theta_zero_location("E")
    
    # Set up ROI indices
    if rois is None:
        rois = list(range(tuning_functions.shape[0]))
    elif isinstance(rois, (int, np.integer)):
        rois = [rois]
    
    # Set up colors for ROIs
    if colors is None:
        base_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    else:
        base_colors = colors
    
    # Sort by direction for proper polar plot connectivity
    sort_order = np.argsort(directions_list)
    degrees = np.deg2rad(directions_list)[sort_order]
    
    # Plot each phase for each ROI (only if show_tuning is True)
    if show_tuning:
        for roi_i, roi_idx in enumerate(rois):
            base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else base_colors
            
            for phase_i in range(phase_num):
                # Get tuning function for this ROI and phase
                tuning_function = tuning_functions[roi_idx, sort_order, phase_i]
                
                # Close the loop
                tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
                degrees_closed = np.concatenate((degrees, [degrees[0]]))
                
                # Create phase-specific styling
                alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
                linestyle = ['-', '--', '-.', ':'][phase_i % 4]
                
                # Plot phase
                label = f'ROI {roi_idx} Phase {phase_i+1}' if len(rois) > 1 or phase_num > 1 else f'Phase {phase_i+1}'
                ax.plot(degrees_closed, tuning_function_closed, marker='o', 
                        color=base_color, alpha=alpha, linestyle=linestyle, label=label)
    
    # Add legend if requested and not minimal and there are multiple traces/tuning/vectors shown
    if legend and not minimal and (((len(rois) > 1 or phase_num > 1) and show_tuning) or show_mean_vector or show_orientation_vector):
        # Collect all handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Create manual legend if no automatic handles found
        if not handles and (show_mean_vector or show_orientation_vector):
            from matplotlib.lines import Line2D
            manual_handles = []
            manual_labels = []
            
            if show_mean_vector:
                for roi_i, roi_idx in enumerate(rois):
                    base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else mean_vector_color
                    for phase_i in range(phase_num):
                        # Create phase-specific color
                        if isinstance(base_color, str):
                            import matplotlib.colors as mcolors
                            base_rgb = mcolors.to_rgb(base_color)
                            phase_factor = 0.8 + 0.2 * (phase_i / max(1, phase_num - 1))
                            phase_color = tuple(min(1.0, c * phase_factor) for c in base_rgb)
                        else:
                            phase_color = base_color
                        
                        alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
                        handle = Line2D([0], [0], color=phase_color, marker='>', markersize=8, 
                                      alpha=alpha, linestyle='None')
                        manual_handles.append(handle)
                        manual_labels.append(f'ROI {roi_idx} Direction Vector P{phase_i+1}')
            
            if show_orientation_vector:
                for roi_i, roi_idx in enumerate(rois):
                    for phase_i in range(phase_num):
                        # Create phase-specific color for orientation
                        if isinstance(orientation_vector_color, str):
                            import matplotlib.colors as mcolors
                            base_rgb = mcolors.to_rgb(orientation_vector_color)
                            phase_factor = 1.0 - 0.3 * (phase_i / max(1, phase_num - 1))
                            phase_orient_color = tuple(max(0.0, c * phase_factor) for c in base_rgb)
                        else:
                            phase_orient_color = orientation_vector_color
                        
                        alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
                        handle = Line2D([0], [0], color=phase_orient_color, marker='>', markersize=8, 
                                      alpha=alpha, linestyle='None')
                        manual_handles.append(handle)
                        manual_labels.append(f'ROI {roi_idx} Orientation Vector P{phase_i+1}')
            
            ax.legend(manual_handles, manual_labels, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        else:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Format title (only if not minimal)
    if show_title and not minimal:
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f'Directional tuning ({metric_name}) - {phase_num} phases', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
    ax.grid(True, alpha=0.3)
    
    # Hide theta labels if requested or in minimal mode
    if not show_theta_labels or minimal:
        ax.set_thetagrids([])
    
    # Add mean direction vectors if requested (phase-dependent)
    if show_mean_vector:
        from pygor.timeseries.osds import tuning_metrics
        import matplotlib.colors as mcolors
        
        
        for roi_i, roi_idx in enumerate(rois):
            base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else mean_vector_color
            
            for phase_i in range(phase_num):
                # Get tuning function for this specific phase
                roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
                directions_sorted = np.array(directions_list)[sort_order]
                
                # Compute mean direction vector for this phase
                mean_vector = tuning_metrics.extract_mean_vector(roi_tuning_phase, directions_sorted)
                
                
                if not np.isnan(mean_vector['angle']):
                    # Convert to radians
                    mean_angle_rad = np.deg2rad(mean_vector['angle'])
                    
                    # Scale magnitude for visibility
                    mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning_phase)
                    
                    # Create phase-specific color variations
                    if isinstance(base_color, str):
                        # If base_color is a string, create variations
                        base_rgb = mcolors.to_rgb(base_color)
                        # Lighten for later phases
                        phase_factor = 0.8 + 0.2 * (phase_i / max(1, phase_num - 1))
                        phase_color = tuple(min(1.0, c * phase_factor) for c in base_rgb)
                    else:
                        phase_color = base_color
                    
                    alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
                    arrow_width = 3 if phase_i == 0 else 2  # Make first phase thicker
                    
                    # Plot mean direction vector arrow
                    ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
                               xytext=(0, 0),
                               arrowprops=dict(arrowstyle='->', color=phase_color, lw=arrow_width, alpha=alpha),
                               zorder=10 + phase_i)
                    
                    # Legend will be created manually in the legend section
    
    # Add mean orientation vectors if requested (phase-dependent)
    if show_orientation_vector:
        from pygor.timeseries.osds import tuning_metrics
        import matplotlib.colors as mcolors
        
        for roi_i, roi_idx in enumerate(rois):
            
            for phase_i in range(phase_num):
                # Get tuning function for this specific phase
                roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
                directions_sorted = np.array(directions_list)[sort_order]
                
                # Compute mean orientation vector for this phase
                orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning_phase, directions_sorted)
                
                
                if not np.isnan(orientation_vector['angle']):
                    # Convert to radians
                    orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                    
                    # Scale magnitude for visibility
                    orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning_phase)
                    
                    # Create phase-specific color variations for orientation vector
                    if isinstance(orientation_vector_color, str):
                        base_rgb = mcolors.to_rgb(orientation_vector_color)
                        # Darken for later phases (opposite of direction vectors)
                        phase_factor = 1.0 - 0.3 * (phase_i / max(1, phase_num - 1))
                        phase_orient_color = tuple(max(0.0, c * phase_factor) for c in base_rgb)
                    else:
                        phase_orient_color = orientation_vector_color
                    
                    alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
                    arrow_width = 3 if phase_i == 0 else 2  # Make first phase thicker
                    
                    # Plot mean orientation vector arrow
                    ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
                               xytext=(0, 0),
                               arrowprops=dict(arrowstyle='->', color=phase_orient_color, lw=arrow_width, alpha=alpha),
                               zorder=9 + phase_i)
                    
                    # Legend will be created manually in the legend section
    
    return fig, ax


def plot_tuning_function_polar_overlay(tuning_functions, directions_list, rois=None, 
                                      figsize=(6, 6), colors=None, phase_colors=None,
                                      ax=None, show_title=True, show_theta_labels=True, 
                                      show_tuning=True, show_mean_vector=False, 
                                      mean_vector_color='red', show_orientation_vector=False, 
                                      orientation_vector_color='orange', metric='peak', legend=True, minimal=False):
    """
    Plot multi-phase tuning functions overlaid on the same polar plot.
    
    Parameters:
    -----------
    tuning_functions : np.ndarray
        Array of shape (n_rois, n_directions, n_phases) containing tuning functions
    directions_list : list of int/float
        List of direction values in degrees
    rois : list of int or None
        ROI indices to plot. If None, plots all ROIs
    figsize : tuple
        Figure size (width, height)
    colors : list or None
        Colors for each ROI. If None, uses default color cycle
    phase_colors : list or None
        Colors for each phase. If None, uses default colors
    ax : matplotlib.axes.Axes or None
        Existing polar axes to plot on. If None, creates new figure and axes
    show_title : bool
        Whether to show the title on the plot
    show_theta_labels : bool
        Whether to show the theta (direction) labels on the plot
    show_tuning : bool
        Whether to show the tuning curve itself
    show_mean_vector : bool
        Whether to show mean direction vectors as overlays
    mean_vector_color : str
        Color for mean direction vector arrows
    show_orientation_vector : bool
        Whether to show mean orientation vectors as overlays
    orientation_vector_color : str
        Color for mean orientation vector arrows
    metric : str
        Metric name for title display
    legend : bool
        Whether to show the legend (default True)
    minimal : bool
        Whether to use minimal plotting (no titles or legends) (default False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The polar plot axes object
    """
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection='polar')
        ax.set_theta_zero_location("E")  # 0° at right (90° north)
    else:
        fig = ax.figure
        # Check if provided axes is polar, if not replace with polar axes
        if ax.name != 'polar':
            # Get the position of the current axes
            pos = ax.get_position()
            # Remove the current axes
            ax.remove()
            # Create a new polar axes in the same position
            ax = fig.add_subplot(111, projection='polar')
            ax.set_position(pos)
            ax.set_theta_zero_location("E")
    
    # Set up ROI indices
    if rois is None:
        rois = list(range(tuning_functions.shape[0]))
    elif isinstance(rois, (int, np.integer)):
        rois = [rois]
    
    # Set up colors
    if colors is None:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
    # Sort by direction for proper polar plot connectivity
    sort_order = np.argsort(directions_list)
    degrees = np.deg2rad(directions_list)[sort_order]
    
    n_phases = tuning_functions.shape[2]
    
    # Plot each phase for each ROI (only if show_tuning is True)
    if show_tuning:
        for roi_i, roi_idx in enumerate(rois):
            roi_color = colors[roi_i] if isinstance(colors, (list, np.ndarray)) else colors
            
            for phase_i in range(n_phases):
                # Get tuning function for this ROI and phase
                tuning_function = tuning_functions[roi_idx, sort_order, phase_i]
                
                # Close the loop
                tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
                degrees_closed = np.concatenate((degrees, [degrees[0]]))
                
                # Use phase-specific colors
                phase_color = phase_colors[phase_i % len(phase_colors)]
                
                # Create different line styles for phases
                linestyle = ['-', '--', '-.', ':'][phase_i % 4]
                alpha = 0.8
                
                # Plot phase with descriptive label
                if len(rois) > 1:
                    label = f'ROI {roi_idx} Phase {phase_i+1}'
                else:
                    label = f'Phase {phase_i+1}'
                
                ax.plot(degrees_closed, tuning_function_closed, marker='o', 
                        color=phase_color, linestyle=linestyle, alpha=alpha, 
                        label=label, linewidth=2, markersize=4)
    
    # Add legend if requested and not minimal and there are multiple traces/tuning/vectors shown
    if legend and not minimal and (((len(rois) > 1 or n_phases > 1) and show_tuning) or show_mean_vector or show_orientation_vector):
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Format title (only if not minimal)
    if show_title and not minimal:
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f'Directional tuning ({metric_name}) - Phase Overlay', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
    ax.grid(True, alpha=0.3)
    
    # Hide theta labels if requested or in minimal mode
    if not show_theta_labels or minimal:
        ax.set_thetagrids([])
    
    # Add mean direction vectors if requested (phase-dependent)
    if show_mean_vector:
        from pygor.timeseries.osds import tuning_metrics
        
        for roi_i, roi_idx in enumerate(rois):
            for phase_i in range(n_phases):
                # Get tuning function for this specific phase
                roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
                directions_sorted = np.array(directions_list)[sort_order]
                
                # Compute mean direction vector for this phase
                mean_vector = tuning_metrics.extract_mean_vector(roi_tuning_phase, directions_sorted)
                
                if not np.isnan(mean_vector['angle']):
                    # Convert to radians
                    mean_angle_rad = np.deg2rad(mean_vector['angle'])
                    
                    # Scale magnitude for visibility
                    mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning_phase)
                    
                    # Use phase-specific colors
                    phase_color = phase_colors[phase_i % len(phase_colors)]
                    
                    # Plot mean direction vector arrow
                    ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
                               xytext=(0, 0),
                               arrowprops=dict(arrowstyle='->', color=phase_color, lw=3),
                               zorder=10 + phase_i)
    
    # Add mean orientation vectors if requested (phase-dependent)
    if show_orientation_vector:
        from pygor.timeseries.osds import tuning_metrics
        
        for roi_i, roi_idx in enumerate(rois):
            for phase_i in range(n_phases):
                # Get tuning function for this specific phase
                roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
                directions_sorted = np.array(directions_list)[sort_order]
                
                # Compute mean orientation vector for this phase
                orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning_phase, directions_sorted)
                
                if not np.isnan(orientation_vector['angle']):
                    # Convert to radians
                    orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                    
                    # Scale magnitude for visibility
                    orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning_phase)
                    
                    # Use phase-specific colors with a twist (darker for orientation)
                    import matplotlib.colors as mcolors
                    phase_color = phase_colors[phase_i % len(phase_colors)]
                    base_rgb = mcolors.to_rgb(phase_color)
                    darker_color = tuple(c * 0.7 for c in base_rgb)
                    
                    # Plot mean orientation vector arrow
                    ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color=darker_color, lw=3),
                            zorder=9 + phase_i)
    
    return fig, ax



def plot_tuning_function_with_traces(osds_obj, roi_index, ax=None, show_trials=True, 
                                    metric='peak', trace_scale=0.2, minimal=True, 
                                    polar_color="#3D3AC4", trace_alpha=1, use_phases=None, 
                                    phase_colors=None, orbit_distance=0.5, trace_aspect_x=1.0, 
                                    trace_aspect_y=1.0, separate_phase_axes=False, **kwargs):
    """
    Plot tuning function with floating trace snippets in external axes.
    Designed for poster presentations with clean, minimal styling.
    
    Parameters:
    -----------
    osds_obj : OSDS object
        Object containing directional response data
    roi_index : int
        ROI index to analyze
    ax : matplotlib.axes.Axes, array/list of matplotlib.axes.Axes, or None
        External axes to plot within. If separate_phase_axes=True, should be 
        array/list with one axes per phase. If None, creates axes automatically.
    show_trials : bool
        Whether to show individual trial traces (default True)
    metric : str
        Summary metric ('peak', 'auc', 'mean', etc.)
    trace_scale : float
        Scale factor for trace subplot size (default 0.25)
    minimal : bool
        Use minimal styling (no titles, labels) (default True)
    polar_color : str
        Color for central polar plot (default '#2E8B57')
    trace_alpha : float
        Alpha for trace plots (default 0.7)
    orbit_distance : float
        Distance of trace orbit from polar plot center (default 0.5)
    trace_aspect_x : float
        Horizontal scaling factor for individual trace plots (default 1.0)
    trace_aspect_y : float
        Vertical scaling factor for individual trace plots (default 1.0)
    separate_phase_axes : bool
        If True, create separate polar + orbit plots for each phase.
        If False, overlay all phases on single polar + orbit plots (default False)
    **kwargs
        Additional arguments (data_crop, etc.)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax_polar : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        If separate_phase_axes=False: single polar plot axes
        If separate_phase_axes=True: list of polar axes (one per phase)
    """
    # Automatically use phases if dir_phase_num > 1 and use_phases not specified
    if use_phases is None:
        use_phases = osds_obj.dir_phase_num > 1
    
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ["#000000", "#4B4B4B", '#8B4513', '#483D8B']
    
    # Debug print
    # print(f"DEBUG: dir_phase_num = {osds_obj.dir_phase_num}, use_phases = {use_phases}")
    # print(f"DEBUG: phase_colors = {phase_colors}")

    # Extract data from OSDS object
    data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = osds_obj.directions_list
    
    # Get trial data if requested
    trial_data = None
    if show_trials:
        trial_data = osds_obj.split_snippets_directionally()[:, :, roi_index, :]
    
    # Handle data cropping
    data_crop = kwargs.get('data_crop', None)
    if data_crop is not None:
        data = data[:, data_crop[0]:data_crop[1]]
        if trial_data is not None:
            trial_data = trial_data[:, :, data_crop[0]:data_crop[1]]
    
    # Handle phase splitting for dual-phase data
    individual_phases_data = None
    if use_phases and osds_obj.dir_phase_num > 1:
        # print(f"DEBUG: Processing {osds_obj.dir_phase_num} phases")
        # print(f"DEBUG: Original data shape: {data.shape}")
        
        # Split data into phases
        phase_split = data.shape[1] // osds_obj.dir_phase_num
        # print(f"DEBUG: Phase split at: {phase_split}")
        
        phases_data = []
        phases_trial_data = []
        
        for phase_i in range(osds_obj.dir_phase_num):
            start_idx = phase_i * phase_split
            end_idx = (phase_i + 1) * phase_split if phase_i < osds_obj.dir_phase_num - 1 else data.shape[1]
            # print(f"DEBUG: Phase {phase_i}: indices {start_idx}:{end_idx}")
            phases_data.append(data[:, start_idx:end_idx])
            if trial_data is not None:
                phases_trial_data.append(trial_data[:, :, start_idx:end_idx])
        
        # Store individual phases for polar plot metrics
        individual_phases_data = phases_data.copy()
        
        # Use concatenated phases for traces
        data = np.concatenate(phases_data, axis=1)
        if trial_data is not None:
            trial_data = np.concatenate(phases_trial_data, axis=2)
        # print(f"DEBUG: Concatenated data shape: {data.shape}")
    
    # Convert directions and sort
    angles = np.radians(directions_list)
    directions_deg = np.array(directions_list)
    sort_indices = np.argsort(directions_deg)
    sorted_angles = angles[sort_indices]
    sorted_data = data[sort_indices]
    sorted_directions_deg = directions_deg[sort_indices]
    if trial_data is not None:
        sorted_trial_data = trial_data[sort_indices]
    
    # Calculate metric values for polar plot (handle phases)
    if use_phases and osds_obj.dir_phase_num > 1 and individual_phases_data is not None:
        # print(f"DEBUG: Calculating metrics for {osds_obj.dir_phase_num} phases")
        # Calculate metrics for each phase separately
        phase_values = []
        
        for phase_i in range(osds_obj.dir_phase_num):
            # Use the individual phase data, then apply sort
            phase_data = individual_phases_data[phase_i][sort_indices]
            # print(f"DEBUG: Phase {phase_i} data shape: {phase_data.shape}")
            
            if metric == 'peak':
                phase_vals = np.array([np.max(np.abs(trace)) for trace in phase_data])
            elif metric == 'auc':
                phase_vals = np.array([np.trapz(np.abs(trace)) for trace in phase_data])
            elif metric == 'mean':
                phase_vals = np.array([np.mean(trace) for trace in phase_data])
            elif callable(metric):
                phase_vals = np.array([metric(trace) for trace in phase_data])
            else:
                phase_vals = np.array([np.max(np.abs(trace)) for trace in phase_data])
            
            # print(f"DEBUG: Phase {phase_i} values: {phase_vals[:3]}...")  # Show first 3 values
            phase_values.append(phase_vals)
    else:
        # Single phase calculation
        if metric == 'peak':
            values = np.array([np.max(np.abs(trace)) for trace in sorted_data])
        elif metric == 'auc':
            values = np.array([np.trapz(np.abs(trace)) for trace in sorted_data])
        elif metric == 'mean':
            values = np.array([np.mean(trace) for trace in sorted_data])
        elif callable(metric):
            values = np.array([metric(trace) for trace in sorted_data])
        else:
            values = np.array([np.max(np.abs(trace)) for trace in sorted_data])  # Default to peak
    
    # Branch into separate mode vs overlay mode
    if separate_phase_axes and use_phases and osds_obj.dir_phase_num > 1 and individual_phases_data is not None:
        # SEPARATE AXES MODE: Each phase gets its own polar plot + orbit traces
        # print(f"DEBUG: Using separate axes mode for {osds_obj.dir_phase_num} phases")
        
        # Handle axes creation/validation
        if ax is None:
            # Create figure and axes automatically with proper spacing
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, osds_obj.dir_phase_num, 
                                 figsize=(6*osds_obj.dir_phase_num, 5))
            # Add space between subplots to prevent orbit overlap
            plt.subplots_adjust(wspace=0.8)  # Increase horizontal spacing
            if osds_obj.dir_phase_num == 1:
                ax = [ax]  # Make it a list for consistency
        elif hasattr(ax, '__len__') and hasattr(ax, '__getitem__'):
            # Handle list, tuple, or numpy array of axes
            if len(ax) != osds_obj.dir_phase_num:
                raise ValueError(f"When separate_phase_axes=True, ax array must have {osds_obj.dir_phase_num} axes, got {len(ax)}")
            fig = ax[0].figure
            
            # Intelligently adjust spacing to prevent orbit overlap
            if osds_obj.dir_phase_num > 1:
                # Check current subplot positions to determine if spacing adjustment is needed
                current_positions = [a.get_position() for a in ax]
                
                # Calculate the minimum spacing needed based on orbit_distance and trace_scale
                # Each orbit extends approximately orbit_distance in each direction from the center
                required_spacing = 2 * orbit_distance * trace_scale
                
                # Check if current spacing is sufficient
                if len(current_positions) >= 2:
                    current_spacing = current_positions[1].x0 - (current_positions[0].x0 + current_positions[0].width)
                    if current_spacing < required_spacing:
                        # Adjust spacing to prevent overlap
                        optimal_wspace = max(0.4, required_spacing * 2)  # At least 0.4, or calculated value
                        fig.subplots_adjust(wspace=optimal_wspace)
        else:
            # Single axes provided but we need multiple
            raise ValueError(f"When separate_phase_axes=True, ax must be a list/tuple/array of {osds_obj.dir_phase_num} axes, or None for automatic creation")
        
        if ax is not None:
            fig = ax[0].figure
        ax_polars = []
        all_trace_axes = []
        
        # Calculate global limits for both polar plots and orbit traces
        # Use the same metric-based scaling for consistency
        all_phase_metric_values = np.concatenate(phase_values)
        global_metric_min = np.min(all_phase_metric_values)
        global_metric_max = np.max(all_phase_metric_values)
        
        # Also calculate global limits for raw trace data
        if trial_data is not None:
            all_trial_data = []
            for phase_i in range(osds_obj.dir_phase_num):
                phase_trial_data = phases_trial_data[phase_i][sort_indices]  # Apply sort to trial data
                all_trial_data.append(phase_trial_data.flatten())
            all_raw_data = np.concatenate(all_trial_data)
        else:
            all_raw_data = np.concatenate([phase_data.flatten() for phase_data in individual_phases_data])
        
        y_min_raw = np.min(all_raw_data)
        y_max_raw = np.max(all_raw_data)
        y_range_raw = y_max_raw - y_min_raw
        y_buffer_raw = y_range_raw if minimal else y_range_raw * 0.1
        y_min_global = y_min_raw - y_buffer_raw
        y_max_global = y_max_raw + y_buffer_raw
        
        # Process each phase separately
        for phase_i in range(osds_obj.dir_phase_num):
            curr_ax = ax[phase_i]
            phase_color = phase_colors[phase_i % len(phase_colors)]
            
            # Convert to polar - use subplot positioning instead of absolute coordinates
            ax_gridspec = curr_ax.get_subplotspec()
            curr_ax.remove()
            ax_polar = fig.add_subplot(ax_gridspec, projection='polar')
            ax_polar.set_theta_zero_location('E')
            ax_polars.append(ax_polar)
            
            # Plot polar for this phase only
            polar_angles = np.append(sorted_angles, sorted_angles[0])
            phase_vals = phase_values[phase_i]
            phase_polar_values = np.append(phase_vals, phase_vals[0])
            
            ax_polar.plot(polar_angles, phase_polar_values, color=phase_color, alpha=1)
            ax_polar.fill(polar_angles, phase_polar_values, alpha=0.33, color=phase_color)
            
            # Styling - use global scaling for consistency
            if minimal:
                ax_polar.grid(True, alpha=0.3)
                ax_polar.set_thetagrids([])
                # Use global metric max for consistent scaling across phases
                max_tick = int(np.ceil(global_metric_max))
                if max_tick <= 3:
                    ticks = [1, 2, max_tick] if max_tick > 2 else [1, max_tick]
                else:
                    step = max_tick // 3
                    if step == 0:
                        step = 1
                    ticks = [step, 2*step, 3*step]
                    if ticks[-1] > max_tick:
                        ticks = [step, 2*step, max_tick]
                ticks = sorted(list(set(ticks)))[:3]
                ax_polar.set_rgrids(ticks, alpha=0.7)
                # Set consistent radial limits
                ax_polar.set_ylim(0, global_metric_max * 1.1)
            else:
                ax_polar.grid(True, alpha=0.3)
                ax_polar.set_ylim(0, global_metric_max * 1.1)
            
            # Add orbit traces for this phase
            phase_trace_axes = []
            
            # Get phase-specific data
            phase_data = individual_phases_data[phase_i][sort_indices]
            phase_trial_data = None
            if show_trials and trial_data is not None:
                phase_trial_data = phases_trial_data[phase_i][sort_indices]
            
            # Positioning calculations - get fresh position after any layout changes
            fig.canvas.draw_idle()  # Ensure layout is updated
            axes_bbox = ax_polar.get_position()
            axes_center_x = axes_bbox.x0 + axes_bbox.width / 2
            axes_center_y = axes_bbox.y0 + axes_bbox.height / 2
            
            fig_size = fig.get_size_inches()
            fig_aspect = fig_size[0] / fig_size[1]
            
            base_orbit_distance = orbit_distance
            if fig_aspect > 1:
                orbit_radius_x = base_orbit_distance / fig_aspect
                orbit_radius_y = base_orbit_distance
            else:
                orbit_radius_x = base_orbit_distance
                orbit_radius_y = base_orbit_distance * fig_aspect
            
            base_trace_size = trace_scale * min(axes_bbox.width, axes_bbox.height)
            trace_width = base_trace_size * trace_aspect_x
            trace_height = base_trace_size * trace_aspect_y
            
            # Create orbit traces for this phase
            for i, angle in enumerate(sorted_angles):
                x_center = axes_center_x + orbit_radius_x * np.cos(angle)
                y_center = axes_center_y + orbit_radius_y * np.sin(angle)
                
                trace_ax = fig.add_axes([
                    x_center - trace_width/2,
                    y_center - trace_height/2, 
                    trace_width,
                    trace_height
                ])
                
                # Plot trace data for this phase only
                if show_trials and phase_trial_data is not None:
                    trace_ax.plot(phase_trial_data[i].T, color=phase_color, alpha=0.1)
                trace_ax.plot(phase_data[i], color=phase_color, alpha=trace_alpha)
                
                trace_ax.set_facecolor((1, 1, 1, 0))
                trace_ax.axhline(0, color='gray', alpha=0.4)
                
                # Clean styling
                trace_ax.set_xlim(0, len(phase_data[i]))
                trace_ax.set_ylim(y_min_global, y_max_global)
                trace_ax.set_xticks([])
                trace_ax.set_yticks([])
                
                for spine in trace_ax.spines.values():
                    spine.set_visible(False)
                    
                phase_trace_axes.append(trace_ax)
            
            all_trace_axes.append(phase_trace_axes)
        
        
        return fig, ax_polars
        
    else:
        # OVERLAY MODE: Single polar plot with overlaid phases (original behavior)
        # print("DEBUG: Using overlay mode")
        
        # Handle axes creation/validation
        if ax is None:
            # Create figure and axes automatically
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        elif hasattr(ax, '__len__') and hasattr(ax, '__getitem__'):
            ax = ax[0]  # Use first axes if array/list provided
            fig = ax.figure
        else:
            # Single axes provided
            fig = ax.figure
        
        # Convert to polar using subplot positioning instead of absolute coordinates
        ax_gridspec = ax.get_subplotspec()
        ax.remove()
        ax_polar = fig.add_subplot(ax_gridspec, projection='polar')
        ax_polar.set_theta_zero_location('E')
        
        # Plot central polar (handle phases)
        polar_angles = np.append(sorted_angles, sorted_angles[0])
        
        if use_phases and osds_obj.dir_phase_num > 1 and 'phase_values' in locals():
            # print(f"DEBUG: Plotting {len(phase_values)} phases on polar plot")
            # Plot each phase on polar plot
            for phase_i, phase_vals in enumerate(phase_values):
                phase_polar_values = np.append(phase_vals, phase_vals[0])
                phase_color = phase_colors[phase_i % len(phase_colors)]
                linestyle = ['-', '--', '-.', ':'][phase_i % 4]
                
                # print(f"DEBUG: Phase {phase_i}: color={phase_color}, style={linestyle}")
                ax_polar.plot(polar_angles, phase_polar_values, color=phase_color, 
                             alpha=0.9, linestyle=linestyle)
                ax_polar.fill(polar_angles, phase_polar_values, alpha=0.2, color=phase_color)
        else:
            # print("DEBUG: Single phase polar plot")
            # Single phase polar plot
            polar_values = np.append(values, values[0])
            ax_polar.plot(polar_angles, polar_values, color=polar_color, alpha=0.8)
            ax_polar.fill(polar_angles, polar_values, alpha=0.2, color=polar_color)
        
        # Calculate global limits for both polar plots and orbit traces
        # Get metric values for consistent scaling
        if use_phases and osds_obj.dir_phase_num > 1 and 'phase_values' in locals():
            all_metric_values = np.concatenate(phase_values)
            global_metric_max = max([np.max(phase_vals) for phase_vals in phase_values])
        else:
            all_metric_values = values
            global_metric_max = np.max(values)
        
        # Calculate global y-limits for trace data
        if trial_data is not None:
            all_data = sorted_trial_data.flatten()
        else:
            all_data = sorted_data.flatten()
        
        y_min = np.min(all_data)
        y_max = np.max(all_data)
        y_range = y_max - y_min
        y_buffer = y_range if minimal else y_range * 0.1
        y_min_global = y_min - y_buffer
        y_max_global = y_max + y_buffer
        
        if minimal:
            # Minimal polar styling
            ax_polar.grid(True, alpha=.3)
            ax_polar.set_thetagrids([])  # No angle labels
            
            # Use global metric max for consistent polar scaling
            max_tick = int(np.ceil(global_metric_max))
            if max_tick <= 3:
                ticks = [1, 2, max_tick] if max_tick > 2 else [1, max_tick]
            else:
                # Create 3 evenly spaced integer ticks
                step = max_tick // 3
                if step == 0:
                    step = 1
                ticks = [step, 2*step, 3*step]
                # Adjust if needed to not exceed max_tick
                if ticks[-1] > max_tick:
                    ticks = [step, 2*step, max_tick]
            
            # Remove duplicates and ensure we have at most 3 ticks
            ticks = sorted(list(set(ticks)))[:3]
            ax_polar.set_rgrids(ticks, alpha=0.7)
            # Set consistent radial limits
            ax_polar.set_ylim(0, global_metric_max * 1.1)
        else:
            # Full radial grid for non-minimal mode
            ax_polar.grid(True, alpha=0.3)
            ax_polar.set_ylim(0, global_metric_max * 1.1)

        # Add floating trace snippets around perimeter
        trace_axes = []
        
        # Get axes position and figure aspect ratio - get fresh position after any layout changes
        fig.canvas.draw_idle()  # Ensure layout is updated
        axes_bbox = ax_polar.get_position()
        axes_center_x = axes_bbox.x0 + axes_bbox.width / 2
        axes_center_y = axes_bbox.y0 + axes_bbox.height / 2
        
        # Get figure size in inches to calculate display aspect ratio
        fig_size = fig.get_size_inches()
        fig_aspect = fig_size[0] / fig_size[1]  # width / height
        
        # Use user-specified orbit distance with aspect ratio correction
        base_orbit_distance = orbit_distance
        
        # Adjust orbit radius components to maintain circular appearance in display coordinates
        if fig_aspect > 1:  # Wide figure
            orbit_radius_x = base_orbit_distance / fig_aspect  # Compress horizontally
            orbit_radius_y = base_orbit_distance
        else:  # Tall figure
            orbit_radius_x = base_orbit_distance
            orbit_radius_y = base_orbit_distance * fig_aspect  # Compress vertically
        
        # Base trace size (can be made rectangular with aspect ratios)
        base_trace_size = trace_scale * min(axes_bbox.width, axes_bbox.height)
        trace_width = base_trace_size * trace_aspect_x
        trace_height = base_trace_size * trace_aspect_y
        
        for i, angle in enumerate(sorted_angles):
            # Circular positioning with aspect ratio correction
            x_center = axes_center_x + orbit_radius_x * np.cos(angle)
            y_center = axes_center_y + orbit_radius_y * np.sin(angle)
            
            # Create trace subplot (can be rectangular now)
            trace_ax = fig.add_axes([
                x_center - trace_width/2,
                y_center - trace_height/2, 
                trace_width,
                trace_height
            ])
            
            # Plot trace (handle phases with separation line)
            if show_trials and trial_data is not None:
                trace_ax.plot(sorted_trial_data[i].T, 'k-',alpha=0.1)
            trace_ax.plot(sorted_data[i], 'k-', alpha=trace_alpha)
            # set background to transparent
            trace_ax.set_facecolor((1, 1, 1, 0))
            trace_ax.axhline(0, color='gray', alpha=0.4)
            
            # Add phase separation line if multi-phase
            if use_phases and osds_obj.dir_phase_num > 1 and individual_phases_data is not None:
                phase_split = len(individual_phases_data[0][0])  # Length of one phase
                for phase_i in range(1, osds_obj.dir_phase_num):
                    sep_x = phase_i * phase_split
                    # trace_ax.axvline(sep_x, color='k', alpha=0.2, 
                    #                 linestyle='-', zorder = -1)
                    # Alternating shaded background for phases
                    if phase_i % 2 == 1:
                        trace_ax.axvspan((phase_i - 1) * phase_split, 
                                        phase_i * phase_split, 
                                        facecolor='gray', alpha=0.5, zorder=-1, edgecolor='none')
            
            # Clean minimal styling
            trace_ax.set_xlim(0, len(sorted_data[i]))
            trace_ax.set_ylim(y_min_global, y_max_global)
            trace_ax.set_xticks([])
            trace_ax.set_yticks([])
            
            # Remove all spines for clean look
            for spine in trace_ax.spines.values():
                spine.set_visible(False)
                
            trace_axes.append(trace_ax)
        
        
        return fig, ax_polar