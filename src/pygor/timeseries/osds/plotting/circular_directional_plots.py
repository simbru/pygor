import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_directional_responses_circular(data, directions_list=None, figsize=(8, 8)):
    """
    Plot directional responses in a circular arrangement.

    Parameters:
    -----------
    data : np.ndarray
            Array of shape (n_directions, n_timepoints) containing response traces
    directions_list : list of int/float, optional
            List of direction values in degrees. If None, uses evenly spaced angles.
    figsize : tuple
            Figure size (width, height)
    """
    n_directions = data.shape[0]

    # Convert direction degrees to radians for positioning
    if directions_list is not None:
        angles = np.radians(directions_list)
    else:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    fig = plt.figure(figsize=figsize, facecolor="white")

    for i, angle in enumerate(angles):
        # Calculate subplot position (start from right, go counter-clockwise to match polar)
        x_center = 0.5 + 0.32 * np.cos(angle)
        y_center = 0.5 + 0.32 * np.sin(angle)

        # Create subplot
        ax = fig.add_axes([x_center - 0.06, y_center - 0.06, 0.12, 0.12])

        # Plot trace
        ax.plot(data[i], "k-", linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        # Clean styling
        ax.set_xlim(0, len(data[i]))
        y_range = np.max(np.abs(data[i])) * 1.1 if np.max(np.abs(data[i])) > 0 else 1
        ax.set_ylim(-y_range, y_range)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f8f8f8")

        # Add direction label
        # if directions_list is not None:
        #     label_x = x_center + 0.08 * np.cos(angle - np.pi/2)
        #     label_y = y_center + 0.08 * np.sin(angle - np.pi/2)
        #     fig.text(label_x, label_y, f'{int(directions_list[i])}°',
        #             ha='center', va='center', fontsize=10, fontweight='bold')

    return fig


def plot_directional_responses_circular_with_polar(
    osds_obj,
    directions_list=None,
    figsize=(10, 10),
    metric="peak",
    polar_kwargs=None,
    polar_size=0.3,
    roi_index=-1,
    show_trials=True,
    data_crop=None,
    use_phases=None,
    phase_colors=None,
):
    """
    Plot directional responses in a circular arrangement with central polar plot.

    Parameters:
    -----------
    data : np.ndarray or None
            Array of shape (n_directions, n_timepoints) containing response traces.
            If osds_obj is provided, this can be None.
    directions_list : list of int/float, optional
            List of direction values in degrees. If None, uses evenly spaced angles.
    figsize : tuple
            Figure size (width, height)
    metric : str or callable
            Summary metric to plot in polar plot. Options:
            - 'peak': maximum absolute value
            - 'auc': area under curve (absolute)
            - 'mean': mean response
            - 'peak_positive': maximum positive value
            - 'peak_negative': minimum negative value
            - custom function that takes 1D array and returns scalar
    polar_kwargs : dict, optional
            Additional keyword arguments for polar plot styling
    polar_size : float, optional
            Size of the central polar plot as fraction of figure (default 0.3)
    osds_obj : OSDS object, optional
            If provided, will extract data and optionally show individual trials
    roi_index : int, optional
            ROI index to plot (default -1 for last ROI)
    show_trials : bool, optional
            Whether to show individual trial traces as faint lines (default False)
    data_crop : tuple, optional
            Tuple of (start, end) indices to crop data timepoints
    use_phases : bool or None
            If None, uses osds_obj.dir_phase_num > 1 to decide
            If True, forces phase analysis with overlay in polar plot
            If False, forces single-phase analysis
    phase_colors : list or None
            Colors for each phase. If None, uses default colors

    Returns:
    --------
    fig : matplotlib.figure.Figure
            The figure object
    ax_polar : matplotlib.axes.Axes
            The polar plot axes object
    """

    # Automatically use phases if dir_phase_num > 1 and use_phases not specified
    if use_phases is None:
        use_phases = osds_obj.dir_phase_num > 1
    
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
    # Extract data from OSDS object if provided
    trial_data = None  # Initialize here so it's available throughout the function

    data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = osds_obj.directions_list

    # Get trial data if requested
    if show_trials:
        # Shape: (n_directions, n_rois, n_trials, n_timepoints) -> (n_directions, n_trials, n_timepoints)
        trial_data = osds_obj.split_snippets_directionally()[:, roi_index, :, :]

    if data_crop is not None:
        print(data.shape)
        data = data[:, data_crop[0] : data_crop[1]]
        if trial_data is not None:
            trial_data = trial_data[:, :, data_crop[0] : data_crop[1]]

    # # Handle the case where data is None but osds_obj is provided
    # data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])

    n_directions = data.shape[0]

    # Convert direction degrees to radians for positioning
    if directions_list is not None:
        angles = np.radians(directions_list)
        directions_deg = np.array(directions_list)
    else:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        directions_deg = np.degrees(angles)

    # Calculate summary metric for each direction
    if metric == "peak":
        values = np.array([np.max(np.abs(trace)) for trace in data])
    elif metric == "auc":
        values = np.array([np.trapz(np.abs(trace)) for trace in data])
    elif metric == "mean":
        values = np.array([np.mean(trace) for trace in data])
    elif metric == "peak_positive":
        values = np.array([np.max(trace) for trace in data])
    elif metric == "peak_negative":
        values = np.array([np.min(trace) for trace in data])
    elif callable(metric):
        values = np.array([metric(trace) for trace in data])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort by angle for proper polar plot connectivity
    sort_indices = np.argsort(directions_deg)
    sorted_angles = angles[sort_indices]
    sorted_values = values[sort_indices]
    sorted_data = data[sort_indices]
    sorted_directions_deg = directions_deg[sort_indices]
    # Also sort trial data if it exists
    if show_trials and trial_data is not None:
        sorted_trial_data = trial_data[sort_indices]

    # Set up polar plot defaults
    default_polar_kwargs = {
        "color": "#2E8B57",  # Sea green instead of red
        "linewidth": 2,
        "marker": "o",
        "markersize": 6,
        "alpha": 0.8,
    }
    if polar_kwargs:
        default_polar_kwargs.update(polar_kwargs)

    fig = plt.figure(figsize=figsize, facecolor="white")

    # Create central polar plot (larger and closer to periphery)
    ax_polar = fig.add_axes([0.225, 0.22, 0.55, 0.55], projection="polar")

    # Plot polar data (need to append first point to close the circle)
    polar_angles = np.append(sorted_angles, sorted_angles[0])
    polar_values = np.append(sorted_values, sorted_values[0])

    ax_polar.plot(polar_angles, polar_values, **default_polar_kwargs)
    ax_polar.fill(
        polar_angles, polar_values, alpha=0.3, color=default_polar_kwargs["color"]
    )

    # Style polar plot
    ax_polar.set_theta_zero_location("E")  # 0° at right (90° north)
    # ax_polar.set_theta_direction(-1)  # Clockwise
    ax_polar.set_title(
        f"Directional Tuning\n({metric.replace('_', ' ').title()})",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax_polar.grid(True, alpha=0.3)
    # Calculate global y-limits for all traces
    if show_trials and trial_data is not None:
        # Use trial data for scaling to capture full variability
        all_trace_data = sorted_trial_data.flatten()
    else:
        # Use average data for scaling
        all_trace_data = sorted_data.flatten()

    global_min = np.min(all_trace_data)
    global_max = np.max(all_trace_data)
    global_range = global_max - global_min
    padding = global_range * 0.1  # 25% padding
    y_min_global = global_min - padding
    y_max_global = global_max + padding
    # Set minimum range to (-3, 3)
    y_min_global = min(y_min_global, -5)
    y_max_global = max(y_max_global, 5)
    # Add individual trace plots around the perimeter
    for i, angle in enumerate(sorted_angles):
        # Calculate subplot position (further out to accommodate central polar plot)
        # Fixed angle calculation to match standard directional conventions
        radius = 0.38  # Increased radius to make room for central plot
        x_center = 0.5 + radius * np.cos(angle)  # 0° at right, counter-clockwise
        y_center = 0.5 + radius * np.sin(angle)  # 0° at right, counter-clockwise

        # Create subplot
        subplot_size = 0.08  # Slightly smaller to fit more around
        ax = fig.add_axes(
            [
                x_center - subplot_size / 2,
                y_center - subplot_size / 2,
                subplot_size,
                subplot_size,
            ]
        )

        # Plot trace - use sorted data
        if show_trials:
            ax.plot(sorted_trial_data[i].T, "k-", linewidth=0.5, alpha=0.33)
        ax.plot(sorted_data[i], "k-", linewidth=1)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        # Clean neutral background
        # ax.set_facecolor("#f8f8f8")

        # Clean styling
        ax.set_xlim(0, len(sorted_data[i]))  # Fixed: use sorted_data
        # y_range = np.max(np.abs(data)) * 1.25 if np.max(np.abs(data)) > 5 else 5
        # y_range = data_absmax + data_absmax * 0.5
        # ax.set_ylim(-y_range, y_range)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(f"{directions_deg[i]:.0f}° or {angle:.0f} rad", fontsize=8)
        ax.set_title(f"{sorted_directions_deg[i]:.0f}°", fontsize=8)
        # sns.despine(ax=ax, left=False, bottom=False, right=True, top=False)

    return fig, ax_polar


def plot_directional_responses_dual_phase(
    osds_obj,
    phase_split=None,
    directions_list=None,
    figsize=(12, 10),
    metric="peak",
    polar_kwargs=None,
    roi_index=-1,
    show_trials=True,
    phase_colors=("#2E8B57", "#B8860B"),  # Sea green, Dark goldenrod
):
    """
    Plot directional responses for two stimulus phases (OFF->ON and ON->OFF)
    with overlapping polar plots and separate trace arrangements.

    Parameters:
    -----------
    osds_obj : OSDS object
        Object containing the directional response data
    phase_split : int
        Frame number where phase 1 ends and phase 2 begins (default 3200)
    directions_list : list of int/float, optional
        List of direction values in degrees
    figsize : tuple
        Figure size (width, height)
    metric : str or callable
        Summary metric for polar plots
    polar_kwargs : dict, optional
        Additional keyword arguments for polar plot styling
    roi_index : int, optional
        ROI index to plot (default -1 for last ROI)
    show_trials : bool, optional
        Whether to show individual trial traces
    phase_colors : tuple
        Colors for (phase1, phase2) plots

    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax_polar : matplotlib.axes.Axes
        The central polar plot axes
    """

    # Extract data
    data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = osds_obj.directions_list

    # Get trial data if requested
    trial_data = None
    if show_trials:
        trial_data = osds_obj.split_snippets_directionally()[:, roi_index, :, :]
    if phase_split is None:
        phase_split = phase_split = osds_obj.averages.shape[1] // osds_obj.trigger_mode // 2  # Should be ~2866
        print(f"Phase split: {phase_split}")
    else:
        phase_split = int(phase_split)
    # Split data into two phases
    phase1_data = data[:, :phase_split]  # OFF->ON
    phase2_data = data[:, phase_split:]  # ON->OFF

    if trial_data is not None:
        phase1_trial_data = trial_data[:, :, :phase_split]
        phase2_trial_data = trial_data[:, :, phase_split:]

    n_directions = data.shape[0]

    # Convert directions and sort
    if directions_list is not None:
        angles = np.radians(directions_list)
        directions_deg = np.array(directions_list)
    else:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        directions_deg = np.degrees(angles)

    sort_indices = np.argsort(directions_deg)
    sorted_angles = angles[sort_indices]
    sorted_directions_deg = directions_deg[sort_indices]

    # Sort both phases
    sorted_phase1_data = phase1_data[sort_indices]
    sorted_phase2_data = phase2_data[sort_indices]

    if trial_data is not None:
        sorted_phase1_trial_data = phase1_trial_data[sort_indices]
        sorted_phase2_trial_data = phase2_trial_data[sort_indices]

    # Calculate metrics for both phases
    if metric == "peak":
        phase1_values = np.array(
            [np.max(np.abs(trace)) for trace in sorted_phase1_data]
        )
        phase2_values = np.array(
            [np.max(np.abs(trace)) for trace in sorted_phase2_data]
        )
    elif metric == "auc":
        phase1_values = np.array(
            [np.trapz(np.abs(trace)) for trace in sorted_phase1_data]
        )
        phase2_values = np.array(
            [np.trapz(np.abs(trace)) for trace in sorted_phase2_data]
        )
    elif metric == "mean":
        phase1_values = np.array([np.mean(trace) for trace in sorted_phase1_data])
        phase2_values = np.array([np.mean(trace) for trace in sorted_phase2_data])
    elif metric == "peak_positive":
        phase1_values = np.array([np.max(trace) for trace in sorted_phase1_data])
        phase2_values = np.array([np.max(trace) for trace in sorted_phase2_data])
    elif metric == "peak_negative":
        phase1_values = np.array([np.min(trace) for trace in sorted_phase1_data])
        phase2_values = np.array([np.min(trace) for trace in sorted_phase2_data])
    elif callable(metric):
        phase1_values = np.array([metric(trace) for trace in sorted_phase1_data])
        phase2_values = np.array([metric(trace) for trace in sorted_phase2_data])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate global y-limits using both phases
    if show_trials and trial_data is not None:
        all_trace_data = np.concatenate(
            [sorted_phase1_trial_data.flatten(), sorted_phase2_trial_data.flatten()]
        )
    else:
        all_trace_data = np.concatenate(
            [sorted_phase1_data.flatten(), sorted_phase2_data.flatten()]
        )

    global_min = np.min(all_trace_data)
    global_max = np.max(all_trace_data)
    global_range = global_max - global_min
    padding = global_range * 0.25
    y_min_global = global_min - padding
    y_max_global = global_max + padding

    # Set minimum range to (-3, 3)
    y_min_global = min(y_min_global, -3)
    y_max_global = max(y_max_global, 3)

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor="white")

    # Create central polar plot
    ax_polar = fig.add_axes([0.225, 0.22, 0.55, 0.55], projection="polar")

    # Plot both phases on same polar plot
    for phase_idx, (values, color, label) in enumerate(
        zip([phase1_values, phase2_values], phase_colors, ["OFF→ON", "ON→OFF"])
    ):
        # Close the circle
        polar_angles = np.append(sorted_angles, sorted_angles[0])
        polar_values = np.append(values, values[0])

        # Set up plot style
        plot_kwargs = {
            "color": color,
            "linewidth": 2.5,
            "marker": "o",
            "markersize": 6,
            "alpha": 0.8,
            "label": label,
        }
        if polar_kwargs:
            plot_kwargs.update(polar_kwargs)

        ax_polar.plot(polar_angles, polar_values, **plot_kwargs)
        ax_polar.fill(polar_angles, polar_values, alpha=0.2, color=color)

    # Style polar plot
    # ax_polar.set_theta_zero_location("E")  # 0° at right (90° north)
    # ax_polar.set_theta_direction(-1)
    ax_polar.set_title(
        f"Dual Phase Directional Tuning\n({metric.replace('_', ' ').title()})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_polar.grid(True, alpha=0.3)
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1))

    # Add trace plots around perimeter - Phase 1 (inner ring)
    for i, angle in enumerate(sorted_angles):
        # Single trace plot per direction
        radius = 0.38  
        x_center = 0.5 + radius * np.cos(angle)
        y_center = 0.5 + radius * np.sin(angle)
        
        subplot_size = 0.08
        ax = fig.add_axes([
            x_center - subplot_size / 2,
            y_center - subplot_size / 2,
            subplot_size,
            subplot_size
        ])
        
        # Concatenate both phases into a continuous trace
        if show_trials and trial_data is not None:
            # Concatenate trial data for both phases
            combined_trial_data = np.concatenate([sorted_phase1_trial_data[i], sorted_phase2_trial_data[i]], axis=1)
            ax.plot(combined_trial_data.T, "k-", linewidth=0.3, alpha=0.3)
        
        # Concatenate average traces for both phases
        combined_data = np.concatenate([sorted_phase1_data[i], sorted_phase2_data[i]])
        ax.plot(combined_data, "k-", linewidth=2)
        
        # Add vertical line to separate the two phases
        phase1_len = len(sorted_phase1_data[i])
        ax.axvline(phase1_len, color="k", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(0, len(combined_data))
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{sorted_directions_deg[i]:.0f}°", fontsize=8)
    return fig, ax_polar


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


def plot_orientation_tuning_cartesian(responses, directions_deg, figsize=(8, 6), 
                                     color='black', linewidth=2, marker='o', 
                                     markersize=6, show_osi=True, osi_color='red',
                                     title=None, xlabel='Orientation (degrees)', 
                                     ylabel='Response', show_grid=True):
    """
    Plot orientation tuning curve in cartesian coordinates.
    
    Creates a standard cartesian plot of orientation tuning (0-180°) commonly 
    used in vision research papers. Shows OSI calculation points if requested.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
    figsize : tuple
        Figure size (width, height)
    color : str or tuple
        Color for the main tuning curve
    linewidth : float
        Line width for the tuning curve
    marker : str
        Marker style for data points
    markersize : float
        Size of data point markers
    show_osi : bool
        Whether to show OSI calculation points
    osi_color : str or tuple
        Color for OSI calculation markers
    title : str or None
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    show_grid : bool
        Whether to show grid lines
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    osi_info : dict
        Dictionary containing OSI calculation results
    """
    from ..tuning_metrics import compute_orientation_tuning, compute_orientation_selectivity_index
    
    # Get orientation tuning
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']
    
    # Get OSI information
    osi_info = compute_orientation_selectivity_index(responses, directions_deg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Sort orientations for smooth plotting
    sort_indices = np.argsort(orientations)
    sorted_orientations = orientations[sort_indices]
    sorted_responses = orientation_responses[sort_indices]
    
    # Plot main tuning curve
    ax.plot(sorted_orientations, sorted_responses, color=color, linewidth=linewidth, 
            marker=marker, markersize=markersize, label='Orientation tuning')
    
    # Show OSI calculation points if requested
    if show_osi and not np.isnan(osi_info['osi']):
        # Mark preferred orientation
        ax.scatter(osi_info['preferred_orientation'], osi_info['preferred_response'], 
                  color=osi_color, s=markersize*15, marker='s', 
                  label=f'Preferred ({osi_info["preferred_orientation"]:.0f}°)', 
                  zorder=10, edgecolors='white', linewidth=1)
        
        # Mark orthogonal orientation
        ax.scatter(osi_info['orthogonal_orientation'], osi_info['orthogonal_response'], 
                  color=osi_color, s=markersize*15, marker='^', 
                  label=f'Orthogonal ({osi_info["orthogonal_orientation"]:.0f}°)', 
                  zorder=10, edgecolors='white', linewidth=1)
        
        # Add OSI text
        ax.text(0.02, 0.98, f'OSI = {osi_info["osi"]:.3f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    # Styling
    ax.set_xlim(0, 180)
    ax.set_ylim(0, max(orientation_responses) * 1.1 if max(orientation_responses) > 0 else 1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-axis ticks at common orientations
    ax.set_xticks([0, 45, 90, 135, 180])
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend if showing OSI
    if show_osi and not np.isnan(osi_info['osi']):
        ax.legend(loc='upper right', fontsize=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout()
    
    return fig, ax, osi_info


def plot_orientation_tuning_comparison(responses, directions_deg, figsize=(12, 5)):
    """
    Plot side-by-side comparison of polar and cartesian orientation tuning.
    
    Creates a two-panel figure showing both polar and cartesian representations
    of the same orientation tuning data for easy comparison.
    
    Parameters:
    -----------
    responses : array-like
        Response values for each direction
    directions_deg : array-like
        Direction values in degrees
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : list
        List containing [polar_ax, cartesian_ax]
    osi_info : dict
        Dictionary containing OSI calculation results
    """
    from ..tuning_metrics import compute_orientation_tuning, extract_orientation_vector
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Polar plot (left)
    ax_polar = fig.add_subplot(121, projection='polar')
    
    # Get orientation data
    orientation_data = compute_orientation_tuning(responses, directions_deg)
    orientations = orientation_data['orientations']
    orientation_responses = orientation_data['responses']
    
    # Convert to radians and double for polar plot
    orientations_rad = np.deg2rad(orientations * 2)  # Double for 0-360° display
    
    # Sort for smooth plotting
    sort_indices = np.argsort(orientations)
    sorted_orientations_rad = orientations_rad[sort_indices]
    sorted_responses = orientation_responses[sort_indices]
    
    # Close the polar plot
    sorted_orientations_rad = np.append(sorted_orientations_rad, sorted_orientations_rad[0])
    sorted_responses = np.append(sorted_responses, sorted_responses[0])
    
    # Plot polar tuning curve
    ax_polar.plot(sorted_orientations_rad, sorted_responses, 'b-', linewidth=2, marker='o', markersize=4)
    ax_polar.fill(sorted_orientations_rad, sorted_responses, alpha=0.3, color='blue')
    
    # Add orientation vector
    orient_vector = extract_orientation_vector(responses, directions_deg)
    if not np.isnan(orient_vector['angle']):
        vector_angle_rad = np.deg2rad(orient_vector['angle'] * 2)
        vector_magnitude = orient_vector['magnitude'] * max(sorted_responses)
        ax_polar.annotate('', xy=(vector_angle_rad, vector_magnitude), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=3),
                         zorder=10)
    
    # Polar plot styling
    ax_polar.set_ylim(0, max(sorted_responses) * 1.1)
    ax_polar.set_title('Polar Representation', fontsize=14, fontweight='bold', pad=20)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    
    # Cartesian plot (right)
    ax_cartesian = fig.add_subplot(122)
    fig_temp, ax_temp, osi_info = plot_orientation_tuning_cartesian(
        responses, directions_deg, figsize=(6, 5), show_osi=True
    )
    
    # Copy cartesian plot to our subplot
    for line in ax_temp.get_lines():
        ax_cartesian.plot(line.get_xdata(), line.get_ydata(), 
                         color=line.get_color(), linewidth=line.get_linewidth(),
                         marker=line.get_marker(), markersize=line.get_markersize())
    
    for collection in ax_temp.collections:
        ax_cartesian.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
                           c=collection.get_facecolors(), s=collection.get_sizes(),
                           marker=collection.get_paths()[0] if collection.get_paths() else 'o')
    
    # Copy cartesian styling
    ax_cartesian.set_xlim(ax_temp.get_xlim())
    ax_cartesian.set_ylim(ax_temp.get_ylim())
    ax_cartesian.set_xlabel(ax_temp.get_xlabel(), fontsize=12)
    ax_cartesian.set_ylabel(ax_temp.get_ylabel(), fontsize=12)
    ax_cartesian.set_title('Cartesian Representation', fontsize=14, fontweight='bold', pad=20)
    ax_cartesian.set_xticks([0, 45, 90, 135, 180])
    ax_cartesian.grid(True, alpha=0.3, linestyle='--')
    
    # Add OSI text
    if not np.isnan(osi_info['osi']):
        ax_cartesian.text(0.02, 0.98, f'OSI = {osi_info["osi"]:.3f}', 
                         transform=ax_cartesian.transAxes, fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                         verticalalignment='top')
    
    # Clean up spines
    ax_cartesian.spines['top'].set_visible(False)
    ax_cartesian.spines['right'].set_visible(False)
    
    plt.close(fig_temp)  # Close temporary figure
    plt.tight_layout()
    
    return fig, [ax_polar, ax_cartesian], osi_info


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


def plot_orientation_tuning_cartesian_phases(responses, directions_deg, phase_colors=None,
                                           figsize=(8, 6), linewidth=2, marker='o', 
                                           markersize=6, show_osi=True,
                                           title=None, xlabel='Orientation (degrees)', 
                                           ylabel='Response', show_grid=True):
    """
    Plot orientation tuning curves for multiple phases in cartesian coordinates.
    
    Parameters:
    -----------
    responses : np.ndarray
        Array of shape (n_directions, n_phases) containing response values
    directions_deg : array-like
        Direction values in degrees
    phase_colors : list or None
        Colors for each phase. If None, uses default colors
    figsize : tuple
        Figure size (width, height)
    linewidth : float
        Line width for the tuning curves
    marker : str
        Marker style for data points
    markersize : float
        Size of data point markers
    show_osi : bool
        Whether to show OSI calculation points
    title : str or None
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    show_grid : bool
        Whether to show grid lines
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    osi_info : dict
        Dictionary containing OSI calculation results for each phase
    """
    from ..tuning_metrics import compute_orientation_tuning, compute_orientation_selectivity_index
    
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
    n_phases = responses.shape[1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Store OSI info for each phase
    osi_info = {}
    
    # Plot each phase
    for phase_i in range(n_phases):
        phase_responses = responses[:, phase_i]
        phase_color = phase_colors[phase_i % len(phase_colors)]
        
        # Get orientation tuning for this phase
        orientation_data = compute_orientation_tuning(phase_responses, directions_deg)
        orientations = orientation_data['orientations']
        orientation_responses = orientation_data['responses']
        
        # Get OSI information for this phase
        phase_osi_info = compute_orientation_selectivity_index(phase_responses, directions_deg)
        osi_info[f'phase_{phase_i}'] = phase_osi_info
        
        # Sort orientations for smooth plotting
        sort_indices = np.argsort(orientations)
        sorted_orientations = orientations[sort_indices]
        sorted_responses = orientation_responses[sort_indices]
        
        # Plot tuning curve for this phase
        linestyle = ['-', '--', '-.', ':'][phase_i % 4]
        ax.plot(sorted_orientations, sorted_responses, color=phase_color, 
                linewidth=linewidth, marker=marker, markersize=markersize, 
                linestyle=linestyle, label=f'Phase {phase_i+1}')
        
        # Show OSI calculation points if requested
        if show_osi and not np.isnan(phase_osi_info['osi']):
            # Mark preferred orientation
            ax.scatter(phase_osi_info['preferred_orientation'], phase_osi_info['preferred_response'], 
                      color=phase_color, s=markersize*12, marker='s', 
                      zorder=10, edgecolors='white', linewidth=1, alpha=0.8)
            
            # Mark orthogonal orientation
            ax.scatter(phase_osi_info['orthogonal_orientation'], phase_osi_info['orthogonal_response'], 
                      color=phase_color, s=markersize*12, marker='^', 
                      zorder=10, edgecolors='white', linewidth=1, alpha=0.8)
    
    # Add OSI text for each phase
    if show_osi:
        osi_text = []
        for phase_i in range(n_phases):
            phase_osi = osi_info[f'phase_{phase_i}']
            if not np.isnan(phase_osi['osi']):
                osi_text.append(f'Phase {phase_i+1}: OSI = {phase_osi["osi"]:.3f}')
        
        if osi_text:
            ax.text(0.02, 0.98, '\n'.join(osi_text), 
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    verticalalignment='top')
    
    # Styling
    ax.set_xlim(0, 180)
    max_response = np.max(responses) if np.max(responses) > 0 else 1
    ax.set_ylim(0, max_response * 1.1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-axis ticks at common orientations
    ax.set_xticks([0, 45, 90, 135, 180])
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Orientation Tuning - Phase Comparison', fontsize=14, fontweight='bold', pad=20)
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout()
    
    return fig, ax, osi_info


def plot_tuning_function_with_traces(osds_obj, roi_index, ax=None, show_trials=True, 
                                    metric='peak', trace_scale=0.2, minimal=True, 
                                    polar_color='#2E8B57', trace_alpha=1, use_phases=None, 
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
        trial_data = osds_obj.split_snippets_directionally()[:, roi_index, :, :]
    
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
                    trace_ax.axvline(sep_x, color='k', alpha=0.8, 
                                    linestyle='-', zorder = -1)
            
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


def plot_orientation_tuning_comparison_phases(responses, directions_deg, phase_colors=None,
                                            figsize=(15, 5)):
    """
    Plot side-by-side comparison of polar and cartesian orientation tuning for multiple phases.
    
    Parameters:
    -----------
    responses : np.ndarray
        Array of shape (n_directions, n_phases) containing response values
    directions_deg : array-like
        Direction values in degrees
    phase_colors : list or None
        Colors for each phase. If None, uses default colors
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : list
        List containing [polar_ax, cartesian_ax]
    osi_info : dict
        Dictionary containing OSI calculation results for each phase
    """
    from ..tuning_metrics import compute_orientation_tuning, extract_orientation_vector
    
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
    n_phases = responses.shape[1]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Polar plot (left)
    ax_polar = fig.add_subplot(121, projection='polar')
    
    # Plot each phase on polar plot
    for phase_i in range(n_phases):
        phase_responses = responses[:, phase_i]
        phase_color = phase_colors[phase_i % len(phase_colors)]
        
        # Get orientation data
        orientation_data = compute_orientation_tuning(phase_responses, directions_deg)
        orientations = orientation_data['orientations']
        orientation_responses = orientation_data['responses']
        
        # Convert to radians and double for polar plot
        orientations_rad = np.deg2rad(orientations * 2)  # Double for 0-360° display
        
        # Sort for smooth plotting
        sort_indices = np.argsort(orientations)
        sorted_orientations_rad = orientations_rad[sort_indices]
        sorted_responses = orientation_responses[sort_indices]
        
        # Close the polar plot
        sorted_orientations_rad = np.append(sorted_orientations_rad, sorted_orientations_rad[0])
        sorted_responses = np.append(sorted_responses, sorted_responses[0])
        
        # Plot polar tuning curve
        linestyle = ['-', '--', '-.', ':'][phase_i % 4]
        ax_polar.plot(sorted_orientations_rad, sorted_responses, color=phase_color, 
                      linewidth=2, marker='o', markersize=3, linestyle=linestyle,
                      label=f'Phase {phase_i+1}')
        ax_polar.fill(sorted_orientations_rad, sorted_responses, alpha=0.2, color=phase_color)
        
        # Add orientation vector
        orient_vector = extract_orientation_vector(phase_responses, directions_deg)
        if not np.isnan(orient_vector['angle']):
            vector_angle_rad = np.deg2rad(orient_vector['angle'] * 2)
            vector_magnitude = orient_vector['magnitude'] * max(sorted_responses)
            ax_polar.annotate('', xy=(vector_angle_rad, vector_magnitude), xytext=(0, 0),
                             arrowprops=dict(arrowstyle='->', color=phase_color, lw=2),
                             zorder=10)
    
    # Polar plot styling
    ax_polar.set_ylim(0, np.max(responses) * 1.1)
    ax_polar.set_title('Polar Representation', fontsize=14, fontweight='bold', pad=20)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # Cartesian plot (right)
    ax_cartesian = fig.add_subplot(122)
    fig_temp, ax_temp, osi_info = plot_orientation_tuning_cartesian_phases(
        responses, directions_deg, phase_colors=phase_colors, figsize=(8, 5), show_osi=True
    )
    
    # Copy cartesian plot to our subplot
    for line in ax_temp.get_lines():
        ax_cartesian.plot(line.get_xdata(), line.get_ydata(), 
                         color=line.get_color(), linewidth=line.get_linewidth(),
                         marker=line.get_marker(), markersize=line.get_markersize(),
                         linestyle=line.get_linestyle(), label=line.get_label())
    
    for collection in ax_temp.collections:
        if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
            ax_cartesian.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
                               c=collection.get_facecolors(), s=collection.get_sizes(),
                               marker='s' if hasattr(collection, '_marker') else 'o',
                               edgecolors='white', linewidth=1, alpha=0.8)
    
    # Copy cartesian styling
    ax_cartesian.set_xlim(ax_temp.get_xlim())
    ax_cartesian.set_ylim(ax_temp.get_ylim())
    ax_cartesian.set_xlabel(ax_temp.get_xlabel(), fontsize=12)
    ax_cartesian.set_ylabel(ax_temp.get_ylabel(), fontsize=12)
    ax_cartesian.set_title('Cartesian Representation', fontsize=14, fontweight='bold', pad=20)
    ax_cartesian.set_xticks([0, 45, 90, 135, 180])
    ax_cartesian.grid(True, alpha=0.3, linestyle='--')
    ax_cartesian.legend(loc='upper right', fontsize=10)
    
    # Add OSI text for each phase
    osi_text = []
    for phase_i in range(n_phases):
        phase_osi = osi_info[f'phase_{phase_i}']
        if not np.isnan(phase_osi['osi']):
            osi_text.append(f'Phase {phase_i+1}: OSI = {phase_osi["osi"]:.3f}')
    
    if osi_text:
        ax_cartesian.text(0.02, 0.98, '\n'.join(osi_text), 
                         transform=ax_cartesian.transAxes, fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                         verticalalignment='top')
    
    # Clean up spines
    ax_cartesian.spines['top'].set_visible(False)
    ax_cartesian.spines['right'].set_visible(False)
    
    plt.close(fig_temp)  # Close temporary figure
    plt.tight_layout()
    
    return fig, [ax_polar, ax_cartesian], osi_info
