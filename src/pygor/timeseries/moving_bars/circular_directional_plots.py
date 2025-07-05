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
    moving_bars_obj,
    directions_list=None,
    figsize=(10, 10),
    metric="peak",
    polar_kwargs=None,
    polar_size=0.3,
    roi_index=-1,
    show_trials=True,
    data_crop=None,
):
    """
    Plot directional responses in a circular arrangement with central polar plot.

    Parameters:
    -----------
    data : np.ndarray or None
            Array of shape (n_directions, n_timepoints) containing response traces.
            If moving_bars_obj is provided, this can be None.
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
    moving_bars_obj : MovingBars object, optional
            If provided, will extract data and optionally show individual trials
    roi_index : int, optional
            ROI index to plot (default -1 for last ROI)
    show_trials : bool, optional
            Whether to show individual trial traces as faint lines (default False)

    Returns:
    --------
    fig : matplotlib.figure.Figure
            The figure object
    ax_polar : matplotlib.axes.Axes
            The polar plot axes object
    """

    # Extract data from MovingBars object if provided
    trial_data = None  # Initialize here so it's available throughout the function

    # if moving_bars_obj is not None:
    data = np.squeeze(moving_bars_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = moving_bars_obj.directions_list

    # Get trial data if requested
    if show_trials:
        # Shape: (n_directions, n_rois, n_trials, n_timepoints) -> (n_directions, n_trials, n_timepoints)
        trial_data = moving_bars_obj.split_snippets_directionally()[:, roi_index, :, :]

    if data_crop is not None:
        print(data.shape)
        data = data[:, data_crop[0] : data_crop[1]]
        if trial_data is not None:
            trial_data = trial_data[:, :, data_crop[0] : data_crop[1]]

    # # Handle the case where data is None but moving_bars_obj is provided
    # data = np.squeeze(moving_bars_obj.split_averages_directionally()[:, [roi_index]])

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
    moving_bars_obj,
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
    moving_bars_obj : MovingBars object
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
    data = np.squeeze(moving_bars_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = moving_bars_obj.directions_list

    # Get trial data if requested
    trial_data = None
    if show_trials:
        trial_data = moving_bars_obj.split_snippets_directionally()[:, roi_index, :, :]
    if phase_split is None:
        phase_split = phase_split = moving_bars_obj.averages.shape[1] // moving_bars_obj.trigger_mode // 2  # Should be ~2866
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
    ax_polar.set_theta_zero_location("E")  # 0° at right (90° north)
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
        x_center = 0.5 + radius * np.cos(-angle + np.pi / 2)
        y_center = 0.5 + radius * np.sin(-angle + np.pi / 2)
        
        subplot_size = 0.08
        ax = fig.add_axes([
            x_center - subplot_size / 2,
            y_center - subplot_size / 2,
            subplot_size,
            subplot_size
        ])
        
        # Plot both phases on same axes with different colors
        if show_trials and trial_data is not None:
            ax.plot(sorted_phase1_trial_data[i].T, color=phase_colors[0], 
                    linewidth=0.3, alpha=0.3)
            ax.plot(sorted_phase2_trial_data[i].T, color=phase_colors[1], 
                    linewidth=0.3, alpha=0.3)
        
        # Plot average traces for both phases
        ax.plot(sorted_phase1_data[i], color=phase_colors[0], linewidth=2, label='OFF→ON')
        ax.plot(sorted_phase2_data[i], color=phase_colors[1], linewidth=2, label='ON→OFF')
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(0, max(len(sorted_phase1_data[i]), len(sorted_phase2_data[i])))
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


def plot_tuning_function_polar(tuning_functions, directions_list, rois=None, figsize=(6, 6), colors=None, metric='peak', ax=None, show_title=True, show_theta_labels=True):
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
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The polar plot axes object
    """
    if rois is None:
        rois = list(range(tuning_functions.shape[1]))
    elif isinstance(rois, int):
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
    
    # Set up colors
    if colors is None:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    
    for i, roi_idx in enumerate(rois):
        tuning_function = tuning_functions[sort_order, roi_idx]
        
        # Close the loop
        tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
        degrees_closed = np.concatenate((degrees, [degrees[0]]))
        
        ax.plot(degrees_closed, tuning_function_closed, marker='o', 
                color=colors[i], label=f'ROI {roi_idx}')
    
    if len(rois) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Format metric name for title
    if show_title:
        if callable(metric):
            metric_name = getattr(metric, '__name__', 'custom function')
        else:
            metric_name = metric.replace('_', ' ').title()
        
        ax.set_title(f'Directional tuning ({metric_name})', fontsize=12, pad=20)
    
    ax.grid(True, alpha=0.3)
    
    # Hide theta labels if requested
    if not show_theta_labels:
        ax.set_thetagrids([])
    
    return fig, ax
