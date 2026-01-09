import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Helper functions for plot_tuning_function_with_traces
# ============================================================================

def _calculate_metric(data, metric='peak'):
    """
    Calculate summary metric for traces.

    Parameters
    ----------
    data : ndarray
        Array of traces (directions × timepoints)
    metric : str or callable
        'peak', 'auc', 'mean', or custom function

    Returns
    -------
    values : ndarray
        Metric value for each trace
    """
    if metric == 'peak':
        return np.array([np.max(np.abs(trace)) for trace in data])
    elif metric == 'auc':
        return np.array([np.trapz(np.abs(trace)) for trace in data])
    elif metric == 'mean':
        return np.array([np.mean(trace) for trace in data])
    elif callable(metric):
        return np.array([metric(trace) for trace in data])
    else:
        return np.array([np.max(np.abs(trace)) for trace in data])


def _prepare_tuning_data(osds_obj, roi_index, use_phases=None, show_trials=True, data_crop=None):
    """
    Extract and sort directional tuning data from OSDS object.

    Parameters
    ----------
    osds_obj : OSDS object
        Object containing directional response data
    roi_index : int
        ROI index to analyze
    use_phases : bool or None
        Whether to split data by phases. If None, auto-detect from osds_obj
    show_trials : bool
        Whether to include trial data
    data_crop : tuple or None
        (start, end) indices for cropping data

    Returns
    -------
    dict with keys:
        'data': sorted average traces (directions × timepoints)
        'trial_data': sorted trial traces or None
        'angles': sorted angles in radians
        'directions_deg': sorted directions in degrees
        'sort_indices': indices used for sorting
        'phases_data': list of phase-separated average data (if multi-phase)
        'phases_trial_data': list of phase-separated trial data (if multi-phase)
        'use_phases': resolved boolean indicating if phases are used
        'dir_phase_num': number of phases
    """
    # Auto-detect phases
    if use_phases is None:
        use_phases = osds_obj.dir_phase_num > 1

    # Extract data
    data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
    directions_list = osds_obj.directions_list

    # Get trial data if requested
    trial_data = None
    if show_trials:
        trial_data = osds_obj.split_snippets_directionally()[:, :, roi_index, :]

    # Handle data cropping
    if data_crop is not None:
        data = data[:, data_crop[0]:data_crop[1]]
        if trial_data is not None:
            trial_data = trial_data[:, :, data_crop[0]:data_crop[1]]

    # Handle phase splitting for multi-phase data
    phases_data = None
    phases_trial_data = None

    if use_phases and osds_obj.dir_phase_num > 1:
        phase_split = data.shape[1] // osds_obj.dir_phase_num
        phases_data = []
        phases_trial_data = []

        for phase_i in range(osds_obj.dir_phase_num):
            start_idx = phase_i * phase_split
            end_idx = (phase_i + 1) * phase_split if phase_i < osds_obj.dir_phase_num - 1 else data.shape[1]
            phases_data.append(data[:, start_idx:end_idx])
            if trial_data is not None:
                phases_trial_data.append(trial_data[:, :, start_idx:end_idx])

        # Concatenate phases for trace display
        data = np.concatenate(phases_data, axis=1)
        if trial_data is not None:
            trial_data = np.concatenate(phases_trial_data, axis=2)

    # Sort by direction
    angles = np.radians(directions_list)
    directions_deg = np.array(directions_list)
    sort_indices = np.argsort(directions_deg)

    sorted_angles = angles[sort_indices]
    sorted_data = data[sort_indices]
    sorted_directions_deg = directions_deg[sort_indices]
    sorted_trial_data = trial_data[sort_indices] if trial_data is not None else None

    return {
        'data': sorted_data,
        'trial_data': sorted_trial_data,
        'angles': sorted_angles,
        'directions_deg': sorted_directions_deg,
        'sort_indices': sort_indices,
        'phases_data': phases_data,
        'phases_trial_data': phases_trial_data,
        'use_phases': use_phases,
        'dir_phase_num': osds_obj.dir_phase_num
    }


def _setup_figure_and_axes(ax, num_subplots=1):
    """
    Handle figure and axes creation/validation.

    Parameters
    ----------
    ax : Axes, array/list of Axes, or None
        Provided axes or None for automatic creation
    num_subplots : int
        Number of subplots needed

    Returns
    -------
    fig : Figure
    axes : list of Axes
        Always returns a list for consistency
    """
    if ax is None:
        fig, ax = plt.subplots(1, num_subplots, figsize=(6*num_subplots, 5))
        if num_subplots > 1:
            plt.subplots_adjust(wspace=0.8)
        axes = [ax] if num_subplots == 1 else list(ax)
        return fig, axes

    # Handle provided axes
    if hasattr(ax, '__len__') and hasattr(ax, '__getitem__'):
        if len(ax) != num_subplots:
            raise ValueError(f"Expected {num_subplots} axes, got {len(ax)}")
        return ax[0].figure, list(ax)
    else:
        # Single axes provided
        if num_subplots != 1:
            raise ValueError(f"Expected {num_subplots} axes, got 1")
        return ax.figure, [ax]


def _calculate_orbit_positions(fig, ax_polar, angles, orbit_distance=0.5,
                               trace_scale=0.2, trace_aspect_x=1.0, trace_aspect_y=1.0):
    """
    Calculate positions for orbit trace subplots around polar plot.

    Parameters
    ----------
    fig : Figure
    ax_polar : Axes
        The polar plot axes
    angles : ndarray
        Array of angles in radians
    orbit_distance : float
        Distance of trace orbit from polar plot center
    trace_scale : float
        Scale factor for trace subplot size
    trace_aspect_x : float
        Horizontal scaling factor for trace plots
    trace_aspect_y : float
        Vertical scaling factor for trace plots

    Returns
    -------
    positions : list of tuples
        Each tuple is (x_center, y_center, trace_width, trace_height)
    """
    fig.canvas.draw_idle()
    axes_bbox = ax_polar.get_position()
    axes_center_x = axes_bbox.x0 + axes_bbox.width / 2
    axes_center_y = axes_bbox.y0 + axes_bbox.height / 2

    fig_size = fig.get_size_inches()
    fig_aspect = fig_size[0] / fig_size[1]

    # Aspect ratio correction for circular orbit
    if fig_aspect > 1:
        orbit_radius_x = orbit_distance / fig_aspect
        orbit_radius_y = orbit_distance
    else:
        orbit_radius_x = orbit_distance
        orbit_radius_y = orbit_distance * fig_aspect

    base_trace_size = trace_scale * min(axes_bbox.width, axes_bbox.height)
    trace_width = base_trace_size * trace_aspect_x
    trace_height = base_trace_size * trace_aspect_y

    positions = []
    for angle in angles:
        x_center = axes_center_x + orbit_radius_x * np.cos(angle)
        y_center = axes_center_y + orbit_radius_y * np.sin(angle)
        positions.append((x_center, y_center, trace_width, trace_height))

    return positions


def _calculate_global_limits(data_list, trial_data_list=None, minimal=True):
    """
    Calculate consistent y-limits across all data.

    Parameters
    ----------
    data_list : list of ndarrays
        List of data arrays (can be single-element list)
    trial_data_list : list of ndarrays or None
        List of trial data arrays
    minimal : bool
        Whether to use minimal buffering

    Returns
    -------
    y_min, y_max : float
        Global y-limits
    """
    if trial_data_list is not None and any(td is not None for td in trial_data_list):
        all_data = np.concatenate([td.flatten() for td in trial_data_list if td is not None])
    else:
        all_data = np.concatenate([d.flatten() for d in data_list])

    y_min = np.min(all_data)
    y_max = np.max(all_data)
    y_range = y_max - y_min
    y_buffer = y_range * 0.05 if minimal else y_range * 0.1

    return y_min - y_buffer, y_max + y_buffer


def _style_polar_plot(ax_polar, angles, values, color, minimal=True, global_max=None):
    """
    Apply consistent polar plot styling.

    Parameters
    ----------
    ax_polar : PolarAxes
        The polar axes to style
    angles : ndarray
        Array of angles in radians
    values : ndarray
        Values to plot at each angle
    color : str
        Color for the plot
    minimal : bool
        Whether to use minimal styling
    global_max : float or None
        Maximum value for consistent scaling across multiple plots
    """
    polar_angles = np.append(angles, angles[0])
    polar_values = np.append(values, values[0])

    ax_polar.plot(polar_angles, polar_values, color=color, alpha=1)
    ax_polar.fill(polar_angles, polar_values, alpha=0.33, color=color)

    max_val = global_max if global_max is not None else np.max(values)

    if minimal:
        ax_polar.grid(True, alpha=0.3)
        ax_polar.set_thetagrids([])

        max_tick = int(np.ceil(max_val))

        # Smart tick calculation
        if max_tick <= 3:
            ticks = [1, 2, max_tick] if max_tick > 2 else [1, max_tick]
        else:
            step = max(1, max_tick // 3)
            ticks = [step, 2*step, 3*step]
            if ticks[-1] > max_tick:
                ticks[-1] = max_tick

        ticks = sorted(list(set(ticks)))[:3]
        ax_polar.set_rgrids(ticks, alpha=0.7)
        ax_polar.set_ylim(0, max_val * 1.1)
    else:
        ax_polar.grid(True, alpha=0.3)
        ax_polar.set_ylim(0, max_val * 1.1)


def _add_orbit_trace(fig, position, trace_data, trial_data, color, y_limits,
                     show_trials=True, trace_alpha=1.0):
    """
    Create a single orbit trace subplot.

    Parameters
    ----------
    fig : Figure
    position : tuple
        (x_center, y_center, trace_width, trace_height)
    trace_data : ndarray
        Average trace to plot
    trial_data : ndarray or None
        Individual trial traces
    color : str
        Color for the trace
    y_limits : tuple
        (y_min, y_max)
    show_trials : bool
        Whether to show individual trials
    trace_alpha : float
        Alpha for average trace

    Returns
    -------
    trace_ax : Axes
        The created trace axes
    """
    x_center, y_center, trace_width, trace_height = position

    trace_ax = fig.add_axes([
        x_center - trace_width/2,
        y_center - trace_height/2,
        trace_width,
        trace_height
    ])

    if show_trials and trial_data is not None:
        trace_ax.plot(trial_data.T, color=color, alpha=0.1)

    trace_ax.plot(trace_data, color=color, alpha=trace_alpha)
    trace_ax.set_facecolor((1, 1, 1, 0))
    trace_ax.axhline(0, color='gray', alpha=0.4)

    trace_ax.set_xlim(0, len(trace_data))
    trace_ax.set_ylim(y_limits[0], y_limits[1])
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])

    for spine in trace_ax.spines.values():
        spine.set_visible(False)

    return trace_ax


# ============================================================================
# Main plotting function
# ============================================================================

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
    # Set default phase colors
    if phase_colors is None:
        phase_colors = ["#000000", "#4B4B4B", '#8B4513', '#483D8B']

    # Prepare all data using helper function
    data_crop = kwargs.get('data_crop', None)
    data_dict = _prepare_tuning_data(osds_obj, roi_index, use_phases, show_trials, data_crop)

    # Unpack prepared data
    sorted_data = data_dict['data']
    sorted_trial_data = data_dict['trial_data']
    sorted_angles = data_dict['angles']
    sorted_directions_deg = data_dict['directions_deg']
    sort_indices = data_dict['sort_indices']
    individual_phases_data = data_dict['phases_data']
    phases_trial_data = data_dict['phases_trial_data']
    use_phases = data_dict['use_phases']
    dir_phase_num = data_dict['dir_phase_num']
    
    # Calculate metric values for polar plot (handle phases)
    if use_phases and dir_phase_num > 1 and individual_phases_data is not None:
        # Calculate metrics for each phase separately using helper function
        phase_values = []
        for phase_i in range(dir_phase_num):
            phase_data = individual_phases_data[phase_i][sort_indices]
            phase_vals = _calculate_metric(phase_data, metric)
            phase_values.append(phase_vals)
    else:
        # Single phase calculation
        values = _calculate_metric(sorted_data, metric)
    
    # Branch into separate mode vs overlay mode
    if separate_phase_axes and use_phases and osds_obj.dir_phase_num > 1 and individual_phases_data is not None:
        # SEPARATE AXES MODE: Each phase gets its own polar plot + orbit traces
        # print(f"DEBUG: Using separate axes mode for {osds_obj.dir_phase_num} phases")
        
        # Handle axes creation/validation using helper function
        fig, ax = _setup_figure_and_axes(ax, dir_phase_num)

        # Adjust spacing to prevent orbit overlap if axes were provided
        if dir_phase_num > 1 and ax is not None:
            current_positions = [a.get_position() for a in ax]
            required_spacing = 2 * orbit_distance * trace_scale
            if len(current_positions) >= 2:
                current_spacing = current_positions[1].x0 - (current_positions[0].x0 + current_positions[0].width)
                if current_spacing < required_spacing:
                    optimal_wspace = max(0.4, required_spacing * 2)
                    fig.subplots_adjust(wspace=optimal_wspace)

        ax_polars = []
        all_trace_axes = []
        
        # Calculate global limits using helper functions
        global_metric_max = max([np.max(phase_vals) for phase_vals in phase_values])

        # Calculate global y-limits for trace data
        if sorted_trial_data is not None:
            trial_data_list = [phases_trial_data[i][sort_indices] for i in range(dir_phase_num)]
            y_min_global, y_max_global = _calculate_global_limits(
                [individual_phases_data[i] for i in range(dir_phase_num)],
                trial_data_list,
                minimal
            )
        else:
            y_min_global, y_max_global = _calculate_global_limits(
                [individual_phases_data[i] for i in range(dir_phase_num)],
                None,
                minimal
            )
        
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
            
            # Plot and style polar for this phase using helper function
            phase_vals = phase_values[phase_i]
            _style_polar_plot(ax_polar, sorted_angles, phase_vals, phase_color, minimal, global_metric_max)
            
            # Add orbit traces for this phase using helper functions
            phase_data = individual_phases_data[phase_i][sort_indices]
            phase_trial_data = None
            if show_trials and sorted_trial_data is not None:
                phase_trial_data = phases_trial_data[phase_i][sort_indices]

            # Calculate orbit positions
            positions = _calculate_orbit_positions(
                fig, ax_polar, sorted_angles, orbit_distance,
                trace_scale, trace_aspect_x, trace_aspect_y
            )

            # Create orbit traces
            phase_trace_axes = []
            for i, position in enumerate(positions):
                trace_ax = _add_orbit_trace(
                    fig, position, phase_data[i],
                    phase_trial_data[i] if phase_trial_data is not None else None,
                    phase_color, (y_min_global, y_max_global),
                    show_trials, trace_alpha
                )
                phase_trace_axes.append(trace_ax)
            
            all_trace_axes.append(phase_trace_axes)
        
        
        return fig, ax_polars
        
    else:
        # OVERLAY MODE: Single polar plot with overlaid phases (original behavior)

        # Handle axes creation/validation using helper function
        fig, axes_list = _setup_figure_and_axes(ax, 1)
        ax = axes_list[0]

        # Convert to polar using subplot positioning instead of absolute coordinates
        ax_gridspec = ax.get_subplotspec()
        ax.remove()
        ax_polar = fig.add_subplot(ax_gridspec, projection='polar')
        ax_polar.set_theta_zero_location('E')
        
        # Plot central polar (handle phases)
        if use_phases and dir_phase_num > 1 and 'phase_values' in locals():
            # Multi-phase: plot each phase with different style
            polar_angles = np.append(sorted_angles, sorted_angles[0])
            for phase_i, phase_vals in enumerate(phase_values):
                phase_polar_values = np.append(phase_vals, phase_vals[0])
                phase_color = phase_colors[phase_i % len(phase_colors)]
                linestyle = ['-', '--', '-.', ':'][phase_i % 4]
                ax_polar.plot(polar_angles, phase_polar_values, color=phase_color,
                             alpha=0.9, linestyle=linestyle)
                ax_polar.fill(polar_angles, phase_polar_values, alpha=0.2, color=phase_color)
            global_metric_max = max([np.max(phase_vals) for phase_vals in phase_values])
        else:
            # Single phase: use helper function for styling
            _style_polar_plot(ax_polar, sorted_angles, values, polar_color, minimal, None)
            global_metric_max = np.max(values)

        # Apply styling for multi-phase plots (single-phase already styled above)
        if use_phases and dir_phase_num > 1:
            if minimal:
                ax_polar.grid(True, alpha=0.3)
                ax_polar.set_thetagrids([])
                max_tick = int(np.ceil(global_metric_max))
                if max_tick <= 3:
                    ticks = [1, 2, max_tick] if max_tick > 2 else [1, max_tick]
                else:
                    step = max(1, max_tick // 3)
                    ticks = [step, 2*step, 3*step]
                    if ticks[-1] > max_tick:
                        ticks[-1] = max_tick
                ticks = sorted(list(set(ticks)))[:3]
                ax_polar.set_rgrids(ticks, alpha=0.7)
                ax_polar.set_ylim(0, global_metric_max * 1.1)
            else:
                ax_polar.grid(True, alpha=0.3)
                ax_polar.set_ylim(0, global_metric_max * 1.1)

        # Calculate global y-limits for trace data using helper function
        y_min_global, y_max_global = _calculate_global_limits(
            [sorted_data],
            [sorted_trial_data] if sorted_trial_data is not None else None,
            minimal
        )

        # Add floating trace snippets around perimeter using helper functions
        positions = _calculate_orbit_positions(
            fig, ax_polar, sorted_angles, orbit_distance,
            trace_scale, trace_aspect_x, trace_aspect_y
        )

        trace_axes = []
        for i, position in enumerate(positions):
            trace_ax = _add_orbit_trace(
                fig, position, sorted_data[i],
                sorted_trial_data[i] if sorted_trial_data is not None else None,
                'k', (y_min_global, y_max_global),
                show_trials, trace_alpha
            )

            # Add phase separation shading if multi-phase
            if use_phases and dir_phase_num > 1 and individual_phases_data is not None:
                phase_split = len(individual_phases_data[0][0])
                for phase_i in range(1, dir_phase_num):
                    if phase_i % 2 == 1:
                        trace_ax.axvspan((phase_i - 1) * phase_split,
                                        phase_i * phase_split,
                                        facecolor='gray', alpha=0.5, zorder=-1, edgecolor='none')

            trace_axes.append(trace_ax)
        
        
        return fig, ax_polar
    




# def calculate_directional_selectivity_index(values):
#     """
#     Calculate directional selectivity index (DSI).
#     DSI = (preferred - null) / (preferred + null)

#     Parameters:
#     -----------
#     values : array-like
#             Response values for each direction

#     Returns:
#     --------
#     float : DSI value between 0 (non-selective) and 1 (highly selective)
#     """
#     preferred = np.max(values)
#     # Find null direction (opposite to preferred)
#     preferred_idx = np.argmax(values)
#     n_dirs = len(values)
#     null_idx = (preferred_idx + n_dirs // 2) % n_dirs
#     null = values[null_idx]

#     if preferred + null == 0:
#         return 0
#     return (preferred - null) / (preferred + null)


# def calculate_orientation_selectivity_index(values, directions_list):
#     """
#     Calculate orientation selectivity index by grouping opposite directions.

#     Parameters:
#     -----------
#     values : array-like
#             Response values for each direction
#     directions_list : array-like
#             Direction values in degrees

#     Returns:
#     --------
#     float : OSI value
#     """
#     directions = np.array(directions_list)

#     # Group opposite directions and sum their responses
#     orientation_responses = []
#     used_indices = set()

#     for i, direction in enumerate(directions):
#         if i in used_indices:
#             continue

#         # Find opposite direction (±180°)
#         opposite_dir = (direction + 180) % 360
#         opposite_idx = None

#         for j, other_dir in enumerate(directions):
#             if abs(other_dir - opposite_dir) < 10:  # Allow some tolerance
#                 opposite_idx = j
#                 break

#         if opposite_idx is not None:
#             combined_response = values[i] + values[opposite_idx]
#             orientation_responses.append(combined_response)
#             used_indices.update([i, opposite_idx])
#         else:
#             orientation_responses.append(values[i])
#             used_indices.add(i)

#     # Calculate OSI
#     orientation_responses = np.array(orientation_responses)
#     if len(orientation_responses) < 2:
#         return 0

#     preferred = np.max(orientation_responses)
#     orthogonal_idx = (
#         np.argmax(orientation_responses) + len(orientation_responses) // 2
#     ) % len(orientation_responses)
#     orthogonal = orientation_responses[orthogonal_idx]

#     if preferred + orthogonal == 0:
#         return 0
#     return (preferred - orthogonal) / (preferred + orthogonal)


# def plot_tuning_function_polar(tuning_functions, directions_list, rois=None, figsize=(6, 6), colors=None, metric='peak', ax=None, show_title=True, show_theta_labels=True, show_tuning=True, show_mean_vector=False, mean_vector_color='red', show_orientation_vector=False, orientation_vector_color='orange', legend=True, minimal=False):
#     """
#     Plot tuning functions as polar plots.
    
#     Parameters:
#     -----------
#     tuning_functions : np.ndarray
#         Array of shape (n_directions, n_rois) containing tuning functions
#     directions_list : list of int/float
#         List of direction values in degrees
#     rois : list of int or None
#         ROI indices to plot. If None, plots all ROIs
#     figsize : tuple
#         Figure size (width, height)
#     colors : list or None
#         Colors for each ROI. If None, uses default color cycle
#     metric : str or callable
#         Metric used to compute tuning function (for title display)
#     ax : matplotlib.axes.Axes or None
#         Existing polar axes to plot on. If None, creates new figure and axes
#     show_title : bool
#         Whether to show the title on the plot (default True)
#     show_theta_labels : bool
#         Whether to show the theta (direction) labels on the plot (default True)
#     show_tuning : bool
#         Whether to show the tuning curve itself (default True). When False, only shows vectors.
#     show_mean_vector : bool
#         Whether to show mean direction vectors as overlays (default False)
#     mean_vector_color : str
#         Color for mean direction vector arrows (default 'red')
#     show_orientation_vector : bool
#         Whether to show mean orientation vectors as overlays (default False)
#     orientation_vector_color : str
#         Color for mean orientation vector arrows (default 'orange')
#     legend : bool
#         Whether to show the legend (default True)
#     minimal : bool
#         Whether to use minimal plotting (no titles or legends) (default False)
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The figure object
#     ax : matplotlib.axes.Axes
#         The polar plot axes object
#     """
#     if rois is None:
#         rois = list(range(tuning_functions.shape[1]))
#     elif isinstance(rois, (int, np.integer)):
#         rois = [rois]
    
#     # Sort by direction for proper polar plot connectivity
#     sort_order = np.argsort(directions_list)
#     degrees = np.deg2rad(directions_list)[sort_order]
    
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax = plt.subplot(projection='polar')
#         # Set polar plot orientation to match other functions
#         ax.set_theta_zero_location("E")  # 0° at right (90° north)
#     else:
#         fig = ax.figure
#         # Check if provided axes is polar, if not replace with polar axes
#         if ax.name != 'polar':
#             # Get the position of the current axes
#             pos = ax.get_position()
#             # Remove the current axes
#             ax.remove()
#             # Create a new polar axes in the same position
#             ax = fig.add_subplot(111, projection='polar')
#             ax.set_position(pos)
#             ax.set_theta_zero_location("E")
    
#     # Set up colors
#     if colors is None:
#         colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    
#     # Plot tuning curves only if show_tuning is True
#     if show_tuning:
#         for i, roi_idx in enumerate(rois):
#             tuning_function = tuning_functions[sort_order, roi_idx]
            
#             # Close the loop
#             tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
#             degrees_closed = np.concatenate((degrees, [degrees[0]]))
            
#             ax.plot(degrees_closed, tuning_function_closed, marker='o', 
#                     color=colors[i], label=f'ROI {roi_idx}')
    
#     # Show legend if requested and not minimal and there are multiple ROIs/tuning/vectors shown
#     if legend and not minimal and ((len(rois) > 1 and show_tuning) or show_mean_vector or show_orientation_vector):
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
#     # Format metric name for title (only if not minimal)
#     if show_title and not minimal:
#         if callable(metric):
#             metric_name = getattr(metric, '__name__', 'custom function')
#         else:
#             metric_name = metric.replace('_', ' ').title()
        
#         ax.set_title(f'Directional tuning ({metric_name})', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
#     ax.grid(True, alpha=0.3)
    
#     # Hide theta labels if requested or in minimal mode
#     if not show_theta_labels or minimal:
#         ax.set_thetagrids([])
    
#     # Add mean direction vectors if requested
#     if show_mean_vector:
#         from pygor.timeseries.osds import tuning_metrics
        
#         for i, roi_idx in enumerate(rois):
#             # Get tuning function for this ROI
#             roi_tuning = tuning_functions[:, roi_idx][sort_order]
#             directions_sorted = np.array(directions_list)[sort_order]
            
#             # Compute mean direction vector
#             mean_vector = tuning_metrics.extract_mean_vector(roi_tuning, directions_sorted)
            
#             if not np.isnan(mean_vector['angle']):
#                 # Convert to radians
#                 mean_angle_rad = np.deg2rad(mean_vector['angle'])
                
#                 # Scale magnitude for visibility
#                 mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning)
                
#                 # Use the same color as the tuning function for this ROI
#                 arrow_color = colors[i] if isinstance(colors, (list, np.ndarray)) else mean_vector_color
                
#                 # Plot mean direction vector arrow with label for legend
#                 ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
#                            xytext=(0, 0),
#                            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3),
#                            zorder=10)
                
#                 # Add invisible line for legend entry
#                 ax.plot([], [], color=arrow_color, marker='>', markersize=8, 
#                        linestyle='None', label=f'ROI {roi_idx} Direction Vector')
    
#     # Add mean orientation vectors if requested
#     if show_orientation_vector:
#         from pygor.timeseries.osds import tuning_metrics
        
#         for i, roi_idx in enumerate(rois):
#             # Get tuning function for this ROI
#             roi_tuning = tuning_functions[:, roi_idx][sort_order]
#             directions_sorted = np.array(directions_list)[sort_order]
            
#             # Compute mean orientation vector
#             orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning, directions_sorted)
            
#             if not np.isnan(orientation_vector['angle']):
#                 # Convert to radians
#                 orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                
#                 # Scale magnitude for visibility
#                 orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning)
                
#                 # Use orientation vector color
#                 orient_arrow_color = orientation_vector_color
                
#                 # Plot mean orientation vector arrow
#                 ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
#                            xytext=(0, 0),
#                            arrowprops=dict(arrowstyle='->', color=orient_arrow_color, lw=3),
#                            zorder=9)
                
#                 # Add invisible line for legend entry
#                 ax.plot([], [], color=orient_arrow_color, marker='>', markersize=8, 
#                        linestyle='None', label=f'ROI {roi_idx} Orientation Vector')
    
#     return fig, ax


# def plot_tuning_function_multi_phase(tuning_functions, directions_list, phase_num, rois=None, 
#                                     figsize=(6, 6), colors=None, ax=None, show_title=True, 
#                                     show_theta_labels=True, show_tuning=True, show_mean_vector=False, 
#                                     mean_vector_color='red', show_orientation_vector=False, 
#                                     orientation_vector_color='orange', metric='peak', legend=True, minimal=False):
#     """
#     Plot multi-phase tuning functions as polar plots with phase-dependent vectors.
    
#     Parameters:
#     -----------
#     tuning_functions : np.ndarray
#         Array of shape (n_rois, n_directions, n_phases) containing tuning functions for each phase
#     directions_list : list of int/float
#         List of direction values in degrees
#     phase_num : int
#         Number of phases
#     rois : list of int or None
#         ROI indices to plot. If None, plots all ROIs
#     figsize : tuple
#         Figure size (width, height)
#     colors : list or None
#         Colors for each ROI. If None, uses default color cycle
#     ax : matplotlib.axes.Axes or None
#         Existing polar axes to plot on. If None, creates new figure and axes
#     show_title : bool
#         Whether to show the title on the plot (default True)
#     show_theta_labels : bool
#         Whether to show the theta (direction) labels on the plot (default True)
#     show_tuning : bool
#         Whether to show the tuning curve itself (default True). When False, only shows vectors.
#     show_mean_vector : bool
#         Whether to show mean direction vectors as overlays (default False)
#     mean_vector_color : str
#         Color for mean direction vector arrows (default 'red')
#     show_orientation_vector : bool
#         Whether to show mean orientation vectors as overlays (default False)
#     orientation_vector_color : str
#         Color for mean orientation vector arrows (default 'orange')
#     metric : str
#         Metric name for title display
#     legend : bool
#         Whether to show the legend (default True)
#     minimal : bool
#         Whether to use minimal plotting (no titles or legends) (default False)
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The figure object
#     ax : matplotlib.axes.Axes
#         The polar plot axes object
#     """
#     # Create figure if not provided
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax = plt.subplot(projection='polar')
#         ax.set_theta_zero_location("E")  # 0° at right (90° north)
#     else:
#         fig = ax.figure
#         # Check if provided axes is polar, if not replace with polar axes
#         if ax.name != 'polar':
#             # Get the position of the current axes
#             pos = ax.get_position()
#             # Remove the current axes
#             ax.remove()
#             # Create a new polar axes in the same position
#             ax = fig.add_subplot(111, projection='polar')
#             ax.set_position(pos)
#             ax.set_theta_zero_location("E")
    
#     # Set up ROI indices
#     if rois is None:
#         rois = list(range(tuning_functions.shape[0]))
#     elif isinstance(rois, (int, np.integer)):
#         rois = [rois]
    
#     # Set up colors for ROIs
#     if colors is None:
#         base_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
#     else:
#         base_colors = colors
    
#     # Sort by direction for proper polar plot connectivity
#     sort_order = np.argsort(directions_list)
#     degrees = np.deg2rad(directions_list)[sort_order]
    
#     # Plot each phase for each ROI (only if show_tuning is True)
#     if show_tuning:
#         for roi_i, roi_idx in enumerate(rois):
#             base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else base_colors
            
#             for phase_i in range(phase_num):
#                 # Get tuning function for this ROI and phase
#                 tuning_function = tuning_functions[roi_idx, sort_order, phase_i]
                
#                 # Close the loop
#                 tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
#                 degrees_closed = np.concatenate((degrees, [degrees[0]]))
                
#                 # Create phase-specific styling
#                 alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
#                 linestyle = ['-', '--', '-.', ':'][phase_i % 4]
                
#                 # Plot phase
#                 label = f'ROI {roi_idx} Phase {phase_i+1}' if len(rois) > 1 or phase_num > 1 else f'Phase {phase_i+1}'
#                 ax.plot(degrees_closed, tuning_function_closed, marker='o', 
#                         color=base_color, alpha=alpha, linestyle=linestyle, label=label)
    
#     # Add legend if requested and not minimal and there are multiple traces/tuning/vectors shown
#     if legend and not minimal and (((len(rois) > 1 or phase_num > 1) and show_tuning) or show_mean_vector or show_orientation_vector):
#         # Collect all handles and labels
#         handles, labels = ax.get_legend_handles_labels()
        
#         # Create manual legend if no automatic handles found
#         if not handles and (show_mean_vector or show_orientation_vector):
#             from matplotlib.lines import Line2D
#             manual_handles = []
#             manual_labels = []
            
#             if show_mean_vector:
#                 for roi_i, roi_idx in enumerate(rois):
#                     base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else mean_vector_color
#                     for phase_i in range(phase_num):
#                         # Create phase-specific color
#                         if isinstance(base_color, str):
#                             import matplotlib.colors as mcolors
#                             base_rgb = mcolors.to_rgb(base_color)
#                             phase_factor = 0.8 + 0.2 * (phase_i / max(1, phase_num - 1))
#                             phase_color = tuple(min(1.0, c * phase_factor) for c in base_rgb)
#                         else:
#                             phase_color = base_color
                        
#                         alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
#                         handle = Line2D([0], [0], color=phase_color, marker='>', markersize=8, 
#                                       alpha=alpha, linestyle='None')
#                         manual_handles.append(handle)
#                         manual_labels.append(f'ROI {roi_idx} Direction Vector P{phase_i+1}')
            
#             if show_orientation_vector:
#                 for roi_i, roi_idx in enumerate(rois):
#                     for phase_i in range(phase_num):
#                         # Create phase-specific color for orientation
#                         if isinstance(orientation_vector_color, str):
#                             import matplotlib.colors as mcolors
#                             base_rgb = mcolors.to_rgb(orientation_vector_color)
#                             phase_factor = 1.0 - 0.3 * (phase_i / max(1, phase_num - 1))
#                             phase_orient_color = tuple(max(0.0, c * phase_factor) for c in base_rgb)
#                         else:
#                             phase_orient_color = orientation_vector_color
                        
#                         alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
#                         handle = Line2D([0], [0], color=phase_orient_color, marker='>', markersize=8, 
#                                       alpha=alpha, linestyle='None')
#                         manual_handles.append(handle)
#                         manual_labels.append(f'ROI {roi_idx} Orientation Vector P{phase_i+1}')
            
#             ax.legend(manual_handles, manual_labels, loc='upper right', bbox_to_anchor=(1.3, 1.1))
#         else:
#             ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
#     # Format title (only if not minimal)
#     if show_title and not minimal:
#         metric_name = metric.replace('_', ' ').title()
#         ax.set_title(f'Directional tuning ({metric_name}) - {phase_num} phases', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
#     ax.grid(True, alpha=0.3)
    
#     # Hide theta labels if requested or in minimal mode
#     if not show_theta_labels or minimal:
#         ax.set_thetagrids([])
    
#     # Add mean direction vectors if requested (phase-dependent)
#     if show_mean_vector:
#         from pygor.timeseries.osds import tuning_metrics
#         import matplotlib.colors as mcolors
        
        
#         for roi_i, roi_idx in enumerate(rois):
#             base_color = base_colors[roi_i] if isinstance(base_colors, (list, np.ndarray)) else mean_vector_color
            
#             for phase_i in range(phase_num):
#                 # Get tuning function for this specific phase
#                 roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
#                 directions_sorted = np.array(directions_list)[sort_order]
                
#                 # Compute mean direction vector for this phase
#                 mean_vector = tuning_metrics.extract_mean_vector(roi_tuning_phase, directions_sorted)
                
                
#                 if not np.isnan(mean_vector['angle']):
#                     # Convert to radians
#                     mean_angle_rad = np.deg2rad(mean_vector['angle'])
                    
#                     # Scale magnitude for visibility
#                     mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning_phase)
                    
#                     # Create phase-specific color variations
#                     if isinstance(base_color, str):
#                         # If base_color is a string, create variations
#                         base_rgb = mcolors.to_rgb(base_color)
#                         # Lighten for later phases
#                         phase_factor = 0.8 + 0.2 * (phase_i / max(1, phase_num - 1))
#                         phase_color = tuple(min(1.0, c * phase_factor) for c in base_rgb)
#                     else:
#                         phase_color = base_color
                    
#                     alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
#                     arrow_width = 3 if phase_i == 0 else 2  # Make first phase thicker
                    
#                     # Plot mean direction vector arrow
#                     ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
#                                xytext=(0, 0),
#                                arrowprops=dict(arrowstyle='->', color=phase_color, lw=arrow_width, alpha=alpha),
#                                zorder=10 + phase_i)
                    
#                     # Legend will be created manually in the legend section
    
#     # Add mean orientation vectors if requested (phase-dependent)
#     if show_orientation_vector:
#         from pygor.timeseries.osds import tuning_metrics
#         import matplotlib.colors as mcolors
        
#         for roi_i, roi_idx in enumerate(rois):
            
#             for phase_i in range(phase_num):
#                 # Get tuning function for this specific phase
#                 roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
#                 directions_sorted = np.array(directions_list)[sort_order]
                
#                 # Compute mean orientation vector for this phase
#                 orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning_phase, directions_sorted)
                
                
#                 if not np.isnan(orientation_vector['angle']):
#                     # Convert to radians
#                     orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                    
#                     # Scale magnitude for visibility
#                     orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning_phase)
                    
#                     # Create phase-specific color variations for orientation vector
#                     if isinstance(orientation_vector_color, str):
#                         base_rgb = mcolors.to_rgb(orientation_vector_color)
#                         # Darken for later phases (opposite of direction vectors)
#                         phase_factor = 1.0 - 0.3 * (phase_i / max(1, phase_num - 1))
#                         phase_orient_color = tuple(max(0.0, c * phase_factor) for c in base_rgb)
#                     else:
#                         phase_orient_color = orientation_vector_color
                    
#                     alpha = 0.7 + 0.3 * (phase_i / (phase_num - 1)) if phase_num > 1 else 1.0
#                     arrow_width = 3 if phase_i == 0 else 2  # Make first phase thicker
                    
#                     # Plot mean orientation vector arrow
#                     ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
#                                xytext=(0, 0),
#                                arrowprops=dict(arrowstyle='->', color=phase_orient_color, lw=arrow_width, alpha=alpha),
#                                zorder=9 + phase_i)
                    
#                     # Legend will be created manually in the legend section
    
#     return fig, ax


# def plot_tuning_function_polar_overlay(tuning_functions, directions_list, rois=None, 
#                                       figsize=(6, 6), colors=None, phase_colors=None,
#                                       ax=None, show_title=True, show_theta_labels=True, 
#                                       show_tuning=True, show_mean_vector=False, 
#                                       mean_vector_color='red', show_orientation_vector=False, 
#                                       orientation_vector_color='orange', metric='peak', legend=True, minimal=False):
#     """
#     Plot multi-phase tuning functions overlaid on the same polar plot.
    
#     Parameters:
#     -----------
#     tuning_functions : np.ndarray
#         Array of shape (n_rois, n_directions, n_phases) containing tuning functions
#     directions_list : list of int/float
#         List of direction values in degrees
#     rois : list of int or None
#         ROI indices to plot. If None, plots all ROIs
#     figsize : tuple
#         Figure size (width, height)
#     colors : list or None
#         Colors for each ROI. If None, uses default color cycle
#     phase_colors : list or None
#         Colors for each phase. If None, uses default colors
#     ax : matplotlib.axes.Axes or None
#         Existing polar axes to plot on. If None, creates new figure and axes
#     show_title : bool
#         Whether to show the title on the plot
#     show_theta_labels : bool
#         Whether to show the theta (direction) labels on the plot
#     show_tuning : bool
#         Whether to show the tuning curve itself
#     show_mean_vector : bool
#         Whether to show mean direction vectors as overlays
#     mean_vector_color : str
#         Color for mean direction vector arrows
#     show_orientation_vector : bool
#         Whether to show mean orientation vectors as overlays
#     orientation_vector_color : str
#         Color for mean orientation vector arrows
#     metric : str
#         Metric name for title display
#     legend : bool
#         Whether to show the legend (default True)
#     minimal : bool
#         Whether to use minimal plotting (no titles or legends) (default False)
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The figure object
#     ax : matplotlib.axes.Axes
#         The polar plot axes object
#     """
#     # Create figure if not provided
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax = plt.subplot(projection='polar')
#         ax.set_theta_zero_location("E")  # 0° at right (90° north)
#     else:
#         fig = ax.figure
#         # Check if provided axes is polar, if not replace with polar axes
#         if ax.name != 'polar':
#             # Get the position of the current axes
#             pos = ax.get_position()
#             # Remove the current axes
#             ax.remove()
#             # Create a new polar axes in the same position
#             ax = fig.add_subplot(111, projection='polar')
#             ax.set_position(pos)
#             ax.set_theta_zero_location("E")
    
#     # Set up ROI indices
#     if rois is None:
#         rois = list(range(tuning_functions.shape[0]))
#     elif isinstance(rois, (int, np.integer)):
#         rois = [rois]
    
#     # Set up colors
#     if colors is None:
#         colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(rois)))
    
#     # Set default phase colors
#     if phase_colors is None:
#         phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
#     # Sort by direction for proper polar plot connectivity
#     sort_order = np.argsort(directions_list)
#     degrees = np.deg2rad(directions_list)[sort_order]
    
#     n_phases = tuning_functions.shape[2]
    
#     # Plot each phase for each ROI (only if show_tuning is True)
#     if show_tuning:
#         for roi_i, roi_idx in enumerate(rois):
#             roi_color = colors[roi_i] if isinstance(colors, (list, np.ndarray)) else colors
            
#             for phase_i in range(n_phases):
#                 # Get tuning function for this ROI and phase
#                 tuning_function = tuning_functions[roi_idx, sort_order, phase_i]
                
#                 # Close the loop
#                 tuning_function_closed = np.concatenate((tuning_function, [tuning_function[0]]))
#                 degrees_closed = np.concatenate((degrees, [degrees[0]]))
                
#                 # Use phase-specific colors
#                 phase_color = phase_colors[phase_i % len(phase_colors)]
                
#                 # Create different line styles for phases
#                 linestyle = ['-', '--', '-.', ':'][phase_i % 4]
#                 alpha = 0.8
                
#                 # Plot phase with descriptive label
#                 if len(rois) > 1:
#                     label = f'ROI {roi_idx} Phase {phase_i+1}'
#                 else:
#                     label = f'Phase {phase_i+1}'
                
#                 ax.plot(degrees_closed, tuning_function_closed, marker='o', 
#                         color=phase_color, linestyle=linestyle, alpha=alpha, 
#                         label=label, linewidth=2, markersize=4)
    
#     # Add legend if requested and not minimal and there are multiple traces/tuning/vectors shown
#     if legend and not minimal and (((len(rois) > 1 or n_phases > 1) and show_tuning) or show_mean_vector or show_orientation_vector):
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
#     # Format title (only if not minimal)
#     if show_title and not minimal:
#         metric_name = metric.replace('_', ' ').title()
#         ax.set_title(f'Directional tuning ({metric_name}) - Phase Overlay', fontsize=plt.rcParams['font.size'] * 1.2, pad=20)
    
#     ax.grid(True, alpha=0.3)
    
#     # Hide theta labels if requested or in minimal mode
#     if not show_theta_labels or minimal:
#         ax.set_thetagrids([])
    
#     # Add mean direction vectors if requested (phase-dependent)
#     if show_mean_vector:
#         from pygor.timeseries.osds import tuning_metrics
        
#         for roi_i, roi_idx in enumerate(rois):
#             for phase_i in range(n_phases):
#                 # Get tuning function for this specific phase
#                 roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
#                 directions_sorted = np.array(directions_list)[sort_order]
                
#                 # Compute mean direction vector for this phase
#                 mean_vector = tuning_metrics.extract_mean_vector(roi_tuning_phase, directions_sorted)
                
#                 if not np.isnan(mean_vector['angle']):
#                     # Convert to radians
#                     mean_angle_rad = np.deg2rad(mean_vector['angle'])
                    
#                     # Scale magnitude for visibility
#                     mean_mag_scaled = mean_vector['magnitude'] * np.max(roi_tuning_phase)
                    
#                     # Use phase-specific colors
#                     phase_color = phase_colors[phase_i % len(phase_colors)]
                    
#                     # Plot mean direction vector arrow
#                     ax.annotate('', xy=(mean_angle_rad, mean_mag_scaled), 
#                                xytext=(0, 0),
#                                arrowprops=dict(arrowstyle='->', color=phase_color, lw=3),
#                                zorder=10 + phase_i)
    
#     # Add mean orientation vectors if requested (phase-dependent)
#     if show_orientation_vector:
#         from pygor.timeseries.osds import tuning_metrics
        
#         for roi_i, roi_idx in enumerate(rois):
#             for phase_i in range(n_phases):
#                 # Get tuning function for this specific phase
#                 roi_tuning_phase = tuning_functions[roi_idx, sort_order, phase_i]
#                 directions_sorted = np.array(directions_list)[sort_order]
                
#                 # Compute mean orientation vector for this phase
#                 orientation_vector = tuning_metrics.extract_orientation_vector(roi_tuning_phase, directions_sorted)
                
#                 if not np.isnan(orientation_vector['angle']):
#                     # Convert to radians
#                     orient_angle_rad = np.deg2rad(orientation_vector['angle'])
                    
#                     # Scale magnitude for visibility
#                     orient_mag_scaled = orientation_vector['magnitude'] * np.max(roi_tuning_phase)
                    
#                     # Use phase-specific colors with a twist (darker for orientation)
#                     import matplotlib.colors as mcolors
#                     phase_color = phase_colors[phase_i % len(phase_colors)]
#                     base_rgb = mcolors.to_rgb(phase_color)
#                     darker_color = tuple(c * 0.7 for c in base_rgb)
                    
#                     # Plot mean orientation vector arrow
#                     ax.annotate('', xy=(orient_angle_rad, orient_mag_scaled), 
#                             xytext=(0, 0),
#                             arrowprops=dict(arrowstyle='->', color=darker_color, lw=3),
#                             zorder=9 + phase_i)
    
#     return fig, ax

