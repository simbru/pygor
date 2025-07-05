import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def plot_experiment_directional_analysis(
    experiment,
    roi=2,
    colour_list=None,
    colourmap=None,
    figsize=None,
    metric="peak",
    marker_alpha=0.25,
    polar_alpha=0.6,
    title_fontsize=10,
):
    """
    Plot directional analysis for an experiment with line traces and polar tuning plots.

    Parameters:
    -----------
    experiment : Experiment object
        Pygor experiment object containing multiple recordings
    roi : int
        ROI index to analyze (default 2)
    colour_list : list or None
        List of colors for each recording. Must match number of recordings.
        If None, uses default colormap.
    colourmap : str or matplotlib colormap, optional
        Colormap to generate colors from. Overrides colour_list if provided.
        Examples: 'viridis', 'plasma', plt.cm.Set1
    figsize : tuple or None
        Figure size (width, height). If None, scales with number of recordings.
    metric : str
        Metric for tuning function computation (default 'peak')
    marker_alpha : float
        Alpha transparency for marker lines (default 0.25)
    polar_alpha : float
        Alpha transparency for second half polar plots (default 0.6)
    title_fontsize : int
        Font size for polar plot titles (default 10)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    n_recordings = len(experiment.recording)
    
    # Handle color assignment
    if colourmap is not None:
        # Generate colors from colormap
        if isinstance(colourmap, str):
            cmap = plt.cm.get_cmap(colourmap)
        else:
            cmap = colourmap
        colors = [cmap(i / max(1, n_recordings - 1)) for i in range(n_recordings)]
    elif colour_list is not None:
        # Validate user-provided color list
        if len(colour_list) != n_recordings:
            raise ValueError(f"colour_list length ({len(colour_list)}) must match "
                           f"number of recordings ({n_recordings})")
        colors = colour_list
    else:
        # Default to viridis colormap
        colors = [plt.cm.viridis(i / max(1, n_recordings - 1)) for i in range(n_recordings)]
    
    # Handle figure size
    if figsize is None:
        figsize = (15, 2 * n_recordings)

    # Calculate shared amplitude range for all polar plots
    all_tuning_values = []
    for rec in experiment.recording:
        dir_length = len(rec.averages[0]) // rec.dir_num
        # Get tuning values for both halves
        tuning_1st = rec.compute_tuning_function(
            roi_index=roi, window=(0, dir_length // 2), metric=metric
        )
        tuning_2nd = rec.compute_tuning_function(
            roi_index=roi, window=(dir_length // 2, dir_length), metric=metric
        )
        all_tuning_values.extend([tuning_1st, tuning_2nd])

    # Calculate shared range
    all_values = np.concatenate(all_tuning_values)
    shared_max = np.max(all_values) * 1.1  # Add 10% padding

    # Create figure with custom grid spacing
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        n_recordings,
        3,
        figure=fig,
        width_ratios=[2, 0.5, 0.5],
        wspace=0.3,
        hspace=0.3,
    )

    # Get markers and directions from first recording
    markers = experiment.recording[0].get_average_markers()
    directions_list = experiment.recording[0].directions_list

    ax_traces = []  # Store axes for sharing
    for n, rec in enumerate(experiment.recording):
        # First column - line traces with shared axes
        if n == 0:
            ax_trace = fig.add_subplot(gs[n, 0])
            ax_traces.append(ax_trace)
        else:
            ax_trace = fig.add_subplot(
                gs[n, 0], sharex=ax_traces[0], sharey=ax_traces[0]
            )
            ax_traces.append(ax_trace)

        ax_trace.plot(rec.averages[roi], color=colors[n], label=rec.name.split("_")[-1])
        for i in markers:
            ax_trace.axvline(i, color="k", linestyle="-", alpha=marker_alpha)
        ax_trace.legend(loc='upper right')
        
        # Hide x-axis labels for all except the last row
        if n < n_recordings - 1:
            ax_trace.tick_params(labelbottom=False)

        # Add column titles and secondary x-axis only to top row
        if n == 0:
            ax_trace.set_title("Time traces", fontsize=title_fontsize + 2, pad=20)
            secondary_xaxis = ax_trace.secondary_xaxis("top")
            secondary_xaxis.set_xticks(
                np.linspace(3000, len(rec.averages[0]) - 3000, 8),
                labels=[str(i) + "°" for i in directions_list],
            )

        # Get the length of each direction's data
        dir_length = len(rec.averages[0]) // rec.dir_num

        # Second column - polar plot for first half
        ax_polar1 = fig.add_subplot(gs[n, 1], projection="polar")
        rec.plot_tuning_function(
            rois=roi,
            ax=ax_polar1,
            window=(0, dir_length // 2),
            metric=metric,
            colors=[colors[n]],
            show_title=False,
            show_theta_labels=(n == 0),
        )
        if n == 0:
            ax_polar1.set_title("First half", fontsize=title_fontsize + 2, pad=20)
        ax_polar1.set_ylim(0, shared_max)

        # Third column - polar plot for second half (more transparent)
        ax_polar2 = fig.add_subplot(gs[n, 2], projection="polar")
        # Create a more transparent version of the color
        base_color = mcolors.to_rgba(colors[n])
        transparent_color = (*base_color[:3], polar_alpha)

        rec.plot_tuning_function(
            rois=roi,
            ax=ax_polar2,
            window=(dir_length // 2, dir_length),
            metric=metric,
            colors=[transparent_color],
            show_title=False,
            show_theta_labels=(n == 0),
        )
        if n == 0:
            ax_polar2.set_title("Second half", fontsize=title_fontsize + 2, pad=20)
        ax_polar2.set_ylim(0, shared_max)

    return fig


def plot_experiment_directional_analysis_averaged(
    experiment,
    colour_list=None,
    colourmap=None,
    figsize=None,
    metric="peak",
    marker_alpha=0.25,
    polar_alpha=0.6,
    title_fontsize=10,
    group_by_index=True,
    custom_groups=None,
    individual_alpha=0.3,
):
    """
    Plot directional analysis for grouped/averaged recordings from an experiment.
    
    This function first averages across all ROIs for each recording, then groups 
    recordings by their naming convention (e.g., R, UV, White) and plots the 
    average response for each group, with individual ROI-averaged recordings shown
    as faint background traces.

    Parameters:
    -----------
    experiment : Experiment object
        Pygor experiment object containing multiple recordings
    colour_list : list or None
        List of colors for each group. Must match number of groups.
        If None, uses default colormap.
    colourmap : str or matplotlib colormap, optional
        Colormap to generate colors from. Overrides colour_list if provided.
        Examples: 'viridis', 'plasma', plt.cm.Set1
    figsize : tuple or None
        Figure size (width, height). If None, scales with number of groups.
    metric : str
        Metric for tuning function computation (default 'peak')
    marker_alpha : float
        Alpha transparency for marker lines (default 0.25)
    polar_alpha : float
        Alpha transparency for second half polar plots (default 0.6)
    title_fontsize : int
        Font size for polar plot titles (default 10)
    group_by_index : bool
        If True, groups recordings by their position (assumes recordings are ordered
        by condition). If False, uses recording names to group by suffix after last '_'.
    custom_groups : dict or None
        Custom grouping dictionary: {group_name: [recording_indices]}.
        Overrides group_by_index if provided.
    individual_alpha : float
        Alpha transparency for individual recording traces (default 0.3)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    groups : dict
        Dictionary of groups with their recordings for reference
    """
    
    # Group recordings
    if custom_groups is not None:
        groups = custom_groups
        group_names = list(groups.keys())
    elif group_by_index:
        # Group by position - assumes recordings are ordered by condition
        # For now, default to treating each recording as its own group
        # This can be extended based on your specific grouping needs
        groups = {}
        for i, rec in enumerate(experiment.recording):
            group_name = rec.name.split('_')[-1] if '_' in rec.name else f'Group_{i}'
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(i)
        group_names = list(groups.keys())
    else:
        # Group by suffix after last underscore
        groups = {}
        for i, rec in enumerate(experiment.recording):
            group_name = rec.name.split('_')[-1] if '_' in rec.name else 'Ungrouped'
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(i)
        group_names = list(groups.keys())
    
    n_groups = len(groups)
    
    # Handle color assignment
    if colourmap is not None:
        # Generate colors from colormap
        if isinstance(colourmap, str):
            cmap = plt.cm.get_cmap(colourmap)
        else:
            cmap = colourmap
        colors = [cmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
    elif colour_list is not None:
        # Validate user-provided color list
        if len(colour_list) != n_groups:
            raise ValueError(f"colour_list length ({len(colour_list)}) must match "
                           f"number of groups ({n_groups})")
        colors = colour_list
    else:
        # Default to viridis colormap
        colors = [plt.cm.viridis(i / max(1, n_groups - 1)) for i in range(n_groups)]
    
    # Handle figure size
    if figsize is None:
        figsize = (15, 2 * n_groups)

    # Calculate averages for each group
    group_averages = {}
    all_tuning_values = []
    
    for group_name, rec_indices in groups.items():
        # Get all recordings for this group
        group_recordings = [experiment.recording[i] for i in rec_indices]
        
        # Calculate average trace for this group (first average across all ROIs for each recording)
        roi_averaged_traces = [np.mean(rec.averages, axis=0) for rec in group_recordings]
        all_traces = np.array(roi_averaged_traces)
        avg_trace = np.mean(all_traces, axis=0)
        
        # Also collect all individual ROI traces for this group (for gray background lines)
        all_individual_roi_traces = []
        for rec in group_recordings:
            for roi_trace in rec.averages:  # Each ROI's trace
                all_individual_roi_traces.append(roi_trace)
        
        group_averages[group_name] = {
            'avg_trace': avg_trace,
            'individual_traces': all_traces,  # ROI-averaged per recording
            'individual_roi_traces': all_individual_roi_traces,  # Each ROI trace
            'recordings': group_recordings
        }
        
        # Calculate tuning functions - compute for each ROI trace, then average
        dir_length = len(avg_trace) // group_recordings[0].dir_num
        all_roi_tunings_1st, all_roi_tunings_2nd = [], []
        
        for roi_trace in all_individual_roi_traces:
            # Split trace by directions and compute tuning for both halves
            tuning_1st_roi = []
            tuning_2nd_roi = []
            
            for direction in range(group_recordings[0].dir_num):
                start_idx = direction * dir_length
                mid_idx = start_idx + dir_length // 2
                end_idx = start_idx + dir_length
                
                trace_1st_half = roi_trace[start_idx:mid_idx]
                trace_2nd_half = roi_trace[mid_idx:end_idx]
                
                # Apply metric
                if metric == 'peak':
                    tuning_1st_roi.append(np.max(np.abs(trace_1st_half)))
                    tuning_2nd_roi.append(np.max(np.abs(trace_2nd_half)))
                elif metric == 'max':
                    tuning_1st_roi.append(np.max(trace_1st_half))
                    tuning_2nd_roi.append(np.max(trace_2nd_half))
                elif metric == 'mean':
                    tuning_1st_roi.append(np.mean(trace_1st_half))
                    tuning_2nd_roi.append(np.mean(trace_2nd_half))
                else:  # Default to peak
                    tuning_1st_roi.append(np.max(np.abs(trace_1st_half)))
                    tuning_2nd_roi.append(np.max(np.abs(trace_2nd_half)))
            
            all_roi_tunings_1st.append(tuning_1st_roi)
            all_roi_tunings_2nd.append(tuning_2nd_roi)
        
        # Average across all ROI tuning functions
        all_roi_tunings_1st = np.array(all_roi_tunings_1st)
        all_roi_tunings_2nd = np.array(all_roi_tunings_2nd)
        tuning_1st = np.mean(all_roi_tunings_1st, axis=0)
        tuning_2nd = np.mean(all_roi_tunings_2nd, axis=0)
        
        # Store results
        all_tuning_values.extend([tuning_1st, tuning_2nd])
        group_averages[group_name].update({
            'tuning_1st': tuning_1st,
            'tuning_2nd': tuning_2nd,
            'individual_tunings_1st': all_roi_tunings_1st,
            'individual_tunings_2nd': all_roi_tunings_2nd
        })

    # Calculate shared range for polar plots
    all_values = np.concatenate(all_tuning_values)
    shared_max = np.max(all_values) * 1.1  # Add 10% padding

    # Create figure with custom grid spacing
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        n_groups,
        3,
        figure=fig,
        width_ratios=[2, 0.5, 0.5],
        wspace=0.3,
        hspace=0.3,
    )

    # Get markers and directions from first recording
    markers = experiment.recording[0].get_average_markers()
    directions_list = experiment.recording[0].directions_list

    ax_traces = []  # Store axes for sharing
    for n, (group_name, group_data) in enumerate(group_averages.items()):
        # First column - line traces with shared axes
        if n == 0:
            ax_trace = fig.add_subplot(gs[n, 0])
            ax_traces.append(ax_trace)
        else:
            ax_trace = fig.add_subplot(
                gs[n, 0], sharex=ax_traces[0], sharey=ax_traces[0]
            )
            ax_traces.append(ax_trace)

        # Plot individual ROI traces as faint gray lines
        for individual_roi_trace in group_data['individual_roi_traces']:
            ax_trace.plot(individual_roi_trace, color='gray', alpha=0.1, linewidth=0.8, zorder=1)
        
        # Plot group average as bold colored line
        ax_trace.plot(group_data['avg_trace'], color=colors[n], linewidth=3, label=f'{group_name} (avg)', zorder=2)
        
        for i in markers:
            ax_trace.axvline(i, color="k", linestyle="-", alpha=marker_alpha)
        ax_trace.legend(loc='upper right')
        
        # Hide x-axis labels for all except the last row
        if n < n_groups - 1:
            ax_trace.tick_params(labelbottom=False)

        # Add column titles and secondary x-axis only to top row
        if n == 0:
            ax_trace.set_title("Time traces (ROI & group averaged)", fontsize=title_fontsize + 2, pad=20)
            secondary_xaxis = ax_trace.secondary_xaxis("top")
            secondary_xaxis.set_xticks(
                np.linspace(3000, len(group_data['avg_trace']) - 3000, 8),
                labels=[str(i) + "°" for i in directions_list],
            )

        # Get the length of each direction's data
        dir_length = len(group_data['avg_trace']) // experiment.recording[0].dir_num

        # Second column - polar plot for first half
        ax_polar1 = fig.add_subplot(gs[n, 1], projection="polar")
        
        # Sort for proper polar plot connectivity
        sort_order = np.argsort(directions_list)
        degrees = np.deg2rad(directions_list)[sort_order]
        
        # Plot individual ROI tuning functions as faint gray lines
        for individual_tuning in group_data['individual_tunings_1st']:
            individual_sorted = individual_tuning[sort_order]
            individual_closed = np.concatenate((individual_sorted, [individual_sorted[0]]))
            degrees_closed = np.concatenate((degrees, [degrees[0]]))
            ax_polar1.plot(degrees_closed, individual_closed, color='gray', alpha=0.1, linewidth=0.8, zorder=1)
        
        # Plot group average as bold colored line
        tuning_1st_sorted = group_data['tuning_1st'][sort_order]
        tuning_1st_closed = np.concatenate((tuning_1st_sorted, [tuning_1st_sorted[0]]))
        degrees_closed = np.concatenate((degrees, [degrees[0]]))
        
        ax_polar1.plot(degrees_closed, tuning_1st_closed, marker='o', color=colors[n], linewidth=3, zorder=2)
        ax_polar1.set_theta_zero_location("E")
        if n == 0:
            ax_polar1.set_title("First half", fontsize=title_fontsize + 2, pad=20)
            ax_polar1.set_thetagrids([])  # Keep consistent with original
        else:
            ax_polar1.set_thetagrids([])
        ax_polar1.set_ylim(0, shared_max)
        ax_polar1.grid(True, alpha=0.3)

        # Third column - polar plot for second half (more transparent)
        ax_polar2 = fig.add_subplot(gs[n, 2], projection="polar")
        
        # Plot individual ROI tuning functions as faint gray lines
        for individual_tuning in group_data['individual_tunings_2nd']:
            individual_sorted = individual_tuning[sort_order]
            individual_closed = np.concatenate((individual_sorted, [individual_sorted[0]]))
            degrees_closed = np.concatenate((degrees, [degrees[0]]))
            ax_polar2.plot(degrees_closed, individual_closed, color='gray', alpha=0.1, linewidth=0.8, zorder=1)
        
        # Plot group average as bold colored line (transparent)
        tuning_2nd_sorted = group_data['tuning_2nd'][sort_order]
        tuning_2nd_closed = np.concatenate((tuning_2nd_sorted, [tuning_2nd_sorted[0]]))
        degrees_closed = np.concatenate((degrees, [degrees[0]]))
        
        # Create a more transparent version of the color
        base_color = mcolors.to_rgba(colors[n])
        transparent_color = (*base_color[:3], polar_alpha)
        
        ax_polar2.plot(degrees_closed, tuning_2nd_closed, marker='o', color=transparent_color, linewidth=3, zorder=2)
        ax_polar2.set_theta_zero_location("E")
        if n == 0:
            ax_polar2.set_title("Second half", fontsize=title_fontsize + 2, pad=20)
            ax_polar2.set_thetagrids([])
        else:
            ax_polar2.set_thetagrids([])
        ax_polar2.set_ylim(0, shared_max)
        ax_polar2.grid(True, alpha=0.3)

    return fig, groups


# Example usage (uncomment to test):
# Default viridis colors: fig = plot_experiment_directional_analysis(experiment, roi=2)
# Custom colors: fig = plot_experiment_directional_analysis(experiment, roi=2, colour_list=['red', 'blue', 'green'])
# Custom colormap: fig = plot_experiment_directional_analysis(experiment, roi=2, colourmap='plasma')
# Averaged groups: fig, groups = plot_experiment_directional_analysis_averaged(experiment)
# plt.show()
