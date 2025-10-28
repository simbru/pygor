
# def plot_directional_responses_circular(data, directions_list=None, figsize=(8, 8)):
#     """
#     Plot directional responses in a circular arrangement.

#     Parameters:
#     -----------
#     data : np.ndarray
#             Array of shape (n_directions, n_timepoints) containing response traces
#     directions_list : list of int/float, optional
#             List of direction values in degrees. If None, uses evenly spaced angles.
#     figsize : tuple
#             Figure size (width, height)
#     """
#     n_directions = data.shape[0]

#     # Convert direction degrees to radians for positioning
#     if directions_list is not None:
#         angles = np.radians(directions_list)
#     else:
#         angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

#     fig = plt.figure(figsize=figsize, facecolor="white")

#     for i, angle in enumerate(angles):
#         # Calculate subplot position (start from right, go counter-clockwise to match polar)
#         x_center = 0.5 + 0.32 * np.cos(angle)
#         y_center = 0.5 + 0.32 * np.sin(angle)

#         # Create subplot
#         ax = fig.add_axes([x_center - 0.06, y_center - 0.06, 0.12, 0.12])

#         # Plot trace
#         ax.plot(data[i], "k-", linewidth=1.2)
#         ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

#         # Clean styling
#         ax.set_xlim(0, len(data[i]))
#         y_range = np.max(np.abs(data[i])) * 1.1 if np.max(np.abs(data[i])) > 0 else 1
#         ax.set_ylim(-y_range, y_range)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         ax.set_facecolor("#f8f8f8")

#         # Add direction label
#         # if directions_list is not None:
#         #     label_x = x_center + 0.08 * np.cos(angle - np.pi/2)
#         #     label_y = y_center + 0.08 * np.sin(angle - np.pi/2)
#         #     fig.text(label_x, label_y, f'{int(directions_list[i])}°',
#         #             ha='center', va='center', fontsize=10, fontweight='bold')

#     return fig


# def plot_directional_responses_circular_with_polar(
#     osds_obj,
#     directions_list=None,
#     figsize=(10, 10),
#     metric="peak",
#     polar_kwargs=None,
#     polar_size=0.3,
#     roi_index=-1,
#     show_trials=True,
#     data_crop=None,
#     use_phases=None,
#     phase_colors=None,
# ):
#     """
#     Plot directional responses in a circular arrangement with central polar plot.

#     Parameters:
#     -----------
#     data : np.ndarray or None
#             Array of shape (n_directions, n_timepoints) containing response traces.
#             If osds_obj is provided, this can be None.
#     directions_list : list of int/float, optional
#             List of direction values in degrees. If None, uses evenly spaced angles.
#     figsize : tuple
#             Figure size (width, height)
#     metric : str or callable
#             Summary metric to plot in polar plot. Options:
#             - 'peak': maximum absolute value
#             - 'auc': area under curve (absolute)
#             - 'mean': mean response
#             - 'peak_positive': maximum positive value
#             - 'peak_negative': minimum negative value
#             - custom function that takes 1D array and returns scalar
#     polar_kwargs : dict, optional
#             Additional keyword arguments for polar plot styling
#     polar_size : float, optional
#             Size of the central polar plot as fraction of figure (default 0.3)
#     osds_obj : OSDS object, optional
#             If provided, will extract data and optionally show individual trials
#     roi_index : int, optional
#             ROI index to plot (default -1 for last ROI)
#     show_trials : bool, optional
#             Whether to show individual trial traces as faint lines (default False)
#     data_crop : tuple, optional
#             Tuple of (start, end) indices to crop data timepoints
#     use_phases : bool or None
#             If None, uses osds_obj.dir_phase_num > 1 to decide
#             If True, forces phase analysis with overlay in polar plot
#             If False, forces single-phase analysis
#     phase_colors : list or None
#             Colors for each phase. If None, uses default colors

#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#             The figure object
#     ax_polar : matplotlib.axes.Axes
#             The polar plot axes object
#     """

#     # Automatically use phases if dir_phase_num > 1 and use_phases not specified
#     if use_phases is None:
#         use_phases = osds_obj.dir_phase_num > 1
    
#     # Set default phase colors
#     if phase_colors is None:
#         phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
#     # Extract data from OSDS object if provided
#     trial_data = None  # Initialize here so it's available throughout the function

#     data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
#     directions_list = osds_obj.directions_list

#     # Get trial data if requested
#     if show_trials:
#         # Shape: (n_directions, n_rois, n_trials, n_timepoints) -> (n_directions, n_trials, n_timepoints)
#         trial_data = osds_obj.split_snippets_directionally()[:, roi_index, :, :]

#     if data_crop is not None:
#         print(data.shape)
#         data = data[:, data_crop[0] : data_crop[1]]
#         if trial_data is not None:
#             trial_data = trial_data[:, :, data_crop[0] : data_crop[1]]

#     # # Handle the case where data is None but osds_obj is provided
#     # data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])

#     n_directions = data.shape[0]

#     # Convert direction degrees to radians for positioning
#     if directions_list is not None:
#         angles = np.radians(directions_list)
#         directions_deg = np.array(directions_list)
#     else:
#         angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
#         directions_deg = np.degrees(angles)

#     # Calculate summary metric for each direction
#     if metric == "peak":
#         values = np.array([np.max(np.abs(trace)) for trace in data])
#     elif metric == "auc":
#         values = np.array([np.trapz(np.abs(trace)) for trace in data])
#     elif metric == "mean":
#         values = np.array([np.mean(trace) for trace in data])
#     elif metric == "peak_positive":
#         values = np.array([np.max(trace) for trace in data])
#     elif metric == "peak_negative":
#         values = np.array([np.min(trace) for trace in data])
#     elif callable(metric):
#         values = np.array([metric(trace) for trace in data])
#     else:
#         raise ValueError(f"Unknown metric: {metric}")

#     # Sort by angle for proper polar plot connectivity
#     sort_indices = np.argsort(directions_deg)
#     sorted_angles = angles[sort_indices]
#     sorted_values = values[sort_indices]
#     sorted_data = data[sort_indices]
#     sorted_directions_deg = directions_deg[sort_indices]
#     # Also sort trial data if it exists
#     if show_trials and trial_data is not None:
#         sorted_trial_data = trial_data[sort_indices]

#     # Set up polar plot defaults
#     default_polar_kwargs = {
#         "color": "#2E8B57",  # Sea green instead of red
#         "linewidth": 2,
#         "marker": "o",
#         "markersize": 6,
#         "alpha": 0.8,
#     }
#     if polar_kwargs:
#         default_polar_kwargs.update(polar_kwargs)

#     fig = plt.figure(figsize=figsize, facecolor="white")

#     # Create central polar plot (larger and closer to periphery)
#     ax_polar = fig.add_axes([0.225, 0.22, 0.55, 0.55], projection="polar")

#     # Plot polar data (need to append first point to close the circle)
#     polar_angles = np.append(sorted_angles, sorted_angles[0])
#     polar_values = np.append(sorted_values, sorted_values[0])

#     ax_polar.plot(polar_angles, polar_values, **default_polar_kwargs)
#     ax_polar.fill(
#         polar_angles, polar_values, alpha=0.3, color=default_polar_kwargs["color"]
#     )

#     # Style polar plot
#     ax_polar.set_theta_zero_location("E")  # 0° at right (90° north)
#     # ax_polar.set_theta_direction(-1)  # Clockwise
#     ax_polar.set_title(
#         f"Directional Tuning\n({metric.replace('_', ' ').title()})",
#         fontsize=12,
#         fontweight="bold",
#         pad=20,
#     )
#     ax_polar.grid(True, alpha=0.3)
#     # Calculate global y-limits for all traces
#     if show_trials and trial_data is not None:
#         # Use trial data for scaling to capture full variability
#         all_trace_data = sorted_trial_data.flatten()
#     else:
#         # Use average data for scaling
#         all_trace_data = sorted_data.flatten()

#     global_min = np.min(all_trace_data)
#     global_max = np.max(all_trace_data)
#     global_range = global_max - global_min
#     padding = global_range * 0.1  # 25% padding
#     y_min_global = global_min - padding
#     y_max_global = global_max + padding
#     # Set minimum range to (-3, 3)
#     y_min_global = min(y_min_global, -5)
#     y_max_global = max(y_max_global, 5)
#     # Add individual trace plots around the perimeter
#     for i, angle in enumerate(sorted_angles):
#         # Calculate subplot position (further out to accommodate central polar plot)
#         # Fixed angle calculation to match standard directional conventions
#         radius = 0.38  # Increased radius to make room for central plot
#         x_center = 0.5 + radius * np.cos(angle)  # 0° at right, counter-clockwise
#         y_center = 0.5 + radius * np.sin(angle)  # 0° at right, counter-clockwise

#         # Create subplot
#         subplot_size = 0.08  # Slightly smaller to fit more around
#         ax = fig.add_axes(
#             [
#                 x_center - subplot_size / 2,
#                 y_center - subplot_size / 2,
#                 subplot_size,
#                 subplot_size,
#             ]
#         )

#         # Plot trace - use sorted data
#         if show_trials:
#             ax.plot(sorted_trial_data[i].T, "k-", linewidth=0.5, alpha=0.33)
#         ax.plot(sorted_data[i], "k-", linewidth=1)
#         ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

#         # Clean neutral background
#         # ax.set_facecolor("#f8f8f8")

#         # Clean styling
#         ax.set_xlim(0, len(sorted_data[i]))  # Fixed: use sorted_data
#         # y_range = np.max(np.abs(data)) * 1.25 if np.max(np.abs(data)) > 5 else 5
#         # y_range = data_absmax + data_absmax * 0.5
#         # ax.set_ylim(-y_range, y_range)
#         ax.set_ylim(y_min_global, y_max_global)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # ax.set_title(f"{directions_deg[i]:.0f}° or {angle:.0f} rad", fontsize=8)
#         ax.set_title(f"{sorted_directions_deg[i]:.0f}°", fontsize=8)
#         # sns.despine(ax=ax, left=False, bottom=False, right=True, top=False)

#     return fig, ax_polar


# def plot_directional_responses_dual_phase(
#     osds_obj,
#     phase_split=None,
#     directions_list=None,
#     figsize=(12, 10),
#     metric="peak",
#     polar_kwargs=None,
#     roi_index=-1,
#     show_trials=True,
#     phase_colors=("#2E8B57", "#B8860B"),  # Sea green, Dark goldenrod
# ):
#     """
#     Plot directional responses for two stimulus phases (OFF->ON and ON->OFF)
#     with overlapping polar plots and separate trace arrangements.

#     Parameters:
#     -----------
#     osds_obj : OSDS object
#         Object containing the directional response data
#     phase_split : int
#         Frame number where phase 1 ends and phase 2 begins (default 3200)
#     directions_list : list of int/float, optional
#         List of direction values in degrees
#     figsize : tuple
#         Figure size (width, height)
#     metric : str or callable
#         Summary metric for polar plots
#     polar_kwargs : dict, optional
#         Additional keyword arguments for polar plot styling
#     roi_index : int, optional
#         ROI index to plot (default -1 for last ROI)
#     show_trials : bool, optional
#         Whether to show individual trial traces
#     phase_colors : tuple
#         Colors for (phase1, phase2) plots

#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#     ax_polar : matplotlib.axes.Axes
#         The central polar plot axes
#     """

#     # Extract data
#     data = np.squeeze(osds_obj.split_averages_directionally()[:, [roi_index]])
#     directions_list = osds_obj.directions_list

#     # Get trial data if requested
#     trial_data = None
#     if show_trials:
#         trial_data = osds_obj.split_snippets_directionally()[:, roi_index, :, :]
#     if phase_split is None:
#         phase_split = phase_split = osds_obj.averages.shape[1] // osds_obj.trigger_mode // 2  # Should be ~2866
#         print(f"Phase split: {phase_split}")
#     else:
#         phase_split = int(phase_split)
#     # Split data into two phases
#     phase1_data = data[:, :phase_split]  # OFF->ON
#     phase2_data = data[:, phase_split:]  # ON->OFF

#     if trial_data is not None:
#         phase1_trial_data = trial_data[:, :, :phase_split]
#         phase2_trial_data = trial_data[:, :, phase_split:]

#     n_directions = data.shape[0]

#     # Convert directions and sort
#     if directions_list is not None:
#         angles = np.radians(directions_list)
#         directions_deg = np.array(directions_list)
#     else:
#         angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
#         directions_deg = np.degrees(angles)

#     sort_indices = np.argsort(directions_deg)
#     sorted_angles = angles[sort_indices]
#     sorted_directions_deg = directions_deg[sort_indices]

#     # Sort both phases
#     sorted_phase1_data = phase1_data[sort_indices]
#     sorted_phase2_data = phase2_data[sort_indices]

#     if trial_data is not None:
#         sorted_phase1_trial_data = phase1_trial_data[sort_indices]
#         sorted_phase2_trial_data = phase2_trial_data[sort_indices]

#     # Calculate metrics for both phases
#     if metric == "peak":
#         phase1_values = np.array(
#             [np.max(np.abs(trace)) for trace in sorted_phase1_data]
#         )
#         phase2_values = np.array(
#             [np.max(np.abs(trace)) for trace in sorted_phase2_data]
#         )
#     elif metric == "auc":
#         phase1_values = np.array(
#             [np.trapz(np.abs(trace)) for trace in sorted_phase1_data]
#         )
#         phase2_values = np.array(
#             [np.trapz(np.abs(trace)) for trace in sorted_phase2_data]
#         )
#     elif metric == "mean":
#         phase1_values = np.array([np.mean(trace) for trace in sorted_phase1_data])
#         phase2_values = np.array([np.mean(trace) for trace in sorted_phase2_data])
#     elif metric == "peak_positive":
#         phase1_values = np.array([np.max(trace) for trace in sorted_phase1_data])
#         phase2_values = np.array([np.max(trace) for trace in sorted_phase2_data])
#     elif metric == "peak_negative":
#         phase1_values = np.array([np.min(trace) for trace in sorted_phase1_data])
#         phase2_values = np.array([np.min(trace) for trace in sorted_phase2_data])
#     elif callable(metric):
#         phase1_values = np.array([metric(trace) for trace in sorted_phase1_data])
#         phase2_values = np.array([metric(trace) for trace in sorted_phase2_data])
#     else:
#         raise ValueError(f"Unknown metric: {metric}")

#     # Calculate global y-limits using both phases
#     if show_trials and trial_data is not None:
#         all_trace_data = np.concatenate(
#             [sorted_phase1_trial_data.flatten(), sorted_phase2_trial_data.flatten()]
#         )
#     else:
#         all_trace_data = np.concatenate(
#             [sorted_phase1_data.flatten(), sorted_phase2_data.flatten()]
#         )

#     global_min = np.min(all_trace_data)
#     global_max = np.max(all_trace_data)
#     global_range = global_max - global_min
#     padding = global_range * 0.25
#     y_min_global = global_min - padding
#     y_max_global = global_max + padding

#     # Set minimum range to (-3, 3)
#     y_min_global = min(y_min_global, -3)
#     y_max_global = max(y_max_global, 3)

#     # Create figure
#     fig = plt.figure(figsize=figsize, facecolor="white")

#     # Create central polar plot
#     ax_polar = fig.add_axes([0.225, 0.22, 0.55, 0.55], projection="polar")

#     # Plot both phases on same polar plot
#     for phase_idx, (values, color, label) in enumerate(
#         zip([phase1_values, phase2_values], phase_colors, ["OFF→ON", "ON→OFF"])
#     ):
#         # Close the circle
#         polar_angles = np.append(sorted_angles, sorted_angles[0])
#         polar_values = np.append(values, values[0])

#         # Set up plot style
#         plot_kwargs = {
#             "color": color,
#             "linewidth": 2.5,
#             "marker": "o",
#             "markersize": 6,
#             "alpha": 0.8,
#             "label": label,
#         }
#         if polar_kwargs:
#             plot_kwargs.update(polar_kwargs)

#         ax_polar.plot(polar_angles, polar_values, **plot_kwargs)
#         ax_polar.fill(polar_angles, polar_values, alpha=0.2, color=color)

#     # Style polar plot
#     # ax_polar.set_theta_zero_location("E")  # 0° at right (90° north)
#     # ax_polar.set_theta_direction(-1)
#     ax_polar.set_title(
#         f"Dual Phase Directional Tuning\n({metric.replace('_', ' ').title()})",
#         fontsize=14,
#         fontweight="bold",
#         pad=20,
#     )
#     ax_polar.grid(True, alpha=0.3)
#     ax_polar.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1))

#     # Add trace plots around perimeter - Phase 1 (inner ring)
#     for i, angle in enumerate(sorted_angles):
#         # Single trace plot per direction
#         radius = 0.38  
#         x_center = 0.5 + radius * np.cos(angle)
#         y_center = 0.5 + radius * np.sin(angle)
        
#         subplot_size = 0.08
#         ax = fig.add_axes([
#             x_center - subplot_size / 2,
#             y_center - subplot_size / 2,
#             subplot_size,
#             subplot_size
#         ])
        
#         # Concatenate both phases into a continuous trace
#         if show_trials and trial_data is not None:
#             # Concatenate trial data for both phases
#             combined_trial_data = np.concatenate([sorted_phase1_trial_data[i], sorted_phase2_trial_data[i]], axis=1)
#             ax.plot(combined_trial_data.T, "k-", linewidth=0.3, alpha=0.3)
        
#         # Concatenate average traces for both phases
#         combined_data = np.concatenate([sorted_phase1_data[i], sorted_phase2_data[i]])
#         ax.plot(combined_data, "k-", linewidth=2)
        
#         # Add vertical line to separate the two phases
#         phase1_len = len(sorted_phase1_data[i])
#         ax.axvline(phase1_len, color="k", linestyle="--", alpha=0.5, linewidth=1.5)
#         ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        
#         ax.set_xlim(0, len(combined_data))
#         ax.set_ylim(y_min_global, y_max_global)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f"{sorted_directions_deg[i]:.0f}°", fontsize=8)
#     return fig, ax_polar


# def plot_orientation_tuning_cartesian(responses, directions_deg, figsize=(8, 6), 
#                                      color='black', linewidth=2, marker='o', 
#                                      markersize=6, show_osi=True, osi_color='red',
#                                      title=None, xlabel='Orientation (degrees)', 
#                                      ylabel='Response', show_grid=True):
#     """
#     Plot orientation tuning curve in cartesian coordinates.
    
#     Creates a standard cartesian plot of orientation tuning (0-180°) commonly 
#     used in vision research papers. Shows OSI calculation points if requested.
    
#     Parameters:
#     -----------
#     responses : array-like
#         Response values for each direction
#     directions_deg : array-like
#         Direction values in degrees
#     figsize : tuple
#         Figure size (width, height)
#     color : str or tuple
#         Color for the main tuning curve
#     linewidth : float
#         Line width for the tuning curve
#     marker : str
#         Marker style for data points
#     markersize : float
#         Size of data point markers
#     show_osi : bool
#         Whether to show OSI calculation points
#     osi_color : str or tuple
#         Color for OSI calculation markers
#     title : str or None
#         Plot title
#     xlabel : str
#         X-axis label
#     ylabel : str
#         Y-axis label
#     show_grid : bool
#         Whether to show grid lines
        
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     ax : matplotlib.axes.Axes
#         Axes object
#     osi_info : dict
#         Dictionary containing OSI calculation results
#     """
#     from ..tuning_metrics import compute_orientation_tuning, compute_orientation_selectivity_index
    
#     # Get orientation tuning
#     orientation_data = compute_orientation_tuning(responses, directions_deg)
#     orientations = orientation_data['orientations']
#     orientation_responses = orientation_data['responses']
    
#     # Get OSI information
#     osi_info = compute_orientation_selectivity_index(responses, directions_deg)
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
#     # Sort orientations for smooth plotting
#     sort_indices = np.argsort(orientations)
#     sorted_orientations = orientations[sort_indices]
#     sorted_responses = orientation_responses[sort_indices]
    
#     # Plot main tuning curve
#     ax.plot(sorted_orientations, sorted_responses, color=color, linewidth=linewidth, 
#             marker=marker, markersize=markersize, label='Orientation tuning')
    
#     # Show OSI calculation points if requested
#     if show_osi and not np.isnan(osi_info['osi']):
#         # Mark preferred orientation
#         ax.scatter(osi_info['preferred_orientation'], osi_info['preferred_response'], 
#                   color=osi_color, s=markersize*15, marker='s', 
#                   label=f'Preferred ({osi_info["preferred_orientation"]:.0f}°)', 
#                   zorder=10, edgecolors='white', linewidth=1)
        
#         # Mark orthogonal orientation
#         ax.scatter(osi_info['orthogonal_orientation'], osi_info['orthogonal_response'], 
#                   color=osi_color, s=markersize*15, marker='^', 
#                   label=f'Orthogonal ({osi_info["orthogonal_orientation"]:.0f}°)', 
#                   zorder=10, edgecolors='white', linewidth=1)
        
#         # Add OSI text
#         ax.text(0.02, 0.98, f'OSI = {osi_info["osi"]:.3f}', 
#                 transform=ax.transAxes, fontsize=12, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                 verticalalignment='top')
    
#     # Styling
#     ax.set_xlim(0, 180)
#     ax.set_ylim(0, max(orientation_responses) * 1.1 if max(orientation_responses) > 0 else 1)
#     ax.set_xlabel(xlabel, fontsize=12)
#     ax.set_ylabel(ylabel, fontsize=12)
    
#     # Set x-axis ticks at common orientations
#     ax.set_xticks([0, 45, 90, 135, 180])
    
#     if title:
#         ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
#     if show_grid:
#         ax.grid(True, alpha=0.3, linestyle='--')
    
#     # Add legend if showing OSI
#     if show_osi and not np.isnan(osi_info['osi']):
#         ax.legend(loc='upper right', fontsize=10)
    
#     # Clean up spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_linewidth(1)
#     ax.spines['bottom'].set_linewidth(1)
    
#     plt.tight_layout()
    
#     return fig, ax, osi_info


# def plot_orientation_tuning_comparison(responses, directions_deg, figsize=(12, 5)):
#     """
#     Plot side-by-side comparison of polar and cartesian orientation tuning.
    
#     Creates a two-panel figure showing both polar and cartesian representations
#     of the same orientation tuning data for easy comparison.
    
#     Parameters:
#     -----------
#     responses : array-like
#         Response values for each direction
#     directions_deg : array-like
#         Direction values in degrees
#     figsize : tuple
#         Figure size (width, height)
        
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     axes : list
#         List containing [polar_ax, cartesian_ax]
#     osi_info : dict
#         Dictionary containing OSI calculation results
#     """
#     from ..tuning_metrics import compute_orientation_tuning, extract_orientation_vector
    
#     # Create figure with two subplots
#     fig = plt.figure(figsize=figsize, facecolor='white')
    
#     # Polar plot (left)
#     ax_polar = fig.add_subplot(121, projection='polar')
    
#     # Get orientation data
#     orientation_data = compute_orientation_tuning(responses, directions_deg)
#     orientations = orientation_data['orientations']
#     orientation_responses = orientation_data['responses']
    
#     # Convert to radians and double for polar plot
#     orientations_rad = np.deg2rad(orientations * 2)  # Double for 0-360° display
    
#     # Sort for smooth plotting
#     sort_indices = np.argsort(orientations)
#     sorted_orientations_rad = orientations_rad[sort_indices]
#     sorted_responses = orientation_responses[sort_indices]
    
#     # Close the polar plot
#     sorted_orientations_rad = np.append(sorted_orientations_rad, sorted_orientations_rad[0])
#     sorted_responses = np.append(sorted_responses, sorted_responses[0])
    
#     # Plot polar tuning curve
#     ax_polar.plot(sorted_orientations_rad, sorted_responses, 'b-', linewidth=2, marker='o', markersize=4)
#     ax_polar.fill(sorted_orientations_rad, sorted_responses, alpha=0.3, color='blue')
    
#     # Add orientation vector
#     orient_vector = extract_orientation_vector(responses, directions_deg)
#     if not np.isnan(orient_vector['angle']):
#         vector_angle_rad = np.deg2rad(orient_vector['angle'] * 2)
#         vector_magnitude = orient_vector['magnitude'] * max(sorted_responses)
#         ax_polar.annotate('', xy=(vector_angle_rad, vector_magnitude), xytext=(0, 0),
#                          arrowprops=dict(arrowstyle='->', color='orange', lw=3),
#                          zorder=10)
    
#     # Polar plot styling
#     ax_polar.set_ylim(0, max(sorted_responses) * 1.1)
#     ax_polar.set_title('Polar Representation', fontsize=14, fontweight='bold', pad=20)
#     ax_polar.set_theta_zero_location('E')
#     ax_polar.set_theta_direction(1)
    
#     # Cartesian plot (right)
#     ax_cartesian = fig.add_subplot(122)
#     fig_temp, ax_temp, osi_info = plot_orientation_tuning_cartesian(
#         responses, directions_deg, figsize=(6, 5), show_osi=True
#     )
    
#     # Copy cartesian plot to our subplot
#     for line in ax_temp.get_lines():
#         ax_cartesian.plot(line.get_xdata(), line.get_ydata(), 
#                          color=line.get_color(), linewidth=line.get_linewidth(),
#                          marker=line.get_marker(), markersize=line.get_markersize())
    
#     for collection in ax_temp.collections:
#         ax_cartesian.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
#                            c=collection.get_facecolors(), s=collection.get_sizes(),
#                            marker=collection.get_paths()[0] if collection.get_paths() else 'o')
    
#     # Copy cartesian styling
#     ax_cartesian.set_xlim(ax_temp.get_xlim())
#     ax_cartesian.set_ylim(ax_temp.get_ylim())
#     ax_cartesian.set_xlabel(ax_temp.get_xlabel(), fontsize=12)
#     ax_cartesian.set_ylabel(ax_temp.get_ylabel(), fontsize=12)
#     ax_cartesian.set_title('Cartesian Representation', fontsize=14, fontweight='bold', pad=20)
#     ax_cartesian.set_xticks([0, 45, 90, 135, 180])
#     ax_cartesian.grid(True, alpha=0.3, linestyle='--')
    
#     # Add OSI text
#     if not np.isnan(osi_info['osi']):
#         ax_cartesian.text(0.02, 0.98, f'OSI = {osi_info["osi"]:.3f}', 
#                          transform=ax_cartesian.transAxes, fontsize=12, fontweight='bold',
#                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                          verticalalignment='top')
    
#     # Clean up spines
#     ax_cartesian.spines['top'].set_visible(False)
#     ax_cartesian.spines['right'].set_visible(False)
    
#     plt.close(fig_temp)  # Close temporary figure
#     plt.tight_layout()
    
#     return fig, [ax_polar, ax_cartesian], osi_info


# def plot_orientation_tuning_comparison_phases(responses, directions_deg, phase_colors=None,
#                                             figsize=(15, 5)):
#     """
#     Plot side-by-side comparison of polar and cartesian orientation tuning for multiple phases.
    
#     Parameters:
#     -----------
#     responses : np.ndarray
#         Array of shape (n_directions, n_phases) containing response values
#     directions_deg : array-like
#         Direction values in degrees
#     phase_colors : list or None
#         Colors for each phase. If None, uses default colors
#     figsize : tuple
#         Figure size (width, height)
        
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     axes : list
#         List containing [polar_ax, cartesian_ax]
#     osi_info : dict
#         Dictionary containing OSI calculation results for each phase
#     """
#     from ..tuning_metrics import compute_orientation_tuning, extract_orientation_vector
    
#     # Set default phase colors
#     if phase_colors is None:
#         phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']
    
#     n_phases = responses.shape[1]
    
#     # Create figure with two subplots
#     fig = plt.figure(figsize=figsize, facecolor='white')
    
#     # Polar plot (left)
#     ax_polar = fig.add_subplot(121, projection='polar')
    
#     # Plot each phase on polar plot
#     for phase_i in range(n_phases):
#         phase_responses = responses[:, phase_i]
#         phase_color = phase_colors[phase_i % len(phase_colors)]
        
#         # Get orientation data
#         orientation_data = compute_orientation_tuning(phase_responses, directions_deg)
#         orientations = orientation_data['orientations']
#         orientation_responses = orientation_data['responses']
        
#         # Convert to radians and double for polar plot
#         orientations_rad = np.deg2rad(orientations * 2)  # Double for 0-360° display
        
#         # Sort for smooth plotting
#         sort_indices = np.argsort(orientations)
#         sorted_orientations_rad = orientations_rad[sort_indices]
#         sorted_responses = orientation_responses[sort_indices]
        
#         # Close the polar plot
#         sorted_orientations_rad = np.append(sorted_orientations_rad, sorted_orientations_rad[0])
#         sorted_responses = np.append(sorted_responses, sorted_responses[0])
        
#         # Plot polar tuning curve
#         linestyle = ['-', '--', '-.', ':'][phase_i % 4]
#         ax_polar.plot(sorted_orientations_rad, sorted_responses, color=phase_color, 
#                       linewidth=2, marker='o', markersize=3, linestyle=linestyle,
#                       label=f'Phase {phase_i+1}')
#         ax_polar.fill(sorted_orientations_rad, sorted_responses, alpha=0.2, color=phase_color)
        
#         # Add orientation vector
#         orient_vector = extract_orientation_vector(phase_responses, directions_deg)
#         if not np.isnan(orient_vector['angle']):
#             vector_angle_rad = np.deg2rad(orient_vector['angle'] * 2)
#             vector_magnitude = orient_vector['magnitude'] * max(sorted_responses)
#             ax_polar.annotate('', xy=(vector_angle_rad, vector_magnitude), xytext=(0, 0),
#                              arrowprops=dict(arrowstyle='->', color=phase_color, lw=2),
#                              zorder=10)
    
#     # Polar plot styling
#     ax_polar.set_ylim(0, np.max(responses) * 1.1)
#     ax_polar.set_title('Polar Representation', fontsize=14, fontweight='bold', pad=20)
#     ax_polar.set_theta_zero_location('E')
#     ax_polar.set_theta_direction(1)
#     ax_polar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
#     # Cartesian plot (right)
#     ax_cartesian = fig.add_subplot(122)
#     fig_temp, ax_temp, osi_info = plot_orientation_tuning_cartesian_phases(
#         responses, directions_deg, phase_colors=phase_colors, figsize=(8, 5), show_osi=True
#     )
    
#     # Copy cartesian plot to our subplot
#     for line in ax_temp.get_lines():
#         ax_cartesian.plot(line.get_xdata(), line.get_ydata(), 
#                          color=line.get_color(), linewidth=line.get_linewidth(),
#                          marker=line.get_marker(), markersize=line.get_markersize(),
#                          linestyle=line.get_linestyle(), label=line.get_label())
    
#     for collection in ax_temp.collections:
#         if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
#             ax_cartesian.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
#                                c=collection.get_facecolors(), s=collection.get_sizes(),
#                                marker='s' if hasattr(collection, '_marker') else 'o',
#                                edgecolors='white', linewidth=1, alpha=0.8)
    
#     # Copy cartesian styling
#     ax_cartesian.set_xlim(ax_temp.get_xlim())
#     ax_cartesian.set_ylim(ax_temp.get_ylim())
#     ax_cartesian.set_xlabel(ax_temp.get_xlabel(), fontsize=12)
#     ax_cartesian.set_ylabel(ax_temp.get_ylabel(), fontsize=12)
#     ax_cartesian.set_title('Cartesian Representation', fontsize=14, fontweight='bold', pad=20)
#     ax_cartesian.set_xticks([0, 45, 90, 135, 180])
#     ax_cartesian.grid(True, alpha=0.3, linestyle='--')
#     ax_cartesian.legend(loc='upper right', fontsize=10)
    
#     # Add OSI text for each phase
#     osi_text = []
#     for phase_i in range(n_phases):
#         phase_osi = osi_info[f'phase_{phase_i}']
#         if not np.isnan(phase_osi['osi']):
#             osi_text.append(f'Phase {phase_i+1}: OSI = {phase_osi["osi"]:.3f}')
    
#     if osi_text:
#         ax_cartesian.text(0.02, 0.98, '\n'.join(osi_text), 
#                          transform=ax_cartesian.transAxes, fontsize=10, fontweight='bold',
#                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                          verticalalignment='top')
    
#     # Clean up spines
#     ax_cartesian.spines['top'].set_visible(False)
#     ax_cartesian.spines['right'].set_visible(False)
    
#     plt.close(fig_temp)  # Close temporary figure
#     plt.tight_layout()
    
#     return fig, [ax_polar, ax_cartesian], osi_info
