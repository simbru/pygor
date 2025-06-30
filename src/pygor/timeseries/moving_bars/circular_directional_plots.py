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
        # Calculate subplot position (start from top, go clockwise)
        x_center = 0.5 + 0.32 * np.cos(angle - np.pi / 2)
        y_center = 0.5 + 0.32 * np.sin(angle - np.pi / 2)

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
    data_crop=None
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
        data = data[:, data_crop[0]:data_crop[1]]
        if trial_data is not None:
            trial_data = trial_data[:, :, data_crop[0]:data_crop[1]]

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
    ax_polar.set_theta_zero_location("N")  # 0° at top
    ax_polar.set_theta_direction(-1)  # Clockwise
    ax_polar.set_title(
        f"Directional Tuning\n({metric.replace('_', ' ').title()})",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax_polar.grid(True, alpha=0.3)

    # Add individual trace plots around the perimeter
    for i, angle in enumerate(angles):
        # Calculate subplot position (further out to accommodate central polar plot)
        radius = 0.38  # Increased radius to make room for central plot
        x_center = 0.5 + radius * np.cos(angle - np.pi / 2)
        y_center = 0.5 + radius * np.sin(angle - np.pi / 2)

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

        # Plot trace
        if show_trials:
            ax.plot(trial_data[i].T, "k-", linewidth=.5, alpha=0.33)
        ax.plot(data[i], "k-", linewidth=1)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        # Clean neutral background
        # ax.set_facecolor("#f8f8f8")

        # Clean styling
        ax.set_xlim(0, len(data[i]))
        y_range = np.max(np.abs(data)) * 1.25 if np.max(np.abs(data)) > 5 else 5
        ax.set_ylim(-y_range, y_range)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=False, bottom=False, right=True, top=True)

        
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
