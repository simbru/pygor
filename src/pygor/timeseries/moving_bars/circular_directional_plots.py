import numpy as np
import matplotlib.pyplot as plt

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
        angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    for i, angle in enumerate(angles):
        # Calculate subplot position (start from top, go clockwise)
        x_center = 0.5 + 0.32 * np.cos(angle - np.pi/2)
        y_center = 0.5 + 0.32 * np.sin(angle - np.pi/2)
        
        # Create subplot
        ax = fig.add_axes([x_center - 0.06, y_center - 0.06, 0.12, 0.12])
        
        # Plot trace
        ax.plot(data[i], 'k-', linewidth=1.2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Clean styling
        ax.set_xlim(0, len(data[i]))
        y_range = np.max(np.abs(data[i])) * 1.1 if np.max(np.abs(data[i])) > 0 else 1
        ax.set_ylim(-y_range, y_range)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8f8f8')
        
        # Add direction label
        # if directions_list is not None:
        #     label_x = x_center + 0.08 * np.cos(angle - np.pi/2)
        #     label_y = y_center + 0.08 * np.sin(angle - np.pi/2)
        #     fig.text(label_x, label_y, f'{int(directions_list[i])}°', 
        #             ha='center', va='center', fontsize=10, fontweight='bold')
    
    return fig



def plot_directional_responses_circular_with_polar(data, directions_list=None, figsize=(10, 10), 
                                                  metric='peak', polar_kwargs=None):
    """
    Plot directional responses in a circular arrangement with central polar plot.
    
    Parameters:
    -----------
    data : np.ndarray
        Array of shape (n_directions, n_timepoints) containing response traces
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
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax_polar : matplotlib.axes.Axes
        The polar plot axes object
    """
    n_directions = data.shape[0]
    
    # Convert direction degrees to radians for positioning
    if directions_list is not None:
        angles = np.radians(directions_list)
        directions_deg = np.array(directions_list)
    else:
        angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        directions_deg = np.degrees(angles)
    
    # Calculate summary metric for each direction
    if metric == 'peak':
        values = np.array([np.max(np.abs(trace)) for trace in data])
    elif metric == 'auc':
        values = np.array([np.trapz(np.abs(trace)) for trace in data])
    elif metric == 'mean':
        values = np.array([np.mean(trace) for trace in data])
    elif metric == 'peak_positive':
        values = np.array([np.max(trace) for trace in data])
    elif metric == 'peak_negative':
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
        'color': '#2E8B57',  # Sea green instead of red
        'linewidth': 2,
        'marker': 'o',
        'markersize': 6,
        'alpha': 0.8,
    }
    if polar_kwargs:
        default_polar_kwargs.update(polar_kwargs)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Create central polar plot
    ax_polar = fig.add_axes([0.225, 0.225, 0.55, 0.55], projection='polar')
    
    # Plot polar data (need to append first point to close the circle)
    polar_angles = np.append(sorted_angles, sorted_angles[0])
    polar_values = np.append(sorted_values, sorted_values[0])
    
    ax_polar.plot(polar_angles, polar_values, **default_polar_kwargs)
    ax_polar.fill(polar_angles, polar_values, alpha=0.3, color=default_polar_kwargs['color'])
    
    # Style polar plot
    ax_polar.set_theta_zero_location('N')  # 0° at top
    ax_polar.set_theta_direction(-1)  # Clockwise
    if callable(metric):
        name = metric.__name__
    else:
        name = metric
    ax_polar.set_title(f'Directional Tuning\n({name.replace("_", " ").title()})', 
                    fontsize=12, fontweight='bold', pad=20)
    ax_polar.grid(True, alpha=0.3)
    
    # Add individual trace plots around the perimeter
    for i, angle in enumerate(angles):
        # Calculate subplot position (further out to accommodate central polar plot)
        radius = 0.38  # Increased radius to make room for central plot
        x_center = 0.5 + radius * np.cos(angle - np.pi/2)
        y_center = 0.5 + radius * np.sin(angle - np.pi/2)
        
        # Create subplot
        subplot_size = 0.08  # Slightly smaller to fit more around
        ax = fig.add_axes([x_center - subplot_size/2, y_center - subplot_size/2, 
                          subplot_size, subplot_size])
        
        # Plot trace
        ax.plot(data[i], 'k-', linewidth=1.2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Clean neutral background
        ax.set_facecolor('#f8f8f8')
        
        # Clean styling
        ax.set_xlim(0, len(data[i]))
        y_range = np.max(np.abs(data)) * 1.1 if np.max(np.abs(data)) > 0 else 1
        ax.set_ylim(-y_range, y_range)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add direction label (only for key directions to reduce clutter)
        # if i % 2 == 0 or len(directions_deg) <= 4:  # Show every other label, or all if few directions
        #     label_radius = radius + 0.07
        #     label_x = 0.5 + label_radius * np.cos(angle - np.pi/2)
        #     label_y = 0.5 + label_radius * np.sin(angle - np.pi/2)
        #     fig.text(label_x, label_y, f'{int(directions_deg[i])}°', 
        #             ha='center', va='center', fontsize=9, fontweight='bold')
    
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
