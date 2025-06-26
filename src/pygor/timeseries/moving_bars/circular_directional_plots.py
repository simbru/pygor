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
        if directions_list is not None:
            label_x = x_center + 0.08 * np.cos(angle - np.pi/2)
            label_y = y_center + 0.08 * np.sin(angle - np.pi/2)
            fig.text(label_x, label_y, f'{int(directions_list[i])}°', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    return fig