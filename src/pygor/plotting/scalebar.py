import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms


def rotate(p, origin=(0, 0), degrees=0):
    """
    Rotate a point (or an array of points) around a given origin by a given
    number of degrees

    Parameters:
    p : numpy.ndarray
        A point or array of points to rotate
    origin : tuple, optional
        The origin of rotation. Default is (0, 0)
    degrees : int, optional
        The number of degrees to rotate. Default is 0.

    Returns:
    numpy.ndarray
        The rotated points
    """
    # Convert the rotation angle from degrees to radians
    angle = np.deg2rad(degrees)

    # Create the 2D rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Ensure that the origin and the point(s) are 2D arrays
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    # Compute the rotated point(s) by applying the rotation matrix
    # to the vector connecting the origin to the point(s), then
    # translating the result back to the origin
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def add_scalebar(
    length,
    x=None,
    y=None,
    ax=None,
    string=None,
    orientation="v",
    flip_text=False,
    offset_modifier=1,
    text_size=None,
    line_width=None,
    transform=None,
    align_to_axis=False,
):
    """
    Adds a scalebar to a plot with improved positioning and alignment.

    Parameters:
    -----------
    length : float
        The length of the scalebar in the same units as the plot.
    x : float, optional
        The x-coordinate of the scalebar (as a fraction of the plot width).
        Default is None (auto-positioned based on orientation).
    y : float, optional
        The y-coordinate of the scalebar (as a fraction of the plot height).
        Default is None (auto-positioned based on orientation).
    ax : matplotlib.axes.Axes, optional
        The axes object to add the scalebar to. Default is None (current axes).
    string : str, optional
        The text to display along the scalebar. Default is None.
    orientation : str, optional
        The orientation of the scalebar ('v' for vertical, 'h' for horizontal).
        Default is 'v'.
    flip_text : bool, optional
        Whether to flip the text to the other side of the scalebar. Default is False.
    offset_modifier : float, optional
        Modifier for the text offset distance. Default is 1.
    text_size : float, optional
        The size of the text. Default is None (uses axes label size).
    line_width : float, optional
        The width of the scalebar line. Default is None (uses rcParams['axes.linewidth']).
    transform : str or Transform, optional
        Text transform to use. Default is None (data coordinates with offset).
    align_to_axis : bool, optional
        Whether to adjust positioning to account for scalebar length at axis edges. 
        Default is False.
        When True:
        - x=1.0 aligns the RIGHT end of horizontal scalebars to the right axis edge
        - x=0.0 aligns the LEFT end of horizontal scalebars to the left axis edge  
        - y=1.0 aligns the TOP end of vertical scalebars to the top axis edge
        - y=0.0 aligns the BOTTOM end of vertical scalebars to the bottom axis edge
        - Values between 0-1 still work as normal positioning with smart interpolation
    
    Examples:
    ---------
    # Basic usage
    add_scalebar(length=10, string='10 μm')
    
    # Align right end of horizontal scalebar to right axis edge
    add_scalebar(length=50, string='50 ms', orientation='h', x=1.0, align_to_axis=True)
    
    # Align top end of vertical scalebar to top axis edge  
    add_scalebar(length=20, string='20 units', orientation='v', y=1.0, align_to_axis=True)
    
    # Manual positioning with custom styling (traditional behavior)
    add_scalebar(length=20, x=0.8, y=0.1, string='20 units', line_width=3.0)
    """
    # If axes object is not provided, use the current axes
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    # If text size is not provided, set it to a default value
    if text_size is None:
        try:
            text_size = ax.get_figure().get_axes()[0].xaxis.label.get_size()
        except AttributeError:
            try:
                text_size = ax.get_figure().get_axes()[0].yaxis.label.get_size()
            except:
                text_size = 15

    # Get the size of the figure and the aspect ratio
    fig_width, fig_height = ax.get_figure().get_size_inches()
    fig_aspect = fig_width / fig_height
    ax_dpi = ax.get_figure().dpi

    # Set line width from rcParams if not specified (user can still override manually)
    if line_width is None:
        line_width = plt.rcParams['axes.linewidth']  # Use consistent line width with axes

    # Get the limits of the axes
    ax_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax_aspect = ax_width / ax_height
    ax_ylowerlim, ax_yupperlim = ax.get_ybound()
    ax_xlowerlim, ax_xupperlim = ax.get_xbound()

    # Set the x and y coordinates of the scalebar if not provided
    if x is None:
        x = -0.1 if orientation == "v" else 0
    if y is None:
        y = 0 if orientation == "v" else -0.1

    # Calculate the start and stop points of the scalebar
    if orientation == "v":
        x_ = x * ax_xupperlim + (1 - x) * ax_xlowerlim if x <= 1 else ax_xlowerlim + x * ax_width
        
        # When align_to_axis=True, adjust y positioning to account for scalebar length
        if align_to_axis and y >= 0:
            # For y=1.0, align the TOP of the scalebar to the top axis
            # For y=0.0, align the BOTTOM of the scalebar to the bottom axis
            if y == 1.0:
                # Align top of scalebar to top of axis
                start_y = ax_yupperlim - length
            elif y == 0.0:
                # Align bottom of scalebar to bottom of axis  
                start_y = ax_ylowerlim
            else:
                # Interpolate, but still account for length when near top
                base_y = y * ax_yupperlim + (1 - y) * ax_ylowerlim
                if y > 0.5:  # Near top, account for length
                    start_y = base_y - (y - 0.5) * 2 * length
                else:
                    start_y = base_y
        else:
            start_y = y * ax_yupperlim + (1 - y) * ax_ylowerlim
            
        start = np.array([x_, start_y])
        stop = np.array([x_, start_y + length])
        rotation_angle = 180
        text_rotation = -90
        
    else:  # 'h'
        y_ = y * ax_yupperlim + (1 - y) * ax_ylowerlim if y <= 1 else ax_ylowerlim + y * ax_height
        
        # When align_to_axis=True, adjust x positioning to account for scalebar length
        if align_to_axis and x >= 0:
            # For x=1.0, align the RIGHT end of the scalebar to the right axis
            # For x=0.0, align the LEFT end of the scalebar to the left axis
            if x == 1.0:
                # Align right end of scalebar to right edge of axis
                start_x = ax_xupperlim - length
            elif x == 0.0:
                # Align left end of scalebar to left edge of axis
                start_x = ax_xlowerlim
            else:
                # Interpolate, but still account for length when near right edge
                base_x = x * ax_xupperlim + (1 - x) * ax_xlowerlim
                if x > 0.5:  # Near right edge, account for length
                    start_x = base_x - (x - 0.5) * 2 * length
                else:
                    start_x = base_x
        else:
            start_x = x * ax_xupperlim + (1 - x) * ax_xlowerlim
            
        start = np.array([start_x, y_])
        stop = np.array([start_x + length, y_])
        rotation_angle = 0
        text_rotation = 0

    if flip_text is True:
        offset_flip = -1
        if orientation == "v":
            text_rotation += 180
    else:
        text_rotation += 0
        offset_flip = 1

    # Rotate the points of the scalebar
    points = np.array([start, stop])
    midpoint = np.mean(points, axis=0)
    points = rotate(points, origin=midpoint, degrees=rotation_angle)

    # Add the scalebar line and text to the axes
    line = plt.Line2D(points[:, 0], points[:, 1], color='k', linewidth=line_width,
                    clip_on=False, clip_box=ax.bbox, mew=1, solid_capstyle="butt")
    ax.add_line(line)
    
    # Calculate aesthetically pleasing, scale-robust text spacing using font metrics
    # Create a temporary text object to measure actual rendered size
    temp_text = ax.text(0, 0, string or 'Ag', fontsize=text_size, transform=ax.transData)
    temp_text.set_visible(False)
    
    # Get the actual rendered text height in data coordinates
    fig.canvas.draw()  # Ensure text is rendered for measurement
    text_bbox = temp_text.get_window_extent(fig.canvas.get_renderer())
    text_height_points = text_bbox.height
    temp_text.remove()  # Clean up temporary text
    
    # Calculate spacing based on rendered text height for true scale robustness
    # Be VERY generous with spacing - user needs substantial breathing room
    base_aesthetic_gap = text_height_points * 1.5  # Much more generous baseline
    line_clearance = line_width * 2.0  # Double the line width clearance
    
    # Use the larger of aesthetic gap or line clearance, then add substantial buffer
    primary_spacing = max(base_aesthetic_gap, line_clearance)
    additional_buffer = text_height_points * 0.8  # Large buffer for excellent typography
    
    text_offset_points = (primary_spacing + additional_buffer) * offset_modifier
    min_gap_fraction = 0.2  # Higher minimum for better appearance
    
    # Ensure minimum spacing to prevent overlap, but no maximum (let it scale naturally)
    text_offset_points = max(text_height_points * min_gap_fraction, text_offset_points)
    
    if orientation == "v":
        text_x = points[0, 0] 
        text_y = midpoint[1]
        # Offset text horizontally from the line
        dx = -text_offset_points / 72 * offset_flip  # Convert points to inches
        dy = 0
        text_ha = 'right' if not flip_text else 'left'
        text_va = 'center'  # Vertical text always centered on scalebar midpoint
    else:  # 'h' 
        text_x = midpoint[0]
        text_y = points[0, 1]
        # Offset text vertically from the line  
        dx = 0
        dy = -text_offset_points / 72 * offset_flip  # Convert points to inches
        text_ha = 'center'  # Horizontal text always centered on scalebar midpoint
        # CRITICAL: Proper alignment to prevent text from overlapping scalebar
        if not flip_text:
            # Text below scalebar: align TOP edge of text to the offset position
            text_va = 'top'  
        else:
            # Text above scalebar: align BOTTOM edge of text to the offset position  
            text_va = 'bottom'

    # Set up text transform
    if transform is None:
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        transform = ax.transData + offset
    elif transform == "axis":
        transform = ax.transAxes
    elif transform == "figure":
        transform = fig.transFigure
    elif transform == "data":
        transform = ax.transData
    
    # Add text with improved alignment
    if string is not None:
        ax.text(
            text_x,
            text_y,
            string,
            ha=text_ha,
            va=text_va,
            fontsize=text_size,
            rotation=text_rotation + rotation_angle,
            transform=transform,
            clip_on=False,
        )
