from logging import exception
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Ensure that the origin and the point(s) are 2D arrays
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    # Compute the rotated point(s) by applying the rotation matrix
    # to the vector connecting the origin to the point(s), then
    # translating the result back to the origin
    return np.squeeze((R @ (p.T - o.T) + o.T).T)



def add_scalebar(length, x=None, y=None, ax=None, string=None,
                text_align="mid", orientation='v', rotation=0,
                text_size=None, line_width=None):
    """
    Adds a scalebar to a plot.

    Parameters:
    length : float
        The length of the scalebar in the same units as the plot.
    x : float, optional
        The x-coordinate of the scalebar (as a fraction of the plot width).
        Default is None.
    y : float, optional
        The y-coordinate of the scalebar (as a fraction of the plot height).
        Default is None.
    ax : matplotlib.axes.Axes, optional
        The axes object to add the scalebar to. Default is None.
    string : str, optional
        The text to display along the scalebar. Default is None.
    text_align : str, optional
        The alignment of the text along the scalebar ('close', 'mid', or 'far').
        Default is 'mid'.
    orientation : str, optional
        The orientation of the scalebar ('v' for vertical, 'h' for horizontal).
        Default is 'v'.
    rotation : float, optional
        The rotation of the scalebar in degrees. Default is 0.
    text_size : float, optional
        The size of the text along the scalebar. Default is None.
    line_width : float, optional
        The width of the scalebar line. Default is None.
    """
    if ax is None:
        ax = plt.gca()

    if text_size is None:
        try:
            text_size = ax.get_figure().get_axes()[0].xaxis.label.get_size()
        except AttributeError:
            try:
                text_size = ax.get_figure().get_axes()[0].yaxis.label.get_size()
            except:
                text_size = 15

    fig_width, fig_height = ax.get_figure().get_size_inches()
    fig_aspect = fig_width / fig_height
    if line_width is None:
        line_width = 1 * np.max([fig_width, fig_height])

    if orientation == 'v':
        offset = 0.025 / fig_aspect
    else:  # 'h'
        offset = 0.025 / np.reciprocal(fig_aspect)

    ax_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax_aspect = ax_width / ax_height
    ax_ylowerlim, ax_yupperlim = ax.get_ybound()
    ax_xlowerlim, ax_xupperlim = ax.get_xbound()

    if x is None:
        x = -.1 if orientation == 'v' else 0
    if y is None:
        y = 0 if orientation == 'v' else -.1

    if orientation == 'v':
        x_ = x * ax_xupperlim + (1 - x) * ax_xlowerlim
        start = np.array([x_, y * ax_yupperlim + ax_ylowerlim])
        stop = np.array([x_, y * ax_yupperlim + ax_ylowerlim + length])
        rotation_angle = 90 + rotation
    else:  # 'h'
        y_ = y * ax_yupperlim + (1 - y) * ax_ylowerlim
        start = np.array([x * ax_xupperlim + ax_xlowerlim, y_])
        stop = np.array([x * ax_xupperlim + ax_xupperlim + length, y_])
        rotation_angle = rotation

    points = np.array([start, stop])
    midpoint = np.mean(points, axis=0)
    points = rotate(points, origin=midpoint, degrees=rotation)

    if orientation == 'v':
        text_x = points[0, 0] - offset * (ax.get_xlim()[1] - ax.get_xlim()[0])
        text_y = {'close': points[0, 1], 'mid': midpoint[1], 'far': points[1, 1]}[text_align]
        if rotation == 180:
            text_x = points[0, 0] + offset * (ax.get_xlim()[1] - ax.get_xlim()[0])
    else:  # 'h'
        text_x = {'close': points[0, 0], 'mid': midpoint[0], 'far': points[1, 0]}[text_align]
        text_y = points[0, 1] - offset * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if rotation == 180:
            text_y = points[0, 1] + offset * (ax.get_ylim()[1] - ax.get_ylim()[0])

    line = plt.Line2D(points[:, 0], points[:, 1], color='k', linewidth=line_width,
                    clip_on=False, clip_box=True, mew=1, solid_capstyle="butt")
    ax.add_line(line)
    ax.text(text_x, text_y, string, ha='center', va='center', fontsize=text_size, rotation=rotation_angle)
