from logging import exception
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)

def add_scalebar(x, y, length, ax = None, string = None, text_align = "mid", orientation = 'v', rotation = 0, text_size = None):
    if ax is None:
        ax = plt.gca()
        ax.set_clip_on(False)
    fig = ax.get_figure()
    line_width = 8
    line_width = 8
    if text_size == None: #"steal" text size from parent plot
        try:
            fig.get_axes()[0].xaxis.label.get_size()
        except AttributeError:
            try:
                fig.get_axes().yaxis.label.get_size()
            except: # Default to a constant
                text_size = 15

    start = np.array([x, y])
    if orientation == 'v':
        stop = np.array([x, y + length])
    else:  # 'h'
        stop = np.array([x + length, y])

    points = np.array([start, stop])

    midpoint = np.mean(points, axis=0)
    points = rotate(points, origin=midpoint, degrees=rotation)
    
    line = plt.Line2D(points[:, 0], points[:, 1], color='k', linewidth=line_width,
        clip_on=False, clip_box = False, mew = 0, solid_capstyle = "butt")    
    ax.add_line(line)

    if orientation == 'v':
        text_x = points[0, 0] - 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        text_y = {'close': points[0, 1], 'mid': midpoint[1], 'far': points[1, 1]}[text_align]
        if rotation == 180:
            text_x = points[0, 0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        rotation_angle = 90 + rotation
    else:  # 'h'
        text_x = {'close': points[0, 0], 'mid': midpoint[0], 'far': points[1, 0]}[text_align]
        text_y = points[0, 1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if rotation == 180:
            text_y = points[0, 1] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        rotation_angle = 0 + rotation

    ax.text(text_x, text_y, string, ha='center', va='center', fontsize=text_size, rotation=rotation_angle)

    # Adjust the subplot parameters to ensure there is enough space for the scalebar
    if orientation == 'v':
        plt.subplots_adjust(left=0.2)
    else:  # 'h'
        plt.subplots_adjust(bottom=0.2)