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

def add_scalebar(length, x=None, y=None, ax=None, string=None, text_align="mid", orientation='v', rotation=0, text_size=None):
    if ax is None:
        ax = plt.gca()
        ax.set_clip_on(False)
    fig = ax.get_figure()

    if text_size is None:  # "steal" text size from parent plot
        try:
            text_size = fig.get_axes()[0].xaxis.label.get_size()
        except AttributeError:
            try:
                text_size = fig.get_axes()[0].yaxis.label.get_size()
            except:  # Default to a constant
                text_size = 15

    # Get fig stats
    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()
    fig_aspect = fig_width / fig_height
    fig_dpi = fig.dpi
    # Get ax stats
    ax_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax_aspect = ax_width / ax_height
    ax_ylowerlim, ax_yupperlim = ax.get_ybound()
    ax_xlowerlim, ax_xupperlim = ax.get_xbound()

    if orientation == 'v':
        if x is None:
            x = -.1
        if y is None:
            y = 0
        x_ = (x * ax_xupperlim) + (ax_xlowerlim * (1 - x) / 1)
        start = np.array([x_, y * ax_yupperlim + ax_ylowerlim])
        stop = np.array([x_, y * ax_yupperlim + ax_ylowerlim + length])
    else:  # 'h'
        if x is None:
            x = 0
        if y is None:
            y = -.1
        y_ = (y * ax_yupperlim) + (ax_ylowerlim * (1 - y) / 1)
        start = np.array([x * ax_xupperlim + ax_xlowerlim, y_])
        stop = np.array([x * ax_xupperlim + ax_xlowerlim + length, y_])

    points = np.array([start, stop])

    midpoint = np.mean(points, axis=0)
    points = rotate(points, origin=midpoint, degrees=rotation)
    
    line_width = 2 * np.max([fig_width, fig_height])
    myster_offset = 0.05
    if orientation == 'v':
        text_x = points[0, 0] - myster_offset * (ax.get_xlim()[1] - ax.get_xlim()[0])
        text_y = {'close': points[0, 1], 'mid': midpoint[1], 'far': points[1, 1]}[text_align]
        if rotation == 180:
            text_x = points[0, 0] + myster_offset * (ax.get_xlim()[1] - ax.get_xlim()[0])
        rotation_angle = 90 + rotation
        # # # Offset the text to avoid clipping
        # text_x_offset =  1 / ((line_width + text_size) / ax_width)
        # if rotation == 0:
        #     text_x -= text_x_offset
        # elif rotation == 180:
        #     text_x += text_x_offset
    else:  # 'h'
        text_x = {'close': points[0, 0], 'mid': midpoint[0], 'far': points[1, 0]}[text_align]
        text_y = points[0, 1] - 1/text_size * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if rotation == 180:
            text_y = points[0, 1] + myster_offset * (ax.get_ylim()[1] - ax.get_ylim()[0])
        rotation_angle = 0 + rotation
        # # Offset the text to avoid clipping
        # text_y_offset =  1 / ((line_width + text_size) / ax_height) #+ (line_width * 1/ax_height)  #+ (text_size + line_width) * 2 / fig_dpi
        # if rotation == 0:
        #     text_y -= text_y_offset
        # elif rotation == 180:
        #     text_y += text_y_offset

    line = plt.Line2D(points[:, 0], points[:, 1], color='k', linewidth=line_width,
        clip_on=False, clip_box=False, mew=0, solid_capstyle="butt")    
    ax.add_line(line)

    ax.text(text_x, text_y, string, ha='center', va='center', fontsize=text_size, rotation=rotation_angle)

    # # Adjust the subplot parameters to ensure there is enough space for the scalebar
    # if orientation == 'v':
    #     if x <= 0:
    #         plt.subplots_adjust(left=0.2)
    #     if x >= 1:
    #         plt.subplots_adjust(right=0.8) 
    # else:  # 'h'
    #     if y <= 0:
    #         plt.subplots_adjust(bottom=0.2)
    #     if y >= 1:
    #         plt.subplots_adjust(top=0.8)
