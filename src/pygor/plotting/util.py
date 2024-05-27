import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def add_scalebar(x, y, length, ax = None, string = None, text_align = "mid", orientation = 'v', rotation = 0):
    if ax == None:
        ax = plt.gca()
    line_width = 8
    text_size = 15
    start = np.array([x + 1, y + 1])
    stop = np.array([x + 1, y + 1 + length]).T
    points=np.array([start, stop])
    rotation = rotation * -1
    if orientation == 'v':
        midpoint = ((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2)
        points = np.array(rotate(points, origin=midpoint, degrees=0 + rotation))
        line = matplotlib.lines.Line2D(points[:, 0], points[:, 1], c = 'k', lw = line_width)
        line = matplotlib.lines.Line2D(points[:, 0], points[:, 1], c = 'k', lw = line_width)
        mapping = {
            "close" : line.get_ydata()[0],
            "mid"   : np.sum(line.get_ydata()) / 2,
            "far"   : line.get_ydata()[-1],}
        text_x = (line.get_xdata()[0] + line.get_xdata()[0] * .1)
        text_y = mapping[text_align]
        text_points = np.array([text_x, text_y])
        print(text_points)
        #text_points = np.array(rotate(text_points, origin=midpoint, degrees = rotation))
        print(text_points)
        #test_x, test_y = new_points[:, 0], new_points[:, 1]
        ax.text(text_points[0], text_points[1], string, ha = "center", va = "center", size = text_size, rotation = 90 + rotation)
    if orientation == 'h':
        # # Calculate the midpoint of the line for the origin of rotation
        # midpoint = ((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2)
        midpoint = start
        # # Rotate the points
        points = np.array(rotate(points, origin=midpoint, degrees=90 + rotation))
        line = matplotlib.lines.Line2D(points[:, 0], points[:, 1], c = 'k', lw = line_width)
        mapping = {
            "close" : line.get_ydata()[0],
            "mid"   : np.sum(line.get_ydata()) / 2,
            "far"   : line.get_ydata()[-1],}
        ax.text((line.get_xdata()[0]) - (text_size + line_width) * np.cos(rotation), mapping[text_align] + (line_width) * np.cos(rotation), 
                string, ha = "center", va = "center", size = text_size, rotation = 0 + rotation)
    line.set_clip_on(False)

    ax.add_line(line)
    
    