import matplotlib

uv_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv", [(0.0, "white"), (1, "violet")])
uv_r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv_r", [(0.0, "violet"), (1, "white")])
chromatic_palette = ['red', 'green', 'blue', 'violet'] # R G B UV
rgbuv_palette = ["r", "g", "b", "violet"]
chromatic_hues = ["R", "G", "B", "UV"]
achromatic_palette = ['black', 'brown'] # BW BWnoUV 
achromatic_hues = ["BW", "BWnoUV"]
all_palette = chromatic_palette + achromatic_palette
all_hues = ["R", "G", "B", "UV", "BW", "BWnoUV"]
rguv_hues = ["R", "G", "UV"]
nanometers = ["588", "478", "422", "375"]
fish_palette = ["#ef8e00", "teal","#5600fe", "fuchsia"]
polarity_palette = ["lightslategray", "k", "tab:olive", "snow"] # [off, on, opp, mix]
# polarity_palette = ["lightsteelblue", "lightcoral", "lightpink"]

compare_conditions = {2 : ["grey", "tomato"],
                3 : ["grey", "tomato", "darkblue"],
                4 : ["grey", "tomato", "darkblue", "purple"],
                5 : ["grey", "tomato", "darkblue", "purple", "darkgreen"]}

# red_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey", "white","#ef8e00","darkred"])
# green_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","mediumaquamarine","teal"])
# blue_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","#5600fe","#4400cb"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","fuchsia","#b22cb2"])
# red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#ef8e00"])
# green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "teal"])
# blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#5600fe"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#ff00ff"])
# red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#ef8e00"])
# green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "teal"])
# blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#5600fe"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#ff00ff"])
red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#ef8e00"])
green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "teal"])
blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#5600fe"])
violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#ff00ff"])


maps_concat = [red_map, green_map, blue_map, violet_map]

def label_ax_colour(ax, x = 0.1, y = .9, marker = 'o', colour = fish_palette[0], relative_axis = True, **kwargs):
        '''The function `label_ax_colour` labels a point on an axis with a marker of a specified color and
        size, allowing for customization of the position and appearance.
        
        Parameters
        ----------
        ax
                The `ax` parameter in the `label_ax_colour` function is the axis object on which you want to plot
        the marker. It is typically obtained by calling `plt.gca()` in matplotlib or by creating a subplot
        using `plt.subplots()`.
        x
                The `x` parameter in the `label_ax_colour` function is used to specify the x-coordinate where the
        marker will be placed on the plot.
        y
                The `y` parameter in the `label_ax_colour` function is used to specify the y-coordinate of the
        point where the marker will be placed on the plot. It determines the vertical position of the marker
        within the plot area.
        marker, optional
                The `marker` parameter in the `label_ax_colour` function is used to specify the marker style for
        the scatter plot. In this function, the `marker` parameter is set to 'o' by default, which
        represents a circle marker. However, you can customize this parameter by passing different marker
        colour
                The `colour` parameter in the `label_ax_colour` function is used to specify the color of the marker
        that will be plotted on the axis. It is set to `fish_palette[0]` by default, which suggests that it
        is likely a predefined color from a palette named `fish_palette
        relative_axis, optional
                The `relative_axis` parameter in the `label_ax_colour` function determines the coordinate system in
        which the marker will be placed. It can take on the following values:
        

        **kwargs are passed to the `ax.scatter` function
        '''
        fig = ax.get_figure()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if 's' not in kwargs or 'size' not in kwargs:
                width, height = bbox.width, bbox.height
                width *= fig.dpi
                height *= fig.dpi
                kwargs["s"] = (width * height) / 100
        if relative_axis is True:
                transform_axis = ax.transAxes
        if relative_axis is False:
                transform_axis = ax.transData
        if relative_axis == 'x':
                transform_axis = ax.get_yaxis_transform()
        if relative_axis == 'y':
                transform_axis = ax.get_xaxis_transform()
        if relative_axis == "fig":
                transform_axis = ax.transFigure
        else: 
                AttributeError("relative_axis must be bool or str 'x', 'y' or 'fig'")    
        ax.scatter(x, y, marker = marker, c = colour, transform = transform_axis, **kwargs)