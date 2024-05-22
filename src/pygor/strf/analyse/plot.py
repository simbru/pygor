from typing import Callable
try:
    from collections import Iterable
except:
    from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pygor.plotting
import matplotlib
import copy
import scipy.stats
import warnings

label_mappings = {
    "588"  : "LWS",
    "478"  : "RH2",
    "422"  : "SWS2",
    "375"  : "SWS1",
}

color_mappings = {
    "588"  : pygor.plotting.fish_palette[0],
    "478"  : pygor.plotting.fish_palette[1],
    "422"  : pygor.plotting.fish_palette[2],
    "375"  : pygor.plotting.fish_palette[3],
}

title_mappings = {
    "area" : "Area (° vis. ang.$^2$)",
    "diam" : "Diameter (° vis. ang.)",
    "centdom" : "Spectral centroid (Hz)",
}

pval_mappings = {
    "area" : "pval_space",
    "diam" : "pval_space",
    "centdom" : "pval_time",
    "centneg" : "pval_time",
    "centpos" : "pval_time",
}

def plot_areas_vs(df, rowX : str, rowY : str, colour = None, ax : plt.axes = None, legend : bool = True, 
        labels : tuple = None,  strategy : str = 'singles', pval_map = True) -> (plt.figure, plt.axis):
    '''The function `plot_areas_vs` generates a plot comparing two variables from a
    DataFrame, with options for customizing colors, labels, and plotting strategies.
    
    Parameters
    ----------
    df
        The function `plot_areas_vs` takes several parameters to create a plot based on
    the input data. Here is an explanation of the parameters:
    rowX : str
        `rowX` is a string parameter that represents the column name in the DataFrame
    `df` that will be used as the x-axis values for plotting the areas.
    rowY : str
        The `rowY` parameter in the `plot_areas_vs` function represents the column name
    in the DataFrame `df` that will be plotted on the y-axis. It is a string that
    specifies the variable or statistic you want to visualize in relation to the
    `rowX` variable on the x-axis
    colour
        The `colour` parameter in the `plot_areas_vs` function allows you to specify the
    color or colors to be used in the plot. It can take different forms: str or tuple
    ax : plt.axes
        The `ax` parameter in the `plot_areas_vs` function is used to specify the
    matplotlib axes on which the plot will be drawn. If `ax` is not provided, a new
    set of axes will be created within the function.
    legend : bool, optional
        The `legend` parameter in the `plot_areas_vs` function is a boolean flag that
    determines whether a legend should be displayed on the plot. If `legend` is set to
    `True`, a legend will be included in the plot to help identify different elements
    of the visualization. If set to `
    labels : tuple
        The `labels` parameter in the `plot_areas_vs` function is used to specify the
    labels for the x and y axes in the plot. If `labels` is not provided, the function
    will attempt to determine appropriate labels based on the input query. If you want
    to customize the labels, you
    strategy : str, optional
        The `strategy` parameter in the `plot_areas_vs` function determines how the data
    will be plotted. It can take two values:
    
    Returns
    -------
        The function `plot_areas_vs` returns a tuple containing a matplotlib figure
    object and a matplotlib axis object.
    
    '''
    # Handle differential inputs
    if rowX.split('_')[0] != rowY.split('_')[0]:
        test_statistic = (rowX, rowY)
        filtered_df = df.filter(items = (test_statistic))
    #     raise AttributeError("Input queries are not the same statistic.")
    else:
        test_statistic = rowX.split('_')[0]
        filtered_df = df.filter(like = test_statistic)
    nm_X = rowX.split("_")[-1]
    nm_Y = rowY.split("_")[-1]
    # Handle colour input
    if isinstance(colour, tuple):
        cX, cY = colour
    if isinstance(colour, str):
        cX = colour
        cY = colour
    elif colour is None:
        if nm_X in color_mappings.keys() and nm_Y in color_mappings.keys():   
            cX = color_mappings[nm_X]
            cY = color_mappings[nm_Y]
        else:
            cX = "lightgrey"
            cY = "darkgrey"
    # rowY_df = df.filter(like = test_statistic)[[rowX, rowY]]
    # Generate queries for either and both
    existing_rows = filtered_df.columns
    query_str_x =  f'{rowX} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != rowX]) + " == 0"
    query_str_y =  f'{rowY} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != rowY]) + " == 0"
    if strategy == 'singles':
        rowX_df = filtered_df.query(query_str_x)[rowX] # if you want only singles
        rowY_df = filtered_df.query(query_str_y)[rowY]
    if strategy == 'pooled':
        rowX_df = filtered_df.query(f"{rowX} > 0 & {rowY} == 0")[rowX] # if you want all other combined cases
        rowY_df = filtered_df.query(f"{rowX} == 0 & {rowY} > 0")[rowY]
    if strategy != 'singles' and strategy != 'pooled':
        raise AttributeError("Strategy not recognised. Please use 'single' or 'pooled'.")
    both_df = df[[rowX, rowY]].query(f"{rowX} > 0 & {rowY} > 0").replace(0, np.nan) # contains all vals but nan empties
        # Get max value for axis limit
    max_val = np.nanmax(filtered_df)
    max_val_leeway = max_val * 1.05
    # max_val = np.nanmax([both_df[rowX], both_df[rowY]]) 
    # max_val_leeway = max_val + (max_val * 0.025)
    # min_val = np.nanmin([both_df[rowX], both_df[rowY]])
    # min_val_leeway = min_val - (max_val * 0.2)
    # Generate figure
    fig = plt.figure(figsize = (5, 5))
    # Generate gridspec
    gs = fig.add_gridspec(3, 3,  width_ratios=(4, 1, 1), height_ratios=(1,1,4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=.01, hspace=0.01)
    # Start building main subplot
    ax = fig.add_subplot(gs[2, 0])
    ax.set_xlabel(rowX)
    ax.set_ylabel(rowY)
    # Dynamically handle labels
    if labels is None:
        if nm_X in label_mappings.keys() and nm_Y in label_mappings.keys():   
            labels = (label_mappings[nm_X], label_mappings[nm_Y])
        else:
            labels = (rowX, rowY)
    if labels is not None and isinstance(labels, Iterable) is False: 
        raise ValueError("Labels must be a tuple or list-like of strings")
    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    # Plot the central figure
    ax.scatter(both_df[rowX], both_df[rowY], s = 20, c = "k", label = f"{labels[0]} and {labels[1]}")
    # Handle pval_mapping 
    if pval_map != None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_pvals = df.filter(like = "pval_space")[["pval_space_375", "pval_space_588"]]
            combined = scipy.stats.combine_pvalues(df_pvals, axis = 1)[1]
        scatter_color = df
    else:
        scatter_color = 'k'
    ax.plot(np.arange(0, max_val_leeway * 1.5, 1), np.arange(0, max_val_leeway * 1.5, 1), color = "grey", ls = "--", alpha = .5)
    # Plot X KDE and stripplot
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.kdeplot(both_df[rowX], color =  "k", ax = ax_histx)
    sns.kdeplot(rowX_df, color = cX, ax = ax_histx)
    ax_stackx = fig.add_subplot(gs[0, 0], sharex=ax)
    sns.stripplot(x = rowX_df, s = 5, c = cX, label = f"{labels[0]} {strategy}", ax = ax_stackx)
    try:
        ax_stackx.legend_.remove()
    except AttributeError:
        pass
    # Plot Y KDE and stripplot
    ax_histy = fig.add_subplot(gs[2, 1], sharey=ax)
    sns.kdeplot(y = both_df[rowY], color =  "k", ax = ax_histy)
    sns.kdeplot(y = rowY_df, color = cY, ax = ax_histy)
    ax_stacky = fig.add_subplot(gs[2, 2], sharey=ax)
    sns.stripplot(y = rowY_df, s = 5, c =cY, label = f"{labels[1]} {strategy}", ax = ax_stacky)
    try:
        ax_stacky.legend_.remove()
    except AttributeError:
        pass
    # Remove their axes for beauty
    ax_histx.axis("off")
    ax_histy.axis("off")
    ax_stackx.axis("off")
    ax_stacky.axis("off")
    # Fix axes limits
    ax.set_xlim(0, max_val_leeway)
    ax.set_ylim(0, max_val_leeway)
    if test_statistic in title_mappings.keys():
        ax.set_title(title_mappings[test_statistic], y = -.3)
    else:
        ax.set_title(test_statistic, y = -.3)
    # Create legend
    if legend is True:
        dots_labels = [ax.get_legend_handles_labels() for ax in fig.axes] # combine handles and labels
        dots, dot_labels = [sum(lol, []) for lol in zip(*dots_labels)]
        fig.legend(reversed(dots), reversed(dot_labels), bbox_to_anchor=(.9, .8))
    plt.show(block=False)
    plt.pause(.01)
    plt.close()
    return fig, ax

def plot_distribution(chroma_df, columns_like = "area", animate = True):
    data = chroma_df.replace(0, np.nan).filter(like = columns_like)
    present_columns = data.columns
    current_statistic = data.columns[0].split('_')[0]
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"
    # plot_settings['animation.embed_limit'] = 2**128
    with matplotlib.rc_context(rc=plot_settings):
        fig, ax = plt.subplots(figsize = (5, 5))
        max_val = np.max(data.replace(np.nan, 0).to_numpy())
        max_val = max_val + max_val * 0.025
        if current_statistic in title_mappings.keys():
            ax.set_ylabel(title_mappings[current_statistic])
        #ax.set_ylim(bottom = 6, top = max_val)
        sns.boxplot(data, 
            palette=reversed(pygor.plotting.custom.fish_palette))
        sns.stripplot(data, 
            palette=r"dark:k", 
            alpha = .5, ax = ax, zorder = 1, linewidth=.5, edgecolors = 'k',size = 3, jitter =.2)
        ax.set_title("Cone input", y = -.15)
        labels = ax.get_xticklabels()
        for n, i in enumerate(labels):
            if present_columns[n].split('_')[-1] in label_mappings:
                labels[n] = label_mappings[present_columns[n].split('_')[-1]]
        ax.set_xticklabels(labels)
        if animate:
            initial_line = data.iloc[0]
            initial_line = initial_line[initial_line > 0]
            ax.plot(initial_line, color = "grey", alpha = .5, marker = "o", ms = 6, 
                    zorder = 2, label = "ROI", mew = 1, mec = 'k',)
            legend = ax.legend()
            def line_sweep(num):
                each_line = data.iloc[num].to_numpy()
                # each_line = np.ma.masked_where(each_line == 0, each_line)
                each_line = each_line[each_line > 0]
                ax.get_lines()[0].set_xdata(np.where(each_line > 0)[0])
                ax.get_lines()[0].set_ydata(each_line)
                legend.get_texts()[0].set_text(f"ROI {num}",) #Update label each at frame
            animation = matplotlib.animation.FuncAnimation(fig, line_sweep, frames = len(data), interval = 150, repeat_delay = 500)
            animation.to_html5_video()
            plt.close()
            return animation
        
