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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend_handler import HandlerTuple

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
        labels : tuple = None,  strategy : str = 'singles', pval_map = True,
        axlim = None) -> (plt.figure, plt.axis):
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
        filtered_df = df.filter(regex = test_statistic  + '_' + r"\d+")
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
    if strategy == 'singles':
        query_str_x =  f'{rowX} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != rowX]) + " == 0"
        query_str_y =  f'{rowY} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != rowY]) + " == 0"
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
    ## Handle pval_mapping 
    if pval_map:
        if f'pval_combined_{nm_Y}' in df.columns and f'pval_combined_{nm_X}' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_pairs = df.iloc[both_df.index].filter(items = (f'pval_combined_{nm_X}', f'pval_combined_{nm_Y}'))
                combined_pvals = scipy.stats.combine_pvalues(pval_pairs, axis = 1, method = 'pearson')[1]
                scatter_colour = combined_pvals
            scatter = ax.scatter(both_df[rowX], both_df[rowY], s = 20, c = scatter_colour, 
                                cmap = "Greys_r", label = f"{labels[0]} and {labels[1]}",
                                alpha = .33, edgecolors = "k")
    else:
        scatter = ax.scatter(both_df[rowX], both_df[rowY], s = 20, c = 'k', 
                            label = f"{labels[0]} and {labels[1]}",
                            alpha = 0.33)
#    plt.colorbar(scatter, cax = ax, orientation = "horizontal", pad = 2)
    
    
    ax.plot(np.arange(0, max_val_leeway * 1.5, 1), np.arange(0, max_val_leeway * 1.5, 1), color = "grey", ls = "--", alpha = .5)
    # Plot X KDE and stripplot
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.kdeplot(both_df[rowX], color =  "k", ax = ax_histx)
    sns.kdeplot(rowX_df, color = cX, ax = ax_histx)
    ax_stackx = fig.add_subplot(gs[0, 0], sharex=ax)
    sns.stripplot(x = rowX_df, s = 5, c = cX, label = f"{labels[0]} {strategy}", ax = ax_stackx, alpha = .33, edgecolor='gray', linewidth=.6)
    try:
        ax_stackx.legend_.remove()
    except AttributeError:
        pass
    # Plot Y KDE and stripplot
    ax_histy = fig.add_subplot(gs[2, 1], sharey=ax)
    sns.kdeplot(y = both_df[rowY], color =  "k", ax = ax_histy)
    sns.kdeplot(y = rowY_df, color = cY, ax = ax_histy)
    ax_stacky = fig.add_subplot(gs[2, 2], sharey=ax)
    sns.stripplot(y = rowY_df, s = 5, c =cY, label = f"{labels[1]} {strategy}", ax = ax_stacky, alpha = .33, edgecolor='gray', linewidth=.6)
    try:
        ax_stacky.legend_.remove()
    except AttributeError:
        pass
    # if pval_map:
    #     fig.colorbar(scatter, ax = ax_stacky, orientation="vertical", pad=0)
    # Remove their axes for beauty
    ax_histx.axis("off")
    ax_histy.axis("off")
    ax_stackx.axis("off")
    ax_stacky.axis("off")
    # Fix axes limits
    if axlim is None:
        ax.set_xlim(0, max_val_leeway)
        ax.set_ylim(0, max_val_leeway)
    else:
        ax.set_xlim(axlim[0], axlim[1])
        ax.set_ylim(axlim[0], axlim[1])
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
        
def ipl_summary_chroma(roi_df, numcolours = 4, figsize = (12, 7)):
    polarities = [-1, 1]
    colours = ["R", "G", "B", "UV"]
    fig, ax = plt.subplots(2, 4, figsize = figsize, sharex = True, sharey=True)
    bins = 10
    # sns.set_style("whitegrid")
    for n, i in enumerate(polarities):
        for m, j in enumerate(colours):
            hist_vals_per_condition = np.histogram(roi_df.query(f"polarity ==  {i} & colour == '{j}'")["ipl_depths"], bins = bins, range = (0, 100))[0]
            hist_vals_population = np.histogram(roi_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins, range = (0, 100))[0]
            # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
            percentages = hist_vals_per_condition  / np.sum(hist_vals_population) * 100
            # percentages = hist_vals_per_condition
            ax[n, m].barh(np.arange(0, 100, 10), width= percentages, height=10, color = pygor.plotting.custom.fish_palette[m], edgecolor="black", alpha = 0.75)        
            ax[n, m].grid(False)
            ipl_border = 55
            ax[n, m].axhline(ipl_border, c = "k", ls = "--")
            # ax[n, m].get_xaxis().set_visible(False)
            # ax[n, m].spines["bottom"].set_visible(False)
            if m == 0:
                ax[n, m].set_ylabel("IPL depth (%)")
                ax[n, m].text(x = ax[n, m].get_xlim()[1] - ax[n, m].get_xlim()[1] * 0.175, y = ipl_border + 5, va = 'center', s = "OFF", size = 10)
                ax[n, m].text(x = ax[n, m].get_xlim()[1] - ax[n, m].get_xlim()[1] * 0.175, y = ipl_border - 5, va = 'center', s = "ON", size = 10)
                if i == -1:
                    ax[n, m].set_title("OFF", weight = "bold", c = "grey", loc = "left")
                if i == 1:
                    ax[n, m].set_title("ON", weight = "bold", loc = "left")
            #ax[0, m].set_title(custom.nanometers[m] + "nm", size = 12)
            num_cells = int(len(np.unique(roi_df.index)) / numcolours)
            ax[1, 0].set_xlabel(f"Percentage by colour (n = {num_cells})", size = 10)
    plt.show()
    return fig, ax 

def ipl_summary_polarity_roi(roi_df, numcolours = 4, figsize = (8, 4), polarities = [-1, 1]):
    skip_df = roi_df
    # roi_df.iloc[roi_df["ipl_depths"].dropna().index]
    fig, axs = plt.subplots(1, len(polarities), sharex = True, sharey=True, figsize = figsize)
    bins = 10
    titles = ["OFF", "ON", "Mixed polarity", "other"]
    palette = pygor.plotting.custom.polarity_palette
    palette.append('r')
    # sns.set_style("whitegrid")
    tot_sum = 0
    for n, ax in enumerate(axs.flatten()):
        query_df = skip_df.query(f"polarity ==  {polarities[n]}")["ipl_depths"]
        hist = np.histogram(query_df, bins = bins, range = (0, 100))
        tot_sum += len(query_df)
        hist_vals_per_condition = hist[0]
        hist_vals_population = np.histogram(skip_df["ipl_depths"], bins = bins, range = (0, 100))[0]
        # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
        percentages = hist_vals_per_condition  / np.sum(hist_vals_population) * 100
        #percentages = hist_vals_per_condition
        ax.barh(np.arange(0, 100, 10), width= percentages, height=10, color = palette[n], edgecolor="black", alpha = 0.75)        
        ax.set_title(titles[n], size = 12)
        ipl_border = 55
        ax.axhline(ipl_border, c = "k", ls = "--")
    axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border + 5, va = 'center', s = "OFF", size = 10)
    axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border - 5, va = 'center', s = "ON", size = 10)
    num_cells = len(skip_df.index)
    print(num_cells, tot_sum)
    axs[0].set_xlabel(f"Percentage by polarity (n = {num_cells})", size = 10)
    plt.show()
    return fig, ax

def ipl_summary_polarity_chroma(chroma_df, numcolours = 4, figsize = (8, 4), cat_pol = ['off', 'on']):
    polarities = ['off', 'on']#, 'opp']
    fig, axs = plt.subplots(1, len(polarities), sharex = True, sharey=True, figsize = (4.8, 2.5))
    bins = 10
    titles = ["OFF", "ON", "Opponent", "empty", "mix"]
    palette = pygor.plotting.custom.polarity_palette
    palette.append('r')
    # sns.set_style("whitegrid")
    tot_sum = 0
    for n, ax in enumerate(axs.flatten()):
        query_df = chroma_df.query(f"cat_pol ==  '{polarities[n]}'")["ipl_depths"]
        hist = np.histogram(query_df, bins = bins, range = (0, 100))
        tot_sum += len(query_df)
        hist_vals_per_condition = hist[0]
        hist_vals_population = np.histogram(chroma_df["ipl_depths"], bins = bins, range = (0, 100))[0]
        # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
        percentages = hist_vals_per_condition  / np.sum(hist_vals_population) * 100
        #percentages = hist_vals_per_condition
        ax.barh(np.arange(0, 100, 10), width= percentages, height=10, color = palette[n], edgecolor="black", alpha = 0.75)        
        ax.set_title(titles[n], size = 12)
        ipl_border = 55
        ax.axhline(ipl_border, c = "k", ls = "--")
    axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border + 5, va = 'center', s = "OFF", size = 10)
    axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border - 5, va = 'center', s = "ON", size = 10)
    num_cells = len(chroma_df.index)
    print(num_cells, tot_sum)
    axs[0].set_xlabel(f"Percentage by polarity (n = {num_cells})", size = 10)
    plt.show()
    return fig, ax

def _multi_vs_single_vert(df, metric, subset_list, colour = None, labels = None) -> (plt.figure, plt.axis):
    # Generate filtered dataframe by metric
    metric_df = df.filter(like = metric)
    # Generate figures according to subset list
    fig, ax = plt.subplots(1, 4, figsize = (5, 5), sharex=False, sharey=True,
                        gridspec_kw={'width_ratios': [1.25, 1, 1, 1.25], 'wspace': 0})
    ax[0].get_shared_x_axes().join(ax[0], ax[3])
    # Loop through subsets and plot data
    labels_used = []
    for n, subset in enumerate(subset_list):
        existing_rows = metric_df.columns
        current_subset = '_'.join([metric, subset])
        query_str =  f'{metric}_{subset} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != current_subset]) + " == 0"
        print(query_str)
        singles = metric_df.query(query_str)[current_subset] # if you want only singles
        pooled = metric_df.query(f"{current_subset} > 0")[current_subset] # if you want all other combined cases        
        pooled = pooled.drop(singles.index)
        # # Handle colour input
        if isinstance(colour, tuple):
            c = colour
        if isinstance(colour, str):
            c = colour
        elif colour is None:
            if subset in color_mappings.keys() and subset in color_mappings.keys():   
                c = color_mappings[subset]
            else:
                c = "lightgrey"
        # Dynamically handle labels
        if labels is not None and isinstance(labels, Iterable) is False: 
            raise ValueError("Labels must be a tuple or list-like of strings")
        if labels is None:
            if subset in label_mappings.keys():   
                label = label_mappings[subset]
        else:
            label = labels[n]
        labels_used.append(label)
        alpha_val = .66        
        sns.kdeplot(y = singles, ax = ax[1], color = c, alpha = alpha_val, lw = 2)
        sns.kdeplot(y = pooled, ax = ax[2], color = c, alpha = alpha_val, lw = 2)
        sns.stripplot(y = singles, x = n, color = c, ax = ax[0], jitter=True, 
                    alpha = alpha_val/3, orient = 'v', edgecolor = 'k', linewidth = .5,
                    marker = 'D', label = label)
        sns.stripplot(y = pooled, x = n, color = c, ax = ax[3], jitter=True,
                    alpha = alpha_val/3, orient = 'v', edgecolor = 'k', linewidth = .5,
                    marker = 'o', label = label)
    if metric in title_mappings.keys():
        ax[0].set_ylabel(title_mappings[metric])
    else:
        ax[0].set_ylabel(metric)
    # Take care of eaxes
    for cax in ax:
        cax.set_xticklabels([])
        cax.set_xlabel("")
        cax.spines.right.set_visible(False)
        cax.spines.left.set_visible(False)
        cax.grid(True, axis = 'y')
    ax[0].spines.left.set_visible(True)
    ax[-1].spines.right.set_visible(True)
    ax[2].spines.left.set_visible(True)
    if ax[1].get_xlim()[-1] > ax[2].get_xlim()[-1]:
        ax[2].set_xlim(ax[1].get_xlim())
    if ax[1].get_xlim()[-1] < ax[2].get_xlim()[-1]:
        ax[1].set_xlim(ax[2].get_xlim())
    ax[1].invert_xaxis()    
    ax[0].set_title("Singular-colour RFs", loc = 'left')
    ax[3].set_title("Multi-colour RF RFs", loc = 'right')
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[-1].get_legend_handles_labels()
    hand_labl = np.array([[handles1, labels1], [handles2, labels2]])
    handles = [(i, j) for i, j in zip(handles1, handles2)]
    print(handles1[0], labels1)
    ax[0].legend(handles, labels_used, handler_map={tuple: HandlerTuple(ndivide=None)})
    l = ax[-1].legend()
    l.remove()
    plt.show()

def _multi_vs_single_horz(df, metric, subset_list, colour = None, labels = None) -> (plt.figure, plt.axis):
    # Generate filtered dataframe by metric
    metric_df = df.filter(like = metric)
    # Generate figures according to subset list
    fig, ax = plt.subplots(4, 1, figsize = (5, 5), sharey=False, sharex=True,
                        gridspec_kw={'height_ratios': [1.25, 1, 1, 1.25], 'hspace': 0})
    # Loop through subsets and plot data
    labels_used = []
    for n, subset in enumerate(subset_list):
        existing_rows = metric_df.columns
        current_subset = '_'.join([metric, subset])
        query_str =  f'{metric}_{subset} > 0 & ' + ' == 0 & '.join([i for i in existing_rows if i != current_subset]) + " == 0"
        print(query_str)
        singles = metric_df.query(query_str)[current_subset] # if you want only singles
        pooled = metric_df.query(f"{current_subset} > 0")[current_subset] # if you want all other combined cases        
        pooled = pooled.drop(singles.index)
        # # Handle colour input
        if isinstance(colour, tuple):
            c = colour
        if isinstance(colour, str):
            c = colour
        elif colour is None:
            if subset in color_mappings.keys() and subset in color_mappings.keys():   
                c = color_mappings[subset]
            else:
                c = "lightgrey"
                # Dynamically handle labels
        if labels is not None and isinstance(labels, Iterable) is False: 
            raise ValueError("Labels must be a tuple or list-like of strings")
        if labels is None:
            if subset in label_mappings.keys():   
                label = label_mappings[subset]
        else:
            label = labels[n]
        labels_used.append(label)
        alpha_val = .66
        sns.kdeplot(x = pooled, ax = ax[2], color = c, alpha = alpha_val, lw = 2)
        sns.kdeplot(x = singles, ax = ax[1], color = c, alpha = alpha_val, lw = 2)
        sns.stripplot(x = singles, y = n, color = c, ax = ax[0], jitter=True, size = 4,
                    alpha = alpha_val/3, orient = 'h', edgecolor = 'k', linewidth = .5,
                    marker = 'D', label = label)
        sns.stripplot(x = pooled, y = n, color = c, ax = ax[3], jitter=True, size = 4,
                    alpha = alpha_val/3, orient = 'h', edgecolor = 'k', linewidth = .5,
                    label = label)
    if metric in title_mappings.keys():
        ax[-1].set_xlabel(title_mappings[metric])
    else:
        ax[-1].set_xlabel(metric)
    # Take care of eaxes
    for cax in ax:
        cax.set_yticklabels([])
        cax.set_ylabel("")
        cax.spines.top.set_visible(False)
        cax.spines.bottom.set_visible(False)
        cax.grid(True, axis = 'x')
    ax[0].spines.top.set_visible(True)
    ax[-1].spines.bottom.set_visible(True)
    ax[1].spines.bottom.set_visible(True)
    ax[2].spines.top.set_visible(True)
    if ax[1].get_ylim()[-1] > ax[2].get_ylim()[-1]:
        ax[2].set_ylim(ax[1].get_ylim())
    if ax[1].get_ylim()[-1] < ax[2].get_ylim()[-1]:
        ax[1].set_ylim(ax[2].get_ylim())
    ax[2].invert_yaxis()
    ax[0].set_ylabel("Singular-colour RFs", loc = 'top')
    ax[3].set_ylabel("Multi-colour RFs", loc = 'bottom')    
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[-1].get_legend_handles_labels()
    hand_labl = np.array([[handles1, labels1], [handles2, labels2]])
    handles = [(i, j) for i, j in zip(handles1, handles2)]
    ax[0].legend(handles, labels_used, handler_map={tuple: HandlerTuple(ndivide=None)})
    l = ax[-1].legend()
    l.remove()
    plt.show()
    
def plot_multi_vs_single(df, metric, subset_list, orientation = 'v', labels = None):
    if orientation == 'v' or orientation == 'vertical':
        _multi_vs_single_vert(df, metric, subset_list)
    if orientation == 'h' or orientation == 'horizontal':
        _multi_vs_single_horz(df, metric, subset_list)
        
