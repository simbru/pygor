try:
    from collections import Iterable
except:
    from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygor.plotting
import matplotlib
import scipy.stats
import warnings
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
from statannotations.Annotator import Annotator
from collections import defaultdict

sns.set_context("notebook")
sns.set_style("white")

label_mappings = {
    # "588"  : "588 nm",
    # "478"  : "478 nm",
    # "422"  : "422 nm",
    # "375"  : "375 nm",
    "588": "Red",
    "478": "Green",
    "422": "Blue",
    "375": "UV",
}


def no_chroma_value():
    return "grey"


color_mappings = defaultdict(no_chroma_value)
color_mappings["588"] = pygor.plotting.custom.fish_palette[0]
color_mappings["478"] = pygor.plotting.custom.fish_palette[1]
color_mappings["422"] = pygor.plotting.custom.fish_palette[2]
color_mappings["375"] = pygor.plotting.custom.fish_palette[3]

# color_mappings = {
#     "588"  : pygor.plotting.fish_palette[0],
#     "478"  : pygor.plotting.fish_palette[1],
#     "422"  : pygor.plotting.fish_palette[2],
#     "375"  : pygor.plotting.fish_palette[3],
# }

title_mappings = {
    "area": "Area (°)",
    "diameter": "Diameter (°)",
    "diam": "Diameter (°)",
    "centdom": "Spectral centroid (Hz)",
    "ampl": "Amplitude (SD)",
    "absampl": "Absolute amplitude (SD)",
}

title_mappings_simple = {
    "area": "°$^2$",
    "diam": "°",
    "diameter": "° vis. ang.",
    "centdom": "Hz",
    "ampl": "SD",
}

stat_mappings = {
    "neg_contour_area": "Area for negative contours (°$)",
    "pos_contour_area": "Area for positive contours (°)",
    "contour_area_total": "Total area (°)",
    "total_contour_area_largest": "Largest contour area(°)",
    "diameter": "Diameter (°)",
    "dom_centroids": "Spectral centroid (Hz)",
}

pval_mappings = {
    "area": "pval_space",
    "diam": "pval_space",
    "centdom": "pval_time",
    "centneg": "pval_time",
    "centpos": "pval_time",
}

sns_stat_map = defaultdict(lambda: "")
sns_stat_map["percent"] = "%"
sns_stat_map["count"] = "ROIs"
sns_stat_map["count"] = "ROIs"


def plot_metric_vs(
    df,
    rowX: str,
    rowY: str,
    colour=None,
    ax: plt.axes = None,
    legend: bool = True,
    labels: tuple = None,
    strategy: str = "singles",
    pval_map=True,
    axlim=None,
) -> (plt.figure, plt.axis):
    """The function `plot_areas_vs` generates a plot comparing two variables from a
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

    """
    # Handle differential inputs
    if rowX.split("_")[0] != rowY.split("_")[0]:
        test_statistic = (rowX, rowY)
        filtered_df = df.filter(items=(test_statistic))
    #     raise AttributeError("Input queries are not the same statistic.")
    else:
        test_statistic = rowX.split("_")[0]
        filtered_df = df.filter(regex=test_statistic + "_" + r"\d+")
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
    if strategy == "singles":
        query_str_x = (
            f"{rowX} > 0 & "
            + " == 0 & ".join([i for i in existing_rows if i != rowX])
            + " == 0"
        )
        query_str_y = (
            f"{rowY} > 0 & "
            + " == 0 & ".join([i for i in existing_rows if i != rowY])
            + " == 0"
        )
        rowX_df = filtered_df.query(query_str_x)[rowX]  # if you want only singles
        rowY_df = filtered_df.query(query_str_y)[rowY]
    if strategy == "pooled":
        rowX_df = filtered_df.query(f"{rowX} > 0 & {rowY} == 0")[
            rowX
        ]  # if you want all other combined cases
        rowY_df = filtered_df.query(f"{rowX} == 0 & {rowY} > 0")[rowY]
    if strategy == "skip":
        rowX_df = None
        rowY_df = None
    elif strategy != "singles" and strategy != "pooled":
        raise AttributeError(
            "Strategy not recognised. Please use 'single' or 'pooled'."
        )
    both_df = (
        df[[rowX, rowY]].query(f"{rowX} > 0 & {rowY} > 0").replace(0, np.nan)
    )  # contains all vals but nan empties
    # Get max value for axis limit
    max_val = np.nanmax(filtered_df)
    max_val_leeway = max_val * 1.05
    # Generate figure
    fig = plt.figure(figsize=(5, 5))
    # Generate gridspec
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=(4, 1, 1),
        height_ratios=(1, 1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.01,
        hspace=0.01,
    )
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
        if (
            f"pval_combined_{nm_Y}" in df.columns
            and f"pval_combined_{nm_X}" in df.columns
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_pairs = df.iloc[both_df.index].filter(
                    items=(f"pval_combined_{nm_X}", f"pval_combined_{nm_Y}")
                )
                combined_pvals = scipy.stats.combine_pvalues(
                    pval_pairs, axis=1, method="pearson"
                )[1]
                scatter_colour = combined_pvals
            scatter = ax.scatter(
                both_df[rowX],
                both_df[rowY],
                s=20,
                c=scatter_colour,
                cmap="Greys_r",
                label=f"{labels[0]} and {labels[1]}",
                alpha=0.33,
                edgecolors=None,
            )
    else:
        scatter = ax.scatter(
            both_df[rowX],
            both_df[rowY],
            s=20,
            c="k",
            label=f"{labels[0]} and {labels[1]}",
            alpha=0.33,
        )
    #    plt.colorbar(scatter, cax = ax, orientation = "horizontal", pad = 2)

    ax.plot(
        np.arange(0, max_val_leeway * 1.5, 1),
        np.arange(0, max_val_leeway * 1.5, 1),
        color="grey",
        ls="--",
        alpha=0.5,
    )
    # Plot X KDE and stripplot
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.kdeplot(x=both_df[rowX], color="k", ax=ax_histx)
    sns.kdeplot(x=rowX_df, color=cX, ax=ax_histx)
    ax_stackx = fig.add_subplot(gs[0, 0], sharex=ax)
    sns.stripplot(
        x=rowX_df,
        s=5,
        c=cX,
        label=f"{labels[0]} {strategy}",
        ax=ax_stackx,
        alpha=0.33,
        edgecolor="gray",
        linewidth=0.6,
    )
    try:
        ax_stackx.legend_.remove()
    except AttributeError:
        pass
    # Plot Y KDE and stripplot
    ax_histy = fig.add_subplot(gs[2, 1], sharey=ax)
    sns.kdeplot(y=both_df[rowY], color="k", ax=ax_histy)
    sns.kdeplot(y=rowY_df, color=cY, ax=ax_histy)
    ax_stacky = fig.add_subplot(gs[2, 2], sharey=ax)
    sns.stripplot(
        y=rowY_df,
        s=5,
        c=cY,
        label=f"{labels[1]} {strategy}",
        ax=ax_stacky,
        alpha=0.33,
        edgecolor="gray",
        linewidth=0.6,
    )
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
        ax.set_title(title_mappings[test_statistic], y=-0.3)
    else:
        ax.set_title(test_statistic, y=-0.3)
    # Create legend
    if legend is True:
        dots_labels = [
            ax.get_legend_handles_labels() for ax in fig.axes
        ]  # combine handles and labels
        dots, dot_labels = [sum(lol, []) for lol in zip(*dots_labels)]
        fig.legend(reversed(dots), reversed(dot_labels), bbox_to_anchor=(0.9, 0.8))
    plt.show(block=False)
    plt.pause(0.01)
    plt.close()
    return fig, ax


def plot_distribution(chroma_df, columns_like="area", animate=True):
    data = chroma_df.replace(0, np.nan).filter(like=columns_like)
    present_columns = data.columns
    current_statistic = data.columns[0].split("_")[0]
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"
    # plot_settings['animation.embed_limit'] = 2**128
    with matplotlib.rc_context(rc=plot_settings):
        fig, ax = plt.subplots(figsize=(5, 5))
        max_val = np.max(data.replace(np.nan, 0).to_numpy())
        max_val = max_val + max_val * 0.025
        if current_statistic in title_mappings.keys():
            ax.set_ylabel(title_mappings[current_statistic])
        # ax.set_ylim(bottom = 6, top = max_val)
        sns.boxplot(data=data, palette=reversed(pygor.plotting.custom.fish_palette))
        sns.stripplot(
            data=data,
            palette=r"dark:k",
            alpha=0.5,
            ax=ax,
            zorder=1,
            linewidth=0.5,
            edgecolors="k",
            size=3,
            jitter=0.2,
        )
        ax.set_title("Cone input", y=-0.15)
        labels = ax.get_xticklabels()
        for n, i in enumerate(labels):
            if present_columns[n].split("_")[-1] in label_mappings:
                labels[n] = label_mappings[present_columns[n].split("_")[-1]]
        ax.set_xticklabels(labels)
        if animate:
            initial_line = data.iloc[0]
            initial_line = initial_line[initial_line > 0]
            ax.plot(
                initial_line,
                color="grey",
                alpha=0.5,
                marker="o",
                ms=6,
                zorder=2,
                label="ROI",
                mew=1,
                mec="k",
            )
            legend = ax.legend()

            def line_sweep(num):
                each_line = data.iloc[num].to_numpy()
                # each_line = np.ma.masked_where(each_line == 0, each_line)
                each_line = each_line[each_line > 0]
                ax.get_lines()[0].set_xdata(np.where(each_line > 0)[0])
                ax.get_lines()[0].set_ydata(each_line)
                legend.get_texts()[0].set_text(
                    f"ROI {num}",
                )  # Update label each at frame

            animation = matplotlib.animation.FuncAnimation(
                fig, line_sweep, frames=len(data), interval=150, repeat_delay=500
            )
            animation.to_html5_video()
            plt.close()
            return animation


def ipl_summary_chroma(
    roi_df,
    numcolours=4,
    figsize=(12, 7),
    legend=True,
    ipl_border=55,
    ax=None,
    split_polarity=False,
):
    if split_polarity is True:
        if ax is None:
            fig, ax = plt.subplots(
                1, 4, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True
            )
        else:
            fig = plt.gcf()
        ax = np.repeat(np.expand_dims(ax, 0), 2, axis=0)
        for n, i in enumerate(ax[0, :].flat):
            bbox = i.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= fig.dpi
            height *= fig.dpi
            plotarea = width * height
            i.scatter(
                0.9,
                0.9,
                marker="o",
                c=pygor.plotting.custom.fish_palette[n],
                s=plotarea / 100,
                transform=i.transAxes,
            )
        colormap = pygor.plotting.custom.polarity_palette
    if split_polarity is False:
        if ax is None:
            fig, ax = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
        else:
            fig = plt.gcf()
        colormap = pygor.plotting.custom.fish_palette
    else:
        AttributeError("split_polarity must be bool")
    polarities = [-1, 1]
    colours = ["R", "G", "B", "UV"]
    colours_map = {
        "R": "Red",
        "G": "Green",
        "B": "Blue",
        "UV": "UV",
    }
    polarity_map = {-1: "Negative", 1: "Positive", 2: "Mix"}
    bins = 10
    # sns.set_style("whitegrid")
    for n, i in enumerate(polarities):
        for m, j in enumerate(colours):
            if split_polarity is True:
                curr_colour = colormap[n]
                label = polarity_map[i]
            if split_polarity is False:
                curr_colour = colormap[m]
                label = colours_map[j]
            hist_vals_per_condition = np.histogram(
                roi_df.query(
                    f"polarity ==  {i} & colour == '{j}' & total_contour_area_largest > 0"
                )["ipl_depths"],
                bins=bins,
                range=(0, 100),
            )[0]
            hist_vals_population = np.histogram(
                roi_df.query(f"colour == '{j}' & total_contour_area_largest > 0")[
                    "ipl_depths"
                ],
                bins=bins,
                range=(0, 100),
            )[0]
            percentages = hist_vals_per_condition / np.sum(hist_vals_population) * 100
            ax[n, m].barh(
                np.arange(0, 100, 10),
                width=percentages,
                height=10,
                color=curr_colour,
                edgecolor="white",
                alpha=0.65,
                label=label,
            )
            ax[n, m].grid(False)
            ax[n, m].axhline(ipl_border, c="k", ls="--")
            if m == 0 and split_polarity is False:
                if i == -1:
                    ax[n, m].set_title(
                        "OFF",
                        weight="bold",
                        loc="left",
                        c=pygor.plotting.polarity_palette[0],
                    )
                if i == 1:
                    ax[n, m].set_title(
                        "ON",
                        weight="bold",
                        loc="left",
                        c=pygor.plotting.polarity_palette[1],
                    )
            num_cells = int(len(np.unique(roi_df.index)) / numcolours)
            num_strfs = int(
                len(np.unique(roi_df.query("total_contour_area_largest > 0").index))
                / numcolours
            )
            percent = np.round(num_strfs / num_cells * 100, 2)
            # ax[1, 0].set_xlabel(f"Percentage by colour (n = {num_cells})", size = 10)
            ax[1, 0].set_xlabel("% STRFs", size=12)
    for cax in ax[:, 0].flat:
        cax.set_ylabel("IPL depth (%)")
        cax.text(
            x=0.9,
            y=(ipl_border / 100) + 0.1,
            va="baseline",
            ha="center",
            s="OFF",
            size=10,
            transform=cax.transAxes,
        )
        cax.text(
            x=0.9,
            y=(ipl_border / 100) - 0.1,
            va="bottom",
            ha="center",
            s="ON",
            size=10,
            transform=cax.transAxes,
        )
    if legend is True:
        if split_polarity is True:
            handles, labels = [], []
            handles, labels = ax[0, 0].get_legend_handles_labels()
            labels_ = ["OFF", "ON"]
            handles_ = [i for i in handles[: len(labels_)]]
            # exact location does not matter as we will move axis later
            ax.flat[-1].legend(
                handles_,
                labels_,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                bbox_to_anchor=(1.04, 1.4),
                loc="upper left",
            )
        if split_polarity is False:
            handles, labels = [], []
            for i in ax.flat:
                handle, label = i.get_legend_handles_labels()
                handles.append(handle)
                labels.append(label)
            labels_ = ["Red", "Green", "Blue", "UV"]
            handles_ = [i[0] for i in handles[: len(labels_)]]
            ax.flat[-1].legend(
                handles_,
                labels_,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                bbox_to_anchor=(1.04, 1.4),
                loc="upper left",
            )
        sns.move_legend(
            ax.flat[-1],
            "lower center",
            bbox_to_anchor=(-1.3, -0.4),
            ncol=4,
            title=None,
            frameon=False,
        )
    sns.despine()
    return fig, ax


def ipl_summary_polarity_roi(
    roi_df,
    numcolours=4,
    figsize=(8, 4),
    polarities=[-1, 1],
    legend=True,
    ipl_border=55,
    ax=None,
):
    if ax is None:
        fig, axs = plt.subplots(
            len(polarities), 1, sharex=True, sharey=True, figsize=figsize
        )
    else:
        axs = ax
        fig = plt.gcf()
    skip_df = roi_df.query("total_contour_area_largest > 0")
    # roi_df.iloc[roi_df["ipl_depths"].dropna().index]
    bins = 10
    titles = ["OFF", "ON", "Mixed polarity", "other"]
    palette = pygor.plotting.custom.polarity_palette
    # sns.set_style("whitegrid")
    tot_sum = 0
    for n, ax in enumerate(axs.flatten()):
        query_df = skip_df.query(f"polarity ==  {polarities[n]}")["ipl_depths"]
        hist = np.histogram(query_df, bins=bins, range=(0, 100))
        tot_sum += len(query_df)
        hist_vals_per_condition = hist[0]
        hist_vals_population = np.histogram(
            skip_df["ipl_depths"], bins=bins, range=(0, 100)
        )[0]
        # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
        percentages = hist_vals_per_condition / np.sum(hist_vals_population) * 100
        # percentages = hist_vals_per_condition
        ax.barh(
            np.arange(0, 100, 10),
            width=percentages,
            height=10,
            color=palette[n],
            edgecolor="white",
            alpha=0.65,
            label=titles[n],
        )
        # ax.set_title(titles[n], size = 12)
        ax.axhline(ipl_border, c="k", ls="--")
    # axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border + 5, va = 'center', s = "OFF", size = 10)
    # axs[0].text(x = axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175, y = ipl_border - 5, va = 'center', s = "ON", size = 10)
    num_cells = len(skip_df.index)
    print(num_cells, tot_sum)
    # axs[0].set_xlabel(f"Percentage by polarity (n = {num_cells})", size = 10)
    if legend is True:
        handles, labels = [], []
        for i in axs:
            handle, label = i.get_legend_handles_labels()
            handles.append(handle)
            labels.append(label)
            # ax[0, 3].legend(handles, labels)
        # labels_ = natsort.natsorted(list(set([i[0] for i in labels])))
        # labels_ = list(set([i[0] for i in labels]))
        labels_ = ["OFF", "ON"]
        handles_ = [i[0] for i in handles[: len(labels_)]]
        axs[-1].legend(
            handles_,
            labels_,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            bbox_to_anchor=(1.04, 1.4),
            loc="upper left",
        )
        sns.move_legend(
            axs[-1],
            "lower center",
            bbox_to_anchor=(0.5, -0.5),
            ncol=4,
            title=None,
            frameon=False,
        )
    sns.despine()
    return fig, ax


def ipl_summary(df):
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(12, 6))
    pygor.strf.analyse.plot.ipl_summary_chroma(df, ax=ax[:, 0:4])
    pygor.strf.analyse.plot.ipl_summary_polarity_roi(df, ax=ax[:, -1])
    return fig, ax


def plot_roi_hist(
    roi_df,
    statistic="contour_area_total",
    conditional="default",
    category="colour",
    bins="auto",
    binwidth=None,
    colour_list=None,
    kde=False,
    scalebar=True,
    avg_line=False,
    ax=None,
    legend=True,
    main_ax="x",
    **kwargs,
):
    """This function `plot_roi_hist` creates histograms of a specified statistic within regions of interest
    (ROIs) based on different categories such as color or polarity.

    Parameters
    ----------
    roi_df
        The `roi_df` parameter is a DataFrame containing data related to regions of interest (ROIs) that
    you want to plot histograms for.
    stat, optional
        The `stat` parameter in the `plot_roi_hist` function represents the statistical measure that will
    be plotted in the histogram. It could be values like "contour_area_total", "contour_complexity",
    "dom_peaktime", etc., depending on the data you are working with.
    conditional
        The `conditional` parameter in the `plot_roi_hist` function is used to specify a condition that the
    data must meet in order to be included in the histogram plot (condtional > 0). If the `conditional` parameter is not
    provided, it defaults to the value of the `stat` parameter. This condition is used
    category, optional
        The `category` parameter in the `plot_roi_hist` function specifies how the data should be grouped
    or categorized for plotting. It can take on two possible values:
    bins, optional
        The `bins` parameter in the `plot_roi_hist` function specifies the number of bins to use for the
    histogram. If set to "auto", the function will automatically determine the number of bins to use
    based on the data. If you have a specific number of bins in mind, you can provide
    binwidth
        The `binwidth` parameter in the `plot_roi_hist` function specifies the width of each bin in the
    histogram. It allows you to control the size of the bins in the histogram plot. If you set a
    specific `binwidth`, the histogram will be divided into bins of that width.

    """
    if "hue" in kwargs and "palette" not in kwargs:
        if kwargs["hue"] == "polarity":
            kwargs["palette"] = pygor.plotting.custom.polarity_palette
        elif kwargs["hue"] == "Group" and len(roi_df[kwargs["hue"]].unique()) == 2:
            kwargs["palette"] = pygor.plotting.custom.compare_conditions[2]
    try:
        title = stat_mappings[statistic]
    except KeyError:
        title = statistic
    if conditional == "default":
        roi_df = roi_df.query("contour_area_total > 0")
    elif conditional != None or conditional != "default":
        roi_df = roi_df.query(f"{conditional}")
    # if category == "colour":fi
    categories_specified = np.unique(roi_df[category])
    if (
        "R" in categories_specified
        and "G" in categories_specified
        and "B" in categories_specified
        and "UV" in categories_specified
    ):
        categories_specified = ["R", "G", "B", "UV"]
    n_categories = len(categories_specified)
    if colour_list == None:
        if category == "polarity":
            colour_list = pygor.plotting.custom.polarity_palette
        else:
            colour_list = pygor.plotting.custom.fish_palette
    if ax is None:
        fig, ax = plt.subplots(
            n_categories,
            1,
            figsize=(4, 6),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0},
        )
    else:
        fig = ax.flat[0].get_figure()
    # Loop through each category and plot the histogram to axes
    for n, c in enumerate(categories_specified):
        if isinstance(c, str):
            df = roi_df.query(f"{category} == '{c}'")
        else:
            df = roi_df.query(f"{category} == {c}")
        # Populate each histogram subplot
        if main_ax == "y":
            currplot = sns.histplot(
                data=df,
                y=statistic,
                color=colour_list[n],
                ax=ax.flat[n],
                kde=kde,
                bins=bins,
                binwidth=binwidth,
                legend=legend,
                **kwargs,
            )
        if main_ax == "x":
            currplot = sns.histplot(
                data=df,
                x=statistic,
                color=colour_list[n],
                ax=ax.flat[n],
                kde=kde,
                bins=bins,
                binwidth=binwidth,
                legend=legend,
                **kwargs,
            )
        else:
            AttributeError("main_ax must be either 'x' or 'y'")
        if legend is True:
            # Fetch legend contents
            currlegend = currplot.get_legend()
            if legend is not None:
                try:
                    # Get the handles and text out of the legend
                    handles, labels = (
                        currlegend.legendHandles,
                        [text.get_text() for text in currlegend.get_texts()],
                    )
                    # Remove the legend again, we will plot it later if hue is used
                    currplot.get_legend().remove()
                except AttributeError:
                    pass
    # Set up axes
    for a, c in zip(ax.flat, categories_specified):
        a.set_ylabel("")
        if avg_line is True:
            a.axvline(
                np.median(roi_df.query(f"{category} == '{c}'")[statistic]),
                c="k",
                ls="--",
            )
        a.set_yticks([])
    a.set_xlabel(title)
    if "hue" in kwargs:
        if legend is True:
            # Handle residual legend nonesense
            fig.legend(
                title=kwargs["hue"],
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncols=2,
            )
        for n, i in enumerate(ax.flat):
            bbox = i.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= fig.dpi
            height *= fig.dpi
            plotarea = width * height
            i.scatter(
                0.08,
                0.8,
                marker="o",
                c=colour_list[n],
                s=plotarea / 100,
                transform=i.transAxes,
            )
    if scalebar is True:
        if "stat" in kwargs:
            if kwargs["stat"] != "count" or kwargs["stat"] != "auto":
                # Just round here, because often we value will not be above 10, forcing it to 0
                auto_num = int(np.round(ax[-1].get_ylim()[1] / 3))
            if kwargs["stat"] == "count":
                # Nearest 10
                auto_num = int(np.round(ax[-1].get_ylim()[1] / 3 / 10) * 10)
        else:
            # Nearest 10
            auto_num = int(np.round(ax[-1].get_ylim()[1] / 3 / 10) * 10)
        if auto_num < 10:  # Make sure a bar is plotted if there is less than 10
            auto_num = int(np.round(ax[-1].get_ylim()[1] / 3))
        if auto_num == 0:
            auto_num = np.round(ax[-1].get_ylim(), 3)[1]
        if "stat" in kwargs:
            auto_str = sns_stat_map[kwargs["stat"]]
            scalebar_str = str(auto_num) + " " + auto_str
        else:
            scalebar_str = str(auto_num)
        pygor.plotting.add_scalebar(
            auto_num,
            string=f"{scalebar_str}",
            ax=ax[-1],
            orientation="v",
            x=1.1,
            flip_text=True,
        )
    a.set_xlabel(title)
    sns.despine()
    try:
        sns.move_legend(
            fig,
            "lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            title=None,
            frameon=False,
        )
    except ValueError:
        # warnings.warn("Could not move legend")
        pass
    return fig, ax


def ipl_summary_polarity_chroma(
    chroma_df, numcolours=4, figsize=(8, 4), cat_pol=["off", "on"]
):
    polarities = ["off", "on"]  # , 'opp']
    fig, axs = plt.subplots(
        1, len(polarities), sharex=True, sharey=True, figsize=(4.8, 2.5)
    )
    bins = 10
    titles = ["OFF", "ON", "Opponent", "empty", "mix"]
    palette = pygor.plotting.custom.polarity_palette
    palette.append("r")
    # sns.set_style("whitegrid")
    tot_sum = 0
    for n, ax in enumerate(axs.flatten()):
        query_df = chroma_df.query(f"cat_pol ==  '{polarities[n]}'")["ipl_depths"]
        hist = np.histogram(query_df, bins=bins, range=(0, 100))
        tot_sum += len(query_df)
        hist_vals_per_condition = hist[0]
        hist_vals_population = np.histogram(
            chroma_df["ipl_depths"], bins=bins, range=(0, 100)
        )[0]
        # hist_vals_population = np.histogram(chroma_df.query(f"colour == '{j}'")["ipl_depths"], bins = bins)[0]
        percentages = hist_vals_per_condition / np.sum(hist_vals_population) * 100
        # percentages = hist_vals_per_condition
        ax.barh(
            np.arange(0, 100, 10),
            width=percentages,
            height=10,
            color=palette[n],
            edgecolor="black",
            alpha=0.75,
        )
        ax.set_title(titles[n], size=12)
        ipl_border = 55
        ax.axhline(ipl_border, c="k", ls="--")
    axs[0].text(
        x=axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175,
        y=ipl_border + 5,
        va="center",
        s="OFF",
        size=10,
    )
    axs[0].text(
        x=axs[0].get_xlim()[1] - axs[0].get_xlim()[1] * 0.175,
        y=ipl_border - 5,
        va="center",
        s="ON",
        size=10,
    )
    num_cells = len(chroma_df.index)
    print(num_cells, tot_sum)
    axs[0].set_xlabel(f"Percentage by polarity (n = {num_cells})", size=10)
    plt.show()
    return fig, ax


def _multi_vs_single_vert(
    df, metric, subset_list, colour=None, labels=None, **kwargs
) -> (plt.figure, plt.axis):
    # Generate filtered dataframe by metric
    metric_df = df.filter(like=metric)
    # Generate figures according to subset list
    fig, ax = plt.subplots(
        1,
        4,
        figsize=(5, 5),
        sharex=False,
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1.25], "wspace": 0},
    )
    ax[0].get_shared_x_axes().join(ax[0], ax[3])
    # Loop through subsets and plot data
    labels_used = []
    for n, subset in enumerate(subset_list):
        existing_rows = [i for i in metric_df.columns if "bool" not in i]
        current_subset = "_".join([metric, subset])
        query_str = (
            f"{metric}_{subset} > 0 & "
            + " == 0 & ".join([i for i in existing_rows if i != current_subset])
            + " == 0"
        )
        print(query_str)
        singles = metric_df.query(query_str)[current_subset]  # if you want only singles
        pooled = metric_df.query(f"{current_subset} > 0")[
            current_subset
        ]  # if you want all other combined cases
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
        alpha_val = 0.66
        sns.kdeplot(y=singles, ax=ax[1], color=c, alpha=alpha_val, lw=2)
        sns.kdeplot(y=pooled, ax=ax[2], color=c, alpha=alpha_val, lw=2)
        sns.stripplot(
            y=singles,
            x=n,
            color=c,
            ax=ax[0],
            jitter=True,
            alpha=alpha_val / 3,
            orient="v",
            edgecolor="k",
            linewidth=0.5,
            marker="D",
            label=label,
        )
        sns.stripplot(
            y=pooled,
            x=n,
            color=c,
            ax=ax[3],
            jitter=True,
            alpha=alpha_val / 3,
            orient="v",
            edgecolor="k",
            linewidth=0.5,
            marker="o",
            label=label,
        )
        if "lim" in kwargs.keys():
            ax[1].set_ylim(kwargs["lim"])
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
        cax.grid(True, axis="y")
    ax[0].spines.left.set_visible(True)
    ax[-1].spines.right.set_visible(True)
    ax[2].spines.left.set_visible(True)
    if ax[1].get_xlim()[-1] > ax[2].get_xlim()[-1]:
        ax[2].set_xlim(ax[1].get_xlim())
    if ax[1].get_xlim()[-1] < ax[2].get_xlim()[-1]:
        ax[1].set_xlim(ax[2].get_xlim())
    ax[1].invert_xaxis()
    ax[0].set_title("Singular-colour RFs", loc="left")
    ax[3].set_title("Multi-colour RF RFs", loc="right")
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[-1].get_legend_handles_labels()
    # hand_labl = np.array([[handles1, labels1], [handles2, labels2]])
    handles = [(i, j) for i, j in zip(handles1, handles2)]
    ax[0].legend(handles, labels_used, handler_map={tuple: HandlerTuple(ndivide=None)})
    legd = ax[-1].legend()
    legd.remove()
    plt.show()


def _multi_vs_single_horz(
    df, metric, subset_list, colour=None, labels=None, **kwargs
) -> (plt.figure, plt.axis):
    # Generate filtered dataframe by metric
    metric_df = df.filter(like=metric)
    # Generate figures according to subset list
    fig, ax = plt.subplots(
        4,
        1,
        figsize=(5, 5),
        sharey=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1, 1, 1.25], "hspace": 0},
    )
    # Loop through subsets and plot data
    labels_used = []
    for n, subset in enumerate(subset_list):
        existing_rows = [i for i in metric_df.columns if "bool" not in i]
        current_subset = "_".join([metric, subset])
        query_str = (
            f"{metric}_{subset} > 0 & "
            + " == 0 & ".join([i for i in existing_rows if i != current_subset])
            + " == 0"
        )
        print(query_str)
        singles = metric_df.query(query_str)[current_subset]  # if you want only singles
        pooled = metric_df.query(f"{current_subset} > 0")[
            current_subset
        ]  # if you want all other combined cases
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
        alpha_val = 0.66
        sns.kdeplot(x=pooled, ax=ax[2], color=c, alpha=alpha_val, lw=2)
        sns.kdeplot(x=singles, ax=ax[1], color=c, alpha=alpha_val, lw=2)
        sns.stripplot(
            x=singles,
            y=n,
            color=c,
            ax=ax[0],
            jitter=True,
            size=4,
            alpha=alpha_val / 3,
            orient="h",
            edgecolor="k",
            linewidth=0.5,
            marker="D",
            label=label,
        )
        sns.stripplot(
            x=pooled,
            y=n,
            color=c,
            ax=ax[3],
            jitter=True,
            size=4,
            alpha=alpha_val / 3,
            orient="h",
            edgecolor="k",
            linewidth=0.5,
            label=label,
        )
        if "lim" in kwargs.keys():
            ax[1].set_xlim(kwargs["lim"])
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
        cax.grid(True, axis="x")
    ax[0].spines.top.set_visible(True)
    ax[-1].spines.bottom.set_visible(True)
    ax[1].spines.bottom.set_visible(True)
    ax[2].spines.top.set_visible(True)
    if ax[1].get_ylim()[-1] > ax[2].get_ylim()[-1]:
        ax[2].set_ylim(ax[1].get_ylim())
    if ax[1].get_ylim()[-1] < ax[2].get_ylim()[-1]:
        ax[1].set_ylim(ax[2].get_ylim())
    ax[2].invert_yaxis()
    ax[0].set_ylabel("Singular-colour RFs", loc="top")
    ax[3].set_ylabel("Multi-colour RFs", loc="bottom")
    handles1, _ = ax[0].get_legend_handles_labels()
    handles2, _ = ax[-1].get_legend_handles_labels()
    #    hand_labl = np.array([[handles1, labels1], [handles2, labels2]])
    handles = [(i, j) for i, j in zip(handles1, handles2)]
    ax[0].legend(handles, labels_used, handler_map={tuple: HandlerTuple(ndivide=None)})
    legd = ax[-1].legend()
    legd.remove()
    plt.show()


def plot_multi_vs_single(
    df, metric, subset_list, orientation="v", labels=None, **kwargs
):
    if orientation == "v" or orientation == "vertical":
        _multi_vs_single_vert(df, metric, subset_list, **kwargs)
    if orientation == "h" or orientation == "horizontal":
        _multi_vs_single_horz(df, metric, subset_list, **kwargs)


def compare_groups_violin(
    df,
    pairs,
    metric="area",
    hue="Group",
    group_by="colour",
    compare="Group",
    orient="v",
    test="Kruskal",
):
    # Reshape and coax into the right format for simpler seaborn handling
    colour_categories = pd.unique(df.filter(like=metric + "_").columns)
    colour_categories = np.append(colour_categories, "Group")
    df = df[colour_categories.tolist()]
    df_reshaped = pd.melt(df, id_vars=["Group"])
    df_reshaped = df_reshaped.mask(df_reshaped == 0, np.nan, inplace=False)
    df_reshaped = df_reshaped.rename(
        columns={"variable": f"{group_by}", "value": f"{metric}"}
    )
    if group_by == "colour":
        df_reshaped.replace(
            {
                f"{metric}_375": "375 nm",
                f"{metric}_422": "422 nm",
                f"{metric}_478": "478 nm",
                f"{metric}_588": "588 nm",
            },
            inplace=True,
        )
    # Generate plot

    figsize = [12, 5]
    if orient == "h" or orient == "horizontal":
        figsize = list(reversed(figsize))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Now define our parameters and handle axis orientations
    if orient == "v" or orient == "vertical":
        x = group_by
        y = metric
        ax.set_xticklabels([])
        try:
            ax.set_ylabel(pygor.strf.analyse.plot.title_mappings[metric])
        except KeyError:
            ax.set_ylabel(y)
    if orient == "h" or orient == "horizontal":
        x = metric
        y = group_by
        ax.set_yticklabels([])
        try:
            ax.set_xlabel(pygor.strf.analyse.plot.title_mappings[metric])
        except KeyError:
            ax.set_xlabel(x)
    if orient not in ["v", "vertical", "h", "horizontal"]:
        raise ValueError("Orientation must be 'v', 'vertical', 'h', or 'horizontal'")
    hue = hue
    hue_order = ["AC block", "Control"]
    order = list(reversed(["375 nm", "422 nm", "478 nm", "588 nm"]))
    palmap = list(reversed(pygor.plotting.compare_conditions[2]))
    # Populate plot
    sns.stripplot(
        df_reshaped,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
        dodge=True,
        linewidth=0.5,
        palette="dark:k",
        legend=False,
        alpha=0.33,
        size=2,
        ax=ax,
    )  # ,palette = compare_conditions[2]
    cax = sns.violinplot(
        df_reshaped,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
        linewidth=1,
        palette=palmap,
        legend=True,
        split=True,
        density_norm="area",
        common_norm=False,
        saturation=0.8,
        gap=0.2,
        ax=ax,
        bw_adjust=1,
        inner_kws=dict(box_width=8, whis_width=2),
    )
    if group_by == "colour":
        for n, i in enumerate(pairs):
            if orient == "v" or orient == "vertical":
                rel_ax = "y"
                cx = n
                cy = 0
            if orient == "h" or orient == "horizontal":
                rel_ax = "x"
                cx = 0
                cy = n
            pygor.plotting.custom.label_ax_colour(
                cax,
                x=cx,
                y=cy,
                marker=".",
                colour=(pygor.plotting.fish_palette)[n],
                clip_on=False,
                relative_axis=rel_ax,
                zorder=4,
            )
    # Override seaborn axis labelling
    if orient == "v" or orient == "vertical":
        ax.set_xlabel("")
        ax.set_xticklabels([])
        sns.despine(bottom=True)
    if orient == "h" or orient == "horizontal":
        ax.set_ylabel("")
        ax.set_yticklabels([])
        sns.despine(left=True)
    if test is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotator = Annotator(
                cax,
                pairs,
                data=df_reshaped,
                x=x,
                y=y,
                hue=hue,
                hue_order=hue_order,
                order=order,
                orient=orient,
            )
            annotator.configure(
                test=test, text_format="star", loc="outside", show_test_name=False
            )
            annotator.apply_and_annotate()
    return fig, ax


# def metric_boxplot(df, metric, sigbar = True, sigtest = 'Kruskal',
#     ):
#     # Formatting, reshaping, renaming
#     colour_categories = pd.unique(df.filter(like = "area_").columns)
#     colour_categories = np.append(colour_categories, "Group")
#     df = df[colour_categories.tolist()]
#     df_reshaped = pd.melt(df, id_vars = ["Group"])
#     df_reshaped = df_reshaped.mask(df_reshaped == 0, np.nan, inplace = False)
#     df_reshaped = df_reshaped.rename(columns = {"variable":"colour", "value":"area"})
#     df_reshaped.replace({"area_375":"375 nm", "area_422":"422 nm", "area_478":"478 nm", "area_588":"588 nm"}, inplace = True)
#     df_reshaped
#     x = "colour"
#     y = metric
#     hue = None
#     hue_order = None

#     # Plot
#     cax = sns.boxplot(data = df_reshaped, y = y, x = x, hue = hue, hue_order=hue_order,
#                 showfliers = False, palette=reversed(pygor.plotting.fish_palette), boxprops=dict(alpha=0.5))
#     sns.stripplot(df_reshaped, y = "area", x = "colour", hue = hue, hue_order=hue_order, dodge=False, linewidth=.5,
#                 color = 'k', legend = False, alpha = 0.33, size = 2)
#     if sigbar == True:
#         pairs = [
#             ("588 nm", "478 nm"),
#             ("588 nm", "422 nm"),
#             ("588 nm", "375 nm"),
#         ]
#         ax = plt.gca()
#         try:
#             ax.set_ylabel(pygor.strf.analyse.plot.title_mappings[y])
#         except KeyError:
#             ax.set_ylabel(y)
#         annotator = Annotator(cax, pairs, data=df_reshaped, x=x, y=y, hue = hue, hue_order=hue_order)
#         annotator.configure(test=sigtest, text_format='star', loc='inside')
#         annotator.apply_and_annotate()
