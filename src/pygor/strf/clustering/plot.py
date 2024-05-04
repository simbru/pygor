import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import seaborn
import natsort
import pandas as pd
import pygor.strf.clustering

def pc_project(pca_DF, pca, axis_ranks = [(0, 1)], alpha=1, cmap = "viridis", ax = None):
    # (X_projected, pca, axis_ranks, labels = None, alpha=1, clust_labels=None, cmap = "viridis"):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''
    pca_df_to_plot = pca_DF
    labels = pca_df_to_plot["cluster"]
    clust_labels = pca_df_to_plot["cluster_id"]
    X_projected = pca_df_to_plot.filter(like = "PC").to_numpy()

    if ax == None:
        fig, ax = plt.subplots(figsize=(7,6))

    # For each factorial plane
    for d1,d2 in axis_ranks: 
            # Initialise the matplotlib figure      
            colormap = matplotlib.cm.get_cmap(cmap, len(np.unique(labels)) + 1)
            colormap = colormap(range(np.max(labels) + 1))
            # colormap = colormap(range(0,255))
            # Display the points
            if clust_labels is None:
                ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                clust_labels = np.array(clust_labels)
                for n, value in enumerate(natsort.natsorted(np.unique(clust_labels))):
                    selected = np.where(clust_labels == value)
                    ax.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, color = colormap[n], label=value)
                ax.legend(bbox_to_anchor = (1.25, .5), loc = "center")
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            ax.set_xlim([-boundary,boundary])
            ax.set_ylim([-boundary,boundary])
            # Display grid lines
            ax.plot([-100, 100], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            ax.set_xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            ax.set_title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)

def plot_df_tuning(post_cluster_df, clusters = 0, group_by = "cluster", plot_cols = "all", specify_cluster = None, 
    print_stat = False, ax = None, add_cols = ["ipl_depths"], sharey = False):
    if group_by not in post_cluster_df.columns:
        raise AttributeError(f"Please ensure {group_by} columns exists in input DF.")
    if plot_cols == "all":
        # This goes through the DataFrames and identifies which columns are present and extracts them 
        # independent of wavelength (more on parsing that below)
        chromatic_cols = post_cluster_df.filter(regex = r"_\d").columns
        unique_cols_sans_wavelength = list(np.unique([i.split('_')[0] for i in chromatic_cols]))
        for i in add_cols:
            unique_cols_sans_wavelength.append(i)
        num_stats = len(unique_cols_sans_wavelength)
    else:
        raise NotImplementedError("plot_cols must currently be = 'all'")
    if np.all(ax == None):
        fig, ax = plt.subplots(1, num_stats, figsize=np.array([num_stats, .75])*4, sharex=True, sharey = sharey)
        fig.tight_layout()
        if num_stats == 1:
            ax = [ax]
    for m, clust_num in enumerate(clusters):
        for n, (i, param) in enumerate(zip(ax.flat, unique_cols_sans_wavelength)): 
            if m == 0 and ax.any() == None:
                i.set_title(param)
            if param == "ipl_depths":
                i.axhspan(0, 55, color = "lightgrey", lw = 0)
                i.axhspan(60, 100, color = "lightgrey", lw = 0)
                sns.histplot(data = post_cluster_df.query(f"{group_by} == {clust_num}"), y = "ipl_depths", binrange = (0, 100), binwidth = 10, ax = i)
                i.set_xlim(0, 20)
            else:
                df = post_cluster_df.query(f"cluster == {clust_num}").filter(like=f"{param}")
                sns.boxplot(df, palette = pygor.plotting.fish_palette, ax = i)
                sns.stripplot(df, palette = 'dark:k', ax = i)
                i.set_xticks([])
                # i.sharey()

def stats_summary(clust_df, cat = "on", **kwargs):
    # # Vizualize n clusters
    clust_labels = pd.unique(clust_df.query(f'cat_pol == "{cat}"')["cluster"])
    # print(np.unique(clust_labels))
    # print(pd.unique(pca_df.query(f"cat_pol == '{cat}'")["cluster_id"]))
    chromatic_cols = clust_df.filter(regex = r"_\d").columns
    unique_cols_sans_wavelength = list(np.unique([i.split('_')[0] for i in chromatic_cols]))
    if "ipl_depths" in clust_df.columns:
        unique_cols_sans_wavelength.append("ipl_depths")
    num_stats = len(unique_cols_sans_wavelength)
    # # pruned_df.filter(regex = "ampl|area|cluster")
    fig, ax = plt.subplots(len(clust_labels), num_stats, figsize = (num_stats*3.5, len(clust_labels)*3), dpi = 80)
    for n, i in enumerate(clust_labels):
        # Assign label accordingly
        if n == 0:
            for a, param_label in zip(ax[0, 0:num_stats], unique_cols_sans_wavelength):
                a.set_title(param_label)
        # Do the rest of the plotting
        plot_df_tuning(clust_df.query(f"cat_pol == '{cat}'").filter(regex = "ampl|area|cent|peak|ipl|cluster|cluster_id"), [i], ax = ax[n, 0:num_stats], **kwargs)
    fig.tight_layout() #merged_stats_df
    plt.suptitle(f"Category: {cat}", size = 20, y = 1.01)

def pc_summary(clust_pca_df, pca_dict, axis_ranks = [(1, 0)]):
    categories = pd.unique(clust_pca_df["cat_pol"])
    fig, axs = plt.subplots(1, len(categories), figsize = (7.5 * len(categories), 5.5))
    for category, ax in zip(categories, axs):
        pygor.strf.clustering.plot.pc_project(clust_pca_df.query(f"cat_pol == '{category}'"), 
        pca_dict[category], axis_ranks, ax = ax)
        ax.set_title(f"Category '{category}' for PC{axis_ranks[0][0]} and PC{axis_ranks[0][1]}")
    fig.tight_layout()