import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
import matplotlib
import numpy as np
import seaborn
import natsort
import pandas as pd
import pygor.plotting
import pygor.strf.clustering
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import warnings
from joblib import Parallel, delayed
import joblib

param_map = {
    "ampl" : "Max abs. amplitude (SD)",
    "area" : "Area (° vis. ang.$^2$)",
    "centdom":"Speed (Hz)" ,
    "centneg":"Neg. speed (Hz)",
    "centpos":"Pos. speed (Hz)",
    "ipl_depths":"Proportion of IPL pop. (%)",
}

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

def pc_summary(clust_pca_df, pca_dict, axis_ranks = [(1, 0)]):
    categories = pd.unique(clust_pca_df["cat_pol"])
    fig, axs = plt.subplots(1, len(categories), figsize = (7.5 * len(categories), 5.5))
    for category, ax in zip(categories, axs):
        pygor.strf.clustering.plot.pc_project(clust_pca_df.query(f"cat_pol == '{category}'"), 
        pca_dict[category], axis_ranks, ax = ax)
        ax.set_title(f"Category '{category}' for PC{axis_ranks[0][0]} and PC{axis_ranks[0][1]}")
    fig.tight_layout()

def plot_df_tuning(post_cluster_df, cluster_ids, group_by = "cluster_id", plot_cols = "all",        
    print_stat = False, ax = None, add_cols = ["ipl_depths"], ipl_percentage = True,
    boxplot = True, scatter = True):
    
    """
    TODO:
        if user passes str or int to "cluster_ids", handle the query 
        accordingly (either encase it in '' or not.) This would make 
        all more flexible. Then can rename cluster_ids param to specify_cluster or 
        search or something...
    """
    if isinstance(cluster_ids, str) is True:
        cluster_ids = [cluster_ids]
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
    if np.all(ax == None): # prevent ValueError on passing multipel axes
        fig, _ax = plt.subplots(len(cluster_ids), num_stats, figsize=np.array([num_stats, .75 * len(cluster_ids)])*4, sharex=False, sharey=False)
        fig.tight_layout()
        if num_stats == 1:
            _ax = [_ax]
    else:
        fig = plt.gcf()
    population_hist_vals_population = np.histogram(post_cluster_df["ipl_depths"], bins = 10, range=(0, 100))[0]
    # Determine category based on cluster_ids
    if isinstance(cluster_ids, str) is False:
        categories = list(set([i.split('_')[0] for i in cluster_ids]))
    else:
        categories = cluster_ids.split('_')[0]
    # Semi redundant, because this could simply be done in next loop...
    for cat in categories:
        for m, clust_id in enumerate(cluster_ids):
            analyse_df = post_cluster_df.query(f"cat_pol == '{cat}'").filter(regex = "^\w+_\d+$|^ipl_depths$|^cluster")
            # Handle external (ax) and internal (_ax) axes assignment differentially
            if np.all(ax == None):
                cax = _ax[m, :]
            else:
                cax = ax
                _ax = ax #nasty solution, but works ㄟ(≧◇≦)ㄏ
            for n, (i, param) in enumerate(zip(cax.flat, unique_cols_sans_wavelength)): 
                if m == 0 and cax.any() == None:
                    i.set_title(param_map[param])
                if param == "ipl_depths":
                    i.axhspan(0, 55, color = "lightgrey", lw = 0)
                    i.axhspan(60, 100, color = "lightgrey", lw = 0)
                    if ipl_percentage == True:
                        percentage_hist_vals_condition = np.histogram(analyse_df.query(f"cluster_id == '{clust_id}'")["ipl_depths"], bins = 10, range=(0, 100))[0]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            percentages = percentage_hist_vals_condition / population_hist_vals_population * 100
                        i.barh(np.arange(0, 100, 10), width= percentages, height=10, color = 'b', edgecolor="black", alpha = .75)
                        i.set_label("skip_this")
                        i.set_xlim(0,40)
                    else:
                        hist = np.histogram(analyse_df.query(f"{group_by} == '{clust_id}'")["ipl_depths"], bins = 10, range=(0, 100))[0]
                        i.barh(np.arange(0, 100, 10), width= hist, height=10, color = 'b', edgecolor="black", alpha = .75)
                        i.set_label("skip_this")
                else:
                    df = analyse_df.query(f"{group_by} == '{clust_id}'").filter(like=f"{param}")
                    colour_scheme = reversed(pygor.plotting.fish_palette)
                    if boxplot == True:
                        sns.boxplot(df, palette = colour_scheme, ax = i)
                    if boxplot == False and scatter == True:
                        sns.stripplot(df, palette = colour_scheme, ax = i, alpha = .2)
                    elif scatter == True:
                        sns.stripplot(df, palette = 'dark:k', ax = i, alpha = .5)
                    i.set_xticks([])
                    i.invert_xaxis()
        for ax in _ax.flat:
            ax.axhline(0, color = "grey", ls = "--")
        # Okay, now we need to figure out which columns to lower the oppacity on 
        # depending on if area == 0... Hold my beer:
        ## First let's find where we need to make changes 
        where_no_area = np.all(post_cluster_df.query(f"{group_by} == '{clust_id}'").filter(regex = "area_\d+$") == 0, axis = 0)
        index_true = np.where(where_no_area == True)[0]
        index_mapping = {"375" : 0, "422" : 1, "478" : 2, "588" : 3}
        wavelength_only = [i.split('_')[-1] for i in where_no_area[index_true].index]
        change_index = [index_mapping[i] for i in wavelength_only]
        alpha_val = 0.1
        ## Each box has 6 associated lines: 2 whiskers, 2 caps, and 1 median (PLEASE MATPLOTLIB DON'T CHANGE THIS:/ )
        lines_per_box = 6
        for cax in _ax.flat:
            ## Now deal with main portion of plot
            for idx in change_index:
                if cax.get_label() == "skip_this": #prevents messing with histogram
                    continue
                # Modify stripplot points
                try:
                    cax.collections[idx].set_alpha(alpha_val)
                except IndexError:
                    pass
                # Now deal with boxplots
                try:
                    patch = cax.patches[idx]
                except IndexError:
                    patch = cax.patch
                fc = patch.get_facecolor()
                patch.set_facecolor(mlp.colors.to_rgba(fc, alpha_val))
                patch.set_edgecolor((0.33, 0.33, 0.33, alpha_val))
                # Modify dots
                start_index = idx * lines_per_box
                end_index = start_index + lines_per_box
                for line in cax.lines[start_index:end_index]:
                    line.set_alpha(alpha_val)    
    return fig, ax

def stats_summary(clust_df, cat = "on", boxplot = True, scatter = True, figsize = None, figsize_scaler = 2, **kwargs):
    # # Vizualize n clusters
    clust_labels = natsort.natsorted(pd.unique(clust_df.query(f'cat_pol == "{cat}"')["cluster"]))
    clust_ids    = natsort.natsorted(pd.unique(clust_df.query(f'cat_pol == "{cat}"')["cluster_id"]))
    # print(np.unique(clust_labels))
    # print(pd.unique(pca_df.query(f"cat_pol == '{cat}'")["cluster_id"]))
    chromatic_cols = clust_df.filter(regex = r"_\d").columns
    unique_cols_sans_wavelength = list(np.unique([i.split('_')[0] for i in chromatic_cols]))
    if "ipl_depths" in clust_df.columns:
        unique_cols_sans_wavelength.append("ipl_depths")
    num_stats = len(unique_cols_sans_wavelength)
    # # pruned_df.filter(regex = "ampl|area|cluster")
    if figsize == None:
        figsize = (num_stats * figsize_scaler *1.2, len(clust_labels) * figsize_scaler)
    fig, ax = plt.subplots(len(clust_labels), num_stats, figsize = figsize, dpi = 100)
    for n, i in enumerate(clust_ids):
        # Assign label accordingly
        if n == 0:
            for a, param_label in zip(ax[0, 0:num_stats], unique_cols_sans_wavelength):
                a.set_title(param_map[param_label], size = 10)
        # Do the rest of the plotting # Regex for fetching all columns wiht name_000 combo, and ipl_depths, and cluser columns
        plot_df_tuning(clust_df, [i], ax = ax[n, 0:num_stats], boxplot = boxplot, scatter = scatter, **kwargs)
    for i, cl_id in zip(ax[:, 0], clust_ids):
        i.set_ylabel(f"{cl_id}", rotation = 0,  labelpad = 30)
    fig.tight_layout() #merged_stats_df
    #plt.suptitle(f"Category: {cat}", size = 20, y = .98)
    plt.subplots_adjust(top = .95)

    for col in range(num_stats):
        ylims = np.array([i.get_ylim() for i in ax[:, col].flat])
        index = np.argmax(np.abs(ylims), axis = 0)
        min_val = np.min(ylims[index][0, :])
        max_val = np.max(ylims[index][1, :])
        min_val, max_val
        for a in ax[:, col]:
            a.set_ylim(min_val, max_val)
    return fig, ax

def _imshow_spatial_reconstruct(df, cluster_id_str, axs=None, parallel=None, **kwargs):
    # Figure out how many columns we need 
    chromatic_cols = df.filter(regex=r"_\d").columns
    unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Deal with passing axes    
    if axs is None:
        fig, axs = plt.subplots(1, len(unique_wavelengths))
    else:
        fig = plt.gcf()
    # Reconstruct the RFs (here optionally passing parallel workers)    
    rf_recons = pygor.strf.clustering.reconstruct.reconstruct_cluster_spatial(df, cluster_id_str, parallel=parallel)
    max_abs = np.max(np.abs(rf_recons))
    # Loop over axes and plot, etc:
    for (n, ax) in enumerate(rf_recons):
        if n == 0:
            axs.flat[n].set_ylabel(cluster_id_str, rotation = 0,  labelpad = 20)
        im = axs.flat[n].imshow(rf_recons[n], cmap=pygor.plotting.custom.maps_concat[n], origin = "lower")
        #axs.flat[n].axis('off')
        axs.flat[n].spines["top"].set_visible(False)
        axs.flat[n].spines["bottom"].set_visible(False)
        axs.flat[n].spines["left"].set_visible(False)
        axs.flat[n].spines["right"].set_visible(False)
        axs.flat[n].set_xticks([])
        axs.flat[n].set_yticks([])
        im.set_clim(-max_abs, max_abs)
        
def _plot_temporal_reconstruct(df, cluster_id_str, axs=None, parallel=None, **kwargs):
    # Figure out how many columns we need 
    chromatic_cols = df.filter(regex=r"_\d").columns
    unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Deal with passing axes    
    if axs is None:
        fig, axs = plt.subplots(1, len(unique_wavelengths))
    else:
        fig = plt.gcf()
    # Reconstruct the RFs (here optionally passing parallel workers)    
    times_recons = pygor.strf.clustering.reconstruct.reconstruct_cluster_temporal(df, cluster_id_str, parallel=parallel)
    # Loop over axes and plot, etc:
    for (n, ax) in enumerate(times_recons):
        if n != 0:
            axs.flat[n].sharey(axs.flat[n-1])
        plot = axs.flat[n].plot(times_recons[n, 0].T, c = "grey")
        plot = axs.flat[n].plot(times_recons[n, 1].T, c = pygor.plotting.fish_palette[n])#, cmap=pygor.plotting.custom.maps_concat[n])
        axs.flat[n].axis('off')
    pygor.plotting.add_scalebar(2.5, ax = axs.flat[-1], rotation = 180, x = 1.1, line_width = 5)

def plot_spatial_reconstruct(clust_df, cluster_id_strings, parallel=True):
    # Figure out how many columns we need 
    chromatic_cols = clust_df.filter(regex=r"_\d").columns
    unique_wavelengths = list(set([i.split('_')[-1] for i in chromatic_cols]))
    if isinstance(cluster_id_strings, str):
        cluster_id_strings = [cluster_id_strings]
    # Determine how many rows and columns
    rows, columns = len(cluster_id_strings), len(unique_wavelengths)
    # Create final plot (and wrap ax in array)
    fig, ax = plt.subplots(rows, columns, figsize = (columns*1.9, rows),
    gridspec_kw = {'wspace' : 0.1, 'hspace' : 0.0, 'bottom': 0.01, 'top': .99})
    if len(cluster_id_strings) == 1:
        ax = np.array([ax])
    # If parallel, initialise the worker
    if parallel:
        # Granted, this LOOKS like its thread un-safe! However, the worker only gets used for 
        # calculations within the _imshow_spatial_reconstruct. And, in fact, the plotting is 
        # done serially after the calculations are done. So it's fine. I think. Lord have mercy.
        with Parallel(n_jobs=4) as worker:
            for n, c_id in enumerate(cluster_id_strings):
                _imshow_spatial_reconstruct(clust_df, c_id, axs=ax[n, 0:4], parallel=worker)
    # Otherwise pass None, which gets processed serially
    else:
        for n, c_id in enumerate(cluster_id_strings):
            _imshow_spatial_reconstruct(clust_df, c_id, axs=ax[n, 0:4], parallel=None)
    # Now post-process plot however you'd like:

    plt.show()

def plot_spacetime_reconstruct(clust_df, cluster_id_strings, parallel=True):
    # Figure out how many columns we need 
    chromatic_cols = clust_df.filter(regex=r"_\d").columns
    unique_wavelengths = list(set([i.split('_')[-1] for i in chromatic_cols]))
    if isinstance(cluster_id_strings, str):
        cluster_id_strings = [cluster_id_strings]
    # Determine how many rows and columns
    rows, columns = len(cluster_id_strings), len(unique_wavelengths) * 2
    # Create final plot (and wrap ax in array)
    fig, ax = plt.subplots(rows, columns, figsize = (columns*1.9, rows),
    gridspec_kw = {'wspace' : 0.1, 'hspace' : 0.0, 'bottom': 0.01, 'top': .99})
    if len(cluster_id_strings) == 1:
        ax = np.array([ax])
    # If parallel, initialise the worker
    if parallel:
        # Granted, this LOOKS like its thread un-safe! However, the worker only gets used for 
        # calculations within the _imshow_spatial_reconstruct. And, in fact, the plotting is 
        # done serially after the calculations are done. So it's fine. I think. Lord have mercy.
        with Parallel(n_jobs=4) as worker:
            for n, c_id in enumerate(cluster_id_strings):
                _imshow_spatial_reconstruct(clust_df, c_id, axs=ax[n, 0:4], parallel=worker)
            for n, c_id in enumerate(cluster_id_strings):
                _plot_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=worker)
    # Otherwise pass None, which gets processed serially
    else:
        for n, c_id in enumerate(cluster_id_strings):
            _imshow_spatial_reconstruct(clust_df, c_id, axs=ax[n, 0:4], parallel=None)
        for n, c_id in enumerate(cluster_id_strings):
            _plot_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=None)
    # Now post-process plot however you'd like:
    pygor.plotting.add_scalebar(4.6153, string = "300 ms", ax = ax[-1, -4], x = 0, y = .1, orientation = 'h', line_width = 5, text_size = 8)
    pygor.plotting.add_scalebar(10, string = f"35.3 °", ax = ax[-1, 0], x = 0, orientation = 'h', line_width = 5, text_size = 8)
    
    plt.show()
    return fig, ax
# def strf_summary():
