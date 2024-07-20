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
import pygor.utilities
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import warnings
from joblib import Parallel, delayed
import joblib
import pygor.strf.pixconverter as pix
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
                        i.set_label("skip_this") #trust me, this works (see later lines)
                        i.set_xlim(0,40)
                    else:
                        hist = np.histogram(analyse_df.query(f"{group_by} == '{clust_id}'")["ipl_depths"], bins = 10, range=(0, 100))[0]
                        i.barh(np.arange(0, 100, 10), width= hist, height=10, color = 'b', edgecolor="black", alpha = .75)
                        i.set_label("skip_this")
                else:
                    df = analyse_df.query(f"{group_by} == '{clust_id}'").filter(like=f"{param}")
                    colour_scheme = reversed(pygor.plotting.fish_palette)
                    if boxplot == True:
                        sns.boxplot(data = df, palette = colour_scheme, ax = i)
                    if boxplot == False and scatter == True:
                        sns.stripplot(data = df, palette = colour_scheme, ax = i, alpha = .2)
                    elif scatter == True:
                        sns.stripplot(data = df, palette = 'dark:k', ax = i, alpha = .5)
                    i.set_xticks([])
                    i.invert_xaxis()
        for ax in _ax.flat:
            ax.axhline(0, color = "grey", ls = "--")
        # Okay, now we need to figure out which columns to lower the oppacity on 
        # depending on if area == 0... Hold my beer:
        ## First let's find where we need to make changes 
        where_no_area = np.all(post_cluster_df.query(f"{group_by} == '{clust_id}'").filter(regex = "area_\d+$") == 0, axis = 0)
        index_true = np.where(where_no_area == True)[0]
        index_mapping = {"375" : 0, "422" : 1, "478" : 2, "588" : 3} # TODO make this more robust and handle nm labels automatically
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
        
def _plot_temporal_reconstruct(df, cluster_id_str, axs=None, parallel=None, scalebar = False,**kwargs):
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
    if scalebar is True:
        pygor.plotting.add_scalebar(2.5, ax = axs.flat[-1], flip_text = True, x = 1, line_width = 5)

def _plot_temporal_reconstruct_stack(df, cluster_id_str, axs=None, parallel=True, drop_surround = True, scalebar = False,**kwargs):
    # Figure out how many columns we need 
    chromatic_cols = df.filter(regex=r"_\d").columns
    unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Deal with passing axes    
    if axs is None:
        fig, axs = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    # Reconstruct the RFs (here optionally passing parallel workers)
    times_recons = pygor.strf.clustering.reconstruct.reconstruct_cluster_temporal(df, cluster_id_str, parallel=parallel)
    if drop_surround == True:
        # Get only the largest (we dont care about the surround component in these plots)
        times_recons = pygor.utilities.select_absmax(times_recons, axis = 1)
    # Loop over axes and plot, etc:
    for i in range(times_recons.shape[0]):
        axs.plot(times_recons[i].T,  color = pygor.plotting.fish_palette[i])
    axs.axis('off')    
    if scalebar is True:
        pygor.plotting.add_scalebar(2.5, ax = axs, flip_text = True, x = 1, line_width = 5)

def _imshow_temporal_reconstruct(df, cluster_id_str, axs=None, parallel=None, **kwargs):
    # Figure out how many columns we need
    chromatic_cols = df.filter(regex=r"_\d").columns
    unique_wavelengths = list(np.unique([i.split('_')[-1] for i in chromatic_cols]))
    # Deal with passing axes
    if axs is None:
        fig, axs = plt.subplots(1, len(unique_wavelengths))
    else:
        fig = plt.gcf()
    # Reconstruct the RFs (here optionally passing parallel workers)
    strf_recons = pygor.strf.clustering.reconstruct.fetch_cluster_strfs(df, cluster_id_str, parallel=parallel)
    strf_recons_avgd = np.average(strf_recons, axis = 1)
    # Loop over axes and plot, etc:
    for (n, ax) in enumerate(strf_recons_avgd):
        if n == 0:
            axs.flat[n].set_ylabel(cluster_id_str, rotation = 0,  labelpad = 20)
        pygor.strf.plot.spacetime_plot(ax = axs.flat[n], strf_arr = strf_recons_avgd[n], 
                cmap = pygor.plotting.custom.maps_concat[n], **kwargs)
        axs.flat[n].axis('off')
        # axs.flat[n].spines["top"].set_visible(False)
        # axs.flat[n].spines["bottom"].set_visible(False)
    #pygor.plotting.add_scalebar(10, ax = axs.flat[-1], flip_text = True, x = 1, line_width = 5)



def plot_spatial_reconstruct(clust_df, cluster_id_strings = None, parallel=True):
    if cluster_id_strings is None:
        cluster_id_strings = natsort.natsorted(pd.unique(clust_df["cluster_id"]).dropna())
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

def plot_spacetime_reconstruct(clust_df, cluster_id_strings, block_size = 200, jitter_div = 4,
                            time_durS = 1.3, scalebar_S = 0.3, scalebar_deg = 10, ipl_percentage = True,parallel=True, 
                            time = "1d",screen_size_deg = pix.screen_width_height_visang, norm_time_ylim = True):
    # Figure out how many columns we need 
    chromatic_cols = clust_df.filter(regex=r"_\d").columns
    unique_wavelengths = list(set([i.split('_')[-1] for i in chromatic_cols]))
    if isinstance(cluster_id_strings, str):
        cluster_id_strings = [cluster_id_strings]
    # Determine how many rows and columns
    if time == "1dstack":
        rows, columns = len(cluster_id_strings), len(unique_wavelengths) + 1
    else:
        rows, columns = len(cluster_id_strings), len(unique_wavelengths) * 2
    if ipl_percentage:
        columns += 1
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
            if time == "1d":
                for n, c_id in enumerate(cluster_id_strings):
                    _plot_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=worker)
            if time =="1dstack":
                for n, c_id in enumerate(cluster_id_strings):
                    _plot_temporal_reconstruct_stack(clust_df, c_id, axs=ax[n, 4], parallel=worker)
            elif time == "2d":
                for n, c_id in enumerate(cluster_id_strings):
                    _imshow_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=worker, aspect = "equal")
    # Otherwise pass None, which gets processed serially
    else:
        if time == "1d":
            for n, c_id in enumerate(cluster_id_strings):
                _plot_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=None)
        if time =="1dstack":
            for n, c_id in enumerate(cluster_id_strings):
                _plot_temporal_reconstruct_stack(clust_df, c_id, axs=ax[n, 4], parallel=None)
        elif time == "2d":
            for n, c_id in enumerate(cluster_id_strings):
                _imshow_temporal_reconstruct(clust_df, c_id, axs=ax[n, 4:8], parallel=None, aspect = "equal")
    if ipl_percentage is True:
        population_hist_vals_population = np.histogram(clust_df["ipl_depths"], bins = 10, range=(0, 100))[0]
        for i, clust in zip(ax[:, -1], cluster_id_strings):
            i.axhspan(-5, 55, color = "lightgrey", lw = 0)
            i.axhspan(60, 95, color = "lightgrey", lw = 0)
            percentage_hist_vals_condition = np.histogram(clust_df.query(f"cluster_id == '{clust}'")["ipl_depths"], bins = 10, range=(0, 100))[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                percentages = percentage_hist_vals_condition / population_hist_vals_population * 100
            i.barh(np.arange(0, 100, 10), width= percentages, height=10, color = 'b', edgecolor="black", alpha = .75)
            i.set_label("skip_this")
            i.set_xlim(0,40)
            i.set_axis_off()
    # Now post-process plot however you'd like:
    # Time scalebar
    time_plot_index = len(unique_wavelengths)
    time_len_xaxis =  np.abs(ax[0, time_plot_index].get_xlim()[0]) + ax[0, time_plot_index].get_xlim()[1] #xaxis has a -0.5 offset that this summation takes care of
    frame_s = time_durS / time_len_xaxis
    scalebar_time_len = scalebar_S / frame_s 
    pygor.plotting.add_scalebar(scalebar_time_len, string = f"{np.rint(scalebar_S * 1000).astype(int)} ms", ax = ax[-1, time_plot_index], x = 0, orientation = 'h', line_width = 3, text_size = 8, offset_modifier=.6)
    # Space scalebar 
    if isinstance(screen_size_deg, np.ndarray) is False:
        screen_size_deg = np.array(screen_size_deg)
    else:
        screen_size_deg = screen_size_deg
    space_len_xaxis =  np.abs(ax[0, 0].get_xlim()[0]) + ax[0, 0].get_xlim()[1] #xaxis has a -0.5 offset that this summation takes care of
    one_pix_visang = pix.visang_to_pix(space_len_xaxis, block_size=block_size)
    scalebar_space_len = scalebar_deg / one_pix_visang
    pygor.plotting.add_scalebar(scalebar_space_len, string = f"{np.rint(scalebar_deg).astype(int)}°", ax = ax[-1, 0], x = 0, orientation = 'h', line_width = 3, text_size = 8, offset_modifier=.6)
    if time == "2d":
        pygor.plotting.add_scalebar(scalebar_space_len, string = f"{np.rint(scalebar_deg).astype(int)}°", ax = ax[-1, time_plot_index], x = -.1, orientation = 'v', line_width = 3, text_size = 8, offset_modifier=.4)

    # Conditionally normalise axes
    if time == "1dstack":
        time_plot_index_last = time_plot_index + 1
    else:
        time_plot_index_last = time_plot_index + len(unique_wavelengths) - 1
    if norm_time_ylim is True and time != "2d":
        # Get all the limits for time axes, and find the max
        ylims = np.squeeze([i.get_ylim() for i in ax[:, time_plot_index:time_plot_index_last].flat])
        # Apply that to all of those axes
        for i in ax[:, time_plot_index:time_plot_index_last].flat:
            i.set_ylim(-np.abs(ylims.max()), np.abs(ylims.max()))
        pygor.plotting.add_scalebar(5, string = f"5 SD", ax = ax[-1, time_plot_index], x = 1, orientation = 'v', line_width = 3, text_size = 8, offset_modifier=.6)
    # Plot
    plt.show()
    return fig, ax
# def strf_summary():

def compare_clusters_conditions(df_0, df_1, metric = "area_"):
    # Mangle, reshape, and coerce the data into the right shape (ew I hate this)
    colour_categories = pd.unique(df_0.filter(like = f"{metric}").columns)
    df_0_reshaped = pd.melt(df_0[colour_categories.tolist() + ["cluster_id"]], id_vars = ["cluster_id"])
    df_0_reshaped = df_0_reshaped.mask(df_0_reshaped == 0, np.nan, inplace = False)
    df_0_reshaped = df_0_reshaped.rename(columns = {"variable":"colour", "value":"area"})
    df_1_reshaped = pd.melt(df_1[colour_categories.tolist() + ["cluster_id"]], id_vars = ["cluster_id"])
    df_1_reshaped = df_1_reshaped.mask(df_1_reshaped == 0, np.nan, inplace = False)
    df_1_reshaped = df_1_reshaped.rename(columns = {"variable":"colour", "value":"area"})
    df0 = df_0_reshaped
    df1 = df_1_reshaped
    df0 = df0.query("cluster_id != 'nan'")[~df0["cluster_id"].str.startswith("mix")]
    df1 = df1.query("cluster_id != 'nan'")[~df1["cluster_id"].str.startswith("mix")]
    df0["cluster_id"] = df0["cluster_id"].cat.remove_unused_categories()
    df1["cluster_id"] = df1["cluster_id"].cat.remove_unused_categories()
    # Generate the figure
    fig_scale = 15
    fig, ax = plt.subplots(4, 2, figsize = (fig_scale/1.5, fig_scale), sharex = False, sharey = True, gridspec_kw={'hspace': 0, 'wspace': .1})
    # Do the plotting
    for index, df in enumerate([df0, df1]):
        order = natsort.natsorted(df["cluster_id"].unique())
        for n, i in enumerate(reversed(pd.unique(df0["colour"]))):
            # Sort the order of the clusters
            # Standardise (but not share, semi-confusing) the x axis
            ax[n ,index].set_xticks(range(len(order)))
            ax[n, index].set_xticklabels(order, rotation = 90, fontsize = 8)
            # Plot the actual data
            sns.stripplot(data = df.query("colour == @i"), y = "area", x="cluster_id", orient="v", ax = ax[n, index], dodge = False, 
                    color=pygor.plotting.compare_conditions[2][index], alpha = 1, order = order, linewidth=.5, size = 3, zorder = 2)
            sns.boxplot(data = df.query("colour == @i"), y = "area", x="cluster_id", ax = ax[n, index], showfliers = False,
                    boxprops=dict(alpha=1), color = pygor.plotting.compare_conditions[2][index], order = order, zorder = 3)
            # Clean up the axes
            if n != 3:
                ax[n, index].set_xticks([])
                ax[n, index].set_xlabel("")
                ax[n, index].set_yticklabels([])
        # Shade background according to cluster categories
        for n, cax in enumerate(ax[:, index]):
            cax.axvspan(-1, len(order), color = pygor.plotting.fish_palette[n], alpha = .33, zorder = 0)
            cax.set_xlim(-.5, len(order)-.5)
            #cax.set_ylim(bottom = 0)
            on_locs = np.where(df["cluster_id"].cat.categories.str.contains("on") == True)[0]
            cax.axvspan(on_locs[0]-.5, on_locs[-1]+.5, color = 'white', alpha = .3, zorder = 0)
            off_locs = np.where(df["cluster_id"].cat.categories.str.contains("off") == True)[0]
            cax.axvspan(off_locs[0]-.5, off_locs[-1]+.5, color = 'k', alpha = .3, zorder = 0)
            opp_locs = np.where(df["cluster_id"].cat.categories.str.contains("opp") == True)[0]
            cax.axvspan(opp_locs[0]-.5, opp_locs[-1]+.5, color = 'grey', alpha = .07, zorder = 0)
            # Do other axis things
            cax.set_ylabel("")
    # Set the axis labels 
    ax[-1, 0].tick_params(axis='x', labelrotation=45)
    ax[-1, 1].tick_params(axis='x', labelrotation=45)
    ax[-1, 0].set_ylabel("Area (° vis. ang.$^2$)")
    ax[-1, 0].set_xlabel("Clusters (control)")
    ax[-1, 1].set_xlabel("Clusters (AC block)")
    pygor.plotting.scalebar.add_scalebar(100, ax = ax[-1, 0], rotation = 0, x = -.05, line_width = 5, string = "100")
    return fig, ax 