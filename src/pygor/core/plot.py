try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.plotting.basic

# Dependencies
import matplotlib.pyplot as plt
import numpy as np

def plot_averages(
    self, rois=None, 
    figsize=(None, None), 
    figsize_scale=None, 
    axs=None, 
    independent_scale = False, 
    n_rois_raster = 50,
    sort_order = None,
    skip_trigger = 1,
    phase_dur_mod = 1,
    kill_phase = None,
    include_snippets = True,
    text_size = None,
    **kwargs,
):
    """
    A function to plot the averages of specified regions of interest (rois) on separate subplots within a figure.

    Parameters
    ----------
    rois : Iterable, optional
        Regions of interest to plot. If not specified, all rois will be plotted.
    figsize : tuple, optional
        Size of the figure to plot the subplots. Default is calculated based on the number of rois.

    Keyword arguments
    ----------
    filter_by : Tuple, optional
        Tuple in format (function, "operator", value) where 'function' is a mathematical function that
        can be applied along axis = 1 for self.vverages, '"operator"' is a mathematical operator (e.g,
        "<", ">=", or "==") in string format, and 'value' is the threshold metric.
    sort_by : String, optional
        String representing attribute of data object, where the metric is ROI-by-ROI, such as a list
        or array where each element represents the metric of each ROI
    label_by : String, optional
        As above, but instead of changing the order of pygor.plotting.plots, changes the label associated with
        each ROI to be the specified metric.

    Returns
    -------
    None
    """
    if text_size is None:
        text_size = plt.rcParams['font.size']
    # Handle arguments, keywords, and exceptions
    if self.averages is None:
        raise AttributeError(
            "self.averages is nan, meaning it has likely not been generated"
        )
    if 'ax' in kwargs:
        axs = kwargs['ax']
    if rois is None:
        rois = np.arange(0, self.num_rois)

    if isinstance(rois, int) is True:
        rois = [rois]
    if isinstance(rois, Iterable) is False:
        rois = [rois]
    if isinstance(rois, np.ndarray) is False:
        rois = np.array(rois)
    if rois is not None and isinstance(rois, Iterable) is False:
        rois = np.array([rois])
    if "sort_by" in kwargs:
        rois = rois[
            np.argsort(self.__keyword_lables[kwargs["sort_by"]][rois].astype(int))
        ]
    if "label_by" in kwargs:
        roi_labels = self.__keyword_lables[kwargs["label_by"]][rois]
    else:
        roi_labels = rois
    if "filter_by" in kwargs:
        filter_result = self.__compare_ops_map[kwargs["filter_by"][1]](
            kwargs["filter_by"][0](self.averages, axis=1), kwargs["filter_by"][2]
        )
        rois = rois[filter_result]
    if figsize == (None, None):
        # determine if we are plotting raster or subplots
        if len(rois) >= n_rois_raster:
            # For rasters: aim for consistent pixel density with logarithmic damping
            # ~0.03 inches per ROI, damped for large numbers
            height = max(5, 0.03 * len(rois) * (1 / np.log10(len(rois) + 10)))
            figsize = (10, height)
        else:
            # scale inverse to number of rois such that more rois does not balloon the figure size
            figsize =  (10, 1 + 0.3 * len(rois))
    if figsize_scale is not None:
        figsize = np.array(figsize) * np.array(figsize_scale)
    if len(rois) < n_rois_raster:
        # Generate matplotlib plot
        # colormap = plt.cm.jet_r(np.linspace(1, 0, len(rois)))
        colormap = plt.cm.gist_rainbow(np.linspace(0, 1, len(rois)))
        # Handle axis input first
        provided_axs = axs is not None
        
        if axs is None:
            fig, axs = plt.subplots(
                len(rois), figsize=figsize, sharey=not independent_scale, sharex=True
            )
        else:
            # Convert single axis to array format FIRST
            if not hasattr(axs, 'flat'):
                axs = np.array([axs])
            # Get figure from provided axis
            fig = axs.flat[0].figure
            if independent_scale:
                for ax in axs.flat:
                    ax.get_shared_y_axes().remove(ax)
            
        # Handle single ROI case for internally created axes
        if not provided_axs and len(rois) == 1:
            axs = np.array([axs])
        sd_ratio_scalebar = 1
        phase_dur = self.ms_dur / self.trigger_mode * phase_dur_mod
        if sort_order is None:
            loop_through = enumerate(zip(axs.flat, rois, roi_labels))
        else:
            loop_through = enumerate(zip(axs.flat, rois[sort_order], roi_labels[sort_order]))
        # Build alternating x-axis shading edges based on stimulus timing
        x_max = len(self.averages[0])
        shade_edges = None
        try:
            markers_arr = self.calc_mean_triggertimes()
        except Exception:
            markers_arr = None
        if markers_arr is not None and len(markers_arr) > 0:
            markers_arr = np.sort(np.unique(markers_arr))
            shade_edges = np.concatenate(([0], markers_arr, [x_max]))
        else:
            if phase_dur_mod != 1:
                inner_loop = np.arange(self.trigger_mode * (1 / phase_dur_mod))
            else:
                inner_loop = np.arange(self.trigger_mode)
            if len(inner_loop) > 0:
                shade_edges = np.concatenate((inner_loop * phase_dur, [inner_loop[-1] * phase_dur + phase_dur]))

        for n, (ax, roi, label) in loop_through:
            if include_snippets is True and self.snippets is not None:
                ax.plot(self.snippets[roi].T, c="grey", alpha=0.5)
            ax.plot(self.averages[roi], color=colormap[n])
            ax.set_xlim(0, len(self.averages[0]))
            if independent_scale is False and self.snippets is not None:
                ax.set_ylim(np.min(self.snippets[rois]), np.max(self.snippets[rois]))
            else:
                if independent_scale:
                    if self.snippets is not None:
                        y_min = np.min(self.snippets[roi])
                        y_max = np.max(self.snippets[roi])
                    else:
                        y_min = np.min(self.averages[roi])
                        y_max = np.max(self.averages[roi])
                    if y_min == y_max:
                        y_min -= 1
                        y_max += 1
                    ax.set_ylim(y_min, y_max)
                closest_sd = np.ceil(np.max(self.averages[roi])*sd_ratio_scalebar)
                pygor.plotting.add_scalebar(closest_sd, string = f"{closest_sd.astype(int)} SD",ax=ax, flip_text=True, x=1.015, y = 0.1, text_size = text_size)
            ax.set_yticklabels([])
            ax.set_ylabel(label, rotation=0, verticalalignment="center", fontsize=plt.rcParams['font.size'])
            ax.spines[["top", "bottom", "right"]].set_visible(False)
            # Alternating shading along x-axis to show stimulus timing
            if shade_edges is not None:
                for i in range(0, len(shade_edges) - 1, 2):
                    ax.axvspan(
                        shade_edges[i],
                        shade_edges[i + 1],
                        alpha=0.12,
                        color="gray",
                        lw=0,
                        zorder=0,
                    )

            ax.grid(False)
        if independent_scale is False:
            closest_sd = np.ceil(np.max(np.abs(self.averages[rois])*sd_ratio_scalebar))
            pygor.plotting.add_scalebar(closest_sd, string = f"{closest_sd.astype(int)} SD",ax=axs.flat[-1], flip_text=True, x=1.015, y = 0.1, text_size = text_size)
        # ax.set_xlabel("Time (ms)")
        fig.subplots_adjust(hspace=0)
        cax = axs.flat[-1]
        cax.set_xticks(np.ceil(cax.xaxis.get_majorticklocs()), np.ceil(cax.xaxis.get_majorticklocs() / 1000))
        cax.set_xlim(0, len(self.averages[0]))
        cax.set_xlabel("Time (s)", fontsize=plt.rcParams['font.size'])
        cax.tick_params(labelsize=plt.rcParams['font.size'])
        return fig, axs
    else:
        if sort_order is None:
            rois = rois
        else:
            rois = rois[sort_order]
        # Generate raster plot
        # Add vertical lines for average trigger times
        markers_arr = self.calc_mean_triggertimes() 
        # Use the passed figsize parameter instead of hardcoded (5, 3)
        if axs is None:
            fig, axs = plt.subplots(1, figsize=figsize)
            axs = np.array([axs])  # Make it consistent with the other branch
        else:
            # Use provided axis - ensure it's in array format
            if not hasattr(axs, 'flat'):
                axs = np.array([axs])
            fig = axs.flat[0].figure
        ax = axs.flat[0]  # Get the axis for plotting
        # import sklearn.preprocessing
        # scaler = sklearn.preprocessing.MaxAbsScaler()
        arr = self.averages
        # arr = scaler.fit_transform(arr)
        maxabs = np.max(np.abs(arr))
        if rois is not None:
            arr = arr[rois]
        if "clim" in kwargs:
            img = ax.imshow(arr, aspect="auto", cmap = "Greys_r", clim = kwargs["clim"], interpolation="None")
        else:
            img = ax.imshow(arr, aspect="auto", cmap = "Greys_r", interpolation="None")
        # ax.set_xticklabels(np.round(load.triggertimes, 3))
        for i in markers_arr:
            ax.axvline(x=i, color="r", alpha=0.5)
            # ax.axvline(x=i + (avg_epoch_dur * (1 / load.linedur_s)/load.trigger_mode)/2, color="blue", alpha=0.5)
        plt.colorbar(img)
        ax.set_xticks(np.ceil(ax.xaxis.get_majorticklocs()), np.ceil(ax.xaxis.get_majorticklocs() / 1000))
        ax.set_xlim(0, len(self.averages[0]))
        ax.set_xlabel("Time (s)", fontsize=plt.rcParams['font.size'])
        return fig, axs
    # plt.tight_layout()
    
    

