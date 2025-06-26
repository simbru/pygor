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
    text_size = 10,
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
        figsize = (5*2, len(rois)*2)
    if figsize_scale is not None:
        figsize = np.array(figsize) * np.array(figsize_scale)
    if len(rois) < n_rois_raster:
        # Generate matplotlib plot
        colormap = plt.cm.jet_r(np.linspace(1, 0, len(rois)))
        if axs is None:
            fig, axs = plt.subplots(
                len(rois), figsize=figsize, sharey=True, sharex=True
            )
        else:
            fig = plt.gcf()
        # Loop through and plot wwithin axes
        if len(
            rois == 1
        ):  # This takes care of passing just 1 roi, not breaking axs.flat in the next line
            axs = np.array([axs])
        sd_ratio_scalebar = 0.5
        phase_dur = self.ms_dur / self.trigger_mode * phase_dur_mod
        """
        TODO: Fix so it doesn't just put shading 
        at regular intervals but instead looks at the
        trigger times. 
        """
        if sort_order is None:
            loop_through = enumerate(zip(axs.flat, rois, roi_labels))
        else:
            loop_through = enumerate(zip(axs.flat, rois[sort_order], roi_labels[sort_order]))
        for n, (ax, roi, label) in loop_through:
            if include_snippets is True:
                ax.plot(self.snippets[roi].T, c="grey", alpha=0.5)
            ax.plot(self.averages[roi], color=colormap[n])
            ax.set_xlim(0, len(self.averages[0]))
            if independent_scale is False:
                ax.set_ylim(np.min(self.snippets[rois]), np.max(self.snippets[rois]))
            else:
                closest_sd = np.ceil(np.max(self.averages[roi])*sd_ratio_scalebar)
                pygor.plotting.add_scalebar(closest_sd, string = f"{closest_sd.astype(int)} SD",ax=ax, flip_text=True, x=1.015, y = 0.1, text_size = text_size)
            ax.set_yticklabels([])
            ax.set_ylabel(label, rotation=0, verticalalignment="center")
            ax.spines[["top", "bottom", "right"]].set_visible(False)
            # Now we need to add axvspans (don't think I can avoid for loops inside for loops...)
            if phase_dur_mod != 1:
                inner_loop = np.arange(
                    self.trigger_mode * (1/phase_dur_mod)
                )
            else:
                inner_loop = np.arange(self.trigger_mode)
            if isinstance(kill_phase, Iterable) is False:
                kill_phase = [kill_phase]
            if self.trigger_mode != 1:
#               for i in self.get_average_markers():
#                   ax.axvline(i, color="k", lw=2, alpha = 0.33, zorder = 0)
                for interval in inner_loop[::2][::skip_trigger]:
                    ax.axvline(interval*phase_dur, color="k", lw=2, alpha = 0.2, zorder = 0)
                    if interval in kill_phase:
                        continue
                    ax.axvspan(
                        interval * phase_dur,
                        (interval + 1) * phase_dur,
                        alpha=0.2,
                        color="gray",
                        lw=0,
                    )

            ax.grid(False)
        if independent_scale is False:
            closest_sd = np.ceil(np.max(self.averages[rois])*sd_ratio_scalebar)
            pygor.plotting.add_scalebar(closest_sd, string = f"{closest_sd.astype(int)} SD",ax=axs.flat[-1], flip_text=True, x=1.015, y = 0.1, text_size = text_size)
        # ax.set_xlabel("Time (ms)")
        fig.subplots_adjust(hspace=0)
        cax = axs.flat[-1]
        cax.set_xticks(np.ceil(ax.xaxis.get_majorticklocs()), np.ceil(ax.xaxis.get_majorticklocs() / 1000))
        cax.set_xlim(0, len(self.averages[0]))
        cax.set_xlabel("Time (s)")
    else:
        if sort_order is None:
            rois = rois
        else:
            rois = rois[sort_order]
        # Generate raster plot
        avg_epoch_dur = np.average(np.diff(self.triggertimes.reshape(-1, self.trigger_mode)[:, 0]))
        epoch_reshape = self.triggertimes.reshape(-1, self.trigger_mode)
        temp_arr = np.empty(epoch_reshape.shape)
        for n, i in enumerate(epoch_reshape):
            temp_arr[n] = i - (avg_epoch_dur * n)
        avg_epoch_triggertimes = np.average(temp_arr, axis=0)
        markers_arr = avg_epoch_triggertimes * (1 / self.linedur_s)
        markers_arr -= markers_arr[0]
        fig, ax = plt.subplots(1,)
        # import sklearn.preprocessing
        # scaler = sklearn.preprocessing.MaxAbsScaler()
        arr = self.averages
        # arr = scaler.fit_transform(arr)
        maxabs = np.max(np.abs(arr))
        if rois is not None:
            print(rois)
            print(arr.shape)
            arr = arr[rois]
        img = ax.imshow(arr, aspect="auto", cmap = "Greys_r")
        # ax.set_xticklabels(np.round(load.triggertimes, 3))
        for i in markers_arr:
            ax.axvline(x=i, color="r", alpha=0.5)
            # ax.axvline(x=i + (avg_epoch_dur * (1 / load.linedur_s)/load.trigger_mode)/2, color="blue", alpha=0.5)
        plt.colorbar(img)
        ax.set_xticks(np.ceil(ax.xaxis.get_majorticklocs()), np.ceil(ax.xaxis.get_majorticklocs() / 1000))
        ax.set_xlim(0, len(self.averages[0]))
        ax.set_xlabel("Time (s)")
    # plt.tight_layout()
    return fig, axs

