from dataclasses import dataclass
from typing import Optional
from pygor.classes.core_data import Core

import numpy as np
import matplotlib.pyplot as plt


@dataclass(kw_only=True, repr=False)
class CenterSurround(Core):

    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        #super().__dict__["data_types"].append(self.type)
        super().__post_init__()

    # def plot_phasic(self, roi=None, stims=None, bar_interval=1, plot_avg=False):
    #     """
    #     TODO
    #     - Docstring
    #     - Add bar_everyother
    #     """

    #     if stims == None:
    #         stims: int  # type annotation
    #         stims = self.trigger_mode
    #     if roi == None:
    #         times = self.averages
    #     else:
    #         times = self.averages[roi]
    #     if times.ndim == 1:
    #         times = np.array([times])
    #     for i in times:
    #         plt.plot(i, label=i)
    #         try:
    #             sections = np.split(i, stims * 2)
    #             # dur = times.shape[1]/2/stims
    #         except ValueError:
    #             len_min_remainder = len(i) - len(i) % (stims * 2 + 1)
    #             sections = np.split(i[:len_min_remainder], stims * 2 + 1)
    #             # dur = len_min_remainder
    #         if plot_avg == True:
    #             for i in range(len(sections)):
    #                 dur = 1
    #                 raise NotImplementedError("Not implemented yet")
    #                 point1 = [dur * i, dur * (i + 1)]
    #                 point2 = [np.average(sections[i]), np.average(sections[i])]
    #                 plt.plot(point1, point2, "-")
    #     print(stims / bar_interval)
    #     # for i in range(stims / bar_interval)[::bar_interval]:
    #     #     span_dur = snippets.shape[1]/stims
    #     #     plt.axvspan(span_dur * i, span_dur * (i+1) ,alpha = 0.25)

    #     # for i in range(stims * bar_interval)[::bar_interval]:
    #     #     print(i * dur)
    #     #     dur = times.shape[1]/stims/bar_interval
    #     #     plt.axvspan(dur * i, dur * (i+1) ,alpha = 0.25)
    #     plt.axhline(0, c="grey", ls="--")

    def split_trials(self, n_trials: Optional[int] = None):
        if n_trials is None:
            n_trials = self.trigger_mode
        n_roi, n_time = self.averages.shape
        n_time = n_time - (n_time % n_trials)
        return self.averages[:, :n_time].reshape(n_roi, n_trials, -1)

    def get_tuning_curves(self, baseline_window: tuple, response_window: tuple, n_trials:Optional[int] = None):
        if n_trials is None:
            n_trials = self.trigger_mode
        trials = self.split_trials(n_trials=n_trials)
        baselines = np.mean(trials[:, :, baseline_window[0]:baseline_window[1]], axis=-1)
        responses = np.mean(trials[:, :, response_window[0]:response_window[1]], axis=-1)
        tuning_curves = responses - baselines
        return tuning_curves
        # Split averages into groups and stimuli, then compute mean response and baseline for each ROI and stimulus, then compute tuning metric as response minus baseline (already in SD units)

    def get_tuning_curves_by_group(self, group_size: int, baseline_window: tuple, response_window: tuple, n_trials:Optional[int] = None):
        # Split averages into groups and stimuli, then compute mean response and baseline for each ROI and stimulus, then compute tuning metric as response minus baseline (already in SD units)
        tuning_curves = self.get_tuning_curves(baseline_window=baseline_window, response_window=response_window, n_trials=n_trials)
        # Reshape to get [ROI, group, stimuli] shape for predictability
        tuning_curves_grouped = tuning_curves.reshape(tuning_curves.shape[0], group_size, -1)
        return tuning_curves_grouped