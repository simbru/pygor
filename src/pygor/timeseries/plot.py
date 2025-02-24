import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Union

import pygor.core.methods

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

def plot_AB_delta(
    self,
    roi: Union[int, list] = 0,
    phase : int = 0,
    a_b_ratio: float = 0.5,
    ignore_percentage: float = 0.1,
) -> None: 
    if isinstance(roi, int) is False:
        raise TypeError("roi must be int")
    crop_points = pygor.core.methods.determine_epoch_markers_ms(self).astype(int)
    split_list = np.split(self.averages, crop_points[1:], axis = 1)
    # curr_arr = split_list[phase][roi]
    # A_len = int(curr_arr.shape[0] * a_b_ratio)
    # delta_period = int(ignore_percentage * A_len)
    # A = curr_arr[delta_period:A_len]
    # B = curr_arr[delta_period+A_len:]
    # A_val = np.average(A)
    # B_val = np.average(B)
    # delta = B_val - A_val
    curr_arr = split_list[phase][roi]
    phase_len = curr_arr.shape[0]
    # Determine the split between A and B part of phase
    a_len = int(a_b_ratio * phase_len)
    b_len = phase_len - a_len
    # Determine indices to be ignored (often rising/falling flanks)
    ignore_indices_a = int(ignore_percentage * a_len)
    ignore_indices_b = int(ignore_percentage * b_len)
    # Calculate A and B values, in this case mean
    a_val = np.mean(curr_arr[ignore_indices_a:a_len])
    b_val = np.mean(curr_arr[b_len + ignore_indices_b :])
    delta = a_val - b_val
    print(a_len, b_len)
    if delta != self.get_AB_deltas(roi)[phase]:
        print(delta, self.get_AB_deltas(roi)[phase])
        raise ValueError("Plotted delta value and calculated delta value do not match. Manual fix required.")
    fig, ax = plt.subplots(1, 2, figsize = (16*1, 4*1))
    mask = np.zeros(curr_arr.shape)
    mask[ignore_indices_a:a_len] = 1
    mask[b_len + ignore_indices_b :] = 1
    curr_arr = np.ma.masked_array(curr_arr, mask = mask)
    self.plot_averages(roi, axs = ax[0])
    ax[0].axvline(crop_points[phase])
    ax[0].axvline(crop_points[phase] + a_len*2)
    ax[1].plot(curr_arr, linestyle = "--", c = "C0")
    ax[1].plot(np.ma.masked_array(curr_arr.data, mask = np.logical_not(mask)), c = "C0")
    ax[1].vlines(a_len, a_val, b_val)
    ax[1].hlines(a_val, 0, a_len , zorder = 3)
    ax[1].hlines(b_val, a_len, len(curr_arr), zorder = 3)
    ax[1].text(.5, .85, r"$\Delta$ (A-B) = " + str(delta), ha = "center", va = "center", transform = ax[1].transAxes)
    fig.tight_layout()
    sns.despine()