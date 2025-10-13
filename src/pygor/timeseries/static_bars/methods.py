import pygor.core.methods
import numpy as np
from typing import Union


def calculate_AB_deltas(
    self,
    roi: Union[int, list] = None,
    a_b_ratio: float = 0.5,
    ignore_percentage: float = 0.1,
) -> np.ndarray:
    # Get epoch markers in ms
    crop_points = pygor.core.methods.determine_epoch_markers_ms(self).astype(int)
    # Split calculated averages (in ms) to into A_B segments according to epoch markers and a_b_ratio
    split_list = np.split(self.averages, crop_points[1:], axis=1)
    # Determine number of rois
    num_rois = self.averages.shape[0] if roi is None else 1
    # Determine number of phases
    num_phases = len(split_list)
    # Loop over rois to calculate deltas per phase in epoch
    deltas = np.zeros((num_rois, num_phases))
    rois = range(self.averages.shape[0]) if roi is None else [roi]
    for i, r in enumerate(rois):
        for phase in range(num_phases):
            curr_arr = split_list[phase][r]
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
            # Assign deltas to array
            deltas[i, phase] = a_val - b_val
    return deltas if roi is None else deltas.flatten()

# def calculate_AFF_deltas(
#     self,
#     roi: Union[int, list] = None,
#     a_b_ratio: float = 0.5,
#     ignore_percentage: float = 0.1,
#     FFF_index = 7
# ):
    