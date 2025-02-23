import numpy as np

def markers(self):
    """
    Calculate the time of each marker in seconds, relative to the first marker.

    The first step is to calculate the average time difference between epochs. This is done by taking the average of the differences between the first element of each epoch in the reshaped triggertimes array.

    Then, the reshaped triggertimes array is iterated over, and each epoch is subtracted by the average epoch duration multiplied by the epoch number. This essentially "un-reshapes" the array, and gives the time of each marker relative to the first marker in the first epoch.

    The average of the "un-reshaped" array is then taken along the first axis, to get the average time of each marker in seconds.

    Finally, the first element of the average array is subtracted from the rest of the array, to get the time of each marker relative to the first marker.

    Parameters
    ----------
    None

    Returns
    -------
    markers_arr : np.ndarray
        An array of the time of each marker in seconds, relative to the first marker.
    """
    
    avg_epoch_dur = np.average(np.diff(self.triggertimes.reshape(-1, self.trigger_mode)[:, 0]))
    epoch_reshape = self.triggertimes.reshape(-1, self.trigger_mode)
    temp_arr = np.empty(epoch_reshape.shape)
    for n, i in enumerate(epoch_reshape):
        temp_arr[n] = i - (avg_epoch_dur * n)
    avg_epoch_triggertimes = np.average(temp_arr, axis=0)
    markers_arr = avg_epoch_triggertimes * (1 / self.linedur_s)
    markers_arr -= markers_arr[0]
    return markers_arr