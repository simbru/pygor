import numpy as np

def determine_epoch_markers_ms(self):
    """

    Parameters
    ----------
    None

    Returns
    -------
    markers_arr : np.ndarray
        An array of the time of each marker in seconds, relative to the first marker.
    """
    # Figure out how long the average epoch is (epoch is defined by trigger mode)
    # and represents one set of stimuli in a given stimulus loop.
    avg_epoch_dur = np.average(np.diff(self.triggertimes.reshape(-1, self.trigger_mode)[:, 0]))
    # Reshape trigger times into epochs
    epoch_reshape = self.triggertimes.reshape(-1, self.trigger_mode)
    # Loop through and subtract the average epoch duration, to get the time deltas in each epoch
    temp_arr = np.empty(epoch_reshape.shape)
    for n, i in enumerate(epoch_reshape):
        temp_arr[n] = i - (avg_epoch_dur * n)    
    # Average the trigger times in each epoch, to generate the average epoch trigger times
    avg_epoch_triggertimes = np.average(temp_arr, axis=0)
    # Divide the average epoch trigger times by the line duration, to get the marker times in ms
    markers_ms_arr = avg_epoch_triggertimes * (1 / self.linedur_s)
    # Subtract the first marker time, to remove pre-start time from epoch trigger tiems
    markers_ms_arr -= markers_ms_arr[0]
    return markers_ms_arr