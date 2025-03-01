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

def correlation_map(array_3d, border=0):
    """
    Calculate Pearson correlation between each pixel's time series and its neighbors.
    """
    if isinstance(array_3d, np.ma.MaskedArray):
        correlation_map = np.ma.masked_array(np.zeros((array_3d.shape[1], array_3d.shape[2])),
                                        mask=np.zeros((array_3d.shape[1], array_3d.shape[2]), dtype=bool))
    else:
        correlation_map = np.zeros((array_3d.shape[1], array_3d.shape[2]))

    for x in range(border, array_3d.shape[1] - border):
        for y in range(border, array_3d.shape[2] - border):
            if np.ma.is_masked(array_3d[:, x, y]):
                correlation_map.mask[x, y] = True
                continue  # Skip masked pixels
            
            # Central time series
            centre_pix = array_3d[:, x, y]

            # Collect valid neighbors safely
            neighbors = []
            for dx, dy in [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]:
                xn, yn = x + dx, y + dy
                if 0 <= xn < array_3d.shape[1] and 0 <= yn < array_3d.shape[2]:  # Bounds check
                    neighbors.append(array_3d[:, xn, yn])

            # Compute correlation coefficients
            if neighbors:
                corr_values = [np.corrcoef(centre_pix, n)[0, 1] for n in neighbors]
                correlation_map[x, y] = np.nanmean(corr_values)  # Handle NaNs if needed
    return correlation_map
