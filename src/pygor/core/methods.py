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
    # Handle cases where stimulation was cut off mid-loop
    total_triggers = len(self.triggertimes)
    complete_epochs = total_triggers // self.trigger_mode
    
    if total_triggers % self.trigger_mode != 0:
        # Incomplete final epoch - truncate to complete epochs only
        valid_triggers = complete_epochs * self.trigger_mode
        triggertimes_complete = self.triggertimes[:valid_triggers]
        print(f"Warning: Stimulation appears to have been cut off mid-loop. "
              f"Using {complete_epochs} complete epochs out of {total_triggers} total triggers.")
    else:
        triggertimes_complete = self.triggertimes
    
    # Reshape trigger times into epochs
    epoch_reshape = triggertimes_complete.reshape(-1, self.trigger_mode)
    
    # Check if epoch timing is even or uneven (with tolerance for timing jitter)
    epoch_durations = np.diff(epoch_reshape[:, 0])
    timing_tolerance_ms = 0.1  # 100 microseconds tolerance for timing jitter
    max_deviation = np.max(epoch_durations) - np.min(epoch_durations)
    is_even_timing = max_deviation <= timing_tolerance_ms
    if is_even_timing:
        # Even timing - use original efficient method
        avg_epoch_dur = np.average(epoch_durations)
        temp_arr = np.empty(epoch_reshape.shape)
        for n, i in enumerate(epoch_reshape):
            temp_arr[n] = i - (avg_epoch_dur * n)
    else:
        # Uneven timing - normalize each epoch to remove cumulative drift
        print(f"Warning: Detected uneven epoch timing. Using epoch-by-epoch normalization.")
        temp_arr = np.empty(epoch_reshape.shape)
        cumulative_offset = 0
        
        for n, epoch_triggers in enumerate(epoch_reshape):
            if n == 0:
                # First epoch is the reference
                temp_arr[n] = epoch_triggers - epoch_triggers[0]
            else:
                # For subsequent epochs, subtract cumulative time offset
                epoch_start_offset = epoch_durations[:n].sum()
                temp_arr[n] = epoch_triggers - epoch_reshape[0, 0] - epoch_start_offset
    
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
