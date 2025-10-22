import numpy as np

def lowpass_filter(data, cutoff, fs, order=4):

    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    
    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def sort_by_response(traces, all_stimtypes, trigger_inds, stimtype1, opponent=False, stimtype2=None):
    '''
    function to sort the traces by their cumulative response to a given stimulus type
    '''
    stimtype1_startloc = all_stimtypes.index(stimtype1)

    response1 = traces[:, int(trigger_inds[stimtype1_startloc]):int(trigger_inds[stimtype1_startloc+1])]
    cumulative_response1 = response1.cumsum(axis=1)[:,-1]

    if opponent==True:
        
        stimtype2_startloc = all_stimtypes.index(stimtype2)
        response2 = traces[:, int(trigger_inds[stimtype2_startloc]):int(trigger_inds[stimtype2_startloc+1])]
        cumulative_response2 = response2.cumsum(axis=1)[:,-1]
        preference = cumulative_response1 - cumulative_response2
        sort_order = np.argsort(preference)

    else: 
        sort_order = np.argsort(cumulative_response1)

    return sort_order

def flexible_reshape(array, columns):
    # Calculate the number of elements to keep
    elements_to_keep = (len(array) // columns) * columns
    # Reshape the array
    reshaped_array = array[:elements_to_keep].reshape(-1, columns)
    return reshaped_array

def cluster_correlation(trace):
        if len(trace)<=1:
            average_correlation = 0 
        
        else: 
            correlation_matrix = np.corrcoef(trace.transpose(), rowvar=False)
            n = correlation_matrix.shape[0]
            upper_triangular = correlation_matrix[np.triu_indices(n, k=1)]
            average_correlation = np.mean(upper_triangular)
        return average_correlation

def z_normalize_columns(data):

    # Calculate the mean and standard deviation for each column
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    
    # Avoid division by zero in case any column has a standard deviation of zero
    stds[stds == 0] = 1
    
    # Z-normalize each column
    normalized_data = ((data.transpose() - means) / stds).transpose()
    
    return normalized_data

def calc_tracestats(trace, prestim_dur_inds, samplerate_inds, resample_factor):
        prestim_dur = int(prestim_dur_inds/resample_factor)
        baseline = np.mean(trace[0:prestim_dur])
        trace_corrected = trace-baseline
        pk_loc = np.argmax(abs(trace_corrected))
        amplitude = trace_corrected[pk_loc]
        area = np.cumsum(trace_corrected)[-1]
        pk_latency = pk_loc
        return amplitude, area, pk_latency

def sort_rows_by_correlation(data):
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(np.cumsum(data, axis=1))

    # Calculate mean correlation for each row
    mean_correlations = np.mean(correlation_matrix, axis=1)

    # Sort rows based on mean correlation
    sorted_indices = np.argsort(mean_correlations)[::-1]

    return sorted_indices

def get_triggertimes(inputdata):
    """calculates when triggers happen (in ms, which equals indices after resampling) in each stimulus loop, then averages to get the triggertimes 
    for the average response. then adds the last timepoint of the average. """
    triggertimes_ms = (inputdata.triggertimes_frame-inputdata.triggertimes_frame[0])/inputdata.frame_hz/inputdata.linedur_s*inputdata.n_planes
    mean_triggertimes_ms = flexible_reshape(triggertimes_ms, inputdata.trigger_mode)
    mean_triggertimes_ms = np.mean(mean_triggertimes_ms, axis=0)-np.mean(mean_triggertimes_ms, axis=0)[0]
    mean_triggertimes_extended = np.round(np.append(mean_triggertimes_ms, len(inputdata.averages.transpose()))).astype(int)
    print('Triggers at ms:', mean_triggertimes_extended)
    print('get_triggertimes is not accurate and will be disabled in a future release. use the .calc_mean_triggertimes_s() method instead')
    return mean_triggertimes_extended

def filter_ROIs(df, traces, quality='quality_indices>0.5', std_threshold=30):
    """
    Filters ROIs to include only with quality index > 0.5 (default) and below 
    a certain standard deviation (to remove too jittery ones).
    
    Parameters
    ----------
    
    traces : np.ndarray
        2D array of shape (n_rois, n_timepoints) containing the traces
    df : pd.DataFrame
        DataFrame containing ROI metadata, e.g. quality criterion, response amplitudes, roi sizes, which can be sorted by
    """
    df = df.query(quality)
    traces = traces[df.index.values]
    std_mask = np.where(np.std(traces, axis=1)<std_threshold)[0]
    traces = traces[std_mask]
    df = df.iloc[std_mask]
    df = df.reset_index(drop=True)  # Add this line to reset the index
    return traces, df

#TODO: should maybe add function to sort and filter dataframe in object already? might require storing df as a property. 