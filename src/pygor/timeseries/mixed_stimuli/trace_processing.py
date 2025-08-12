import numpy as np

def lowpass_filter(data, cutoff, fs, order=4):

    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    
    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

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
    triggertimes_ms = (inputdata.triggerstime_frame-inputdata.triggerstime_frame[0])/inputdata.frame_hz/inputdata.linedur_s*inputdata.n_planes
    mean_triggertimes_ms = flexible_reshape(triggertimes_ms, inputdata.trigger_mode)
    mean_triggertimes_ms = np.mean(mean_triggertimes_ms, axis=0)-np.mean(mean_triggertimes_ms, axis=0)[0]
    mean_triggertimes_extended = np.round(np.append(mean_triggertimes_ms, len(inputdata.averages.transpose()))).astype(int)
    print('Triggers at ms:', mean_triggertimes_extended)
    return mean_triggertimes_extended