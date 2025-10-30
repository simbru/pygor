import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rois_and_traces(rec, exp, df):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

    time = np.arange(len(exp.recording[0].averages.T))*exp.recording[0].linedur_s
    pref_cmap = plt.get_cmap('jet', df.shape[0])
    averages = exp.fetch_averages()

    if len(time) > averages.shape[1]:
        time = time[:averages.shape[1]]

    elif averages.shape[1] > len(time):
        averages = averages[:, :len(time)]

    # Show average image on left, spanning all rows
    ax_img = fig.add_subplot(gs[:, 0])
    avg_img = exp.recording[rec].average_stack
    p_low, p_high = np.percentile(avg_img, (2, 98))
    ax_img.imshow(avg_img, cmap='gray', origin='lower', vmin=p_low, vmax=p_high)

    # ROI label map for this recording
    rois_to_use = exp.recording[rec].rois
# subset DataFrame to this recording
    rec_idxs = np.where(df['recording_id'] == rec)[0]
    for rec_idx in rec_idxs:
        local_roi_id = int(df.iloc[rec_idx]['roi_id'])
        color = pref_cmap(rec_idx)
    # create RGBA overlay for this ROI (labels are negative)
        mask = (rois_to_use == -(local_roi_id+1))
        rgba = np.zeros(rois_to_use.shape + (4,), dtype=float)
        rgba[mask] = [*color[:3], 0.4]
        ax_img.imshow(rgba, origin='lower')
    ax_img.set_title(exp.recording[rec].name)
    ax_img.axis('off')

    #find which rec_idxs are less than the cutoff
    top_indices = rec_idxs[rec_idxs < int(0.2*df.shape[0])]
    top_rois = df.iloc[top_indices].index.values
    top_averages = averages[top_rois]
    average_trace = np.mean(top_averages, axis=0)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(time, top_averages.T, color='blue', alpha=0.1, linewidth=1)
    ax_top.plot(time, average_trace, color='blue', linewidth=2)

    #find which rec_idxs are intermediate
    mid_indices = rec_idxs[(rec_idxs >= int(0.375*df.shape[0])) & (rec_idxs < int(0.625*df.shape[0]))]
    mid_rois = df.iloc[mid_indices].index.values
    mid_averages = averages[mid_rois]
    average_trace = np.mean(mid_averages, axis=0)
    ax_mid = fig.add_subplot(gs[1, 1])
    ax_mid.plot(time, mid_averages.T, color='green', alpha=0.1, linewidth=1)
    ax_mid.plot(time, average_trace, color='green', linewidth=2)

    #find which rec_idxs are in the bottom 25%
    bot_indices = rec_idxs[rec_idxs >= int(0.8*df.shape[0])]
    bot_rois = df.iloc[bot_indices].index.values
    bot_averages = averages[bot_rois]
    average_trace = np.mean(bot_averages, axis=0)
    ax_bot = fig.add_subplot(gs[2, 1])
    ax_bot.plot(time, bot_averages.T, color='red', alpha=0.1, linewidth=1)
    ax_bot.plot(time, average_trace, color='red', linewidth=2)
    
    sns.despine()
    plt.tight_layout()
    plt.show()

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