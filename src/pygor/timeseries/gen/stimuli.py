import numpy as np

def bars_onoff_w_fff(stims_dims, trigger_mode, bars_n, min_val, max_val):
    """
    stims_dims : tuple
        (frames, width, height), for example (999, 4, 128)
    trigger_mode : int
    """
    # if orientaiton == "v":
    frames = stims_dims[0]
    stim_arr = np.ones((frames, stims_dims[1], stims_dims[2])) * min_val
    delta = np.floor(frames/trigger_mode).astype(int)
    bar_size = int(stims_dims[2] / bars_n)
    iterator = range(frames)[::delta]
    # ON bars
    for n, i in enumerate(iterator[:bars_n]):
        stim_arr[i:int(i+delta/2), :, n*bar_size:(n+1)*bar_size] = max_val
    # ON fff
    stim_arr[iterator[int(trigger_mode/2)-1]:, :, :] = max_val
    # OFF bars
    for n, i in enumerate(iterator[bars_n+1:]):
        stim_arr[i:int(i+delta/2), :, n*bar_size:(n+1)*bar_size] = min_val
    # OFF fff
    stim_arr[iterator[trigger_mode-1]:, :, :] = min_val
    return stim_arr