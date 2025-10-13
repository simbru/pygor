import imageio.v3 as iio
import scipy.ndimage
import numpy as np
import pathlib
import warnings
def write_gif(arr2d, out_path = None, upscale = 1, fps = None, percentile_range = [0, 100], total_duration = None, global_range = None):
    """
    Write a 2D array as an animated GIF.
    
    Parameters:
    -----------
    arr2d : array-like
        2D array where first dimension is time (frames)
    out_path : str or Path, optional
        Output file path. Defaults to "output_adaptive.gif"
    upscale : int, optional
        Factor to upscale each frame. Default is 1
    fps : float, optional
        Frames per second. Default is 50 if neither fps nor total_duration specified. Cannot be used with total_duration
    percentile_range : list, optional
        Percentile range for contrast scaling. Default is [0, 100]
    total_duration : float, optional
        Total duration of GIF in seconds. Cannot be used with fps
    global_range : tuple, optional
        Global min/max values for contrast scaling
    """
    video_array = arr2d
    # Adaptive contrast limits (percentile-based to avoid extreme clipping)
    if global_range is not None:
        # Calculate percentiles from data, then clamp to global range
        vmin_data, vmax_data = np.percentile(video_array, [percentile_range[0], percentile_range[1]])
        vmin = max(vmin_data, global_range[0])  # Don't go below global min
        vmax = min(vmax_data, global_range[1])  # Don't go above global max
    else:
        vmin, vmax = np.percentile(video_array, [percentile_range[0], percentile_range[1]])  # Avoids extreme outliers
    # Clip
    video_array_clipped = np.clip(video_array, vmin, vmax)
    # Scale to RGB space
    video_array = ((video_array_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    if upscale > 1:
        video_array = np.array([scipy.ndimage.zoom(i, upscale, order = 0) for i in video_array])
    # Save as GIF
    if out_path is None:
        out_path = pathlib.Path("output_adaptive.gif")
    else:
        out_path = pathlib.Path(out_path)
    if out_path.suffix != ".gif":
        out_path = out_path.with_suffix(".gif")
    # Handle duration calculation - either from fps or total_duration
    if fps is not None and total_duration is not None:
        raise ValueError("Cannot specify both fps and total_duration. Choose one.")
    
    if total_duration is not None:
        # Calculate per-frame duration from total duration
        num_frames = len(video_array)
        frame_duration_ms = (total_duration / num_frames) * 1000
    else:
        # Use fps to calculate per-frame duration
        if fps is None:
            fps = 50  # Default fps
        if fps > 50:
            warnings.warn("fps is higher than 50, gif playback likely to look slow.")
        frame_duration_ms = 1/fps * 1000
    iio.imwrite(out_path, video_array, duration=frame_duration_ms, loop=0, quantizer = "octree")