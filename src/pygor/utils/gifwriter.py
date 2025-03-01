import imageio.v3 as iio
import scipy.ndimage
import numpy as np
import pathlib
import warnings
def write_gif(arr2d, out_path = None, upscale = 1, fps = 50, percentile_range = [0, 100]):
    # video_array = bar.calculate_image_average()
    video_array = arr2d
    # Adaptive contrast limits (percentile-based to avoid extreme clipping)
    vmin, vmax = np.percentile(video_array, [percentile_range[0], percentile_range[1]])  # Avoids extreme outliers
    # Clip
    video_array_clipped = np.clip(video_array, vmin, vmax)
    # Scale to RGB space
    video_array = ((video_array_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    if upscale > 1:
        video_array = np.array([scipy.ndimage.zoom(i, 5, order = 0) for i in video_array])
    # Save as GIF
    if out_path is None:
        out_path = pathlib.Path("output_adaptive.gif")
    else:
        out_path = pathlib.Path(out_path)
    if out_path.suffix != ".gif":
        out_path = out_path.with_suffix(".gif")
    if fps > 50:
        warnings.warn("fps is higher than 50, gif playback likely to look slow.")
    duration = 1/fps * 1000 #ms
    iio.imwrite(out_path, video_array, duration=duration, loop=0, quantizer = "octree")