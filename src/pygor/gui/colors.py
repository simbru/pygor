"""Colour palette for ROI labels.

napari's default label colormap includes desaturated entries that read as
grey against a greyscale image stack. Hues here are stepped by the golden
ratio for separation and kept at high saturation, so no entry can come out
grey.
"""

import colorsys

import numpy as np

# Every colour is generated at one of these (saturation, value) pairs.
# Saturation is bounded well away from zero, which is what greyness is.
_TONES = ((0.95, 1.00), (0.95, 0.70), (0.60, 1.00), (0.75, 0.85))

# Reciprocal golden ratio: successive hues land far apart on the wheel
_HUE_STEP = 0.61803398875

MIN_SATURATION = 0.55


def roi_colors(n=48):
    """Return ``n`` distinct, non-grey RGBA colours as a float array."""
    colors = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        hue = (i * _HUE_STEP) % 1.0
        saturation, value = _TONES[i % len(_TONES)]
        colors[i, :3] = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i, 3] = 1.0
    return colors


def roi_colormap(n=48):
    """Build a cyclic label colormap with a transparent background entry."""
    from napari.utils.colormaps import CyclicLabelColormap

    colors = np.vstack([np.zeros((1, 4), dtype=np.float32), roi_colors(n)])
    return CyclicLabelColormap(colors=colors, display_name="pygor_rois")
