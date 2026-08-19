"""Colour palette for ROI labels.

The palette spans a rainbow colormap sampled once per ROI, so the colours
stay as far apart as the ROI count allows and the sequence reads in a
predictable order. It is resampled whenever ROIs are added, which does
mean existing ROIs change colour as the count grows.

``gist_rainbow`` is used rather than ``rainbow``: it is fully saturated
across its whole range, so no ROI can come out grey against the greyscale
image stack, whereas ``rainbow`` drops to 0.36 saturation in its
cyan-green region.
"""

import numpy as np

DEFAULT_CMAP = "gist_rainbow"

# Any sample from the default colormap sits far above this
MIN_SATURATION = 0.55

# napari maps label i to colors[1 + (i - 1) % n], so index 0 is background
_BACKGROUND = np.zeros((1, 4), dtype=np.float32)


def roi_colors(n_rois, cmap_name=DEFAULT_CMAP):
    """Sample a colormap once per ROI, returning an (n, 4) RGBA array."""
    import matplotlib

    n = max(int(n_rois), 1)
    positions = np.linspace(0, 1, n)
    return matplotlib.colormaps[cmap_name](positions).astype(np.float32)


def roi_colormap(n_rois, cmap_name=DEFAULT_CMAP):
    """Build a label colormap covering ``n_rois`` ROIs plus a background."""
    from napari.utils.colormaps import CyclicLabelColormap

    colors = np.vstack([_BACKGROUND, roi_colors(n_rois, cmap_name)])
    return CyclicLabelColormap(colors=colors, display_name="pygor_rois")


def apply_roi_colormap(labels_layer, n_rois=None):
    """Resample the palette so it spans exactly the ROIs present.

    Rebuilding the colormap is not free, so this is a no-op when the ROI
    count has not moved. The count is remembered on the layer's metadata
    rather than tracked by the caller, so any code path that adds ROIs
    gets the update by calling this.
    """
    if labels_layer is None:
        return None

    if n_rois is None:
        n_rois = int(np.asarray(labels_layer.data).max())
    n_rois = max(int(n_rois), 1)

    if labels_layer.metadata.get("pygor_palette_n") == n_rois:
        return labels_layer.colormap

    labels_layer.colormap = roi_colormap(n_rois)
    labels_layer.metadata["pygor_palette_n"] = n_rois
    return labels_layer.colormap


def apply_metric_colormap(labels_layer, roi_labels, values, cmap_name="viridis"):
    """Colour ROIs by a per-ROI metric rather than by identity.

    Values are normalised across the ROIs present. ROIs whose value is not
    finite are drawn transparent, so a metric that could not be computed
    for a cell shows as absent rather than as an extreme.

    The identity palette is restored by calling :func:`apply_roi_colormap`;
    the remembered palette size is cleared so that call rebuilds it.
    """
    import matplotlib
    from napari.utils.colormaps import DirectLabelColormap

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        low, high = float(values[finite].min()), float(values[finite].max())
    else:
        low, high = 0.0, 1.0
    span = (high - low) or 1.0

    cmap = matplotlib.colormaps[cmap_name]
    color_dict = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
    for label, value in zip(roi_labels, values):
        if np.isfinite(value):
            color_dict[int(label)] = tuple(float(c) for c in cmap((value - low) / span))
        else:
            color_dict[int(label)] = (0.0, 0.0, 0.0, 0.0)

    labels_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    labels_layer.metadata["pygor_palette_n"] = None
    return low, high
