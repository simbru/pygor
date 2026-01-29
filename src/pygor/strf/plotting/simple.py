import pygor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections.abc import Iterable


def _normalize_roi_indices(roi, num_slices):
    if roi is None:
        return list(range(num_slices))
    if isinstance(roi, (int, np.integer)):
        roi_indices = [int(roi)]
    elif isinstance(roi, Iterable) and not isinstance(roi, (str, bytes)):
        roi_indices = [int(r) for r in roi]
    else:
        raise TypeError("roi must be None, an int, or an iterable of ints")
    for r in roi_indices:
        if r < 0 or r >= num_slices:
            raise ValueError(f"roi {r} out of range [0, {num_slices - 1}]")
    return roi_indices


def _build_grid_image(array, max_x):
    num_slices = array.shape[0]
    num_rows = int(np.ceil(num_slices / max_x)) if num_slices > 0 else 0
    if num_rows == 0:
        return np.zeros((0, 0)), 0, 0
    empty_slices = num_rows * max_x - num_slices
    if np.ma.isMaskedArray(array):
        pad_array = np.ma.masked_all((empty_slices, array.shape[1], array.shape[2]))
        array_padded = (
            np.ma.vstack([array, pad_array]) if empty_slices > 0 else array
        )
    else:
        pad_array = np.zeros((empty_slices, array.shape[1], array.shape[2]))
        array_padded = np.vstack([array, pad_array]) if empty_slices > 0 else array
    array_grid = array_padded.reshape(num_rows, max_x, array.shape[1], array.shape[2])
    blocks = [[array_grid[i, j] for j in range(max_x)] for i in range(num_rows)]
    if np.ma.isMaskedArray(array_grid):
        row_blocks = [np.ma.hstack(row) for row in blocks]
        image = np.ma.vstack(row_blocks)
    else:
        image = np.block(blocks)
    return image, num_rows, num_slices


def _normalize_max_x(max_x):
    if isinstance(max_x, (int, np.integer)) and max_x > 0:
        return int(max_x)
    raise ValueError("max_x must be a positive int")


def _symmetric_cval(image, cval):
    if cval is not None:
        return cval
    if np.ma.isMaskedArray(image):
        data = image.compressed()
        if data.size == 0:
            raise ValueError("no unmasked data available to compute color limits")
        percentile = np.percentile(data, [1, 99])
    else:
        percentile = np.percentile(image, [1, 99])
    return max(abs(percentile[0]), abs(percentile[1]))


def _upper_cval(image, cval):
    if cval is not None:
        return cval
    if np.ma.isMaskedArray(image):
        data = image.compressed()
        if data.size == 0:
            raise ValueError("no unmasked data available to compute color limits")
        return np.percentile(data, 99)
    return np.percentile(image, 99)


def _masked_minmax(image):
    if np.ma.isMaskedArray(image):
        data = image.compressed()
        if data.size == 0:
            raise ValueError("no unmasked data available to compute color limits")
        return float(np.min(data)), float(np.max(data))
    return float(np.min(image)), float(np.max(image))


def _masked_percentile(image, lo, hi):
    if np.ma.isMaskedArray(image):
        data = image.compressed()
        if data.size == 0:
            raise ValueError("no unmasked data available to compute color limits")
        return tuple(np.percentile(data, [lo, hi]))
    return tuple(np.percentile(image, [lo, hi]))


def _masked_mad(image):
    if np.ma.isMaskedArray(image):
        data = image.compressed()
    else:
        data = image.ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("no valid data available to compute MAD")
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return float(median), float(mad)


def _weighted_mad(values, weights):
    values = values.ravel()
    weights = weights.ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        raise ValueError("no valid data available to compute weighted MAD")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    median = np.interp(0.5, cdf, values)
    abs_dev = np.abs(values - median)
    order = np.argsort(abs_dev)
    abs_dev = abs_dev[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    mad = np.interp(0.5, cdf, abs_dev)
    return float(median), float(mad)


def _weighted_percentile(values, weights, lo, hi):
    values = values.ravel()
    weights = weights.ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        raise ValueError("no valid data available to compute weighted percentiles")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    lo_q, hi_q = lo / 100.0, hi / 100.0
    vmin = np.interp(lo_q, cdf, values)
    vmax = np.interp(hi_q, cdf, values)
    return float(vmin), float(vmax)


def _center_per_slice(array, center):
    if np.ma.isMaskedArray(array):
        if center == "mean":
            centres = np.ma.mean(array, axis=(1, 2))
        else:
            centres = np.ma.median(array, axis=(1, 2))
        centres = np.ma.filled(centres, 0.0)
    else:
        if center == "mean":
            centres = np.nanmean(array, axis=(1, 2))
        else:
            centres = np.nanmedian(array, axis=(1, 2))
        centres = np.nan_to_num(centres, nan=0.0)
    return array - centres[:, None, None], centres


def _weighted_center_per_slice(array, weights):
    """Center each slice using weighted median.

    Parameters
    ----------
    array : ndarray
        Shape (n_rois, height, width).
    weights : ndarray
        Same shape as array, weights for each pixel.

    Returns
    -------
    centered : ndarray
        Array with weighted median subtracted per slice.
    centres : ndarray
        The weighted median for each slice.
    """
    n_rois = array.shape[0]
    centres = np.zeros(n_rois)
    centered = array.copy()

    for i in range(n_rois):
        roi_slice = array[i]
        w = weights[i]
        # Use _weighted_mad to get the weighted median (first return value)
        weighted_median, _ = _weighted_mad(roi_slice, w)
        centres[i] = weighted_median
        centered[i] = roi_slice - weighted_median

    return centered, centres


def _normalize_per_roi(array, scale_k=2, alpha_weights=None):
    """Normalize each ROI independently to [-1, 1] using consistent weighted statistics.

    When alpha_weights are provided, uses BOTH weighted median (for centering)
    AND weighted MAD (for scaling). This ensures the normalization is consistent -
    the center and spread are computed from the same weighted distribution.

    Parameters
    ----------
    array : ndarray
        Shape (n_rois, height, width). Will be re-centered using weighted median.
    scale_k : float
        Multiplier for MAD to define normalization range. scale_k=2 means
        values within ±2*MAD map to ±1.
    alpha_weights : ndarray or None
        Weights for each pixel (e.g., from collapsed STRF magnitude).
        Higher weights = more influence on centering and scale calculation.

    Returns
    -------
    normalized : ndarray
        Same shape, values centered and scaled based on high-signal pixels.
    scales : ndarray
        Per-ROI scale factors (scale_k * MAD) used for normalization.
    """
    n_rois = array.shape[0]
    if np.ma.isMaskedArray(array):
        normalized = np.ma.array(array.copy())
    else:
        normalized = array.copy()
    scales = np.zeros(n_rois)

    for i in range(n_rois):
        roi_slice = array[i]

        if alpha_weights is not None:
            # Use weighted median and MAD - both from same weighted distribution
            weights = alpha_weights[i]
            weighted_median, mad = _weighted_mad(roi_slice, weights)
            # Re-center using weighted median (not the unweighted one from earlier)
            roi_centered = roi_slice - weighted_median
        else:
            # Unweighted fallback
            median, mad = _masked_mad(roi_slice)
            roi_centered = roi_slice - median

        scale = scale_k * mad if mad > 0 and np.isfinite(mad) else 1.0

        scales[i] = scale
        normalized[i] = np.clip(roi_centered / scale, -1, 1)

    return normalized, scales


def _annotate_grid(ax, roi_indices, num_rows, max_x, h, w):
    num_slices = len(roi_indices)
    for i in range(num_rows):
        for j in range(max_x):
            slice_idx = i * max_x + j
            if slice_idx >= num_slices:
                continue
            x_pos = j * w + 1
            y_pos = i * h + 1
            ax.text(
                x_pos,
                y_pos,
                str(roi_indices[slice_idx]),
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )



def plot_collapsed_strfs(
    self,
    cval=None,
    channel=None,
    cmap="bwr",
    origin="upper",
    roi=None,
    max_x=10,
    show_cbar=True,
    cbar_kwargs=None,
    show_labels=True,
):
    array = self.collapse_times(force_recompute=True)
    if channel is not None:
        array = pygor.utilities.multicolour_reshape(array, channel)[channel - 1]
    roi_indices = _normalize_roi_indices(roi, array.shape[0])
    if not roi_indices:
        raise ValueError("roi selection is empty")
    array = array[roi_indices]

    # Grid layout
    max_x = _normalize_max_x(max_x)
    image, num_rows, num_slices = _build_grid_image(array, max_x)

    cval = _symmetric_cval(image, cval)

    # Display
    fig, ax = plt.subplots(figsize=(max_x, max(2, num_rows)))
    im = ax.imshow(image, cmap=cmap, interpolation="none", clim=(-cval, cval), origin=origin)
    ax.axis("off")
    if show_cbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)

    # Overlay slice numbers
    if show_labels:
        h, w = array.shape[1], array.shape[2]
        _annotate_grid(ax, roi_indices, num_rows, max_x, h, w)
    return fig, ax


def plot_peaktime_strfs(
    self,
    roi=None,
    channel=None,
    use_segmentation=False,
    seg_kwargs=None,
    cmap="jet",
    clim=None,
    origin="upper",
    max_x=10,
    show_cbar=True,
    cbar_kwargs=None,
    show_labels=True,
    bad_color="black",
    alpha_mode="strf_abs",
):
    """Plot raw peak timing values for each ROI.

    Shows when each pixel's response peaked (absolute time, no centering).

    Parameters
    ----------
    roi : int, list of int, or None
        ROI indices to plot. None plots all ROIs.
    channel : int or None
        Color channel to plot (1-indexed). None uses all channels.
    use_segmentation : bool
        If True, mask pixels outside the segmented receptive field.
    seg_kwargs : dict or None
        Keyword arguments for get_centre_only_seg().
    cmap : str
        Colormap name.
    clim : tuple of (vmin, vmax) or None
        Color limits. None auto-scales to data range.
    origin : str
        Image origin ('upper' or 'lower').
    max_x : int
        Maximum ROIs per row in grid.
    show_cbar : bool
        Whether to show colorbar.
    cbar_kwargs : dict or None
        Keyword arguments for colorbar.
    show_labels : bool
        Whether to show ROI number labels.
    bad_color : str
        Color for masked pixels.
    alpha_mode : str
        Alpha transparency mode. "strf_abs" uses collapsed STRF magnitude.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    array = self.get_strf_peak_times()
    if channel is not None:
        array = pygor.utilities.multicolour_reshape(array, channel)[channel - 1]
    roi_indices = _normalize_roi_indices(roi, array.shape[0])
    if not roi_indices:
        raise ValueError("roi selection is empty")
    array = array[roi_indices]

    if use_segmentation:
        seg_kwargs = {} if seg_kwargs is None else seg_kwargs
        seg_masks = self.get_centre_only_seg(**seg_kwargs)[roi_indices]
        seg_masks = seg_masks.astype(bool)
        if seg_masks.shape != array.shape:
            raise ValueError("segmentation masks shape does not match data array")
        empty_masks = np.sum(seg_masks, axis=(1, 2)) == 0
        if np.any(empty_masks):
            seg_masks[empty_masks] = True
        array = np.ma.array(array, mask=~seg_masks)

    # Compute alpha weights for transparency
    alpha = None
    if alpha_mode == "strf_abs":
        alpha_array_raw = self.get_amplitude_weights()
        if channel is not None:
            alpha_array_raw = pygor.utilities.multicolour_reshape(alpha_array_raw, channel)[
                channel - 1
            ]
        alpha_array_raw = alpha_array_raw[roi_indices]
        if alpha_array_raw.shape == array.shape:
            finite_alpha = alpha_array_raw[np.isfinite(alpha_array_raw)]
            if finite_alpha.size > 0:
                alpha_lo, alpha_hi = np.percentile(finite_alpha, (50, 95))
                if alpha_hi > alpha_lo:
                    alpha_array = np.clip(alpha_array_raw, alpha_lo, alpha_hi)
                    alpha_array = (alpha_array - alpha_lo) / (alpha_hi - alpha_lo)
                    alpha_array = np.clip(alpha_array, 0.0, 1.0)
                    alpha_array = np.nan_to_num(alpha_array, nan=0.0, posinf=1.0, neginf=0.0)
                    alpha_array = np.clip(alpha_array, 0.0, 1.0)
                    alpha, _, _ = _build_grid_image(alpha_array, max_x)

    # Grid layout
    max_x = _normalize_max_x(max_x)
    image, num_rows, num_slices = _build_grid_image(array, max_x)

    # Color limits
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = _masked_minmax(image)

    # Display
    fig, ax = plt.subplots(figsize=(max_x, max(2, num_rows)))
    if np.ma.isMaskedArray(image):
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color=bad_color)
    else:
        cmap_obj = cmap
    if alpha is not None:
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    if alpha is not None and np.ma.isMaskedArray(image):
        mask = np.ma.getmaskarray(image)
        alpha = np.where(mask, 0.0, alpha)
    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
    im = ax.imshow(
        image,
        cmap=cmap_obj,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        alpha=alpha,
    )
    ax.axis("off")

    if show_cbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
        cbar.set_label("Peak time (s)")

    if show_labels:
        h, w = array.shape[1], array.shape[2]
        _annotate_grid(ax, roi_indices, num_rows, max_x, h, w)

    return fig, ax


def plot_deltatime_strfs(
    self,
    roi=None,
    channel=None,
    use_segmentation=False,
    seg_kwargs=None,
    scale="global",
    mad_scale=2,
    cmap="jet",
    clim=None,
    origin="upper",
    max_x=10,
    show_cbar=True,
    cbar_kwargs=None,
    show_labels=True,
    bad_color="black",
    alpha_mode="strf_abs",
):
    """Plot relative timing differences for each ROI.

    Shows timing differences relative to each ROI's weighted median,
    revealing temporal gradients within receptive fields.

    Parameters
    ----------
    roi : int, list of int, or None
        ROI indices to plot. None plots all ROIs.
    channel : int or None
        Color channel to plot (1-indexed). None uses all channels.
    use_segmentation : bool
        If True, mask pixels outside the segmented receptive field.
    seg_kwargs : dict or None
        Keyword arguments for get_centre_only_seg().
    scale : str
        Scaling mode:
        - "global": Shared centering and scale across ROIs, colorbar in time units
        - "global_centered": Per-ROI centering + global scale, colorbar in time units.
          Best of both worlds: removes baseline differences while keeping comparable scale.
        - "per_roi": Each ROI normalized independently to [-1, 1]
    mad_scale : float
        Colorbar range is ±(mad_scale × MAD). Used for "global" and "global_centered".
        Ignored for scale="per_roi".
    cmap : str
        Colormap name.
    clim : tuple of (vmin, vmax) or None
        Override auto scaling with explicit limits.
    origin : str
        Image origin ('upper' or 'lower').
    max_x : int
        Maximum ROIs per row in grid.
    show_cbar : bool
        Whether to show colorbar.
    cbar_kwargs : dict or None
        Keyword arguments for colorbar.
    show_labels : bool
        Whether to show ROI number labels.
    bad_color : str
        Color for masked pixels.
    alpha_mode : str
        Alpha transparency mode. "strf_abs" uses collapsed STRF magnitude.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if scale not in {"global", "global_centered", "per_roi"}:
        raise ValueError("scale must be 'global', 'global_centered', or 'per_roi'")

    # Get centered delta times from the data method
    array, _ = self.get_strf_delta_times(
        roi=roi, channel=channel,
        use_segmentation=use_segmentation, seg_kwargs=seg_kwargs,
    )
    # Ensure 3D even for single ROI
    if array.ndim == 2:
        array = array[np.newaxis]

    # Resolve roi_indices for alpha weights (must match array shape)
    all_weights = self.get_amplitude_weights()
    if channel is not None:
        all_weights = pygor.utilities.multicolour_reshape(all_weights, channel)[channel - 1]
    roi_indices = _normalize_roi_indices(roi, all_weights.shape[0])
    if not roi_indices:
        raise ValueError("roi selection is empty")

    # Compute alpha weights for display
    alpha = None
    alpha_array_raw = None
    if alpha_mode == "strf_abs":
        alpha_array_raw = all_weights[roi_indices]
        if alpha_array_raw.shape != array.shape:
            raise ValueError("alpha map shape does not match data array")
        # Normalized version for display alpha
        finite_alpha = alpha_array_raw[np.isfinite(alpha_array_raw)]
        if finite_alpha.size > 0:
            alpha_lo, alpha_hi = np.percentile(finite_alpha, (50, 95))
            if alpha_hi > alpha_lo:
                alpha_array = np.clip(alpha_array_raw, alpha_lo, alpha_hi)
                alpha_array = (alpha_array - alpha_lo) / (alpha_hi - alpha_lo)
                alpha_array = np.clip(alpha_array, 0.0, 1.0)
                alpha_array = np.nan_to_num(alpha_array, nan=0.0, posinf=1.0, neginf=0.0)
                alpha_array = np.clip(alpha_array, 0.0, 1.0)
                alpha, _, _ = _build_grid_image(alpha_array, max_x)

    # Display-specific scaling
    is_normalized = False
    per_roi_mads = None
    if scale == "per_roi":
        # Per-ROI normalization to [-1, 1]
        array, _ = _normalize_per_roi(array, mad_scale, alpha_array_raw)
        is_normalized = True
    elif scale == "global_centered":
        # Compute per-ROI MADs for color limit scaling
        per_roi_mads = []
        for i in range(array.shape[0]):
            roi_slice = array[i]
            if alpha_array_raw is not None:
                _, mad = _weighted_mad(roi_slice, alpha_array_raw[i])
            else:
                _, mad = _masked_mad(roi_slice)
            per_roi_mads.append(mad if np.isfinite(mad) else 0.0)
        per_roi_mads = np.array(per_roi_mads)

    # Grid layout
    max_x = _normalize_max_x(max_x)
    image, num_rows, num_slices = _build_grid_image(array, max_x)

    # Determine color limits
    if clim is not None:
        vmin, vmax = clim
    elif is_normalized:
        vmin, vmax = -1.0, 1.0
    elif scale == "global_centered" and per_roi_mads is not None:
        # Use median of per-ROI MADs as the scale
        # This represents "typical" within-ROI variance
        mad = np.median(per_roi_mads[per_roi_mads > 0]) if np.any(per_roi_mads > 0) else 0.0
        if mad == 0 or not np.isfinite(mad):
            vmin, vmax = _masked_minmax(image)
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs
        else:
            max_abs = mad_scale * mad
            vmin, vmax = -max_abs, max_abs
    else:
        # Global scaling with single weighted MAD across all data
        if alpha_array_raw is not None:
            _, mad = _weighted_mad(array, alpha_array_raw)
        else:
            _, mad = _masked_mad(image)
        if mad == 0 or not np.isfinite(mad):
            vmin, vmax = _masked_minmax(image)
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs
        else:
            max_abs = mad_scale * mad
            vmin, vmax = -max_abs, max_abs

    # Display
    fig, ax = plt.subplots(figsize=(max_x, max(2, num_rows)))
    if np.ma.isMaskedArray(image):
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color=bad_color)
    else:
        cmap_obj = cmap
    if alpha is not None:
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    if alpha is not None and np.ma.isMaskedArray(image):
        mask = np.ma.getmaskarray(image)
        alpha = np.where(mask, 0.0, alpha)
    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
    im = ax.imshow(
        image,
        cmap=cmap_obj,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        alpha=alpha,
    )
    ax.axis("off")

    if show_cbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
        if is_normalized:
            cbar.set_label("Relative timing (normalized)")
        else:
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs < 1:
                time_scale = 1000.0
                unit = "ms"
            else:
                time_scale = 1.0
                unit = "s"
            cbar.set_label(f"\u0394t ({unit})")
            cbar.formatter = mticker.FuncFormatter(
                lambda x, pos: f"{x * time_scale:g}"
            )
            cbar.update_ticks()

    if show_labels:
        h, w = array.shape[1], array.shape[2]
        _annotate_grid(ax, roi_indices, num_rows, max_x, h, w)

    return fig, ax


