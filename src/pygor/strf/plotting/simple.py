from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pygor


def _validate_channel(channel, n_colours):
    """Validate 0-indexed channel parameter."""
    if channel is not None:
        if not isinstance(channel, (int, np.integer)):
            raise TypeError(
                f"channel must be an int or None, got {type(channel).__name__}"
            )
        if channel < 0 or channel >= n_colours:
            raise ValueError(
                f"channel={channel} out of range. Use 0-{n_colours - 1} "
                f"(0-indexed) or None for all channels."
            )


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
        array_padded = np.ma.vstack([array, pad_array]) if empty_slices > 0 else array
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


# ---------------------------------------------------------------------------
# Colour-mosaic helpers (fast single-imshow rendering for multicolour data)
# ---------------------------------------------------------------------------


def _compute_mosaic_row_limits(
    data_4d, scale, mad_scale=2, alpha_weights_4d=None, clim=None
):
    """Compute per-ROI (vmin, vmax) colour limits for a 4D colour mosaic.

    Parameters
    ----------
    data_4d : ndarray, shape (n_colours, n_rois, h, w)
    scale : {"per_roi", "global", "global_centered"}
    mad_scale : float
    alpha_weights_4d : ndarray or None, same shape as data_4d
    clim : tuple (vmin, vmax) or None

    Returns
    -------
    list of (vmin, vmax) tuples, length n_rois.
    """
    n_rois = data_4d.shape[1]

    if clim is not None:
        return [clim] * n_rois

    if scale == "per_roi":
        row_limits = []
        for roi in range(n_rois):
            roi_data = data_4d[:, roi]
            if alpha_weights_4d is not None:
                try:
                    _, mad = _weighted_mad(roi_data, alpha_weights_4d[:, roi])
                except ValueError:
                    mad = 0.0
            else:
                try:
                    _, mad = _masked_mad(roi_data)
                except ValueError:
                    mad = 0.0
            if mad > 0 and np.isfinite(mad):
                vm = mad_scale * mad
            else:
                vm = float(np.nanmax(np.abs(roi_data)))
            row_limits.append((-vm, vm) if vm > 0 else (-1.0, 1.0))
        return row_limits

    elif scale == "global_centered":
        per_roi_mads = []
        for roi in range(n_rois):
            roi_data = data_4d[:, roi]
            if alpha_weights_4d is not None:
                try:
                    _, mad = _weighted_mad(roi_data, alpha_weights_4d[:, roi])
                except ValueError:
                    mad = 0.0
            else:
                try:
                    _, mad = _masked_mad(roi_data)
                except ValueError:
                    mad = 0.0
            per_roi_mads.append(mad if np.isfinite(mad) else 0.0)
        per_roi_mads = np.array(per_roi_mads)
        positive = per_roi_mads[per_roi_mads > 0]
        mad = float(np.median(positive)) if positive.size > 0 else 1.0
        vm = mad_scale * mad
        return [(-vm, vm)] * n_rois

    else:  # "global"
        if alpha_weights_4d is not None:
            try:
                _, mad = _weighted_mad(data_4d, alpha_weights_4d)
            except ValueError:
                mad = 0.0
        else:
            try:
                _, mad = _masked_mad(data_4d)
            except ValueError:
                mad = 0.0
        if mad > 0 and np.isfinite(mad):
            vm = mad_scale * mad
        else:
            vm = float(np.nanmax(np.abs(data_4d)))
        return [(-vm, vm)] * n_rois


def _build_colour_mosaic(
    data_4d,
    row_limits,
    cmap="jet",
    alpha_weights_4d=None,
    bad_color="white",
    gap_px=2,
    cbar_width_px=2,
    bg_color=(1, 1, 1, 1),
):
    """Build an RGBA mosaic image: ROIs as rows, colour channels as columns.

    Parameters
    ----------
    data_4d : ndarray, shape (n_colours, n_rois, h, w)
    row_limits : list of (vmin, vmax), length n_rois
    cmap : str or Colormap
    alpha_weights_4d : ndarray or None, same shape as data_4d
    bad_color : str
    gap_px : int
    cbar_width_px : int
    bg_color : tuple of 4 floats (RGBA)

    Returns
    -------
    mosaic : ndarray, shape (total_h, total_w, 4)
    layout_info : dict with keys n_colours, n_rois, ny, nx, gap_px,
                  cbar_width_px, row_width
    """
    n_colours, n_rois, ny, nx = data_4d.shape
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(bad_color)
    bg_rgba = np.array(bg_color, dtype=np.float32)

    row_width = n_colours * nx + (n_colours - 1) * gap_px + gap_px + cbar_width_px
    total_h = n_rois * ny + (n_rois - 1) * gap_px
    mosaic = np.broadcast_to(bg_rgba, (total_h, row_width, 4)).copy()

    for roi in range(n_rois):
        vmin, vmax = row_limits[roi]
        norm = Normalize(vmin=vmin, vmax=vmax)
        y0 = roi * (ny + gap_px)
        for c in range(n_colours):
            x0 = c * (nx + gap_px)
            rgba = colormap(norm(data_4d[c, roi]))  # (ny, nx, 4)
            if alpha_weights_4d is not None:
                rgba[..., 3] = alpha_weights_4d[c, roi]
            mosaic[y0 : y0 + ny, x0 : x0 + nx] = rgba
        # Per-row colourbar strip: vertical gradient from vmax (top) to vmin (bottom)
        cbar_vals = np.linspace(1, 0, ny)[:, None] * np.ones((1, cbar_width_px))
        cbar_x0 = n_colours * nx + (n_colours - 1) * gap_px + gap_px
        mosaic[y0 : y0 + ny, cbar_x0 : cbar_x0 + cbar_width_px] = colormap(cbar_vals)

    layout_info = dict(
        n_colours=n_colours,
        n_rois=n_rois,
        ny=ny,
        nx=nx,
        gap_px=gap_px,
        cbar_width_px=cbar_width_px,
        row_width=row_width,
    )
    return mosaic, layout_info


def _build_multi_cmap_mosaic(
    data_4d,
    row_limits,
    cmaps,
    extra_rgba_columns=None,
    crosshairs=False,
    gap_px=1,
    bg_color=(1, 1, 1, 1),
):
    """Build an RGBA mosaic with per-column colormaps and optional extra RGBA columns.

    Parameters
    ----------
    data_4d : ndarray, shape (n_colours, n_rois, h, w)
        Colormapped spatial data.
    row_limits : list of (vmin, vmax), length n_rois
        Symmetric colour limits per ROI row.
    cmaps : list of colormap
        One colormap per colour column (length n_colours).
    extra_rgba_columns : list of ndarray or None
        Each array has shape (n_rois, h, w, 3) or (n_rois, h, w, 4).
        These are placed directly into the mosaic without colormap application.
    crosshairs : bool
        Draw crosshair lines on spatial tiles.
    gap_px : int
        Pixel gap between tiles.
    bg_color : tuple of 4 floats
        RGBA background colour.

    Returns
    -------
    mosaic : ndarray, shape (total_h, total_w, 4)
    layout_info : dict
    """
    n_colours, n_rois, ny, nx = data_4d.shape
    colormaps = [plt.get_cmap(c).copy() if isinstance(c, str) else c for c in cmaps]
    for cm in colormaps:
        cm.set_bad("black")
    bg_rgba = np.array(bg_color, dtype=np.float32)

    n_extra = len(extra_rgba_columns) if extra_rgba_columns is not None else 0
    total_cols = n_colours + n_extra
    row_width = total_cols * nx + (total_cols - 1) * gap_px
    total_h = n_rois * ny + (n_rois - 1) * gap_px
    mosaic = np.broadcast_to(bg_rgba, (total_h, row_width, 4)).copy()

    crosshair_color = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    for roi in range(n_rois):
        vmin, vmax = row_limits[roi]
        norm = Normalize(vmin=vmin, vmax=vmax)
        y0 = roi * (ny + gap_px)
        # Colormapped columns
        for c in range(n_colours):
            x0 = c * (nx + gap_px)
            rgba = colormaps[c](norm(data_4d[c, roi]))  # (ny, nx, 4)
            mosaic[y0 : y0 + ny, x0 : x0 + nx] = rgba
            if crosshairs:
                cy = y0 + ny // 2
                cx = x0 + nx // 2
                mosaic[cy, x0 : x0 + nx] = crosshair_color
                mosaic[y0 : y0 + ny, cx] = crosshair_color
        # Extra RGBA columns (RGB/RGU composites)
        if extra_rgba_columns is not None:
            for e, extra in enumerate(extra_rgba_columns):
                x0 = (n_colours + e) * (nx + gap_px)
                tile = extra[roi]  # (h, w, 3) or (h, w, 4)
                if tile.ndim == 3 and tile.shape[2] == 3:
                    # Pad RGB to RGBA
                    rgba_tile = np.ones((*tile.shape[:2], 4), dtype=np.float32)
                    rgba_tile[..., :3] = tile
                else:
                    rgba_tile = tile
                mosaic[y0 : y0 + ny, x0 : x0 + nx] = rgba_tile

    layout_info = dict(
        n_colours=n_colours,
        n_extra=n_extra,
        total_cols=total_cols,
        n_rois=n_rois,
        ny=ny,
        nx=nx,
        gap_px=gap_px,
        row_width=row_width,
    )
    return mosaic, layout_info


def _annotate_colour_mosaic(
    ax,
    layout_info,
    roi_indices,
    row_limits,
    colour_labels=None,
    show_labels=True,
    show_cbar_labels=True,
    cbar_label_format="time",
):
    """Add column titles, ROI labels, and colourbar annotations to a mosaic plot.

    Parameters
    ----------
    ax : matplotlib Axes
    layout_info : dict from _build_colour_mosaic
    roi_indices : list of int
    row_limits : list of (vmin, vmax)
    colour_labels : list of str or None
    show_labels : bool
    show_cbar_labels : bool
    cbar_label_format : str
        "time" formats as ms/s, "generic" uses plain numbers.
    """
    info = layout_info
    nc, ny, nx, gap = info["n_colours"], info["ny"], info["nx"], info["gap_px"]
    cbar_w = info["cbar_width_px"]

    if colour_labels is None:
        colour_labels = [f"Ch{i}" for i in range(nc)]

    # Column titles
    for c in range(nc):
        x_center = c * (nx + gap) + nx / 2
        ax.text(
            x_center,
            -2,
            colour_labels[c],
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if not show_labels and not show_cbar_labels:
        return

    cbar_x_right = nc * nx + (nc - 1) * gap + gap + cbar_w
    for i, roi_idx in enumerate(roi_indices):
        y_center = i * (ny + gap) + ny / 2
        y0 = i * (ny + gap)
        vmin, vmax = row_limits[i]

        if show_labels:
            ax.text(-2, y_center, str(roi_idx), ha="right", va="center", fontsize=6)

        if show_cbar_labels:
            if cbar_label_format == "time":
                max_abs = max(abs(vmin), abs(vmax))
                if max_abs < 1 and max_abs > 0:
                    fmt_top = f"{vmax * 1000:.0f}ms"
                    fmt_bot = f"{vmin * 1000:.0f}ms"
                else:
                    fmt_top = f"{vmax:.2f}s"
                    fmt_bot = f"{vmin:.2f}s"
            else:
                fmt_top = f"{vmax:.2g}"
                fmt_bot = f"{vmin:.2g}"
            ax.text(cbar_x_right + 1, y0 + 1, fmt_top, ha="left", va="top", fontsize=4)
            ax.text(
                cbar_x_right + 1,
                y0 + ny - 1,
                fmt_bot,
                ha="left",
                va="bottom",
                fontsize=4,
            )


def _compute_alpha_weights_4d(self, roi_indices):
    """Compute normalised 4D alpha weights for mosaic rendering.

    Parameters
    ----------
    self : STRF object
    roi_indices : list of int

    Returns
    -------
    alpha_4d : ndarray (n_colours, n_selected_rois, h, w) or None
    """
    raw_weights = self.get_amplitude_weights()
    raw_weights = pygor.utilities.multicolour_reshape(raw_weights, self.n_colours)
    raw_weights = raw_weights[:, roi_indices]
    finite = raw_weights[np.isfinite(raw_weights)]
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, (50, 95))
    if hi <= lo:
        return None
    alpha = np.clip(raw_weights, lo, hi)
    alpha = (alpha - lo) / (hi - lo)
    alpha = np.nan_to_num(alpha, nan=0.0)
    return alpha


def _mosaic_figure(
    mosaic,
    layout_info,
    roi_indices,
    row_limits,
    colour_labels,
    show_labels,
    show_cbar,
    px_scale,
    cbar_label_format="time",
):
    """Create a figure from a colour mosaic and annotate it.

    Returns
    -------
    fig, ax
    """
    fig_w = mosaic.shape[1] * px_scale
    fig_h = mosaic.shape[0] * px_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(mosaic, aspect="equal", interpolation="nearest", origin="lower")
    _annotate_colour_mosaic(
        ax,
        layout_info,
        roi_indices,
        row_limits,
        colour_labels=colour_labels,
        show_labels=show_labels,
        show_cbar_labels=show_cbar,
        cbar_label_format=cbar_label_format,
    )
    ax.set_xlim(-15, layout_info["row_width"] + 20)
    ax.set_ylim(mosaic.shape[0], -5)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


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
    gap_px=2,
    cbar_width_px=2,
    colour_labels=None,
    px_scale=0.05,
):
    """Plot time-collapsed spatial STRFs.

    For multicolour data with ``channel=None``, produces a fast mosaic
    layout with colour channels as columns and ROIs as rows.

    Parameters
    ----------
    cval : float or None
        Symmetric colour limit. None auto-scales.
    channel : int or None
        Color channel to plot (1-indexed). None uses all channels.
        For multicolour data, None triggers the mosaic layout.
    cmap : str
        Colormap name.
    origin : str
        Image origin ('upper' or 'lower').
    roi : int, list of int, or None
        ROI indices to plot. None plots all ROIs.
    max_x : int
        Maximum ROIs per row in grid (single-channel mode only).
    show_cbar : bool
        Whether to show colorbar.
    cbar_kwargs : dict or None
        Keyword arguments for colorbar (single-channel mode only).
    show_labels : bool
        Whether to show ROI number labels.
    gap_px : int
        Pixel gap between panels in mosaic mode.
    cbar_width_px : int
        Width of per-row colourbar strips in mosaic mode.
    colour_labels : list of str or None
        Labels for colour channel columns in mosaic mode.
    px_scale : float
        Figure size scaling factor in mosaic mode.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- Mosaic mode: multicolour data with no specific channel ---
    use_mosaic = hasattr(self, "multicolour") and self.multicolour and channel is None
    if use_mosaic:
        data_4d = self.collapse_times_by_channel(
            force_recompute=True,
        )  # (n_colours, n_rois, h, w)
        n_rois_per_colour = data_4d.shape[1]
        roi_indices = _normalize_roi_indices(roi, n_rois_per_colour)
        if not roi_indices:
            raise ValueError("roi selection is empty")
        data_4d = data_4d[:, roi_indices]

        # Symmetric limits per row
        n_rois = len(roi_indices)
        if cval is not None:
            row_limits = [(-cval, cval)] * n_rois
        else:
            row_limits = []
            for r in range(n_rois):
                roi_data = data_4d[:, r]
                cv = _symmetric_cval(roi_data, None)
                row_limits.append((-cv, cv))

        mosaic, layout_info = _build_colour_mosaic(
            data_4d,
            row_limits,
            cmap,
            None,
            "white",
            gap_px,
            cbar_width_px if show_cbar else 0,
        )
        return _mosaic_figure(
            mosaic,
            layout_info,
            roi_indices,
            row_limits,
            colour_labels,
            show_labels,
            show_cbar,
            px_scale,
            cbar_label_format="generic",
        )

    # --- Single-channel / single-colour grid mode (existing path) ---
    _validate_channel(channel, self.n_colours)
    array = self.collapse_times(force_recompute=True)
    if channel is not None:
        array = pygor.utilities.multicolour_reshape(array, self.n_colours)[channel]
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
    im = ax.imshow(
        image, cmap=cmap, interpolation="none", clim=(-cval, cval), origin=origin
    )
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
    gap_px=2,
    cbar_width_px=2,
    colour_labels=None,
    px_scale=0.05,
):
    """Plot raw peak timing values for each ROI.

    Shows when each pixel's response peaked (absolute time, no centering).
    For multicolour data with ``channel=None``, produces a fast mosaic
    layout with colour channels as columns and ROIs as rows.

    Parameters
    ----------
    roi : int, list of int, or None
        ROI indices to plot. None plots all ROIs.
    channel : int or None
        Color channel to plot (1-indexed). None uses all channels.
        For multicolour data, None triggers the mosaic layout.
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
        Maximum ROIs per row in grid (single-channel mode only).
    show_cbar : bool
        Whether to show colorbar.
    cbar_kwargs : dict or None
        Keyword arguments for colorbar (single-channel mode only).
    show_labels : bool
        Whether to show ROI number labels.
    bad_color : str
        Color for masked pixels.
    alpha_mode : str
        Alpha transparency mode. "strf_abs" uses collapsed STRF magnitude.
    gap_px : int
        Pixel gap between panels in mosaic mode.
    cbar_width_px : int
        Width of per-row colourbar strips in mosaic mode.
    colour_labels : list of str or None
        Labels for colour channel columns in mosaic mode.
    px_scale : float
        Figure size scaling factor in mosaic mode.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- Mosaic mode: multicolour data with no specific channel ---
    use_mosaic = hasattr(self, "multicolour") and self.multicolour and channel is None
    if use_mosaic:
        data_4d = self.get_strf_peak_times_by_channel()  # (n_colours, n_rois, h, w)
        n_rois_per_colour = data_4d.shape[1]
        roi_indices = _normalize_roi_indices(roi, n_rois_per_colour)
        if not roi_indices:
            raise ValueError("roi selection is empty")
        data_4d = data_4d[:, roi_indices]

        alpha_4d = None
        if alpha_mode == "strf_abs":
            alpha_4d = _compute_alpha_weights_4d(self, roi_indices)

        # For peak times, use global min/max per row (not MAD-based)
        n_rois = len(roi_indices)
        if clim is not None:
            row_limits = [clim] * n_rois
        else:
            row_limits = []
            for r in range(n_rois):
                roi_data = data_4d[:, r]
                finite = roi_data[np.isfinite(roi_data)]
                if hasattr(roi_data, "compressed"):
                    finite = roi_data.compressed()
                if finite.size > 0:
                    row_limits.append((float(np.min(finite)), float(np.max(finite))))
                else:
                    row_limits.append((0.0, 1.0))

        mosaic, layout_info = _build_colour_mosaic(
            data_4d,
            row_limits,
            cmap,
            alpha_4d,
            bad_color,
            gap_px,
            cbar_width_px if show_cbar else 0,
        )
        return _mosaic_figure(
            mosaic,
            layout_info,
            roi_indices,
            row_limits,
            colour_labels,
            show_labels,
            show_cbar,
            px_scale,
        )

    # --- Single-channel / single-colour grid mode (existing path) ---
    _validate_channel(channel, self.n_colours)
    array = self.get_strf_peak_times()
    if channel is not None:
        array = pygor.utilities.multicolour_reshape(array, self.n_colours)[channel]
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
            alpha_array_raw = pygor.utilities.multicolour_reshape(
                alpha_array_raw, self.n_colours
            )[channel]
        alpha_array_raw = alpha_array_raw[roi_indices]
        if alpha_array_raw.shape == array.shape:
            finite_alpha = alpha_array_raw[np.isfinite(alpha_array_raw)]
            if finite_alpha.size > 0:
                alpha_lo, alpha_hi = np.percentile(finite_alpha, (50, 95))
                if alpha_hi > alpha_lo:
                    alpha_array = np.clip(alpha_array_raw, alpha_lo, alpha_hi)
                    alpha_array = (alpha_array - alpha_lo) / (alpha_hi - alpha_lo)
                    alpha_array = np.clip(alpha_array, 0.0, 1.0)
                    alpha_array = np.nan_to_num(
                        alpha_array, nan=0.0, posinf=1.0, neginf=0.0
                    )
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
    gap_px=2,
    cbar_width_px=2,
    colour_labels=None,
    px_scale=0.05,
):
    """Plot relative timing differences for each ROI.

    Shows timing differences relative to each ROI's weighted median,
    revealing temporal gradients within receptive fields.

    For multicolour data with ``channel=None``, produces a fast mosaic
    layout with colour channels as columns and ROIs as rows.

    Parameters
    ----------
    roi : int, list of int, or None
        ROI indices to plot. None plots all ROIs.
    channel : int or None
        Color channel to plot (1-indexed). None uses all channels.
        For multicolour data, None triggers the mosaic layout.
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
        Colorbar range is ±(mad_scale × MAD).
    cmap : str
        Colormap name.
    clim : tuple of (vmin, vmax) or None
        Override auto scaling with explicit limits.
    origin : str
        Image origin ('upper' or 'lower').
    max_x : int
        Maximum ROIs per row in grid (single-channel mode only).
    show_cbar : bool
        Whether to show colorbar.
    cbar_kwargs : dict or None
        Keyword arguments for colorbar (single-channel mode only).
    show_labels : bool
        Whether to show ROI number labels.
    bad_color : str
        Color for masked pixels.
    alpha_mode : str
        Alpha transparency mode. "strf_abs" uses collapsed STRF magnitude.
    gap_px : int
        Pixel gap between panels in mosaic mode.
    cbar_width_px : int
        Width of per-row colourbar strips in mosaic mode.
    colour_labels : list of str or None
        Labels for colour channel columns in mosaic mode.
    px_scale : float
        Figure size scaling factor in mosaic mode.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if scale not in {"global", "global_centered", "per_roi"}:
        raise ValueError("scale must be 'global', 'global_centered', or 'per_roi'")

    # --- Mosaic mode: multicolour data with no specific channel ---
    use_mosaic = hasattr(self, "multicolour") and self.multicolour and channel is None
    if use_mosaic:
        data_4d = self.get_strf_delta_times_by_channel(
            use_segmentation=use_segmentation,
            seg_kwargs=seg_kwargs,
        )  # (n_colours, n_rois, h, w)
        # Resolve ROI indices from the per-colour ROI count
        n_rois_per_colour = data_4d.shape[1]
        roi_indices = _normalize_roi_indices(roi, n_rois_per_colour)
        if not roi_indices:
            raise ValueError("roi selection is empty")
        data_4d = data_4d[:, roi_indices]

        alpha_4d = None
        if alpha_mode == "strf_abs":
            alpha_4d = _compute_alpha_weights_4d(self, roi_indices)

        row_limits = _compute_mosaic_row_limits(
            data_4d,
            scale,
            mad_scale,
            alpha_4d,
            clim,
        )
        mosaic, layout_info = _build_colour_mosaic(
            data_4d,
            row_limits,
            cmap,
            alpha_4d,
            bad_color,
            gap_px,
            cbar_width_px if show_cbar else 0,
        )
        return _mosaic_figure(
            mosaic,
            layout_info,
            roi_indices,
            row_limits,
            colour_labels,
            show_labels,
            show_cbar,
            px_scale,
        )

    # --- Single-channel / single-colour grid mode (existing path) ---
    _validate_channel(channel, self.n_colours)

    # Get centered delta times from the data method
    array = self.get_strf_delta_times(
        roi=roi,
        channel=channel,
        use_segmentation=use_segmentation,
        seg_kwargs=seg_kwargs,
    )
    # Ensure 3D even for single ROI
    if array.ndim == 2:
        array = array[np.newaxis]

    # Resolve roi_indices for alpha weights (must match array shape)
    all_weights = self.get_amplitude_weights()
    if channel is not None:
        all_weights = pygor.utilities.multicolour_reshape(all_weights, self.n_colours)[
            channel
        ]
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
                alpha_array = np.nan_to_num(
                    alpha_array, nan=0.0, posinf=1.0, neginf=0.0
                )
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
        mad = (
            np.median(per_roi_mads[per_roi_mads > 0])
            if np.any(per_roi_mads > 0)
            else 0.0
        )
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
            cbar.formatter = mticker.FuncFormatter(lambda x, pos: f"{x * time_scale:g}")
            cbar.update_ticks()

    if show_labels:
        h, w = array.shape[1], array.shape[2]
        _annotate_grid(ax, roi_indices, num_rows, max_x, h, w)

    return fig, ax
