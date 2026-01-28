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
    print(max_x, num_slices)
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


def plot_spacetime_strfs(
    self,
    cval=None,
    channel=None,
    cmap="jet",
    origin="upper",
    roi=None,
    max_x=10,
    use_segmentation=False,
    seg_kwargs=None,
    bad_color="black",
    show_cbar=True,
    cbar_kwargs=None,
    scale_k=2,
    show_labels=True,
    alpha_mode="strf_abs",
    scale_weighted=True,
    relative=True,
    relative_center="median",
    relative_scale="per_roi",
):
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
        full_masks = np.sum(seg_masks, axis=(1, 2)) == seg_masks.shape[1] * seg_masks.shape[2]
        if np.any(full_masks):
            array[full_masks] = np.ma.masked

    if relative:
        # Fixed to median; keep arg for backward compatibility.
        array, _ = _center_per_slice(array, "median")
        if relative_scale not in {"global", "per_roi"}:
            raise ValueError("relative_scale must be 'global' or 'per_roi'")

    alpha = None
    alpha_array = None
    if alpha_mode == "strf_abs":
        alpha_array = np.abs(self.collapse_times())
        if channel is not None:
            alpha_array = pygor.utilities.multicolour_reshape(alpha_array, channel)[
                channel - 1
            ]
        alpha_array = alpha_array[roi_indices]
        if alpha_array.shape != array.shape:
            raise ValueError("alpha map shape does not match data array")
        finite_alpha = alpha_array[np.isfinite(alpha_array)]
        if finite_alpha.size > 0:
            alpha_lo, alpha_hi = np.percentile(finite_alpha, (50, 95))
            if alpha_hi > alpha_lo:
                alpha_array = np.clip(alpha_array, alpha_lo, alpha_hi)
                alpha_array = (alpha_array - alpha_lo) / (alpha_hi - alpha_lo)
                alpha_array = np.clip(alpha_array, 0.0, 1.0)
                alpha_array = alpha_array
                alpha_array = np.nan_to_num(
                    alpha_array, nan=0.0, posinf=1.0, neginf=0.0
                )
                alpha_array = np.clip(alpha_array, 0.0, 1.0)
                alpha, _, _ = _build_grid_image(alpha_array, max_x)

    # Grid layout
    max_x = _normalize_max_x(max_x)
    image, num_rows, num_slices = _build_grid_image(array, max_x)

    if relative and relative_scale == "per_roi":
        per_roi_mad = []
        for roi_idx in range(array.shape[0]):
            roi_slice = array[roi_idx]
            if np.ma.isMaskedArray(roi_slice):
                data = roi_slice.compressed()
            else:
                data = roi_slice.ravel()
            if data.size == 0:
                per_roi_mad.append(0.0)
                continue
            if scale_weighted and alpha_array is not None:
                _, mad = _weighted_mad(roi_slice, alpha_array[roi_idx])
            else:
                _, mad = _masked_mad(roi_slice)
            per_roi_mad.append(abs(mad))
        scale = np.median(per_roi_mad) if per_roi_mad else 0.0
        if not np.isfinite(scale) or scale == 0.0:
            vmin, vmax = _masked_minmax(image)
            if relative:
                max_abs = max(abs(vmin), abs(vmax))
                vmin, vmax = -max_abs, max_abs
        else:
            max_abs = scale_k * scale
            vmin, vmax = -max_abs, max_abs
    else:
        if scale_weighted and alpha_array is not None:
            _, mad = _weighted_mad(array, alpha_array)
        else:
            _, mad = _masked_mad(image)
        if mad == 0 or not np.isfinite(mad):
            vmin, vmax = _masked_minmax(image)
        else:
            max_abs = scale_k * mad
            vmin, vmax = -max_abs, max_abs
    if cval is not None:
        vmax = cval
        if relative:
            vmin = -cval

    # Display
    fig, ax = plt.subplots(figsize=(max_x, max(2, num_rows)))
    if np.ma.isMaskedArray(image):
        cmap = plt.cm.get_cmap(cmap).copy()
        cmap.set_bad(color=bad_color)
    if alpha is not None:
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    if alpha is not None and np.ma.isMaskedArray(image):
        mask = np.ma.getmaskarray(image)
        alpha = np.where(mask, 0.0, alpha)
    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
    im = ax.imshow(
        image,
        cmap=cmap,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        alpha=alpha,
    )
    ax.axis("off")
    print(max_x, num_slices)
    if show_cbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
        if relative:
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs < 1:
                scale = 1000.0
                unit = "ms"
            else:
                scale = 1.0
                unit = "s"
            cbar.set_label(f"\u0394t ({unit})")
            cbar.formatter = mticker.FuncFormatter(
                lambda x, pos: f"{x * scale:g}"
            )
            cbar.update_ticks()
        else:
            cbar.set_label("t (s)")

    # Overlay slice numbers
    if show_labels:
        h, w = array.shape[1], array.shape[2]
        _annotate_grid(ax, roi_indices, num_rows, max_x, h, w)
    return fig, ax
