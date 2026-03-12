import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.plotting
import pygor.strf.contouring
import pygor.strf.pixconverter
import pygor.strf.spatial
import pygor.strf.temporal
import pygor.utilities
from pygor.plotting.custom import blue_map, fish_palette, green_map, red_map, violet_map
from pygor.strf.plotting.simple import _build_multi_cmap_mosaic

pygor.plotting.fish_palette.append("dimgrey")
pygor.plotting.fish_palette.append("grey")


def chroma_overview(
    data_strf_object,
    specify_rois=None,
    ipl_sort=False,
    y_crop=(0, 0),
    x_crop=(0, 0),
    column_titles=None,
    colour_maps=[red_map, green_map, blue_map, violet_map, "Greys_r", "Greys_r"],
    centre_dots=True,
    contours=False,
    crosshairs=False,
    ax=None,
    high_contrast=True,
    remove_border=False,
    labels=None,
    clim="roi",
    with_times=False,
    with_rgb=True,
    time_setting="1d",
    time_dur_ms=None,
    scalebar=True,
    figsize=None,
    colour_idx=None,
):
    # Optimised by Claude Opus 4.6, incredibly fast now.
    # Create iterators depneding on desired output
    if isinstance(specify_rois, np.int64) or isinstance(specify_rois, np.int32):
        specify_rois = int(specify_rois)  # handle silly numpy int types issue
    if isinstance(
        specify_rois, int
    ):  # user specifies number of rois from "start", although negative is also allowed
        rois_specified = [specify_rois]
        # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable):  # user specifies specific rois
        rois_specified = specify_rois  # lol
        if ipl_sort == True:
            rois_specified = data_strf_object.ipl_depths[specify_rois].argsort()
    elif specify_rois == None:  # user wants all rois
        rois_specified = range(data_strf_object.num_rois)
        if ipl_sort == True:
            rois_specified = data_strf_object.ipl_depths.argsort()
    if specify_rois is None:
        rois_specified = range(data_strf_object.num_rois)
        specify_rois = range(data_strf_object.num_rois)
    if isinstance(colour_maps, Iterable) is False:
        colour_maps = [colour_maps] * len(column_titles)
    # if isinstance(data_strf_object, pygor.classes.strf_data.STRF) is False:
    if str(data_strf_object) != "<class 'pygor.classes.strf_data.STRF'>":
        print(data_strf_object)
        raise AttributeError("Input object is not a STRF object.")
        # warnings.warn(
        #     "Input object is not a STRF object. Attempting to treat as nxm Numpy array. Use-case not intended, expect errors."
        # )
        # strfs_chroma = data_strf_object
        # numcolour = strfs_chroma.shape[0]
        # remove_border = False
    # else:
    numcolour = data_strf_object.numcolour

    # Handle colour index filtering
    if colour_idx is None:
        colour_idx = list(range(numcolour))
    elif isinstance(colour_idx, int):
        colour_idx = [colour_idx]
    else:
        colour_idx = list(colour_idx)

    # Filter colour maps to match selected indices
    if isinstance(colour_maps, list) and len(colour_maps) >= numcolour:
        colour_maps = [colour_maps[i] for i in colour_idx]
    elif not isinstance(colour_maps, list):
        colour_maps = [colour_maps] * len(colour_idx)

    # Filter column titles if provided
    if column_titles is not None:
        if len(column_titles) >= numcolour:
            column_titles = [column_titles[i] for i in colour_idx]
        else:
            column_titles = None  # Fallback if titles don't match

    num_cols = len(colour_idx)

    # Handle time duration - use object's duration if not specified
    if time_dur_ms is None:
        time_dur_ms = data_strf_object.strf_dur_ms

    # Handle colour limits
    if isinstance(clim, str) and clim == "all":
        # Use the abs max of the entre input
        abs_max = np.max(np.abs(data_strf_object.strfs_chroma))
        clim_vals = (-abs_max, abs_max)
    else:
        # Otherwise use user input (two floats or ints)
        clim_vals = clim
    if ax is not None:
        # Legacy path: external axes provided, use old per-axis approach
        return _chroma_overview_legacy(
            data_strf_object,
            rois_specified,
            colour_idx,
            colour_maps,
            column_titles,
            numcolour,
            clim,
            clim_vals,
            remove_border,
            y_crop,
            x_crop,
            contours,
            crosshairs,
            high_contrast,
            labels,
            with_times,
            with_rgb,
            time_setting,
            time_dur_ms,
            scalebar,
            figsize,
            ax,
        )

    # ---- Optimised mosaic path ----
    num_rois = len(rois_specified) if not isinstance(rois_specified, int) else 1
    rois_list = (
        list(rois_specified)
        if not isinstance(rois_specified, int)
        else [rois_specified]
    )

    # Phase A: Vectorised data preparation
    all_collapsed_chroma = data_strf_object.collapse_times_chroma()
    # data_4d shape: (len(colour_idx), num_rois, h, w)
    data_4d = all_collapsed_chroma[np.ix_(colour_idx, rois_list)]
    if remove_border is True:
        data_4d = np.copy(pygor.utilities.auto_remove_border(data_4d))
    ycropper = y_crop if y_crop != (0, 0) else (None, None)
    xcropper = x_crop if x_crop != (0, 0) else (None, None)
    data_4d = data_4d[:, :, ycropper[0] : ycropper[1], xcropper[0] : xcropper[1]]

    # Compute per-ROI symmetric colour limits
    if isinstance(clim, str) and clim == "all":
        abs_max = np.max(np.abs(data_4d))
        row_limits = [(-abs_max, abs_max)] * num_rois
    elif clim == "roi" or clim is None:
        row_limits = []
        for n in range(num_rois):
            cv = np.max(np.abs(data_4d[:, n]))
            row_limits.append((-cv, cv))
    else:
        row_limits = [clim_vals] * num_rois

    # Phase B: Build RGB/RGU composites
    extra_rgba_columns = None
    if with_rgb and data_4d.shape[0] >= 4:
        rgb_composites = []
        rgu_composites = []
        for n in range(num_rois):
            rgb_data = np.abs(data_4d[[0, 1, 2], n])  # (3, h, w)
            rgu_data = np.abs(data_4d[[0, 1, 3], n])
            rgb_max = rgb_data.max()
            rgu_max = rgu_data.max()
            if rgb_max > 0:
                rgb_data = rgb_data / rgb_max
            if rgu_max > 0:
                rgu_data = rgu_data / rgu_max
            rgb_composites.append(np.transpose(rgb_data, (1, 2, 0)))  # (h, w, 3)
            rgu_composites.append(np.transpose(rgu_data, (1, 2, 0)))
        extra_rgba_columns = [
            np.array(rgb_composites, dtype=np.float32),
            np.array(rgu_composites, dtype=np.float32),
        ]

    # Phase C: Build mosaic
    mosaic, layout_info = _build_multi_cmap_mosaic(
        data_4d,
        row_limits,
        colour_maps,
        extra_rgba_columns=extra_rgba_columns,
        crosshairs=crosshairs,
        gap_px=1,
    )
    ny = layout_info["ny"]
    nx = layout_info["nx"]
    gap_px = layout_info["gap_px"]

    # Phase D: Create figure using manual axes positioning to avoid
    # tight_layout / aspect="equal" conflicts that cause whitespace.
    px_scale = 0.03
    mosaic_fig_w = mosaic.shape[1] * px_scale
    mosaic_fig_h = mosaic.shape[0] * px_scale
    if with_times:
        time_frac = 0.1  # fraction of total width for temporal panel
        gap_frac = 0.02  # gap between imshow and plot
        margin = 0.04  # margin for scalebar text
        line_pix = 1  # thickness of lines (optional)
        trace_scalar = 0.9
        mosaic_frac = 1.0 - time_frac - gap_frac
        total_w = mosaic_fig_w / (mosaic_frac - margin)
        fig_h = mosaic_fig_h  # small extra for scalebar text below
        if figsize is None:
            figsize = (total_w, fig_h)
        fig = plt.figure(figsize=figsize)
        ax_mosaic = fig.add_axes([margin, margin, mosaic_frac - margin, 1 - 2 * margin])
        ax_time = fig.add_axes(
            [mosaic_frac + gap_frac, margin, time_frac - margin, 1 - 2 * margin],
            sharey=ax_mosaic,
        )
    else:
        if figsize is None:
            figsize = (max(mosaic_fig_w, 2), mosaic_fig_h + 0.5)
        fig, ax_mosaic = plt.subplots(1, 1, figsize=figsize)
        ax_time = None

    ax_mosaic.imshow(mosaic, aspect="auto", interpolation="nearest", origin="lower")
    ax_mosaic.autoscale(
        enable=False
    )  # lock limits — prevents scalebar plot() from expanding via sharey
    ax_mosaic.axis("off")

    # Phase E: Annotations on mosaic
    if column_titles is not None:
        for plot_idx in range(len(colour_idx)):
            x_center = plot_idx * (nx + gap_px) + nx / 2
            ax_mosaic.text(
                x_center,
                -2,
                column_titles[plot_idx],
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
    if labels is not None:
        if labels == "auto":
            labels = [f"ROI {roi}" for roi in rois_list]
        for n, label in enumerate(labels):
            y_center = n * (ny + gap_px) + ny / 2
            ax_mosaic.text(
                -2, y_center, label, ha="right", va="center", fontsize=6, color="white"
            )

    # Contours on mosaic
    if contours:
        for n, roi in enumerate(rois_list):
            y_offset = n * (ny + gap_px)
            for plot_idx, ci in enumerate(colour_idx):
                x_offset = plot_idx * (nx + gap_px)
                _contours_plotter_mosaic(
                    data_strf_object,
                    roi=roi,
                    index=ci,
                    ax=ax_mosaic,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    high_contrast=high_contrast,
                )

    # Phase F: Temporal traces — y-offsets use mosaic pixel coordinates so
    # each trace aligns with its corresponding mosaic row (origin="lower").
    mosaic_h = mosaic.shape[0]
    if with_times and ax_time is not None:
        all_max_val = 0
        # First pass: find global max amplitude
        for n, roi in enumerate(rois_list):
            start_index = roi * numcolour
            end_index = start_index + numcolour
            fetch_indices = range(start_index, end_index)
            times = np.squeeze(
                pygor.utilities.multicolour_reshape(
                    data_strf_object.get_timecourses(
                        fetch_indices, method="segmentation", mask_empty=True
                    ),
                    numcolour,
                )
            )
            times = times[colour_idx]
            curr_max = np.max(np.abs(times))
            if curr_max > all_max_val:
                all_max_val = curr_max
        # Scale traces to fit within each mosaic row height
        trace_scale = (ny * trace_scalar) / all_max_val if all_max_val > 0 else 1.0
        # Second pass: plot traces at mosaic row centers
        for n, roi in enumerate(rois_list):
            start_index = roi * numcolour
            end_index = start_index + numcolour
            fetch_indices = range(start_index, end_index)
            times = np.squeeze(
                pygor.utilities.multicolour_reshape(
                    data_strf_object.get_timecourses(
                        fetch_indices, method="segmentation", mask_empty=True
                    ),
                    numcolour,
                )
            )
            times = times[colour_idx]
            deviation = np.std(times, axis=-1)
            close_to_zero = np.isclose(deviation, 0, rtol=0.1, atol=0.4)
            time_frames = times.shape[-1]
            time_axis_ms = np.linspace(0, time_dur_ms, time_frames)
            # y_offset = center of this ROI's row in mosaic pixel coords
            y_offset = n * (ny + gap_px) + ny / 2
            for enum, plotme in enumerate(times):
                original_color_idx = colour_idx[enum]
                close_val = close_to_zero[enum]
                if hasattr(close_val, "any"):
                    is_close_to_zero = close_val.any()
                else:
                    is_close_to_zero = bool(close_val)
                alpha = 0.5 if is_close_to_zero else 1.0
                ax_time.plot(
                    time_axis_ms,
                    plotme.T * trace_scale + y_offset,
                    color=pygor.plotting.fish_palette[original_color_idx],
                    alpha=alpha,
                    lw=line_pix,
                )
        ax_time.set_xlim(0, time_dur_ms)
        # y-limits are shared with ax_mosaic via sharey — guaranteed 1:1 alignment
        ax_time.axis("off")
        # Scalebars drawn directly in data coords
        if scalebar:
            # SD bar — only show if there's actual amplitude to measure
            if all_max_val > 0:
                value = 5 if all_max_val > 5 else np.round(all_max_val, 2)
                scaled_value = value * trace_scale
                sb_x = time_dur_ms * 1.02
                t_ymid = (mosaic_h - 1) / 2  # vertical midpoint of mosaic
                ax_time.plot(
                    [sb_x, sb_x],
                    [t_ymid - scaled_value / 2, t_ymid + scaled_value / 2],
                    "k-",
                    lw=3,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                ax_time.text(
                    sb_x + time_dur_ms * 0.03,
                    t_ymid,
                    f"{value} SD",
                    ha="left",
                    va="center",
                    fontsize=8,
                    rotation=-90,
                )
            # 300ms bar — below the mosaic bottom edge (matching spatial scalebar)
            scalebar_target_ms = 300
            sb_y = -0.5 - ny * 0.15
            ax_time.plot(
                [0, scalebar_target_ms],
                [sb_y, sb_y],
                "k-",
                lw=3,
                clip_on=False,
                solid_capstyle="butt",
            )
            ax_time.text(
                scalebar_target_ms / 2,
                sb_y - ny * 0.08,
                f"{scalebar_target_ms} ms",
                ha="center",
                va="top",
                fontsize=8,
            )

    # Spatial scalebar — bottom-left of mosaic, drawn in data coords
    degrees = 15
    visang_to_space = pygor.strf.pixconverter.visang_to_pix(
        degrees,
        pixwidth=40,
    )
    if scalebar:
        # imshow sets xlim/ylim to image extent (-0.5 to dim-0.5)
        sb_y = -0.5 - ny * 0.15  # just below the mosaic bottom edge
        ax_mosaic.plot(
            [0, visang_to_space],
            [sb_y, sb_y],
            "k-",
            lw=3,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax_mosaic.text(
            visang_to_space / 2,
            sb_y - ny * 0.08,
            f"{degrees}\u00b0",
            ha="center",
            va="top",
            fontsize=8,
        )

    return fig, ax_mosaic


def _contours_plotter_mosaic(
    data_strf_object,
    roi,
    index=None,
    x_offset=0,
    y_offset=0,
    high_contrast=True,
    ax=None,
):
    """Draw contours on a mosaic axis with pixel-coordinate offsets."""
    if ax is None:
        return
    contours = pygor.utilities.multicolour_reshape(
        data_strf_object.fit_contours(), data_strf_object.numcolour
    )[:, roi]
    neg_contours = contours[:, 0]
    pos_contours = contours[:, 1]
    if index is None:
        index = range(len(contours))
    if isinstance(index, Iterable) is False:
        index = [index]
    for colour in index:
        for contour_n in neg_contours[colour]:
            if high_contrast:
                ax.plot(
                    contour_n[:, 1] + x_offset,
                    contour_n[:, 0] + y_offset,
                    lw=1.5,
                    ls="-",
                    c="white",
                    alpha=1,
                )
                ax.plot(
                    contour_n[:, 1] + x_offset,
                    contour_n[:, 0] + y_offset,
                    lw=1.5,
                    ls="dashed",
                    c=fish_palette[colour],
                    alpha=1,
                )
            else:
                ax.plot(
                    contour_n[:, 1] + x_offset,
                    contour_n[:, 0] + y_offset,
                    lw=1,
                    ls="-",
                    c=fish_palette[colour],
                    alpha=1,
                )
        for contour_p in pos_contours[colour]:
            if high_contrast:
                ax.plot(
                    contour_p[:, 1] + x_offset,
                    contour_p[:, 0] + y_offset,
                    lw=1.5,
                    ls="-",
                    c="white",
                    alpha=1,
                )
                ax.plot(
                    contour_p[:, 1] + x_offset,
                    contour_p[:, 0] + y_offset,
                    lw=1.5,
                    ls="dashed",
                    c=fish_palette[colour],
                    alpha=1,
                )
            else:
                ax.plot(
                    contour_p[:, 1] + x_offset,
                    contour_p[:, 0] + y_offset,
                    lw=1,
                    ls="-",
                    c=fish_palette[colour],
                    alpha=1,
                )


def _chroma_overview_legacy(
    data_strf_object,
    rois_specified,
    colour_idx,
    colour_maps,
    column_titles,
    numcolour,
    clim,
    clim_vals,
    remove_border,
    y_crop,
    x_crop,
    contours,
    crosshairs,
    high_contrast,
    labels,
    with_times,
    with_rgb,
    time_setting,
    time_dur_ms,
    scalebar,
    figsize,
    ax,
):
    """Legacy per-axis chroma_overview for when external axes are provided."""
    fig = plt.gcf()
    # Ensure ax is always 2D for consistent indexing
    if ax.ndim == 1:
        ax = ax.reshape(1, -1)
    all_collapsed_chroma = data_strf_object.collapse_times_chroma()
    for n, roi in enumerate(rois_specified):
        start_index = roi * numcolour
        end_index = start_index + numcolour
        fetch_indices = range(start_index, end_index)
        strfs_chroma = np.squeeze(all_collapsed_chroma[:, roi])
        if remove_border is True:
            border_tup = pygor.utilities.check_border(strfs_chroma)
            strfs_chroma = np.copy(pygor.utilities.auto_remove_border(strfs_chroma))
        else:
            border_tup = (0, 0, 0, 0)
        if clim == "roi" or clim is None:
            clim_vals = (-np.max(np.abs(strfs_chroma)), np.max(np.abs(strfs_chroma)))
        ycropper = y_crop if y_crop != (0, 0) else (None, None)
        xcropper = x_crop if x_crop != (0, 0) else (None, None)
        strfs_chroma = strfs_chroma[
            :, ycropper[0] : ycropper[1], xcropper[0] : xcropper[1]
        ]
        for plot_idx, color_idx in enumerate(colour_idx):
            strf = ax[n, plot_idx].imshow(
                strfs_chroma[color_idx], cmap=colour_maps[plot_idx], origin="lower"
            )
            strf.set_clim(clim_vals)
            if n == 0 and column_titles is not None:
                ax[n, plot_idx].set_title(column_titles[plot_idx])
            if contours:
                _contours_plotter(
                    data_strf_object,
                    roi=roi,
                    index=color_idx,
                    ax=ax[-n - 1, plot_idx],
                    xy_offset=(-border_tup[0], -border_tup[2]),
                    high_contrast=high_contrast,
                )
        if with_times:
            time_ax_col = len(colour_idx)
            times = np.squeeze(
                pygor.utilities.multicolour_reshape(
                    data_strf_object.get_timecourses(
                        fetch_indices, method="segmentation", mask_empty=True
                    ),
                    numcolour,
                )
            )
            times = times[colour_idx]
            deviation = np.std(times, axis=-1)
            close_to_zero = np.isclose(deviation, 0, rtol=0.1, atol=0.4)
            time_frames = times.shape[-1]
            time_axis_ms = np.linspace(0, time_dur_ms, time_frames)
            for enum, plotme in enumerate(times):
                original_color_idx = colour_idx[enum]
                close_val = close_to_zero[enum]
                if hasattr(close_val, "any"):
                    is_close_to_zero = close_val.any()
                else:
                    is_close_to_zero = bool(close_val)
                alpha = 0.5 if is_close_to_zero else 1.0
                ax[n, time_ax_col].plot(
                    time_axis_ms,
                    plotme.T,
                    color=pygor.plotting.fish_palette[original_color_idx],
                    alpha=alpha,
                )
        if with_rgb:
            rgb_start_col = len(colour_idx) + (1 if with_times else 0)
            rgb_data = strfs_chroma[[0, 1, 2]]
            rgu_data = strfs_chroma[[0, 1, 3]]
            rgb_normalized = np.abs(rgb_data)
            rgu_normalized = np.abs(rgu_data)
            rgb_global_max = np.max([rgb_normalized[i].max() for i in range(3)])
            rgu_global_max = np.max([rgu_normalized[i].max() for i in range(3)])
            if rgb_global_max > 0:
                for i in range(3):
                    rgb_normalized[i] = rgb_normalized[i] / rgb_global_max
            if rgu_global_max > 0:
                for i in range(3):
                    rgu_normalized[i] = rgu_normalized[i] / rgu_global_max
            processed_rgb = np.transpose(rgb_normalized, (1, 2, 0))
            processed_rgu = np.transpose(rgu_normalized, (1, 2, 0))
            ax[n, rgb_start_col].imshow(
                processed_rgb, origin="lower", interpolation="none"
            )
            ax[n, rgb_start_col + 1].imshow(
                processed_rgu, origin="lower", interpolation="none"
            )
            ax[n, rgb_start_col].axis(False)
            ax[n, rgb_start_col + 1].axis(False)
    for n, axis in enumerate(ax.flat):
        axis.axis(False)
        if crosshairs:
            if len(axis.images) > 0:
                xlim = axis.get_xlim()
                ylim = axis.get_ylim()
                axis.axhline(
                    y=ylim[0] + (ylim[1] - ylim[0]) / 2, color="k", alpha=0.3, lw=1
                )
                axis.axvline(
                    x=xlim[0] + (xlim[1] - xlim[0]) / 2, color="k", alpha=0.3, lw=1
                )
    if labels is not None:
        if labels == "auto":
            labels = [f"ROI {roi}" for roi in rois_specified]
        for axis, label in zip(ax[:, 0].flat, labels):
            axis.axis(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_ylabel(label, rotation="horizontal", labelpad=15)
    if with_times:
        ref_ax = ax[0, 0]
        time_ax = len(colour_idx)
        asp = np.diff(ref_ax.get_ylim())[0] / np.diff(ref_ax.get_xlim())[0]
        max_val = np.max(np.abs([ax.get_ylim() for ax in ax[:, time_ax].flat]))
        for axis in ax[:, time_ax].flat:
            axis.set_ylim(-max_val, max_val)
            time_data_range = 1 * max_val
            spatial_data_height = ref_ax.get_ylim()[1] - ref_ax.get_ylim()[0]
            spatial_data_width = ref_ax.get_xlim()[1] - ref_ax.get_xlim()[0]
            time_asp = (
                asp
                * (spatial_data_height / time_data_range)
                * (time_dur_ms / spatial_data_width)
            )
            axis.set_aspect(time_asp)
        if scalebar:
            y_limits = ax[-1, time_ax].get_ylim()
            value = 5 if y_limits[0] < -5 and y_limits[1] > 5 else y_limits[1]
            value = np.round(value, 2)
            pygor.plotting.add_scalebar(
                value,
                ax=ax[-1, time_ax],
                string=f"{value} SD",
                x=1.025,
                flip_text=True,
                y=0.5,
                line_width=3,
            )
            scalebar_target_ms = 300
            pygor.plotting.add_scalebar(
                scalebar_target_ms,
                ax=ax[-1, time_ax],
                string=f"{scalebar_target_ms} ms",
                line_width=3,
                orientation="h",
                y=0,
            )
    degrees = 15
    visang_to_space = pygor.strf.pixconverter.visang_to_pix(degrees, pixwidth=40)
    if scalebar:
        pygor.plotting.add_scalebar(
            visang_to_space,
            ax=ax[-1, 0],
            string=f"{degrees}\u00b0",
            orientation="h",
            line_width=3,
        )
    return fig, ax


def _contours_plotter(
    data_strf_object, roi, index=None, xy_offset=(0, 0), high_contrast=True, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    contours = pygor.utilities.multicolour_reshape(
        data_strf_object.fit_contours(), data_strf_object.numcolour
    )[:, roi]
    neg_contours = contours[:, 0]
    pos_contours = contours[:, 1]
    if index is None:
        index = range(len(contours))
    if isinstance(index, Iterable) is False:
        index = [index]
    for colour in index:
        for contour_n in neg_contours[colour]:
            if high_contrast == True:
                ax.plot(
                    contour_n[:, 1] + xy_offset[1],
                    contour_n[:, 0] + xy_offset[0],
                    lw=2,
                    ls="-",
                    c="white",
                    alpha=1,
                )  # contour
                ax.plot(
                    contour_n[:, 1] + xy_offset[1],
                    contour_n[:, 0] + xy_offset[0],
                    lw=2,
                    ls="dashed",
                    c=fish_palette[colour],
                    alpha=1,
                )  # contour
            else:
                ax.plot(
                    contour_n[:, 1] + xy_offset[1],
                    contour_n[:, 0] + xy_offset[0],
                    lw=1,
                    ls="-",
                    c=fish_palette[colour],
                    alpha=1,
                )  # contour
        for contour_p in pos_contours[colour]:
            if high_contrast == True:
                ax.plot(
                    contour_p[:, 1] + xy_offset[1],
                    contour_p[:, 0] + xy_offset[0],
                    lw=2,
                    ls="-",
                    c="white",
                    alpha=1,
                )
                ax.plot(
                    contour_p[:, 1] + xy_offset[1],
                    contour_p[:, 0] + xy_offset[0],
                    lw=2,
                    ls="dashed",
                    c=fish_palette[colour],
                    alpha=1,
                )
            else:
                ax.plot(
                    contour_p[:, 1] + xy_offset[1],
                    contour_p[:, 0] + xy_offset[0],
                    lw=1,
                    ls="-",
                    c=fish_palette[colour],
                    alpha=1,
                )


def rgb_representation(
    data_strf_object,
    specify_rois=None,
    colours_dims=[0, 1, 2, 3],
    ipl_sort=False,
    y_crop=(0, 0),
    x_crop=(0, 0),
    ax=None,
    contours=False,
    remove_border=False,
):
    # Create iterators depneding on desired output
    if isinstance(specify_rois, int) or isinstance(
        specify_rois, np.int32
    ):  # user specifies number of rois from "start", although negative is also allowed
        specify_rois = range(specify_rois, specify_rois + data_strf_object.numcolour)
    # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable):  # user specifies specific rois
        specify_rois = specify_rois  # lol
    elif specify_rois == None:  # user wants all rois
        specify_rois = None
    if ipl_sort == True:
        specify_rois = data_strf_object.ipl_depths.argsort()
    n_cols = 1
    # If more than can be represnted as RGB, we need to spill over into another column
    if isinstance(colours_dims, Iterable) is False:
        colours_dims = [colours_dims]
    if len(colours_dims) > 3:
        n_cols = np.ceil(len(colours_dims) / 3).astype(
            "int"
        )  # At most, RGB can be represented in one column
    # Generate axes accordingly
    if ax is None:
        fig, axs = plt.subplots(
            len(specify_rois),
            n_cols,
            sharex=True,
            sharey=True,
            figsize=(n_cols * 4, len(specify_rois) * 2),
        )
    else:
        axs = ax
        fig = plt.gcf()
    rois = list(specify_rois) * 2
    if len(specify_rois) == data_strf_object.numcolour:
        axs = [axs]
    for n, ax in enumerate(axs):
        roi = specify_rois[n]  # Because each row represents a roi
        if y_crop != (0, 0):
            ycropper = y_crop
        else:
            ycropper = (None, None)
        if x_crop != (0, 0):
            xcropper = x_crop
        else:
            xcropper = (None, None)
        # Summary of spatial components
        processed_rgb = data_strf_object.to_rgb(
            roi, rgb_channels=[0, 1, 2], remove_borders=remove_border
        )[:, ycropper[0] : ycropper[1], xcropper[0] : xcropper[1]]
        processed_rgu = data_strf_object.to_rgb(
            roi, rgb_channels=[0, 1, 3], remove_borders=remove_border
        )[:, ycropper[0] : ycropper[1], xcropper[0] : xcropper[1]]
        # for cax in roi_ax[0]:
        ax[0].imshow(processed_rgb, origin="lower", interpolation="none")
        ax[1].imshow(processed_rgu, origin="lower", interpolation="none")
        ax[0].axis(False)
        ax[1].axis(False)
        if contours is True:
            _contours_plotter(
                data_strf_object, index=[0, 1, 2], roi=roi, ax=ax[0]
            )  # , xy_offset = (led_offset, led_offset))
            _contours_plotter(
                data_strf_object, index=[0, 1, 3], roi=roi, ax=ax[1]
            )  # , xy_offset = (led_offset, led_offset[0]))
    # fig.tight_layout(pad = 0.1, h_pad = .1, w_pad=.1)
    return fig, ax


def visualise_summary(
    data_strf_object, specify_rois, ipl_sort=False, y_crop=(0, 0), x_crop=(0, 0)
):
    strfs_chroma = pygor.utilities.multicolour_reshape(
        data_strf_object.collapse_times(), 4
    )
    strfs_rgb = np.abs(np.rollaxis((np.delete(strfs_chroma, 3, 0)), 0, 4))
    strfs_rgu = np.abs(np.rollaxis((np.delete(strfs_chroma, 2, 0)), 0, 4))

    # Proportional normalization - preserve relative magnitudes between RGB channels
    for roi_idx in range(strfs_rgb.shape[0]):
        # RGB normalization
        rgb_channels = [strfs_rgb[roi_idx, :, :, i] for i in range(3)]
        rgb_global_max = max(channel.max() for channel in rgb_channels)
        if rgb_global_max > 0:
            for i in range(3):
                strfs_rgb[roi_idx, :, :, i] = (
                    strfs_rgb[roi_idx, :, :, i] / rgb_global_max
                )

        # RGU normalization
        rgu_channels = [strfs_rgu[roi_idx, :, :, i] for i in range(3)]
        rgu_global_max = max(channel.max() for channel in rgu_channels)
        if rgu_global_max > 0:
            for i in range(3):
                strfs_rgu[roi_idx, :, :, i] = (
                    strfs_rgu[roi_idx, :, :, i] / rgu_global_max
                )

    # Create iterators depneding on desired output
    if isinstance(
        specify_rois, int
    ):  # user specifies number of rois from "start", although negative is also allowed
        specify_rois = range(specify_rois, specify_rois + 1)
        # who cares what ipl_sort does here, the input is an int. What's it supposed to do?!
    elif isinstance(specify_rois, Iterable):  # user specifies specific rois
        specify_rois = specify_rois  # lol
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths[specify_rois].argsort()
    elif specify_rois == None:  # user wants all rois
        specify_rois = range(len(strfs_chroma[0, :]))
        if ipl_sort == True:
            specify_rois = data_strf_object.ipl_depths.argsort()
    fig, ax = plt.subplots(
        len(specify_rois), 3, figsize=(3 * 1.3 * 2, len(specify_rois) * 1.7)
    )
    for n, roi in enumerate(reversed(specify_rois)):
        # Summary of spatial components
        # spaces = np.copy(pygor.utilities.auto_remove_border(strfs_chroma[:, roi])) # this works
        spaces = strfs_chroma[:, roi]
        # Prepare for RGB representation (by intiger)
        spaces[3] = np.roll(
            spaces[3],
            np.round(data_strf_object.calc_LED_offset(), 0).astype("int"),
            axis=(0, 1),
        )
        if y_crop != (0, 0) or x_crop != (0, 0):
            spaces = spaces[:, y_crop[0] : y_crop[1], x_crop[0] : x_crop[1]]
        r, g, b, uv = spaces[0], spaces[1], spaces[2], spaces[3]
        rgb = np.abs(np.array([r, g, b]))
        rgu = np.abs(np.array([r, g, uv]))
        processed_rgb = np.rollaxis(
            pygor.utilities.min_max_norm(rgb, 0, 1), axis=0, start=3
        )
        processed_rgu = np.rollaxis(
            pygor.utilities.min_max_norm(rgu, 0, 1), axis=0, start=3
        )
        # Handle axes differently depending on number of rois (trust me, makes lif easier)
        if len(specify_rois) > 1:
            roi_ax = ax[n]
        else:
            roi_ax = ax
        for cax in roi_ax.flat[::3]:
            rgb_plot = cax.imshow(processed_rgb, origin="lower", interpolation="none")
            cax.axis(False)
            _contours_plotter(data_strf_object, roi=roi, ax=cax)
        for cax in roi_ax.flat[1::3]:
            rgu_plot = cax.imshow(processed_rgu, origin="lower", interpolation="none")
            cax.axis(False)
            _contours_plotter(data_strf_object, roi=roi, ax=cax)
        # Reshape times for convenience
        times = np.ma.copy(
            pygor.utilities.multicolour_reshape(data_strf_object.get_timecourses(), 4)
        )[:, roi]
        for cax in roi_ax.flat[2::3]:
            for colour in range(4):
                curr_colour = times[colour].T
                cax.plot(curr_colour, c=fish_palette[colour])
                cax.set_xticks(
                    np.linspace(0, 20, 5), np.round(np.linspace(0, 1.3, 5), 2)
                )
    plt.tight_layout()
    return fig, ax


def tiling(
    Data_strf_object,
    deletion_threshold=0,
    chromatic=False,
    x_lim=None,
    y_lim=None,
    **kwargs: Any,
):
    """
    Visualizes the tiling of spectro-temporal receptive fields (STRFs).

    This function takes a Data_strf_object, which is assumed to have methods for
    fitting contours and collapsing times. It optionally shrinks the contours and
    filters out contours based on a deletion threshold. It then generates a series
    of plots that show the minimum and maximum projections of the collapsed times,
    as well as a combined plot with optional chromatic or monochromatic contour
    overlays.

    Parameters
    ----------
    Data_strf_object : object
        An object with methods for fitting contours and collapsing times, which are
        used to compute and visualize the STRF tilings.
    deletion_threshold : float, optional
        Threshold below which contours are deleted from the visualization based on
        the maximum amplitude across collapsed times (default is 0).
    chromatic : bool, optional
        If True, contours are plotted in color; otherwise, they are plotted in red
        and blue (default is False).
    x_lim : tuple of int, optional
        Limits for the x-axis of the plots (default is None).
    y_lim : tuple of int, optional
        Limits for the y-axis of the plots (default is None).
    **kwargs : dict
        Additional keyword arguments. Can include 'shrink_factor' to determine the
        scaling factor by which the contours are shrunk.

    Returns
    -------
    None
        The function does not return any values but generates matplotlib plots.
    """

    def _shrink_contour(coordinates, scale_factor):
        # Step 1: Find the center of the contour
        center = np.mean(coordinates, axis=0)
        # Step 2: Translate coordinates to make the center the origin
        translated_coordinates = coordinates - center
        # Step 3: Scale the coordinates to shrink the contour
        scaled_coordinates = scale_factor * translated_coordinates
        # Step 4: Translate coordinates back to their original position
        final_coordinates = scaled_coordinates + center
        return final_coordinates

    def _transform_contours(contours, transform_funct, *params):
        new_contours = []
        for lower, upper in contours:
            curr_upper = []
            curr_lower = []
            for i in upper:
                inner_upper = transform_funct(i, *params)
                curr_upper.append(inner_upper)
            for j in lower:
                inner_lower = transform_funct(j, *params)
                curr_lower.append(inner_lower)
            new_contours.append([curr_lower, curr_upper])
        # return np.array(new_lowers, dtype = "object"), np.array(new_uppers, dtype = "object")
        return np.array(new_contours, dtype="object")

    # Make a copy of the array view
    contours = np.copy(Data_strf_object.fit_contours())
    if "shrink_factor" in kwargs and kwargs["shrink_factor"] != None:
        contours = _transform_contours(
            contours, _shrink_contour, kwargs["shrink_factor"]
        )
    if "shrink_factor" not in kwargs:
        kwargs["shrink_factor"] = 1
    absolute_version = np.abs(Data_strf_object.collapse_times())
    indeces_to_delete = np.unique(
        np.where(np.max(absolute_version, axis=(1, 2)) < deletion_threshold)[0]
    )  # filtering criteria based on amplitudes
    # Kick out contours accordingly
    cleaned_version = np.delete(
        Data_strf_object.collapse_times(), indeces_to_delete, axis=0
    )
    # cleaned_contours = list(np.delete(np.array(contours, dtype = "object"), indeces_to_delete, axis = 0))
    # print(cleaned_contours)
    contours[:, :][indeces_to_delete] = [
        [[]]
    ]  # I dont understand why this works but it does
    #    cleaned_contours = contours
    # Make projectsion
    min_projection = np.min(cleaned_version, axis=0)
    max_projection = np.max(cleaned_version, axis=0)
    min_val = np.min(min_projection)
    max_val = np.max(max_projection)
    # create plot
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    # Plot projections
    minproj = ax[0].imshow(min_projection, cmap="RdBu", origin="lower")
    minproj.set_clim(min_val, max_val)
    plt.colorbar(minproj, ax=ax[0])
    # Plot the other projection
    maxproj = ax[1].imshow(max_projection, cmap="RdBu", origin="lower")
    maxproj.set_clim(min_val, max_val)
    plt.colorbar(maxproj, ax=ax[1])
    # Plot their combination
    combined2 = ax[2].imshow(
        np.abs(min_projection) + np.abs(max_projection),
        cmap="Greys",
        alpha=1,
        origin="lower",
    )
    combined2.set_clim(0, max_val)
    plt.colorbar(combined2, ax=ax[2])
    # Finally plot contours accordingly
    if chromatic == True:
        n = 0
        for i in contours:
            upper, lower = i
            if len(upper) != 0:
                for contour_up in upper:
                    ax[0].plot(
                        contour_up[:, 1],
                        contour_up[:, 0],
                        lw=2,
                        c=pygor.plotting.custom.fish_palette[n],
                        alpha=0.5,
                    )  # contour
                    ax[2].plot(
                        contour_up[:, 1],
                        contour_up[:, 0],
                        lw=2,
                        c=pygor.plotting.custom.fish_palette[n],
                        alpha=0.5,
                    )  # contour
            if len(lower) != 0:
                for contour_low in lower:
                    ax[1].plot(
                        contour_low[:, 1],
                        contour_low[:, 0],
                        lw=2,
                        c=pygor.plotting.custom.fish_palette[n],
                        alpha=0.5,
                    )  # contour
                    ax[2].plot(
                        contour_low[:, 1],
                        contour_low[:, 0],
                        lw=2,
                        c=pygor.plotting.custom.fish_palette[n],
                        alpha=0.5,
                    )  # contour
            n += 1
            if n == 4:
                n = 0
    else:
        for i in contours:
            upper, lower = i
            if len(upper) != 0:
                for contour_up in upper:
                    ax[0].plot(
                        contour_up[:, 1] / kwargs["shrink_factor"],
                        contour_up[:, 0] / kwargs["shrink_factor"],
                        lw=2,
                        c="red",
                        alpha=0.25,
                    )  # contour
                    ax[2].plot(
                        contour_up[:, 1] / kwargs["shrink_factor"],
                        contour_up[:, 0] / kwargs["shrink_factor"],
                        lw=2,
                        c="red",
                        alpha=0.4,
                    )  # contour
            if len(lower) != 0:
                for contour_low in lower:
                    ax[1].plot(
                        contour_low[:, 1] / kwargs["shrink_factor"],
                        contour_low[:, 0] / kwargs["shrink_factor"],
                        lw=2,
                        c="blue",
                        alpha=0.25,
                    )  # contour
                    ax[2].plot(
                        contour_low[:, 1] / kwargs["shrink_factor"],
                        contour_low[:, 0] / kwargs["shrink_factor"],
                        lw=2,
                        c="blue",
                        alpha=0.4,
                    )  # contour
    if x_lim != None:
        for a in ax.flat:
            a.set_xlim(x_lim[0], x_lim[1])
    if y_lim != None:
        for a in ax.flat:
            a.set_ylim(y_lim[0], y_lim[1])
    # ax[3].imshow(np.average(load.images, axis = 0), cmap = 'Greys_r', origin = "lower")
    # plt.savefig(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Conferences\Life Sciences PhD Careers Symposium 2023\RF_tiling.svg")


def multi_chroma_movie(strf_object, roi, show_cbar=False, **kwargs: Any):
    # This is way more efficient than the legacy version and does not rely on ipywidgets
    # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib

    # Return default matplotlib plotting parameters to new dict and change those needed
    plot_settings = plt.rcParams
    plot_settings["animation.html"] = "jshtml"
    plot_settings["figure.dpi"] = 100
    plot_settings["savefig.facecolor"] = "white"

    num_colours = strf_object.numcolour
    multichrom = pygor.utilities.multicolour_reshape(strf_object.strfs, num_colours)[
        :, roi
    ]
    # Use RC context manager to temporarily use the modified rc dict
    animation = pygor.plotting.play_movie_4d(
        multichrom, show_cbar=show_cbar, cmap_list=pygor.plotting.maps_concat
    )
    return animation


def spatial_colors(d3_srf_arr):
    minmax_abs = np.max(np.abs(d3_srf_arr))
    fig, axs = plt.subplots(1, 4, figsize=(10, 4))
    for n, ax in enumerate(axs):
        ax.pcolormesh(
            d3_srf_arr[n],
            vmin=-minmax_abs,
            vmax=minmax_abs,
            cmap=pygor.plotting.maps_concat[n],
        )
        ax.set_aspect("equal")
        ax.axis("off")
    plt.close()
    return fig


def spacetime_plot(
    strf_arr, slice_along="y", avg_sides=3, ax=None, cmap=None, clim=None, **kwargs: Any
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prune = pygor.utilities.auto_remove_border(strf_arr)
    prune = pygor.strf.spatial.centre_on_max(prune)
    collapsed = np.ma.var(prune, axis=0)
    maxindex = np.unravel_index(np.argmax(np.abs(collapsed)), collapsed.shape)
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    if slice_along == "y":
        if avg_sides == None:
            arr = prune[:, :, maxindex[1]].T
        else:
            avg_from = maxindex[1] - avg_sides
            avg_to = maxindex[1] + avg_sides
            arr = prune[:, :, avg_from:avg_to].T
            arr = np.average(arr, axis=0)
    if slice_along == "x":
        if avg_sides == None:
            arr = prune[:, maxindex[0]].T
        else:
            avg_from = maxindex[0] - avg_sides
            avg_to = maxindex[0] + avg_sides
            arr = prune[:, avg_from:avg_to, :].T
            arr = np.average(arr, axis=1)
    if cmap == None:
        cmap = plt.get_cmap()
    if clim == None:
        clim = (-np.max(np.abs(arr)), np.max(np.abs(arr)))
        if clim[1] < 5:
            clim = (-5, 5)
    ax.imshow(arr, cmap=cmap, clim=clim, origin="lower", **kwargs)
    return fig, ax
