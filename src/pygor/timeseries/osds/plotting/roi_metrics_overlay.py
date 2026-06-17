"""
Spatial overlays of per-ROI directional metrics on the imaging field of view.

The main entry point, :func:`plot_ds_overlay`, fills each ROI's mask region with a
colour mapped from its directional selectivity index (DSI) and draws an arrow in the
ROI's preferred direction (length proportional to DSI), over the average imaging stack.
Moving-bar recordings have two phases (ON edge / OFF edge), so the figure has one
subplot per phase.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


def _as_2d_phase(arr):
    """Coerce a metric array to shape (n_phases, n_rois)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def _resolve_background(osds_obj, background):
    """Return a 2D grayscale background image."""
    if background == 'average_stack':
        bg = getattr(osds_obj, 'average_stack', None)
        if bg is not None:
            return np.asarray(bg)
        images = getattr(osds_obj, 'images', None)
        if images is not None:
            return np.mean(images, axis=0)
        raise ValueError(
            "No background available: both average_stack and images are None."
        )
    # Allow passing an explicit image array.
    return np.asarray(background)


def plot_ds_overlay(osds_obj, axes=None, metric=None,
                    cmap='viridis', vmin=None, vmax=None,
                    fill_alpha=0.7, show_arrows=True, arrow_scale=15.0,
                    arrow_color='white', phase_labels=("ON edge", "OFF edge"),
                    show_colorbar=True, background='average_stack', figsize=None):
    """
    Overlay per-ROI DSI (colour fill) and preferred direction (arrows) on the FOV.

    Parameters
    ----------
    osds_obj : OSDS object
        Recording with ROIs, averages and tuning data.
    axes : array/list of Axes or None
        One axes per phase. If None, axes are created automatically.
    metric : str or callable or None
        Tuning metric used to compute DSI / preferred direction. If None, uses
        ``osds_obj.tuning_metric``.
    cmap : str or Colormap
        Colormap mapping DSI to fill colour.
    vmin, vmax : float or None
        DSI range for the colour normalisation. If None (default), the colormap
        autoscales to the actual min/max of the DSI values.
    fill_alpha : float
        Alpha of the per-ROI colour fill over the background.
    show_arrows : bool
        Draw a preferred-direction arrow per ROI (length proportional to DSI).
    arrow_scale : float
        Arrow length in pixels at DSI = 1.
    arrow_color : str
        Colour of the direction arrows.
    phase_labels : sequence of str
        Title per phase subplot.
    show_colorbar : bool
        Add a shared DSI colorbar.
    background : str or ndarray
        'average_stack' (default, falls back to mean of ``images``) or an explicit image.
    figsize : tuple or None
        Figure size. Auto-scaled from the number of phases if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        One axes per phase.
    """
    if getattr(osds_obj, 'rois', None) is None:
        raise ValueError("No ROIs defined on this recording. Impossible to overlay.")

    # --- Metrics (phase-aware), coerced to (n_phases, n_rois) ----------------
    dsi = _as_2d_phase(osds_obj.get_dsi(metric=metric, phase_aware=True))
    pref = _as_2d_phase(osds_obj.get_preferred_direction(metric=metric, phase_aware=True))
    n_phases, n_rois = dsi.shape

    # --- Spatial references --------------------------------------------------
    bg = _resolve_background(osds_obj, background)
    rois_alt = osds_obj.rois_alt          # 0-based labels, background NaN
    centroids = osds_obj.roi_centroids    # (n_rois, 2) as [y, x]

    # Autoscale to the actual DSI range when limits are not given.
    if vmin is None or vmax is None:
        finite = dsi[np.isfinite(dsi)]
        data_min = float(np.min(finite)) if finite.size else 0.0
        data_max = float(np.max(finite)) if finite.size else 1.0
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    # --- Axes ----------------------------------------------------------------
    if axes is None:
        if figsize is None:
            figsize = (5.5 * n_phases, 5.0)
        fig, axes = plt.subplots(1, n_phases, figsize=figsize, squeeze=False)
        axes = list(axes[0])
    else:
        axes = list(np.atleast_1d(axes).ravel())
        if len(axes) < n_phases:
            raise ValueError(f"Expected at least {n_phases} axes, got {len(axes)}.")
        fig = axes[0].figure

    # --- Per-phase rendering -------------------------------------------------
    for p in range(n_phases):
        ax = axes[p]
        ax.imshow(bg, cmap='Greys_r', origin='lower')

        # Build a single RGBA buffer of all ROI fills for this phase.
        rgba = np.zeros((*bg.shape, 4), dtype=float)
        for i in range(n_rois):
            d = dsi[p, i]
            if np.isnan(d):
                continue
            mask = rois_alt == i
            if not mask.any():
                continue
            colour = cmap(norm(d))
            rgba[mask] = (colour[0], colour[1], colour[2], fill_alpha)
        ax.imshow(rgba, origin='lower')

        # Preferred-direction arrows, length proportional to DSI.
        if show_arrows:
            for i in range(n_rois):
                d, ang = dsi[p, i], pref[p, i]
                if np.isnan(d) or np.isnan(ang):
                    continue
                cy, cx = centroids[i]
                theta = np.deg2rad(ang)
                length = arrow_scale * np.clip(d, 0.0, 1.0)
                if length <= 0:
                    continue
                ax.annotate(
                    '', xy=(cx + length * np.cos(theta), cy + length * np.sin(theta)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5),
                )

        label = phase_labels[p] if p < len(phase_labels) else f"phase {p}"
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Shared colorbar -----------------------------------------------------
    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04, label="DSI")

    return fig, axes
