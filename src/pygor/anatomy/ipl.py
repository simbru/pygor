"""
IPL (Inner Plexiform Layer) depth estimation utilities.

Provides functions for:
- Interpolating and smoothing boundary polylines
- Determining scan orientation (horizontal vs vertical)
- Computing IPL depth percentages for ROI centroids
- Automated boundary estimation from ROI spatial distributions

AI-generated: This module was created with Claude Code assistance.
"""

import warnings
import numpy as np
from scipy.signal import savgol_filter


def interp_boundary(coords, n_points=1000, smooth=True, smooth_window=None,
                    poly_order_smooth=1):
    """Interpolate a polyline boundary to evenly-spaced points with optional smoothing.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Raw polyline coordinates in (y, x) format.
    n_points : int, optional
        Number of evenly-spaced output points (default: 1000).
    smooth : bool, optional
        Whether to apply Savitzky-Golay smoothing (default: True).
    smooth_window : int or None, optional
        Window length for the Savitzky-Golay filter. If None, defaults to
        ``n_points // 2`` (forced odd).
    poly_order_smooth : int, optional
        Polynomial order for the Savitzky-Golay filter (default: 3).

    Returns
    -------
    np.ndarray, shape (n_points, 2)
        Interpolated (and optionally smoothed) boundary coordinates in (y, x).
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be shape (N, 2), got {coords.shape}")
    if coords.shape[0] < 2:
        raise ValueError("Need at least 2 points to interpolate a boundary")

    # Cumulative arc-length parameterisation
    distances = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)

    # Evenly-spaced samples along arc length
    interp_distances = np.linspace(0, distances[-1], n_points)
    y_interp = np.interp(interp_distances, distances, coords[:, 0])
    x_interp = np.interp(interp_distances, distances, coords[:, 1])

    if not smooth:
        return np.column_stack((y_interp, x_interp))

    # Savitzky-Golay smoothing
    if smooth_window is None:
        smooth_window = int(n_points / 2)
    if smooth_window % 2 == 0:
        smooth_window -= 1
    smooth_window = max(smooth_window, poly_order_smooth + 2)  # must exceed poly order

    y_smooth = savgol_filter(y_interp, smooth_window, poly_order_smooth)
    x_smooth = savgol_filter(x_interp, smooth_window, poly_order_smooth)
    return np.column_stack((y_smooth, x_smooth))


def determine_orientation(x_coords, y_coords):
    """Determine whether a boundary curve runs horizontally or vertically.

    Compares normalised derivative ranges in X vs Y to decide which axis
    the boundary varies along most.

    Parameters
    ----------
    x_coords : array-like
        X component of the boundary curve.
    y_coords : array-like
        Y component of the boundary curve.

    Returns
    -------
    str
        ``"horizontal"`` if X variations dominate, ``"vertical"`` otherwise.
    """
    x_func = np.asarray(x_coords, dtype=float)
    y_func = np.asarray(y_coords, dtype=float)

    # Normalise
    x_max = np.max(np.abs(x_func)) if np.any(x_func) else 1.0
    y_max = np.max(np.abs(y_func)) if np.any(y_func) else 1.0
    x_calc = x_func / x_max
    y_calc = y_func / y_max

    # Numerical derivative
    calc_dif_x = np.diff(x_calc)
    calc_dif_y = np.diff(y_calc)

    # Range as product of extrema (captures sign changes → large oscillations)
    x_range = (np.max(calc_dif_x) * np.min(calc_dif_x)) if len(calc_dif_x) > 0 else 0
    y_range = (np.max(calc_dif_y) * np.min(calc_dif_y)) if len(calc_dif_y) > 0 else 0

    return "horizontal" if x_range >= y_range else "vertical"


def _auto_detect_orientation(roi_centroids):
    """Detect scan orientation from the spatial spread of ROI centroids.

    Parameters
    ----------
    roi_centroids : np.ndarray, shape (n_rois, 2)
        ROI centroid positions in (y, x) format.

    Returns
    -------
    str
        ``"horizontal"`` if the centroids span wider in X than Y,
        ``"vertical"`` otherwise.
    """
    y_spread = np.ptp(roi_centroids[:, 0])
    x_spread = np.ptp(roi_centroids[:, 1])
    return "horizontal" if x_spread >= y_spread else "vertical"


def calculate_ipl_depths(roi_centroids, upper_boundary, lower_boundary,
                         orientation=None):
    """Compute IPL depth percentage for each ROI between two boundaries.

    For each ROI centroid, finds the closest point on each boundary (matching
    by X coordinate for horizontal scans, Y for vertical) and linearly
    interpolates the depth position as a percentage (0 % = lower/outer
    boundary, 100 % = upper/inner boundary).

    Parameters
    ----------
    roi_centroids : array-like, shape (n_rois, 2)
        ROI centroid positions in (y, x) format.
    upper_boundary : array-like, shape (M, 2)
        The 100 % (inner) boundary in (y, x) format.
    lower_boundary : array-like, shape (M, 2)
        The 0 % (outer) boundary in (y, x) format.
    orientation : str or None, optional
        ``"horizontal"`` or ``"vertical"``. If None, auto-detected from
        the lower boundary curve via :func:`determine_orientation`.

    Returns
    -------
    np.ndarray, shape (n_rois,)
        Depth percentages for each ROI (0–100 range, may exceed bounds for
        ROIs outside the boundary pair).
    """
    roi_centroids = np.asarray(roi_centroids, dtype=float)
    upper = np.asarray(upper_boundary, dtype=float)
    lower = np.asarray(lower_boundary, dtype=float)

    if orientation is None:
        orientation = determine_orientation(lower[:, 1], lower[:, 0])

    if orientation == "horizontal":
        return _depths_horizontal(lower, upper, roi_centroids)
    else:
        return _depths_vertical(lower, upper, roi_centroids)


def _depths_horizontal(lower, upper, roi_centroids):
    """Depth calculation for horizontal scans (match by X, measure along Y)."""
    upper_x = upper[:, 1]
    lower_x = lower[:, 1]
    roi_x = roi_centroids[:, 1]
    roi_y = roi_centroids[:, 0]

    # Closest X match on each boundary
    closest_upper = np.argmin(np.abs(upper_x[:, None] - roi_x), axis=0)
    closest_lower = np.argmin(np.abs(lower_x[:, None] - roi_x), axis=0)

    y_upper = upper[closest_upper, 0]
    y_lower = lower[closest_lower, 0]

    return (roi_y - y_lower) / (y_upper - y_lower) * 100


def _depths_vertical(lower, upper, roi_centroids):
    """Depth calculation for vertical scans (match by Y, measure along X)."""
    upper_y = upper[:, 0]
    lower_y = lower[:, 0]
    roi_y = roi_centroids[:, 0]
    roi_x = roi_centroids[:, 1]

    # Closest Y match on each boundary
    closest_upper = np.argmin(np.abs(upper_y[:, None] - roi_y), axis=0)
    closest_lower = np.argmin(np.abs(lower_y[:, None] - roi_y), axis=0)

    x_upper = upper[closest_upper, 1]
    x_lower = lower[closest_lower, 1]

    # Handle reversed boundaries (0 % on the right / higher-X side)
    if np.mean(x_lower) > np.mean(x_upper):
        return (x_lower - roi_x) / (x_lower - x_upper) * 100
    else:
        return (roi_x - x_lower) / (x_upper - x_lower) * 100


def estimate_ipl_boundaries(roi_centroids, n_bins=1, upper_percentile=0,
                            lower_percentile=100.0,
                            orientation=None, margin_fraction=.1,
                            n_points=100):
    """Estimate upper and lower IPL boundaries automatically from ROI positions.

    Bins the ROI centroids along the scan axis and takes percentiles in each
    bin to approximate the outer (0 %) and inner (100 %) boundaries of the
    IPL. A running median filter removes single-bin outlier spikes, followed
    by light Savitzky-Golay smoothing for a clean curve.

    Parameters
    ----------
    roi_centroids : array-like, shape (n_rois, 2)
        ROI centroid positions in (y, x) format.
    n_bins : int, optional
        Number of bins along the scan axis (default: 15).
    upper_percentile : float, optional
        Percentile for the inner (100 %) boundary within each bin
        (default: 5.0).
    lower_percentile : float, optional
        Percentile for the outer (0 %) boundary within each bin
        (default: 95.0).
    orientation : str or None, optional
        ``"horizontal"`` or ``"vertical"``. If None, auto-detected from
        centroid spread.
    margin_fraction : float, optional
        Fraction of the boundary gap to expand outward, so the boundaries
        wrap past the outermost ROIs (default: 0.05).
    n_points : int, optional
        Number of points for the output boundaries (default: 1000).

    Returns
    -------
    upper_boundary : np.ndarray, shape (n_points, 2)
        The estimated 100 % (inner) boundary in (y, x) format.
    lower_boundary : np.ndarray, shape (n_points, 2)
        The estimated 0 % (outer) boundary in (y, x) format.

    Raises
    ------
    ValueError
        If fewer than 4 ROI centroids are provided.
    """
    from scipy.ndimage import median_filter

    roi_centroids = np.asarray(roi_centroids, dtype=float)
    if roi_centroids.shape[0] < 4:
        raise ValueError(
            f"Need at least 4 ROI centroids for boundary estimation, "
            f"got {roi_centroids.shape[0]}"
        )

    if orientation is None:
        orientation = _auto_detect_orientation(roi_centroids)

    # scan_axis: the axis along which we bin (X for horizontal, Y for vertical)
    # depth_axis: the axis along which depth varies (Y for horizontal, X for vertical)
    if orientation == "horizontal":
        scan_coords = roi_centroids[:, 1]   # X
        depth_coords = roi_centroids[:, 0]  # Y
    else:
        scan_coords = roi_centroids[:, 0]   # Y
        depth_coords = roi_centroids[:, 1]  # X

    # Create bins along the scan axis
    scan_min, scan_max = np.min(scan_coords), np.max(scan_coords)
    bin_edges = np.linspace(scan_min, scan_max, n_bins + 1)
    bin_indices = np.digitize(scan_coords, bin_edges) - 1
    # Clamp to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    upper_depths = []  # 100 % boundary depth values
    lower_depths = []  # 0 % boundary depth values
    scan_centers = []
    valid_bins = 0

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) < 2:
            continue
        valid_bins += 1
        bin_scan = scan_coords[mask]
        bin_depth = depth_coords[mask]
        scan_centers.append(np.mean(bin_scan))

        upper_depths.append(np.percentile(bin_depth, upper_percentile))
        lower_depths.append(np.percentile(bin_depth, lower_percentile))

    if valid_bins < 3:
        raise ValueError(
            f"Only {valid_bins} bins had >= 2 ROIs. Increase ROI count or "
            f"decrease n_bins (currently {n_bins})."
        )

    if valid_bins < n_bins * 0.5:
        warnings.warn(
            f"Only {valid_bins}/{n_bins} bins had sufficient ROIs. "
            f"Boundary estimation may be unreliable.",
            stacklevel=2,
        )

    scan_centers = np.array(scan_centers)
    upper_depths = np.array(upper_depths)
    lower_depths = np.array(lower_depths)

    # --- Median filter to remove single-bin outlier spikes ---
    # A kernel of 3 knocks out isolated bins where an outlier ROI pulled the
    # boundary inward, without affecting the overall trend.
    if valid_bins >= 3:
        upper_depths = median_filter(upper_depths, size=3, mode='nearest')
        lower_depths = median_filter(lower_depths, size=3, mode='nearest')

    # Expand boundaries outward by margin_fraction
    gap = lower_depths - upper_depths  # per-bin gap
    upper_depths = upper_depths - margin_fraction * gap
    lower_depths = lower_depths + margin_fraction * gap

    # Extend the scan range to cover the full image width/height
    # This ensures boundaries extend past cut-off borders for proper depth stats
    scan_range = scan_max - scan_min
    extend_amount = scan_range * 0.025  # extend 2.5% beyond each edge
    extended_scan_start = scan_min - extend_amount
    extended_scan_end = scan_max + extend_amount

    # Prepend and append points at the extended scan positions
    # using the edge depth values (flat extrapolation in depth direction)
    scan_centers = np.concatenate([
        [extended_scan_start], scan_centers, [extended_scan_end]
    ])
    upper_depths = np.concatenate([
        [upper_depths[0]], upper_depths, [upper_depths[-1]]
    ])
    lower_depths = np.concatenate([
        [lower_depths[0]], lower_depths, [lower_depths[-1]]
    ])

    # Convert back to (y, x) format for interp_boundary
    if orientation == "horizontal":
        # scan = X, depth = Y → (y, x) = (depth, scan)
        upper_yx = np.column_stack((upper_depths, scan_centers))
        lower_yx = np.column_stack((lower_depths, scan_centers))
    else:
        # scan = Y, depth = X → (y, x) = (scan, depth)
        upper_yx = np.column_stack((scan_centers, upper_depths))
        lower_yx = np.column_stack((scan_centers, lower_depths))

    # Interpolate to n_points with light smoothing.  The median-filtered
    # bin points are clean enough for gentle Savitzky-Golay; use a window
    # proportional to the number of raw points (not n_points) to avoid
    # over-smoothing while still producing a visually smooth curve.
    raw_pts = len(scan_centers)
    # Map raw-point spacing to upsampled spacing, then use ~3 raw-point
    # spans as the smooth window (covers ~3 bins worth of signal).
    smooth_win = max(int(n_points / raw_pts * 3), 5)
    if smooth_win % 2 == 0:
        smooth_win += 1
    # Cap at half of n_points to stay within savgol limits
    smooth_win = min(smooth_win, n_points // 2)
    if smooth_win % 2 == 0:
        smooth_win -= 1

    upper_boundary = interp_boundary(
        upper_yx, n_points=n_points, smooth=True, smooth_window=smooth_win,
    )
    lower_boundary = interp_boundary(
        lower_yx, n_points=n_points, smooth=True, smooth_window=smooth_win,
    )

    return upper_boundary, lower_boundary


def plot_ipl_estimation(image, roi_centroids, upper_boundary, lower_boundary,
                        depths, ax=None, figsize=(10, 6), cmap_image="Greys_r",
                        depth_bins=10, depth_range=(0.0, 100.0)):
    """Plot IPL boundary estimation with image, centroids, boundaries, and depth histogram.

    Creates a two-panel figure: the left panel shows the mean image with
    ROI centroids and fitted boundaries overlaid; the right panel shows a
    histogram of the estimated IPL depths aligned to the image height.

    Parameters
    ----------
    image : np.ndarray, 2D
        Background image (e.g. mean of imaging stack).
    roi_centroids : array-like, shape (n_rois, 2)
        ROI centroid positions in (y, x) format.
    upper_boundary : np.ndarray, shape (M, 2)
        The 100 % (inner) boundary in (y, x) format.
    lower_boundary : np.ndarray, shape (M, 2)
        The 0 % (outer) boundary in (y, x) format.
    depths : np.ndarray, shape (n_rois,)
        Estimated IPL depth percentages.
    ax : None
        Reserved for future use. Currently ignored (always creates new figure).
    figsize : tuple, optional
        Figure size (default: (10, 6)).
    cmap_image : str, optional
        Colormap for the background image (default: ``"Greys_r"``).
    depth_bins : int, optional
        Number of histogram bins for depth values (default: 10).
    depth_range : tuple, optional
        Min and max depth values for the histogram (default: (0, 100)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : tuple of matplotlib.axes.Axes
        ``(ax_image, ax_hist)`` — the image axis and the histogram axis.
    """
    import matplotlib.pyplot as plt

    roi_centroids = np.asarray(roi_centroids)
    depths = np.asarray(depths)

    fig, (ax_img, ax_kde) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": (3, 1)},
    )

    # --- Left panel: image + centroids + boundaries ---
    ax_img.imshow(image, cmap=cmap_image, origin='lower')
    ax_img.scatter(
        roi_centroids[:, 1], roi_centroids[:, 0],
        c=depths, cmap="coolwarm", s=8, edgecolors="none", zorder=3,
        vmin=0, vmax=100,
    )
    ax_img.plot(
        upper_boundary[:, 1], upper_boundary[:, 0],
        color="red", linewidth=1.5, label="100% (inner)",
    )
    ax_img.plot(
        lower_boundary[:, 1], lower_boundary[:, 0],
        color="blue", linewidth=1.5, label="0% (outer)",
    )
    ax_img.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax_img.set_title("IPL boundary estimation")
    ax_img.set_xlabel("X (px)")
    ax_img.set_ylabel("Y (px)")

    # Right panel: histogram of depths
    valid_depths = depths[np.isfinite(depths)]
    depth_min, depth_max = depth_range
    if depth_max <= depth_min:
        raise ValueError("depth_range must be (min, max) with max > min")

    # Map depth values to pixel Y using boundary medians
    y_upper_med = np.median(upper_boundary[:, 0])
    y_lower_med = np.median(lower_boundary[:, 0])

    def _depth_to_y(depth_vals):
        norm = (depth_vals - depth_min) / (depth_max - depth_min)
        return y_lower_med + norm * (y_upper_med - y_lower_med)

    if len(valid_depths) > 0:
        in_range = (valid_depths >= depth_min) & (valid_depths <= depth_max)
        depth_vals = valid_depths[in_range]
        if len(depth_vals) > 0:
            counts, edges = np.histogram(
                depth_vals, bins=depth_bins, range=depth_range
            )
            y_edges = _depth_to_y(edges)
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
            heights = np.abs(np.diff(y_edges))
            ax_kde.barh(
                y_centers, counts, height=heights,
                color="steelblue", alpha=0.6, edgecolor="none",
            )

    # Set y-limits to match image dimensions exactly
    ax_img.set_ylim(-0.5, image.shape[0] - 0.5)
    ax_img.set_xlim(-0.5, image.shape[1] - 0.5)
    
    # Set histogram y-limits to match image
    ax_kde.set_ylim(-0.5, image.shape[0] - 0.5)
    ax_kde.set_xlabel("Count")
    ax_kde.set_xlim(left=0)
    
    # Add IPL depth labels on left side of histogram
    ax_kde.set_yticks([y_lower_med, y_upper_med])
    ax_kde.set_yticklabels(['0%', '100%'])
    # ax_kde.set_ylabel("IPL depth")
    ax_kde.tick_params(axis="y", which="both", left=True, labelleft=True)
    
    ax_kde.spines['left'].set_visible(True)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['bottom'].set_visible(True)
    ax_img.spines['top'].set_visible(False)
    ax_img.spines['right'].set_visible(False)
    ax_img.spines['bottom'].set_visible(True)
    ax_img.spines['left'].set_visible(True)
    
    # Adjust figure height based on image aspect ratio
    img_aspect = image.shape[0] / image.shape[1]
    fig_width = fig.get_figwidth()
    # Account for width_ratios (3:1) - image takes 3/4 of width
    img_width_inches = fig_width * 0.75 * 0.8  # adjusted for margins
    fig_height = img_width_inches * img_aspect + 1  # add space for labels
    fig.set_figheight(fig_height)
    
    plt.tight_layout()
    return fig, (ax_img, ax_kde)
