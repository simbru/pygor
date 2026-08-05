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
    interpolates the depth position as a percentage.

    ``upper_boundary`` is 0 % and ``lower_boundary`` is 100 % by argument
    position alone -- neither name implies a geometric or anatomical side.
    The interpolation is signed, so passing the two the other way round
    returns ``100 - depth``, silently.

    Lab convention for what those percentages mean anatomically:

    ====== ================================================
    0 %    ON layer, proximal, GCL side
    100 %  OFF layer, distal, INL side
    ====== ================================================

    The ON/OFF border sits near 40 %. Whichever boundary is passed as
    ``upper_boundary`` defines the 0 % end, so it is the caller's job to
    make that the GCL/ON side consistently across recordings -- this
    function cannot check it.

    Parameters
    ----------
    roi_centroids : array-like, shape (n_rois, 2)
        ROI centroid positions in (y, x) format.
    upper_boundary : array-like, shape (M, 2)
        The 0 % boundary in (y, x) format.
    lower_boundary : array-like, shape (M, 2)
        The 100 % boundary in (y, x) format.
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

    return (roi_y - y_upper) / (y_lower - y_upper) * 100


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

    return (roi_x - x_upper) / (x_lower - x_upper) * 100


def estimate_ipl_boundaries(roi_centroids, n_bins=8, upper_percentile=0,
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
        Number of bins along the scan axis (default: 8, matching
        ``Core.estimate_ipl_depths``). At least 3 bins must end up with 2 or
        more ROIs, so values below 3 always raise.
    upper_percentile : float, optional
        Percentile for the outer (0 %) boundary within each bin
        (default: 5.0).
    lower_percentile : float, optional
        Percentile for the inner (100 %) boundary within each bin
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
        The estimated 0 % (outer) boundary in (y, x) format.
    lower_boundary : np.ndarray, shape (n_points, 2)
        The estimated 100 % (inner) boundary in (y, x) format.

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

    upper_depths = []  # 0 % boundary depth values (upper_percentile, ON/GCL side)
    lower_depths = []  # 100 % boundary depth values (lower_percentile, OFF/INL side)
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


def orientation_from_anatomy(image, smooth_sigma=1.5, border_crop=6):
    """Detect scan orientation from the image rather than the ROI centroids.

    The depth axis is the one running across the IPL, along which mean
    intensity rises and falls; along the scan axis the band is roughly
    uniform. Comparing the coefficient of variation of the two mean profiles
    is scale-free, so unequal sampling between axes (e.g. a 64 x 128 frame)
    does not bias the result the way comparing raw pixel spreads does.

    Unlike :func:`_auto_detect_orientation`, this does not depend on the ROI
    set, so it is unaffected by ROIs having been culled.

    Parameters
    ----------
    image : np.ndarray, 2D
        Anatomical image, e.g. the mean of the imaging stack.
    smooth_sigma : float, optional
        Gaussian pre-smoothing in pixels (default: 1.5).
    border_crop : int, optional
        Pixels trimmed from each edge before profiling, to exclude the light
        artifact strip (default: 6).

    Returns
    -------
    str
        ``"horizontal"`` (depth along Y) or ``"vertical"`` (depth along X).
    """
    from scipy.ndimage import gaussian_filter

    image = np.asarray(image, dtype=float)
    cropped = _crop_borders(image, border_crop)[0]
    img = gaussian_filter(cropped, smooth_sigma)
    prof_y = img.mean(axis=1)
    prof_x = img.mean(axis=0)
    cv_y = prof_y.std() / (prof_y.mean() + 1e-12)
    cv_x = prof_x.std() / (prof_x.mean() + 1e-12)
    return "horizontal" if cv_y >= cv_x else "vertical"


def _crop_borders(image, border_crop):
    """Trim the light-artifact frame edge; returns (cropped, y_offset, x_offset)."""
    c = int(border_crop)
    if c <= 0 or 2 * c >= min(image.shape):
        return image, 0, 0
    return image[c:-c, c:-c], c, c


def _contiguous_support(profile, centre, fraction):
    """Indices of the contiguous run around ``centre`` above ``fraction`` of peak.

    Returns a single index when ``centre`` itself is below the cut-off, which
    happens when the profile is bimodal and the intensity-weighted centre falls
    in the dip between two lobes. The caller drops such bins (zero width), and
    that is deliberate: re-anchoring on the profile peak instead recovers bins
    whose band is genuinely ill-defined, and they then distort the fit.
    """
    above = profile >= fraction * profile.max()
    if not above[centre]:
        return np.array([centre])
    lo = centre
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = centre
    while hi < len(profile) - 1 and above[hi + 1]:
        hi += 1
    return np.arange(lo, hi + 1)


def estimate_ipl_boundaries_anatomy(image, orientation=None, n_bins=16,
                                    k_sigma=2.0, smooth_sigma=1.5,
                                    border_crop=6, baseline_percentile=10.0,
                                    centre_poly_deg=2, width_poly_deg=1,
                                    n_points=200, trim_vertical=True,
                                    trim_fraction=0.25):
    """Estimate IPL boundaries from the anatomy instead of the ROI cloud.

    :func:`estimate_ipl_boundaries` takes per-bin percentiles of the ROI
    centroids as the band edges, which requires the ROIs to sample the full
    depth of the IPL. That fails when non-responsive ROIs have been removed:
    the extremes that define the boundaries are gone, unevenly. This function
    reads the band out of the image, so culling ROIs does not affect it.

    Within each bin along the scan axis, the intensity profile across depth
    gives an intensity-weighted centre and standard deviation; the band is
    taken as centre +/- ``k_sigma`` SD. Centre and half-width are then fitted
    separately across the scan axis -- the centre carries the retinal
    curvature, the half-width is near-constant -- which is far more stable
    than fitting the two edges independently.

    Parameters
    ----------
    image : np.ndarray, 2D
        Anatomical image, e.g. the mean of the imaging stack.
    orientation : str or None, optional
        ``"horizontal"``/``"vertical"``. Detected via
        :func:`orientation_from_anatomy` if None.
    n_bins : int, optional
        Bins along the scan axis (default: 16).
    k_sigma : float, optional
        Half-width of the band in intensity-weighted SD (default: 2.0).
    smooth_sigma : float, optional
        Gaussian pre-smoothing in pixels (default: 1.5).
    border_crop : int, optional
        Pixels trimmed from each frame edge (default: 6).
    baseline_percentile : float, optional
        Per-profile background level subtracted before weighting (default: 10).
    centre_poly_deg, width_poly_deg : int, optional
        Polynomial degrees for the centre line and half-width (default: 2, 1).
    n_points : int, optional
        Points in each returned boundary (default: 200).
    trim_vertical : bool, optional
        For vertical sections, take the moments over the band's own support
        rather than the whole profile (default: True). See the note below.
    trim_fraction : float, optional
        Support cut-off as a fraction of peak, used when trimming (default: 0.25).

    Returns
    -------
    upper_boundary, lower_boundary : np.ndarray, shape (n_points, 2)
        Boundaries in (y, x) format, as :func:`calculate_ipl_depths` expects.
        ``upper_boundary`` is always the band edge at the *smaller* coordinate
        along the depth axis -- smaller Y when horizontal, smaller X when
        vertical -- and so becomes 0 %. See the polarity note below.
    orientation : str
        The orientation used.

    Raises
    ------
    ValueError
        If the band could not be tracked across at least 3 bins.

    Notes
    -----
    Polarity is assigned geometrically, not anatomically. Nothing in the
    intensity profile distinguishes the INL border from the GCL border, so
    the smaller-coordinate edge is called 0 % unconditionally. A section
    scanned or mounted the other way round therefore yields ``100 - depth``
    with no warning and no way to detect it from the image alone. Depths are
    only comparable across recordings that share a scan/mounting convention;
    where that is in doubt, draw the boundaries by hand with
    :class:`~pygor.core.gui.methods.NapariDepthPrompt`, which takes the
    assignment from the user instead.

    Validated against IGOR-drawn ``Positions`` on one horizontal recording
    (``examples/strf_demo_data.h5``): r = 0.99, with the geometric convention
    matching. The vertical branch has no ground-truth validation.
    """
    from scipy.ndimage import gaussian_filter, median_filter

    image = np.asarray(image, dtype=float)
    if orientation is None:
        orientation = orientation_from_anatomy(image, smooth_sigma, border_crop)

    cropped, y_off, x_off = _crop_borders(image, border_crop)
    img = gaussian_filter(cropped, smooth_sigma)
    # Work in (depth, scan) regardless of orientation.
    depth_scan = img if orientation == "horizontal" else img.T
    depth_off, scan_off = (y_off, x_off) if orientation == "horizontal" else (x_off, y_off)
    n_depth, n_scan = depth_scan.shape
    depth_idx = np.arange(n_depth, dtype=float)

    edges = np.linspace(0, n_scan, n_bins + 1).astype(int)
    centres, band_mid, band_half = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        profile = depth_scan[:, lo:hi].mean(axis=1)
        profile = np.clip(profile - np.percentile(profile, baseline_percentile), 0, None)
        total = profile.sum()
        if total <= 0:
            continue
        weights = profile
        if trim_vertical and orientation == "vertical":
            # A vertical section puts the depth axis along the long side of the
            # frame, so the band sits in ~50 % dark margin; including that in
            # the moment inflates the SD and pushes the borders into the dark.
            # A horizontal section's band nearly fills its axis, where trimming
            # would instead cut into the band, so it is left alone.
            centre = int(np.clip(round((depth_idx * profile).sum() / total), 0, n_depth - 1))
            support = _contiguous_support(profile, centre, trim_fraction)
            weights = np.zeros_like(profile)
            weights[support] = profile[support]
            total = weights.sum()
            if total <= 0:
                continue
        mid = float((depth_idx * weights).sum() / total)
        var = float((weights * (depth_idx - mid) ** 2).sum() / total)
        half = k_sigma * np.sqrt(max(var, 0.0))
        if half <= 0:
            continue
        centres.append((lo + hi) / 2.0)
        band_mid.append(mid)
        band_half.append(half)

    if len(centres) < 3:
        raise ValueError(
            "Could not track the IPL band across the scan axis; the image may "
            "not show a clear band, or n_bins is too high for its size."
        )

    centres = np.asarray(centres)
    band_mid = median_filter(np.asarray(band_mid), size=3, mode="nearest")
    band_half = median_filter(np.asarray(band_half), size=3, mode="nearest")

    # Reject bins whose centre or width is far from the rest before fitting.
    # A single bad bin at either end of the scan axis will otherwise drag a
    # degree-2 fit into a pronounced bow.
    keep = np.ones(len(centres), dtype=bool)
    for values in (band_mid, band_half):
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad > 0:
            keep &= np.abs(values - med) <= 4.0 * mad
    if keep.sum() >= 3:
        centres, band_mid, band_half = centres[keep], band_mid[keep], band_half[keep]

    def _fit(values, deg):
        deg = int(np.clip(deg, 0, max(0, len(centres) - 2)))
        return np.polyfit(centres, values, deg)

    # The border crop keeps the light artifact out of the profiling, but the
    # returned boundaries should still span the whole frame -- otherwise ROIs
    # in the outer few pixels have no boundary at their scan position and get
    # matched to the truncated end instead. Fit inside the cropped region,
    # report across the full extent.
    n_scan_full = image.shape[1] if orientation == "horizontal" else image.shape[0]
    scan_fine = np.linspace(0, n_scan_full - 1, n_points)
    # Evaluate only within the fitted span, holding flat outside, so a
    # higher-order term cannot run away past the outermost bin.
    scan_clamped = np.clip(scan_fine - scan_off, centres.min(), centres.max())
    mid_fine = np.polyval(_fit(band_mid, centre_poly_deg), scan_clamped)
    half_fine = np.clip(np.polyval(_fit(band_half, width_poly_deg), scan_clamped), 1.0, None)

    upper_depths = mid_fine - half_fine + depth_off
    lower_depths = mid_fine + half_fine + depth_off
    scan_coords = scan_fine

    if orientation == "horizontal":  # depth = y, scan = x
        upper = np.column_stack((upper_depths, scan_coords))
        lower = np.column_stack((lower_depths, scan_coords))
    else:                            # depth = x, scan = y
        upper = np.column_stack((scan_coords, upper_depths))
        lower = np.column_stack((scan_coords, lower_depths))
    return upper, lower, orientation


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
        The 0 % (outer) boundary in (y, x) format.
    lower_boundary : np.ndarray, shape (M, 2)
        The 100 % (inner) boundary in (y, x) format.
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
        color="blue", linewidth=1.5, label="0% (outer)",
    )
    ax_img.plot(
        lower_boundary[:, 1], lower_boundary[:, 0],
        color="red", linewidth=1.5, label="100% (inner)",
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
        # upper_boundary is 0 %, matching calculate_ipl_depths. Mapping these
        # the other way round mirrored the histogram against the image panel.
        norm = (depth_vals - depth_min) / (depth_max - depth_min)
        return y_upper_med + norm * (y_lower_med - y_upper_med)

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
    ax_kde.set_yticks([y_upper_med, y_lower_med])
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
