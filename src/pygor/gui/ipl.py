"""IPL boundary drawing inside the main viewer.

Depth is measured per ROI between two polylines: the 0 % (outer) and 100 %
(inner) boundaries of the inner plexiform layer. `NapariDepthPrompt` in
`pygor/core/gui/methods.py` already does this, but in a viewer of its own
that only computes once the window is closed. Here the boundaries are
layers alongside the ROIs and the depths are computed on request, so the
result can be looked at and the boundaries adjusted without reopening
anything.
"""

import numpy as np

UPPER_LAYER_NAME = "IPL 0% boundary"
LOWER_LAYER_NAME = "IPL 100% boundary"

# Matches the colours the standalone prompt used
_UPPER_COLOUR = "red"
_LOWER_COLOUR = "cyan"


def boundary_layers(viewer):
    """Return the (upper, lower) boundary layers, missing ones as None."""
    upper = viewer.layers[UPPER_LAYER_NAME] if UPPER_LAYER_NAME in viewer.layers else None
    lower = viewer.layers[LOWER_LAYER_NAME] if LOWER_LAYER_NAME in viewer.layers else None
    return upper, lower


def ensure_boundary_layers(viewer, ready_to_draw=True):
    """Add the two boundary Shapes layers, returning them.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer to add the layers to.
    ready_to_draw : bool
        Put the outer boundary into polyline mode and select it, so the
        next click starts drawing rather than needing the tool picked
        first.
    """
    upper, lower = boundary_layers(viewer)

    if lower is None:
        lower = viewer.add_shapes(
            name=LOWER_LAYER_NAME, edge_color=_LOWER_COLOUR, edge_width=1
        )
    if upper is None:
        upper = viewer.add_shapes(
            name=UPPER_LAYER_NAME, edge_color=_UPPER_COLOUR, edge_width=1
        )

    if ready_to_draw:
        lower.mode = "add_polyline"
        upper.mode = "add_polyline"
        viewer.layers.selection = {upper}

    return upper, lower


def boundary_coords(layer):
    """Return the drawn polyline as (N, 2) in (y, x), or None if empty.

    The last shape drawn wins, so redrawing a boundary supersedes an
    earlier attempt without needing the old one deleted.
    """
    if layer is None or len(layer.data) == 0:
        return None
    coords = np.asarray(layer.data[-1], dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None
    return coords[:, -2:]


def smooth_boundary(coords, n_points=1000):
    """Interpolate a drawn polyline to evenly spaced, smoothed points."""
    from pygor.anatomy.ipl import interp_boundary

    return interp_boundary(coords, n_points=n_points)


def compute_depths(recording, viewer, orientation=None):
    """Compute IPL depths from the drawn boundaries.

    Returns
    -------
    tuple
        ``(depths, message)``. ``depths`` is None when the boundaries are
        not both drawn or the calculation fails, and ``message`` says why.
    """
    from pygor.anatomy.ipl import calculate_ipl_depths

    upper_layer, lower_layer = boundary_layers(viewer)
    upper = boundary_coords(upper_layer)
    lower = boundary_coords(lower_layer)

    missing = [
        name
        for name, coords in ((UPPER_LAYER_NAME, upper), (LOWER_LAYER_NAME, lower))
        if coords is None
    ]
    if missing:
        return None, f"Draw both boundaries first — missing {', '.join(missing)}"

    if getattr(recording, "rois", None) is None:
        return None, "No ROIs to measure"

    try:
        depths = calculate_ipl_depths(
            recording.roi_centroids,
            upper_boundary=smooth_boundary(upper),
            lower_boundary=smooth_boundary(lower),
            orientation=orientation,
        )
    except Exception as exc:
        return None, f"Depth calculation failed: {exc}"

    depths = np.asarray(depths, dtype=float)
    n_rois = int(getattr(recording, "num_rois", depths.size))
    if depths.size != n_rois:
        return None, f"Got {depths.size} depths for {n_rois} ROIs"

    return depths, (
        f"IPL depths for {depths.size} ROIs: "
        f"{np.nanmin(depths):.0f}% to {np.nanmax(depths):.0f}%"
    )


def estimate_depths(recording):
    """Estimate boundaries automatically, without drawing anything."""
    if not hasattr(recording, "estimate_ipl_depths"):
        return None, "This recording cannot estimate IPL depths"
    try:
        depths = np.asarray(recording.estimate_ipl_depths(plot=False), dtype=float)
    except Exception as exc:
        return None, f"Estimation failed: {exc}"
    return depths, (
        f"Estimated IPL depths for {depths.size} ROIs: "
        f"{np.nanmin(depths):.0f}% to {np.nanmax(depths):.0f}%"
    )
