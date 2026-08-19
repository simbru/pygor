"""Per-ROI metrics the population panel can plot.

Each metric reduces a recording to one number per ROI. Where the trace
dock picks one array for one ROI, this picks one scalar for every ROI, so
a whole recording can be looked at before drilling into a cell.

Metrics differ from trace sources in three ways that shape this module:
they are computed rather than looked up, their cost ranges from free to a
thousand permutations, and which ones exist depends on the analysis class.
So a spec carries a cheap availability check, a compute callable and an
``expensive`` flag, and results are cached per recording.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    """One per-ROI metric.

    Attributes
    ----------
    key : str
        Stable identifier, used for caching.
    label : str
        Text shown in the dropdown.
    compute : callable
        ``compute(recording) -> array`` of one value per ROI.
    applies : callable
        Cheap test for whether the recording can provide this metric. Must
        not do the work itself.
    expensive : bool
        If True the panel waits for an explicit request rather than
        computing on selection.
    description : str
        Shown as the dropdown tooltip.
    """

    key: str
    label: str
    compute: Callable
    applies: Callable = field(default=lambda rec: True)
    expensive: bool = False
    description: str = ""


def _has_traces(recording):
    traces = getattr(recording, "traces_znorm", None)
    return traces is not None and np.asarray(traces).ndim == 2


def _traces(recording):
    return np.asarray(recording.traces_znorm, dtype=float)


def _roi_areas(recording):
    """Pixel count per ROI, in ROI id order."""
    from pygor.gui.roi_bridge import roi_ids_in_order

    mask = np.asarray(recording.rois)
    return np.array([int((mask == roi_id).sum()) for roi_id in roi_ids_in_order(mask)])


def _method_applies(name):
    """Availability check for a metric backed by a method call."""
    return lambda rec: callable(getattr(rec, name, None))


def _from_method(name, **kwargs):
    """Compute a metric by calling a recording method."""
    return lambda rec: np.asarray(getattr(rec, name)(**kwargs), dtype=float)


# Ordered as they should appear in the dropdown. Anything whose `applies`
# returns False is simply left out for that recording.
METRICS = (
    MetricSpec(
        key="quality_index",
        label="Quality index",
        compute=lambda rec: np.asarray(rec.quality_indices, dtype=float),
        applies=lambda rec: getattr(rec, "quality_indices", None) is not None,
        description="Response quality criterion, as exported by IGOR",
    ),
    MetricSpec(
        key="trace_range",
        label="Trace range",
        compute=lambda rec: np.ptp(_traces(rec), axis=1),
        applies=_has_traces,
        description="Peak-to-peak of the z-normalised trace",
    ),
    MetricSpec(
        key="trace_sd",
        label="Trace SD",
        compute=lambda rec: _traces(rec).std(axis=1),
        applies=_has_traces,
        description="Standard deviation of the z-normalised trace",
    ),
    MetricSpec(
        key="trace_max",
        label="Trace max",
        compute=lambda rec: _traces(rec).max(axis=1),
        applies=_has_traces,
        description="Largest value of the z-normalised trace",
    ),
    MetricSpec(
        key="roi_area",
        label="ROI area (px)",
        compute=_roi_areas,
        applies=lambda rec: getattr(rec, "rois", None) is not None,
        description="Number of pixels in each ROI",
    ),
    MetricSpec(
        key="ipl_depth",
        label="IPL depth",
        compute=lambda rec: np.asarray(rec.ipl_depths, dtype=float),
        applies=lambda rec: getattr(rec, "ipl_depths", None) is not None,
        description="Estimated inner plexiform layer depth",
    ),
    # Analysis-class metrics. Their `applies` only checks that the method
    # exists, so nothing is computed while building the dropdown.
    MetricSpec(
        key="dsi",
        label="DSI",
        compute=_from_method("get_dsi"),
        applies=_method_applies("get_dsi"),
        description="Direction selectivity index",
    ),
    MetricSpec(
        key="osi",
        label="OSI",
        compute=_from_method("get_osi"),
        applies=_method_applies("get_osi"),
        description="Orientation selectivity index",
    ),
    MetricSpec(
        key="preferred_direction",
        label="Preferred direction",
        compute=_from_method("get_preferred_direction"),
        applies=_method_applies("get_preferred_direction"),
        description="Preferred direction in degrees",
    ),
    MetricSpec(
        key="dsi_pvalue",
        label="DSI p-value",
        compute=_from_method("get_dsi_pvalue"),
        applies=_method_applies("get_dsi_pvalue"),
        expensive=True,
        description="Permutation test p-value for direction selectivity",
    ),
    MetricSpec(
        key="contours_count",
        label="Contour count",
        compute=_from_method("get_contours_count"),
        applies=_method_applies("get_contours_count"),
        description="Number of receptive field contours",
    ),
)


def available_metrics(recording):
    """Return the metrics this recording can provide, in dropdown order."""
    if recording is None:
        return ()
    return tuple(spec for spec in METRICS if _safe_applies(spec, recording))


def _safe_applies(spec, recording):
    try:
        return bool(spec.applies(recording))
    except Exception:
        return False


def metric_by_key(key):
    """Look up a spec by key, or None."""
    for spec in METRICS:
        if spec.key == key:
            return spec
    return None


def compute_metric(spec, recording, n_rois=None):
    """Compute a metric, returning None if it does not come back usable.

    A metric that raises, or that does not return one finite-shaped value
    per ROI, is dropped rather than allowed to misalign the panel against
    the ROI ids.
    """
    try:
        values = spec.compute(recording)
    except Exception as exc:
        warnings.warn(f"Metric {spec.key!r} failed: {exc}", stacklevel=2)
        return None

    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim != 1:
        warnings.warn(
            f"Metric {spec.key!r} returned shape {values.shape}, expected one "
            "value per ROI",
            stacklevel=2,
        )
        return None

    if n_rois is not None and values.size != n_rois:
        warnings.warn(
            f"Metric {spec.key!r} returned {values.size} values for "
            f"{n_rois} ROIs",
            stacklevel=2,
        )
        return None
    return values
