"""What is under the cursor: which pixel, and what that pixel's signal does.

The terminal reports the pointer in character cells, so the finest the probe
can ever be is one cell -- roughly two pixels on a 64-row scan in a 30-row
pane. That is why the pixel fallback averages a small box rather than reading a
single pixel: a box the size of the thing you can actually point at.

Nothing here draws. It answers "which pixel" and "what trace", and the screen
decides how to show it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Half the side of the box averaged for a pixel trace, in pixels.
BOX = 1


@dataclass(frozen=True)
class Trace:
    """A probed signal and what it is, ready to be plotted."""

    label: str
    values: object  # np.ndarray
    seconds: float | None
    source: str  # "roi" or "pixel"


def pixel_at(x: int, y: int, cell_width: int, cell_height: int, shape,
             flipped: bool = True) -> tuple[int, int] | None:
    """The image pixel under a cell offset into a widget of that size.

    ``flipped`` because every view is drawn with ``origin="lower"``, matching
    pygor's own plots: the top row of cells is the last row of the array.
    """
    rows, cols = shape
    if cell_width <= 0 or cell_height <= 0:
        return None
    if not (0 <= x < cell_width and 0 <= y < cell_height):
        return None
    col = int(x / cell_width * cols)
    row = int(y / cell_height * rows)
    if flipped:
        row = rows - 1 - row
    return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)


def roi_ids(rois):
    """The ROI ids present, in the order :func:`extract_traces` writes rows.

    Deriving the row from the id arithmetically (``abs(id) - 1``) is wrong the
    moment a mask has a gap in it -- deleting ROI -3 leaves -1, -2, -4 over
    three rows -- and a trace attributed to the wrong cell is worse than none.
    """
    import numpy as np

    if rois is None:
        return []
    ids = np.unique(rois)
    return sorted(ids[ids < 0].tolist(), reverse=True)


def _roi_trace(recording, roi_id):
    ids = roi_ids(getattr(recording, "rois", None))
    if roi_id not in ids:
        return None
    index = ids.index(roi_id)
    for attribute, name in (("traces_znorm", "znorm"), ("traces_raw", "raw")):
        traces = getattr(recording, attribute, None)
        if traces is None or index >= len(traces):
            continue
        return traces[index], name
    return None


def at_pixel(recording, row: int, col: int) -> Trace:
    """The trace to show for a probe at ``(row, col)``.

    An ROI's own extracted trace where there is one, because that is the
    signal the analysis uses and it has the averaging already done. Off an ROI,
    or on a recording nobody has segmented, the raw neighbourhood -- which is
    also what tells you whether an artifact is in the data or in the
    segmentation.
    """
    import numpy as np

    hz = getattr(recording, "frame_hz", None)
    rois = getattr(recording, "rois", None)

    if rois is not None:
        roi_id = int(rois[row, col])
        if roi_id < 0:
            found = _roi_trace(recording, roi_id)
            if found is not None:
                values, name = found
                n_px = int((rois == roi_id).sum())
                return Trace(
                    label=f"ROI {roi_id}  ·  {name}  ·  {n_px} px",
                    values=np.asarray(values),
                    seconds=len(values) / hz if hz else None,
                    source="roi",
                )

    images = getattr(recording, "images", None)
    if images is None:
        return Trace("no images in this recording", np.zeros(0), None, "pixel")
    rows, cols = images.shape[1], images.shape[2]
    r0, r1 = max(row - BOX, 0), min(row + BOX + 1, rows)
    c0, c1 = max(col - BOX, 0), min(col + BOX + 1, cols)
    values = images[:, r0:r1, c0:c1].mean(axis=(1, 2), dtype=np.float64)
    side = f"{r1 - r0}x{c1 - c0}"
    return Trace(
        label=f"pixel ({row}, {col})  ·  raw  ·  {side} mean",
        values=values,
        seconds=len(values) / hz if hz else None,
        source="pixel",
    )


# The narrowest a zoom may get, in samples. Below this the y axis rescales to
# the noise of a handful of frames, which reads as structure.
MIN_WINDOW = 8


def clamp_window(window, n: int) -> tuple[float, float]:
    """``window`` as fractions of the trace, kept inside it and at least
    :data:`MIN_WINDOW` samples wide.

    Fractions rather than samples so the window survives moving the probe: a
    pixel trace and an ROI trace from the same recording need not be the same
    length, but the same stretch of the recording is the same fraction of both.
    """
    lo, hi = window
    least = min(1.0, MIN_WINDOW / n) if n else 1.0
    width = min(max(hi - lo, least), 1.0)
    lo = min(max(lo, 0.0), 1.0 - width)
    return lo, lo + width


def zoom(window, factor: float, anchor: float, n: int) -> tuple[float, float]:
    """Scale ``window`` by ``factor`` about ``anchor``, a fraction across it.

    ``factor`` below one zooms in. The point under ``anchor`` stays where it is
    on screen, so scrolling over a feature closes in on that feature.
    """
    lo, hi = window
    at = lo + anchor * (hi - lo)
    width = (hi - lo) * factor
    return clamp_window((at - anchor * width, at - anchor * width + width), n)


def pan(window, by: float, n: int) -> tuple[float, float]:
    """Slide ``window`` by ``by`` of its own width, stopping at either end."""
    lo, hi = window
    step = (hi - lo) * by
    return clamp_window((lo + step, hi + step), n)


def windowed(trace: Trace, window) -> tuple[Trace, float, float]:
    """The part of ``trace`` inside ``window``, and where it starts and ends.

    Start and end are in seconds when the recording has a frame rate and in
    frames when it does not, the same unit the axis will be labelled in.
    """
    from dataclasses import replace

    n = len(trace.values)
    lo, hi = clamp_window(window, n)
    i0, i1 = int(lo * n), max(int(round(hi * n)), int(lo * n) + 1)
    per_sample = trace.seconds / n if trace.seconds and n else 1.0
    part = replace(trace, values=trace.values[i0:i1],
                   seconds=(i1 - i0) * per_sample if trace.seconds else None)
    return part, i0 * per_sample, i1 * per_sample


def trigger_summary(times) -> str:
    """Triggers in one line: how many, how fast, and whether any are missing.

    At 5 Hz over an hour there are fifteen thousand of them and no drawing can
    show them one at a time, so this is the part that carries the information.
    A dropped trigger is an interval at twice the usual one, which the median
    and the gap count catch and a picture of a solid bar does not.
    """
    import numpy as np

    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return "no triggers"
    if times.size == 1:
        return f"1 trigger at {times[0]:.2f}s"

    intervals = np.diff(np.sort(times))
    median = float(np.median(intervals))
    spread = float(np.median(np.abs(intervals - median)))
    parts = [f"{times.size} trig", f"{median:.3f}s ± {spread:.3f}"]

    # Half a period either way: anything outside that is a trigger the system
    # did not send or one it sent twice, not jitter.
    gaps = int((intervals > median * 1.5).sum())
    extra = int((intervals < median * 0.5).sum())
    if gaps:
        parts.append(f"{gaps} gap{'s' if gaps != 1 else ''} (max {intervals.max():.3f}s)")
    if extra:
        parts.append(f"{extra} short (min {intervals.min():.3f}s)")
    if not gaps and not extra:
        parts.append("no gaps")
    return "  ·  ".join(parts)


def triggers_of(recording):
    """A recording's trigger times in seconds, or None if it has none."""
    times = getattr(recording, "triggertimes", None)
    if times is None or len(times) == 0:
        return None
    return times


def display_range(images, low: float = 1.0, high: float = 99.5):
    """Percentile limits for the stack, from a subsample of its frames.

    Computed once over the whole recording rather than per frame: scrubbing
    through frames each scaled to its own range makes the tissue pulse, which
    reads as a signal that is not there. The subsample is because the
    percentile of a 5000-frame stack is not worth the second it takes.
    """
    import numpy as np

    step = max(1, len(images) // 40)
    sample = np.asarray(images[::step], dtype=np.float32)
    lo, hi = np.percentile(sample, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)
