"""Hand edits to an ROI mask: a disk, a polygon, a deletion.

Pure array operations, no drawing and no widgets, so the rules are testable
without a terminal. The mask follows the IGOR convention the rest of pygor
uses: ROIs are negative integers, background is non-negative (1 in files that
came through IGOR, 0 in some that did not -- whatever is there is kept).

Where a new shape overlaps an existing ROI, the existing ROI keeps its pixels
and the new one takes only what is unclaimed. Redrawing a cell therefore
means deleting it first, which is deliberate: a stroke that lands a pixel too
far should not quietly shave the neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Added:
    """What :func:`add` did: the new mask and how much of the shape it got."""

    rois: object  # np.ndarray
    roi_id: int | None
    taken: int
    blocked: int


def background_value(rois) -> int:
    """The value the mask already uses for "no ROI"."""
    import numpy as np

    free = rois[rois >= 0]
    return int(free.flat[0]) if free.size else 1


def disk(shape, centre, radius: float):
    """Pixels within ``radius`` of ``centre`` (row, col), as a boolean mask.

    Radius 0 is the single pixel, which is a legitimate ROI on a sparse
    label where a terminal is two pixels across.
    """
    import numpy as np

    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row, col = centre
    return (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2 + 1e-9


def polygon(shape, vertices):
    """The pixels a closed polygon through ``vertices`` (row, col) covers.

    Interior and outline together: ``skimage.draw.polygon`` alone drops much
    of the boundary, so a lasso drawn tightly around a small cell would come
    back smaller than the line the reader drew.
    """
    import numpy as np
    from skimage.draw import line
    from skimage.draw import polygon as fill

    region = np.zeros(shape, dtype=bool)
    if not vertices:
        return region
    if len(vertices) >= 3:
        rs = np.array([v[0] for v in vertices], dtype=float)
        cs = np.array([v[1] for v in vertices], dtype=float)
        region[fill(rs, cs, shape=shape)] = True
    # The outline segment by segment, closing back to the start.
    closed = [*vertices, vertices[0]] if len(vertices) > 2 else list(vertices)
    for (r0, c0), (r1, c1) in zip(closed, closed[1:] or closed):
        rr, cc = line(int(r0), int(c0), int(r1), int(c1))
        keep = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        region[rr[keep], cc[keep]] = True
    return region


def add(rois, region) -> Added:
    """``rois`` with ``region`` added as a new ROI, on unclaimed pixels only.

    The new id is one below the lowest in use, so no existing ROI is
    renumbered. Returns ``roi_id=None`` and the mask unchanged when every
    pixel of the shape is already taken.
    """
    free = region & (rois >= 0)
    taken = int(free.sum())
    blocked = int((region & (rois < 0)).sum())
    if not taken:
        return Added(rois, None, 0, blocked)
    in_use = rois[rois < 0]
    roi_id = int(in_use.min()) - 1 if in_use.size else -1
    out = rois.copy()
    out[free] = roi_id
    return Added(out, roi_id, taken, blocked)


def remove(rois, roi_id: int):
    """``rois`` without ``roi_id``; its pixels go back to background."""
    out = rois.copy()
    out[out == roi_id] = background_value(rois)
    return out


def compact(rois):
    """Renumber to -1..-n without gaps, keeping the existing order.

    Deleting leaves holes in the ids, and while the inspector copes (see
    ``probe.roi_ids``), plenty of pygor still maps id to row as ``abs(id)-1``.
    So a mask is compacted before it leaves the editor, never during: ids
    stay stable while the reader is working.
    """
    import numpy as np

    out = rois.copy()
    ids = sorted(np.unique(rois[rois < 0]).tolist(), reverse=True)
    for new, old in enumerate(ids, start=1):
        out[rois == old] = -new
    return out


def count(rois) -> int:
    import numpy as np

    return int(np.unique(rois[rois < 0]).size)
