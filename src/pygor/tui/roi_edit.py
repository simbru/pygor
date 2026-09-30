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
    # The outline segment by segment, closing back to the start. A single
    # vertex is a segment from itself to itself: one pixel.
    closed = [*vertices, vertices[0]] if len(vertices) > 2 else list(vertices)
    for (r0, c0), (r1, c1) in zip(closed, closed[1:] or closed, strict=False):
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


@dataclass(frozen=True)
class Reconciled:
    """An edited mask laid out against the one it was edited from."""

    rois: object  # np.ndarray, ids -1..-n
    kept: list  # row indices into the old mask's ROIs, in order
    added: int  # ROIs after the kept ones that are new


def reconcile(old, edited) -> Reconciled:
    """Lay ``edited`` out so every per-ROI array of ``old`` can follow it.

    An old ROI survives if exactly its pixels are one ROI in ``edited``;
    anything else -- a deleted cell, a new one, an old one reshaped -- is not a
    survivor. Survivors take rows 0..k-1 in their old order, so an array
    indexed like ``old`` follows by keeping rows ``kept``; everything else
    comes after, and has no row to inherit.

    Rows of ``old`` are positional (see ``probe.roi_ids``), so a mask with
    gaps in its ids is handled; the result never has gaps.
    """
    import numpy as np

    old_ids = sorted(np.unique(old[old < 0]).tolist(), reverse=True) if old is not None else []
    edited_ids = sorted(np.unique(edited[edited < 0]).tolist(), reverse=True)

    survivor_of = {}  # edited id -> old row
    for row, old_id in enumerate(old_ids):
        pixels = old == old_id
        values = np.unique(edited[pixels])
        if (values.size == 1 and values[0] < 0
                and int((edited == values[0]).sum()) == int(pixels.sum())):
            survivor_of[int(values[0])] = row

    kept_ids = sorted(survivor_of, key=survivor_of.get)
    order = kept_ids + [i for i in edited_ids if i not in survivor_of]
    out = np.full(edited.shape, background_value(edited), dtype=edited.dtype)
    for n, roi_id in enumerate(order):
        out[edited == roi_id] = -(n + 1)
    return Reconciled(out, [survivor_of[i] for i in kept_ids], len(order) - len(kept_ids))


def count(rois) -> int:
    import numpy as np

    return int(np.unique(rois[rois < 0]).size)
