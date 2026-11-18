"""Keep every per-ROI array on a recording aligned with its ROI mask.

Row ``n`` of a per-ROI array belongs to the ROI with mask id ``-(n+1)``.
An edit -- deleting, merging, redrawing, re-segmenting -- changes which ROIs
exist, and unless every per-ROI array is reindexed with the mask, together,
a value ends up describing a different cell: an IPL depth or an RF one row
off its trace, with nothing to show for it.

The rule applied here: **an ROI keeps its data only if exactly its pixels
are one ROI after the edit.** Matching is by pixels, not by id, so
renumbering is not a change. Survivors take rows ``0..k-1`` in their old
order; every other ROI -- new, merged, reshaped -- follows, in id order, with
NaN wherever a value would need recomputing. NaN means "not known", never
"a neighbour's".

Every per-ROI attribute pygor stores is listed below. Anything derived from
them that is cheap to rebuild is dropped and rebuilt on next use; anything
that is not is reindexed. :func:`check_alignment` verifies the result.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: Stored, one row per ROI, not recomputable from the mask: reindexed.
#: A wrong length here means the recording was already out of step, and the
#: edit is refused rather than moving the error somewhere harder to see.
ROI_ROWS = ("ipl_depths", "traces_deconvolved")

#: One row per ROI, re-extracted or recomputed after every edit.
RECOMPUTED_ROWS = ("traces_raw", "traces_znorm", "snippets", "averages")

#: One row per ROI, recomputed only when averages are; reindexed otherwise.
#: Derived, so a wrong length drops it with a warning instead of refusing.
DERIVED_ROI_ROWS = ("quality_indices",)

#: Stored, one row per ROI and colour (flat index ``roi * n_colours + c``).
STRF_ROWS = ("strfs",)

#: Derived per STRF, too slow to drop (a bootstrap each): reindexed.
DERIVED_STRF_ROWS = ("_pval_time", "_pval_space")

#: Derived per STRF and rebuilt on demand when missing: dropped.
DROPPED_STRF_STATE = (
    "_STRF__contours_centroids",
    "_STRF__centres_by_pol",
    "_STRF__strf_masks",
    "_spatial_centered_collapse",
)

#: Per-ROI results whose ``None`` means "not computed yet": reset.
RESET_TO_NONE = ("_OSDS__tuning_metrics_results", "_OSDS__von_mises_results")


# -- masks ---------------------------------------------------------------------


def background_value(rois) -> int:
    """The value the mask already uses for "no ROI": 1 from IGOR, 0 from some."""
    free = rois[rois >= 0]
    return int(free.flat[0]) if free.size else 1


def roi_ids(rois):
    """ROI ids in row order: -1, -2, ... skipping any gaps."""
    if rois is None:
        return []
    return sorted(np.unique(rois[rois < 0]).tolist(), reverse=True)


def count(rois) -> int:
    return len(roi_ids(rois))


def compact(rois):
    """Renumber to -1..-n without gaps, keeping the existing order."""
    out = rois.copy()
    for new, old in enumerate(roi_ids(rois), start=1):
        out[rois == old] = -new
    return out


@dataclass(frozen=True)
class Reconciled:
    """An edited mask laid out against the one it was edited from."""

    rois: object  # np.ndarray, ids -1..-n
    kept: list  # row indices into the old mask's ROIs, in order
    added: int  # ROIs after the kept ones that are new

    @property
    def n_rois(self) -> int:
        return len(self.kept) + self.added

    @property
    def unknown_rows(self) -> list:
        """Rows with nothing to inherit: new, merged or reshaped ROIs."""
        return list(range(len(self.kept), self.n_rois))


def reconcile(old, edited) -> Reconciled:
    """Lay ``edited`` out so every per-ROI array of ``old`` can follow it.

    An old ROI survives if exactly its pixels are one ROI in ``edited``;
    anything else is not a survivor. Survivors take rows 0..k-1 in their old
    order, so an array indexed like ``old`` follows by keeping rows ``kept``;
    everything else comes after and has no row to inherit.

    Rows of ``old`` are positional, so a mask with gaps in its ids is
    handled; the result never has gaps.
    """
    old_ids = roi_ids(old)
    edited_ids = roi_ids(edited)

    survivor_of = {}  # edited id -> old row
    for row, old_id in enumerate(old_ids):
        pixels = old == old_id
        values = np.unique(edited[pixels])
        if (
            values.size == 1
            and values[0] < 0
            and int((edited == values[0]).sum()) == int(pixels.sum())
        ):
            survivor_of[int(values[0])] = row

    kept_ids = sorted(survivor_of, key=survivor_of.get)
    order = kept_ids + [i for i in edited_ids if i not in survivor_of]
    out = np.full(edited.shape, background_value(edited), dtype=edited.dtype)
    for n, roi_id in enumerate(order):
        out[edited == roi_id] = -(n + 1)
    return Reconciled(out, [survivor_of[i] for i in kept_ids], len(order) - len(kept_ids))


# -- arrays ----------------------------------------------------------------------


def _present(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 1 and value.size > 0


def _reindexed(array, rows, n_out):
    """``array[rows]`` followed by NaN rows up to ``n_out``.

    A masked array stays masked, with its mask reindexed alongside: STRFs
    come masked from both the H5 loader and ``calculate_strf``, and copying
    only ``.data`` would unmask the border as if it were measured.
    """
    rows = np.asarray(rows, dtype=int)
    if isinstance(array, np.ma.MaskedArray):
        data = _reindexed(np.ma.getdata(array), rows, n_out)
        mask = np.zeros(data.shape, dtype=bool)
        mask[: rows.size] = np.ma.getmaskarray(array)[rows]
        return np.ma.MaskedArray(data, mask=mask, fill_value=array.fill_value)
    if array.dtype == object:
        out = np.empty((n_out, *array.shape[1:]), dtype=object)
        out[...] = np.nan
    else:
        dtype = np.result_type(array.dtype, np.float32)
        out = np.full((n_out, *array.shape[1:]), np.nan, dtype=dtype)
    out[: rows.size] = array[rows]
    return out


def _strf_rows(rows, n_colours):
    rows = np.asarray(rows, dtype=int)
    return (rows[:, None] * n_colours + np.arange(n_colours)).ravel()


def strf_keys(n_rois, n_colours):
    """STRF names in flat order, as ``calculate_strf`` writes them."""
    return [f"STRF{c}_{roi}" for roi in range(n_rois) for c in range(n_colours)]


def n_colours_of(recording, n_rois=None):
    """Colours per ROI in ``strfs``, read off the array when unset."""
    n_colours = getattr(recording, "n_colours", None)
    strfs = getattr(recording, "strfs", None)
    if n_rois is None:
        n_rois = count(getattr(recording, "rois", None))
    if not n_colours and _present(strfs) and n_rois:
        n_colours = len(strfs) // n_rois
    return int(n_colours or 1)


def check_alignment(recording) -> list:
    """Every way a per-ROI array disagrees with the mask, as messages.

    An empty list means every stored per-ROI array has one row per ROI (or
    per ROI and colour, for STRFs), and ``num_rois`` matches the mask.
    """
    rois = getattr(recording, "rois", None)
    n = count(rois) if isinstance(rois, np.ndarray) else 0
    problems = []
    if int(getattr(recording, "num_rois", n) or 0) != n:
        problems.append(f"num_rois is {recording.num_rois}, mask has {n} ROIs")
    if isinstance(rois, np.ndarray) and n and roi_ids(rois) != list(range(-1, -n - 1, -1)):
        problems.append("mask ids have gaps")
    for name in ROI_ROWS + RECOMPUTED_ROWS + DERIVED_ROI_ROWS + ("roi_sizes",):
        value = getattr(recording, name, None)
        if _present(value) and len(value) != n:
            problems.append(f"{name} has {len(value)} rows for {n} ROIs")
    strfs = getattr(recording, "strfs", None)
    if _present(strfs):
        nc = n_colours_of(recording, n)
        for name in STRF_ROWS + DERIVED_STRF_ROWS:
            value = getattr(recording, name, None)
            if _present(value) and len(value) != n * nc:
                problems.append(f"{name} has {len(value)} rows for {n} ROIs x {nc} colours")
        keys = getattr(recording, "strf_keys", None)
        if keys is not None and len(keys) != n * nc:
            problems.append(f"strf_keys has {len(keys)} entries for {n * nc} STRFs")
    return problems


@dataclass
class Applied:
    """What :func:`apply_mask` did, for reporting back to whoever edited."""

    kept: int
    deleted: int
    added: int
    unknown_rows: list = field(default_factory=list)
    dropped: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.deleted or self.added)


def _renumber_rf_rejections(recording, kept):
    """Keep rejected RFs on the same ROIs through an edit.

    ``rf_rejected`` (pygor.strf.rejection) holds ROI indices, which an edit
    renumbers. A rejection follows its ROI to its new index; one on a deleted
    or redrawn ROI goes with it, and a new ROI starts unrejected.
    """
    from pygor.strf.rejection import ATTR, rejected

    if getattr(recording, ATTR, None) is None:
        return
    new_index = {int(old): new for new, old in enumerate(kept)}
    setattr(recording, ATTR, np.array(
        sorted(new_index[r] for r in rejected(recording).tolist() if r in new_index),
        dtype=int,
    ))


def apply_mask(recording, mask, refresh_traces=True) -> Applied:
    """Put an edited mask on ``recording`` and keep every per-ROI array aligned.

    ROIs whose pixels are unchanged keep everything they had, in their old
    order -- traces and averages included, so each one's trace, RF and depth
    still describe the same data. Every other ROI is appended with NaN STRFs,
    IPL depths and the like: not computed, rather than a neighbour's. With
    ``refresh_traces`` their traces are extracted (and snippets, averages and
    quality computed, if the recording had them); without, they stay NaN
    until the caller extracts. ``roi_sizes`` is recounted from the mask.

    Everything is checked before anything changes: a stored array that does
    not already have one row per ROI raises ``ValueError`` and the
    recording is left as it was.
    """
    old = getattr(recording, "rois", None)
    if not isinstance(old, np.ndarray):
        old = None
    mask = np.asarray(mask)
    layout = reconcile(old, mask)
    n_old = count(old)
    n_new = layout.n_rois
    kept = layout.kept

    # -- check, changing nothing ------------------------------------------------
    for name in ROI_ROWS:
        value = getattr(recording, name, None)
        if _present(value) and len(value) != n_old:
            raise ValueError(
                f"{name} has {len(value)} rows for {n_old} ROIs; this recording is "
                f"already misaligned, so {name} cannot be carried through an edit"
            )
    strfs = getattr(recording, "strfs", None)
    n_colours = 1
    if _present(strfs) and n_old:
        if len(strfs) % n_old:
            raise ValueError(
                f"{len(strfs)} STRFs do not divide among {n_old} ROIs; this recording's "
                f"STRFs are already misaligned, so they cannot be carried through an edit"
            )
        n_colours = len(strfs) // n_old
        stated = getattr(recording, "n_colours", None)
        if stated and int(stated) != n_colours:
            raise ValueError(
                f"{len(strfs)} STRFs for {n_old} ROIs is {n_colours} colours, but "
                f"n_colours is {stated}; the STRFs are misaligned"
            )

    had_averages = _present(getattr(recording, "averages", None))
    dropped = []

    # -- reindex -----------------------------------------------------------------
    for name in ROI_ROWS:
        value = getattr(recording, name, None)
        if _present(value):
            setattr(recording, name, _reindexed(value, kept, n_new))

    # Reindexed even when about to be recomputed, so that whatever is not
    # recomputed (snippets without averages, say) still lines up
    for name in RECOMPUTED_ROWS:
        value = getattr(recording, name, None)
        if _present(value) and len(value) == n_old:
            setattr(recording, name, _reindexed(value, kept, n_new))
        elif _present(value):
            # Out of step already (a transfer's NaN-padded rows): re-extracted below
            setattr(recording, name, None)

    for name in DERIVED_ROI_ROWS:
        value = getattr(recording, name, None)
        if not _present(value):
            continue
        if len(value) == n_old:
            setattr(recording, name, _reindexed(value, kept, n_new))
        else:
            setattr(recording, name, None)
            dropped.append(name)

    if _present(strfs) and n_old:
        flat = _strf_rows(kept, n_colours)
        recording.strfs = _reindexed(strfs, flat, n_new * n_colours).astype(
            strfs.dtype if np.issubdtype(strfs.dtype, np.floating) else np.float32,
            copy=False,
        )
        for name in DERIVED_STRF_ROWS:
            value = getattr(recording, name, None)
            if not _present(value):
                continue
            if len(value) == n_old * n_colours:
                setattr(recording, name, _reindexed(value, flat, n_new * n_colours))
            else:
                delattr(recording, name)
                dropped.append(name)
        recording.n_colours = n_colours
        recording.strf_keys = strf_keys(n_new, n_colours)
        recording.num_strfs = len(recording.strfs)
    for name in DROPPED_STRF_STATE:
        if name in vars(recording):
            delattr(recording, name)
    _renumber_rf_rejections(recording, kept)
    if hasattr(recording, "_invalidate_strf_caches"):
        recording._invalidate_strf_caches()
    for name in RESET_TO_NONE:
        if name in vars(recording):
            setattr(recording, name, None)

    # -- the mask itself -----------------------------------------------------------
    dtype = old.dtype if old is not None else np.int16
    recording.rois = layout.rois.astype(dtype)
    recording.num_rois = n_new
    previous = getattr(recording, "roi_origin", None)
    # A transfer's expected/lost ids name the old mask's labels, and
    # extraction drops rows by them; carried over, they would drop the wrong ones
    recording.roi_origin = {
        "method": "manual",
        "source": getattr(recording, "name", None),
        "edited_from": previous.get("method") if isinstance(previous, dict) else None,
    }
    labels = getattr(recording, "_Core__keyword_lables", None)
    if isinstance(labels, dict) and "ipl_depths" in labels:
        labels["ipl_depths"] = getattr(recording, "ipl_depths", None)
    # A pixel count, so recounted rather than carried
    recording.roi_sizes = np.array(
        [int((recording.rois == -(n + 1)).sum()) for n in range(n_new)], dtype=int
    )

    # -- traces ----------------------------------------------------------------
    # Without a refresh the new rows stay NaN until the caller extracts
    if n_new == 0:
        for name in RECOMPUTED_ROWS:
            setattr(recording, name, np.nan)
    elif refresh_traces:
        # Survivors keep the rows they had: a trace from IGOR, say, is what
        # their RF was computed from, and swapping in a re-extraction would
        # leave each survivor's trace and RF describing different data.
        # Fresh values only fill the rows that have nothing to keep.
        carried = {
            name: getattr(recording, name, None)
            for name in RECOMPUTED_ROWS + DERIVED_ROI_ROWS
        }
        recording.extract_traces_from_rois()
        if had_averages:
            recording.compute_snippets_and_averages()
        for name, before in carried.items():
            after = getattr(recording, name, None)
            if (
                _present(before)
                and _present(after)
                and len(before) == len(after) == n_new
                and before.shape[1:] == after.shape[1:]
            ):
                merged = np.array(after, dtype=np.result_type(after.dtype, before.dtype), copy=True)
                merged[: len(kept)] = before[: len(kept)]
                setattr(recording, name, merged)

    problems = check_alignment(recording)
    if problems:
        # A bug here, not in the data: everything was checked on the way in
        raise RuntimeError("ROI arrays out of step after an edit: " + "; ".join(problems))
    if dropped:
        warnings.warn(
            f"Dropped {', '.join(dropped)}: already misaligned before the edit",
            stacklevel=2,
        )
    return Applied(
        kept=len(kept),
        deleted=n_old - len(kept),
        added=layout.added,
        unknown_rows=layout.unknown_rows,
        dropped=dropped,
    )


def fill_missing_traces(recording) -> list:
    """Extract traces for the ROIs that have none (NaN rows), only.

    Every other row is left as it was, so a trace stays the one its RF was
    computed from. Returns the rows that were filled.
    """
    before = {name: getattr(recording, name, None) for name in ("traces_raw", "traces_znorm")}
    raw = before["traces_raw"]
    n = count(getattr(recording, "rois", None))
    if not _present(raw) or len(raw) != n:
        recording.extract_traces_from_rois()
        return list(range(n))
    missing = [int(i) for i in np.flatnonzero(np.isnan(np.asarray(raw, dtype=float)).all(axis=1))]
    if not missing:
        return []
    recording.extract_traces_from_rois()
    for name, old in before.items():
        new = getattr(recording, name, None)
        if _present(old) and _present(new) and old.shape == new.shape:
            keep = np.ones(n, dtype=bool)
            keep[missing] = False
            merged = np.array(new, copy=True)
            merged[keep] = old[keep]
            setattr(recording, name, merged)
    return missing


def rois_missing_strfs(recording) -> list:
    """ROI rows whose STRFs are all NaN: new or edited, not yet computed."""
    strfs = getattr(recording, "strfs", None)
    n = count(getattr(recording, "rois", None))
    if not _present(strfs) or not n:
        return []
    nc = n_colours_of(recording, n)
    per_roi = np.asarray(strfs, dtype=float).reshape(n, nc, -1)
    return [int(i) for i in np.flatnonzero(np.isnan(per_roi).all(axis=(1, 2)))]
