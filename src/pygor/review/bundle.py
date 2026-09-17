"""One field of view and the recordings that share its ROIs.

A cell is recorded under several stimuli. ROIs are segmented once, on a master
recording, and transferred to the others, so the same cell has a row in each --
and a judgement about it ("the ROIs landed off the cells", "this receptive field
is noise") is usually a judgement about one of those recordings, or about how
well they line up, rather than about the cell in the abstract. This module is
what lets a reviewer hold all of them at once.

Two tiers, because the cost difference is three orders of magnitude:

* the light tier reads projections, masks and provenance straight out of HDF5 in
  about a millisecond, and drives everything navigational -- the field-of-view
  list, the alignment panels, the metrics;
* :meth:`FovBundle.full` loads a real pygor object, which costs seconds and about
  a gigabyte, and is only worth paying once a reviewer has chosen a field of view
  to walk cell by cell.

Grouping is driven by callables supplied by the dataset, not by rules baked in
here: which stem is the noise recording and which the direction one, and how a
condition tag is stripped out of a name, are properties of an experiment, not of
pygor.
"""

from __future__ import annotations

import dataclasses
import gc
import pathlib
import warnings
from collections import OrderedDict

import numpy as np

import pygor.load
from pygor.review.index import RecordingRef, read_arrays

# Measured: a loaded recording sits at roughly 2.3x its file size in memory.
# Rounded up, because the consequence of underestimating is swapping.
MEMORY_PER_BYTE_ON_DISK = 2.5

DEFAULT_CACHE_BYTES = 6_000_000_000


class MissingPartner(LookupError):
    """A recording the field of view should have, but which is not on disk."""


def master_roi_index(roi_origin, num_rois) -> np.ndarray:
    """Row index -> the master's ROI index, for a recording that inherited ROIs.

    ``transfer_rois_from`` drops ROIs that fall outside the frame once the shift
    is applied and renumbers what survives from zero, so row j of a recording
    that lost cells is not master ROI j. Reading the mapping back out of the
    provenance is what keeps a positional join from being silently wrong.

    A self-segmented recording defines its own indexing and gets ``arange``.

    Mirrors ``analyses._common.master_roi_index`` but takes the provenance dict
    rather than a loaded object, so a work queue can use it without paying for a
    full load.

    Raises
    ------
    RuntimeError
        When the provenance and the object disagree about how many ROIs
        survived. Neither mapping is trustworthy then, and saying so is better
        than guessing.
    """
    origin = roi_origin or {}
    if origin.get("method") != "transferred":
        return np.arange(num_rois)
    expected = [int(x) for x in origin.get("expected_roi_ids", [])]
    lost = {int(x) for x in origin.get("lost_roi_ids", [])}
    survivors = [roi_id for roi_id in expected if roi_id not in lost]
    if len(survivors) != num_rois:
        raise RuntimeError(
            f"roi_origin lists {len(survivors)} surviving ROIs but the "
            f"recording has {num_rois}"
        )
    return np.array([abs(roi_id) - 1 for roi_id in survivors], dtype=int)


@dataclasses.dataclass(frozen=True)
class AlignmentInfo:
    """What the ROI transfer onto one recording did, and whether to trust it."""

    role: str
    method: str
    source: str = ""
    shift: tuple = ()
    error: float = float("nan")
    correlation: float = float("nan")
    n_rois: int = 0
    n_expected: int = 0
    lost_roi_ids: tuple = ()
    usable: bool = True
    problem: str = ""

    @property
    def transferred(self) -> bool:
        return self.method == "transferred"

    @property
    def n_lost(self) -> int:
        return len(self.lost_roi_ids)


def alignment_of(ref: RecordingRef, role="") -> AlignmentInfo:
    """Read one recording's transfer provenance, without loading it."""
    origin = ref.roi_origin or {}
    method = origin.get("method") or "unknown"
    shift = origin.get("shift") or ()
    problem = ""
    usable = True
    try:
        master_roi_index(origin, ref.num_rois)
    except RuntimeError as error:
        usable = False
        problem = str(error)
    return AlignmentInfo(
        role=role or "",
        method=method,
        source=str(origin.get("source", "")),
        shift=tuple(shift) if isinstance(shift, (list, tuple)) else (shift,),
        error=float(origin.get("error", float("nan"))),
        correlation=ref.transfer_correlation,
        n_rois=ref.num_rois,
        n_expected=len(origin.get("expected_roi_ids", [])),
        lost_roi_ids=tuple(ref.lost_roi_ids),
        usable=usable,
        problem=problem,
    )


class BundleCache:
    """LRU of loaded recordings, bounded by estimated memory rather than count.

    Counting objects would be the wrong bound: these files range from 58 MB to
    1.3 GB, so "three recordings" is anywhere between 150 MB and 3 GB of
    resident memory.
    """

    def __init__(self, max_bytes=DEFAULT_CACHE_BYTES):
        self.max_bytes = max_bytes
        self._items: OrderedDict[pathlib.Path, object] = OrderedDict()
        self._sizes: dict[pathlib.Path, int] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def nbytes(self) -> int:
        return sum(self._sizes.values())

    def estimate(self, ref: RecordingRef) -> int:
        return int(ref.size * MEMORY_PER_BYTE_ON_DISK)

    def get(self, ref: RecordingRef):
        key = ref.path
        if key in self._items:
            self.hits += 1
            self._items.move_to_end(key)
            return self._items[key]
        self.misses += 1
        obj = self._load(ref)
        self._items[key] = obj
        self._sizes[key] = self.estimate(ref)
        self._trim()
        return obj

    def _load(self, ref: RecordingRef):
        cls = getattr(pygor.load, ref.class_name, None)
        if cls is None:  # a class the installed pygor does not define
            cls = pygor.load.Core
        return cls.load_object(ref.path)

    def _trim(self) -> None:
        # Keep the most recent entry even when it alone exceeds the budget:
        # evicting what was just asked for would loop forever.
        while len(self._items) > 1 and self.nbytes > self.max_bytes:
            key, _ = self._items.popitem(last=False)
            self._sizes.pop(key, None)
            self.evictions += 1
            gc.collect()

    def loaded(self, ref: RecordingRef) -> bool:
        """Whether this recording is in memory, without putting it there."""
        return ref.path in self._items

    def discard(self, ref: RecordingRef) -> None:
        self._items.pop(ref.path, None)
        self._sizes.pop(ref.path, None)
        gc.collect()

    def clear(self) -> None:
        self._items.clear()
        self._sizes.clear()
        gc.collect()

    def stats(self) -> dict:
        return {
            "items": len(self._items),
            "bytes": self.nbytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


@dataclasses.dataclass
class FovBundle:
    """The recordings of one field of view, under one condition."""

    fov_uid: str
    condition: str
    session: str
    prefix: str
    refs: dict[str, RecordingRef]
    master_role: str
    expected: dict[str, bool] = dataclasses.field(default_factory=dict)
    cache: BundleCache | None = None
    _cells = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = BundleCache()

    # -- cheap ------------------------------------------------------------

    @property
    def roles(self) -> tuple:
        return tuple(self.refs)

    @property
    def master(self) -> RecordingRef:
        return self.refs[self.master_role]

    @property
    def n_cells(self) -> int:
        """ROI count of the master, which defines the field of view's numbering."""
        return self.master.num_rois

    @property
    def strf_role(self) -> str:
        """The recording that actually holds receptive fields.

        Not the master: on a paired field of view the master is the direction
        recording, which has no STRFs at all. A panel asking the master for a
        receptive field gets an AttributeError.
        """
        for role, ref in self.refs.items():
            if ref.num_strfs:
                return role
        return "swn" if "swn" in self.refs else self.master_role

    def peek(self, role) -> RecordingRef:
        try:
            return self.refs[role]
        except KeyError as error:
            raise MissingPartner(f"{self.fov_uid} has no {role} recording") from error

    def missing(self) -> dict[str, str]:
        """Roles the field of view should have had, and why they are absent."""
        out = {}
        for role, expected in self.expected.items():
            if expected and role not in self.refs:
                out[role] = "expected from the CSV but no processed file on disk"
        for role, ref in self.refs.items():
            if ref.status not in ("ok", "unknown"):
                out[role] = f"status={ref.status}: {ref.status_reason}".rstrip(": ")
        return out

    def alignment(self, role) -> AlignmentInfo:
        return alignment_of(self.peek(role), role=role)

    def alignments(self) -> dict[str, AlignmentInfo]:
        return {role: alignment_of(ref, role=role) for role, ref in self.refs.items()}

    def roi_map(self, role) -> np.ndarray:
        """Row index -> master ROI index for one recording."""
        ref = self.peek(role)
        return master_roi_index(ref.roi_origin, ref.num_rois)

    def master_to_row(self, role, master_index) -> int | None:
        """Where master ROI ``master_index`` sits in ``role``'s own arrays.

        ``None`` when that cell was lost in the transfer, which is the case a
        positional join would otherwise get wrong.
        """
        rows = np.flatnonzero(self.roi_map(role) == master_index)
        return int(rows[0]) if rows.size else None

    def projection(self, role) -> np.ndarray:
        """The anatomy image, read without loading the recording."""
        arrays = read_arrays(self.peek(role).path, keys=("average_stack",))
        if "average_stack" not in arrays:
            raise KeyError(f"{role} recording has no average_stack")
        return arrays["average_stack"]

    def roi_mask(self, role) -> np.ndarray:
        """The ROI mask in IGOR convention (background 1, ROIs -1, -2, ...)."""
        arrays = read_arrays(self.peek(role).path, keys=("rois",))
        if "rois" not in arrays:
            raise KeyError(f"{role} recording has no rois")
        return arrays["rois"]

    def light(self, role) -> dict[str, np.ndarray]:
        """Everything a field-of-view panel needs, in one read."""
        return read_arrays(
            self.peek(role).path,
            keys=("rois", "average_stack", "correlation_projection"),
        )

    # -- expensive --------------------------------------------------------

    def full(self, role):
        """Load the real pygor object. Seconds, and about a gigabyte.

        Only worth paying once a reviewer is walking this field of view cell by
        cell; every navigational panel should use the light tier instead.
        """
        return self.cache.get(self.peek(role))

    #: Derived quantities that compute for every cell on first touch and cache.
    #: Measured on a 167-cell recording: 1.8 s and 4.7 s respectively, after
    #: which a receptive-field panel renders in about 20 ms instead of 5 s.
    WARM_CALLS = (
        ("collapse_times_chroma", {"roi": 0}),
        ("get_timecourses", {}),
    )

    def warm(self, role="swn") -> None:
        """Pay the per-recording derived costs once, before walking its cells.

        Both caches are whole-recording: touching one cell computes them for all
        of them. Priming up front turns the first cell from the slowest into the
        same speed as the rest, which matters because the first cell is the one a
        reviewer waits on before deciding whether to stay in this field of view.
        """
        recording = self.full(role)
        for name, kwargs in self.WARM_CALLS:
            call = getattr(recording, name, None)
            if call is None:
                continue
            try:
                call(**kwargs)
            except Exception as error:  # no usable STRFs is itself a finding
                warnings.warn(
                    f"could not warm {self.fov_uid} {role} via {name}: {error}",
                    stacklevel=2,
                )

    def release(self) -> None:
        for ref in self.refs.values():
            self.cache.discard(ref)

    # -- CSV rows ---------------------------------------------------------

    def attach_cells(self, frame) -> None:
        """Hold this field of view's rows from the aggregate CSV.

        Filtered on condition as well as field of view by the caller: one
        ``fov_uid`` has meant more than one field of view in the past, and the
        rows of the other one are a different set of cells.
        """
        self._cells = frame

    @property
    def cells(self):
        return self._cells

    def cell_row(self, master_index):
        """The CSV row for one cell, or None when it has none."""
        if self._cells is None or self._cells.empty:
            return None
        match = self._cells[self._cells.roi_id == master_index]
        return None if match.empty else match.iloc[0]


def group_refs(refs, *, prefix_of, classify, fov_uid_of, pick=None, expected=None):
    """Group scanned recordings into bundles, one per field of view.

    The dataset supplies the rules: ``prefix_of(stem)`` pulls out the
    block/plane identity, ``classify(stem)`` says which role a recording plays,
    and ``fov_uid_of(condition, session, prefix)`` mints the id. Keying includes
    the condition because a session can hold the same plane recorded under two
    of them, and treating those as one field of view merges two different sets
    of cells.

    ``pick`` chooses between several candidates for a role, as when a field of
    view has repeat recordings of the same stimulus.
    """
    if pick is None:
        pick = lambda candidates: sorted(candidates, key=lambda r: len(r.stem))[0]

    grouped: dict[tuple, dict[str, list]] = {}
    for ref in refs:
        prefix = prefix_of(ref.stem)
        if prefix is None:
            continue
        role = classify(ref.stem)
        if role is None:
            continue
        key = (ref.condition, ref.session, prefix)
        grouped.setdefault(key, {}).setdefault(role, []).append(ref)

    bundles = []
    for (condition, session, prefix), by_role in sorted(grouped.items()):
        chosen = {role: pick(cands) for role, cands in by_role.items() if cands}
        if not chosen:
            continue
        # ROIs come from the direction recording when there is one, so a cell
        # keeps one identity across stimuli; otherwise the noise recording
        # segmented itself and defines its own.
        master_role = "osds" if "osds" in chosen else next(iter(chosen))
        if "osds" not in chosen and "swn" in chosen:
            master_role = "swn"
        bundles.append(
            FovBundle(
                fov_uid=fov_uid_of(condition, session, prefix),
                condition=condition,
                session=session,
                prefix=prefix,
                refs=chosen,
                master_role=master_role,
                expected=dict(expected or {}),
            )
        )
    return bundles
