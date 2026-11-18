"""Per-ROI data follows its ROI's pixels through any mask edit.

Every per-ROI value in these tests is a *fingerprint* of the pixels it was
computed from, so after an edit each row can be checked against the ROI that
now sits at that index: a value one row off is caught, not just a wrong count.
"""

import os
import pathlib

import numpy as np
import pytest

from pygor.core import roi_alignment as ra

SHAPE = (24, 24)
N_COLOURS = 2


def fingerprint(mask, roi_id):
    """Sum of the flat pixel indices of one ROI: equal only for equal pixels
    (in practice, for these small grids)."""
    return float(np.flatnonzero(mask.ravel() == roi_id).sum())


def grid_mask(n_rois=9):
    """2x2 ROIs on a grid, ids -1..-n in reading order, background 1."""
    mask = np.ones(SHAPE, dtype=np.int16)
    for n in range(n_rois):
        r, c = divmod(n, 4)
        mask[2 + 5 * r : 4 + 5 * r, 2 + 5 * c : 4 + 5 * c] = -(n + 1)
    return mask


class FingerprintRecording:
    """Stand-in for an STRF recording whose per-ROI data encode their ROI."""

    name = "fingerprint"

    def __init__(self, mask):
        self.rois = mask
        self.num_rois = ra.count(mask)
        # A pixel's value is its flat index at every frame, so an ROI's mean
        # trace is its mean pixel index: also a function of its pixels alone
        self.images = np.broadcast_to(
            np.arange(mask.size, dtype=float).reshape(SHAPE), (5, *SHAPE)
        ).copy()
        self.extract_traces_from_rois()
        prints = self.fingerprints()
        self.ipl_depths = prints.copy()
        self.traces_deconvolved = np.repeat(prints[:, None], 5, axis=1)
        self.quality_indices = prints.copy()
        self.n_colours = N_COLOURS
        self.strfs = np.stack(
            [np.full((3, 2, 2), p * 10 + c, dtype=np.float32) for p in prints for c in range(N_COLOURS)]
        )
        self.strf_keys = ra.strf_keys(len(prints), N_COLOURS)
        self.num_strfs = len(self.strfs)
        self._pval_time = np.array([p * 10 + c for p in prints for c in range(N_COLOURS)])
        self._STRF__contours_centroids = object()
        self.roi_origin = {"method": "transfer", "lost_roi_ids": [-3], "expected_roi_ids": [-1]}
        self._Core__keyword_lables = {"ipl_depths": self.ipl_depths}
        self.averages = None

    def fingerprints(self):
        return np.array([fingerprint(self.rois, i) for i in ra.roi_ids(self.rois)])

    def known(self):
        """Fingerprints of the ROIs that hold data, as opposed to NaN."""
        prints = self.fingerprints()
        return prints[~np.isnan(self.ipl_depths)]

    def extract_traces_from_rois(self):
        rows = [self.images[:, self.rois == i].mean(axis=1) for i in ra.roi_ids(self.rois)]
        self.traces_raw = np.array(rows).reshape(len(rows), self.images.shape[0])
        self.traces_znorm = self.traces_raw.copy()

    def _invalidate_strf_caches(self):
        self.invalidated = True


def assert_follows_pixels(rec, before_prints):
    """Every row either holds its own ROI's fingerprint or is NaN (unknown),
    and it is unknown exactly when its pixels are not those of an old ROI
    that held data. ``before_prints`` are those old ROIs' fingerprints."""
    assert ra.check_alignment(rec) == []
    old = set(before_prints.tolist())
    for row, roi_id in enumerate(ra.roi_ids(rec.rois)):
        assert roi_id == -(row + 1), "ids are compact and in row order"
        own = fingerprint(rec.rois, roi_id)
        survived = own in old
        for value in (
            rec.ipl_depths[row],
            rec.quality_indices[row],
            rec.traces_deconvolved[row, 0],
            *(rec.strfs[row * N_COLOURS + c, 0, 0, 0] - c for c in range(N_COLOURS)),
            *(rec._pval_time[row * N_COLOURS + c] - c for c in range(N_COLOURS)),
        ):
            if survived:
                # strfs/pvals were encoded as print * 10 + colour
                assert value in (own, own * 10), f"row {row} holds another ROI's value"
            else:
                assert np.isnan(value), f"row {row} is a changed ROI but inherited a value"
        # Traces are re-extracted, so they are right for every ROI
        assert rec.traces_raw[row, 0] == pytest.approx(own / (rec.rois == roi_id).sum())
        assert rec.roi_sizes[row] == (rec.rois == roi_id).sum()


def test_deleting_shifts_everything_together():
    rec = FingerprintRecording(grid_mask())
    before = rec.fingerprints()
    mask = rec.rois.copy()
    mask[mask == -3] = 1
    applied = ra.apply_mask(rec, mask)

    assert (applied.kept, applied.deleted, applied.added) == (8, 1, 0)
    assert_follows_pixels(rec, before)
    # Row 2 is now what was ROI -4
    assert rec.ipl_depths[2] == before[3]


def test_merge_reshape_add_and_renumber_at_once():
    rec = FingerprintRecording(grid_mask())
    before = rec.fingerprints()
    mask = rec.rois.copy()
    mask[mask == -2] = -1  # merge 2 into 1
    mask[9, 2] = -5  # reshape: ROI -5 (rows 7-8, cols 2-3) grows a pixel down
    mask[20, 20] = -40  # new
    mask[mask == -9] = -90  # renumber only: same pixels, new id
    applied = ra.apply_mask(rec, mask)

    assert_follows_pixels(rec, before)
    assert applied.deleted == 3  # 1 and 2 merged away, 5 reshaped
    assert applied.added == 3  # the merged one, the reshaped one, the new one
    # Renumbered -9 survived with its data, at the end of the survivors
    assert before[8] in rec.ipl_depths.tolist()


@pytest.mark.parametrize("seed", range(40))
def test_random_edits_never_shift_data(seed):
    rng = np.random.default_rng(seed)
    rec = FingerprintRecording(grid_mask(int(rng.integers(3, 12))))
    for _ in range(3):  # edits on top of edits
        # ROIs added in an earlier round hold NaN, and still should
        before = rec.known()
        mask = rec.rois.copy()
        ids = ra.roi_ids(mask)
        for roi_id in ids:
            action = rng.choice(["keep", "keep", "delete", "merge", "grow", "relabel"])
            if action == "delete":
                mask[mask == roi_id] = 1
            elif action == "merge" and len(ids) > 1:
                other = int(rng.choice([i for i in ids if i != roi_id]))
                mask[mask == roi_id] = other
            elif action == "grow":
                rows, cols = np.nonzero(mask == roi_id)
                if rows.size:
                    r, c = rows.max() + 1, cols.max()
                    if r < SHAPE[0] and mask[r, c] == 1:
                        mask[r, c] = roi_id
            elif action == "relabel":
                mask[mask == roi_id] = roi_id - 100
        if rng.random() < 0.5:
            mask[22, int(rng.integers(0, 24))] = -77
        ra.apply_mask(rec, mask)
        assert_follows_pixels(rec, before)


def test_unchanged_mask_changes_nothing():
    rec = FingerprintRecording(grid_mask())
    depths, strfs = rec.ipl_depths.copy(), rec.strfs.copy()
    applied = ra.apply_mask(rec, rec.rois.copy())
    assert not applied.changed
    assert np.array_equal(rec.ipl_depths, depths)
    assert np.array_equal(rec.strfs, strfs)


def test_misaligned_recording_is_refused_untouched():
    rec = FingerprintRecording(grid_mask())
    rec.ipl_depths = rec.ipl_depths[:-1]
    before = rec.rois.copy()
    mask = before.copy()
    mask[mask == -1] = 1
    with pytest.raises(ValueError, match="ipl_depths .* misaligned"):
        ra.apply_mask(rec, mask)
    assert np.array_equal(rec.rois, before)
    assert len(rec.strfs) == 9 * N_COLOURS


def test_derived_state_is_dropped_and_provenance_cleared():
    rec = FingerprintRecording(grid_mask())
    mask = rec.rois.copy()
    mask[mask == -1] = 1
    ra.apply_mask(rec, mask)
    assert not hasattr(rec, "_STRF__contours_centroids")
    assert rec.invalidated
    assert "lost_roi_ids" not in rec.roi_origin, "stale ids would drop the wrong trace rows"
    assert rec._Core__keyword_lables["ipl_depths"] is rec.ipl_depths
    assert rec.strf_keys == ra.strf_keys(8, N_COLOURS)


def test_rois_missing_strfs_lists_the_nan_rows():
    rec = FingerprintRecording(grid_mask(4))
    mask = rec.rois.copy()
    mask[mask == -2] = -1
    ra.apply_mask(rec, mask)
    # -3, -4 survive (rows 0, 1); the merged ROI comes after them
    assert ra.rois_missing_strfs(rec) == [2]


def test_deleting_everything_leaves_empty_aligned_arrays():
    rec = FingerprintRecording(grid_mask(3))
    ra.apply_mask(rec, np.ones(SHAPE, dtype=np.int16))
    assert rec.num_rois == 0
    assert rec.strfs.shape[0] == 0
    assert ra.check_alignment(rec) == []


# -- a real recording ------------------------------------------------------------

_DEMO = pathlib.Path(
    os.environ.get(
        "PYGOR_TEST_H5",
        pathlib.Path(__file__).parents[3].joinpath("examples/strf_demo_data.h5"),
    )
)


@pytest.mark.skipif(not _DEMO.exists(), reason=f"no recording at {_DEMO}")
def test_real_strf_recording_keeps_rfs_depths_and_traces_with_their_pixels():
    import pygor.load

    rec = pygor.load.STRF(_DEMO)
    rec.extract_traces_from_rois()  # compare against pygor's own extraction
    ids = ra.roi_ids(rec.rois)
    nc = len(rec.strfs) // len(ids)
    by_pixels = {
        fingerprint(rec.rois, roi_id): (
            float(rec.ipl_depths[row]),
            rec.strfs[row * nc : (row + 1) * nc].copy(),
            rec.traces_raw[row].copy(),
        )
        for row, roi_id in enumerate(ids)
    }

    mask = rec.rois.copy()
    mask[mask == ids[0]] = 1  # delete the first ROI
    rows, cols = np.nonzero(mask == 1)
    mask[rows[0], cols[0]] = -50  # draw a one-pixel ROI
    ra.apply_mask(rec, mask)

    assert ra.check_alignment(rec) == []
    assert rec.num_rois == len(ids)
    for row, roi_id in enumerate(ra.roi_ids(rec.rois)):
        found = by_pixels.get(fingerprint(rec.rois, roi_id))
        if found is None:
            assert row == rec.num_rois - 1, "the new ROI comes last"
            assert np.isnan(rec.ipl_depths[row])
            assert np.isnan(rec.strfs[row * nc : (row + 1) * nc]).all()
            continue
        depth, strfs, trace = found
        got = rec.strfs[row * nc : (row + 1) * nc]
        assert rec.ipl_depths[row] == depth
        assert np.array_equal(np.ma.getdata(got), np.ma.getdata(strfs))
        # The loader masks part of each STRF; a copy of .data alone would unmask it
        assert isinstance(rec.strfs, np.ma.MaskedArray) == isinstance(strfs, np.ma.MaskedArray)
        assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(strfs))
        assert np.allclose(rec.traces_raw[row], trace)
    assert ra.rois_missing_strfs(rec) == [rec.num_rois - 1]


def test_survivors_keep_their_traces_rather_than_a_re_extraction():
    """A survivor's trace is what its RF was computed from; it must not be
    swapped for a fresh extraction that may differ (IGOR vs pygor)."""
    rec = FingerprintRecording(grid_mask(4))
    rec.traces_raw = rec.traces_raw + 1000.0  # as if extracted elsewhere
    kept = rec.traces_raw[1:].copy()
    mask = rec.rois.copy()
    mask[mask == -1] = 1
    mask[22, 3] = -50
    ra.apply_mask(rec, mask)
    assert np.array_equal(rec.traces_raw[:3], kept)
    new_id = ra.roi_ids(rec.rois)[3]
    assert rec.traces_raw[3, 0] == pytest.approx(fingerprint(rec.rois, new_id))


def test_fill_missing_traces_only_fills_nan_rows():
    rec = FingerprintRecording(grid_mask(3))
    mask = rec.rois.copy()
    mask[22, 3] = -50
    ra.apply_mask(rec, mask, refresh_traces=False)
    rec.traces_raw[:3] += 1000.0  # survivors' traces, from elsewhere
    survivors = rec.traces_raw[:3].copy()
    assert np.isnan(rec.traces_raw[3]).all()

    assert ra.fill_missing_traces(rec) == [3]
    assert np.array_equal(rec.traces_raw[:3], survivors)
    assert not np.isnan(rec.traces_raw[3]).any()


def test_rejected_rfs_follow_their_rois_through_an_edit():
    rec = FingerprintRecording(grid_mask())
    before = rec.fingerprints()
    # Rows 1 (ROI -2), 4 (ROI -5) and 7 (ROI -8) rejected
    rec.rf_rejected = np.array([1, 4, 7])
    mask = rec.rois.copy()
    mask[mask == -1] = 1  # delete one before them: everything shifts down
    mask[mask == -2] = 1  # delete a rejected one: its rejection goes with it
    mask[20, 20] = -40  # add one: starts unrejected
    ra.apply_mask(rec, mask, refresh_traces=False)

    rows = [int(np.flatnonzero(rec.fingerprints() == p)[0]) for p in (before[4], before[7])]
    assert rec.rf_rejected.tolist() == rows == [2, 5]


def test_a_recording_without_rejections_stays_without():
    rec = FingerprintRecording(grid_mask())
    mask = rec.rois.copy()
    mask[mask == -1] = 1
    ra.apply_mask(rec, mask, refresh_traces=False)
    assert not hasattr(rec, "rf_rejected")
