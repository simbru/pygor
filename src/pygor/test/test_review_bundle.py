"""Unit tests for pygor.review.bundle.

The ROI-index mapping is the part worth the most care: a positional join that is
quietly off by the cells a transfer dropped produces plausible numbers for the
wrong cells, which is exactly the failure this whole review stack exists to
catch.
"""

import numpy as np
import pytest

from pygor.review import bundle as review_bundle
from pygor.review.bundle import (
    BundleCache,
    FovBundle,
    MissingPartner,
    alignment_of,
    group_refs,
    master_roi_index,
)
from pygor.test.test_review_index import write_recording

from pygor.review import index as review_index

SEGMENTED = {"method": "segmented", "mode": "blob", "source": "x"}


def transferred(expected, lost, source="master"):
    return {
        "method": "transferred",
        "source": source,
        "correlation": 0.77,
        "error": 0.02,
        "shift": [1.5, -0.5],
        "expected_roi_ids": expected,
        "lost_roi_ids": lost,
    }


class TestMasterRoiIndex:
    def test_segmented_is_identity(self):
        assert list(master_roi_index(SEGMENTED, 4)) == [0, 1, 2, 3]

    def test_no_provenance_is_identity(self):
        assert list(master_roi_index(None, 3)) == [0, 1, 2]

    def test_clean_transfer_is_identity(self):
        origin = transferred([-1, -2, -3, -4], [])
        assert list(master_roi_index(origin, 4)) == [0, 1, 2, 3]

    def test_lossy_transfer_skips_the_lost_cells(self):
        """Row 2 of this recording is master ROI 3, not master ROI 2."""
        origin = transferred([-1, -2, -3, -4, -5], [-3])
        assert list(master_roi_index(origin, 4)) == [0, 1, 3, 4]

    def test_inconsistent_provenance_raises(self):
        """Neither mapping is trustworthy, so guessing would be worse."""
        origin = transferred([-1, -2, -3, -4], [-3])
        with pytest.raises(RuntimeError, match="surviving ROIs"):
            master_roi_index(origin, 99)

    @pytest.mark.parametrize(
        "origin,num_rois",
        [
            (SEGMENTED, 4),
            (transferred([-1, -2, -3, -4], []), 4),
            (transferred([-1, -2, -3, -4, -5], [-2, -5]), 3),
        ],
    )
    def test_parity_with_the_pipeline_implementation(self, origin, num_rois):
        """The pipeline and the review tool must agree, or joins diverge."""
        common = pytest.importorskip("analyses._common")

        class FakeRecording:
            def __init__(self, roi_origin, num_rois):
                self.roi_origin = roi_origin
                self.num_rois = num_rois

        expected = common.master_roi_index(FakeRecording(origin, num_rois))
        assert list(master_roi_index(origin, num_rois)) == list(expected)


class TestAlignmentInfo:
    def test_reads_transfer_fields(self, tmp_path):
        path = write_recording(
            tmp_path, roi_origin=transferred([-1, -2, -3], [-2]), num_rois=2
        )
        info = alignment_of(review_index.scan_recording(path), role="swn")
        assert info.transferred
        assert info.n_lost == 1
        assert info.n_expected == 3
        assert info.correlation == pytest.approx(0.77)
        assert info.shift == (1.5, -0.5)
        assert info.usable

    def test_marks_inconsistent_provenance_unusable(self, tmp_path):
        """A recording whose provenance disagrees with it is a finding itself."""
        path = write_recording(
            tmp_path, roi_origin=transferred([-1, -2, -3], [-2]), num_rois=7
        )
        info = alignment_of(review_index.scan_recording(path))
        assert not info.usable
        assert "surviving ROIs" in info.problem

    def test_segmented_has_no_correlation(self, tmp_path):
        path = write_recording(tmp_path, roi_origin=SEGMENTED)
        info = alignment_of(review_index.scan_recording(path))
        assert not info.transferred
        assert np.isnan(info.correlation)


class TestBundleCache:
    class FakeRef:
        def __init__(self, name, size):
            self.path = name
            self.size = size
            self.class_name = "Core"

    def _cache(self, loaded):
        cache = BundleCache(max_bytes=1000)
        cache._load = lambda ref: loaded.setdefault(ref.path, object())
        return cache

    def test_hit_does_not_reload(self):
        loaded = {}
        cache = self._cache(loaded)
        ref = self.FakeRef("a", 10)
        first = cache.get(ref)
        assert cache.get(ref) is first
        assert cache.stats()["hits"] == 1

    def test_evicts_by_bytes_not_count(self):
        """Three small recordings fit where one large one does not."""
        cache = self._cache({})
        for name in "abc":
            cache.get(self.FakeRef(name, 100))
        assert cache.stats()["items"] == 3
        cache.get(self.FakeRef("big", 900))
        assert cache.stats()["evictions"] > 0
        assert cache.nbytes <= cache.max_bytes or cache.stats()["items"] == 1

    def test_keeps_the_newest_even_when_oversized(self):
        """Evicting what was just requested would loop forever."""
        cache = self._cache({})
        cache.get(self.FakeRef("huge", 10_000))
        assert cache.stats()["items"] == 1

    def test_discard_and_clear(self):
        cache = self._cache({})
        ref = self.FakeRef("a", 10)
        cache.get(ref)
        cache.discard(ref)
        assert cache.stats()["items"] == 0
        cache.get(ref)
        cache.clear()
        assert cache.stats()["items"] == 0


@pytest.fixture
def fov(tmp_path):
    """A field of view whose ROIs were segmented on the direction recording."""
    write_recording(tmp_path, stem="0_0_OSDS_2x_vel", num_rois=5, roi_origin=SEGMENTED)
    write_recording(
        tmp_path,
        stem="0_0_ColourSWN_200",
        num_rois=4,
        roi_origin=transferred([-1, -2, -3, -4, -5], [-3], source="0_0_OSDS_2x_vel"),
    )
    refs = review_index.scan_processed(tmp_path, cache=False)
    by_stem = {r.stem: r for r in refs}
    return FovBundle(
        fov_uid="240101 test::control::0_0",
        condition="control",
        session="240101 test",
        prefix="0_0",
        refs={"osds": by_stem["0_0_OSDS_2x_vel"], "swn": by_stem["0_0_ColourSWN_200"]},
        master_role="osds",
        expected={"osds": True, "swn": True, "fff": True},
    )


class TestFovBundle:
    def test_master_defines_the_cell_count(self, fov):
        assert fov.master_role == "osds"
        assert fov.n_cells == 5
        assert set(fov.roles) == {"osds", "swn"}

    def test_missing_partner_raises_a_named_error(self, fov):
        with pytest.raises(MissingPartner, match="fff"):
            fov.peek("fff")

    def test_missing_reports_the_expected_but_absent(self, fov):
        assert "fff" in fov.missing()

    def test_roi_map_accounts_for_the_lost_cell(self, fov):
        """The noise recording lost master ROI 2, so its rows skip it."""
        assert list(fov.roi_map("osds")) == [0, 1, 2, 3, 4]
        assert list(fov.roi_map("swn")) == [0, 1, 3, 4]

    def test_master_to_row_translates(self, fov):
        assert fov.master_to_row("swn", 0) == 0
        assert fov.master_to_row("swn", 3) == 2  # not 3
        assert fov.master_to_row("osds", 3) == 3

    def test_master_to_row_is_none_for_a_lost_cell(self, fov):
        """The case a positional join would get silently wrong."""
        assert fov.master_to_row("swn", 2) is None

    def test_row_in_survivor_space(self, fov):
        """An array with num_rois rows was renumbered to the survivors."""
        n_rows = fov.peek("swn").num_rois  # 4 survivors of 5
        assert fov.row_in("swn", 0, n_rows) == 0
        assert fov.row_in("swn", 3, n_rows) == 2
        assert fov.row_in("swn", 2, n_rows) is None  # lost

    def test_row_in_master_space(self, fov):
        """An array with the master's row count is NaN-padded, not renumbered.

        This is how the pipeline leaves traces and STRFs on a transferred
        recording. Translating into survivor space here would show the wrong
        cell's receptive field under the right cell's number.
        """
        n_rows = fov.n_cells  # 5, the master's count
        assert fov.row_in("swn", 0, n_rows) == 0
        assert fov.row_in("swn", 3, n_rows) == 3  # not 2
        assert fov.row_in("swn", 2, n_rows) is None  # lost: its row is padding

    def test_row_in_master_role_is_identity(self, fov):
        assert fov.row_in("osds", 4, fov.n_cells) == 4
        assert fov.row_in("osds", 99, fov.n_cells) is None

    def test_row_in_refuses_an_unrecognised_length(self, fov):
        """Neither space fits, so guessing would silently mis-index."""
        with pytest.raises(RuntimeError, match="neither"):
            fov.row_in("swn", 0, 17)

    def test_light_tier_reads_without_loading(self, fov):
        assert fov.projection("osds").shape == (8, 8)
        assert fov.roi_mask("osds").shape == (8, 8)
        assert set(fov.light("swn")) >= {"rois", "average_stack"}
        assert fov.cache.stats()["items"] == 0  # nothing was loaded

    def test_alignments_cover_every_role(self, fov):
        alignments = fov.alignments()
        assert set(alignments) == {"osds", "swn"}
        assert alignments["swn"].transferred
        assert not alignments["osds"].transferred

    def test_cell_row_lookup(self, fov):
        import pandas as pd

        fov.attach_cells(
            pd.DataFrame({"roi_id": [0, 1, 4], "strf_pass_bool_ch0": [True, False, True]})
        )
        assert fov.cell_row(0).strf_pass_bool_ch0
        assert not fov.cell_row(1).strf_pass_bool_ch0
        assert fov.cell_row(2) is None  # roi_id 2 has no row

    def test_cell_row_without_a_csv_is_none(self, fov):
        assert fov.cell_row(0) is None


class TestGroupRefs:
    def _rules(self):
        def prefix_of(stem):
            import re

            match = re.match(r"^(?:CTinj)?(\d+_\d+)", stem)
            return match.group(1) if match else None

        def classify(stem):
            low = stem.lower()
            if "swn" in low:
                return "swn"
            if "fff" in low:
                return "fff"
            if "osds" in low:
                return "osds"
            return None

        return prefix_of, classify, lambda c, s, p: f"{s}::{c}::{p}"

    def test_groups_by_condition_session_prefix(self, tmp_path):
        write_recording(tmp_path, stem="0_0_ColourSWN_200")
        write_recording(tmp_path, stem="0_0_OSDS_2x_vel")
        write_recording(tmp_path, stem="0_1_ColourSWN_200")
        prefix_of, classify, fov_uid_of = self._rules()
        bundles = group_refs(
            review_index.scan_processed(tmp_path, cache=False),
            prefix_of=prefix_of,
            classify=classify,
            fov_uid_of=fov_uid_of,
        )
        assert len(bundles) == 2
        first = [b for b in bundles if b.prefix == "0_0"][0]
        assert set(first.roles) == {"swn", "osds"}
        assert first.master_role == "osds"

    def test_same_prefix_in_two_conditions_stays_two_fovs(self, tmp_path):
        """The collision that cost the aggregate CSV 200 cells.

        Two recordings of one plane, one per condition, share a prefix once the
        condition tag is stripped. They are different sets of cells and must not
        be merged into one field of view.
        """
        write_recording(tmp_path, condition="control", session="240124 inj",
                        stem="1_0_ColourSWN_200", num_rois=58)
        write_recording(tmp_path, condition="acblock", session="240124 inj",
                        stem="CTinj1_0_ColourSWN_200", num_rois=137)
        prefix_of, classify, fov_uid_of = self._rules()
        bundles = group_refs(
            review_index.scan_processed(tmp_path, cache=False),
            prefix_of=prefix_of,
            classify=classify,
            fov_uid_of=fov_uid_of,
        )
        assert len(bundles) == 2
        assert len({b.fov_uid for b in bundles}) == 2
        assert {b.n_cells for b in bundles} == {58, 137}

    def test_unclassifiable_stems_are_skipped(self, tmp_path):
        write_recording(tmp_path, stem="0_0_something_else")
        prefix_of, classify, fov_uid_of = self._rules()
        assert (
            group_refs(
                review_index.scan_processed(tmp_path, cache=False),
                prefix_of=prefix_of,
                classify=classify,
                fov_uid_of=fov_uid_of,
            )
            == []
        )

    def test_pick_chooses_between_repeats(self, tmp_path):
        write_recording(tmp_path, stem="0_0_OSDS_2x_vel")
        write_recording(tmp_path, stem="0_0_OSDS_2x_vel_repeat")
        write_recording(tmp_path, stem="0_0_ColourSWN_200")
        prefix_of, classify, fov_uid_of = self._rules()
        bundles = group_refs(
            review_index.scan_processed(tmp_path, cache=False),
            prefix_of=prefix_of,
            classify=classify,
            fov_uid_of=fov_uid_of,
        )
        assert len(bundles) == 1
        assert bundles[0].refs["osds"].stem == "0_0_OSDS_2x_vel"

    def test_swn_is_master_when_there_is_no_osds(self, tmp_path):
        write_recording(tmp_path, stem="0_0_ColourSWN_200")
        write_recording(tmp_path, stem="0_0_FFF_RGBUV")
        prefix_of, classify, fov_uid_of = self._rules()
        bundles = group_refs(
            review_index.scan_processed(tmp_path, cache=False),
            prefix_of=prefix_of,
            classify=classify,
            fov_uid_of=fov_uid_of,
        )
        assert bundles[0].master_role == "swn"
