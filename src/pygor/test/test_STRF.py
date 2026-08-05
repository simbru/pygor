"""Integration tests for the STRF class against the demo recording."""

import numpy as np
import pandas as pd
import pytest

import pygor.data_helpers
import pygor.load
import pygor.utils.helpinfo
from pygor.test.helpers import callable_with_roi_only, callable_without_arguments

pytestmark = pytest.mark.demo_data

# Errors that mean an internal name went stale, as opposed to a method
# deliberately refusing the arguments it was given.
PLUMBING_ERRORS = (AttributeError, NameError, UnboundLocalError)

# Methods that write next to the source file, so calling them blind would
# clobber the demo recording. They are covered by the export tests instead.
WRITES_TO_SOURCE_DIR = {"export_to_h5", "save", "save_object"}

# run_bootstrap signals a misconfigured object with AttributeError, which the
# smoke test cannot tell from a stale name. TestBootstrap covers it properly.
SMOKE_EXCLUDED = WRITES_TO_SOURCE_DIR | {"run_bootstrap"}

# Failures on the demo recording that look like real defects rather than
# deliberate refusals, recorded as what was observed rather than as a diagnosis.
# The suite stays green while they stay visible, and each flips to XPASS the
# moment it is fixed.
_OVERLAP_RESHAPE = "ValueError: cannot reshape 9792 elements into (4, 58, -1)"
KNOWN_BROKEN = {
    "to_rgb": "IndexError: index 16 is out of bounds for axis 0 with size 16",
    "plot_space": "TypeError: imshow got a 3D array",
    "plot_averages": "TypeError: divides by frame_hz while averages is None",
    "spatial_overlap_blue_uv": _OVERLAP_RESHAPE,
    "spatial_overlap_green_blue": _OVERLAP_RESHAPE,
    "spatial_overlap_green_uv": _OVERLAP_RESHAPE,
    "spatial_overlap_index_mean": _OVERLAP_RESHAPE,
    "spatial_overlap_index_min": _OVERLAP_RESHAPE,
    "spatial_overlap_index_stats": _OVERLAP_RESHAPE,
    "spatial_overlap_index_std": _OVERLAP_RESHAPE,
    "spatial_overlap_index_var": _OVERLAP_RESHAPE,
    "spatial_overlap_red_blue": _OVERLAP_RESHAPE,
    "spatial_overlap_red_green": _OVERLAP_RESHAPE,
    "spatial_overlap_red_uv": _OVERLAP_RESHAPE,
}


def _params(names):
    """Attach an xfail to the names listed in KNOWN_BROKEN."""
    for name in names:
        reason = KNOWN_BROKEN.get(name)
        marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
        yield pytest.param(name, marks=marks)


def _call_and_check(obj, method_name, *args):
    """Call a method and complain only about errors that mean a stale name.

    This does not check the answer is right, only that the call reaches its own
    code. It is what catches an attribute renamed in one place and not another,
    which is the failure mode that keeps showing up here.
    """
    if method_name in KNOWN_BROKEN:
        getattr(obj, method_name)(*args)  # let xfail see the real error
        return
    try:
        getattr(obj, method_name)(*args)
    except PLUMBING_ERRORS as e:
        pytest.fail(f"{method_name}() hit a stale name: {type(e).__name__}: {e}")
    except Exception as e:
        assert str(e), f"{method_name}() raised {type(e).__name__} with no message"


@pytest.mark.parametrize(
    "method_name",
    list(_params(callable_without_arguments(pygor.load.STRF, exclude=SMOKE_EXCLUDED))),
)
def test_method_is_wired_up(scratch_strf, method_name):
    _call_and_check(scratch_strf, method_name)


@pytest.mark.parametrize(
    "method_name",
    list(_params(callable_with_roi_only(pygor.load.STRF, exclude=SMOKE_EXCLUDED))),
)
def test_roi_method_is_wired_up(scratch_strf, method_name):
    _call_and_check(scratch_strf, method_name, 0)


def test_attributes_readable(strf):
    for name in pygor.utils.helpinfo.get_attribute_list(strf, with_types=False):
        getattr(strf, name)


def test_strf_axis_order(strf):
    """STRF arrays are [cell, time, y, x] with time the short axis."""
    assert strf.strfs.ndim == 4
    n_cells, n_time, n_y, n_x = strf.strfs.shape
    assert min(n_cells, n_time, n_y, n_x) > 0
    assert n_time <= min(n_y, n_x)


def test_by_channel_splits_by_colour(strf):
    """The generated _by_channel wrappers reshape a flat result per colour."""
    if not strf.multicolour:
        pytest.skip("single-colour recording")
    flat = strf.spatial_polarity_index()
    per_channel = strf.spatial_polarity_index_by_channel()
    assert per_channel.shape[0] == strf.n_colours
    assert per_channel.size == np.size(flat)


def test_no_by_channel_wrapper_for_gui_methods():
    """A _by_channel copy of a napari method would be a second way to hang."""
    for name in ("draw_rois", "napari_strfs", "get_depth", "view_images_interactive"):
        assert not hasattr(pygor.load.STRF, f"{name}_by_channel"), (
            f"{name}_by_channel would open a blocking window"
        )


def test_contours_fit(fresh_strf):
    fresh_strf.fit_contours()


def test_save_pkl_roundtrip(fresh_strf, tmp_path):
    fresh_strf.save_pkl(str(tmp_path), "roundtrip.pkl")
    assert (tmp_path / "roundtrip.pkl").exists()


def test_get_help_writes_to_stdout(strf, capsys):
    strf.get_help(hints=True, types=True)
    assert capsys.readouterr().out.strip(), "get_help() printed nothing"


class TestExtremaTiming:
    def test_shape_matches_strfs(self, strf):
        timing = strf.map_extrema_timing()
        n_cells, _, n_y, n_x = strf.strfs.shape
        assert timing.shape == (n_cells, n_y, n_x)

    def test_single_roi_is_2d(self, strf):
        assert strf.map_extrema_timing(roi=0).ndim == 2

    def test_roi_out_of_range(self, strf):
        with pytest.raises(IndexError):
            strf.map_extrema_timing(roi=strf.strfs.shape[0])

    def test_higher_threshold_masks_more(self, strf):
        low = np.isnan(strf.map_extrema_timing(threshold=1.0)).sum()
        high = np.isnan(strf.map_extrema_timing(threshold=5.0)).sum()
        assert high >= low

    def test_timing_values_are_frame_indices(self, strf):
        """Values index the time axis after the excluded first and last frames."""
        timing = strf.map_extrema_timing(threshold=1.0, exclude_firstlast=(1, 1))
        finite = timing[np.isfinite(timing)]
        if finite.size == 0:
            pytest.skip("nothing above threshold in the demo recording")
        assert finite.min() >= 0
        assert finite.max() < strf.strfs.shape[1] - 2


class TestMulticolourAlignment:
    @pytest.fixture(autouse=True)
    def skip_single_colour(self, strf):
        if not strf.multicolour:
            pytest.skip("single-colour recording")

    def test_alignment_matrices_are_square_over_colours(self, strf):
        results = strf.analyze_spatial_alignment(roi=0)
        n = strf.n_colours
        for key in ("correlation_matrix", "overlap_matrix", "distance_matrix"):
            assert results[key].shape == (n, n)

    def test_alignment_summary_keys(self, strf):
        results = strf.analyze_spatial_alignment(roi=0)
        assert set(results) >= {"summary_stats", "channel_centroids", "pairwise_metrics"}

    def test_correlation_matrix_is_symmetric(self, strf):
        matrix = strf.analyze_spatial_alignment(roi=0)["correlation_matrix"]
        assert np.allclose(matrix, matrix.T, equal_nan=True)

    def test_channel_overlap_pair(self, strf):
        overlap = strf.compute_colour_channel_overlap(roi=0, colour_channels=(0, 1))
        assert isinstance(overlap, dict)


class TestBootstrap:
    """Bootstrap mutates the object, so these run on their own copy."""

    @pytest.fixture
    def bootstrapped(self, fresh_strf):
        fresh_strf.set_bootstrap_bool(True)
        fresh_strf.update_bootstrap_settings(
            pygor.data_helpers.create_bs_dict(space_bs_n=5, time_bs_n=5)
        )
        fresh_strf.run_bootstrap()
        return fresh_strf

    def test_settings_are_applied(self, fresh_strf):
        fresh_strf.set_bootstrap_bool(True)
        fresh_strf.update_bootstrap_settings(
            pygor.data_helpers.create_bs_dict(space_bs_n=10, time_bs_n=10)
        )
        settings = fresh_strf.get_bootstrap_settings()
        assert settings["space_bs_n"] == 10
        assert settings["time_bs_n"] == 10

    def test_run_sets_flag(self, bootstrapped):
        assert bootstrapped.bs_settings["bs_already_ran"]

    def test_rerun_without_a_terminal_refuses(self, bootstrapped):
        """Unattended callers must get an error, not a prompt they cannot answer.

        This is the deadlock that made the suite unrunnable: run_bootstrap()
        asked on stdin and waited forever.
        """
        with pytest.raises(RuntimeError, match="force=True"):
            bootstrapped.run_bootstrap()

    def test_rerun_with_force_succeeds(self, bootstrapped):
        bootstrapped.run_bootstrap(force=True)

    def test_pvals_table_columns(self, bootstrapped):
        table = bootstrapped.get_pvals_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0
        if bootstrapped.multicolour:
            expected = ["space_R", "time_R", "sig_R", "sig_any"]
        else:
            expected = ["space", "time", "sig"]
        for column in expected:
            assert column in table.columns


class TestPlotting:
    """Plots are checked for running headless, not for what they look like."""

    def test_timecourse(self, strf):
        strf.plot_timecourse(0)

    def test_chromatic_overview(self, strf):
        strf.plot_chromatic_overview()
