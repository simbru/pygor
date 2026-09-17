"""Bootstrap settings must survive a save/load round trip.

``set_bootstrap_settings_default`` used to be called only on the H5 branch of
``STRF.__post_init__``, so every object built from a ScanM file -- which is how
the analysis pipeline builds all of them -- had no ``bs_settings`` at all. Since
``fit_contours`` and the pvals read it unconditionally, contouring raised
``AttributeError`` on every saved recording in the dataset.

Reprocessing those files was not an option, so the fix restores the defaults on
load as well as setting them on construction. These tests cover both halves.
"""

import h5py
import numpy as np
import pytest

import pygor.load


@pytest.fixture
def saved_without_bs_settings(fresh_strf, tmp_path):
    """A ``.recording.h5`` with the bootstrap attrs stripped out.

    That is what the pipeline's own files look like, so stripping them here is
    how a test reaches the case without a 350 MB fixture.
    """
    path = tmp_path / "stripped"
    saved = fresh_strf.save_object(path, overwrite=True)
    with h5py.File(saved, "r+") as handle:
        group = handle["recording_000"]
        for key in [k for k in group.attrs if "bs_settings" in k]:
            del group.attrs[key]
        if "bs_settings" in group:
            del group["bs_settings"]
    return saved


@pytest.mark.demo_data
class TestBootstrapSettingsRestored:
    def test_loading_restores_missing_settings(self, saved_without_bs_settings):
        recording = pygor.load.STRF.load_object(saved_without_bs_settings)
        assert hasattr(recording, "bs_settings")
        assert "do_bootstrap" in recording.bs_settings

    def test_fit_contours_works_after_load(self, saved_without_bs_settings):
        """The actual failure: AttributeError on every contour call."""
        recording = pygor.load.STRF.load_object(saved_without_bs_settings)
        contours = recording.fit_contours(roi=0)
        assert len(contours) == 1

    def test_existing_settings_are_not_overwritten(self, fresh_strf, tmp_path):
        """A recording that was bootstrapped must keep its own settings."""
        fresh_strf.bs_settings["space_sig_thresh"] = 0.123
        saved = fresh_strf.save_object(tmp_path / "kept", overwrite=True)
        recording = pygor.load.STRF.load_object(saved)
        assert recording.bs_settings["space_sig_thresh"] == pytest.approx(0.123)

    def test_fit_contours_honours_the_roi_argument(self, fresh_strf):
        """The no-bootstrap branch used to ignore `roi` and contour everything.

        `roi` indexes the flat (cell x colour) STRF axis, matching
        collapse_times, so one index is one STRF and not one cell.
        """
        fresh_strf.bs_settings["do_bootstrap"] = False
        assert len(fresh_strf.fit_contours(roi=0)) == 1
        assert len(fresh_strf.fit_contours()) == fresh_strf.num_strfs
