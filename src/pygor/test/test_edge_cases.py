"""Unit tests for the STRF analysis functions, using synthetic arrays only.

These need no recording, so they run on a fresh clone.
"""

import numpy as np
import pytest

import pygor.strf.extrema_timing as extrema_timing
import pygor.strf.spatial_alignment as spatial_alignment

# map_extrema_timing drops this many frames off each end before taking argmax,
# so the indices it returns are offset by the leading value.
CROP = (1, 1)


@pytest.fixture
def known_peaks():
    """One STRF with a positive peak at t=5 and a negative one at t=3."""
    strf = np.zeros((1, 10, 5, 5))
    strf[0, 5, 2, 2] = 10.0
    strf[0, 3, 1, 1] = -8.0
    return strf


class TestExtremaTiming:
    def test_zeros_are_all_below_threshold(self):
        timing = extrema_timing.map_extrema_timing(np.zeros((1, 10, 5, 5)), threshold=1.0)
        assert np.all(np.isnan(timing))

    def test_noise_is_mostly_below_threshold(self):
        noise = np.random.randn(1, 10, 5, 5) * 0.1
        timing = extrema_timing.map_extrema_timing(noise, threshold=3.0)
        assert np.isnan(timing).mean() > 0.5

    def test_peaks_land_on_the_right_frame(self, known_peaks):
        """Indices are relative to the cropped time axis, not the original."""
        timing = extrema_timing.map_extrema_timing(
            known_peaks, threshold=5.0, exclude_firstlast=CROP
        )
        assert timing[0, 2, 2] == 5 - CROP[0]
        assert timing[0, 1, 1] == 3 - CROP[0]

    def test_threshold_above_every_peak_masks_everything(self, known_peaks):
        timing = extrema_timing.map_extrema_timing(known_peaks, threshold=100.0)
        assert np.all(np.isnan(timing))

    def test_zero_threshold_keeps_pixels_with_any_signal(self, known_peaks):
        """The comparison is strict, so a flat-zero pixel is still masked out."""
        timing = extrema_timing.map_extrema_timing(known_peaks, threshold=0.0)
        assert not np.isnan(timing[0, 2, 2])
        assert not np.isnan(timing[0, 1, 1])
        assert np.isnan(timing[0, 0, 0])

    def test_three_dimensional_input_drops_the_cell_axis(self):
        timing = extrema_timing.map_extrema_timing(np.random.randn(10, 20, 20))
        assert timing.ndim == 2

    def test_four_dimensional_input_keeps_the_cell_axis(self):
        timing = extrema_timing.map_extrema_timing(np.random.randn(3, 10, 20, 20))
        assert timing.shape == (3, 20, 20)

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="3D or 4D"):
            extrema_timing.map_extrema_timing(np.random.randn(10, 10))

    def test_cropping_away_every_frame_is_rejected(self, known_peaks):
        with pytest.raises(ValueError, match="removes all time points"):
            extrema_timing.map_extrema_timing(known_peaks, exclude_firstlast=(5, 5))

    def test_single_pixel_arrays(self):
        timing = extrema_timing.map_extrema_timing(np.random.randn(1, 5, 1, 1), threshold=0.0)
        assert timing.shape == (1, 1, 1)

    def test_nan_pixels_stay_nan(self, known_peaks):
        strf = known_peaks.copy()
        strf[0, :, 0, 0] = np.nan
        timing = extrema_timing.map_extrema_timing(strf, threshold=1.0)
        assert np.isnan(timing[0, 0, 0])

    def test_infinities_are_picked_as_the_extremum(self, known_peaks):
        strf = known_peaks.copy()
        strf[0, 7, 4, 4] = np.inf
        timing = extrema_timing.map_extrema_timing(
            strf, threshold=1.0, exclude_firstlast=CROP
        )
        assert timing[0, 4, 4] == 7 - CROP[0]

    def test_raising_the_threshold_never_unmasks_a_pixel(self, known_peaks):
        masked = [
            np.isnan(
                extrema_timing.map_extrema_timing(known_peaks, threshold=threshold)
            ).sum()
            for threshold in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
        ]
        assert masked == sorted(masked)

    def test_negative_threshold_keeps_everything(self, known_peaks):
        """A negative threshold cannot exclude anything, since it compares |x|."""
        timing = extrema_timing.map_extrema_timing(known_peaks, threshold=-1.0)
        assert not np.any(np.isnan(timing))

    def test_a_large_array_completes(self):
        timing = extrema_timing.map_extrema_timing(
            np.random.randn(5, 50, 100, 100), threshold=2.0
        )
        assert timing.shape == (5, 100, 100)


class TestSpatialOverlap:
    def test_zero_maps_give_no_correlation_and_no_centroid(self):
        blank = np.zeros((20, 20))
        metrics = spatial_alignment.compute_spatial_overlap_metrics(
            blank, blank, threshold=1.0, method="all"
        )
        assert np.isnan(metrics["spatial_correlation"])
        assert np.isnan(metrics["centroid_distance"])
        assert metrics["jaccard_index"] == 0

    def test_identical_maps_overlap_completely(self):
        spatial_map = np.zeros((20, 20))
        spatial_map[8:12, 8:12] = 5.0
        metrics = spatial_alignment.compute_spatial_overlap_metrics(
            spatial_map, spatial_map, threshold=1.0, method="all"
        )
        assert metrics["jaccard_index"] == 1
        assert metrics["centroid_distance"] == pytest.approx(0.0)

    def test_disjoint_maps_do_not_overlap(self):
        left = np.zeros((20, 20))
        left[2:5, 2:5] = 5.0
        right = np.zeros((20, 20))
        right[15:18, 15:18] = 5.0
        metrics = spatial_alignment.compute_spatial_overlap_metrics(
            left, right, threshold=1.0, method="all"
        )
        assert metrics["jaccard_index"] == 0
        assert metrics["centroid_distance"] > 0
