"""Tests for pygor.anatomy.ipl — IPL depth estimation utilities.

AI-generated: Tests created with Claude Code assistance.
"""

import unittest
import warnings
import numpy as np

from pygor.anatomy.ipl import (
    interp_boundary,
    determine_orientation,
    calculate_ipl_depths,
    estimate_ipl_boundaries,
    _auto_detect_orientation,
)


class TestInterpBoundary(unittest.TestCase):
    """Tests for interp_boundary."""

    def test_basic_shape(self):
        coords = np.array([[0, 0], [0, 5], [0, 10]])
        result = interp_boundary(coords, n_points=50, smooth=False)
        self.assertEqual(result.shape, (50, 2))

    def test_output_endpoints_preserved(self):
        coords = np.array([[0, 0], [10, 10]])
        result = interp_boundary(coords, n_points=100, smooth=False)
        np.testing.assert_allclose(result[0], [0, 0], atol=1e-10)
        np.testing.assert_allclose(result[-1], [10, 10], atol=1e-10)

    def test_smoothing_runs(self):
        # Noisy coords — smoothing should not crash
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 30)
        y = 50 + rng.normal(0, 5, 30)
        coords = np.column_stack((y, x))
        result = interp_boundary(coords, n_points=200, smooth=True)
        self.assertEqual(result.shape, (200, 2))

    def test_rejects_single_point(self):
        with self.assertRaises(ValueError):
            interp_boundary(np.array([[1, 2]]))

    def test_rejects_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            interp_boundary(np.array([[1, 2, 3], [4, 5, 6]]))


class TestDetermineOrientation(unittest.TestCase):
    """Tests for determine_orientation."""

    def test_horizontal_line(self):
        # X varies a lot, Y is constant → horizontal
        x = np.linspace(0, 100, 50)
        y = np.ones(50) * 25
        self.assertEqual(determine_orientation(x, y), "horizontal")

    def test_vertical_line(self):
        # Y varies a lot, X is constant → vertical
        x = np.ones(50) * 25
        y = np.linspace(0, 100, 50)
        self.assertEqual(determine_orientation(x, y), "vertical")


class TestAutoDetectOrientation(unittest.TestCase):
    """Tests for _auto_detect_orientation."""

    def test_wider_in_x(self):
        centroids = np.array([[10, 0], [10, 50], [10, 100]])  # (y, x)
        self.assertEqual(_auto_detect_orientation(centroids), "horizontal")

    def test_taller_in_y(self):
        centroids = np.array([[0, 10], [50, 10], [100, 10]])  # (y, x)
        self.assertEqual(_auto_detect_orientation(centroids), "vertical")

    def test_square_defaults_horizontal(self):
        centroids = np.array([[0, 0], [100, 100]])  # equal spread
        self.assertEqual(_auto_detect_orientation(centroids), "horizontal")


class TestCalculateIplDepths(unittest.TestCase):
    """Tests for calculate_ipl_depths."""

    def _flat_horizontal_setup(self):
        """Flat horizontal boundaries: upper at y=10, lower at y=110."""
        x = np.linspace(0, 100, 200)
        upper = np.column_stack((np.full(200, 10.0), x))   # y=10 (100%)
        lower = np.column_stack((np.full(200, 110.0), x))   # y=110 (0%)
        return upper, lower

    def test_midpoint_roi_gets_50_percent(self):
        upper, lower = self._flat_horizontal_setup()
        roi = np.array([[60.0, 50.0]])  # y=60, midpoint between 10 and 110
        depths = calculate_ipl_depths(roi, upper, lower, orientation="horizontal")
        np.testing.assert_allclose(depths, [50.0], atol=1.0)

    def test_roi_on_upper_boundary(self):
        upper, lower = self._flat_horizontal_setup()
        roi = np.array([[10.0, 50.0]])  # y=10 = upper boundary = 100%
        depths = calculate_ipl_depths(roi, upper, lower, orientation="horizontal")
        np.testing.assert_allclose(depths, [100.0], atol=1.0)

    def test_roi_on_lower_boundary(self):
        upper, lower = self._flat_horizontal_setup()
        roi = np.array([[110.0, 50.0]])  # y=110 = lower boundary = 0%
        depths = calculate_ipl_depths(roi, upper, lower, orientation="horizontal")
        np.testing.assert_allclose(depths, [0.0], atol=1.0)

    def test_multiple_rois(self):
        upper, lower = self._flat_horizontal_setup()
        rois = np.array([
            [10.0, 20.0],   # 100%
            [60.0, 50.0],   # 50%
            [110.0, 80.0],  # 0%
            [35.0, 40.0],   # 75%
        ])
        depths = calculate_ipl_depths(rois, upper, lower, orientation="horizontal")
        np.testing.assert_allclose(depths, [100.0, 50.0, 0.0, 75.0], atol=1.0)

    def test_vertical_orientation(self):
        # Vertical boundaries: upper at x=10, lower at x=110
        y = np.linspace(0, 100, 200)
        upper = np.column_stack((y, np.full(200, 10.0)))   # x=10 (100%)
        lower = np.column_stack((y, np.full(200, 110.0)))   # x=110 (0%)
        roi = np.array([[50.0, 60.0]])  # x=60, midpoint between 10 and 110
        depths = calculate_ipl_depths(roi, upper, lower, orientation="vertical")
        np.testing.assert_allclose(depths, [50.0], atol=1.0)

    def test_auto_orientation_detection(self):
        """Should auto-detect horizontal from boundary shape."""
        x = np.linspace(0, 100, 200)
        upper = np.column_stack((np.full(200, 10.0), x))
        lower = np.column_stack((np.full(200, 110.0), x))
        roi = np.array([[60.0, 50.0]])
        depths = calculate_ipl_depths(roi, upper, lower)  # no orientation given
        np.testing.assert_allclose(depths, [50.0], atol=1.0)


class TestEstimateIplBoundaries(unittest.TestCase):
    """Tests for estimate_ipl_boundaries."""

    def _make_grid_rois(self, n_x=20, n_y=10, x_range=(0, 200),
                        y_range=(20, 80)):
        """Create a uniform grid of ROI centroids."""
        x = np.linspace(x_range[0], x_range[1], n_x)
        y = np.linspace(y_range[0], y_range[1], n_y)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((yy.ravel(), xx.ravel()))  # (y, x) format

    def test_output_shapes(self):
        rois = self._make_grid_rois()
        upper, lower = estimate_ipl_boundaries(rois, n_points=500)
        self.assertEqual(upper.shape, (500, 2))
        self.assertEqual(lower.shape, (500, 2))

    def test_boundaries_bracket_rois(self):
        """Upper boundary should be above (lower y) most ROIs,
        lower boundary should be below (higher y) most ROIs."""
        rois = self._make_grid_rois(y_range=(30, 70))
        upper, lower = estimate_ipl_boundaries(
            rois, n_bins=10, upper_percentile=5, lower_percentile=95
        )
        # With margin, upper y values should mostly be < 30
        # and lower y values should mostly be > 70
        self.assertLess(np.median(upper[:, 0]), 35)
        self.assertGreater(np.median(lower[:, 0]), 65)

    def test_depths_from_estimated_boundaries(self):
        """ROIs estimated from auto boundaries should have reasonable depths."""
        rois = self._make_grid_rois(n_x=30, n_y=15, y_range=(20, 80))
        upper, lower = estimate_ipl_boundaries(rois, n_bins=10)
        depths = calculate_ipl_depths(rois, upper, lower, orientation="horizontal")
        # With 5/95 percentiles + median filter + margin, the vast majority
        # of ROIs should be within a reasonable range
        in_range = np.mean((depths >= -10) & (depths <= 110))
        self.assertGreater(in_range, 0.90)

    def test_too_few_rois_raises(self):
        with self.assertRaises(ValueError):
            estimate_ipl_boundaries(np.array([[10, 20], [30, 40]]))

    def test_sparse_bins_warns(self):
        # ROIs clustered in part of the scan range → many empty bins
        rng = np.random.default_rng(42)
        # 30 ROIs clustered in x=[0,20] out of x=[0,100] range
        x_cluster = rng.uniform(0, 20, 25)
        x_spread = rng.uniform(80, 100, 5)  # a few outliers to widen bin range
        x = np.concatenate([x_cluster, x_spread])
        y = rng.uniform(20, 80, 30)
        rois = np.column_stack((y, x))
        with self.assertWarns(UserWarning):
            estimate_ipl_boundaries(rois, n_bins=20)

    def test_vertical_scan(self):
        """Vertical grid should be auto-detected and handled."""
        # Taller than wide → vertical
        x = np.linspace(30, 70, 5)
        y = np.linspace(0, 200, 20)
        xx, yy = np.meshgrid(x, y)
        rois = np.column_stack((yy.ravel(), xx.ravel()))
        upper, lower = estimate_ipl_boundaries(rois)
        depths = calculate_ipl_depths(rois, upper, lower, orientation="vertical")
        self.assertEqual(depths.shape, (rois.shape[0],))
        # Should be reasonable values
        self.assertGreater(np.mean((depths >= -10) & (depths <= 110)), 0.8)


if __name__ == "__main__":
    unittest.main()
