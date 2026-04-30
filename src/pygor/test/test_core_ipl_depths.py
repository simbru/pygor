"""Targeted tests for Core.estimate_ipl_depths clipping behavior."""

import unittest
from unittest.mock import patch

import numpy as np

from pygor.classes.core_data import Core


class TestCoreEstimateIplDepths(unittest.TestCase):
    """Regression tests for out-of-range IPL depth normalization."""

    def _core_stub(self, n_rois=6):
        core = Core.__new__(Core)
        core.name = "test_recording"
        rois = np.ones((64, 64), dtype=float)
        coords = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60)]
        for idx, (y, x) in enumerate(coords[:n_rois], start=1):
            rois[y, x] = -idx
        core.rois = rois
        core.images = np.zeros((1, 64, 64), dtype=float)
        core.ipl_depths = None
        return core

    @patch("pygor.anatomy.ipl.calculate_ipl_depths")
    @patch("pygor.anatomy.ipl.estimate_ipl_boundaries")
    def test_clips_negative_and_high_values(self, mock_estimate, mock_calculate):
        """Depths outside [0, 100] should be clipped by default."""
        mock_estimate.return_value = (
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float),
            np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float),
        )
        mock_calculate.return_value = np.array(
            [-300.0, -2.5, 25.0, 75.0, 120.0, 1000.0],
            dtype=float,
        )

        core = self._core_stub(n_rois=6)
        with self.assertWarns(UserWarning):
            depths = core.estimate_ipl_depths(plot=False)

        expected = np.array([0.0, 0.0, 25.0, 75.0, 100.0, 100.0], dtype=float)
        np.testing.assert_allclose(depths, expected)
        np.testing.assert_allclose(core.ipl_depths, expected)

    @patch("pygor.anatomy.ipl.calculate_ipl_depths")
    @patch("pygor.anatomy.ipl.estimate_ipl_boundaries")
    def test_non_finite_values_become_nan(self, mock_estimate, mock_calculate):
        """Non-finite outputs should be normalized to NaN."""
        mock_estimate.return_value = (
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float),
            np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float),
        )
        mock_calculate.return_value = np.array([10.0, np.inf, -np.inf, np.nan])

        core = self._core_stub(n_rois=4)

        with self.assertWarns(UserWarning):
            depths = core.estimate_ipl_depths(plot=False)

        self.assertEqual(depths[0], 10.0)
        self.assertTrue(np.isnan(depths[1]))
        self.assertTrue(np.isnan(depths[2]))
        self.assertTrue(np.isnan(depths[3]))


if __name__ == "__main__":
    unittest.main()
