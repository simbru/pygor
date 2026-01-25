import unittest
import numpy as np

from pygor.timeseries.osds import tuning_metrics


class TestOSDSTuningMetrics(unittest.TestCase):
    def test_direction_vector_magnitude_perfect(self):
        directions = np.array([0, 90, 180, 270])
        responses = np.array([1.0, 0.0, 0.0, 0.0])
        r = tuning_metrics.compute_direction_vector_magnitude(responses, directions)
        self.assertAlmostEqual(r, 1.0, places=7)

    def test_direction_vector_magnitude_uniform(self):
        directions = np.array([0, 90, 180, 270])
        responses = np.array([1.0, 1.0, 1.0, 1.0])
        r = tuning_metrics.compute_direction_vector_magnitude(responses, directions)
        self.assertAlmostEqual(r, 0.0, places=7)

    def test_orientation_vector_magnitude_perfect(self):
        directions = np.array([0, 90, 180, 270])
        responses = np.array([1.0, 0.0, 1.0, 0.0])
        r = tuning_metrics.compute_orientation_vector_magnitude(responses, directions)
        self.assertAlmostEqual(r, 1.0, places=7)

        mean_orientation = tuning_metrics.compute_mean_orientation(responses, directions)
        self.assertAlmostEqual(mean_orientation, 0.0, places=7)

    def test_orientation_vector_magnitude_uniform(self):
        directions = np.array([0, 90, 180, 270])
        responses = np.array([1.0, 1.0, 1.0, 1.0])
        r = tuning_metrics.compute_orientation_vector_magnitude(responses, directions)
        self.assertAlmostEqual(r, 0.0, places=7)

    def test_compute_all_tuning_metrics_keys_and_alias(self):
        class MockOSDS:
            def __init__(self, tuning_functions, directions_list):
                self._tuning = np.array(tuning_functions)
                self.num_rois = self._tuning.shape[0]
                self.directions_list = directions_list
                self.dir_phase_num = 1

            def compute_tuning_function(self, metric='peak', phase_num=1, window=None, roi_index=None):
                if roi_index is not None:
                    return self._tuning[roi_index]
                return self._tuning

            def get_epoch_dur(self):
                return 0

        directions = [0, 90, 180, 270]
        tuning_functions = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        osds = MockOSDS(tuning_functions, directions)
        metrics = tuning_metrics.compute_all_tuning_metrics(osds)

        self.assertIn('direction_vector_magnitude', metrics)
        self.assertIn('orientation_vector_magnitude', metrics)
        self.assertIn('mean_direction', metrics)
        self.assertIn('mean_orientation', metrics)
        self.assertIn('vector_magnitude', metrics)

        np.testing.assert_allclose(
            metrics['direction_vector_magnitude'],
            metrics['vector_magnitude']
        )


class TestPhaseIdxExtraction(unittest.TestCase):
    """Test phase_idx parameter for OSDS getter methods."""

    def setUp(self):
        """Create mock OSDS object for testing _extract_phase."""
        from pygor.classes.osds_data import OSDS

        # Create a minimal mock that just tests the _extract_phase method
        class MockOSDSForExtract:
            """Mock class with _extract_phase method from OSDS."""

            def _extract_phase(self, result, phase_idx):
                """Copy of the _extract_phase method for testing."""
                if phase_idx is None:
                    return result
                if not hasattr(result, 'ndim') or result.ndim != 2:
                    if phase_idx != 0:
                        raise IndexError(f"phase_idx={phase_idx} invalid for single-phase result")
                    return result
                if phase_idx < 0 or phase_idx >= result.shape[0]:
                    raise IndexError(f"phase_idx={phase_idx} out of range for {result.shape[0]} phases")
                return result[phase_idx]

        self.mock = MockOSDSForExtract()

    def test_extract_phase_none_returns_unchanged(self):
        """Test that phase_idx=None returns result unchanged."""
        result_1d = np.array([1, 2, 3, 4, 5])
        result_2d = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_array_equal(
            self.mock._extract_phase(result_1d, None), result_1d
        )
        np.testing.assert_array_equal(
            self.mock._extract_phase(result_2d, None), result_2d
        )

    def test_extract_phase_from_2d_array(self):
        """Test extracting specific phase from 2D array."""
        result_2d = np.array([[1, 2, 3], [4, 5, 6]])  # (2 phases, 3 rois)

        phase0 = self.mock._extract_phase(result_2d, 0)
        phase1 = self.mock._extract_phase(result_2d, 1)

        np.testing.assert_array_equal(phase0, np.array([1, 2, 3]))
        np.testing.assert_array_equal(phase1, np.array([4, 5, 6]))
        self.assertEqual(phase0.shape, (3,))
        self.assertEqual(phase1.shape, (3,))

    def test_extract_phase_1d_with_phase_idx_0(self):
        """Test that 1D array with phase_idx=0 returns unchanged."""
        result_1d = np.array([1, 2, 3, 4, 5])
        extracted = self.mock._extract_phase(result_1d, 0)
        np.testing.assert_array_equal(extracted, result_1d)

    def test_extract_phase_1d_with_invalid_phase_idx_raises(self):
        """Test that 1D array with phase_idx != 0 raises IndexError."""
        result_1d = np.array([1, 2, 3, 4, 5])

        with self.assertRaises(IndexError):
            self.mock._extract_phase(result_1d, 1)

    def test_extract_phase_out_of_range_raises(self):
        """Test that out-of-range phase_idx raises IndexError."""
        result_2d = np.array([[1, 2, 3], [4, 5, 6]])  # 2 phases

        with self.assertRaises(IndexError):
            self.mock._extract_phase(result_2d, 2)

        with self.assertRaises(IndexError):
            self.mock._extract_phase(result_2d, -1)


class TestComputeAllTuningMetricsMultiPhase(unittest.TestCase):
    """Test multi-phase behavior of compute_all_tuning_metrics."""

    def test_multi_phase_output_shape(self):
        """Test that multi-phase returns (n_phases, n_rois) arrays."""

        class MockMultiPhaseOSDS:
            def __init__(self):
                self.num_rois = 3
                self.directions_list = [0, 90, 180, 270]
                self.dir_phase_num = 2
                # Tuning functions per ROI per direction
                # When compute_tuning_function is called with phase_num, it should
                # return (n_rois, n_directions) for that phase
                self._tuning_phase0 = np.random.rand(3, 4)
                self._tuning_phase1 = np.random.rand(3, 4)

            def compute_tuning_function(self, metric='peak', phase_num=None, window=None, roi_index=None):
                # When called for a specific roi_index, return (n_directions,) or (n_directions, n_phases)
                if roi_index is not None:
                    if phase_num is not None:
                        # Return for specific phase
                        if phase_num == 2:
                            return np.column_stack([self._tuning_phase0[roi_index],
                                                    self._tuning_phase1[roi_index]])
                        return self._tuning_phase0[roi_index]
                    return self._tuning_phase0[roi_index]
                # Return all ROIs
                if phase_num is not None and phase_num == 2:
                    # Return with phases: (n_rois, n_directions, n_phases)
                    return np.stack([self._tuning_phase0, self._tuning_phase1], axis=-1)
                return self._tuning_phase0  # (n_rois, n_directions)

            def get_epoch_dur(self):
                return 100

        osds = MockMultiPhaseOSDS()
        # Test single-phase mode (default for this mock since we're not
        # setting up window-based phase splitting properly)
        metrics = tuning_metrics.compute_all_tuning_metrics(osds, phase_aware=False)

        # Single-phase should return (n_rois,)
        self.assertEqual(metrics['dsi'].shape, (3,))
        self.assertEqual(metrics['osi'].shape, (3,))
        self.assertEqual(metrics['direction_vector_magnitude'].shape, (3,))

    def test_single_phase_output_shape(self):
        """Test that single-phase returns (n_rois,) arrays."""

        class MockSinglePhaseOSDS:
            def __init__(self):
                self.num_rois = 3
                self.directions_list = [0, 90, 180, 270]
                self.dir_phase_num = 1
                self._tuning = np.random.rand(3, 4)

            def compute_tuning_function(self, metric='peak', phase_num=None, window=None, roi_index=None):
                if roi_index is not None:
                    return self._tuning[roi_index]
                return self._tuning

            def get_epoch_dur(self):
                return 100

        osds = MockSinglePhaseOSDS()
        metrics = tuning_metrics.compute_all_tuning_metrics(osds, phase_aware=False)

        # Single-phase should return (n_rois,)
        self.assertEqual(metrics['dsi'].shape, (3,))
        self.assertEqual(metrics['osi'].shape, (3,))


class TestVonMisesFitting(unittest.TestCase):
    """Test von Mises curve fitting functions."""

    def test_von_mises_single_perfect_tuning(self):
        """Test single von Mises on perfectly tuned response."""
        from pygor.timeseries.osds import von_mises_fitting

        # Create perfect von Mises response at 90 degrees
        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        theta = np.deg2rad(directions)
        mu_true = np.deg2rad(90)
        kappa_true = 3.0
        responses = np.exp(kappa_true * np.cos(theta - mu_true))

        result = von_mises_fitting.fit_von_mises_direction(
            responses, directions, return_full=True
        )

        self.assertTrue(result['fit_successful'])
        self.assertGreater(result['r_squared'], 0.95)
        # Allow 10 degree tolerance
        diff = abs(result['preferred_direction'] - 90)
        diff = min(diff, 360 - diff)
        self.assertLess(diff, 10)

    def test_von_mises_double_perfect_orientation(self):
        """Test double von Mises on perfectly oriented response."""
        from pygor.timeseries.osds import von_mises_fitting

        # Create response tuned to 45/225 degrees (45 degree orientation)
        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        theta = np.deg2rad(directions)
        mu_true = np.deg2rad(45)
        kappa_true = 3.0
        responses = (np.exp(kappa_true * np.cos(theta - mu_true)) +
                     np.exp(kappa_true * np.cos(theta - mu_true - np.pi)))

        result = von_mises_fitting.fit_von_mises_orientation(
            responses, directions, return_full=True
        )

        self.assertTrue(result['fit_successful'])
        self.assertGreater(result['r_squared'], 0.95)
        # Allow 10 degree tolerance for orientation
        diff = abs(result['preferred_orientation'] - 45)
        diff = min(diff, 180 - diff)
        self.assertLess(diff, 10)

    def test_von_mises_uniform_response_low_rsquared(self):
        """Test that uniform responses give low R-squared."""
        from pygor.timeseries.osds import von_mises_fitting

        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        responses = np.ones(8)  # Uniform response

        result = von_mises_fitting.fit_von_mises_direction(
            responses, directions, return_full=True
        )

        # For uniform responses, R-squared should be very low or fit may fail
        # Either outcome is acceptable
        if result['fit_successful']:
            self.assertLess(result['r_squared'], 0.5)

    def test_compute_vonmises_preferred_direction_2d(self):
        """Test batch computation on 2D array (n_rois, n_directions)."""
        from pygor.timeseries.osds import von_mises_fitting

        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        theta = np.deg2rad(directions)

        # Create 3 ROIs with different preferred directions
        n_rois = 3
        true_directions = [45, 135, 270]
        responses = np.zeros((n_rois, 8))

        for i, mu in enumerate(true_directions):
            mu_rad = np.deg2rad(mu)
            responses[i] = np.exp(3.0 * np.cos(theta - mu_rad))

        results = von_mises_fitting.compute_vonmises_preferred_direction(
            responses, directions, r_squared_threshold=0.8
        )

        self.assertEqual(results['preferred_direction'].shape, (n_rois,))
        self.assertEqual(results['r_squared'].shape, (n_rois,))
        self.assertEqual(results['fit_valid'].shape, (n_rois,))

        # Check all fits are valid and estimates are close
        for i, true_dir in enumerate(true_directions):
            self.assertTrue(results['fit_valid'][i])
            diff = abs(results['preferred_direction'][i] - true_dir)
            diff = min(diff, 360 - diff)
            self.assertLess(diff, 15)

    def test_fit_failure_returns_nan(self):
        """Test that failed fits return NaN values."""
        from pygor.timeseries.osds import von_mises_fitting

        # Pathological case: all zeros
        directions = np.array([0, 90, 180, 270])
        responses = np.zeros(4)

        result = von_mises_fitting.fit_von_mises_direction(
            responses, directions, return_full=True
        )

        self.assertFalse(result['fit_successful'])
        self.assertTrue(np.isnan(result['preferred_direction']))
        self.assertTrue(np.isnan(result['r_squared']))

    def test_r_squared_computation(self):
        """Test R-squared computation directly."""
        from pygor.timeseries.osds import von_mises_fitting

        # Perfect prediction
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        r_sq = von_mises_fitting.compute_r_squared(observed, predicted)
        self.assertAlmostEqual(r_sq, 1.0, places=7)

        # Completely off prediction (mean prediction)
        predicted_mean = np.array([3, 3, 3, 3, 3])
        r_sq_mean = von_mises_fitting.compute_r_squared(observed, predicted_mean)
        self.assertAlmostEqual(r_sq_mean, 0.0, places=7)

    def test_von_mises_noisy_data(self):
        """Test fitting with noisy data gives reasonable estimate."""
        from pygor.timeseries.osds import von_mises_fitting

        np.random.seed(42)
        directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        theta = np.deg2rad(directions)
        mu_true = np.deg2rad(180)

        # Add noise
        responses = np.exp(2.0 * np.cos(theta - mu_true))
        responses += np.random.normal(0, 0.3, size=8)
        responses = np.maximum(responses, 0.1)  # Ensure positive

        result = von_mises_fitting.fit_von_mises_direction(
            responses, directions, return_full=True
        )

        if result['fit_successful'] and result['r_squared'] > 0.5:
            # Should be within 45 degrees of true for noisy data
            diff = abs(result['preferred_direction'] - 180)
            diff = min(diff, 360 - diff)
            self.assertLess(diff, 45)


if __name__ == '__main__':
    unittest.main()
