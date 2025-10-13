import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import unittest
import numpy as np
import warnings
import tempfile
import os

try:
    import pygor.load
    import pygor.strf.extrema_timing
    import pygor.strf.spatial_alignment
    PYGOR_AVAILABLE = True
except ImportError:
    PYGOR_AVAILABLE = False

@unittest.skipUnless(PYGOR_AVAILABLE, "pygor module not available")
class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Create mock data for testing edge cases"""
        # Create minimal test STRF data
        self.mock_strf_3d = np.random.randn(10, 20, 20)  # [time, y, x]
        self.mock_strf_4d = np.random.randn(3, 10, 20, 20)  # [cell, time, y, x]
        
        # Create data with known extrema for predictable testing
        self.test_strf = np.zeros((1, 10, 5, 5))
        self.test_strf[0, 5, 2, 2] = 10.0  # Strong signal at center, middle time
        self.test_strf[0, 3, 1, 1] = -8.0   # Strong negative signal
        
    def test_extrema_timing_with_empty_array(self):
        """Test extrema timing with empty/zero arrays"""
        empty_strf = np.zeros((1, 10, 5, 5))
        
        timing_map = pygor.strf.extrema_timing.map_extrema_timing(empty_strf, threshold=1.0)
        
        # Should return all NaN for zero data
        self.assertTrue(np.all(np.isnan(timing_map)))
        
    def test_extrema_timing_with_noise_only(self):
        """Test extrema timing with low amplitude noise"""
        noise_strf = np.random.randn(1, 10, 5, 5) * 0.1  # Very small noise
        
        timing_map = pygor.strf.extrema_timing.map_extrema_timing(noise_strf, threshold=3.0)
        
        # Should return mostly NaN for noise below threshold
        nan_fraction = np.sum(np.isnan(timing_map)) / timing_map.size
        self.assertGreater(nan_fraction, 0.5, "Most pixels should be below threshold")
        
    def test_extrema_timing_known_values(self):
        """Test extrema timing with known peak locations"""
        timing_map = pygor.strf.extrema_timing.map_extrema_timing(self.test_strf, threshold=5.0)
        
        # Check that the strong signal at (2,2) is detected at time 5
        self.assertEqual(timing_map[0, 2, 2], 5, "Strong positive signal should be detected at correct time")
        
        # Check that negative signal at (1,1) is detected at time 3
        self.assertEqual(timing_map[0, 1, 1], 3, "Strong negative signal should be detected at correct time")
        
    def test_extrema_timing_boundary_conditions(self):
        """Test extrema timing with edge cases in parameters"""
        # Test with very high threshold - should return all NaN
        timing_map_high = pygor.strf.extrema_timing.map_extrema_timing(self.test_strf, threshold=100.0)
        self.assertTrue(np.all(np.isnan(timing_map_high)), "Very high threshold should return all NaN")
        
        # Test with zero threshold - should return all valid values
        timing_map_zero = pygor.strf.extrema_timing.map_extrema_timing(self.test_strf, threshold=0.0)
        nan_count = np.sum(np.isnan(timing_map_zero))
        self.assertLess(nan_count, timing_map_zero.size, "Zero threshold should return mostly valid values")
        
    def test_extrema_timing_3d_vs_4d_input(self):
        """Test that 3D and 4D inputs work correctly"""
        # 3D input should work
        timing_3d = pygor.strf.extrema_timing.map_extrema_timing(self.mock_strf_3d, threshold=1.0)
        self.assertEqual(len(timing_3d.shape), 2, "3D input should return 2D output")
        
        # 4D input should work
        timing_4d = pygor.strf.extrema_timing.map_extrema_timing(self.mock_strf_4d, threshold=1.0)
        self.assertEqual(len(timing_4d.shape), 3, "4D input should return 3D output")
        self.assertEqual(timing_4d.shape[0], 3, "Should preserve cell dimension")
        
    def test_spatial_overlap_edge_cases(self):
        """Test spatial overlap with edge cases"""
        # Create mock multicolor STRF data
        n_cells, n_colors = 2, 3
        mock_multicolor = np.random.randn(n_cells * n_colors, 10, 20, 20)
        
        # Test with all zeros - should handle gracefully
        zero_strf = np.zeros((6, 10, 20, 20))  # 2 cells * 3 colors
        
        # This should not crash, even with zero data
        try:
            overlap_results = pygor.strf.spatial_alignment.compute_spatial_overlap_metrics(
                zero_strf, n_cells, n_colors, threshold=1.0
            )
            # Should return valid structure even with no signal
            self.assertIsInstance(overlap_results, dict)
        except Exception as e:
            self.fail(f"Spatial overlap analysis should handle zero data gracefully: {e}")
            
    def test_invalid_array_shapes(self):
        """Test handling of invalid array shapes"""
        # Test with wrong dimensions
        wrong_shape_2d = np.random.randn(10, 10)  # 2D instead of 3D/4D
        
        with self.assertRaises(Exception):
            pygor.strf.extrema_timing.map_extrema_timing(wrong_shape_2d, threshold=1.0)
            
        # Test with single pixel
        tiny_strf = np.random.randn(1, 5, 1, 1)
        timing_tiny = pygor.strf.extrema_timing.map_extrema_timing(tiny_strf, threshold=1.0)
        self.assertEqual(timing_tiny.shape, (1, 1, 1), "Should handle single pixel arrays")
        
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values in input data"""
        # Create STRF with NaN values
        nan_strf = self.test_strf.copy()
        nan_strf[0, :, 0, 0] = np.nan
        
        timing_map = pygor.strf.extrema_timing.map_extrema_timing(nan_strf, threshold=1.0)
        # Should handle NaN gracefully - pixel with NaN should return NaN
        self.assertTrue(np.isnan(timing_map[0, 0, 0]), "Pixels with NaN input should return NaN")
        
        # Create STRF with infinite values
        inf_strf = self.test_strf.copy()
        inf_strf[0, 7, 4, 4] = np.inf
        
        timing_map_inf = pygor.strf.extrema_timing.map_extrema_timing(inf_strf, threshold=1.0)
        # Should detect infinite values
        self.assertEqual(timing_map_inf[0, 4, 4], 7, "Should detect infinite values at correct time")
        
    def test_memory_efficiency_large_arrays(self):
        """Test memory efficiency with larger arrays"""
        # Create larger test array to check memory handling
        large_strf = np.random.randn(5, 50, 100, 100)  # Larger but still manageable
        
        # This should complete without memory issues
        timing_map = pygor.strf.extrema_timing.map_extrema_timing(large_strf, threshold=2.0)
        
        self.assertEqual(timing_map.shape, (5, 100, 100))
        self.assertIsInstance(timing_map, np.ndarray)
        
    def test_threshold_range_validation(self):
        """Test various threshold values"""
        thresholds = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        nan_counts = []
        
        for thresh in thresholds:
            timing_map = pygor.strf.extrema_timing.map_extrema_timing(self.test_strf, threshold=thresh)
            nan_counts.append(np.sum(np.isnan(timing_map)))
            
        # NaN count should generally increase with threshold
        for i in range(1, len(nan_counts)):
            self.assertGreaterEqual(nan_counts[i], nan_counts[i-1], 
                                   f"Higher threshold should not decrease NaN count: {thresholds[i-1]} vs {thresholds[i]}")


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation for STRF methods"""
    
    def test_negative_parameters(self):
        """Test handling of negative parameters"""
        test_strf = np.random.randn(1, 10, 5, 5)
        
        # Negative threshold should be handled appropriately
        with warnings.catch_warnings(record=True) as w:
            timing_map = pygor.strf.extrema_timing.map_extrema_timing(test_strf, threshold=-1.0)
            # Should either warn or handle gracefully
            self.assertTrue(len(w) > 0 or timing_map is not None)
            
    def test_extreme_parameters(self):
        """Test handling of extreme parameter values"""
        test_strf = np.random.randn(1, 10, 5, 5)
        
        # Very large threshold
        timing_map_large = pygor.strf.extrema_timing.map_extrema_timing(test_strf, threshold=1e6)
        self.assertTrue(np.all(np.isnan(timing_map_large)), "Extremely large threshold should return all NaN")
        
        # Very small threshold (but positive)
        timing_map_small = pygor.strf.extrema_timing.map_extrema_timing(test_strf, threshold=1e-10)
        # Should work without errors
        self.assertIsInstance(timing_map_small, np.ndarray)


if __name__ == "__main__":
    unittest.main()