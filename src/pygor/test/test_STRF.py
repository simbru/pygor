import pygor.load
import pygor.data_helpers
import pygor.utils.helpinfo
import unittest
import warnings
import os
import pathlib
from contextlib import redirect_stdout
import atexit
import numpy as np
import pandas as pd
import matplotlib.figure

file_loc = pathlib.Path(__file__).parents[3]
example_data = file_loc.joinpath(r"examples/strf_demo_data.h5")
out_loc = r"src/pygor/test/test_out_Core.txt"
bs_bool = False

class TestSTRF(unittest.TestCase):
    strfs = pygor.load.STRF(example_data)

    def test_contours(self):
        self.strfs.fit_contours()
    
    def test_attributes_return(self):
        attr_list = pygor.utils.helpinfo.get_attribute_list(self.strfs, with_types=False)
        [getattr(self.strfs, i) for i in attr_list]

    def test_simple_methods_return(self):
        meth_list = pygor.utils.helpinfo.get_methods_list(self.strfs, with_returns=False)
        bs_refs = [i for i in meth_list if "bootstrap" in i or "bs" in i]
        # Only exclude methods that genuinely require interactive input
        ignore = ["draw_rois", "get_depth"]
        # Methods that require parameters or special setup should be tested separately
        requires_params = ["napari_strfs", "plot_averages", "view_stack_projection", "view_stack_rois", "view_drift"]
        # Methods that require bootstrap data to be available
        requires_bootstrap = ["get_pvals_table"]
        meth_set = set(meth_list) - set(bs_refs) - set(ignore) - set(requires_params) - set(requires_bootstrap)
        
        print("Testing simple methods:")
        failed_methods = []
        for method_name in meth_set:
            print(f"- {method_name}")    
            if "plot" not in method_name and "play" not in method_name:
                try:
                    result = getattr(self.strfs, method_name)()
                    # Validate that method returns something reasonable
                    if result is not None:
                        valid_types = (np.ndarray, list, dict, tuple, int, float, bool, str, 
                                     pd.DataFrame, pd.Series, matplotlib.figure.Figure, 
                                     pathlib.Path, type(None))
                        self.assertTrue(isinstance(result, valid_types), 
                                       f"Method {method_name} returned unexpected type: {type(result)}")
                except AttributeError as e:
                    failed_methods.append(f"Method {method_name} gave AttributeError: {e}")
                except TypeError as e:
                    # Only acceptable if method requires parameters
                    if "required positional argument" in str(e) or "missing" in str(e).lower():
                        continue  # This is expected for methods requiring parameters
                    else:
                        failed_methods.append(f"Method {method_name} gave unexpected TypeError: {e}")
                except Exception as e:
                    failed_methods.append(f"Method {method_name} failed with {type(e).__name__}: {e}")
        
        # Fail the test if any methods had unexpected errors
        if failed_methods:
            self.fail(f"Methods failed unexpectedly:\n" + "\n".join(failed_methods))

    def test_saveload(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "test.pkl"
        try:
            self.strfs.save_pkl(dir_path, filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Error in storing .pkl file during test_saveload."
            ) from e

        finally:
            os.remove(pathlib.Path(dir_path, filename))

    def test_bs(self):
        self.strfs.set_bootstrap_bool(True)
        settings = self.strfs.get_bootstrap_settings()
        self.assertIsInstance(settings, dict)
        
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n = 10, time_bs_n = 10)
        self.strfs.update_bootstrap_settings(new_bs_dict)
        
        # Validate bootstrap was updated
        updated_settings = self.strfs.get_bootstrap_settings()
        self.assertEqual(updated_settings['space_bs_n'], 10)
        self.assertEqual(updated_settings['time_bs_n'], 10)
        
        # Run bootstrap and validate results
        self.strfs.run_bootstrap()
        # Bootstrap should create some results - check they exist
        self.assertTrue(hasattr(self.strfs, 'bs_bool'))
        self.assertTrue(self.strfs.bs_bool)

    def test_get_help(self):
        write_to = file_loc.joinpath(out_loc)
        with open(write_to, 'w') as f:
            with redirect_stdout(f):
                self.strfs.get_help(hints = True, types = True)

    def test_map_extrema_timing(self):
        """Test the new extrema timing analysis method"""
        # Test with default parameters - all STRFs
        timing_map = self.strfs.map_extrema_timing()
        
        # Validate output shape and type
        self.assertIsInstance(timing_map, np.ndarray)
        self.assertEqual(len(timing_map.shape), 3)  # Should be 3D array [n_strfs, y, x]
        
        # Check that shape matches STRF spatial dimensions
        expected_shape = (self.strfs.strfs.shape[0], self.strfs.strfs.shape[2], self.strfs.strfs.shape[3])
        self.assertEqual(timing_map.shape, expected_shape)
        
        # Test with specific parameters
        timing_map_thresh = self.strfs.map_extrema_timing(threshold=2.0)
        self.assertIsInstance(timing_map_thresh, np.ndarray)
        self.assertEqual(timing_map_thresh.shape, expected_shape)
        
        # Test with single STRF selection
        timing_map_single = self.strfs.map_extrema_timing(roi=0)
        self.assertIsInstance(timing_map_single, np.ndarray)
        self.assertEqual(len(timing_map_single.shape), 2)  # Should be 2D for single STRF
        
        # Test with specific STRF indices
        n_strfs = self.strfs.strfs.shape[0]
        if n_strfs > 1:
            timing_map_last = self.strfs.map_extrema_timing(roi=n_strfs-1)
            self.assertIsInstance(timing_map_last, np.ndarray)
            self.assertEqual(len(timing_map_last.shape), 2)
        
        # Test with invalid STRF index
        with self.assertRaises(IndexError):
            self.strfs.map_extrema_timing(roi=n_strfs)  # Should be out of bounds

    def test_compute_spatial_overlap_metrics(self):
        """Test the spatial overlap analysis method"""
        # Skip if not multicolor data
        if not (hasattr(self.strfs, 'multicolour') and self.strfs.multicolour):
            self.skipTest("Spatial overlap test requires multicolor data")
        
        overlap_metrics = self.strfs.compute_spatial_overlap_metrics()
        
        # Validate output structure
        self.assertIsInstance(overlap_metrics, dict)
        expected_keys = ['correlation', 'jaccard_index', 'centroid_distance', 'offset_pixels']
        for key in expected_keys:
            self.assertIn(key, overlap_metrics)
            self.assertIsInstance(overlap_metrics[key], np.ndarray)
        
        # Test with specific threshold
        overlap_metrics_thresh = self.strfs.compute_spatial_overlap_metrics(threshold=2.0)
        self.assertIsInstance(overlap_metrics_thresh, dict)

    def test_analyze_multicolor_spatial_alignment(self):
        """Test the comprehensive multicolor alignment analysis"""
        # Skip if not multicolor data
        if not (hasattr(self.strfs, 'multicolour') and self.strfs.multicolour):
            self.skipTest("Multicolor alignment test requires multicolor data")
        
        alignment_results = self.strfs.analyze_multicolor_spatial_alignment()
        
        # Validate output structure
        self.assertIsInstance(alignment_results, dict)
        expected_keys = ['pairwise_metrics', 'summary_stats', 'channel_centroids']
        for key in expected_keys:
            self.assertIn(key, alignment_results)
        
        # Validate pairwise metrics structure
        pairwise = alignment_results['pairwise_metrics']
        self.assertIsInstance(pairwise, dict)
        
        # Validate summary stats
        summary = alignment_results['summary_stats']
        self.assertIsInstance(summary, dict)
        self.assertIn('mean_correlation', summary)
        self.assertIn('mean_jaccard', summary)

    def test_strf_array_structure(self):
        """Test STRF array follows expected [cell, time, y, x] convention"""
        self.assertEqual(len(self.strfs.strfs.shape), 4)
        
        # Validate axes are in correct order
        n_cells, n_time, n_y, n_x = self.strfs.strfs.shape
        self.assertGreater(n_cells, 0)
        self.assertGreater(n_time, 0) 
        self.assertGreater(n_y, 0)
        self.assertGreater(n_x, 0)
        
        # Time dimension should typically be smallest (10-50 frames)
        # Spatial dimensions should be larger (typically 20-100 pixels)
        self.assertLess(n_time, min(n_y, n_x), "Time dimension should be smaller than spatial dimensions")

    def test_threshold_masking_behavior(self):
        """Test that threshold masking works correctly"""
        timing_map_low = self.strfs.map_extrema_timing(threshold=1.0)
        timing_map_high = self.strfs.map_extrema_timing(threshold=5.0)
        
        # Higher threshold should result in more NaN values
        nan_count_low = np.sum(np.isnan(timing_map_low))
        nan_count_high = np.sum(np.isnan(timing_map_high))
        self.assertGreaterEqual(nan_count_high, nan_count_low, 
                               "Higher threshold should create more NaN values")

    def test_pvals_table_with_bootstrap(self):
        """Test get_pvals_table method after setting up bootstrap data"""
        # First set up bootstrap
        self.strfs.set_bootstrap_bool(True)
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n=5, time_bs_n=5)  # Small values for test speed
        self.strfs.update_bootstrap_settings(new_bs_dict)
        self.strfs.run_bootstrap()
        
        # Now test get_pvals_table
        result = self.strfs.get_pvals_table()
        self.assertIsInstance(result, pd.DataFrame, "get_pvals_table should return a DataFrame")
        self.assertGreater(len(result), 0, "DataFrame should not be empty")
        
        # Check that it has expected columns
        if self.strfs.multicolour:
            expected_cols = ["space_R", "space_G", "space_B", "space_UV", "time_R", "time_G", "time_B", "time_UV",
                           "sig_R", "sig_G", "sig_B", "sig_UV", "sig_any"]
        else:
            expected_cols = ["space", "time", "sig"]
        
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Expected column {col} not found in DataFrame")

class TestSTRF_plot(unittest.TestCase):
    strfs = pygor.load.STRF(example_data)

    def find_plot_methods(self):
        plot_list = pygor.utils.helpinfo.get_methods_list(self.strfs, with_returns=False)
        plot_list = [i for i in plot_list if "plot" in i]
        print("Found plot methods:", plot_list)
        disallowed = ["plot_averages"]
        if any(i in plot_list for i in disallowed):
            print("Found disallowed plot methods:", set(plot_list) & set(disallowed))
            print("Ignoring these.")
        plot_set = set(plot_list) - set(disallowed)
        # getattr(strfs, i)()
        return plot_set
    
    def test_timecourse(self):
        self.strfs.plot_timecourse(0)
    
    def test_chromatic_overview(self):
        self.strfs.plot_chromatic_overview()
if os.path.exists(file_loc.joinpath(out_loc)):
    atexit.register(lambda : os.remove(file_loc.joinpath(out_loc)))

if __name__ == "__main__":
    unittest.main()
