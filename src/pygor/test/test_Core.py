import pygor.load
import pygor.utils.helpinfo
import os
import numpy as np
import pandas as pd
import matplotlib.figure
import unittest
import warnings
from contextlib import redirect_stdout
import atexit

file_loc = pathlib.Path(__file__).parents[3]
example_data = file_loc.joinpath(r"examples/strf_demo_data.h5")
out_loc = r"src/pygor/test/test_out_Core.txt"
data = pygor.load.Core(example_data)


class TestCore(unittest.TestCase):
    def test_averages_type(self):
        self.assertTrue(isinstance(data.averages, np.ndarray) or data.averages is None)

    def test_ipl_depths_type(self):
        self.assertTrue(
            isinstance(data.ipl_depths, np.ndarray) or data.ipl_depths is None
        )

    def test_metadata_type(self):
        self.assertTrue(isinstance(data.metadata, dict))

    def test_num_rois_type(self):
        self.assertTrue(isinstance(data.num_rois, int))

    def test_rois_type(self):
        self.assertTrue(isinstance(data.rois, np.ndarray))

    def test_attributes_return(self):
        attr_list = pygor.utils.helpinfo.get_attribute_list(data, with_types=False)
        [getattr(data, i) for i in attr_list]

    def test_simple_methods_return(self):
        meth_list = pygor.utils.helpinfo.get_methods_list(data, with_returns=False)
        # Methods that require interactive input or parameters
        ignore = ["draw_rois", "get_depth", "view_stack_projection", "view_stack_rois", 
                 "view_drift", "plot_averages", "calculate_image_average"]
        
        write_to = file_loc.joinpath(out_loc)
        failed_methods = []
        
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                for method_name in meth_list:
                    if method_name not in ignore and "plot" not in method_name:
                        try:
                            result = getattr(data, method_name)()
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
                                continue  # Expected for methods requiring parameters
                            else:
                                failed_methods.append(f"Method {method_name} gave unexpected TypeError: {e}")
                        except Exception as e:
                            failed_methods.append(f"Method {method_name} failed with {type(e).__name__}: {e}")
        
        # Fail the test if any methods had unexpected errors
        if failed_methods:
            self.fail(f"Methods failed unexpectedly:\n" + "\n".join(failed_methods))

    def test_get_help(self):
        write_to = file_loc.joinpath(out_loc)
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                data.get_help(hints = True, types = True)

    def test_core_data_structure(self):
        """Test that core data follows expected structures"""
        # Test required attributes exist
        required_attrs = ['filename', 'metadata', 'rois', 'type', 'frame_hz', 'num_rois']
        for attr in required_attrs:
            self.assertTrue(hasattr(data, attr), f"Missing required attribute: {attr}")
        
        # Test data types are correct
        if data.averages is not None:
            self.assertEqual(len(data.averages.shape), 3, "Averages should be 3D array [roi, time, repetition] or similar")
        
        if data.images is not None:
            self.assertEqual(len(data.images.shape), 3, "Images should be 3D array [time, y, x]")
            self.assertGreater(data.images.shape[0], 0, "Images should have time dimension > 0")
        
        # Test ROI data consistency
        if data.rois is not None:
            unique_rois = np.unique(data.rois)
            unique_rois = unique_rois[~np.isnan(unique_rois)]  # Remove NaN
            self.assertEqual(len(unique_rois) - 1, data.num_rois, "num_rois should match actual ROI count")

    def test_metadata_structure(self):
        """Test metadata contains expected information"""
        self.assertIsInstance(data.metadata, dict)
        # Check for typical metadata keys (these may vary by dataset)
        expected_keys = ['exp_date']  # At minimum should have experiment date
        for key in expected_keys:
            if key in data.metadata:
                self.assertIsNotNone(data.metadata[key])

    def test_roi_properties(self):
        """Test ROI-related properties work correctly"""
        if hasattr(data, 'roi_centroids'):
            centroids = data.roi_centroids
            if centroids is not None:
                self.assertIsInstance(centroids, np.ndarray)
                self.assertEqual(len(centroids.shape), 2, "Centroids should be 2D array [roi, coordinates]")
                self.assertEqual(centroids.shape[1], 2, "Each centroid should have 2 coordinates (y, x)")

    def test_correlation_map(self):
        """Test correlation map calculation"""
        if hasattr(data, 'get_correlation_map') and data.images is not None:
            corr_map = data.get_correlation_map()
            self.assertIsInstance(corr_map, np.ndarray)
            # Correlation map should have same spatial dimensions as images
            self.assertEqual(corr_map.shape, data.images.shape[1:])

    def test_frame_rate_calculation(self):
        """Test frame rate is reasonable"""
        if hasattr(data, 'frame_hz') and data.frame_hz is not None:
            self.assertGreater(data.frame_hz, 0, "Frame rate should be positive")
            self.assertLess(data.frame_hz, 1000, "Frame rate should be reasonable (< 1000 Hz)")

    def test_trigger_timing(self):
        """Test trigger timing data consistency"""
        if hasattr(data, 'triggertimes') and data.triggertimes is not None:
            self.assertIsInstance(data.triggertimes, np.ndarray)
            # Trigger times should be monotonically increasing
            if len(data.triggertimes) > 1:
                diff = np.diff(data.triggertimes)
                self.assertTrue(np.all(diff >= 0), "Trigger times should be monotonically increasing")
        
        if hasattr(data, 'triggertimes_frame') and data.triggertimes_frame is not None:
            self.assertIsInstance(data.triggertimes_frame, np.ndarray)
            # Frame trigger times should be integers
            self.assertTrue(np.all(data.triggertimes_frame == data.triggertimes_frame.astype(int)),
                           "Frame trigger times should be integers")
if os.path.exists(file_loc.joinpath(out_loc)):
    atexit.register(lambda : os.remove(file_loc.joinpath(out_loc)))

if __name__ == "__main__":
    unittest.main()
