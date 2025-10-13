"""
Test utilities for Pygor test suite
Provides helper functions and mock data generation for consistent testing
"""

import numpy as np
import pathlib
import tempfile
import os
from typing import Tuple, Optional


def create_mock_strf_data(n_cells: int = 2, n_time: int = 15, n_y: int = 25, n_x: int = 25, 
                         n_colors: int = 1, signal_strength: float = 5.0, 
                         noise_level: float = 1.0) -> np.ndarray:
    """
    Create mock STRF data with known structure for testing
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_time : int
        Number of time points
    n_y, n_x : int
        Spatial dimensions
    n_colors : int
        Number of color channels
    signal_strength : float
        Amplitude of inserted signals
    noise_level : float
        Standard deviation of background noise
        
    Returns
    -------
    np.ndarray
        Mock STRF data with shape [n_cells * n_colors, n_time, n_y, n_x]
    """
    total_cells = n_cells * n_colors
    strf = np.random.randn(total_cells, n_time, n_y, n_x) * noise_level
    
    # Add some predictable signals for testing
    for cell_idx in range(total_cells):
        # Add a center signal at middle time
        center_y, center_x = n_y // 2, n_x // 2
        center_time = n_time // 2
        strf[cell_idx, center_time, center_y, center_x] = signal_strength
        
        # Add a negative signal offset in space and time
        offset_y = min(center_y + 3, n_y - 1)
        offset_x = min(center_x + 3, n_x - 1) 
        offset_time = min(center_time + 2, n_time - 1)
        strf[cell_idx, offset_time, offset_y, offset_x] = -signal_strength * 0.8
        
        # Add some weaker signals for threshold testing
        weak_y, weak_x = max(center_y - 3, 0), max(center_x - 3, 0)
        weak_time = max(center_time - 2, 0)
        strf[cell_idx, weak_time, weak_y, weak_x] = signal_strength * 0.4
        
    return strf


def create_mock_core_data_dict() -> dict:
    """
    Create mock data dictionary simulating HDF5 file contents
    
    Returns
    -------
    dict
        Dictionary with typical Core data attributes
    """
    n_time, n_y, n_x = 100, 50, 50
    n_rois = 5
    
    mock_data = {
        'images': np.random.randn(n_time, n_y, n_x) * 100 + 1000,  # Typical fluorescence values
        'rois': create_mock_roi_mask(n_y, n_x, n_rois),
        'averages': np.random.randn(n_rois, 20, 10) * 50 + 500,  # [roi, time, repetition]
        'traces_raw': np.random.randn(n_rois, n_time) * 100 + 1000,
        'traces_znorm': np.random.randn(n_rois, n_time),
        'triggertimes': np.sort(np.random.uniform(0, n_time * 0.1, 20)),  # Sorted trigger times
        'triggertimes_frame': np.sort(np.random.randint(0, n_time, 20)),
        'frame_hz': 10.0,  # Typical frame rate
        'metadata': {
            'exp_date': '2024-01-01',
            'experimenter': 'test_user',
            'stimulus_type': 'test_stimulus'
        }
    }
    
    return mock_data


def create_mock_roi_mask(n_y: int, n_x: int, n_rois: int) -> np.ndarray:
    """
    Create mock ROI mask with circular ROIs
    
    Parameters
    ----------
    n_y, n_x : int
        Spatial dimensions
    n_rois : int
        Number of ROIs to create
        
    Returns
    -------
    np.ndarray
        ROI mask with negative values for ROIs, 1 for background
    """
    roi_mask = np.ones((n_y, n_x))  # Background = 1
    
    # Create circular ROIs
    radius = min(n_y, n_x) // (2 * int(np.sqrt(n_rois)) + 1)
    
    for roi_idx in range(n_rois):
        # Distribute ROIs across image
        row = (roi_idx // int(np.sqrt(n_rois))) * (n_y // int(np.sqrt(n_rois))) + n_y // (2 * int(np.sqrt(n_rois)))
        col = (roi_idx % int(np.sqrt(n_rois))) * (n_x // int(np.sqrt(n_rois))) + n_x // (2 * int(np.sqrt(n_rois)))
        
        # Create circular ROI
        y, x = np.ogrid[:n_y, :n_x]
        mask = (y - row)**2 + (x - col)**2 <= radius**2
        roi_mask[mask] = -(roi_idx + 2)  # ROIs are negative, starting from -2
        
    return roi_mask


def assert_array_properties(test_case, array: np.ndarray, expected_shape: Optional[Tuple] = None,
                           expected_dtype: Optional[type] = None, allow_nan: bool = True,
                           finite_range: Optional[Tuple[float, float]] = None):
    """
    Helper function to assert common array properties
    
    Parameters
    ----------
    test_case : unittest.TestCase
        Test case instance for assertions
    array : np.ndarray
        Array to validate
    expected_shape : tuple, optional
        Expected shape of array
    expected_dtype : type, optional
        Expected data type
    allow_nan : bool
        Whether NaN values are acceptable
    finite_range : tuple, optional
        Expected range (min, max) for finite values
    """
    test_case.assertIsInstance(array, np.ndarray, "Should return numpy array")
    
    if expected_shape is not None:
        test_case.assertEqual(array.shape, expected_shape, f"Array shape should be {expected_shape}")
        
    if expected_dtype is not None:
        test_case.assertEqual(array.dtype, expected_dtype, f"Array dtype should be {expected_dtype}")
        
    if not allow_nan:
        test_case.assertFalse(np.any(np.isnan(array)), "Array should not contain NaN values")
        
    # Check for infinite values
    test_case.assertFalse(np.any(np.isinf(array)), "Array should not contain infinite values")
    
    if finite_range is not None:
        finite_values = array[np.isfinite(array)]
        if len(finite_values) > 0:
            test_case.assertGreaterEqual(np.min(finite_values), finite_range[0], 
                                       f"Finite values should be >= {finite_range[0]}")
            test_case.assertLessEqual(np.max(finite_values), finite_range[1],
                                    f"Finite values should be <= {finite_range[1]}")


def create_temporary_h5_file(data_dict: dict) -> str:
    """
    Create temporary HDF5 file for testing
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of data to write to file
        
    Returns
    -------
    str
        Path to temporary file
    """
    import h5py
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_file.close()
    
    # Write data to HDF5 file
    with h5py.File(temp_file.name, 'w') as f:
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (like metadata)
                grp = f.create_group(key)
                for subkey, subvalue in value.items():
                    grp.attrs[subkey] = subvalue
            else:
                f.create_dataset(key, data=value)
                
    return temp_file.name


def cleanup_temp_file(filepath: str):
    """Remove temporary file"""
    if os.path.exists(filepath):
        os.remove(filepath)


class MockSTRFObject:
    """Mock STRF object for testing without requiring full data loading"""
    
    def __init__(self, n_cells: int = 2, n_colors: int = 3, **kwargs):
        self.strfs = create_mock_strf_data(n_cells=n_cells, n_colors=n_colors, **kwargs)
        self.multicolour = n_colors > 1
        self.numcolour = n_colors
        self.num_rois = n_cells
        
    def get_strf_shape(self):
        return self.strfs.shape


def validate_strf_axis_convention(test_case, strf_array: np.ndarray):
    """
    Validate that STRF array follows [cell, time, y, x] convention
    
    Parameters
    ----------
    test_case : unittest.TestCase
        Test case for assertions
    strf_array : np.ndarray
        STRF array to validate
    """
    test_case.assertEqual(len(strf_array.shape), 4, "STRF should be 4D array")
    
    n_cells, n_time, n_y, n_x = strf_array.shape
    test_case.assertGreater(n_cells, 0, "Should have at least one cell")
    test_case.assertGreater(n_time, 0, "Should have at least one time point")
    test_case.assertGreater(n_y, 0, "Should have spatial Y dimension")
    test_case.assertGreater(n_x, 0, "Should have spatial X dimension")
    
    # Typical constraints for retinal data
    test_case.assertLess(n_time, min(n_y, n_x), 
                        "Time dimension should typically be smaller than spatial dimensions")
    test_case.assertLess(n_time, 100, "Time dimension should be reasonable (< 100 frames)")
    test_case.assertGreater(n_time, 3, "Time dimension should be > 3 frames for meaningful analysis")