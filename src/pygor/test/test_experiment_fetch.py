import unittest
import numpy as np
from unittest.mock import MagicMock
from collections import defaultdict

from pygor.classes.experiment import Experiment


class MockRecording:
    """Mock recording object for testing Experiment.fetch methods."""

    def __init__(self, name, rec_type, num_rois):
        self.name = name
        self.type = rec_type
        self.num_rois = num_rois
        self.metadata = {
            'filename': f'/path/to/{name}.h5',
            'exp_date': MagicMock(strftime=lambda x: '01-01-2024')
        }
        self._dsi_data = np.random.rand(num_rois)
        self._depths_data = np.random.rand(num_rois) * 100

    def get_dsi(self, metric='peak', phase_idx=None, **kwargs):
        """Return mock DSI data."""
        return self._dsi_data

    def get_osi(self, metric='peak', phase_idx=None, **kwargs):
        """Return mock OSI data."""
        return np.random.rand(self.num_rois)

    @property
    def ipl_depths(self):
        return self._depths_data


class MockOSDSRecording(MockRecording):
    """Mock OSDS recording with multi-phase support."""

    def __init__(self, name, num_rois, n_phases=2):
        super().__init__(name, 'OSDS', num_rois)
        self.n_phases = n_phases
        self._dsi_multiphase = np.random.rand(n_phases, num_rois)

    def get_dsi(self, metric='peak', phase_idx=None, **kwargs):
        """Return mock DSI data with phase support."""
        if phase_idx is not None:
            return self._dsi_multiphase[phase_idx]
        return self._dsi_multiphase


class TestFetchRaw(unittest.TestCase):
    """Test Experiment.fetch_raw() method."""

    def setUp(self):
        """Create test experiment with mock recordings."""
        self.rec1 = MockRecording('rec1', 'OSDS', num_rois=10)
        self.rec2 = MockRecording('rec2', 'OSDS', num_rois=15)
        self.rec3 = MockRecording('rec3', 'STRF', num_rois=8)

        self.exp = Experiment(recording=[self.rec1, self.rec2, self.rec3])

    def test_fetch_raw_returns_list(self):
        """Test that fetch_raw always returns a list."""
        result = self.exp.fetch_raw('get_dsi')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_fetch_raw_preserves_shapes(self):
        """Test that fetch_raw preserves original result shapes."""
        result = self.exp.fetch_raw('get_dsi')
        self.assertEqual(result[0].shape, (10,))
        self.assertEqual(result[1].shape, (15,))
        self.assertEqual(result[2].shape, (8,))

    def test_fetch_raw_handles_missing_method(self):
        """Test that fetch_raw handles recordings without the method."""
        # Create a recording without the method
        class MinimalRecording:
            def __init__(self):
                self.name = 'minimal'
                self.type = 'Minimal'
                self.num_rois = 5
                self.metadata = {
                    'filename': '/path/to/minimal.h5',
                    'exp_date': MagicMock(strftime=lambda x: '01-01-2024')
                }

        minimal = MinimalRecording()
        exp = Experiment(recording=[self.rec1, minimal])

        result = exp.fetch_raw('get_dsi')
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])  # rec1 has get_dsi
        self.assertIsNone(result[1])  # minimal doesn't have get_dsi

    def test_fetch_raw_passes_kwargs(self):
        """Test that fetch_raw passes kwargs to methods."""
        result = self.exp.fetch_raw('get_dsi', metric='mean')
        self.assertEqual(len(result), 3)

    def test_fetch_raw_with_property(self):
        """Test that fetch_raw works with properties (not just methods)."""
        result = self.exp.fetch_raw('ipl_depths')
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertIsInstance(r, np.ndarray)


class TestFetchConcat(unittest.TestCase):
    """Test Experiment.fetch_concat() method."""

    def setUp(self):
        """Create test experiment with mock recordings."""
        self.rec1 = MockRecording('rec1', 'OSDS', num_rois=10)
        self.rec2 = MockRecording('rec2', 'OSDS', num_rois=15)
        self.rec3 = MockRecording('rec3', 'STRF', num_rois=8)

        self.exp = Experiment(recording=[self.rec1, self.rec2, self.rec3])

    def test_fetch_concat_returns_array(self):
        """Test that fetch_concat returns concatenated array."""
        result = self.exp.fetch_concat('ipl_depths')
        self.assertIsInstance(result, np.ndarray)
        # Should have 10 + 15 + 8 = 33 total ROIs
        self.assertEqual(result.shape, (33,))

    def test_fetch_concat_with_type_filter_string(self):
        """Test that type_filter works with string input."""
        result = self.exp.fetch_concat('get_dsi', type_filter='OSDS')
        # Should only include OSDS recordings: 10 + 15 = 25 ROIs
        self.assertEqual(result.shape[0], 25)

    def test_fetch_concat_with_type_filter_list(self):
        """Test that type_filter accepts list of types."""
        result = self.exp.fetch_concat('ipl_depths', type_filter=['OSDS', 'STRF'])
        self.assertEqual(result.shape, (33,))

    def test_fetch_concat_returns_none_if_all_fail(self):
        """Test that fetch_concat returns None if all results fail."""
        result = self.exp.fetch_concat('nonexistent_method')
        self.assertIsNone(result)

    def test_fetch_concat_passes_kwargs(self):
        """Test that kwargs are passed through."""
        result = self.exp.fetch_concat('get_dsi', metric='mean', type_filter='OSDS')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 25)

    def test_fetch_concat_with_phase_idx(self):
        """Test fetch_concat with phase_idx parameter for OSDS data."""
        osds1 = MockOSDSRecording('osds1', num_rois=10, n_phases=2)
        osds2 = MockOSDSRecording('osds2', num_rois=15, n_phases=2)
        exp = Experiment(recording=[osds1, osds2])

        # Without phase_idx, would get (2, n_rois) per recording - can't concat
        # With phase_idx, get (n_rois,) per recording - can concat
        result = exp.fetch_concat('get_dsi', phase_idx=0)
        self.assertEqual(result.shape, (25,))

        result = exp.fetch_concat('get_dsi', phase_idx=1)
        self.assertEqual(result.shape, (25,))


class TestFetchConcatShapeMismatch(unittest.TestCase):
    """Test fetch_concat error handling for shape mismatches."""

    def test_raises_on_shape_mismatch(self):
        """Test that incompatible shapes raise ValueError with helpful message."""

        class Recording2D:
            def __init__(self, name, shape):
                self.name = name
                self.type = 'Test'
                self.num_rois = shape[0]
                self.metadata = {
                    'filename': f'/path/to/{name}.h5',
                    'exp_date': MagicMock(strftime=lambda x: '01-01-2024')
                }
                self._data = np.random.rand(*shape)

            def get_data(self):
                return self._data

        # One recording returns (5, 3), another returns (5, 4) - can't concat
        rec1 = Recording2D('rec1', (5, 3))
        rec2 = Recording2D('rec2', (5, 4))
        exp = Experiment(recording=[rec1, rec2])

        with self.assertRaises(ValueError) as ctx:
            exp.fetch_concat('get_data', axis=0)

        self.assertIn('Cannot concatenate', str(ctx.exception))
        self.assertIn('phase_idx', str(ctx.exception))


class TestFetchBackwardCompatibility(unittest.TestCase):
    """Test that existing fetch() behavior is unchanged."""

    def setUp(self):
        """Create test experiment."""
        self.rec1 = MockRecording('rec1', 'OSDS', num_rois=10)
        self.exp = Experiment(recording=[self.rec1])

    def test_fetch_still_works_with_list(self):
        """Test that existing fetch() API still works with list input."""
        result = self.exp.fetch(['ipl_depths', 'num_rois'])
        self.assertIsInstance(result, dict)
        self.assertIn('ipl_depths', result)
        self.assertIn('num_rois', result)

    def test_fetch_still_works_with_string(self):
        """Test that fetch() works with single string input."""
        result = self.exp.fetch('ipl_depths')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_fetch_still_works_with_dict(self):
        """Test that fetch() works with dict input."""
        result = self.exp.fetch({'depths': 'ipl_depths'})
        self.assertIsInstance(result, dict)
        self.assertIn('depths', result)


if __name__ == '__main__':
    unittest.main()
