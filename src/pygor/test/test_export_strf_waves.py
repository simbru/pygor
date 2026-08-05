import pathlib
import tempfile
import unittest

import h5py
import numpy as np
import pytest

import pygor.load
from pygor.test.helpers import DEMO_DATA as example_data

pytestmark = pytest.mark.demo_data


class TestExportStrfWaves(unittest.TestCase):
    """STRF.export_to_h5 must add IGOR OS_STRFs representation waves on top of the
    base Core export, with IGOR-native shapes, and round-trip through load_strf."""

    @classmethod
    def setUpClass(cls):
        cls.obj = pygor.load.STRF(example_data)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out_path = pathlib.Path(cls._tmp.name) / "strf_export.h5"
        cls.obj.export_to_h5(cls.out_path, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_individual_filters_present_and_shaped(self):
        """One STRF0_{roi}_{colour} kernel per (roi, colour), IGOR (x, y, frame)."""
        obj = self.obj
        with h5py.File(self.out_path, "r") as f:
            names = sorted(k for k in f.keys() if k.startswith("STRF0_"))
            self.assertEqual(len(names), obj.num_rois * obj.n_colours)
            # kernel is [time, y, x] -> transposed to (x, y, time); time is smallest
            kernel = f[names[0]]
            self.assertEqual(kernel.ndim, 3)
            self.assertEqual(kernel.shape[2], obj.strfs.shape[1])  # frames

    def test_collapsed_projection_waves(self):
        """STRF_Corr0 and STRF_SD0 both exist as (nX, nY*nColours, nROI)."""
        obj = self.obj
        with h5py.File(self.out_path, "r") as f:
            for name in ("STRF_Corr0", "STRF_SD0"):
                self.assertIn(name, f)
                self.assertEqual(f[name].shape[2], obj.num_rois)
                self.assertEqual(f[name].shape[1] % obj.n_colours, 0)
            # sourced from the same pygor collapse, so identical by design
            np.testing.assert_array_equal(f["STRF_Corr0"][:], f["STRF_SD0"][:])

    def test_concatenated_and_montage(self):
        obj = self.obj
        with h5py.File(self.out_path, "r") as f:
            self.assertIn("STRFs_concatenated", f)
            self.assertEqual(f["STRFs_concatenated"].shape[1] % obj.n_colours, 0)
            self.assertIn("STRF_Corr_Montage", f)
            self.assertEqual(f["STRF_Corr_Montage"].ndim, 2)

    def test_rgb_montage_only_when_four_or_more_colours(self):
        obj = self.obj
        with h5py.File(self.out_path, "r") as f:
            expected = obj.n_colours >= 4
            self.assertEqual("STRF_Corr_Montage_RGB" in f, expected)
            if expected:
                self.assertEqual(f["STRF_Corr_Montage_RGB"].shape[2], 3)
                self.assertIn("STRF_Corr_Montage_RGB2", f)

    def test_roundtrip_reloads_strfs(self):
        """Exported individual filters reload via load_strf with matching kernels."""
        reloaded = pygor.load.STRF(str(self.out_path))
        self.assertEqual(reloaded.strfs.shape, self.obj.strfs.shape)
        a = np.ma.filled(self.obj.strfs, 0.0).ravel()
        b = np.ma.filled(reloaded.strfs, 0.0).ravel()
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.999)


if __name__ == "__main__":
    unittest.main()
