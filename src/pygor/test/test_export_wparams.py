import pathlib
import tempfile
import unittest

import h5py
import numpy as np

import pygor.load

file_loc = pathlib.Path(__file__).parents[3]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")


def _wparamsnum_lookup(ds, name):
    """Mimic IGOR's FindDimLabel-based read: label[i] names data[i-1]."""
    flat = [str(x) for x in np.asarray(ds.attrs["IGORWaveDimensionLabels"]).reshape(-1)]
    return ds[flat.index(name) - 1]


class TestExportWParams(unittest.TestCase):
    def test_wparamsnum_has_labels_and_linedur_resolves(self):
        """wParamsNum must carry IGORWaveDimensionLabels, and IGOR's only read —
        LineDuration = wParamsNum[%User_dxPix] * wParamsNum[%RealPixDur] * 1e-6 —
        must resolve (labels present) and reproduce the recording's line duration.
        Without labels, IGOR's FindDimLabel returns -1 and wParamsNum[-1] errors."""
        data = pygor.load.Core(example_data)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["wParamsNum"]
                self.assertIn("IGORWaveDimensionLabels", ds.attrs)
                dx = _wparamsnum_lookup(ds, "User_dxPix")
                rp = _wparamsnum_lookup(ds, "RealPixDur")
                self.assertGreater(dx, 0)
                self.assertGreater(rp, 0)
                self.assertAlmostEqual(dx * rp * 1e-6, data.linedur_s, places=6)

    def test_wparamsnum_synthesized_without_capture(self):
        """Raw-ScanM objects (no captured wave) must still get a labelled wParamsNum
        whose line-duration read resolves, from _scanm_header / derived fallback."""
        data = pygor.load.Core(example_data)
        data._wparamsnum_raw = None
        data._wparamsnum_labels_raw = None
        data._scanm_header = None  # force the derived fallback
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["wParamsNum"]
                self.assertIn("IGORWaveDimensionLabels", ds.attrs)
                dx = _wparamsnum_lookup(ds, "User_dxPix")
                rp = _wparamsnum_lookup(ds, "RealPixDur")
                self.assertAlmostEqual(dx * rp * 1e-6, data.linedur_s, places=6)

    def test_wparamsstr_fixed_length_and_datetime(self):
        """wParamsStr must be fixed-length text (matching IGOR /IGOR=8, not vlen —
        the suspected H5Dread cause) with date at [4] and time at [5]."""
        data = pygor.load.Core(example_data)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["wParamsStr"]
                self.assertEqual(ds.dtype.kind, "S")  # fixed-length bytes, not vlen
                vals = [v.decode("utf-8", "replace") for v in ds[:]]
                d = [int(x) for x in vals[4].split("-")]
                self.assertEqual(
                    (d[0], d[1], d[2]),
                    (data.metadata["exp_date"].year,
                     data.metadata["exp_date"].month,
                     data.metadata["exp_date"].day),
                )
                t = [int(x) for x in vals[5].split("-")]
                self.assertEqual(t[0], data.metadata["exp_time"].hour)


if __name__ == "__main__":
    unittest.main()
