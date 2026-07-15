import pathlib
import tempfile
import unittest

import h5py
import numpy as np

import pygor.load
from pygor.preproc.os_parameter_table import OS_PARAMETER_TABLE

file_loc = pathlib.Path(__file__).parents[3]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")


class TestOSParametersRoundTrip(unittest.TestCase):
    def test_h5_roundtrip_preserves_captured_array(self):
        """Loading a genuine IGOR h5 and re-exporting should reproduce the
        captured OS_Parameters array + labels shape (patched only where the
        live object state actually differs)."""
        data = pygor.load.Core(example_data)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["OS_Parameters"]
                labels = ds.attrs["IGORWaveDimensionLabels"]
                self.assertEqual(ds.shape, data._os_parameters_raw.shape)
                self.assertEqual(labels.shape, data._os_parameters_labels_raw.shape)

    def test_h5_roundtrip_self_consistent(self):
        """Export then reload; named scalars must match the original."""
        data = pygor.load.Core(example_data)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            reloaded = pygor.load.Core(out_path)
            self.assertAlmostEqual(reloaded.linedur_s, data.linedur_s, places=6)
            self.assertEqual(reloaded.trigger_mode, data.trigger_mode)
            self.assertEqual(reloaded.n_planes, data.n_planes)

    def test_synthesized_table_has_igor_dimension_labels(self):
        """A from-scratch synthesis (no captured raw wave, e.g. objects loaded
        from raw ScanM files) must reproduce a genuine IGOR OS_Parameters layout:
        data length == label-attr length (IGOR's on-disk convention), a leading
        blank dimension-name slot, and label[i] naming data[i-1]. Emitting an
        N+1 label array (the old bug) makes IGOR mis-apply labels and read
        garbage channel numbers."""
        data = pygor.load.Core(example_data)
        data._os_parameters_raw = None
        data._os_parameters_labels_raw = None
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["OS_Parameters"]
                labels = ds.attrs["IGORWaveDimensionLabels"]
                # IGOR: label attribute has exactly as many rows as data points.
                self.assertEqual(ds.shape, (len(OS_PARAMETER_TABLE),))
                self.assertEqual(labels.shape, (len(OS_PARAMETER_TABLE), 1))
                self.assertEqual(labels[0, 0], "")
                # label[i] names data[i-1]; check the channel params IGOR's
                # trigger detection actually reads.
                flat = [str(x) for x in np.asarray(labels).reshape(-1)]
                for name, expected in (
                    ("Data_Channel", 0.0),
                    ("Trigger_Channel", 2.0),
                    ("Trigger_Mode", float(data.trigger_mode)),
                ):
                    i = flat.index(name)
                    self.assertEqual(ds[i - 1], expected)

    def test_synthesized_layout_matches_installed_ipf_contract(self):
        """Regression guard against the demo-file swap: the synthesized table must
        match the CURRENTLY-INSTALLED OS_ParameterTable.ipf (which the demo file, an
        older script version, does NOT). The averaging scripts read %AvgStack_make /
        %AvgStack_SkipTrig / %AvgStack_firstplane — absent from the demo layout, which
        is what made the average stack build despite the flag and error on SkipTrig."""
        from pygor.preproc.os_parameter_table import IPF_TABLE_LENGTH

        data = pygor.load.Core(example_data)
        data._os_parameters_raw = None
        data._os_parameters_labels_raw = None
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                ds = f["OS_Parameters"]
                labels = ds.attrs["IGORWaveDimensionLabels"]
                flat = [str(x) for x in np.asarray(labels).reshape(-1)]
                # label[i] names data[i-1]: assert the ipf averaging + channel params
                # at their exact installed-ipf indices.
                for name, data_idx, value in (
                    ("AvgStack_make", 28, 0.0),
                    ("AvgStack_SkipTrig", 29, 1.0),
                    ("AvgStack_firstplane", 30, 1.0),
                    ("Data_Channel", 53, 0.0),
                    ("Trigger_Channel", 55, 2.0),
                    ("Trigger_Threshold", 60, 20000.0),
                ):
                    self.assertEqual(flat[data_idx + 1], name)
                    self.assertEqual(ds[data_idx], value)
                # Stale demo-only names must NOT reappear.
                self.assertNotIn("AverageStack_make", flat)
                # Appended STRF param present and set from frame_hz.
                self.assertIn("samp_rate_Hz", flat)
                sr = ds[flat.index("samp_rate_Hz") - 1]
                self.assertAlmostEqual(sr, float(data.frame_hz), places=3)
                # ipf portion length + N-length label convention preserved.
                self.assertGreaterEqual(ds.shape[0], IPF_TABLE_LENGTH)
                self.assertEqual(len(flat), ds.shape[0])

    def test_no_homemade_os_parameters_attribute_key(self):
        """Regression guard: must not reintroduce the old fake attribute name."""
        data = pygor.load.Core(example_data)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "export.h5"
            data.export_to_h5(out_path, overwrite=True)
            with h5py.File(out_path, "r") as f:
                attrs = dict(f["OS_Parameters"].attrs.items())
                self.assertIn("IGORWaveDimensionLabels", attrs)
                self.assertEqual(list(attrs.keys()), ["IGORWaveDimensionLabels"])


if __name__ == "__main__":
    unittest.main()
