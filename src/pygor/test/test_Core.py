import pygor.load
import pathlib
import os
import numpy as np
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
        write_to = file_loc.joinpath(out_loc)
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                for i in meth_list:
                    if i not in ["try_fetch", "draw_rois", "get_depth"]:  # exclusion list
                        try:
                            getattr(data, i)()
                        except AttributeError:
                            warnings.warn(f"Method {i} gave AttributeError")

    def test_get_help(self):
        write_to = file_loc.joinpath(out_loc)
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                data.get_help(hints = True, types = True)
if os.path.exists(file_loc.joinpath(out_loc)):
    atexit.register(lambda : os.remove(file_loc.joinpath(out_loc)))

if __name__ == "__main__":
    unittest.main()
