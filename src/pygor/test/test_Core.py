import pygor.load
import pathlib
import os
import numpy.testing as nptest
import numpy as np
import unittest
import warnings
from contextlib import redirect_stdout


file_loc = pathlib.Path(__file__).parents[1]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")
print(example_data)
bs_bool = False
data = pygor.load.Core(example_data)

class TestCore(unittest.TestCase):
    def test_averages_type(self):
        self.assertTrue(isinstance(data.averages, np.ndarray) or np.isnan(data.averages))

    def test_ipl_depths_type(self):
        self.assertTrue(isinstance(data.ipl_depths, np.ndarray) or np.isnan(data.ipl_depths))

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
        with open(r"src/pygor/test/test_out.txt", 'w') as f:
            with redirect_stdout(f):
                for i in meth_list:
                    if i not in []: # exclusion list
                        try:
                            getattr(data, i)() 
                        except AttributeError:
                            warnings.warn(f"Method {i} gave AttributeError")

if __name__ == "__main__":
    unittest.main()
    print(os.getcwd())