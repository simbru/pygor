import pygor.experiment
import pygor.data_helpers
import numpy.testing as nptest
import numpy as np
import unittest
import warnings
import os, pathlib

loc = r"C:\Users\Simen\Downloads\2023-11-14_0_0_SWN_200_Colours.h5"
bs_bool = False
strfs = pygor.experiment.STRF(loc)


class TestSTRF(unittest.TestCase):
    def test_contours(self):
        strfs.contours
    
    def test_attributes(self):
        attr_list = pygor.data_helpers.get_attribute_list(strfs, with_types=False)
        [getattr(strfs, i) for i in attr_list]

    def test_simple_methods(self):
        meth_list = pygor.data_helpers.get_methods_list(strfs, with_returns=False)
        for i in meth_list:
            if i not in ["save_pkl", "load_pkl", "get_bootstrap_settings", "update_bootstrap_settings", "run_bootstrap"]:
                try:
                    getattr(strfs, i)() 
                except AttributeError:
                    warnings.warn(f"Method {i} gave AttributeError")

    def test_saveload(self):
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "test.pkl"
        strfs.save_pkl(dir_path, filename)
        os.remove(pathlib.Path(dir_path, filename))
    
    def test_bs(self):
        strfs.get_bootstrap_settings()
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n = 10, time_bs_n = 10)
        strfs.update_bootstrap_settings(new_bs_dict)
        strfs.run_bootstrap()

if __name__ == '__main__':
    unittest.main()