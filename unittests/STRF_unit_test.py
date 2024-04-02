import pygor.experiment
import pygor.data_helpers
import numpy.testing as nptest
import numpy as np
import unittest



loc = r"C:\Users\Simen\Downloads\2023-11-14_0_0_SWN_200_Colours.h5"
bs_bool = False
strfs = pygor.experiment.STRF(loc)


class TestSTRF(unittest.TestCase):
    def test_contours(self):
        strfs.contours
    
    def test_attributes(self):
        attr_list = pygor.data_helpers.get_attribute_list(strfs, with_types=False)
        [getattr(strfs, i) for i in attr_list]

    def test_bs(self):
        strfs.get_bootstrap_settings()
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n = 10, time_bs_n = 10)
        strfs.update_bootstrap_settings(new_bs_dict)
        strfs.run_bootstrap()

if __name__ == '__main__':
    unittest.main()