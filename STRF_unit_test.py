import data_objects_beta
import numpy.testing as nptest
import numpy as np
import unittest



loc = r"C:\Users\Simen\Downloads\2023-11-14_0_0_SWN_200_Colours.h5"
bs_bool = False


class TestSTRF(unittest.TestCase):
    file_location : str or Pathlib.path = loc
    def test_contours(self):
        strfs = data_objects_beta.STRF(self.file_location, do_bootstrap=bs_bool)
        nptest.assert_array_equal(strfs.contours, strfs._STRF__contours, 
            "Contour retrieval failed")

    def test_bs(self):
        strfs = data_objects_beta.STRF(self.file_location, do_bootstrap=bs_bool)
        strfs.run_bootstrap()            

if __name__ == '__main__':
    unittest.main()