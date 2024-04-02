import unittest

loc = r"C:\Users\Simen\Downloads\2023-11-14_0_0_SWN_200_Colours.h5"
bs_bool = False

class Import(unittest.TestCase):
    def test_core(self):
        from core import data_objects_beta
        

if __name__ == '__main__':
    unittest.main()