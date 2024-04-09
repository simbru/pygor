import pygor.load
import pathlib
import os
import numpy.testing as nptest
import numpy as np
import unittest


file_loc = pathlib.Path(__file__).parents[1]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")
print(example_data)
bs_bool = False
data = pygor.load.Core(example_data)

class TestCore(unittest.TestCase):
    def test_averages(self):
        data.averages

if __name__ == "__main__":
    unittest.main()