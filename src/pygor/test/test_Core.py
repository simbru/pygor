import pygor.analyses
import pathlib
import os
import numpy.testing as nptest
import numpy as np
import unittest


repo = pathlib.Path(os.getcwd())
example_data = repo.joinpath(repo, "Example_data/example_exp.h5")
print(example_data)
bs_bool = False
data = pygor.analyses.Core(example_data)

class TestCore(unittest.TestCase):
    def test_averages(self):
        data.averages

if __name__ == "__main__":
    unittest.main()