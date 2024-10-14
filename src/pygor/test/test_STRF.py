import pygor.load
import pygor.data_helpers
import pygor.utils.helpinfo
import numpy.testing as nptest
import numpy as np
import unittest
import warnings
import os
import pathlib
from contextlib import redirect_stdout
import atexit

file_loc = pathlib.Path(__file__).parents[1]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")
bs_bool = False

class TestSTRF(unittest.TestCase):
    strfs = pygor.load.STRF(example_data)

    def test_contours(self):
        self.strfs.fit_contours()
    
    def test_attributes_return(self):
        attr_list = pygor.utils.helpinfo.get_attribute_list(self.strfs, with_types=False)
        [getattr(self.strfs, i) for i in attr_list]

    def test_simple_methods_return(self):
        meth_list = pygor.utils.helpinfo.get_methods_list(self.strfs, with_returns=False)
        bs_refs = [i for i in meth_list if "bootstrap" in i or "bs" in i]
        meth_set = set(meth_list) - set(bs_refs)
        print("Testing simple methods:")
        for i in meth_set:
            print(f"- {i}")    
            if "plot" not in i and "play" not in i:
                try:
                    getattr(self.strfs, i)() 
                except AttributeError:
                    warnings.warn(f"Method {i} gave AttributeError")
                except TypeError:
                    warnings.warn(f"Method {i} gave TypeError, likely due to missing inputs")

    def test_saveload(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "test.pkl"
        try:
            self.strfs.save_pkl(dir_path, filename)
        except FileNotFoundError as e:
            raise FileNotFoundError("Error in storing .pkl file during test_saveload.") from e
            
        finally:
            os.remove(pathlib.Path(dir_path, filename))
    
    def test_bs(self):
        self.strfs.set_bootstrap_bool(True)
        self.strfs.get_bootstrap_settings()
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n = 10, time_bs_n = 10)
        self.strfs.update_bootstrap_settings(new_bs_dict)
        self.strfs.run_bootstrap()

    def test_get_help(self):
        write_to = file_loc.joinpath("test/test_out_STRF.txt")
        with open(write_to, 'w') as f:
            with redirect_stdout(f):
                self.strfs.get_help(hints = True, types = True)

class TestSTRF_plot(unittest.TestCase):
    strfs = pygor.load.STRF(example_data)

    def find_plot_methods(self):
        plot_list = pygor.utils.helpinfo.get_methods_list(self.strfs, with_returns=False)
        plot_list = [i for i in plot_list if "plot" in i]
        print("Found plot methods:", plot_list)
        disallowed = ["plot_averages"]
        if any(i in plot_list for i in disallowed):
            print("Found disallowed plot methods:", set(plot_list) & set(disallowed))
            print("Ignoring these.")
        plot_set = set(plot_list) - set(disallowed)
        # getattr(strfs, i)()
        return plot_set
    
    def test_timecourse(self):
        self.strfs.plot_timecourse(0)
    
    def test_chromatic_overview(self):
        self.strfs.plot_chromatic_overview()

atexit.register(lambda : os.remove(file_loc.joinpath("test/test_out_STRF.txt")))

if __name__ == '__main__':
    unittest.main()