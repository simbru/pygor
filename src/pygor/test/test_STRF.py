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

file_loc = pathlib.Path(__file__).parents[1]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")
bs_bool = False
strfs = pygor.load.STRF(example_data)

class TestSTRF(unittest.TestCase):
    def test_contours(self):
        strfs.fit_contours()
    
    def test_attributes_return(self):
        attr_list = pygor.utils.helpinfo.get_attribute_list(strfs, with_types=False)
        [getattr(strfs, i) for i in attr_list]

    def test_simple_methods_return(self):
        meth_list = pygor.utils.helpinfo.get_methods_list(strfs, with_returns=False)
        write_to = file_loc.parent.joinpath("test_out.txt")
        with open(write_to, 'w') as f:
            with redirect_stdout(f):
                for i in meth_list:
                        if "plot" not in i and "play" not in i:
                            try:
                                getattr(strfs, i)() 
                            except AttributeError:
                                warnings.warn(f"Method {i} gave AttributeError")
                            except TypeError:
                                warnings.warn(f"Method {i} gave TypeError, likely due to missing inputs")

    def test_saveload(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "test.pkl"
        try:
            strfs.save_pkl(dir_path, filename)
        except FileNotFoundError as e:
            raise FileNotFoundError("Error in storing .pkl file during test_saveload.") from e
            
        finally:
            os.remove(pathlib.Path(dir_path, filename))
    
    def test_bs(self):
        strfs.get_bootstrap_settings()
        strfs.set_bootstrap_bool(True)
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n = 10, time_bs_n = 10)
        strfs.update_bootstrap_settings(new_bs_dict)
        strfs.run_bootstrap()

    def test_get_help(self):
        write_to = file_loc.joinpath("test/test_out.txt")
        with open(write_to, 'w') as f:
            with redirect_stdout(f):
                strfs.get_help(hints = True, types = True)

class TestSTRF_plot(unittest.TestCase):
    def find_plot_methods(self):
        plot_list = pygor.utils.helpinfo.get_methods_list(strfs, with_returns=False)
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
        strfs.plot_timecourse(0)
    
    def test_chromatic_overview(self):
        strfs.plot_chromatic_overview()

if __name__ == '__main__':
    unittest.main()