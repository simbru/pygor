import pygor.load
import pygor.data_helpers
import pygor.utils.helpinfo
import unittest
import pathlib

file_loc = pathlib.Path(__file__).parents[2]
example_data = file_loc.joinpath("examples/strf_demo_data.h5")
bs_bool = False
strfs = pygor.load.STRF(example_data)

class TestSTRF(unittest.TestCase):
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


    # print(plot_list)