import pygor.load
import pygor.data_helpers
import pygor.utils.helpinfo
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

    def test_simple_methods_return(self):
        """
        Deliberately exclude 'plot', 'play' and 'bootstrap' ,methods, and skips methods that
        require user inputs.
        """
        meth_list = pygor.utils.helpinfo.get_methods_list(strfs, with_returns=False)
        write_to = file_loc.joinpath("test/test_out_STRF.txt")
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                for i in meth_list:
                    if "plot" not in i and "play" not in i and "bootstrap" not in i:
                        try:
                            getattr(strfs, i)()
                        # except AttributeError:
                        #     warnings.warn(f"Method {i} gave AttributeError")
                        except TypeError:
                            warnings.warn(
                                f"Method {i} gave TypeError, likely due to missing inputs"
                            )

    ## TODO:
    # - Write test for non-simple methods that require user inputs

    def test_attributes_return(self):
        attr_list = pygor.utils.helpinfo.get_attribute_list(strfs, with_types=False)
        [getattr(strfs, i) for i in attr_list]

    def test_saveload(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = "test.pkl"
        try:
            strfs.save_pkl(dir_path, filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Error in storing .pkl file during test_saveload."
            ) from e

        finally:
            os.remove(pathlib.Path(dir_path, filename))

    def test_bs(self):
        strfs.get_bootstrap_settings()
        strfs.set_bootstrap_bool(True)
        new_bs_dict = pygor.data_helpers.create_bs_dict(space_bs_n=10, time_bs_n=10)
        strfs.update_bootstrap_settings(new_bs_dict)
        strfs.run_bootstrap()

    def test_get_help(self):
        write_to = file_loc.joinpath("test/test_out_STRF.txt")
        with open(write_to, "w") as f:
            with redirect_stdout(f):
                strfs.get_help(hints=True, types=True)


if __name__ == "__main__":
    unittest.main()
