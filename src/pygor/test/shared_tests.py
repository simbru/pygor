
import 

file_loc = pathlib.Path(__file__).parents[1]

def test_attributes_return(self):
    attr_list = pygor.utils.helpinfo.get_attribute_list(data, with_types=False)
    [getattr(data, i) for i in attr_list]

def test_simple_methods_return(self):
    meth_list = pygor.utils.helpinfo.get_methods_list(data, with_returns=False)
    write_to = file_loc.parent.joinpath("test_out.txt")
    with open(write_to, 'w') as f:
        with redirect_stdout(f):
            for i in meth_list:
                if i not in ['try_fetch']: # exclusion list
                    try:
                        getattr(data, i)() 
                    except AttributeError:
                        warnings.warn(f"Method {i} gave AttributeError")