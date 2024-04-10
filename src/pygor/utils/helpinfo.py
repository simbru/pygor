import pprint
import textwrap
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import types

pp = pprint.PrettyPrinter(width = 110, indent = 2, compact = True)
md = pprint.PrettyPrinter(width = 110, indent = 2, compact = True)

def get_methods_list(obj, with_returns = True) -> list:
    """
    Get a list of methods of a given object.

    Parameters
    ----------
    obj : object
        The object to retrieve the methods from.
    with_returns : bool, optional
        If True, include the return type of each method in the list.
        Default is True.

    Returns
    -------
    list
        A list of methods from the object. If `with_returns` is True,
        each method is followed by its return type in parenthesis.

    """
    method_list = [func for func in dir(obj) if callable(getattr(obj, func)) is True and "__" not in func]
    """
    TODO Sort the methods list alphabetically by the 
    letter in the second part of the funciton name, after '_'
    """
    if with_returns is True:
        method_list = [func_str + f" ({get_return_type(getattr(obj, func_str))})" for func_str in method_list]
    return method_list

def get_attribute_list(obj, with_types = True) -> list:
    """
    Get a list of attributes of a given object.

    Parameters
    ----------
    obj : object
        The object to retrieve the attributes from.
    with_types : bool, optional
        If True, include the types of each attribute in the list.
        Default is True.

    Returns
    -------
    list
        A list of attributes from the object. If `with_types` is True,
        each attribute is followed by its type in parenthesis.

    """
    attribute_list = [attr for attr in dir(obj) if callable(getattr(obj, attr)) is False and "__" not in attr and attr[0] != '_']
    if with_types is True:
        attribute_list = [attr_str + f" ({type(getattr(obj, attr_str)).__name__})" for attr_str in attribute_list]
    return attribute_list

def get_return_type(func):
    function_annotation = func.__annotations__.get('return', '?')
    if function_annotation is None:
        return 'None'
    if function_annotation == '?':
        return '?'
    if isinstance(function_annotation, str) is False and isinstance(function_annotation, Iterable):
        org_type = type(function_annotation)
        return org_type([i.__name__ for i in function_annotation])
    elif isinstance(function_annotation, str) is False and isinstance(function_annotation, Iterable) is False:
        return function_annotation.__name__
    else:
        try:
            raise AssertionError(f"Unaccounted for attribute type in function return annotation for {func.__name__}, manual fix required")
        except AttributeError:
            raise AttributeError(f"Unaccounted for attribute type in function return annotation, manual fix required")

def text_attrs(attribute_list) -> str:
    attrstr = textwrap.indent(pp.pformat(attribute_list), '    ')
    return attrstr

def text_meths(method_list) -> str:
    methstr = textwrap.indent(pp.pformat(method_list), '    ')
    return methstr

def attrs_help(attribute_list, hints = True) -> str:
    attrs_block = f"""
    ## Attributes
        Here's the data you have access to -> pass 'types = True' for type hints, attr (type):
{text_attrs(attribute_list)}
    """
    if hints ==True:
        hint = """
        - You can change these by customising the corresponding attributes or
            @property-decorated functions inside data_objects.py
        - These typically get created at run-time, within the dataclass, and include data 
            properties that are simple and static.
        - Attributes that are complex to generate should be imported from data_helpers.py 
        """
        return attrs_block + hint
    else:
        return attrs_block

def meths_help(methods_list, hints = True) -> str:
    meths_block = f"""
    ## Methods
        Here's some actions you have -> pass 'types = True' for type hints, func (return type):
{text_meths(methods_list)}
"""
    if hints == True:
        hint = """
        - You cam change these by customising the corresponding methods in
            data_objects.py. Please wwrite documentation, preferably in numpy docstring
            but at least a sentence explaining what the function does.
        - These are funcitons that are evoked only when called. Most are configured
            such that they write the result of the called computation to the object
            for future use without re-calculating (lazy initialisation).
        - Fast AND simple functions are exceptions to lazy initialisation.
        """
        return meths_block + hint
    else:
        return meths_block

def welcome_help(data_type_list, metadata_dict, hints = False) -> str:
    print_block = f"""
Welcome to your data object! 
Below are attributes and methods, as well as some metadata.
Pass 'hint = True' for more tips and hints on using the data class.
    ## Class info:
        Current data type: 
{textwrap.indent(pp.pformat(data_type_list), '        ')}
        Current metadata: 
{textwrap.indent(md.pformat(metadata_dict), '       ')}
"""
    if hints == True:
        hint = """
    - You can access these via self.data_types or self.metadata, respectively.
        Here, 'self' referes to the variable name you give your Data object.
    - All attributes (data) and methods (actions) are accessed likewise.
    """
        return print_block + hint
    else:
        return print_block

def text_exit() -> str:
    block = """
    NB: Feel free to write your own attributes (data) and methods (actions)!
    Happy analysin' (:
    """
    return block

def print_help(to_print_list):
    for i in to_print_list:
        print(i, end="")

