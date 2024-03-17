import pprint
import textwrap

pp = pprint.PrettyPrinter(width = 80, indent = 2, compact = True)
md = pprint.PrettyPrinter(width = 160, indent = 2, compact = True)


def text_attrs(attribute_list) -> str:
    attrstr = textwrap.indent(pp.pformat(attribute_list), '    ')
    return attrstr

def text_meths(method_list) -> str:
    methstr = textwrap.indent(pp.pformat(method_list), '    ')
    return methstr

def attrs_help(attribute_list, hints = True) -> str:
    attrs_block = f"""
    ## Attributes
        Here's the data you have access to:
{text_attrs(attribute_list)}
    """
    if hints ==True:
        hint = """
        - You cam change these by customising the corresponding attributes or
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
        Here's some actions you have:
{text_meths(methods_list)}
"""
    if hints == True:
        hint = """
        - You cam change these by customising the corresponding methods in
            data_objects.py. PLease wwrite documentation, preferably in numpy docstring
            but at least a sentence explaining what the function does.
        - These are funcitons that are evoked only when called. Most are configured
            such that they write the result of the called computation to the object
            for future use without re-calculating (lazy initialisation).
        - Fast AND simple functions are exceptions to this.
        """
        return meths_block + hint
    else:
        return meths_block

def welcome_help(data_type_list, metadata_dict, hints = False) -> str:
    print_block = f"""
Welcome to your Data object!
    ## Class info:
        Current data types: 
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
    NB: Feel free to write your own attributes (data) and methods (actions)!:)
    """
    return block

def print_help(to_print_list):
    for i in to_print_list:
        print(i, end="")

