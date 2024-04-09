# Local imports
# from pygor.data_objs.data_parent import Data


# Dependencies
import os 
import sys
import warnings
import pathlib

""" 
The following function imports custom data objects from the specified directory and adds them to the current module.
"""


def import_data_objs() -> None:
    """
    Imports custom data objects from the specified directory and adds them to the current module.

    Parameters:
        None

    Returns:
        None

    Description:
        This function imports custom data objects from the specified directory and adds them to the current module. It searches 
        for Python files in the directory and imports all classes defined in those files. The imported classes are added as 
        attributes to the current module, allowing them to be accessed directly.

        The directory path is constructed using the current file's directory path and the relative path to the custom classes
        directory. The function searches for Python files in the directory and excludes the '__init__.py' file.

        For each Python file found, the module is dynamically imported using the '__import__' function. The classes defined 
        in the module are then extracted and added to the list of classes to import.

        After importing all the classes, the function removes any duplicates from the list and assigns the imported classes 
        as attributes to the current module. It also adds the imported classes to the 'sys.modules' dictionary, allowing them 
        to be imported by other modules.

        If any errors occur during the import process, a warning is issued and the import is skipped for that class.

        Finally, the function prints the number of custom classes found in the directory and the names of the imported classes.

    Example:
        import_data_objs()

            Found custom classes in /path/to/pygor/classes
            Imported as: ['ClassName1', 'ClassName2', ...]
    """
    dir_path = pathlib.Path(os.path.dirname(os.path.abspath('')), "pygor", "src", "pygor","classes")
    files_in_dir = [f for f in os.listdir(dir_path)
                    if f.endswith('.py') and f != '__init__.py']
    for_import = []
    simple_names = []
    for f in files_in_dir:
        module = __import__('.'.join(["pygor.classes", pathlib.Path(f).stem]), fromlist=[f])
        classes_to_import = [getattr(module, x) for x in dir(module) if isinstance(getattr(module, x), type) and "pygor.classes" in getattr(module, x).__module__]
        for_import.extend(classes_to_import)
    # Ignore duplicates
    for_import = list(dict.fromkeys(for_import))
    simple_names = list(dict.fromkeys(simple_names))        
    if len(for_import) > 0:
        for i in for_import:
            try:
                print(i, i.__name__, __name__)
                setattr(sys.modules[__name__], i.__name__, i)
                sys.modules[i.__name__] = i
                simple_names.append(i.__name__)
            except AttributeError:
                warnings.warn(f"Could not import {i.__name__}")
                pass

    print(f"Found custom classes in {dir_path}", for_import)
    print("Imported as:", simple_names)
import_data_objs()

# Get dataobjects
# def import_data_objs() -> None:
#     # Get path for data objects
#     dir_path = pathlib.Path(os.path.dirname(os.path.abspath('')), "src", "pygor","data_objs")
#     # Loop through list, find all .py files
#     files_in_dir = [f for f in os.listdir(dir_path)
#                     if f.endswith('.py') and f != '__init__.py']
#     # Check that they are importable modules with pygor.data_objs in the __module__
#     for f in files_in_dir:
#         module = __import__('.'.join(["pygor.data_objs", pathlib.Path(f).stem]), fromlist=[f])
#         to_import = [getattr(module, x) for x in dir(module) if isinstance(getattr(module, x), type) and "pygor.data_objs" in getattr(module, x).__module__]
#     if len(to_import) > 0:
#         print(f"Found custom classes in {dir_path}", to_import)
#         simple_names = []
#         for i in to_import:
#             try:
#                 setattr(sys.modules[__name__], i.__name__, i)
#                 simple_names.append(i.__name__)
#             except AttributeError:
#                 warnings.warn(f"Could not import {i.__name__}")
#                 pass
#             except ImportError:
#                 warnings.warn(f"Missing library for {i.__name__}")
#                 pass
#         # print("Imported as:", simple_names)
#     else:
#         print(f"No custom classes found in {dir_path}")
# import_data_objs()

#pp = pprint.PrettyPrinter(indent=4)








    
