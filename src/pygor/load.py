# Local imports
# from pygor.data_objs.data_parent import Data


# Dependencies
import os 
import sys
import warnings
import pathlib
import importlib 

# Ensure all warnings are displayed
warnings.simplefilter('always')

def dynamic_import(classes_folder = "classes") -> None:
    """
    Imports custom data objects from the specified directory and adds them to the current module.
    """
    # Determine the directory path containing custom classes
    directory_path = pathlib.Path(__file__).resolve().parent / classes_folder
    # List all Python files in the directory (excluding __init__.py)
    files = [pathlib.Path(f) for f in os.listdir(directory_path) if f.endswith('.py') and f != '__init__.py']
    # List to store imported classes
    imported_classes = []
    # Loop through each Python file
    for file in files:
        # Import the module corresponding to the Python file
        module = importlib.import_module(f"pygor.classes.{file.stem}")
        # Iterate over attributes of the module
        for obj_name, obj in vars(module).items():
            # Check if the attribute is a class defined within the module
            if isinstance(obj, type) and obj.__module__ == module.__name__:
                # Check if class with the same name is already imported
                if hasattr(sys.modules[__name__], obj.__name__):
                    # Print a message and raise a warning if the class is already imported
                    print(f"Class '{obj.__name__}' is already imported, skipping.")
                    warnings.warn(f"Class '{obj.__name__}' is already imported, skipping.", UserWarning)
                else:
                    # Add the class to the list of imported classes
                    imported_classes.append(obj)
    # Remove duplicates from the list of imported classes
    imported_classes = list(dict.fromkeys(imported_classes))
    # Add imported classes to the current module
    for cls in imported_classes:
        setattr(sys.modules[__name__], cls.__name__, cls)
        sys.modules[cls.__name__] = cls
    if len(imported_classes) > 0:
        # Print summary of imported classes
        print(f"Found {len(imported_classes)} custom classes in {directory_path}")
        print("Class names:", [cls.__name__ for cls in imported_classes])
    print(f"Access custom classes using 'from {__name__} import ClassName'")    

# Call the function to import custom data objects, regardless if __name__ == "__main__"
# such that the classes can be imported by importing pygor.load
dynamic_import()