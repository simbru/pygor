import unittest
import importlib
import pygor.load

def get_submodules(main_module_name) -> list[object]:
    main_module = importlib.import_module(main_module_name)
    submodules = []
    for submodule_name in dir(main_module):
        submodule = getattr(main_module, submodule_name)
        if isinstance(submodule, type(main_module)) and submodule.__package__ == main_module.__package__:
            submodules.append(submodule)
    return submodules

def get_submodule_class_names(main_module_name):
    main_module = importlib.import_module(main_module_name)
    submodules_with_classes = []
    for submodule_name in dir(main_module):
        submodule = getattr(main_module, submodule_name)
        if isinstance(submodule, type(main_module)) and submodule.__package__ == main_module.__package__:
            classes = [cls.__name__ for cls in vars(submodule).values() if isinstance(cls, type) and cls.__module__ == submodule.__name__]
            if len(classes) == 1:
                submodules_with_classes.append((submodule_name, classes[0]))
    return submodules_with_classes

class Import(unittest.TestCase):
    def test_dynamic_import(self):
        pygor.load.dynamic_import()

    def test_submodules(self):
        main_module = "pygor.classes"
        submodules_with_classes = get_submodule_class_names(main_module)
        for submodule, user_class in submodules_with_classes:
            submodule_full = main_module + "." + submodule
            submodule_obj = importlib.import_module(submodule_full)
            class_obj = getattr(submodule_obj, user_class)
            self.assertTrue(class_obj)

if __name__ == '__main__':
    unittest.main()