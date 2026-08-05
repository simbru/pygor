"""Tests for the dynamic class discovery in pygor.load.

Dropping a module into pygor/classes/ is supposed to be all it takes to make
its class loadable, so the thing worth checking is that the discovery actually
finds what is on disk.
"""

import importlib
import pathlib
import unittest

import pygor.classes
import pygor.load


def classes_defined_in(module):
    """Classes a module defines itself, ignoring anything it merely imports."""
    return [
        obj.__name__
        for obj in vars(module).values()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    ]


class TestDynamicImport(unittest.TestCase):
    def test_every_class_module_is_importable(self):
        package_dir = pathlib.Path(pygor.classes.__file__).parent
        modules = sorted(
            path.stem
            for path in package_dir.glob("*.py")
            if not path.stem.startswith("_")
        )
        self.assertTrue(modules, "no modules found in pygor.classes")
        for name in modules:
            with self.subTest(module=name):
                importlib.import_module(f"pygor.classes.{name}")

    def test_rerunning_discovery_keeps_the_classes(self):
        """It runs at import, so a second call must be a no-op, not a wipe."""
        pygor.load.dynamic_import()
        self.assertTrue(hasattr(pygor.load, "STRF"))

    def test_documented_classes_are_present(self):
        """The classes the README and CLAUDE.md tell people to load."""
        for name in ("Core", "STRF", "OSDS", "FullField", "Experiment"):
            self.assertTrue(hasattr(pygor.load, name), f"pygor.load.{name} is missing")

    def test_every_class_module_contributes_its_class(self):
        package_dir = pathlib.Path(pygor.classes.__file__).parent
        for path in sorted(package_dir.glob("*.py")):
            if path.stem.startswith("_"):
                continue
            module = importlib.import_module(f"pygor.classes.{path.stem}")
            for cls_name in classes_defined_in(module):
                if cls_name.startswith("_"):
                    continue
                with self.subTest(module=path.stem, cls=cls_name):
                    self.assertTrue(
                        hasattr(pygor.load, cls_name),
                        f"{path.stem}.{cls_name} was not picked up by pygor.load",
                    )


if __name__ == "__main__":
    unittest.main()
