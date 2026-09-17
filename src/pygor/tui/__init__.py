"""Terminal proofreading cockpit.

Imports nothing heavy: the console entry point has to be able to tell someone
the ``[tui]`` extra is missing, which it cannot do if importing this package is
what fails. Textual is imported inside :func:`pygor.tui.__main__.main`.
"""

__all__ = ["__version__"]

__version__ = "0.1"
