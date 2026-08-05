"""
GUI submodule for pygor.core

This module provides GUI functionality for interactive ROI drawing with napari.
Napari is an optional dependency and only imported when GUI methods are used.
"""

# Import the methods submodule so it's accessible as pygor.core.gui.methods
from . import methods


def interactive(func):
    """Mark a method as one that blocks until a human acts on a window or prompt.

    Callers that run unattended (the test suite, batch scripts) filter on
    ``func.__pygor_interactive__`` rather than keeping a list of method names in
    sync, which is how a napari call previously reached the test runner and
    aborted it.
    """
    func.__pygor_interactive__ = True
    return func


def is_interactive(func) -> bool:
    """True if `func` was marked with `interactive`."""
    return getattr(func, "__pygor_interactive__", False)


__all__ = ["methods", "interactive", "is_interactive"]
