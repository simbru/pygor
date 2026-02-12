"""
GUI submodule for pygor.core

This module provides GUI functionality for interactive ROI drawing with napari.
Napari is an optional dependency and only imported when GUI methods are used.
"""

# Import the methods submodule so it's accessible as pygor.core.gui.methods
from . import methods

__all__ = ['methods']
