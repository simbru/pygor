"""
Deprecated module: Use osds_data instead.

This module is maintained for backward compatibility only.
"""
import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'moving_bars_data' module is deprecated and will be removed in a future version. "
    "Please use 'from pygor.classes.osds_data import OSDS' or 'from pygor.classes import OSDS' instead. "
    "The 'MovingBars' name is still available as an alias to OSDS for backward compatibility.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new module for backward compatibility
from .osds_data import OSDS, MovingBars

__all__ = ['OSDS', 'MovingBars']
