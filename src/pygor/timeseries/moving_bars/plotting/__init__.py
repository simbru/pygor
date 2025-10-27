"""
Deprecated module: Use pygor.timeseries.osds.plotting instead.

This module is maintained for backward compatibility only.
"""
import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'pygor.timeseries.moving_bars.plotting' module is deprecated and will be removed in a future version. "
    "Please use 'from pygor.timeseries.osds.plotting import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new module
from pygor.timeseries.osds.plotting import *
from pygor.timeseries.osds.plotting import circular_directional_plots, phase_analysis

__all__ = ['circular_directional_plots', 'phase_analysis']
