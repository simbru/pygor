"""
Deprecated module: Use pygor.timeseries.osds instead.

This module is maintained for backward compatibility only.
"""
import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'pygor.timeseries.moving_bars' module is deprecated and will be removed in a future version. "
    "Please use 'from pygor.timeseries.osds import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new module
from pygor.timeseries.osds import *
from pygor.timeseries.osds import tuning_metrics, tuning_computation

__all__ = ['tuning_metrics', 'tuning_computation']
