"""
Pygor segmentation module.

Provides automated ROI segmentation using various methods.

Example usage:
    data = pygor.load.Core("file.h5")
    data.segment_rois(mode="cellpose+")  # Cellpose + post-processing heuristics

Available modes:
    - "cellpose+": Cellpose with splitting/shrinking heuristics (recommended)
    - "cellpose": Raw Cellpose output only
"""

from pygor.segmentation.core import segment_rois

__all__ = ["segment_rois"]
