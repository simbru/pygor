"""
Cellpose-based segmentation for pygor.

This submodule provides:
- inference: Run trained Cellpose models
- postprocess: Split and shrink ROI heuristics
- training: Train custom Cellpose models
"""

from pygor.segmentation.cellpose.inference import run_cellpose_inference
from pygor.segmentation.cellpose.postprocess import split_large_rois, shrink_rois
from pygor.segmentation.cellpose.training import train_model

__all__ = [
    "run_cellpose_inference",
    "split_large_rois",
    "shrink_rois",
    "train_model",
]
