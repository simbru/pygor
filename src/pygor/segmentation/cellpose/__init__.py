"""
Cellpose-based segmentation for pygor.

This submodule provides:
- segment: Main segmentation function (with optional postprocessing)
- masks: Mask format conversion (Cellpose <-> pygor)
- training: Train custom Cellpose models
- postprocess: Split and shrink ROI heuristics
"""

from pygor.segmentation.cellpose.segment import (
    segment,
    run_cellpose_inference,
    load_cellpose_model,
)
from pygor.segmentation.cellpose.postprocess import split_large_rois, shrink_rois
from pygor.segmentation.cellpose.training import train_model
from pygor.segmentation.cellpose import masks

__all__ = [
    "segment",
    "masks",
    "run_cellpose_inference",
    "load_cellpose_model",
    "split_large_rois",
    "shrink_rois",
    "train_model",
]
