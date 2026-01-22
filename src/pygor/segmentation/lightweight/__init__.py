"""
Lightweight segmentation methods for pygor.

These methods don't require deep learning frameworks (Cellpose/PyTorch) and work
directly on correlation/average projections using classical image processing.

Available methods:
    - watershed: Local maxima seeded watershed segmentation
    - flood_fill: IGOR-style region growing from peaks
"""

from pygor.segmentation.lightweight.segment import (
    segment,
    segment_watershed,
    segment_flood_fill,
    prepare_image,
)

# Use shared mask conversion utilities (no cellpose dependency)
from pygor.segmentation import masks

__all__ = [
    "segment",
    "segment_watershed",
    "segment_flood_fill",
    "prepare_image",
    "masks",
]
