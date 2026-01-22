"""
Blob detection segmentation for pygor.

Uses Difference of Gaussian (DoG) to detect synaptic terminals
and other blob-like structures.
"""

from pygor.segmentation.blob.segment import segment
from pygor.segmentation import masks

__all__ = ["segment", "masks"]
