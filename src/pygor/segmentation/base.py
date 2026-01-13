"""
Base classes and protocols for segmentation methods.

This module defines the interface that all segmentation methods should follow,
enabling easy addition of new methods (e.g., StarDist, threshold-based, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np


class SegmentationMethod(ABC):
    """Abstract base class for segmentation methods.

    All segmentation methods should inherit from this class and implement
    the `segment` method.

    Attributes
    ----------
    name : str
        Human-readable name for the method
    """

    name: str = "base"

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment an image into ROIs.

        Parameters
        ----------
        image : ndarray
            2D image to segment (typically average_stack from pygor)
        **kwargs
            Method-specific parameters

        Returns
        -------
        masks : ndarray
            ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
            This is the intermediate format; conversion to pygor format
            (background=1, ROIs=-1,-2,-3...) happens in the main function.
        """
        raise NotImplementedError


# Registry for segmentation methods
_METHODS: dict[str, SegmentationMethod] = {}


def register_method(name: str, method: SegmentationMethod) -> None:
    """Register a segmentation method.

    Parameters
    ----------
    name : str
        Name to register the method under (e.g., "cellpose", "stardist")
    method : SegmentationMethod
        Instance of the segmentation method
    """
    _METHODS[name] = method


def get_method(name: str) -> SegmentationMethod:
    """Get a registered segmentation method.

    Parameters
    ----------
    name : str
        Name of the method to retrieve

    Returns
    -------
    method : SegmentationMethod
        The registered method

    Raises
    ------
    KeyError
        If the method is not registered
    """
    if name not in _METHODS:
        available = list(_METHODS.keys())
        raise KeyError(
            f"Segmentation method '{name}' not found. "
            f"Available methods: {available}"
        )
    return _METHODS[name]


def list_methods() -> list[str]:
    """List all registered segmentation methods.

    Returns
    -------
    methods : list of str
        Names of all registered methods
    """
    return list(_METHODS.keys())
