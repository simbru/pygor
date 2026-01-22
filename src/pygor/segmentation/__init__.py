"""
Pygor segmentation module.

Provides automated ROI segmentation using various methods.

Example usage:
    data = pygor.load.Core("file.h5")
    data.segment_rois(mode="cellpose+")  # Cellpose + post-processing heuristics
    data.segment_rois(mode="watershed")  # Lightweight, no ML required

Available modes:
    - "cellpose+": Cellpose with splitting/shrinking heuristics (recommended for trained models)
    - "cellpose": Raw Cellpose output only
    - "watershed": Lightweight watershed segmentation (no ML required)
    - "flood_fill": IGOR-style region growing (no ML required)

To add new segmentation methods, create a submodule with a segment() function
and add an entry to _METHODS below.
"""

import numpy as np
import warnings

# Import shared mask utilities (no cellpose/PyTorch dependency)
from pygor.segmentation import masks

# Method registry: mode -> (module_path, postprocess_default, pass_data_object)
# pass_data_object: if True, pass the full data object instead of just the image
_METHODS = {
    "cellpose": ("pygor.segmentation.cellpose", False, False),
    "cellpose+": ("pygor.segmentation.cellpose", True, False),
    "watershed": ("pygor.segmentation.lightweight", False, True),
    "flood_fill": ("pygor.segmentation.lightweight", False, True),
}


def list_methods():
    """List available segmentation methods.

    Returns
    -------
    methods : list of str
        Available mode strings
    """
    return list(_METHODS.keys())


def segment_rois(
    data,
    mode="cellpose+",
    model_path=None,
    model_dir=None,
    overwrite=False,
    verbose=True,
    **kwargs,
):
    """Segment ROIs using automated methods.

    Parameters
    ----------
    data : Core or STRF
        Pygor data object with image data
    mode : str
        Segmentation mode. Use list_methods() for available options:
        - "cellpose+": Cellpose with post-processing heuristics (recommended for trained models)
        - "cellpose": Raw Cellpose output only
        - "watershed": Lightweight watershed segmentation (no ML required)
        - "flood_fill": IGOR-style region growing (no ML required)
    model_path : str or Path, optional
        Direct path to a trained model file (Cellpose modes only)
    model_dir : str or Path, optional
        Directory to search for trained models (Cellpose modes only)
    overwrite : bool
        If True, overwrite existing ROIs in data object
    verbose : bool
        If True, print progress messages
    **kwargs
        Additional parameters passed to the segmentation method.

        Cellpose parameters:
        - diameter : float (default: None, auto-detect)
        - flow_threshold : float (default: 0.9)
        - cellprob_threshold : float (default: 0.5)
        - min_size : int (default: 2)

        Post-processing parameters (for mode="cellpose+"):
        - split_large : bool (default: True)
        - size_multiplier : float (default: 1.5)
        - min_peak_distance : int (default: 1)
        - min_size_after_split : int (default: 4)
        - shrink_iterations : int (default: 1, 0 to disable)
        - shrink_size_threshold : int (default: 30)

        Lightweight parameters (for mode="watershed" or "flood_fill"):
        - input_mode : str (default: "combined") - "correlation", "average", or "combined"
        - threshold : float (watershed: 0.1, flood_fill: 0.15)
        - min_distance : int (default: 1)
        - min_roi_size : int (default: 3)
        - gap_pixels : int (watershed only, default: 2)
        - max_size : int (flood_fill only, default: 20)
        - drop_fraction : float (flood_fill only, default: 0.2)

    Returns
    -------
    masks : ndarray
        ROI mask in pygor format (background=1, ROIs=-1,-2,-3...)

    Examples
    --------
    >>> data = pygor.load.Core("recording.h5")
    >>> data.segment_rois(model_dir="./models/synaptic")  # Cellpose

    >>> # Lightweight segmentation (no ML required)
    >>> data.segment_rois(mode="watershed", input_mode="combined")
    >>> data.segment_rois(mode="flood_fill", max_size=15, drop_fraction=0.3)
    """
    # Check for existing ROIs
    if data.rois is not None and not overwrite:
        warnings.warn("ROIs already exist. Use overwrite=True to replace.")
        return None

    # Get method
    if mode not in _METHODS:
        available = list_methods()
        raise ValueError(f"Unknown mode: '{mode}'. Available: {available}")

    module_path, postprocess_default, pass_data_object = _METHODS[mode]

    # Lazy import the method module
    import importlib
    method_module = importlib.import_module(module_path)

    # Determine what to pass to segment()
    if pass_data_object:
        # Lightweight methods: pass the full data object so they can access
        # params.artifact_width and choose input image
        # Extract method-specific kwarg for lightweight
        method_kwarg = kwargs.pop("method", mode)  # "watershed" or "flood_fill"
        raw_masks = method_module.segment(
            data,
            method=method_kwarg,
            postprocess=postprocess_default,
            verbose=verbose,
            **kwargs,
        )
    else:
        # Cellpose methods: pass the image
        if data.average_stack is None:
            if data.images is not None:
                if verbose:
                    print("Computing average stack from images...")
                image = data.images.mean(axis=0).astype(np.float32)
            else:
                raise ValueError("No image data available for segmentation")
        else:
            image = data.average_stack.astype(np.float32)

        raw_masks = method_module.segment(
            image,
            model_path=model_path,
            model_dir=model_dir,
            postprocess=postprocess_default,
            verbose=verbose,
            **kwargs,
        )

    # Convert to pygor format using shared mask utilities
    pygor_mask = masks.to_pygor(raw_masks)

    # Clean up ROI labels to be consecutive
    pygor_mask = masks.relabel_pygor_consecutive(pygor_mask)

    n_final = len(np.unique(pygor_mask)) - 1  # -1 for background
    if verbose:
        print(f"  Final ROI count: {n_final}")

    return pygor_mask


__all__ = [
    "segment_rois",
    "list_methods",
]
