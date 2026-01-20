"""
Pygor segmentation module.

Provides automated ROI segmentation using various methods.

Example usage:
    data = pygor.load.Core("file.h5")
    data.segment_rois(mode="cellpose+")  # Cellpose + post-processing heuristics

Available modes:
    - "cellpose+": Cellpose with splitting/shrinking heuristics (recommended)
    - "cellpose": Raw Cellpose output only

To add new segmentation methods, create a submodule with a segment() function
and add an entry to _METHODS below.
"""

import numpy as np
import warnings

# Method registry: mode -> (module_path, postprocess_default)
_METHODS = {
    "cellpose": ("pygor.segmentation.cellpose", False),
    "cellpose+": ("pygor.segmentation.cellpose", True),
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
    data : Core
        Pygor data object with average_stack attribute
    mode : str
        Segmentation mode. Use list_methods() for available options:
        - "cellpose+": Cellpose with post-processing heuristics (recommended)
        - "cellpose": Raw Cellpose output only
    model_path : str or Path, optional
        Direct path to a trained model file
    model_dir : str or Path, optional
        Directory to search for trained models
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

    Returns
    -------
    masks : ndarray
        ROI mask in pygor format (background=1, ROIs=-1,-2,-3...)

    Examples
    --------
    >>> data = pygor.load.Core("recording.h5")
    >>> data.segment_rois(model_dir="./models/synaptic")

    >>> # Custom parameters
    >>> data.segment_rois(
    ...     model_dir="./models/synaptic",
    ...     flow_threshold=0.8,
    ...     shrink_iterations=2,
    ... )
    """
    # Validate input
    if data.average_stack is None:
        if data.images is not None:
            if verbose:
                print("Computing average stack from images...")
            image = data.images.mean(axis=0).astype(np.float32)
        else:
            raise ValueError("No image data available for segmentation")
    else:
        image = data.average_stack.astype(np.float32)

    # Check for existing ROIs
    if data.rois is not None and not overwrite:
        warnings.warn("ROIs already exist. Use overwrite=True to replace.")
        return None

    # Get method
    if mode not in _METHODS:
        available = list_methods()
        raise ValueError(f"Unknown mode: '{mode}'. Available: {available}")

    module_path, postprocess_default = _METHODS[mode]

    # Lazy import the method module
    import importlib
    method_module = importlib.import_module(module_path)

    # Run segmentation (returns Cellpose format)
    masks = method_module.segment(
        image,
        model_path=model_path,
        model_dir=model_dir,
        postprocess=postprocess_default,
        verbose=verbose,
        **kwargs,
    )

    # Convert to pygor format using the method's mask converter
    pygor_mask = method_module.masks.to_pygor(masks)

    # Clean up ROI labels to be consecutive
    pygor_mask = method_module.masks.relabel_pygor_consecutive(pygor_mask)

    n_final = len(np.unique(pygor_mask)) - 1  # -1 for background
    if verbose:
        print(f"  Final ROI count: {n_final}")

    return pygor_mask


__all__ = [
    "segment_rois",
    "list_methods",
]
