"""
Core segmentation functionality for pygor.

This module provides the main segment_rois() function that orchestrates
different segmentation methods and handles mask format conversion.
"""

import numpy as np
import warnings


def convert_cellpose_to_pygor(cellpose_mask):
    """Convert Cellpose mask format to pygor format.

    Cellpose format: background=0, ROIs=1,2,3...
    pygor format: background=1, ROIs=-1,-2,-3...

    Parameters
    ----------
    cellpose_mask : ndarray
        Mask in Cellpose format

    Returns
    -------
    pygor_mask : ndarray
        Mask in pygor format
    """
    pygor_mask = cellpose_mask.copy().astype(np.int16)

    # Convert ROIs to negative (do this first while we can identify them)
    roi_pixels = pygor_mask > 0
    pygor_mask[roi_pixels] = -pygor_mask[roi_pixels]

    # Convert background (0) to 1
    pygor_mask[pygor_mask == 0] = 1

    return pygor_mask


def convert_pygor_to_cellpose(pygor_mask):
    """Convert pygor mask format to Cellpose format.

    pygor format: background=1, ROIs=-1,-2,-3...
    Cellpose format: background=0, ROIs=1,2,3...

    Parameters
    ----------
    pygor_mask : ndarray
        Mask in pygor format

    Returns
    -------
    cellpose_mask : ndarray
        Mask in Cellpose format
    """
    if pygor_mask is None:
        return None

    cellpose_mask = pygor_mask.copy()
    # Convert background (1) to 0
    cellpose_mask[cellpose_mask == 1] = 0
    # Convert negative ROI labels to positive
    cellpose_mask = np.abs(cellpose_mask)

    return cellpose_mask.astype(np.uint16)


def segment_rois(
    data,
    mode="cellpose+",
    model_path=None,
    model_dir=None,
    # preview=False,
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
        Segmentation mode:
        - "cellpose+": Cellpose with post-processing heuristics (recommended)
        - "cellpose": Raw Cellpose output only
    model_path : str or Path, optional
        Direct path to a trained Cellpose model file
    model_dir : str or Path, optional
        Directory to search for trained models
    overwrite : bool
        If True, overwrite existing ROIs in data object
    verbose : bool
        If True, print progress messages
    **kwargs
        Additional parameters passed to the segmentation method:

        Cellpose parameters:
        - diameter : float (default: None, auto-detect)
        - flow_threshold : float (default: 0.9)
        - cellprob_threshold : float (default: 0.5)
        - min_size : int (default: 2)

        Post-processing parameters (for mode="cellpose+"):
        - split_large : bool (default: True)
        - size_multiplier : float (default: 1.25)
        - min_peak_distance : int (default: 1)
        - min_size_after_split : int (default: 4)
        - shrink_iterations : int (default: 1, 0 to disable)
        - shrink_size_threshold : int (default: 30)

    Returns
    -------
    masks : ndarray (only if preview=True)
        ROI mask in pygor format (background=1, ROIs=-1,-2,-3...)

    Examples
    --------
    >>> data = pygor.load.Core("recording.h5")
    >>> data.segment_rois(model_dir="./models/synaptic")

    >>> # Preview without saving
    >>> masks = data.segment_rois(model_dir="./models/synaptic")

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
        warnings.warn(
            "ROIs already exist. Use overwrite=True to replace."
        )
        return None

    # Parse mode
    if mode not in ("cellpose", "cellpose+"):
        raise ValueError(f"Unknown segmentation mode: {mode}. Use 'cellpose' or 'cellpose+'")

    # Import cellpose functions
    from pygor.segmentation.cellpose.inference import run_cellpose_inference
    from pygor.segmentation.cellpose.postprocess import split_large_rois, shrink_rois
    from cellpose import models

    # Extract cellpose parameters
    cellpose_params = {
        "diameter": kwargs.pop("diameter", None),
        "flow_threshold": kwargs.pop("flow_threshold", 0.9),
        "cellprob_threshold": kwargs.pop("cellprob_threshold", 0.5),
        "min_size": kwargs.pop("min_size", 2),
    }

    # Extract post-processing parameters
    split_large = kwargs.pop("split_large", True)
    split_params = {
        "size_multiplier": kwargs.pop("size_multiplier", 1.25),
        "min_distance": kwargs.pop("min_peak_distance", 1),
        "min_size_after_split": kwargs.pop("min_size_after_split", 4),
    }
    shrink_iterations = kwargs.pop("shrink_iterations", 1)
    shrink_size_threshold = kwargs.pop("shrink_size_threshold", 30)

    # Warn about unused kwargs
    if kwargs:
        warnings.warn(f"Unused parameters: {list(kwargs.keys())}")

    # Run Cellpose inference
    if verbose:
        print(f"Running Cellpose segmentation...")

    results = run_cellpose_inference(
        image,
        model_path=model_path,
        model_dir=model_dir,
        **cellpose_params,
    )

    masks = results["masks"]
    n_cellpose = results["n_rois"]

    if verbose:
        print(f"  Cellpose detected: {n_cellpose} ROIs")

    # Apply post-processing for "cellpose+" mode
    if mode == "cellpose+":
        # Split large ROIs
        if split_large:
            masks, n_splits = split_large_rois(masks, image, **split_params)
            n_after_split = len(np.unique(masks)) - 1
            if verbose and n_splits > 0:
                print(f"  After splitting: {n_after_split} ROIs (+{n_after_split - n_cellpose})")

        # Shrink ROIs
        if shrink_iterations > 0:
            masks = shrink_rois(
                masks, iterations=shrink_iterations, size_threshold=shrink_size_threshold
            )
            if verbose:
                print(f"  Applied ROI shrinking ({shrink_iterations} iteration(s))")

    # Convert to pygor format
    pygor_mask = convert_cellpose_to_pygor(masks)
    n_final = len(np.unique(pygor_mask)) - 1  # -1 for background

    if verbose:
        print(f"  Final ROI count: {n_final}")

    # Clean up ROI label numbers to be consecutive after post-processing
    # Background is 1, ROIs should be -1, -2, -3, etc.
    unique_labels = np.unique(pygor_mask)
    unique_labels = unique_labels[unique_labels != 1]  # Remove background from remapping

    # Create a temporary array to avoid overwriting issues during remapping
    remapped_mask = pygor_mask.copy()
    for i, old_label in enumerate(unique_labels, start=1):
        remapped_mask[pygor_mask == old_label] = -i
    pygor_mask = remapped_mask

    # Return or save
    # if preview:
    return pygor_mask
    # else:
    #     # Update data object and H5 file
    #     success = data.update_rois(pygor_mask, overwrite=overwrite)
    #     if not success:
    #         warnings.warn("Failed to save ROIs to H5 file")
    #     return None
