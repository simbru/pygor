"""
Cellpose segmentation for pygor.

This module provides the main segment() function for Cellpose-based ROI detection,
as well as lower-level inference utilities.
"""

import numpy as np
from pathlib import Path
import warnings

try:
    from cellpose import models
except ImportError as e:
    raise ImportError(
        "Cellpose is not installed. It is an optional dependency for pygor.\n\n"
        "To install cellpose, run:\n"
        "  uv pip install 'cellpose[gui]>=4.0.8'\n\n"
        "Or install pygor with the cellpose extra:\n"
        "  uv pip install 'pygor[cellpose]'\n\n"
        "Alternatively, use a lightweight segmentation method that doesn't require cellpose:\n"
        "  data.segment_rois(mode='watershed')  # or 'flood_fill', 'blob'\n"
    ) from e
except OSError as e:
    if "DLL" in str(e) or "c10.dll" in str(e):
        raise OSError(
            "Failed to load PyTorch/Cellpose due to a DLL initialization error.\n"
            "This is a known Windows issue with CUDA/PyTorch compatibility.\n\n"
            "Try one of these solutions:\n"
            "  Either\n"
            "   - Reinstall PyTorch with matching CUDA version:\n"
            "     pip uninstall torch torchvision torchaudio\n"
            "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n"
            "  Or\n"
            "   - As a workaround, import cellpose before other pygor operations:\n"
            "     from cellpose import models  # Run this first in your script\n"
        ) from e
    raise


def find_latest_model(models_dir):
    """Find the most recent trained model in a directory.

    Parameters
    ----------
    models_dir : str or Path
        Directory containing trained models

    Returns
    -------
    model_path : Path or None
        Path to the most recent model, or None if not found
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        return None

    # Look for model files in models/ subdirectory first
    model_files = list(models_dir.glob("models/cellpose_*"))

    if not model_files:
        # Try looking directly in the directory
        model_files = list(models_dir.glob("cellpose_*"))

    if not model_files:
        return None

    # Sort by modification time, get newest
    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime)
    return model_files[-1]


def load_cellpose_model(model_path=None, model_dir=None, gpu=True):
    """Load a Cellpose model.

    Parameters
    ----------
    model_path : str or Path, optional
        Direct path to a trained model file
    model_dir : str or Path, optional
        Directory to search for the latest model (ignored if model_path given)
    gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    model : CellposeModel
        Loaded model ready for inference
    model_name : str
        Name/identifier of the loaded model

    Raises
    ------
    FileNotFoundError
        If no model is found and no fallback is available
    """
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
        return model, model_path.name

    if model_dir is not None:
        model_path = find_latest_model(model_dir)
        if model_path is not None:
            model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
            return model, model_path.name

    # No custom model found - this is an error for pygor since we require training
    raise FileNotFoundError(
        "No trained model found. You must train a model first using:\n"
        "  from pygor.segmentation.cellpose import train_model\n"
        "  train_model(data_dirs=[...], save_path='./models/my_model')"
    )


def run_cellpose_inference(
    image,
    model_path=None,
    model_dir=None,
    model_type="cyto3",
    diameter=None,
    flow_threshold=0.9,
    cellprob_threshold=0.5,
    min_size=2,
    gpu=True,
):
    """Run Cellpose inference on an image.

    Parameters
    ----------
    image : ndarray
        2D image to segment (e.g., average_stack from pygor)
    model_path : str or Path, optional
        Direct path to a trained model file
    model_dir : str or Path, optional
        Directory to search for the latest model
    model_type : str
        Pretrained model name to use if no custom model specified.
        Cellpose v4 defaults to "cpsam"; other names must be added as GUI models.
        Default: "cyto3"
    diameter : float, optional
        Expected cell diameter in pixels (None for auto-detect)
    flow_threshold : float
        Flow error threshold (higher = stricter, helps separate adjacent ROIs)
    cellprob_threshold : float
        Cell probability threshold (higher = more confident detections only)
    min_size : int
        Minimum ROI size in pixels
    gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'masks': ROI mask array (background=0, ROIs=1,2,3...)
        - 'flows': Flow field outputs from Cellpose
        - 'n_rois': Number of detected ROIs
        - 'auto_diameter': Auto-detected diameter (if diameter=None)
    """
    # Load model
    print("Loading Cellpose model...")
    if model_path is not None:
        model, model_name = load_cellpose_model(
            model_path=model_path, model_dir=model_dir, gpu=gpu
        )
    else:
        print("  No custom model specified, using Cellpose v4 default model")
        model_name = "cpsam"
        model = models.CellposeModel(gpu=gpu)

    # Ensure image is float32
    image = np.asarray(image, dtype=np.float32)

    # Run inference
    masks, flows, styles = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )

    # Get auto-detected diameter
    auto_diameter = styles[0] if len(styles) > 0 else 0

    results = {
        "masks": masks,
        "flows": flows,
        "n_rois": len(np.unique(masks)) - 1,  # -1 for background
        "auto_diameter": auto_diameter,
        "model_name": model_name,
    }
    return results

def segment(
    image,
    *,
    model_path=None,
    model_dir=None,
    model_type="cyto3",
    postprocess=True,
    verbose=True,
    plot=False,
    **kwargs,
):
    """Segment an image using Cellpose.

    This is the main entry point for Cellpose segmentation. It handles
    inference and optional post-processing (splitting/shrinking).

    Parameters
    ----------
    image : ndarray
        2D image to segment
    model_path : str or Path, optional
        Direct path to a trained model file
    model_dir : str or Path, optional
        Directory to search for the latest model
    model_type : str
        Pretrained model name to use if no custom model specified.
        Cellpose v4 defaults to "cpsam"; other names must be added as GUI models.
        Default: "cyto3"
    postprocess : bool
        If True, apply splitting/shrinking heuristics (default: True)
    verbose : bool
        Print progress messages
    **kwargs
        Additional parameters:

        Cellpose parameters:
        - diameter : float (default: None, auto-detect)
        - flow_threshold : float (default: 0.9)
        - cellprob_threshold : float (default: 0.5)
        - min_size : int (default: 2)

        Post-processing parameters (when postprocess=True):
        - split_large : bool (default: True)
        - size_multiplier : float (default: 1.5)
        - min_peak_distance : int (default: 1)
        - min_size_after_split : int (default: 4)
        - shrink_iterations : int (default: 1, 0 to disable)
        - shrink_size_threshold : int (default: 30)

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
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
        "size_multiplier": kwargs.pop("size_multiplier", 1.5),
        "min_distance": kwargs.pop("min_peak_distance", 1),
        "min_size_after_split": kwargs.pop("min_size_after_split", 4),
    }
    shrink_iterations = kwargs.pop("shrink_iterations", 1)
    shrink_size_threshold = kwargs.pop("shrink_size_threshold", 30)

    # Warn about unused kwargs
    if kwargs:
        warnings.warn(f"Unused parameters: {list(kwargs.keys())}")

    # Run inference
    if verbose:
        print("Running Cellpose segmentation...")

    if model_path is None:
        model_path = find_latest_model(model_dir) if model_dir is not None else None

    results = run_cellpose_inference(
        image,
        model_path=model_path,
        model_dir=model_dir,
        model_type=model_type,
        **cellpose_params,
    )

    masks = results["masks"]
    n_cellpose = results["n_rois"]

    if verbose:
        print(f"  Cellpose detected: {n_cellpose} ROIs")

    # Apply post-processing if requested
    if postprocess:
        from .postprocess import split_large_rois, shrink_rois

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

    # Plot if requested
    if plot:
        from pygor.segmentation.plotting import plot_segmentation
        plot_segmentation(image, masks, input_mode="average", method="cellpose")

    return masks
