"""
Cellpose inference for ROI segmentation.

This module handles loading trained Cellpose models and running inference
on pygor image data.
"""

import numpy as np
from pathlib import Path
from cellpose import models


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
    model, model_name = load_cellpose_model(
        model_path=model_path, model_dir=model_dir, gpu=gpu
    )

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
