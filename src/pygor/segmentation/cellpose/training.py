"""
Cellpose model training for pygor.

This module provides functions to train custom Cellpose models on pygor data.

Key points:
- Uses rescale=True to prevent catastrophic forgetting
- Fine-tunes from cyto3 base model
- Supports training from H5 files or pygor data objects
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import label
import pygor.load


def convert_pygor_mask_to_cellpose(pygor_mask):
    """Convert pygor ROI mask to Cellpose format with sequential relabeling.

    pygor format: background=1, ROIs=-1,-2,-3...
    Cellpose format: background=0, ROIs=1,2,3... (sequential)

    Uses scipy.ndimage.label to ensure sequential labels.

    Parameters
    ----------
    pygor_mask : ndarray
        Mask in pygor format

    Returns
    -------
    cellpose_mask : ndarray
        Mask in Cellpose format
    n_rois : int
        Number of ROIs found
    """
    if pygor_mask is None:
        return None, 0

    # Create binary mask: foreground (any ROI) vs background
    binary_mask = (pygor_mask != 1).astype(np.uint8)

    # Relabel connected components to get sequential 1,2,3...
    labeled_mask, n_components = label(binary_mask)

    return labeled_mask.astype(np.uint16), n_components


def validate_mask(mask, name="mask"):
    """Validate a Cellpose-format mask and return diagnostics.

    Parameters
    ----------
    mask : ndarray
        Mask to validate (Cellpose format)
    name : str
        Name for logging

    Returns
    -------
    results : dict
        Validation results including n_rois, roi_diameters, validity
    """
    results = {
        "name": name,
        "valid": True,
        "issues": [],
    }

    # Check background
    if 0 not in mask:
        results["issues"].append("No background (0) pixels found")

    # Get ROI labels
    unique_labels = np.unique(mask)
    roi_labels = unique_labels[unique_labels > 0]

    results["n_rois"] = len(roi_labels)

    if results["n_rois"] == 0:
        results["valid"] = False
        results["issues"].append("No ROIs found in mask")
        return results

    # Calculate ROI sizes and diameters
    roi_sizes = []
    roi_diameters = []
    for roi_id in roi_labels:
        n_pixels = np.sum(mask == roi_id)
        roi_sizes.append(n_pixels)
        diameter = 2 * np.sqrt(n_pixels / np.pi)
        roi_diameters.append(diameter)

    results["roi_sizes"] = roi_sizes
    results["roi_diameters"] = roi_diameters
    results["median_diameter"] = np.median(roi_diameters)

    if results["issues"]:
        results["valid"] = False

    return results


def load_training_data(data_dirs, max_files=None, verbose=True):
    """Load H5 files for training.

    Parameters
    ----------
    data_dirs : list of str or Path
        Directories containing H5 files with ROIs
    max_files : int, optional
        Maximum number of files to load
    verbose : bool
        Print progress messages

    Returns
    -------
    images : list of ndarray
        Training images
    masks : list of ndarray
        Training masks (Cellpose format)
    stats : dict
        Loading statistics
    """
    data_dirs = [Path(d) for d in data_dirs]

    # Find all H5 files
    h5_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            h5_files.extend(data_dir.glob("**/*.h5"))
        elif verbose:
            print(f"  Warning: Directory not found: {data_dir}")
    h5_files = sorted(h5_files)

    if max_files is not None:
        h5_files = h5_files[:max_files]

    if verbose:
        print(f"Found {len(h5_files)} H5 files to process")

    images = []
    masks = []
    all_diameters = []
    skipped = {"no_image": 0, "no_rois": 0, "empty_mask": 0, "error": 0}

    for i, h5_path in enumerate(h5_files):
        try:
            data = pygor.load.Core(str(h5_path))

            # Check for image
            if data.average_stack is None:
                if data.images is not None:
                    image = data.images.mean(axis=0)
                else:
                    skipped["no_image"] += 1
                    continue
            else:
                image = data.average_stack

            # Check for ROIs
            if data.rois is None:
                skipped["no_rois"] += 1
                continue

            # Convert mask
            cellpose_mask, n_rois = convert_pygor_mask_to_cellpose(data.rois)

            if n_rois == 0:
                skipped["empty_mask"] += 1
                continue

            # Validate
            validation = validate_mask(cellpose_mask, h5_path.name)
            if not validation["valid"]:
                skipped["empty_mask"] += 1
                continue

            images.append(image.astype(np.float32))
            masks.append(cellpose_mask)
            all_diameters.extend(validation["roi_diameters"])

            if verbose and (i + 1) % 20 == 0:
                print(f"  Loaded {len(images)} files...")

        except Exception as e:
            if verbose:
                print(f"  Error loading {h5_path.name}: {str(e)[:50]}")
            skipped["error"] += 1

    stats = {
        "total_files": len(h5_files),
        "loaded": len(images),
        "skipped": skipped,
        "all_diameters": all_diameters,
        "median_diameter": np.median(all_diameters) if all_diameters else 0,
    }

    return images, masks, stats


def load_training_data_from_objects(data_objects, verbose=True):
    """Load training data from pygor data objects.

    Parameters
    ----------
    data_objects : list of Core
        Pygor data objects with ROIs
    verbose : bool
        Print progress messages

    Returns
    -------
    images : list of ndarray
        Training images
    masks : list of ndarray
        Training masks (Cellpose format)
    stats : dict
        Loading statistics
    """
    images = []
    masks = []
    all_diameters = []
    skipped = {"no_image": 0, "no_rois": 0, "empty_mask": 0}

    for data in data_objects:
        # Get image
        if data.average_stack is None:
            if data.images is not None:
                image = data.images.mean(axis=0)
            else:
                skipped["no_image"] += 1
                continue
        else:
            image = data.average_stack

        # Get ROIs
        if data.rois is None:
            skipped["no_rois"] += 1
            continue

        cellpose_mask, n_rois = convert_pygor_mask_to_cellpose(data.rois)
        if n_rois == 0:
            skipped["empty_mask"] += 1
            continue

        validation = validate_mask(cellpose_mask)
        if not validation["valid"]:
            skipped["empty_mask"] += 1
            continue

        images.append(image.astype(np.float32))
        masks.append(cellpose_mask)
        all_diameters.extend(validation["roi_diameters"])

    stats = {
        "total_files": len(data_objects),
        "loaded": len(images),
        "skipped": skipped,
        "all_diameters": all_diameters,
        "median_diameter": np.median(all_diameters) if all_diameters else 0,
    }

    return images, masks, stats


def train_model(
    data_dirs=None,
    data_objects=None,
    save_path="./models/cellpose_custom",
    model_name="cellpose_rois",
    n_epochs=50,
    learning_rate=1e-5,
    weight_decay=0.1,
    train_split=0.8,
    max_files=None,
    gpu=True,
    verbose=True,
):
    """Train a custom Cellpose model on pygor data.

    Parameters
    ----------
    data_dirs : list of str or Path, optional
        Directories containing H5 files with ROIs
    data_objects : list of Core, optional
        Pygor data objects with ROIs (alternative to data_dirs)
    save_path : str or Path
        Directory to save the trained model
    model_name : str
        Name prefix for the model file
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate (1e-5 recommended for fine-tuning)
    weight_decay : float
        Weight decay for regularization
    train_split : float
        Fraction of data for training (rest for validation)
    max_files : int, optional
        Maximum number of files to load
    gpu : bool
        Use GPU acceleration
    verbose : bool
        Print progress messages

    Returns
    -------
    model_path : Path
        Path to the saved model

    Examples
    --------
    >>> from pygor.segmentation.cellpose import train_model
    >>> train_model(
    ...     data_dirs=["./data/recordings"],
    ...     save_path="./models/my_tissue",
    ...     n_epochs=50,
    ... )
    """
    from cellpose import models, train

    if data_dirs is None and data_objects is None:
        raise ValueError("Must provide either data_dirs or data_objects")

    save_path = Path(save_path)

    # Load data
    if verbose:
        print("Loading training data...")

    if data_objects is not None:
        images, masks, stats = load_training_data_from_objects(data_objects, verbose)
    else:
        images, masks, stats = load_training_data(data_dirs, max_files, verbose)

    if len(images) == 0:
        raise ValueError("No valid training data found!")

    if verbose:
        print(f"\nLoaded {stats['loaded']} images with {len(stats['all_diameters'])} total ROIs")
        print(f"Median ROI diameter: {stats['median_diameter']:.1f} px")
        if stats["skipped"]:
            print(f"Skipped: {stats['skipped']}")

    # Train/test split
    n_total = len(images)
    n_train = int(n_total * train_split)
    indices = np.random.permutation(n_total)

    train_images = [images[i] for i in indices[:n_train]]
    train_masks = [masks[i] for i in indices[:n_train]]
    test_images = [images[i] for i in indices[n_train:]]
    test_masks = [masks[i] for i in indices[n_train:]]

    if verbose:
        print(f"\nTraining: {len(train_images)} images")
        print(f"Validation: {len(test_images)} images")

    # Initialize model
    if verbose:
        print("\nInitializing model from cyto3 base...")
    model = models.CellposeModel(gpu=gpu, model_type="cyto3")

    # Create save directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Train
    if verbose:
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"  learning_rate: {learning_rate}")
        print(f"  weight_decay: {weight_decay}")
        print(f"  rescale: True (data augmentation)")

    result = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_masks else None,
        save_path=str(save_path),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_train_masks=1,
        rescale=True,  # CRITICAL: prevents catastrophic forgetting
        model_name=model_name,
    )

    # Extract model path from result
    if isinstance(result, tuple):
        model_path = Path(result[0])
        if verbose:
            train_losses = result[1]
            print(f"\nTraining complete!")
            print(f"  Final loss: {train_losses[-1]:.4f}")
    else:
        model_path = Path(result)

    if verbose:
        print(f"  Model saved to: {model_path}")

    return model_path
