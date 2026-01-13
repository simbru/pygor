"""
Cellpose Training Script - Simplified Version

Clean rewrite focused on:
1. rescale=True for data augmentation (prevents catastrophic forgetting)
2. Proper mask relabeling with scipy.ndimage.label
3. Cellpose v4 recommended parameters
4. Extensive diagnostics at each step
5. Post-training validation

Key finding from debugging:
- Without rescale=True, the model suffers catastrophic forgetting
- Cell probability becomes negative and model detects 0 ROIs
- With rescale=True, augmentation helps maintain generalization
"""

from cellpose import models, train
import pygor.load
import numpy as np
from pathlib import Path
from scipy.ndimage import label
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIRS = [
    Path(r"D:\Igor analyses\SWN BC main"),
    Path(r"D:\Igor analyses\SWN inj"),
    Path(r"D:\Igor analyses\SWN single colour"),
    Path(r"D:\Igor analyses\SWN 6colour"),
    Path(r"D:\Igor analyses\251112 OSDS"),
    Path(r"D:\Igor analyses\SWN achrom"),
    Path(r"D:\Igor analyses\SWN paired w simple stims"),
    Path(r"D:\Igor analyses\OSDS")

]

SAVE_PATH = Path("src/pygor/dev/.models_simple")
MODEL_NAME = "cellpose_rois"
N_EPOCHS = 50
TRAIN_SPLIT = 0.8
MAX_FILES = None  # None = use all files

# Cellpose v4 recommended parameters for fine-tuning
LEARNING_RATE = 1e-5  # Much lower than old default (0.2)
WEIGHT_DECAY = 0.1
MIN_TRAIN_MASKS = 1

# CRITICAL: rescale=True enables data augmentation during training
# Without this, the model suffers catastrophic forgetting and outputs all zeros
RESCALE = True

# =============================================================================
# MASK CONVERSION FUNCTIONS
# =============================================================================


def convert_pygor_mask_to_cellpose(pygor_mask):
    """Convert pygor ROI mask to Cellpose format with sequential relabeling.

    pygor format: background=1, ROIs=-1,-2,-3...
    Cellpose format: background=0, ROIs=1,2,3... (sequential)

    Uses scipy.ndimage.label to ensure sequential labels and handle
    any gaps or issues in original labeling.
    """
    if pygor_mask is None:
        return None, 0

    # Create binary mask: foreground (any ROI) vs background
    binary_mask = (pygor_mask != 1).astype(np.uint8)

    # Relabel connected components to get sequential 1,2,3...
    # This handles: gaps in labels, disconnected regions, etc.
    labeled_mask, n_components = label(binary_mask)

    return labeled_mask.astype(np.uint16), n_components


def convert_cellpose_mask_to_pygor(cellpose_mask):
    """Convert Cellpose mask back to pygor format.

    Cellpose format: background=0, ROIs=1,2,3...
    pygor format: background=1, ROIs=-1,-2,-3...
    """
    pygor_mask = cellpose_mask.copy().astype(np.int16)

    # Convert ROIs to negative (do this first while we can identify them)
    roi_pixels = pygor_mask > 0
    pygor_mask[roi_pixels] = -pygor_mask[roi_pixels]

    # Convert background (0) to 1
    pygor_mask[pygor_mask == 0] = 1

    return pygor_mask


# =============================================================================
# MASK VALIDATION
# =============================================================================


def validate_mask(mask, name="mask"):
    """Validate a Cellpose-format mask and return diagnostics.

    Returns dict with validation results.
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

    # Check sequential labeling
    expected_labels = np.arange(1, len(roi_labels) + 1)
    if not np.array_equal(roi_labels, expected_labels):
        results["issues"].append(
            f"Labels not sequential: {roi_labels} vs expected {expected_labels}"
        )

    # Calculate ROI sizes and diameters
    roi_sizes = []
    roi_diameters = []
    for roi_id in roi_labels:
        n_pixels = np.sum(mask == roi_id)
        roi_sizes.append(n_pixels)
        # Estimate diameter assuming circular ROI
        diameter = 2 * np.sqrt(n_pixels / np.pi)
        roi_diameters.append(diameter)

    results["roi_sizes"] = roi_sizes
    results["roi_diameters"] = roi_diameters
    results["median_diameter"] = np.median(roi_diameters)
    results["min_diameter"] = np.min(roi_diameters)
    results["max_diameter"] = np.max(roi_diameters)

    if results["issues"]:
        results["valid"] = False

    return results


# =============================================================================
# DATA LOADING
# =============================================================================


def load_h5_files(data_dirs, max_files=None):
    """Load H5 files and extract images and ROI masks.

    Returns:
        images: list of numpy arrays (average stack images)
        masks: list of numpy arrays (Cellpose format masks)
        filenames: list of str (for tracking)
        stats: dict with loading statistics
    """
    # Find all H5 files
    h5_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            h5_files.extend(data_dir.glob("**/*.h5"))
        else:
            print(f"  Warning: Directory not found: {data_dir}")
    h5_files = sorted(h5_files)

    if max_files is not None:
        h5_files = h5_files[:max_files]

    print(f"\nFound {len(h5_files)} H5 files to process")

    images = []
    masks = []
    filenames = []
    all_diameters = []
    skipped = {"no_image": 0, "no_rois": 0, "empty_mask": 0, "error": 0}

    for i, h5_path in enumerate(h5_files):
        try:
            # Load data via pygor
            data = pygor.load.Core(str(h5_path))

            # Check for image
            if data.average_stack is None:
                if data.images is not None:
                    data.average_stack = data.images.mean(axis=0)
                else:
                    skipped["no_image"] += 1
                    continue

            # Check for ROIs
            if data.rois is None:
                skipped["no_rois"] += 1
                continue

            # Convert mask to Cellpose format with relabeling
            cellpose_mask, n_rois = convert_pygor_mask_to_cellpose(data.rois)

            if n_rois == 0:
                skipped["empty_mask"] += 1
                continue

            # Validate mask
            validation = validate_mask(cellpose_mask, h5_path.name)

            if not validation["valid"]:
                print(f"  [{i+1}] {h5_path.name}: Invalid mask - {validation['issues']}")
                skipped["empty_mask"] += 1
                continue

            # Store data
            images.append(data.average_stack.astype(np.float32))
            masks.append(cellpose_mask)
            filenames.append(h5_path.name)
            all_diameters.extend(validation["roi_diameters"])

            # Progress indicator (every 20 files)
            if (i + 1) % 20 == 0:
                print(f"  Loaded {len(images)} files so far...")

        except Exception as e:
            print(f"  [{i+1}] {h5_path.name}: Error - {str(e)[:50]}")
            skipped["error"] += 1
            continue

    # Compute statistics
    stats = {
        "total_files": len(h5_files),
        "loaded": len(images),
        "skipped": skipped,
        "all_diameters": all_diameters,
        "median_diameter": np.median(all_diameters) if all_diameters else 0,
    }

    return images, masks, filenames, stats


# =============================================================================
# TRAINING
# =============================================================================


def train_model(train_images, train_masks, test_images, test_masks, save_path, median_diameter):
    """Train Cellpose model with v4 recommended parameters."""

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    # Create model from cyto3 base
    print("\nInitializing model from cyto3 base...")
    model = models.CellposeModel(gpu=True, model_type="cyto3")

    print(f"\nTraining parameters:")
    print(f"  - n_epochs: {N_EPOCHS}")
    print(f"  - learning_rate: {LEARNING_RATE}")
    print(f"  - weight_decay: {WEIGHT_DECAY}")
    print(f"  - min_train_masks: {MIN_TRAIN_MASKS}")
    print(f"  - rescale: {RESCALE} (data augmentation)")
    print(f"  - estimated diameter: {median_diameter:.1f} px")

    # Create save directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\nStarting training for {N_EPOCHS} epochs...")
    print("(Watch for loss values - should decrease, not be NaN)\n")

    try:
        result = train.train_seg(
            model.net,
            train_data=train_images,
            train_labels=train_masks,
            test_data=test_images if test_images else None,
            test_labels=test_masks if test_masks else None,
            save_path=str(save_path),
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            min_train_masks=MIN_TRAIN_MASKS,
            rescale=RESCALE,  # CRITICAL: enables data augmentation
            model_name=MODEL_NAME,
        )

        # train_seg returns (model_path, train_losses, test_losses)
        if isinstance(result, tuple):
            model_path = result[0]
            train_losses = result[1]
            test_losses = result[2] if len(result) > 2 else None
            print(f"\nTraining complete!")
            print(f"  Model saved to: {model_path}")
            print(f"  Final train loss: {train_losses[-1]:.4f}")
            if test_losses is not None and len(test_losses) > 0 and test_losses[-1] > 0:
                print(f"  Final test loss: {test_losses[-1]:.4f}")
        else:
            model_path = result
            print(f"\nTraining complete! Model saved to: {model_path}")

        return model_path

    except Exception as e:
        print(f"\nTraining FAILED with error: {e}")
        return None


# =============================================================================
# POST-TRAINING VALIDATION
# =============================================================================


def validate_trained_model(model_path, test_image, test_mask, median_diameter):
    """Test the trained model on a sample image."""

    print("\n" + "=" * 60)
    print("POST-TRAINING VALIDATION")
    print("=" * 60)

    # Handle case where model_path might be a Path or tuple
    if isinstance(model_path, (tuple, list)):
        model_path = model_path[0]

    # Load trained model
    print(f"\nLoading trained model: {model_path}")
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

    # Count ground truth ROIs
    gt_rois = len(np.unique(test_mask)) - 1  # -1 for background
    print(f"Ground truth: {gt_rois} ROIs")

    # Run inference with various parameters
    print(f"\nRunning inference (diameter={median_diameter:.1f})...")

    try:
        predicted_masks, flows, styles = model.eval(
            test_image,
            diameter=median_diameter,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=5,
        )

        n_predicted = len(np.unique(predicted_masks)) - 1
        print(f"Predicted: {n_predicted} ROIs")

        # Also try with auto diameter
        print(f"\nRunning inference (diameter=None, auto-detect)...")
        predicted_masks_auto, _, styles_auto = model.eval(
            test_image,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=5,
        )
        n_predicted_auto = len(np.unique(predicted_masks_auto)) - 1
        auto_diameter = styles_auto[0] if len(styles_auto) > 0 else 0
        print(f"Predicted (auto): {n_predicted_auto} ROIs, estimated diameter: {auto_diameter:.1f}")

        # Evaluate success
        if n_predicted == 0 and n_predicted_auto == 0:
            print("\n" + "!" * 60)
            print("FAILURE: Model detects 0 ROIs!")
            print("!" * 60)
            print("\nPossible issues:")
            print("  1. Training diverged (check loss values above)")
            print("  2. Learning rate too high/low")
            print("  3. Mask format issue")
            return False, predicted_masks, flows
        else:
            print("\n" + "*" * 60)
            print(f"SUCCESS: Model detects ROIs ({n_predicted} with set diameter, {n_predicted_auto} auto)")
            print("*" * 60)
            return True, predicted_masks, flows

    except Exception as e:
        print(f"\nInference FAILED with error: {e}")
        return False, None, None


def save_diagnostic_figure(
    image, gt_mask, pred_mask, flows, save_path, median_diameter
):
    """Save a diagnostic figure comparing ground truth and predictions."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Image, Ground Truth, Prediction
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    gt_colored = np.ma.masked_where(gt_mask == 0, gt_mask)
    axes[0, 1].imshow(image, cmap="gray")
    axes[0, 1].imshow(gt_colored, cmap="tab20", alpha=0.6)
    n_gt = len(np.unique(gt_mask)) - 1
    axes[0, 1].set_title(f"Ground Truth ({n_gt} ROIs)")
    axes[0, 1].axis("off")

    if pred_mask is not None:
        pred_colored = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[0, 2].imshow(image, cmap="gray")
        axes[0, 2].imshow(pred_colored, cmap="tab20", alpha=0.6)
        n_pred = len(np.unique(pred_mask)) - 1
        axes[0, 2].set_title(f"Prediction ({n_pred} ROIs)")
    else:
        axes[0, 2].text(0.5, 0.5, "No prediction", ha="center", va="center")
        axes[0, 2].set_title("Prediction (FAILED)")
    axes[0, 2].axis("off")

    # Row 2: Flows and probability
    # Note: flows structure can vary by Cellpose version
    # flows[0] is usually cell probability, flows[1] and flows[2] are flow fields
    try:
        if flows is not None and len(flows) >= 1:
            # Cell probability (first element)
            prob = flows[0]
            if prob.ndim == 3:
                prob = prob[0]  # Take first channel if 3D
            axes[1, 0].imshow(prob, cmap="hot")
            axes[1, 0].set_title("Cell Probability")
            axes[1, 0].axis("off")

            if len(flows) >= 2:
                # Flow fields
                flow = flows[1]
                if flow.ndim == 3 and flow.shape[0] == 2:
                    # Shape is (2, H, W) - split into Y and X
                    axes[1, 1].imshow(flow[0], cmap="RdBu_r")
                    axes[1, 1].set_title("Flow Y")
                    axes[1, 1].axis("off")

                    axes[1, 2].imshow(flow[1], cmap="RdBu_r")
                    axes[1, 2].set_title("Flow X")
                    axes[1, 2].axis("off")
                elif flow.ndim == 2:
                    axes[1, 1].imshow(flow, cmap="RdBu_r")
                    axes[1, 1].set_title("Flow")
                    axes[1, 1].axis("off")
                    axes[1, 2].axis("off")
                else:
                    axes[1, 1].axis("off")
                    axes[1, 2].axis("off")
            else:
                axes[1, 1].axis("off")
                axes[1, 2].axis("off")
        else:
            for ax in axes[1, :]:
                ax.axis("off")
    except Exception as e:
        print(f"  Warning: Could not plot flows: {e}")
        for ax in axes[1, :]:
            ax.axis("off")

    plt.suptitle(f"Training Validation (median diameter: {median_diameter:.1f} px)")
    plt.tight_layout()

    fig_path = save_path / "training_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic figure to: {fig_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("CELLPOSE TRAINING - SIMPLE VERSION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading data...")
    print(f"Data directories:")
    for d in DATA_DIRS:
        print(f"  - {d}")

    images, masks, filenames, stats = load_h5_files(DATA_DIRS, max_files=MAX_FILES)

    print(f"\n[DATA SUMMARY]")
    print(f"  Total files found: {stats['total_files']}")
    print(f"  Successfully loaded: {stats['loaded']}")
    print(f"  Skipped (no image): {stats['skipped']['no_image']}")
    print(f"  Skipped (no ROIs): {stats['skipped']['no_rois']}")
    print(f"  Skipped (empty mask): {stats['skipped']['empty_mask']}")
    print(f"  Skipped (error): {stats['skipped']['error']}")

    if len(images) == 0:
        print("\nERROR: No valid data loaded!")
        return

    # -------------------------------------------------------------------------
    # Step 2: Analyze data
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Analyzing data...")

    # Image statistics
    img_shapes = [img.shape for img in images]
    unique_shapes = list(set(img_shapes))
    print(f"\n[IMAGE STATISTICS]")
    print(f"  Unique shapes: {unique_shapes}")
    print(f"  Value range: [{min(img.min() for img in images):.2f}, {max(img.max() for img in images):.2f}]")

    # ROI statistics
    median_diameter = stats["median_diameter"]
    all_diameters = stats["all_diameters"]
    print(f"\n[ROI STATISTICS]")
    print(f"  Total ROIs: {len(all_diameters)}")
    print(f"  Median diameter: {median_diameter:.2f} px")
    print(f"  Diameter range: [{np.min(all_diameters):.2f}, {np.max(all_diameters):.2f}] px")
    print(f"  Diameter percentiles: 10%={np.percentile(all_diameters, 10):.2f}, 90%={np.percentile(all_diameters, 90):.2f}")

    # -------------------------------------------------------------------------
    # Step 3: Train/test split
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Splitting data...")

    n_total = len(images)
    n_train = int(n_total * TRAIN_SPLIT)

    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_images = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    test_masks = [masks[i] for i in test_idx]

    train_rois = sum(len(np.unique(m)) - 1 for m in train_masks)
    test_rois = sum(len(np.unique(m)) - 1 for m in test_masks)

    print(f"\n[SPLIT SUMMARY]")
    print(f"  Training: {len(train_images)} images, {train_rois} ROIs")
    print(f"  Testing: {len(test_images)} images, {test_rois} ROIs")

    # -------------------------------------------------------------------------
    # Step 4: Pre-training validation
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Pre-training validation...")

    # Check for issues
    issues = []

    # Check shapes
    train_shapes = set(img.shape for img in train_images)
    if len(train_shapes) > 1:
        issues.append(f"Multiple image shapes in training data: {train_shapes}")

    # Check for NaN/Inf
    for i, img in enumerate(train_images):
        if np.any(np.isnan(img)) or np.any(np.isinf(img)):
            issues.append(f"NaN/Inf found in training image {i}")

    # Check mask values
    for i, mask in enumerate(train_masks):
        if mask.min() < 0:
            issues.append(f"Negative values in mask {i}")

    if issues:
        print("  WARNINGS:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed!")

    # Save sample visualization
    print("\n  Saving sample visualization...")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(4, len(train_images))):
        axes[0, i].imshow(train_images[i], cmap="gray")
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        mask_colored = np.ma.masked_where(train_masks[i] == 0, train_masks[i])
        axes[1, i].imshow(train_images[i], cmap="gray")
        axes[1, i].imshow(mask_colored, cmap="tab20", alpha=0.6)
        n_rois = len(np.unique(train_masks[i])) - 1
        axes[1, i].set_title(f"Mask ({n_rois} ROIs)")
        axes[1, i].axis("off")

    plt.suptitle("Training Data Samples")
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "training_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {SAVE_PATH / 'training_samples.png'}")

    # -------------------------------------------------------------------------
    # Step 5: Training
    # -------------------------------------------------------------------------
    model_path = train_model(
        train_images, train_masks, test_images, test_masks, SAVE_PATH, median_diameter
    )

    if model_path is None:
        print("\nTraining failed. Exiting.")
        return

    # -------------------------------------------------------------------------
    # Step 6: Post-training validation
    # -------------------------------------------------------------------------
    # Test on first training image (should work well)
    test_img = train_images[0]
    test_msk = train_masks[0]

    success, pred_mask, flows = validate_trained_model(
        model_path, test_img, test_msk, median_diameter
    )

    # Save diagnostic figure
    save_diagnostic_figure(
        test_img, test_msk, pred_mask, flows, SAVE_PATH, median_diameter
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print(f"Median ROI diameter: {median_diameter:.1f} px")
    print(f"\nTo use this model for inference:")
    print(f"  python cellpose_infer_simple.py")
    print(f"\nRecommended inference parameters:")
    print(f"  diameter=None (auto-detect)")
    print(f"  flow_threshold=0.6 (higher helps separate adjacent ROIs)")
    print(f"  cellprob_threshold=0.0")
    print(f"  min_size=3")


if __name__ == "__main__":
    main()
