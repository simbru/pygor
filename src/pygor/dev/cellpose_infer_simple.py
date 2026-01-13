"""
Cellpose Inference Script - Simplified Version

Companion to cellpose_train_simple.py for running inference
with the trained model.
"""

from cellpose import models
import pygor.load
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model path (auto-detect latest model in the directory)
MODELS_DIR = Path("src/pygor/dev/.models_simple")

# Test file - change this to your test data
TEST_H5_FILE = Path(
    r"D:\Igor analyses\OSDS\251112 OSDS\Original_2025-11-12_SMP_0_1_gradient_contrast_400_white.h5"
    # r"D:\Igor analyses\OSDS\251020 different OSDS\w_2025-10-20_SMP_0_0_OSDS_gradient_400_white.h5"
    # r"D:\Igor analyses\OSDS\251020 different OSDS\w_2025-10-20_SMP_0_0_OSDS_bar_pairs_s200_white.h5"
)

# Inference parameters (optimized from testing)
# flow_threshold=0.6 was found to work best for separating adjacent ROIs
DIAMETER = None  # Auto-detect works well
FLOW_THRESHOLD = .9  # Higher value helps separate adjacent ROIs # around .9 seems good
CELLPROB_THRESHOLD = .5 # Default # around 0.5 or a bit higher seems good
MIN_SIZE = 2  # Minimum ROI size in pixels
#Params tuned on 251112 OSDS dataset

# ROI splitting parameters
SIZE_MULTIPLIER = 1.25
  # ROIs larger than median * this are candidates for splitting
MIN_PEAK_DISTANCE = 1  # Minimum pixels between intensity peaks (try 1-3)
MIN_SIZE_AFTER_SPLIT = 4  # Discard fragments smaller than this

# ROI shrinking parameters
SHRINK_ITERATIONS = 1  # Pixels to erode from ROI boundaries (0 to disable)
SHRINK_SIZE_THRESHOLD = 30  # Only shrink ROIs with at least this many pixels

# Output
SAVE_VISUALIZATION = True
OUTPUT_DIR = Path("src/pygor/dev")

# =============================================================================
# MASK CONVERSION
# =============================================================================


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


def convert_pygor_mask_to_cellpose(pygor_mask):
    """Convert pygor ROI mask to Cellpose format.

    pygor format: background=1, ROIs=-1,-2,-3...
    Cellpose format: background=0, ROIs=1,2,3...
    """
    if pygor_mask is None:
        return None

    cellpose_mask = pygor_mask.copy()
    # Convert background (1) to 0
    cellpose_mask[cellpose_mask == 1] = 0
    # Convert negative ROI labels to positive
    cellpose_mask = np.abs(cellpose_mask)

    return cellpose_mask.astype(np.uint16)


# =============================================================================
# ROI SPLITTING HEURISTIC
# =============================================================================


def split_large_rois(masks, image, size_multiplier=SIZE_MULTIPLIER, min_distance=MIN_PEAK_DISTANCE, min_size_after_split=MIN_SIZE_AFTER_SPLIT):
    """Split large ROIs that may contain multiple merged synaptic terminals.

    Uses watershed segmentation seeded by local intensity maxima within
    candidate ROIs. ROIs larger than (median_size * size_multiplier) are
    candidates for splitting.

    Parameters
    ----------
    masks : ndarray
        Cellpose-format mask (background=0, ROIs=1,2,3...)
    image : ndarray
        Original intensity image (used to find local maxima)
    size_multiplier : float
        ROIs larger than median_size * this value are candidates
    min_distance : int
        Minimum pixel distance between peaks to consider them distinct
    min_size_after_split : int
        Minimum size for split fragments (smaller ones are merged back)

    Returns
    -------
    new_masks : ndarray
        Updated mask with split ROIs
    n_splits : int
        Number of ROIs that were split
    """
    new_masks = masks.copy()
    roi_ids = np.unique(masks)
    roi_ids = roi_ids[roi_ids > 0]  # exclude background

    if len(roi_ids) == 0:
        return new_masks, 0

    # Calculate median ROI size
    roi_sizes = [(masks == roi_id).sum() for roi_id in roi_ids]
    median_size = np.median(roi_sizes)
    size_threshold = median_size * size_multiplier

    max_label = masks.max()
    n_splits = 0

    for roi_id in roi_ids:
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()

        if roi_size <= size_threshold:
            continue

        # Extract intensity within this ROI
        roi_image = image * roi_mask

        # Find local maxima within the ROI
        # Use the ROI mask as the labels parameter to restrict search
        coords = peak_local_max(
            roi_image,
            min_distance=min_distance,
            labels=roi_mask.astype(int),
            num_peaks_per_label=10,  # allow multiple peaks per ROI
        )

        if len(coords) <= 1:
            # Only one peak found, no split needed
            continue

        # Create markers for watershed
        markers = np.zeros_like(masks, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            markers[y, x] = i + 1

        # Run watershed on inverted image (watershed finds basins, we want peaks)
        # Use the ROI mask to constrain the watershed
        split_labels = watershed(-roi_image, markers, mask=roi_mask)

        # Check if split produced valid fragments
        split_ids = np.unique(split_labels)
        split_ids = split_ids[split_ids > 0]

        if len(split_ids) <= 1:
            continue

        # Validate fragment sizes and relabel
        valid_splits = []
        for split_id in split_ids:
            fragment_mask = split_labels == split_id
            fragment_size = fragment_mask.sum()
            if fragment_size >= min_size_after_split:
                valid_splits.append((split_id, fragment_mask))

        if len(valid_splits) <= 1:
            # After size filtering, only one valid fragment remains
            continue

        # Clear the original ROI from new_masks
        new_masks[roi_mask] = 0

        # Assign new labels to valid fragments
        for _, fragment_mask in valid_splits:
            max_label += 1
            new_masks[fragment_mask] = max_label

        n_splits += 1

    return new_masks, n_splits


def shrink_rois(masks, iterations=1, size_threshold=10):
    """Shrink large ROIs by eroding their boundaries.

    Parameters
    ----------
    masks : ndarray
        Cellpose-format mask (background=0, ROIs=1,2,3...)
    iterations : int
        Number of pixels to erode from boundaries
    size_threshold : int
        Only shrink ROIs with at least this many pixels (smaller ones kept as-is)

    Returns
    -------
    new_masks : ndarray
        Mask with shrunk ROIs
    """
    from scipy.ndimage import binary_erosion

    if iterations <= 0:
        return masks.copy()

    new_masks = np.zeros_like(masks)
    for roi_id in np.unique(masks):
        if roi_id == 0:
            continue
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()

        if roi_size < size_threshold:
            # Keep small ROIs unchanged
            new_masks[roi_mask] = roi_id
        else:
            # Shrink larger ROIs
            shrunk = binary_erosion(roi_mask, iterations=iterations)
            if shrunk.sum() > 0:
                new_masks[shrunk] = roi_id
            else:
                # Erosion eliminated the ROI, keep original
                new_masks[roi_mask] = roi_id
    return new_masks


# =============================================================================
# MODEL LOADING
# =============================================================================


def find_latest_model(models_dir):
    """Find the most recent trained model in the directory."""
    models_dir = Path(models_dir)

    if not models_dir.exists():
        return None

    # Look for model files
    model_files = list(models_dir.glob("models/cellpose_*"))

    if not model_files:
        # Try looking directly in the directory
        model_files = list(models_dir.glob("cellpose_*"))

    if not model_files:
        return None

    # Sort by modification time, get newest
    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime)
    return model_files[-1]


def load_model(model_path=None):
    """Load a trained Cellpose model."""

    if model_path is None:
        model_path = find_latest_model(MODELS_DIR)

    if model_path is None:
        print("No trained model found!")
        print(f"Looked in: {MODELS_DIR}")
        print("\nFalling back to pretrained cyto3 model...")
        return models.CellposeModel(gpu=True, model_type="cyto3"), "cyto3 (pretrained)"

    print(f"Loading model: {model_path}")
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    return model, model_path.name


# =============================================================================
# INFERENCE
# =============================================================================


def run_inference(model, image, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, min_size=5):
    """Run Cellpose inference on a single image."""
    # image_99th = np.percentile(image, 99)
    
    masks, flows, styles = model.eval(
        # np.clip(image, 0, image_99th),  # Clip extreme values
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
    }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================


def _split_flows(flows):
    """Extract cellprobability and (y, x) flow components regardless of layout."""

    cellprob = None
    flow_y = None
    flow_x = None

    for arr in flows or []:
        if arr is None:
            continue

        a = np.asarray(arr)

        # 2D arrays are typically cell probability / distance maps
        if a.ndim == 2:
            if cellprob is None:
                cellprob = a
            continue

        # 3D flow fields – support both channel-first and channel-last
        if a.ndim == 3:
            if a.shape[0] == 2:  # (2, H, W)
                flow_y, flow_x = a[0], a[1]
                continue
            if a.shape[-1] == 2:  # (H, W, 2)
                flow_y, flow_x = a[..., 0], a[..., 1]
                continue

    return cellprob, flow_y, flow_x


def visualize_results(image, pred_masks, flows, gt_masks=None, split_masks=None, title="Inference Results"):
    """Create visualization of inference results with optional split comparison.

    Parameters
    ----------
    image : ndarray
        Input image
    pred_masks : ndarray
        Original predicted masks from Cellpose
    flows : list
        Flow outputs from Cellpose
    gt_masks : ndarray, optional
        Ground truth masks if available
    split_masks : ndarray, optional
        Masks after splitting heuristic applied
    title : str
        Figure title
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    # Determine layout based on what's provided
    n_cols = 4
    fig, axes = plt.subplots(2, n_cols, figsize=(20, 10))

    # Row 1: Image, Original masks, Split masks, Ground truth
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # Original predicted masks
    pred_colored = np.ma.masked_where(pred_masks == 0, pred_masks)
    axes[0, 1].imshow(image, cmap="gray")
    axes[0, 1].imshow(pred_colored, cmap="prism", alpha=0.4)
    n_pred = len(np.unique(pred_masks)) - 1
    axes[0, 1].set_title(f"Original ({n_pred} ROIs)")
    axes[0, 1].axis("off")

    # Split masks (if provided)
    if split_masks is not None:
        split_colored = np.ma.masked_where(split_masks == 0, split_masks)
        axes[0, 2].imshow(image, cmap="gray")
        axes[0, 2].imshow(split_colored, cmap="prism", alpha=0.4)
        n_split = len(np.unique(split_masks)) - 1
        diff = n_split - n_pred
        axes[0, 2].set_title(f"After Split ({n_split} ROIs, +{diff})")
        axes[0, 2].axis("off")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].set_title("Split (disabled)")

    # Ground truth (if provided)
    if gt_masks is not None:
        gt_colored = np.ma.masked_where(gt_masks == 0, gt_masks)
        axes[0, 3].imshow(image, cmap="gray")
        axes[0, 3].imshow(gt_colored, cmap="prism", alpha=0.4)
        n_gt = len(np.unique(gt_masks)) - 1
        axes[0, 3].set_title(f"Ground Truth ({n_gt} ROIs)")
        axes[0, 3].axis("off")
    else:
        axes[0, 3].axis("off")
        axes[0, 3].set_title("Ground Truth (N/A)")

    # Row 2: Flows and difference visualization
    cellprob, flow_y, flow_x = _split_flows(flows)

    # Cell probability
    if cellprob is not None:
        axes[1, 0].imshow(cellprob, cmap="hot")
        axes[1, 0].set_title("Cell Probability")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    # Flow Y
    if flow_y is not None:
        axes[1, 1].imshow(flow_y, cmap="RdBu_r")
        axes[1, 1].set_title("Flow Y")
        axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    # Flow X
    if flow_x is not None:
        axes[1, 2].imshow(flow_x, cmap="RdBu_r")
        axes[1, 2].set_title("Flow X")
        axes[1, 2].axis("off")
    else:
        axes[1, 2].axis("off")

    # Difference map: show where splits occurred
    if split_masks is not None:
        # Highlight ROIs that were split (exist in split but not original boundaries)
        orig_boundaries = np.zeros_like(pred_masks, dtype=bool)
        for roi_id in np.unique(pred_masks):
            if roi_id == 0:
                continue
            roi_mask = pred_masks == roi_id
            dilated = binary_dilation(roi_mask)
            eroded = binary_erosion(roi_mask)
            orig_boundaries |= dilated & ~eroded

        split_boundaries = np.zeros_like(split_masks, dtype=bool)
        for roi_id in np.unique(split_masks):
            if roi_id == 0:
                continue
            roi_mask = split_masks == roi_id
            dilated = binary_dilation(roi_mask)
            eroded = binary_erosion(roi_mask)
            split_boundaries |= dilated & ~eroded

        # New boundaries from splitting
        new_boundaries = split_boundaries & ~orig_boundaries

        axes[1, 3].imshow(image, cmap="gray")
        axes[1, 3].imshow(orig_boundaries, cmap="Blues", alpha=0.3)
        axes[1, 3].imshow(new_boundaries, cmap="Reds", alpha=0.6)
        axes[1, 3].set_title("Split Boundaries (red=new)")
        axes[1, 3].axis("off")
    else:
        axes[1, 3].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("CELLPOSE INFERENCE - SIMPLE VERSION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load model
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading model...")
    model, model_name = load_model()
    print(f"  Using model: {model_name}")

    # -------------------------------------------------------------------------
    # Step 2: Load test data
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Loading test data...")

    if not TEST_H5_FILE.exists():
        print(f"  ERROR: File not found: {TEST_H5_FILE}")
        return

    print(f"  Loading: {TEST_H5_FILE.name}")
    data = pygor.load.Core(str(TEST_H5_FILE))

    if data.average_stack is None:
        if data.images is not None:
            print("  Computing average stack from images...")
            data.average_stack = data.images.mean(axis=0)
        else:
            print("  ERROR: No image data found!")
            return

    image = data.average_stack.astype(np.float32)

    print(f"\n[IMAGE INFO]")
    print(f"  Shape: {image.shape}")
    print(f"  Value range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Mean: {image.mean():.2f}, Std: {image.std():.2f}")

    # Check for ground truth
    gt_masks = None
    if data.rois is not None:
        gt_masks = convert_pygor_mask_to_cellpose(data.rois)
        n_gt = len(np.unique(gt_masks)) - 1
        print(f"  Ground truth available: {n_gt} ROIs")

    # -------------------------------------------------------------------------
    # Step 3: Run inference
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Running inference...")

    print(f"\n[INFERENCE PARAMETERS]")
    print(f"  diameter: {DIAMETER}")
    print(f"  flow_threshold: {FLOW_THRESHOLD}")
    print(f"  cellprob_threshold: {CELLPROB_THRESHOLD}")
    print(f"  min_size: {MIN_SIZE}")

    # Run with specified diameter
    print(f"\nRunning inference with diameter={DIAMETER}...")
    results = run_inference(
        model,
        image,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
    )
    print(f"  Detected: {results['n_rois']} ROIs")

    # Also run with auto diameter
    print(f"\nRunning inference with diameter=None (auto)...")
    results_auto = run_inference(
        model,
        image,
        diameter=None,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
    )
    print(f"  Detected: {results_auto['n_rois']} ROIs")
    print(f"  Auto-estimated diameter: {results_auto['auto_diameter']:.1f}")

    # Use whichever got more ROIs
    if results_auto["n_rois"] > results["n_rois"]:
        print(f"\n  Using auto-diameter results (more ROIs detected)")
        best_results = results_auto
    else:
        best_results = results

    # -------------------------------------------------------------------------
    # Step 4: ROI splitting heuristic
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Applying ROI splitting heuristic...")

    masks_original = best_results["masks"].copy()
    masks_split, n_splits = split_large_rois(
        masks_original,
        image,
        size_multiplier=SIZE_MULTIPLIER,
        min_distance=MIN_PEAK_DISTANCE,
        min_size_after_split=MIN_SIZE_AFTER_SPLIT,
    )

    n_original = len(np.unique(masks_original)) - 1
    n_after_split = len(np.unique(masks_split)) - 1

    print(f"  ROIs split: {n_splits}")
    print(f"  Original count: {n_original}")
    print(f"  After splitting: {n_after_split} (+{n_after_split - n_original})")

    # -------------------------------------------------------------------------
    # Step 5: Apply ROI shrinking
    # -------------------------------------------------------------------------
    if SHRINK_ITERATIONS > 0:
        print(f"\n[STEP 5] Shrinking ROIs by {SHRINK_ITERATIONS} pixel(s)...")
        masks_final = shrink_rois(masks_split, iterations=SHRINK_ITERATIONS, size_threshold=SHRINK_SIZE_THRESHOLD)
        n_final = len(np.unique(masks_final)) - 1
        n_lost = n_after_split - n_final
        print(f"  ROIs after shrinking: {n_final}" + (f" ({n_lost} lost)" if n_lost > 0 else ""))
    else:
        masks_final = masks_split
        n_final = n_after_split

    # -------------------------------------------------------------------------
    # Step 6: Results summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOriginal ROIs: {n_original}")
    print(f"After splitting: {n_after_split}")
    print(f"After shrinking: {n_final}")
    if gt_masks is not None:
        n_gt = len(np.unique(gt_masks)) - 1
        print(f"Ground truth ROIs: {n_gt}")
        diff_orig = n_original - n_gt
        diff_split = n_after_split - n_gt
        print(f"Difference (original): {diff_orig:+d} ({100*diff_orig/n_gt:+.1f}%)")
        print(f"Difference (split): {diff_split:+d} ({100*diff_split/n_gt:+.1f}%)")

    # -------------------------------------------------------------------------
    # Step 7: Visualization
    # -------------------------------------------------------------------------
    if SAVE_VISUALIZATION:
        print("\n[STEP 6] Saving visualization...")

        title = f"Inference Results - Original: {n_original}, Split: {n_after_split}, Final: {n_final}"
        fig = visualize_results(
            image,
            masks_original,
            best_results["flows"],
            gt_masks=gt_masks,
            split_masks=masks_final,  # Show final (split + shrunk) masks
            title=title,
        )

        output_path = OUTPUT_DIR / "cellpose_inference_result.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {output_path}")
        plt.close()

    # -------------------------------------------------------------------------
    # Step 6: Optional - Save to H5
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    # save_option = input("Save predicted ROIs to H5 file? (y/n): ").lower().strip()

    # if save_option == "y":
    #     # Convert to pygor format
    #     pygor_mask = convert_cellpose_mask_to_pygor(best_results["masks"])

    #     try:
    #         success = data.update_rois(pygor_mask, overwrite=True)
    #         if success:
    #             print(f"  Saved {best_results['n_rois']} ROIs to H5 file")
    #         else:
    #             print("  Failed to save ROIs")
    #     except Exception as e:
    #         print(f"  Error saving ROIs: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
