"""
Shared plotting utilities for segmentation results.

Provides simple visualization of segmentation results overlaid on input images.
"""

import numpy as np


def plot_segmentation(
    img,
    masks,
    input_mode="image",
    method="segmentation",
    anatomy_mask=None,
    enhanced=None,
    average_img=None,
):
    """
    Plot the segmentation result as a 2x2 grid.

    Layout::

        [[average stack]          [enhanced/processed image]]
        [[ROIs over average stack] [ROIs over enhanced image]]

    Parameters
    ----------
    img : ndarray
        The processed/enhanced image used for segmentation
    masks : ndarray
        ROI mask array (background=0, ROIs=1,2,3...)
    input_mode : str
        Image mode used, for labeling (e.g., "combined", "average", "correlation")
    method : str
        Segmentation method used, for labeling (e.g., "blob", "watershed", "cellpose")
    anatomy_mask : ndarray (bool), optional
        If provided, draws the anatomy mask boundary as a contour on the overlay
    enhanced : bool, optional
        If True, indicates the image was enhanced (adds to title). Auto-detected if None.
    average_img : ndarray, optional
        The raw average stack. If None, the left column uses ``img`` as fallback.
    """
    import matplotlib.pyplot as plt
    from skimage import measure

    n_rois = masks.max() if masks.max() > 0 else len(np.unique(masks)) - 1

    if average_img is None:
        average_img = img

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # Build enhanced title
    enhanced_title = f"Input: {input_mode}"
    if enhanced:
        enhanced_title += " (enhanced)"

    # -- Top-left: pure average stack --
    axes[0, 0].imshow(average_img, cmap="gray", origin="lower")
    axes[0, 0].set_title("Average stack")
    axes[0, 0].axis("off")

    # -- Top-right: enhanced / processed image --
    axes[0, 1].imshow(img, cmap="gray", origin="lower")
    axes[0, 1].set_title(enhanced_title)
    axes[0, 1].axis("off")

    # Draw anatomy mask boundary on the enhanced image panel if provided
    if anatomy_mask is not None:
        try:
            contours = measure.find_contours(anatomy_mask.astype(float), 0.5)
            for contour in contours:
                axes[0, 1].plot(
                    contour[:, 1], contour[:, 0], "c-", linewidth=1.5, alpha=0.7
                )
            axes[0, 1].plot([], [], "c-", linewidth=1.5, label="Anatomy mask")
            axes[0, 1].legend(loc="upper right", fontsize=8)
        except Exception:
            pass

    # Prepare masked ROI overlay
    masked = np.ma.masked_where(masks == 0, masks) if n_rois > 0 else None

    # -- Bottom-left: ROIs over average stack --
    axes[1, 0].imshow(average_img, cmap="gray", origin="lower")
    if masked is not None:
        axes[1, 0].imshow(
            masked, cmap="prism", alpha=0.23, interpolation="nearest", origin="lower"
        )
    axes[1, 0].set_title(f"{method}: average + ROIs ({n_rois})")
    axes[1, 0].axis("off")

    # -- Bottom-right: ROIs over enhanced image --
    axes[1, 1].imshow(img, cmap="gray", origin="lower")
    if masked is not None:
        axes[1, 1].imshow(
            masked, cmap="prism", alpha=0.23, interpolation="nearest", origin="lower"
        )
    axes[1, 1].set_title(f"{method}: {input_mode} + ROIs ({n_rois})")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
