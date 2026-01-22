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
):
    """
    Plot the segmentation result overlaid on the input image.

    Parameters
    ----------
    img : ndarray
        The input image used for segmentation (should include any enhancements applied)
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
    """
    import matplotlib.pyplot as plt
    from skimage import measure

    n_rois = masks.max() if masks.max() > 0 else len(np.unique(masks)) - 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Build input title
    input_title = f'Input: {input_mode}'
    if enhanced:
        input_title += ' (enhanced)'

    # Input image
    axes[0].imshow(img, cmap='gray', origin="lower")
    axes[0].set_title(input_title)
    axes[0].axis('off')

    # ROI masks only
    axes[1].imshow(masks, cmap='gray', interpolation='nearest', origin="lower")
    axes[1].set_title(f'ROI Masks ({n_rois} ROIs)')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img, cmap='gray', origin="lower")
    if n_rois > 0:
        masked = np.ma.masked_where(masks == 0, masks)
        axes[2].imshow(masked, cmap='prism', alpha=0.23, interpolation='nearest', origin="lower")

    # Draw anatomy mask boundary if provided
    if anatomy_mask is not None:
        try:
            # Find contours of the anatomy mask
            contours = measure.find_contours(anatomy_mask.astype(float), 0.5)
            for contour in contours:
                axes[0].plot(contour[:, 1], contour[:, 0], 'c-', linewidth=1.5, alpha=0.7)
            # Add to legend
            axes[0].plot([], [], 'c-', linewidth=1.5, label='Anatomy mask')
            axes[0].legend(loc='upper right', fontsize=8)
        except Exception:
            pass  # Skip contour drawing if it fails

    axes[2].set_title(f'{method}: {input_mode} + ROIs')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
