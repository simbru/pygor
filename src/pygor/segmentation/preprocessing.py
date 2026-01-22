"""
Shared preprocessing utilities for segmentation.

These functions provide image enhancement and masking operations that can be
used across different segmentation backends (blob, lightweight, etc.).

Cellpose is intentionally excluded as it was trained on raw data.
"""

import numpy as np
from skimage.filters import unsharp_mask, threshold_otsu
from scipy.ndimage import binary_erosion


def normalize(img):
    """
    Rescale image to [0, 1] range.

    Parameters
    ----------
    img : ndarray
        Input image

    Returns
    -------
    normalized : ndarray
        Image scaled to [0, 1]
    """
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-10)


def enhance_unsharp(img, radius=1.0, amount=2.5):
    """
    Apply unsharp masking to enhance local contrast.

    Useful for making blob-like structures more distinct before detection.

    Parameters
    ----------
    img : ndarray
        Input image (will be normalized first if not already 0-1)
    radius : float
        Gaussian blur radius for the unsharp mask (0.5-2.0 typical)
    amount : float
        Strength of the sharpening effect (1.0-5.0 typical)

    Returns
    -------
    enhanced : ndarray
        Enhanced image, normalized to [0, 1]
    """
    # Ensure input is normalized
    if img.max() > 1.0 or img.min() < 0.0:
        img = normalize(img)

    enhanced = unsharp_mask(img, radius=radius, amount=amount)
    return normalize(enhanced)


def create_anatomy_mask(img, method='otsu', thresh_mult=1.0, erode_iterations=2):
    """
    Create a binary mask of the anatomy/tissue region.

    This mask identifies regions with actual signal (tissue) vs background,
    allowing segmentation algorithms to ignore empty/border regions.

    Parameters
    ----------
    img : ndarray
        Input image (should be normalized to 0-1 for consistent thresholds)
    method : str or float
        Thresholding method:
        - 'otsu': Automatic threshold using Otsu's method
        - float: Manual threshold value (0-1 range for normalized images)
    thresh_mult : float
        Multiply the computed threshold by this factor.
        Lower values (e.g., 0.2) are more permissive, keeping more of the image.
        Higher values are more restrictive, keeping only bright regions.
    erode_iterations : int
        Number of binary erosion iterations to shrink the mask inward.
        Helps exclude edge artifacts. Set to 0 to disable.

    Returns
    -------
    mask : ndarray (bool)
        Binary mask where True = valid anatomy region for segmentation

    Examples
    --------
    >>> # Automatic threshold with conservative erosion
    >>> mask = create_anatomy_mask(img, method='otsu', thresh_mult=0.2, erode_iterations=1)
    >>>
    >>> # Manual threshold, no erosion
    >>> mask = create_anatomy_mask(img, method=0.1, erode_iterations=0)
    """
    if method == 'otsu':
        thresh = threshold_otsu(img)
    else:
        thresh = float(method)

    thresh *= thresh_mult
    mask = img > thresh

    # Erode to push boundary inward (exclude edge artifacts)
    if erode_iterations > 0:
        mask = binary_erosion(mask, iterations=erode_iterations)

    return mask


def filter_points_by_mask(points, mask):
    """
    Keep only points (y, x coordinates) that fall within a mask.

    Useful for filtering detected peaks/blobs to only those in valid regions.

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 2+) where first two columns are (y, x) coordinates.
        Additional columns (e.g., radius, intensity) are preserved.
    mask : ndarray (bool)
        Binary mask where True = valid region

    Returns
    -------
    filtered : ndarray
        Points with centers inside the mask
    """
    if len(points) == 0:
        return points

    centers_y = points[:, 0].astype(int)
    centers_x = points[:, 1].astype(int)

    # Clip to image bounds to avoid index errors
    centers_y = np.clip(centers_y, 0, mask.shape[0] - 1)
    centers_x = np.clip(centers_x, 0, mask.shape[1] - 1)

    valid = mask[centers_y, centers_x]
    return points[valid]


def prepare_image(data, input_mode="combined", artifact_width=None):
    """
    Prepare image for segmentation from a Core/STRF object.

    Extracts the appropriate image based on input_mode, normalizes to 0-1,
    and masks out the light artifact region.

    Parameters
    ----------
    data : Core or STRF
        Pygor data object with images and/or correlation_projection
    input_mode : str
        Which image representation to use:
        - "correlation": Use correlation_projection (requires compute_correlation_projection())
        - "average": Use mean of images stack
        - "std": Use standard deviation of images stack
        - "combined": Correlation weighted by average (recommended for most cases)
    artifact_width : int, optional
        Override artifact width. If None, uses data.params.artifact_width

    Returns
    -------
    img : ndarray
        Normalized image (0-1) with artifact region masked to 0
    artifact_fill_width : int
        Number of columns masked (for reference)

    Notes
    -----
    The artifact region (leftmost columns affected by light artifact) is set to 0
    so it won't be segmented. The number of columns masked is artifact_width + 1
    due to IGOR's inclusive indexing convention.
    """
    # Validate input_mode
    valid_modes = ("correlation", "average", "std", "combined")
    if input_mode not in valid_modes:
        raise ValueError(f"Unknown input_mode: '{input_mode}'. Use one of {valid_modes}")

    # Check data availability upfront
    has_images = data.images is not None
    has_average_stack = data.average_stack is not None
    has_correlation = data.correlation_projection is not None

    needs_correlation = input_mode in ("correlation", "combined")
    needs_images_stack = input_mode == "std"
    needs_average = input_mode in ("average", "combined")

    if needs_correlation and not has_correlation:
        raise ValueError("correlation_projection not available. Call compute_correlation_projection() first.")
    if needs_images_stack and not has_images:
        raise ValueError("Images stack required for 'std' mode but not available.")
    if needs_average and not (has_images or has_average_stack):
        raise ValueError("No image data available for computing average.")

    # Extract image based on mode
    if input_mode == "correlation":
        img = data.correlation_projection.copy()
    elif input_mode == "average":
        img = data.images.mean(axis=0)
    elif input_mode == "std":
        img = data.images.std(axis=0)
    elif input_mode == "combined":
        # Correlation weighted by average (structural + functional)
        corr = data.correlation_projection.copy()
        avg = data.images.mean(axis=0)
        # Normalize each and multiply
        corr_n = normalize(corr)
        avg_n = normalize(avg)
        img = corr_n * avg_n

    # Normalize to 0-1 BEFORE masking artifact (so artifact doesn't affect scaling)
    img = normalize(img)

    # Mask out light artifact region
    if artifact_width is None:
        artifact_width = data.params.artifact_width
    # IGOR uses inclusive indexing, so artifact_width=2 means pixels 0,1,2 are affected
    artifact_fill_width = artifact_width + 1

    # Set artifact region to 0 so it won't be segmented
    img[:, :artifact_fill_width] = 0

    return img.astype(np.float32), artifact_fill_width
